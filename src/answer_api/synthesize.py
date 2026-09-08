from __future__ import annotations

import re

from openai import AsyncOpenAI

from answer_api import search as search_mod
from graph_extract.config import ExtractSettings
from graph_extract.usage import instrument

# Case-insensitive: a URL must NEVER survive (design-decision #2). Stop the
# match at brackets so a URL written flush against a marker ("https://x[1]")
# doesn't swallow the legitimate [1] citation along with it.
_URL_RE = re.compile(r"https?://[^\s\[\]]+", re.IGNORECASE)
_MARKER_RE = re.compile(r"\[(\d+)\]")
# A hyphen / en-dash / em-dash with optional surrounding spaces -- the forms a
# model uses to write a marker RANGE like "[31]-[60]".
_SEP = r"[ \t]*[-–—][ \t]*"
_RANGE_RE = re.compile(r"\[(\d+)\]" + _SEP + r"\[(\d+)\]")

_PROMPT = (
    "You are answering a question about backup products using ONLY the numbered "
    "facts below. Cite every claim inline with its [N] marker. Do NOT use outside "
    "knowledge. Do NOT write any URL or link. If the facts do not answer the "
    "question, reply exactly: "
    "\"I don't have enough information to answer that from the available sources.\"\n\n"
    "FACTS:\n{facts}\n\nQUESTION: {q}\n\nAnswer:"
)

_REFUSAL = "I don't have enough information to answer that from the available sources."


def _strip_markers(text: str, keep: set[int]) -> str:
    """Remove every [N] marker whose N is not in `keep`, then repair the spacing.

    Design decision #2 says a citation is graph-derived and the LLM only emits
    markers. Filtering an unresolvable marker out of the `cited` list but leaving
    it in the prose broke that: readers saw citations the envelope did not have.
    A visible marker must always resolve.

    Ranges get no expander on purpose. "[31]-[60]" is two markers and a
    separator; a model writing a 30-marker span is guessing, not citing, so
    expanding it would manufacture citations it never made. When both ends are
    dropped the separator goes too, or the prose is left with an orphaned dash.
    """
    def _range(m: re.Match[str]) -> str:
        a, b = int(m.group(1)), int(m.group(2))
        if a in keep and b in keep:
            return m.group(0)          # a real span -- leave it intact
        if a in keep:
            return f"[{a}]"
        if b in keep:
            return f"[{b}]"
        return ""                      # neither resolves: markers AND separator go

    text = _RANGE_RE.sub(_range, text)
    text = _MARKER_RE.sub(
        lambda m: m.group(0) if int(m.group(1)) in keep else "", text)
    # Repair spacing WITHOUT touching line breaks: this runs on every answer, and
    # paragraph structure is part of a correct one.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    return text.strip()


def _finalize_answer(raw: str, marker_map: dict[int, dict]) -> tuple[str, list[int]]:
    """Deterministic design-decision #2 enforcement: strip any URL the model
    emitted (it must never write one), keep the ordered-unique [N] markers that
    map to a retrieved fact, and remove the ones that do not from the TEXT as
    well -- a visible marker must always correspond to a real citation."""
    text = _URL_RE.sub("", raw).strip()
    cited: list[int] = []
    for m in _MARKER_RE.findall(text):
        n = int(m)
        if n in marker_map and n not in cited:
            cited.append(n)
    return _strip_markers(text, set(marker_map)), cited


def _build_citations(cited: list[int], marker_map: dict, resolved: dict) -> list[dict]:
    """Assemble citation objects from resolved provenance — one per cited marker
    (design #2: every field is graph-derived, none LLM-authored). Shared by the
    global and DRIFT reducers, whose citation shape is identical."""
    out: list[dict] = []
    for m in cited:
        uuid = marker_map[m]["fact_uuid"]
        r = resolved.get(uuid, {})
        out.append({"marker": m, "fact_uuid": uuid,
                    "valid_at": r.get("valid_at"), "invalid_at": r.get("invalid_at"),
                    "sources": r.get("sources", [])})
    return out


def _synthesis_client_and_model(settings: ExtractSettings) -> tuple[AsyncOpenAI, str]:
    """Synthesis LLM = GLM-5.2 via the judge_* config (shared endpoint for now;
    point at a dedicated synthesis model later without touching the judge)."""
    if not settings.judge_base_url:
        raise ValueError(
            "No synthesis model configured. Set JUDGE_BASE_URL / JUDGE_MODEL / "
            "JUDGE_API_KEY in .env (synthesis currently uses the GLM-5.2 judge endpoint)."
        )
    client = instrument(AsyncOpenAI(
        api_key=settings.judge_api_key or "not-needed", base_url=settings.judge_base_url))
    return client, settings.judge_model


async def answer_local(graphiti, driver, synth_client, synth_model, *,
                       q, k=15, vendor=None, group_id) -> dict:
    """Retrieve facts via search_local, number them [1..N], let the LLM cite
    by marker only (design-decision #2: the LLM never writes a URL), then
    resolve citations deterministically from the marker map. Zero-retrieval
    short-circuits to the fixed refusal without spending any LLM tokens."""
    res = await search_mod.search_local(
        graphiti, driver, q=q, k=k, vendor=vendor, group_id=group_id)
    results = res["results"]
    if not results:
        return {"query": q, "answer": _REFUSAL, "citations": [],
                "retrieved": 0, "cited": 0}
    marker_map = {i: r for i, r in enumerate(results, 1)}
    facts_block = "\n".join(f"[{i}] {r['fact']}" for i, r in marker_map.items())
    resp = await synth_client.chat.completions.create(
        # GLM-5.2 is a reasoning model: a small cap truncates the answer to
        # empty (finish_reason='length', content=''), so give the reasoning
        # headroom -- same lesson as the GLM judge (type_precision).
        model=synth_model, temperature=0, max_tokens=3000,
        messages=[{"role": "user", "content": _PROMPT.format(facts=facts_block, q=q)}])
    raw = resp.choices[0].message.content or ""
    answer, cited = _finalize_answer(raw, marker_map)
    citations = [{"marker": m, "fact": marker_map[m]["fact"],
                  "fact_uuid": marker_map[m]["fact_uuid"],
                  "valid_at": marker_map[m].get("valid_at"),
                  "invalid_at": marker_map[m].get("invalid_at"),
                  "sources": marker_map[m]["sources"]} for m in cited]
    return {"query": q, "answer": answer, "citations": citations,
            "retrieved": len(results), "cited": len(cited)}
