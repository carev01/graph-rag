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

_PROMPT = (
    "You are answering a question about backup products using ONLY the numbered "
    "facts below. Cite every claim inline with its [N] marker. Do NOT use outside "
    "knowledge. Do NOT write any URL or link. If the facts do not answer the "
    "question, reply exactly: "
    "\"I don't have enough information to answer that from the available sources.\"\n\n"
    "FACTS:\n{facts}\n\nQUESTION: {q}\n\nAnswer:"
)

_REFUSAL = "I don't have enough information to answer that from the available sources."


def _finalize_answer(raw: str, marker_map: dict[int, dict]) -> tuple[str, list[int]]:
    """Deterministic design-decision #2 enforcement: strip any URL the model
    emitted (it must never write one), then keep the ordered-unique [N] markers
    that map to a retrieved fact (drop invented ones)."""
    text = _URL_RE.sub("", raw).strip()
    cited: list[int] = []
    for m in _MARKER_RE.findall(text):
        n = int(m)
        if n in marker_map and n not in cited:
            cited.append(n)
    return text, cited


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
