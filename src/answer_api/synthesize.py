from __future__ import annotations

import logging
import re

from openai import AsyncOpenAI

from answer_api import search as search_mod
from graph_extract.config import ExtractSettings
from graph_extract.usage import instrument, usable_content as _usable_content

logger = logging.getLogger(__name__)

# Case-insensitive: a URL must NEVER survive (design-decision #2). Stop the
# match at brackets so a URL written flush against a marker ("https://x[1]")
# doesn't swallow the legitimate [1] citation along with it.
_URL_RE = re.compile(r"https?://[^\s\[\]]+", re.IGNORECASE)
_MARKER_RE = re.compile(r"\[(\d+)\]")
# Models emit a comma list ("[1, 2]") or inner-padded single marker ("[ 9 ]")
# constantly -- `_MARKER_RE` only matches a single bare `[N]`, so left
# unnormalised these forms are invisible to it: the marker stays visible in
# the prose while `cited` sees nothing, breaking the guarantee that a visible
# marker always has a citation entry. Normalising to one-marker-per-bracket
# lets all the existing resolve/strip logic below handle it unchanged.
# `[9a]` is intentionally NOT matched/normalised here -- it is not a marker
# form the prompts ask models to emit, so it is out of scope for this pass
# and is left as ordinary (if odd) prose.
_MARKER_LIST_RE = re.compile(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]")
# Placeholder for an unresolvable marker mid-pass. Never allowed to survive
# into an answer -- _strip_markers strips any pre-existing one on entry.
_SENTINEL = "\x00"
# A separator is only orphaned -- and goes with a removed marker -- when the
# token on the OTHER side of it is ALSO a marker (a surviving [N] or another
# sentinel). Spec 3.3 licenses removal only "where a marker was removed from
# each side"; a dash/em-dash with a marker on just one side is ordinary prose
# punctuation ("enforced [9] — and ...", "Point-in-time [9]-recovery") and
# must survive. `_PAIRED_SEP_RE` uses a lookahead (not a consuming match) for
# the right-hand token so a chain like "[1]-[2]-[3]" resolves left-to-right in
# one pass without fabricating a fused "[1]-[3]" span: each separator is
# dropped only against its immediate neighbour, never both at once.
_MARKER_TOKEN_RE = r"(?:\[\d+\]|" + _SENTINEL + r")"
# Two alternatives, not one broad "marker SEP marker": a separator between two
# SURVIVING markers ("[1]-[2]") had nothing removed from either side and must
# never match here -- that is the legitimate-range case, left untouched. A
# match requires a sentinel on at least one side.
_PAIRED_SEP_RE = re.compile(
    r"(" + _SENTINEL + r")[ \t]*[-–—][ \t]*(?=" + _MARKER_TOKEN_RE + r")"
    r"|(\[\d+\])[ \t]*[-–—][ \t]*(?=" + _SENTINEL + r")")
# Once every separator that had a marker on both sides is gone, whatever
# sentinel remains is truly lone (no adjacent range partner) -- delete it and
# collapse its surrounding spaces/tabs to one, touching nothing else.
_LONE_SENTINEL_RE = re.compile(r"[ \t]*" + _SENTINEL + r"[ \t]*")

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

    Chained ranges ("[1]-[2]-[3]") rule out a single-pass regex substitution:
    replacing "[1]-[2]" first would collapse it to "[1]" and leave "-[3]"
    sitting right next to it, fabricating a "[1]-[3]" span nobody asserted.
    Instead every unresolvable marker becomes a sentinel first, then a second
    pass removes each sentinel together with any separator orphaned by its
    removal -- a separator between two surviving markers is left untouched.
    """
    text = text.replace(_SENTINEL, "")  # defensive: must never reach an answer
    text = _MARKER_RE.sub(
        lambda m: m.group(0) if int(m.group(1)) in keep else _SENTINEL, text)
    text = _PAIRED_SEP_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _LONE_SENTINEL_RE.sub(" ", text)
    # Repair spacing WITHOUT touching line breaks: this runs on every answer, and
    # paragraph structure is part of a correct one.
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def _normalise_marker_lists(text: str) -> str:
    """Rewrite a bracketed comma list into separate single markers so every
    downstream step keeps working on one uniform `[N]` shape:
    `[1, 2]` -> `[1] [2]`, `[ 9 ]` -> `[9]`, `[1]` -> `[1]` (unchanged)."""
    return _MARKER_LIST_RE.sub(
        lambda m: " ".join(f"[{n}]" for n in re.findall(r"\d+", m.group(0))), text)


def _finalize_answer(raw: str, marker_map: dict[int, dict]) -> tuple[str, list[int]]:
    """Deterministic design-decision #2 enforcement: strip any URL the model
    emitted (it must never write one), keep the ordered-unique [N] markers that
    map to a retrieved fact, and remove the ones that do not from the TEXT as
    well -- a visible marker must always correspond to a real citation."""
    text = _URL_RE.sub("", raw).strip()
    # Normalise comma-list/padded markers to single markers BEFORE `cited` is
    # computed, so `cited` sees the same uniform shape the resolve/strip logic
    # below already handles.
    text = _normalise_marker_lists(text)
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


async def _complete_or_none(client: AsyncOpenAI, model: str, prompt: str, *,
                            max_tokens: int) -> str | None:
    """Call the model once; if it returns nothing usable, retry ONCE with a
    3x larger token budget (a reasoning model can burn a small budget entirely
    on reasoning tokens) before giving up and returning None."""
    resp = await client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}])
    content = _usable_content(resp)
    if content is not None:
        return content
    finish_reason = resp.choices[0].finish_reason if resp.choices else "no-choices"
    logger.warning(
        "synthesis model %s returned no usable content (finish_reason=%s); "
        "retrying with max_tokens=%d", model, finish_reason, max_tokens * 3)

    resp = await client.chat.completions.create(
        model=model, temperature=0, max_tokens=max_tokens * 3,
        messages=[{"role": "user", "content": prompt}])
    content = _usable_content(resp)
    if content is not None:
        return content
    finish_reason = resp.choices[0].finish_reason if resp.choices else "no-choices"
    logger.warning(
        "synthesis model %s still returned no usable content after retry "
        "(finish_reason=%s); giving up", model, finish_reason)
    return None


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
    # GLM-5.2 is a reasoning model: a small cap truncates the answer to empty
    # (finish_reason='length', content=None), so give the reasoning headroom --
    # same lesson as the GLM judge -- and _complete_or_none retries once wider
    # rather than serving that empty reply as a successful blank answer.
    raw = await _complete_or_none(synth_client, synth_model,
                                  _PROMPT.format(facts=facts_block, q=q), max_tokens=3000)
    if raw is None:
        return {"query": q, "answer": _REFUSAL, "citations": [],
                "retrieved": len(results), "cited": 0}
    answer, cited = _finalize_answer(raw, marker_map)
    citations = [{"marker": m, "fact": marker_map[m]["fact"],
                  "fact_uuid": marker_map[m]["fact_uuid"],
                  "valid_at": marker_map[m].get("valid_at"),
                  "invalid_at": marker_map[m].get("invalid_at"),
                  "sources": marker_map[m]["sources"]} for m in cited]
    return {"query": q, "answer": answer, "citations": citations,
            "retrieved": len(results), "cited": len(cited)}
