"""Global map-reduce search over the :Community report layer. Shortlist reports
by query-embedding cosine + rating, MAP each to query-relevant key points (with
validated fact IDs), REDUCE into one cited answer. Citations are graph traversal
(design decision #2): the LLM only ever emits fact UUIDs / [N] markers."""
from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass, replace

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.provenance import Provenance
from graph_extract.usage import bounded_llm_client
from answer_api.attribution import ATTRIBUTION_RULES, applies_to, fact_line, in_scope
from answer_api.rerank import rerank, rerank_configured
from answer_api.scope import Scope
from answer_api.synthesize import (
    _build_citations, _complete_or_none, _finalize_answer, _usable_content,
)

logger = logging.getLogger(__name__)

_REFUSAL = "I don't have enough thematic coverage to answer that from the community reports."

# Reader-visible degradation notices. The `degraded` envelope field is
# machine-readable only -- nothing shows it to the person reading the answer.
# Keyed by reason so other degraded modes can adopt it; only rerank-unavailable
# is wired up. drift.py's "no-primer-communities" is deliberately NOT here: the
# reader gets a valid local answer, not a less accurate one.
_DEGRADED_DISCLAIMERS = {
    "rerank-unavailable": (
        "Note: relevance ranking was unavailable for this answer, so the sources it "
        "draws on may be less relevant than usual. Please verify against the cited "
        "sources."),
}


def _with_disclaimer(answer: str, reason: str | None) -> str:
    """Prepend a reader-visible notice to a degraded answer.

    Called AFTER _finalize_answer: marker stripping and whitespace repair must not
    treat this text as answer prose. The notice carries no [N] markers and no URL,
    so it cannot be mistaken for cited content.
    """
    note = _DEGRADED_DISCLAIMERS.get(reason or "")
    if not note or not answer.strip() or answer.strip() == _REFUSAL:
        return answer
    return f"{note}\n\n{answer}"


@dataclass
class CommunityHit:
    community_id: str
    title: str
    summary: str
    level: int
    rating: float
    cited_fact_uuids: list[str]
    full_report: str
    similarity: float
    relevance: float | None = None       # rerank score; None when never reranked


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _rank_hits(query_vec: list[float], rows: list[dict], *, k: int,
               rating_boost: float) -> list[CommunityHit]:
    scored: list[tuple[float, CommunityHit]] = []
    for r in rows:
        if not r.get("cited_fact_uuids"):
            continue                        # nothing citable -> useless for a cited answer
        sim = _cosine(query_vec, r["embedding"])
        score = sim + rating_boost * (r.get("rating") or 0.0) / 10.0
        scored.append((score, CommunityHit(
            community_id=r["community_id"], title=r["title"], summary=r["summary"],
            level=r["level"], rating=r.get("rating") or 0.0,
            cited_fact_uuids=list(r["cited_fact_uuids"]), full_report=r["full_report"],
            similarity=sim)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [h for _, h in scored[:k]]


@dataclass
class RerankStats:
    """Out-param so the two shortlist_communities call sites keep their shape."""
    degraded: str | None = None


def _apply_rerank(hits: list[CommunityHit],
                  scored: list[tuple[int, float]] | None,
                  *, top_n: int, floor: float,
                  stats: RerankStats) -> list[CommunityHit]:
    """Cut the candidate list using rerank scores.

    `scored is None` means the reranker COULD NOT SCORE: fall back to cosine order,
    capped at top_n, and mark the result degraded so the caller can tell the reader.
    An empty list is different -- it means the reranker scored and nothing cleared
    the floor, which is a real verdict and must reach the refusal path.
    """
    if scored is None:
        stats.degraded = "rerank-unavailable"
        return hits[:top_n]
    out: list[CommunityHit] = []
    for idx, score in scored[:top_n]:
        if score < floor:
            continue
        # `replace`, not mutation: a caller may hold the same candidate list, and
        # silently rewriting its objects is the kind of surprise that is very hard
        # to trace later. Same reason report.py's _with_findings uses it.
        out.append(replace(hits[idx], relevance=score))
    return out


def _in_scope_ci(sources: list[dict], scope: Scope) -> bool:
    """Case-insensitive wrapper around `attribution.in_scope` (R3): a `Scope`
    reaching global_search may carry either the ScopeResolver's canonical
    case or raw API-param case, while `sources[].vendor/product` always carry
    the canonical structural case. Fold both sides rather than loosen
    `in_scope`'s exact-match contract, which local/timeline still rely on."""
    if scope.is_empty():
        return True
    folded_scope = Scope(tuple(v.lower() for v in scope.vendors),
                         tuple(p.lower() for p in scope.products), scope.source)
    folded_sources = [{"vendor": (s.get("vendor") or "").lower(),
                       "product": (s.get("product") or "").lower()} for s in sources]
    return in_scope(folded_sources, folded_scope)


async def _scope_shares(driver, group_id: str, hits: list[CommunityHit],
                        scope: Scope) -> dict[str, float]:
    """Community_id -> in-scope share of its `cited_fact_uuids` (spec §4.3 step
    2): the fraction of a community's cited facts whose provenance sources
    intersect the scope's vendor(s)/product(s). A community with no cited
    facts, or whose cited facts resolve to no structural vendor/product at
    all, gets 0.0 rather than being silently missing from the returned dict.

    R4 (post-implementation review): this used to be a per-row custom Cypher
    traversal duplicating Provenance.resolve_citations's chain -- profiled at
    20.6s / ~24M dbHits over 24 communities x 20 facts against a 20k-fact
    graph, vs. 0.7s / ~21k dbHits for one resolve_citations call over the same
    480 uuids (see the fix report's PROFILE evidence). It now makes exactly
    ONE resolve_citations call, over the ordered-unique union of every hit's
    cited_fact_uuids, and computes each share in Python via `_in_scope_ci`
    (case-insensitive per R3; a fact missing from the resolve result -- no
    structural provenance at all -- counts as not in scope, matching the old
    Cypher's `v.id IS NOT NULL AND p.id IS NOT NULL` guard).

    `group_id` is accepted for call-site stability (`_keep_in_scope` passes it
    through) but unused: resolve_citations resolves by fact uuid alone.
    """
    if not hits:
        return {}
    union: list[str] = []
    seen: set[str] = set()
    for h in hits:
        for f in h.cited_fact_uuids:
            if f not in seen:
                seen.add(f)
                union.append(f)
    resolved = await Provenance(driver).resolve_citations(union) if union else {}
    shares: dict[str, float] = {}
    for h in hits:
        facts = h.cited_fact_uuids
        if not facts:
            shares[h.community_id] = 0.0
            continue
        in_scope_count = sum(
            1 for f in facts
            if _in_scope_ci(resolved.get(f, {}).get("sources", []), scope))
        shares[h.community_id] = in_scope_count / len(facts)
    return shares


async def _keep_in_scope(driver, group_id: str, candidates: list[CommunityHit],
                         scope: Scope, settings: ExtractSettings | None,
                         ) -> list[CommunityHit]:
    """Filter `candidates` to those whose in-scope share clears
    `global_scope_min_share` (spec §4.3 step 3). `settings` is only ever None
    from a caller that skipped it entirely; production callers always pass it,
    so 0.5 here is a defensive fallback, not a second source of truth."""
    min_share = settings.global_scope_min_share if settings is not None else 0.5
    shares = await _scope_shares(driver, group_id, candidates, scope)
    kept = [h for h in candidates if shares.get(h.community_id, 0.0) >= min_share]
    logger.info("global scope %s: kept %d of %d candidates", scope, len(kept),
               len(candidates))
    return kept


async def shortlist_communities(driver, embedder, q: str, *, level: int, k: int,
                                group_id: str, rating_boost: float = 0.1,
                                settings: ExtractSettings | None = None,
                                stats: RerankStats | None = None,
                                scope: Scope | None = None) -> list[CommunityHit]:
    query_vec = (await embedder.create_batch([q]))[0]
    async with driver.session() as s:
        r = await s.run(
            "MATCH (c:Community {group_id:$g, level:$lvl}) "
            "RETURN c.community_id AS community_id, c.title AS title, "
            "coalesce(c.summary,'') AS summary, c.level AS level, "
            "coalesce(c.rating,0.0) AS rating, "
            "coalesce(c.cited_fact_uuids,[]) AS cited_fact_uuids, "
            "coalesce(c.full_report,'[]') AS full_report, c.embedding AS embedding",
            g=group_id, lvl=level)
        rows = [dict(rec) async for rec in r if rec["embedding"]]
    # `scope_` (not Optional) once known non-empty, so the two `_keep_in_scope`
    # call sites below don't need to re-narrow `scope: Scope | None` for mypy.
    scope_ = scope if (scope is not None and not scope.is_empty()) else None
    scoped = scope_ is not None
    # The unreranked path must still return exactly k, whatever rerank_candidates
    # is set to, or it silently under-returns (finding: rerank_candidates < k) --
    # so only widen the pre-cut pool once we know it will actually be reranked.
    if settings is None or not rerank_configured(settings):
        pool_k = max(k, settings.global_scope_candidates) if (scoped and settings) else k
        candidates = _rank_hits(query_vec, rows, k=pool_k, rating_boost=rating_boost)
        if scope_ is not None:
            candidates = await _keep_in_scope(driver, group_id, candidates, scope_, settings)
        return candidates[:k]
    pool_k = settings.rerank_candidates
    if scoped:
        pool_k = max(pool_k, settings.global_scope_candidates)
    candidates = _rank_hits(query_vec, rows, k=pool_k, rating_boost=rating_boost)
    if scope_ is not None:
        candidates = await _keep_in_scope(driver, group_id, candidates, scope_, settings)
    st = stats if stats is not None else RerankStats()
    docs = [f"{h.title}: {h.summary}" for h in candidates]
    scored = await rerank(q, docs, top_k=settings.rerank_top_n, settings=settings)
    return _apply_rerank(candidates, scored, top_n=settings.rerank_top_n,
                         floor=settings.rerank_score_floor, stats=st)


@dataclass
class MapResult:
    community_id: str
    title: str
    relevance: float | None
    key_points: list[str]
    fact_ids: list[str]


_MAP_PROMPT = (
    "Extract from this COMMUNITY report only what answers the QUESTION. Using ONLY "
    "the report, respond with JSON: {{\"key_points\": [short strings that answer the "
    "question], \"fact_ids\": [the fact uuids from the report that support those "
    "points]}}. Include ONLY points that bear on the question, and ONLY fact_ids that "
    "support the points you listed. fact_ids MUST be uuids that appear in the report. "
    "The question may span several vendors or topics and this report may cover only one "
    "of them: return what the report says about the part it covers, as fully as the "
    "report supports, and leave the rest to other reports. Every key point must state "
    "something the report SAYS. Do NOT write a point about what the report does not "
    "contain, does not mention, or cannot compare, and do not describe the report's "
    "scope. Absence of evidence is not a finding. Return an empty key_points list "
    "({{\"key_points\": [], \"fact_ids\": []}}) only when nothing in the report bears "
    "on the question. "
    "Do NOT write URLs.\n\n"
    "QUESTION: {q}\n\nCOMMUNITY \"{title}\": {summary}\nFINDINGS: {full_report}"
)


def _map_client_and_model(settings: ExtractSettings) -> tuple[AsyncOpenAI, str]:
    base = settings.map_llm_base_url or settings.judge_base_url
    model = settings.map_llm_model or settings.judge_model
    key = settings.map_llm_api_key or settings.judge_api_key or "not-needed"
    if not base or not model:
        raise ValueError("No map model configured. Set map_llm_* or judge_* (GLM-5.2).")
    return bounded_llm_client(base, key, reasoning_effort=settings.map_reasoning_effort,
                              tier="map", capture_path=settings.llm_capture_path), model


def _extract_json(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


async def map_report(client: AsyncOpenAI, model: str, q: str,
                     hit: CommunityHit) -> MapResult | None:
    prompt = _MAP_PROMPT.format(q=q, title=hit.title, summary=hit.summary,
                                full_report=hit.full_report)
    obj: dict | None = None
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}])
        obj = _extract_json(_usable_content(resp) or "")
        if obj is not None:
            break
    if obj is None:
        return None
    valid = {u for u in hit.cited_fact_uuids}
    fact_ids = [f for f in (obj.get("fact_ids") or []) if f in valid]
    key_points = [str(p) for p in (obj.get("key_points") or [])]
    return MapResult(community_id=hit.community_id, title=hit.title,
                     relevance=hit.relevance, key_points=key_points, fact_ids=fact_ids)


async def _fact_texts(driver, group_id: str, fact_uuids: list[str]) -> dict[str, str]:
    """Fact text for the union of the map step's fact_ids, in ONE round trip.

    BACKLOG 0b: the reducer used to be handed key points plus a bag of markers
    and never saw a fact, so it numbered sentences by position. It now sees
    `[N] <fact>` lines, which needs the text. A fact with no text (edge gone,
    property null) is simply absent from the result -- the caller drops it
    loudly rather than rendering an empty line under a legitimate marker.
    """
    if not fact_uuids:
        return {}
    async with driver.session() as s:
        r = await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $u "
            "RETURN f.uuid AS uuid, f.fact AS fact", g=group_id, u=list(fact_uuids))
        return {rec["uuid"]: rec["fact"] async for rec in r
                if rec["fact"] and str(rec["fact"]).strip()}


def _render_blocks(results: list[MapResult], fact_to_marker: dict[str, int],
                   texts: dict[str, str], sources: dict[str, list[dict]]) -> list[str]:
    """One reduce block per community: `COMMUNITY "title":` then a `[N] <fact>`
    line per selected fact, in the map step's order, numbered by the SAME
    fact_to_marker every citation downstream resolves through. A fact whose
    text is missing is skipped (see _fact_texts); a community left with no
    line contributes no block at all.

    Each line is labelled `(Vendor · Product)` via `fact_line` (spec §3.1);
    `sources` is empty for a fact with no resolved provenance, in which case
    `fact_line` renders the bare `[N] <fact>` line, unchanged."""
    blocks = []
    for m in results:
        lines = [fact_line(fact_to_marker[f], texts[f], sources.get(f, []))
                 for f in m.fact_ids if f in texts]
        if not lines:
            continue
        # m.relevance is the rerank score; None when the shortlist was never
        # reranked (the default deployment). Printing "(relevance 0.0)" then
        # would be false information handed to the reducer, not a placeholder
        # -- omit the parenthetical entirely rather than invent a number.
        suffix = f" (relevance {m.relevance})" if m.relevance is not None else ""
        blocks.append(f"COMMUNITY \"{m.title}\"{suffix}:\n" + "\n".join(lines))
    return blocks


_REDUCE_PROMPT = (
    "Answer the QUESTION by synthesizing across these community findings, organized "
    "by theme and vendor. Each finding is shown as a marker [N] followed by the fact "
    "it refers to.\n"
    + ATTRIBUTION_RULES +
    "Rules:\n"
    # BACKLOG 0b: the findings are now the facts themselves, marker-bound, so
    # the citation rule can say which marker a claim takes: the one printed
    # beside the fact the claim was drawn from.
    "- Cite every claim with the [N] marker of the fact it rests on: a claim drawn "
    "from the fact shown after [N] must cite that [N]. A sentence with no marker "
    "is not allowed.\n"
    # BACKLOG 0d: a range like [1]-[26] finalizes to TWO citations (no expander,
    # by design -- a 26-marker span is a guess, not a citation), so the model
    # must name every marker it means. Tell it why: only what it writes counts.
    "- Write each marker individually: [1] [2] [3]. Never write a range or span "
    "such as [1]-[3]; only the markers you write out are cited.\n"
    "- Use ONLY these findings. Do NOT use outside knowledge.\n"
    "- Do NOT write any URL.\n"
    "- Do NOT comment on what the findings do not contain, and do not explain what "
    "you cannot compare. Absence of evidence is not a finding.\n"
    "- Let the evidence set the length. Say what the findings support and then stop; "
    "do not pad, hedge, or restate.\n"
    "- If the findings do not support an answer, reply exactly: "
    "\"" + _REFUSAL + "\"\n\nQUESTION: {q}\n\nFINDINGS:\n{blocks}\n\nAnswer:"
)


async def global_search(driver, embedder, map_client: AsyncOpenAI, map_model: str,
                        synth_client: AsyncOpenAI, synth_model: str, *, q: str,
                        level: int, k: int, group_id: str,
                        settings: ExtractSettings,
                        scope: Scope | None = None) -> dict:
    stats = RerankStats()
    hits = await shortlist_communities(driver, embedder, q, level=level, k=k,
                                       group_id=group_id, settings=settings, stats=stats,
                                       scope=scope)
    if not hits:
        return {"query": q, "answer": _REFUSAL, "citations": [], "communities_used": [],
                "degraded": stats.degraded, "applies_to": []}
    maps = await asyncio.gather(
        *[map_report(map_client, map_model, q, h) for h in hits],
        return_exceptions=True)
    results: list[MapResult] = []
    for m in maps:
        if isinstance(m, BaseException):
            logger.warning("global map failed: %s", m)
        elif m is not None:
            results.append(m)
    if not results:
        return {"query": q, "answer": _REFUSAL, "citations": [], "communities_used": [],
                "degraded": stats.degraded, "applies_to": []}
    # Resolve provenance ONCE, over the union of every map result's fact_ids
    # BEFORE the scope filter or the numbering below (plan ruling 5): the same
    # `resolved` supplies the in-scope check, the fact-line labels, AND the
    # final citations -- never a second resolve_citations call. Resolved even
    # when unscoped: labels need it regardless (spec §3.1).
    all_fact_ids = sorted({fid for m in results for fid in m.fact_ids})
    resolved = await Provenance(driver).resolve_citations(all_fact_ids)
    if scope is not None and not scope.is_empty():
        # Plan ruling 1: the map step's INPUT is report prose, not fact lines,
        # so the scope filter applies to its OUTPUT -- each MapResult's
        # fact_ids -- before numbering, not to the rendered blocks after.
        # `replace`, not mutation, for the same reason _apply_rerank uses it.
        results = [replace(m, fact_ids=[
            f for f in m.fact_ids
            if _in_scope_ci(resolved.get(f, {}).get("sources", []), scope)])
            for m in results]
        # Review finding B: a community that loses EVERY fact to the scope
        # filter fed nothing into the reduce step and must not appear in
        # communities_used, or the router's global->local fallback (which
        # triggers on `not communities_used`) never fires when the map step
        # happened to pick only out-of-scope facts for every shortlisted
        # community. Scoped path only -- the unscoped path's semantics are
        # untouched: a MapResult with naturally empty fact_ids has always
        # stayed in communities_used there, and still does.
        results = [m for m in results if m.fact_ids]
    # Built once, after any scope filtering/drop above, and reused by every
    # return below: this is "the communities that fed the reduce step", which
    # stays true whether or not the reduce LLM call itself later succeeds.
    communities_used = [{"community_id": m.community_id, "title": m.title,
                         "relevance": m.relevance} for m in results]
    # number the ordered-unique union of fact ids -> marker_map
    marker_map: dict[int, dict] = {}
    fact_to_marker: dict[str, int] = {}
    for m in results:
        for fid in m.fact_ids:
            if fid not in fact_to_marker:
                idx = len(fact_to_marker) + 1
                fact_to_marker[fid] = idx
                marker_map[idx] = {"fact_uuid": fid}
    # BACKLOG 0b: the reducer gets `[N] <fact>` lines, not key points plus a
    # marker bag, so a claim can cite the fact it actually rests on. The
    # numbering above is untouched; only what is printed beside it changed.
    texts = await _fact_texts(driver, group_id, list(fact_to_marker))
    missing = [fid for fid in fact_to_marker if fid not in texts]
    if missing:
        # Not silent: a marker with no fact behind it would either render as
        # an empty line or be cited blind. Drop it from the block AND from the
        # marker map, and say how many and which.
        logger.warning(
            "global reduce: %d of %d selected facts have no readable text and were "
            "dropped from the findings: %s", len(missing), len(fact_to_marker), missing)
        marker_map = {n: v for n, v in marker_map.items() if v["fact_uuid"] in texts}
    if not marker_map:
        logger.warning("global reduce: no fact text for any selected fact; refusing "
                       "without calling the reducer")
        return {"query": q, "answer": _REFUSAL, "citations": [],
                "communities_used": communities_used, "degraded": stats.degraded,
                "applies_to": []}
    sources_by_fact = {fid: resolved.get(fid, {}).get("sources", []) for fid in fact_to_marker}
    blocks = _render_blocks(results, fact_to_marker, texts, sources_by_fact)
    raw = await _complete_or_none(
        synth_client, synth_model,
        _REDUCE_PROMPT.format(q=q, blocks="\n\n".join(blocks)), max_tokens=3000)
    if raw is None:
        # Communities WERE shortlisted and mapped -- only the reduce LLM call
        # failed. Report the same list the success path below reports; `[]`
        # here would collapse "coverage existed, the LLM failed" into "no
        # thematic coverage existed", which is a different, false statement.
        return {"query": q, "answer": _REFUSAL, "citations": [],
                "communities_used": communities_used, "degraded": stats.degraded,
                "applies_to": []}
    answer, cited = _finalize_answer(raw, marker_map)
    citations = _build_citations(cited, marker_map, resolved)
    answer = _with_disclaimer(answer, stats.degraded)
    return {"query": q, "answer": answer, "citations": citations,
            "degraded": stats.degraded, "communities_used": communities_used,
            "applies_to": applies_to(citations)}
