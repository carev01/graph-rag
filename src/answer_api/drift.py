"""DRIFT search: broad->deep. Primer on the :Community report layer drafts a
preliminary answer + follow-up queries; each follow-up runs a graph-biased local
fact search; a final synthesis merges the evidence into one cited answer.
Citations are graph traversal (design decision #2): the LLM emits only [N]
markers / fact UUIDs, never a URL."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from answer_api.attribution import ATTRIBUTION_RULES, applies_to, fact_line
from answer_api.scope import Scope
from answer_api.search import search_local
from answer_api.shape import (Facet, could_not_check_note, group_by_facet,
                              raise_if_cancelled)
from answer_api.synthesize import (
    _build_citations, _complete_or_none, _finalize_answer, _usable_content, answer_local,
)
from answer_api.global_search import (
    RerankStats, _extract_json, _with_disclaimer, shortlist_communities,
)
from graph_extract.config import ExtractSettings
from graph_extract.provenance import Provenance

logger = logging.getLogger(__name__)

_REFUSAL = "I don't have enough information to answer that from the available sources."


@dataclass
class FollowUp:
    query: str
    community_id: str | None
    iteration: int = 1
    scope: Scope | None = None   # the facet scope that runs it; None = request scope


def _parse_followups(obj: dict, hit_ids: set[str], max_followups: int,
                     iteration: int) -> list[FollowUp]:
    scored: list[tuple[float, FollowUp]] = []
    for item in obj.get("follow_ups") or []:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        cid = item.get("community_id")
        cid = cid if (isinstance(cid, str) and cid in hit_ids) else None
        try:
            rel = float(item.get("relevance", 0) or 0)
        except (TypeError, ValueError):
            rel = 0.0
        scored.append((rel, FollowUp(query=query, community_id=cid, iteration=iteration)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [fu for _, fu in scored[:max_followups]]


_PRIMER_PROMPT = (
    "You are starting a broad investigation of a QUESTION about backup products. "
    "Use the COMMUNITY themes below as leads (not as the final answer). Draft a "
    "brief preliminary answer and 3-6 targeted follow-up questions that would "
    "retrieve concrete supporting facts. Tag each follow-up with the community_id "
    "it draws from, or null. Respond with ONLY JSON: {{\"preliminary_answer\": str, "
    "\"follow_ups\": [{{\"query\": str, \"community_id\": str|null, "
    "\"relevance\": 0-10}}]}}. Do NOT write URLs.\n\n"
    "QUESTION: {q}\n\nCOMMUNITIES:\n{blocks}"
)


async def _primer(embedder, synth_client, synth_model, driver, *, q, level, k,
                  max_followups, group_id, settings, stats: RerankStats | None = None,
                  scope: Scope | None = None, facets: Sequence[Facet] | None = None,
                  top_n: int | None = None):
    """Shortlist communities and draft follow-ups. Returns `None` (no communities)
    or `(preliminary, follow_ups, hits, degraded)`. With >= 2 facets each facet
    shortlists in its own scope; a follow-up drawn from a facet's community runs in
    that facet's scope, and a facet left without a follow-up gets the original
    question scoped to it. A facet whose shortlist raised is listed in `degraded`
    (never treated as having nothing to say) and gets no follow-up."""
    facet_of: dict[str, Facet] = {}
    degraded: list[str] = []
    if facets is not None and len(facets) >= 2:
        found = await asyncio.gather(*[
            shortlist_communities(driver, embedder, q, level=level, k=k,
                                  group_id=group_id, settings=settings, stats=stats,
                                  scope=f.scope, top_n=top_n)
            for f in facets], return_exceptions=True)
        raise_if_cancelled(found)   # cancellation propagates; only Exceptions remain
        hits = []
        for f, fh in zip(facets, found):
            if isinstance(fh, BaseException):
                logger.warning("drift facet %r primer failed: %s", f.label, fh)
                degraded.append(f.label)
                continue
            for h in fh:
                if h.community_id not in facet_of:
                    facet_of[h.community_id] = f
                    hits.append(h)
    else:
        hits = await shortlist_communities(driver, embedder, q, level=level, k=k,
                                           group_id=group_id, settings=settings,
                                           stats=stats, scope=scope)
    if not hits:
        return None
    blocks = "\n".join(f'- {h.community_id} "{h.title}": {h.summary}' for h in hits)
    obj: dict | None = None
    for _ in range(2):
        resp = await synth_client.chat.completions.create(
            model=synth_model, temperature=0, max_tokens=2000,
            messages=[{"role": "user",
                       "content": _PRIMER_PROMPT.format(q=q, blocks=blocks)}])
        obj = _extract_json(_usable_content(resp) or "")
        if obj is not None:
            break
    hit_ids = {h.community_id for h in hits}
    if obj is None:
        fus = [FollowUp(query=q, community_id=None, iteration=1)]
        preliminary = ""
    else:
        preliminary = str(obj.get("preliminary_answer", "")).strip()
        fus = _parse_followups(obj, hit_ids, max_followups, iteration=1) \
            or [FollowUp(query=q, community_id=None, iteration=1)]
    if facet_of:
        fus = [FollowUp(query=fu.query, community_id=fu.community_id,
                        iteration=fu.iteration,
                        scope=facet_of[fu.community_id].scope
                        if fu.community_id in facet_of else None) for fu in fus]
        covered = {fu.scope for fu in fus if fu.scope is not None}
        for f in facets or []:
            if f.label not in degraded and f.scope not in covered:
                fus.append(FollowUp(query=q, community_id=None, iteration=1,
                                    scope=f.scope))
    return preliminary, fus, hits, degraded


async def _top_member_entity(driver, group_id, community_id) -> str | None:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (c:Community {group_id:$g, community_id:$cid})<-[:IN_COMMUNITY]-(e:Entity) "
            "OPTIONAL MATCH (e)-[rel:RELATES_TO {group_id:$g}]-() "
            "WITH e, count(rel) AS deg ORDER BY deg DESC LIMIT 1 "
            "RETURN e.uuid AS uuid", g=group_id, cid=community_id)
        rec = await r.single()
        return rec["uuid"] if rec else None


async def _run_followup(graphiti, driver, fu: FollowUp, *, k, group_id,
                        scope: Scope | None = None, fetch_cap: int = 450,
                        scan_max_episodes: int = 6000,
                        settings: ExtractSettings | None = None) -> list[dict]:
    center = (await _top_member_entity(driver, group_id, fu.community_id)
              if fu.community_id else None)
    res = await search_local(graphiti, driver, q=fu.query, k=k, scope=scope,
                             center_node_uuid=center, group_id=group_id,
                             fetch_cap=fetch_cap, scan_max_episodes=scan_max_episodes,
                             settings=settings)
    return res["results"]


_REFINE_PROMPT = (
    "You are investigating a QUESTION and have gathered these FACTS. Draft up to "
    "{n} refined follow-up questions that fill the biggest remaining gaps. Tag each "
    "with a community_id from the list, or null. Respond with ONLY JSON: "
    "{{\"follow_ups\": [{{\"query\": str, \"community_id\": str|null, "
    "\"relevance\": 0-10}}]}}. Do NOT write URLs.\n\n"
    "QUESTION: {q}\n\nCOMMUNITY IDS: {cids}\n\nFACTS:\n{facts}"
)


async def _refine_followups(synth_client, synth_model, *, q, facts, max_followups,
                            hit_ids) -> list[FollowUp]:
    facts_block = "\n".join(f"- {f['fact']}" for f in facts)
    obj: dict | None = None
    for _ in range(2):
        resp = await synth_client.chat.completions.create(
            model=synth_model, temperature=0, max_tokens=1500,
            messages=[{"role": "user", "content": _REFINE_PROMPT.format(
                n=max_followups, q=q, cids=sorted(hit_ids), facts=facts_block)}])
        obj = _extract_json(_usable_content(resp) or "")
        if obj is not None:
            break
    if obj is None:
        return []
    return _parse_followups(obj, set(hit_ids), max_followups, iteration=2)


_SYNTH_PROMPT = (
    "Answer the QUESTION using ONLY the numbered FACTS, refining the DRAFT where "
    "the facts support it. Cite every claim inline with its [N] marker. Do NOT use "
    "outside knowledge. Do NOT write any URL. If the facts do not answer the "
    "question, reply exactly: \"{refusal}\"\n\n"
    + ATTRIBUTION_RULES +
    "\nQUESTION: {q}\n\nDRAFT: {draft}\n\nFACTS:\n{facts}\n\nAnswer:"
)

_SYNTH_FACETS_PROMPT = (
    "Answer the QUESTION using ONLY the numbered FACTS, which are grouped under a "
    "heading for each side of the question, refining the DRAFT where the facts support "
    "it. The question has these sides: {labels}. Address EVERY side, in that order. If "
    "a side shows \"(no facts retrieved)\", say in one sentence that the retrieved "
    "documentation does not cover it, and do not guess. Cite every claim inline with "
    "its [N] marker. Do NOT use outside knowledge. Do NOT write any URL. If no side has "
    "facts that answer the question, reply exactly: \"{refusal}\"\n\n"
    + ATTRIBUTION_RULES +
    "\nQUESTION: {q}\n\nDRAFT: {draft}\n\nFACTS:\n{facts}\n\nAnswer:"
)


def _dedup_facts(facts: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for f in facts:
        u = f["fact_uuid"]
        if u not in seen:
            seen.add(u)
            out.append(f)
    return out


async def _synthesize(synth_client, synth_model, driver, *, q, preliminary_answer,
                      facts, facets: Sequence[Facet] | None = None
                      ) -> tuple[str, list[dict]]:
    """With >= 2 `facets`, facts are grouped under one heading per facet (in facet
    order; a facet with none shows "(no facts retrieved)", unmatched facts go under
    "Other") and the prompt requires every side to be addressed. Callers pass only
    facets whose retrieval worked -- a degraded facet must not get a heading."""
    facts = _dedup_facts(facts)
    if facets is not None and len(facets) >= 2:
        groups = group_by_facet(facts, facets)
        grouped_ids = {f["fact_uuid"] for _, xs in groups for f in xs}
        ordered = [f for _, xs in groups for f in xs]
        rest = [f for f in facts if f["fact_uuid"] not in grouped_ids]
        marker_map = {i: f for i, f in enumerate(ordered + rest, 1)}
        index = {f["fact_uuid"]: i for i, f in marker_map.items()}
        parts = []
        for facet, xs in groups:
            lines = [fact_line(index[f["fact_uuid"]], f["fact"], f.get("sources", []))
                     for f in xs] or ["(no facts retrieved)"]
            parts.append(f"=== {facet.label} ===\n" + "\n".join(lines))
        if rest:
            parts.append("=== Other ===\n" + "\n".join(
                fact_line(index[f["fact_uuid"]], f["fact"], f.get("sources", []))
                for f in rest))
        prompt = _SYNTH_FACETS_PROMPT.format(
            labels="; ".join(f.label for f in facets), refusal=_REFUSAL, q=q,
            draft=preliminary_answer or "(none)", facts="\n\n".join(parts))
    else:
        marker_map = {i: f for i, f in enumerate(facts, 1)}
        facts_block = "\n".join(
            fact_line(i, f["fact"], f.get("sources", [])) for i, f in marker_map.items())
        prompt = _SYNTH_PROMPT.format(refusal=_REFUSAL, q=q,
                                      draft=preliminary_answer or "(none)",
                                      facts=facts_block)
    raw = await _complete_or_none(synth_client, synth_model, prompt, max_tokens=3000)
    if raw is None:
        return _REFUSAL, []
    answer, cited = _finalize_answer(raw, marker_map)
    resolved = await Provenance(driver).resolve_citations(
        [marker_map[m]["fact_uuid"] for m in cited])
    citations = _build_citations(cited, marker_map, resolved)
    return answer, citations


async def drift_search(graphiti, driver, embedder, synth_client, synth_model, *,
                       q, level, iterations, primer_k, max_followups, followup_k,
                       group_id, settings, scope: Scope | None = None,
                       facets: Sequence[Facet] | None = None) -> dict:
    """With >= 2 `facets` the primer shortlists per facet and every follow-up runs
    in its facet's scope (see `_primer`). A facet is reported in `degraded_facets`
    -- never rendered as having no facts -- when its primer shortlist raised or
    when every follow-up that ran in its scope raised; the "Could not check" note
    is appended to the answer on every return path. Fewer than 2 facets: unchanged."""
    multi = facets is not None and len(facets) >= 2
    rounds = max(1, min(iterations, 2))
    stats = RerankStats()
    primed = await _primer(embedder, synth_client, synth_model, driver, q=q,
                           level=level, k=primer_k, max_followups=max_followups,
                           group_id=group_id, settings=settings, stats=stats,
                           scope=scope, facets=facets,
                           top_n=settings.global_facet_top_n if multi else None)
    if primed is None:
        res = await answer_local(graphiti, driver, synth_client, synth_model,
                                 q=q, group_id=group_id, scope=scope, facets=facets,
                                 min_per_facet=settings.local_min_per_facet,
                                 fetch_cap=settings.local_scoped_fetch_cap,
                                 scan_max_episodes=settings.local_scoped_scan_max_episodes,
                                 settings=settings)
        res["degraded"] = "no-primer-communities"
        return res
    preliminary, followups, hits, degraded = primed
    hit_ids = {h.community_id for h in hits}
    facts: list[dict] = []
    executed: list[FollowUp] = []
    tried: set[Scope] = set()       # facet scopes that ran at least one follow-up
    worked: set[Scope] = set()      # facet scopes with at least one follow-up that returned

    async def _execute(fu: FollowUp) -> None:
        try:
            facts.extend(await _run_followup(
                graphiti, driver, fu, k=followup_k, group_id=group_id,
                scope=fu.scope if fu.scope is not None else scope,
                fetch_cap=settings.local_scoped_fetch_cap,
                scan_max_episodes=settings.local_scoped_scan_max_episodes,
                settings=settings))
            if fu.scope is not None:
                worked.add(fu.scope)
        except Exception:
            logger.warning("drift follow-up failed: %s", fu.query, exc_info=True)
        if fu.scope is not None:
            tried.add(fu.scope)
        executed.append(fu)

    for fu in followups:
        await _execute(fu)
    round1_facts = _dedup_facts(facts)
    if rounds == 2 and round1_facts:
        refined = await _refine_followups(synth_client, synth_model, q=q,
                                          facts=round1_facts,
                                          max_followups=max_followups, hit_ids=hit_ids)
        # A refined follow-up drawn from a facet's community stays in that facet.
        scope_of = {fu.community_id: fu.scope for fu in followups
                    if fu.community_id is not None and fu.scope is not None}
        for fu in refined:
            if fu.community_id in scope_of:
                fu.scope = scope_of[fu.community_id]
            await _execute(fu)
    live_facets: list[Facet] | None = None
    if multi:
        assert facets is not None
        for f in facets:
            if f.label not in degraded and f.scope in tried and f.scope not in worked:
                logger.warning("drift facet %r: every follow-up failed", f.label)
                degraded.append(f.label)
        degraded = [f.label for f in facets if f.label in degraded]
        live_facets = [f for f in facets if f.label not in degraded]
    note = could_not_check_note(degraded)
    extra: dict = {"degraded_facets": degraded} if multi else {}
    follow_ups_meta = [{"query": fu.query, "community_id": fu.community_id,
                        "iteration": fu.iteration} for fu in executed]
    communities_used = [{"community_id": h.community_id, "title": h.title} for h in hits]
    deduped = _dedup_facts(facts)
    if not deduped:
        return {"query": q, "answer": _REFUSAL + (" " + note if note else ""),
                "citations": [],
                "follow_ups": follow_ups_meta, "communities_used": communities_used,
                "degraded": stats.degraded, "applies_to": [], **extra}
    answer, citations = await _synthesize(synth_client, synth_model, driver, q=q,
                                          preliminary_answer=preliminary, facts=deduped,
                                          facets=live_facets)
    # Guard against _with_disclaimer's own refusal check, which compares against
    # global_search's _REFUSAL string -- textually distinct from ours. Checking
    # against OUR _REFUSAL here (not global_search's) is what keeps a DRIFT
    # refusal from silently picking up a disclaimer.
    if answer != _REFUSAL:
        answer = _with_disclaimer(answer, stats.degraded)
        if note:
            answer = f"{answer}\n\n{note}"
    elif note:
        answer = f"{answer} {note}"
    return {"query": q, "answer": answer, "citations": citations,
            "follow_ups": follow_ups_meta, "communities_used": communities_used,
            "degraded": stats.degraded, "applies_to": applies_to(citations), **extra}
