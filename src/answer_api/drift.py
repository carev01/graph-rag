"""DRIFT search: broad->deep. Primer on the :Community report layer drafts a
preliminary answer + follow-up queries; each follow-up runs a graph-biased local
fact search; a final synthesis merges the evidence into one cited answer.
Citations are graph traversal (design decision #2): the LLM emits only [N]
markers / fact UUIDs, never a URL."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from answer_api.search import search_local
from answer_api.synthesize import (
    _build_citations, _complete_or_none, _finalize_answer, _usable_content, answer_local,
)
from answer_api.global_search import shortlist_communities, _extract_json
from graph_extract.provenance import Provenance

logger = logging.getLogger(__name__)

_REFUSAL = "I don't have enough information to answer that from the available sources."


@dataclass
class FollowUp:
    query: str
    community_id: str | None
    iteration: int = 1


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
                  max_followups, group_id):
    hits = await shortlist_communities(driver, embedder, q, level=level, k=k,
                                       group_id=group_id)
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
        return "", [FollowUp(query=q, community_id=None, iteration=1)], hits
    preliminary = str(obj.get("preliminary_answer", "")).strip()
    fus = _parse_followups(obj, hit_ids, max_followups, iteration=1)
    if not fus:
        fus = [FollowUp(query=q, community_id=None, iteration=1)]
    return preliminary, fus, hits


async def _top_member_entity(driver, group_id, community_id) -> str | None:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (c:Community {group_id:$g, community_id:$cid})<-[:IN_COMMUNITY]-(e:Entity) "
            "OPTIONAL MATCH (e)-[rel:RELATES_TO {group_id:$g}]-() "
            "WITH e, count(rel) AS deg ORDER BY deg DESC LIMIT 1 "
            "RETURN e.uuid AS uuid", g=group_id, cid=community_id)
        rec = await r.single()
        return rec["uuid"] if rec else None


async def _run_followup(graphiti, driver, fu: FollowUp, *, k, group_id) -> list[dict]:
    center = (await _top_member_entity(driver, group_id, fu.community_id)
              if fu.community_id else None)
    res = await search_local(graphiti, driver, q=fu.query, k=k,
                             center_node_uuid=center, group_id=group_id)
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
    "QUESTION: {q}\n\nDRAFT: {draft}\n\nFACTS:\n{facts}\n\nAnswer:"
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
                      facts) -> tuple[str, list[dict]]:
    facts = _dedup_facts(facts)
    marker_map = {i: f for i, f in enumerate(facts, 1)}
    facts_block = "\n".join(f"[{i}] {f['fact']}" for i, f in marker_map.items())
    raw = await _complete_or_none(
        synth_client, synth_model,
        _SYNTH_PROMPT.format(refusal=_REFUSAL, q=q, draft=preliminary_answer or "(none)",
                             facts=facts_block),
        max_tokens=3000)
    if raw is None:
        return _REFUSAL, []
    answer, cited = _finalize_answer(raw, marker_map)
    resolved = await Provenance(driver).resolve_citations(
        [marker_map[m]["fact_uuid"] for m in cited])
    citations = _build_citations(cited, marker_map, resolved)
    return answer, citations


async def drift_search(graphiti, driver, embedder, synth_client, synth_model, *,
                       q, level, iterations, primer_k, max_followups, followup_k,
                       group_id) -> dict:
    rounds = max(1, min(iterations, 2))
    primed = await _primer(embedder, synth_client, synth_model, driver, q=q,
                           level=level, k=primer_k, max_followups=max_followups,
                           group_id=group_id)
    if primed is None:
        res = await answer_local(graphiti, driver, synth_client, synth_model,
                                 q=q, group_id=group_id)
        res["degraded"] = "no-primer-communities"
        return res
    preliminary, followups, hits = primed
    hit_ids = {h.community_id for h in hits}
    facts: list[dict] = []
    executed: list[FollowUp] = []
    for fu in followups:
        try:
            facts.extend(await _run_followup(graphiti, driver, fu, k=followup_k,
                                             group_id=group_id))
        except Exception:
            logger.warning("drift follow-up failed: %s", fu.query, exc_info=True)
        executed.append(fu)
    round1_facts = _dedup_facts(facts)
    if rounds == 2 and round1_facts:
        refined = await _refine_followups(synth_client, synth_model, q=q,
                                          facts=round1_facts,
                                          max_followups=max_followups, hit_ids=hit_ids)
        for fu in refined:
            try:
                facts.extend(await _run_followup(graphiti, driver, fu, k=followup_k,
                                                 group_id=group_id))
            except Exception:
                logger.warning("drift follow-up failed: %s", fu.query, exc_info=True)
            executed.append(fu)
    follow_ups_meta = [{"query": fu.query, "community_id": fu.community_id,
                        "iteration": fu.iteration} for fu in executed]
    communities_used = [{"community_id": h.community_id, "title": h.title} for h in hits]
    deduped = _dedup_facts(facts)
    if not deduped:
        return {"query": q, "answer": _REFUSAL, "citations": [],
                "follow_ups": follow_ups_meta, "communities_used": communities_used}
    answer, citations = await _synthesize(synth_client, synth_model, driver, q=q,
                                          preliminary_answer=preliminary, facts=deduped)
    return {"query": q, "answer": answer, "citations": citations,
            "follow_ups": follow_ups_meta, "communities_used": communities_used}
