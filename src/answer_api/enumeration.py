"""Enumeration answers (spec 2026-10-12 §3.4): "Which vendors ...?" checks EVERY
vendor with extracted content -- no ceiling. Per vendor: gather scoped evidence,
one cheap-tier check (documented or not, with fact ids drawn from that evidence),
then one synthesis over the documented vendors. The closing "Checked / Documented
for" line is built here, deterministically -- the LLM never decides who was checked.
A vendor whose check fails is reported as "Could not check", never as absent.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence

from answer_api.attribution import ATTRIBUTION_RULES, applies_to, fact_line
from answer_api.global_search import (_extract_json, _fact_texts, _in_scope_ci,
                                      shortlist_communities)
from answer_api.search import search_local
from answer_api.shape import Facet, could_not_check_note, raise_if_cancelled
from answer_api.synthesize import (_REFUSAL, _build_citations, _complete_or_none,
                                   _finalize_answer, _usable_content)
from graph_extract.provenance import Provenance

logger = logging.getLogger(__name__)

NONE_DOCUMENTED = ("None of the checked vendors ({vendors}) document this in the "
                   "available sources.")

_CHECK_PROMPT = (
    "Decide whether the EVIDENCE below, all from {vendor}'s documentation, shows what "
    "the QUESTION asks about for {vendor}. Use ONLY the evidence. Absence of evidence "
    "is not a finding. Respond with ONLY JSON: {{\"documented\": true or false, "
    "\"points\": [short strings stating what the evidence shows], \"fact_ids\": [ids "
    "from the evidence that support those points]}}. If documented is false, points and "
    "fact_ids must be empty. Do NOT write URLs.\n\nQUESTION: {q}\n\nEVIDENCE:\n{evidence}"
)

_ENUM_PROMPT = (
    "Answer the QUESTION by covering EVERY vendor below, in the order given, with one "
    "short paragraph each, using ONLY that vendor's numbered facts. "
    + ATTRIBUTION_RULES +
    "Cite every claim with the [N] marker of the fact it rests on, writing each marker "
    "individually. Do NOT use outside knowledge. Do NOT write any URL. Do NOT mention "
    "any vendor that is not listed below.\n\nQUESTION: {q}\n\n{blocks}\n\nAnswer:"
)


async def _gather(graphiti, driver, embedder, *, q, facet: Facet, group_id, settings):
    n = settings.enum_communities_per_vendor
    communities, facts = await asyncio.gather(
        shortlist_communities(driver, embedder, q, level=settings.global_default_level,
                              k=n, group_id=group_id, settings=settings,
                              scope=facet.scope, top_n=n),
        search_local(graphiti, driver, q=q, k=settings.enum_facts_per_vendor,
                     scope=facet.scope, group_id=group_id,
                     fetch_cap=settings.local_scoped_fetch_cap))
    return communities, facts["results"]


def _community_ids(communities) -> list[tuple[str, list[str]]]:
    """(finding, cited ids) per report finding, ids limited to the community's
    own cited facts."""
    out = []
    for c in communities:
        try:
            findings = json.loads(c.full_report or "[]")
        except ValueError:
            findings = []
        for fd in findings if isinstance(findings, list) else []:
            if not isinstance(fd, dict):
                continue
            ids = [i for i in fd.get("fact_ids") or [] if i in c.cited_fact_uuids]
            if ids:
                out.append((str(fd.get("finding", "")), ids))
    return out


def _evidence(communities, facts, admitted: set[str]) -> tuple[str, set[str]]:
    """Evidence lines and the ids a check may cite. Local facts come from a
    vendor-scoped search and are kept as is; community-cited ids are kept only
    when in `admitted` (their sources belong to this vendor), so a vendor is
    never documented on another vendor's facts."""
    lines, valid = [], set()
    for f in facts:
        lines.append(f"[{f['fact_uuid']}] {f['fact']}")
        valid.add(f["fact_uuid"])
    for finding, ids in _community_ids(communities):
        ids = [i for i in ids if i in admitted]
        if ids:
            lines.append(f"FINDING: {finding} (facts: {', '.join(ids)})")
            valid.update(ids)
    return "\n".join(lines), valid


async def _check(client, model, *, q, vendor, evidence, valid) -> list[str] | None:
    """Supporting fact ids (possibly empty: not documented), or None: unusable."""
    prompt = _CHECK_PROMPT.format(vendor=vendor, q=q, evidence=evidence)
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}])
        obj = _extract_json(_usable_content(resp) or "")
        if obj is not None and isinstance(obj.get("documented"), bool):
            if not obj["documented"]:
                return []
            ids = [i for i in obj.get("fact_ids") or [] if i in valid]
            if ids:
                return ids
            # Invented ids are an unusable reply, like unparseable JSON: retry.
            logger.warning("enumeration check for %s claimed documented with no "
                           "fact id from its evidence", vendor)
    logger.warning("enumeration check for %s returned no usable verdict", vendor)
    return None


async def enumerate_vendors(graphiti, driver, embedder, map_client, map_model,
                            synth_client, synth_model, *, q: str,
                            facets: Sequence[Facet], group_id: str, settings) -> dict:
    attempted = [f.label for f in facets]
    gathered = await asyncio.gather(*[
        _gather(graphiti, driver, embedder, q=q, facet=f, group_id=group_id,
                settings=settings) for f in facets], return_exceptions=True)
    raise_if_cancelled(gathered)   # cancellation propagates, never a degraded vendor
    degraded: set[str] = set()
    supported: dict[str, list[str]] = {}
    communities_used: list[dict] = []
    local_texts: dict[str, str] = {}

    # Community-cited ids, resolved ONCE, so each vendor admits only its own.
    community_ids = sorted({i for got in gathered if not isinstance(got, Exception)
                            for _, ids in _community_ids(got[0]) for i in ids})
    community_sources: dict[str, list] | None = {}
    if community_ids:
        try:
            resolved_c = await Provenance(driver).resolve_citations(community_ids)
            community_sources = {u: resolved_c.get(u, {}).get("sources", [])
                                 for u in community_ids}
        except Exception as exc:
            logger.warning("enumeration could not resolve community facts: %s", exc)
            community_sources = None

    async def _one(f: Facet, got):
        if isinstance(got, Exception):
            logger.warning("enumeration gather for %s failed: %s", f.label, got)
            return f, None
        communities, facts = got
        communities_used.extend({"community_id": c.community_id, "title": c.title,
                                 "relevance": c.relevance} for c in communities)
        local_texts.update({x["fact_uuid"]: x["fact"] for x in facts})
        try:
            if community_sources is None and _community_ids(communities):
                return f, None   # its community evidence could not be verified
            admitted = {u for u, src in (community_sources or {}).items()
                        if _in_scope_ci(src, f.scope)}
            evidence, valid = _evidence(communities, facts, admitted)
            if not evidence:
                return f, []
            return f, await _check(map_client, map_model, q=q, vendor=f.label,
                                   evidence=evidence, valid=valid)
        except Exception as exc:   # one vendor's failure must not sink the rest
            logger.warning("enumeration check for %s failed: %s", f.label, exc)
            return f, None

    for f, ids in await asyncio.gather(*[_one(f, g) for f, g in zip(facets, gathered)]):
        if ids is None:
            degraded.add(f.label)
        elif ids:
            supported[f.label] = ids

    texts: dict[str, str] = {}
    resolved: dict = {}
    all_ids = list(dict.fromkeys(i for ids in supported.values() for i in ids))
    if all_ids:
        texts = {**await _fact_texts(driver, group_id,
                                     [i for i in all_ids if i not in local_texts]),
                 **{i: t for i, t in local_texts.items() if i in all_ids}}
        resolved = await Provenance(driver).resolve_citations(all_ids)
    for v in list(supported):
        renderable = [i for i in supported[v] if i in texts]
        if renderable:
            supported[v] = renderable
        else:   # documented, but no fact to show: a failure, never absence
            logger.warning("enumeration: %s documented but none of its facts has text", v)
            del supported[v]
            degraded.add(v)

    degraded_list = [v for v in attempted if v in degraded]
    checked = [v for v in attempted if v not in degraded]
    documented = [v for v in checked if v in supported]
    note = could_not_check_note(degraded_list)
    # checked_vendors agrees with the closing "Checked:" line; the vendors that
    # could not be checked are in degraded_facets.
    base = {"query": q, "checked_vendors": checked, "documented_vendors": documented,
            "degraded_facets": degraded_list, "communities_used": communities_used}
    if not checked:
        return {**base, "answer": f"{_REFUSAL} {note}", "citations": [],
                "applies_to": []}
    if not documented:
        answer = NONE_DOCUMENTED.format(vendors=", ".join(checked))
        return {**base, "answer": answer + (f" {note}" if note else ""),
                "citations": [], "applies_to": []}
    marker_map: dict[int, dict] = {}
    marker_of: dict[str, int] = {}
    blocks = []
    for v in documented:
        lines = []
        for fid in supported[v]:
            if fid not in marker_of:   # a fact shown under an earlier vendor keeps its marker
                marker_of[fid] = len(marker_map) + 1
                marker_map[marker_of[fid]] = {"fact_uuid": fid}
            lines.append(fact_line(marker_of[fid], texts[fid],
                                   resolved.get(fid, {}).get("sources", [])))
        blocks.append(f"=== {v} ===\n" + "\n".join(lines))
    closing = f"Checked: {', '.join(checked)}. Documented for: {', '.join(documented)}."
    raw = await _complete_or_none(synth_client, synth_model,
                                  _ENUM_PROMPT.format(q=q, blocks="\n\n".join(blocks)),
                                  max_tokens=3000)
    tail = f"{note}\n\n{closing}" if note else closing
    if raw is None:
        return {**base, "answer": f"{_REFUSAL}\n\n{tail}", "citations": [],
                "applies_to": [], "degraded": "synthesis-failed"}
    answer, cited = _finalize_answer(raw, marker_map)
    citations = _build_citations(cited, marker_map, resolved)
    return {**base, "answer": f"{answer}\n\n{tail}", "citations": citations,
            "applies_to": applies_to(citations)}
