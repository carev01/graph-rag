"""Incremental community refresh: decide which communities actually changed since
the last theme-build (via created_at timestamps) and match fresh Leiden communities
to persisted ones (Jaccard) so their stable ids carry across refreshes and only
new/changed communities are LLM-regenerated."""
from __future__ import annotations

from dataclasses import dataclass

from neo4j import AsyncDriver

from theme_builder.detect import Community


@dataclass
class PersistedCommunity:
    community_id: str
    level: int
    members: set[str]
    title: str
    summary: str
    full_report: str
    rating: float
    rating_explanation: str
    tags: list[str]
    cited_fact_uuids: list[str]
    embedding: list[float]
    generated_at: object          # neo4j DateTime; passed back unchanged on reuse


def _jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def match_communities(fresh: list[Community], persisted: list[PersistedCommunity],
                      *, tau: float) -> dict[int, "PersistedCommunity | None"]:
    """Greedy 1:1 Jaccard match of fresh->persisted within the same level. Each
    fresh community maps to its best-overlapping persisted community (>= tau) or
    None (new); a persisted community is claimed by at most one fresh."""
    cands: list[tuple[float, int, int]] = []
    for fi, fc in enumerate(fresh):
        fmembers = set(fc.member_uuids)
        for pi, pc in enumerate(persisted):
            if pc.level != fc.level:
                continue
            j = _jaccard(fmembers, pc.members)
            if j >= tau:
                cands.append((j, fi, pi))
    cands.sort(key=lambda t: t[0], reverse=True)
    out: dict[int, "PersistedCommunity | None"] = {i: None for i in range(len(fresh))}
    used_fresh: set[int] = set()
    used_persisted: set[int] = set()
    for _j, fi, pi in cands:
        if fi in used_fresh or pi in used_persisted:
            continue
        out[fi] = persisted[pi]
        used_fresh.add(fi)
        used_persisted.add(pi)
    return out


def classify(fresh: list[Community], matches: dict[int, "PersistedCommunity | None"],
             touched: set[str] | None) -> tuple[list[int], list[int]]:
    """Split fresh community indexes into (dirty, clean). touched=None -> all dirty
    (cold start). A matched community is clean iff none of its members were touched."""
    dirty: list[int] = []
    clean: list[int] = []
    for i, fc in enumerate(fresh):
        if touched is None or matches[i] is None:
            dirty.append(i)
        elif set(fc.member_uuids) & touched:
            dirty.append(i)
        else:
            clean.append(i)
    return dirty, clean


async def touched_entities(driver: AsyncDriver, group_id: str,
                           prev_cursor: str | None) -> set[str] | None:
    """Entity uuids whose data changed since prev_cursor: entities created after it,
    plus both endpoints of RELATES_TO facts created after it. prev_cursor=None
    (no prior layer) -> None, meaning 'treat everything as new'."""
    if prev_cursor is None:
        return None
    touched: set[str] = set()
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) WHERE e.created_at > datetime($c) "
            "RETURN e.uuid AS uuid", g=group_id, c=prev_cursor)
        touched |= {rec["uuid"] async for rec in r}
        r = await s.run(
            "MATCH (a:Entity {group_id:$g})-[f:RELATES_TO {group_id:$g}]->"
            "(b:Entity {group_id:$g}) WHERE f.created_at > datetime($c) "
            "RETURN a.uuid AS a, b.uuid AS b", g=group_id, c=prev_cursor)
        async for rec in r:
            touched.add(rec["a"])
            touched.add(rec["b"])
        # Facts silently expired by the staleness sweep carry no new created_at:
        # the sweep sets invalid_at=now + expired_by_sweep=true (never deletes the
        # edge). Key the dirty signal on those markers so the community regenerates.
        r = await s.run(
            "MATCH (a:Entity {group_id:$g})-[f:RELATES_TO {group_id:$g}]->"
            "(b:Entity {group_id:$g}) "
            "WHERE f.expired_by_sweep = true AND f.invalid_at > datetime($c) "
            "RETURN a.uuid AS a, b.uuid AS b", g=group_id, c=prev_cursor)
        async for rec in r:
            touched.add(rec["a"])
            touched.add(rec["b"])
    return touched


async def load_persisted(driver: AsyncDriver, group_id: str) -> list[PersistedCommunity]:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (c:Community {group_id:$g}) "
            "OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:Entity) "
            "WITH c, collect(e.uuid) AS members "
            "RETURN c.community_id AS community_id, c.level AS level, members, "
            "c.title AS title, coalesce(c.summary,'') AS summary, "
            "coalesce(c.full_report,'[]') AS full_report, coalesce(c.rating,0.0) AS rating, "
            "coalesce(c.rating_explanation,'') AS rating_explanation, "
            "coalesce(c.tags,[]) AS tags, coalesce(c.cited_fact_uuids,[]) AS cited_fact_uuids, "
            "c.embedding AS embedding, c.generated_at AS generated_at", g=group_id)
        return [PersistedCommunity(
            community_id=x["community_id"], level=x["level"], members=set(x["members"]),
            title=x["title"], summary=x["summary"], full_report=x["full_report"],
            rating=x["rating"], rating_explanation=x["rating_explanation"], tags=x["tags"],
            cited_fact_uuids=x["cited_fact_uuids"], embedding=x["embedding"],
            generated_at=x["generated_at"]) async for x in r]


async def prev_corpus_cursor(driver: AsyncDriver, group_id: str) -> str | None:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (c:Community {group_id:$g}) RETURN toString(max(c.corpus_cursor)) AS c",
            g=group_id)
        rec = await r.single()
        return rec["c"] if rec else None
