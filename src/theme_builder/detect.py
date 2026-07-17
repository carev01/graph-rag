"""GDS hierarchical Leiden community detection over the entity graph. The pure
helpers (`_community_id`, `_build_communities_from_rows`) turn Leiden's per-node
level labels into a hierarchy of communities with deterministic ids and parent
links; `detect_communities` is the thin GDS I/O wrapper (covered by an @live
smoke — the standard Neo4j testcontainer has no GDS plugin)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from neo4j import AsyncDriver


@dataclass
class Community:
    community_id: str
    level: int
    member_uuids: list[str]
    parent_id: str | None


def _community_id(level: int, member_uuids: list[str]) -> str:
    h = hashlib.sha1((f"{level}:" + ",".join(sorted(member_uuids))).encode())
    return h.hexdigest()[:16]


def _build_communities_from_rows(rows: list[dict], *, min_community_size: int,
                                 max_levels: int) -> list[Community]:
    """`rows`: [{"uuid": str, "levels": [finest, ..., coarsest]}]. Build one set of
    communities per level (capped at max_levels), drop those below
    min_community_size, and link each community to the level-above community that
    contains its members (nested Leiden levels)."""
    if not rows:
        return []
    n_levels = min(max(len(r["levels"]) for r in rows), max_levels)
    # level -> {leiden_label -> [uuids]}
    members: list[dict[int, list[str]]] = [{} for _ in range(n_levels)]
    for r in rows:
        for lvl in range(min(len(r["levels"]), n_levels)):
            members[lvl].setdefault(r["levels"][lvl], []).append(r["uuid"])
    # assign ids per surviving community; keep leiden_label -> community_id maps
    label_to_id: list[dict[int, str]] = [{} for _ in range(n_levels)]
    surviving: list[dict[int, list[str]]] = [{} for _ in range(n_levels)]
    for lvl in range(n_levels):
        for label, uuids in members[lvl].items():
            if len(uuids) >= min_community_size:
                cid = _community_id(lvl, uuids)
                label_to_id[lvl][label] = cid
                surviving[lvl][label] = uuids
    row_by_uuid = {r["uuid"]: r for r in rows}
    out: list[Community] = []
    for lvl in range(n_levels):
        for label, uuids in surviving[lvl].items():
            parent_id = None
            if lvl + 1 < n_levels:
                parent_label = None
                for mu in uuids:
                    levels = row_by_uuid[mu]["levels"]
                    if len(levels) > lvl + 1:
                        parent_label = levels[lvl + 1]
                        break
                if parent_label is not None:
                    parent_id = label_to_id[lvl + 1].get(parent_label)
            out.append(Community(community_id=label_to_id[lvl][label], level=lvl,
                                 member_uuids=uuids, parent_id=parent_id))
    return out


# GDS 2.13: project the entity graph with the modern gds.graph.project aggregation
# function, aggregating parallel RELATES_TO into an integer `weight` and marking
# the projection UNDIRECTED (Leiden requires undirected; the key lives on the
# PROJECTION config, not on gds.leiden). Only entities with a RELATES_TO edge are
# projected — isolated entities aren't clustered, which is fine.
_PROJECT = (
    "MATCH (a:Entity {group_id: $g})-[rel:RELATES_TO {group_id: $g}]->(b:Entity {group_id: $g}) "
    "WITH a, b, count(rel) AS w "
    "RETURN gds.graph.project($n, a, b, {relationshipProperties: {weight: w}}, "
    "{undirectedRelationshipTypes: ['*']}) AS res"
)
_DROP_IF_EXISTS = (
    "CALL gds.graph.exists($n) YIELD exists "
    "WITH exists WHERE exists CALL gds.graph.drop($n) YIELD graphName RETURN graphName"
)


async def detect_communities(driver: AsyncDriver, group_id: str, *,
                             min_community_size: int, max_levels: int) -> list[Community]:
    name = f"theme-{group_id}"
    async with driver.session() as s:
        await s.run(_DROP_IF_EXISTS, n=name)   # pre-drop a stale projection
        try:
            await s.run(_PROJECT, g=group_id, n=name)
            r = await s.run(
                "CALL gds.leiden.stream($n, {relationshipWeightProperty: 'weight', "
                "includeIntermediateCommunities: true}) "
                "YIELD nodeId, intermediateCommunityIds "
                "RETURN gds.util.asNode(nodeId).uuid AS uuid, intermediateCommunityIds AS levels",
                n=name)
            rows = [{"uuid": rec["uuid"], "levels": list(rec["levels"])} async for rec in r]
        finally:
            await s.run(_DROP_IF_EXISTS, n=name)
    return _build_communities_from_rows(rows, min_community_size=min_community_size,
                                        max_levels=max_levels)
