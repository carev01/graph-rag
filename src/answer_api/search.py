from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_NODE_DISTANCE, EDGE_HYBRID_SEARCH_RRF)

from answer_api.scope import Scope, scope_episode_uuids
from answer_api.temporal import is_current
from graph_extract.provenance import Provenance

logger = logging.getLogger(__name__)

# The scope's own facts, ranked by similarity to the query (the fallback when the
# corpus-wide top-N holds too few of them, see search_local).
#
# Never scans every RELATES_TO edge: from the scope's episodes (the structural chain,
# as scope._EPISODES) to the DISTINCT entities they mention, then those entities'
# edges, keeping an edge only if its `episodes` list names a scope episode. Any such
# edge is reached: the episode that produced it mentions its endpoints. Entities are
# de-duplicated BEFORE expanding -- expanding once per mention multiplies every hub's
# degree by its mention count (the six Veeam Agent guides timed out at 30 s that way;
# 1.4 s de-duplicated, 2026-10-13). `x IN uu` is not a linear scan per row in
# practice (Neo4j caches the list: the same check over all 240k edges takes ~1 s).
#
# A scope above $max_episodes is not scanned: `LIMIT $episode_cap` (max + 1) bounds
# the chain walk, and n = episode_cap tells the caller it was skipped. Neo4j cosine
# is (1+cos)/2; ranking is unaffected.
_SCAN = (
    "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(s:Source)"
    "-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(e:Episodic) "
    "WHERE v.id IS NOT NULL AND p.id IS NOT NULL "
    "AND (toLower(v.name) IN $vendors OR toLower(p.name) IN $products "
    "OR toLower(s.name) IN $sources) "
    "WITH DISTINCT e LIMIT $episode_cap "
    "WITH collect(e) AS eps "
    "WITH eps, size(eps) AS n "
    "OPTIONAL CALL (eps, n) { "
    "  WITH eps WHERE 0 < n <= $max_episodes "
    "  CALL (eps) { "
    "    UNWIND eps AS e "
    "    MATCH (e)-[:MENTIONS]->(m:Entity) "
    "    RETURN collect(DISTINCT m) AS ms "
    "  } "
    "  WITH [x IN eps | x.uuid] AS uu, ms "
    "  UNWIND ms AS m "
    "  MATCH (m)-[f:RELATES_TO]-(:Entity) "
    "  WHERE f.group_id = $group_id AND NOT f.uuid IN $exclude "
    "  AND ($include_invalid OR f.invalid_at IS NULL OR f.invalid_at > datetime()) "
    "  AND any(x IN f.episodes WHERE x IN uu) "
    "  WITH DISTINCT f "
    "  ORDER BY vector.similarity.cosine(f.fact_embedding, $vec) DESC LIMIT $k "
    "  RETURN f.uuid AS uuid, f.fact AS fact, f.episodes AS episodes, "
    "         f.valid_at AS valid_at, f.invalid_at AS invalid_at "
    "} "
    "RETURN n, uuid, fact, episodes, valid_at, invalid_at")


@dataclass(frozen=True)
class ScannedEdge:
    """The fields search_local reads from a graphiti EntityEdge."""
    uuid: str
    fact: str
    episodes: list[str]
    valid_at: datetime | None = None
    invalid_at: datetime | None = None


def _native(value: Any) -> Any:
    """neo4j.time.DateTime -> datetime, as graphiti's own edges carry."""
    return value.to_native() if hasattr(value, "to_native") else value


async def _retrieve_edges(graphiti, q, *, fetch_limit, group_id,
                          center_node_uuid=None) -> list:
    # SearchConfig.limit is the top-level fetch cap consulted by Graphiti._search
    # (graphiti-core==0.29.2). A center node switches edge reranking from RRF to
    # graph-distance (node_distance) so retrieval is biased toward that node.
    recipe = EDGE_HYBRID_SEARCH_NODE_DISTANCE if center_node_uuid else EDGE_HYBRID_SEARCH_RRF
    config = recipe.model_copy(deep=True)
    config.limit = fetch_limit
    results = await graphiti._search(q, config, group_ids=[group_id],
                                     center_node_uuid=center_node_uuid)
    return list(results.edges)


async def _scoped_scan(graphiti, driver, *, q, scope: Scope, k: int, group_id: str,
                       exclude: set[str], include_invalid: bool = False,
                       max_episodes: int = 20000) -> list[ScannedEdge]:
    """The top `k` RELATES_TO edges of `scope` (not in `exclude`) by cosine to `q`.

    `q` is embedded exactly as graphiti's search embeds it (graphiti-core 0.30.1,
    search/search.py). A scope with more than `max_episodes` episodes returns []
    -- logged, not scanned: large scopes are served by the post-filter."""
    vec = await graphiti.embedder.create(input_data=[q.replace("\n", " ")])
    records, _, _ = await driver.execute_query(
        _SCAN, vec=vec, k=k, exclude=sorted(exclude), group_id=group_id,
        include_invalid=include_invalid, max_episodes=max_episodes,
        episode_cap=max_episodes + 1,
        vendors=[v.lower() for v in scope.vendors],
        products=[p.lower() for p in scope.products],
        sources=[x.lower() for x in scope.sources])
    if records and records[0]["n"] > max_episodes:
        logger.info("scoped scan skipped: scope %s has more than %d episodes",
                    scope.as_dict(), max_episodes)
        return []
    return [ScannedEdge(uuid=r["uuid"], fact=r["fact"], episodes=list(r["episodes"] or []),
                        valid_at=_native(r["valid_at"]), invalid_at=_native(r["invalid_at"]))
            for r in records if r["uuid"] is not None]


async def search_local(graphiti, driver, *, q, k=10, scope: Scope | None = None,
                       include_invalid=False, group_id, center_node_uuid=None,
                       fetch_cap: int = 450, scan_max_episodes: int = 20000):
    scoped = scope is not None and not scope.is_empty()
    # Over-fetch so post-filters (validity, scope) still leave ~k results. A narrow
    # scope can filter out most of one fetch (the six Veeam Agent guides kept 7 of
    # 45, 2026-10-12), so a scoped search re-fetches wider until k survive.
    first = max(k * 3, k)
    limits = [first] if not scoped else list(dict.fromkeys(
        min(n, fetch_cap) for n in (first, k * 10, k * 30)))
    edges: list = []
    for limit in limits:
        edges = await _retrieve_edges(graphiti, q, fetch_limit=limit,
                                      group_id=group_id,
                                      center_node_uuid=center_node_uuid)
        if not include_invalid:
            edges = [e for e in edges if is_current(getattr(e, "invalid_at", None))]
        if scoped:
            assert scope is not None
            allowed = await scope_episode_uuids(
                driver, scope, candidates=[u for e in edges for u in (e.episodes or [])])
            edges = [e for e in edges if allowed.intersection(e.episodes or [])]
        if len(edges) >= k:
            break
    edges = edges[:k]
    if scoped and len(edges) < k:
        # The corpus-wide top-N can hold none of a small scope's facts next to a
        # huge one (Cohesity's 29 DataLock/FortKnox facts vs Veeam's 20k articles,
        # 2026-10-12): rank the scope's own facts and append them after the
        # post-filter's. A failure keeps what the post-filter found -- never raised,
        # which would empty the facet.
        assert scope is not None
        seen = {e.uuid for e in edges}
        try:
            extra = await _scoped_scan(
                graphiti, driver, q=q, scope=scope, k=k - len(edges), group_id=group_id,
                exclude=seen, include_invalid=include_invalid,
                max_episodes=scan_max_episodes)
        except Exception as exc:
            logger.warning("scoped scan failed for scope %s: %s; keeping %d "
                           "post-filter results", scope.as_dict(), exc, len(edges))
            extra = []
        for e in extra:
            if len(edges) >= k:
                break
            if e.uuid not in seen:
                seen.add(e.uuid)
                edges.append(e)
    citations = await Provenance(driver).resolve_citations([e.uuid for e in edges])
    return {
        "query": q, "count": len(edges),
        "results": [{"fact": e.fact, "fact_uuid": e.uuid,
                     "valid_at": getattr(e, "valid_at", None),
                     "invalid_at": getattr(e, "invalid_at", None),
                     "sources": citations.get(e.uuid, {}).get("sources", [])} for e in edges],
    }
