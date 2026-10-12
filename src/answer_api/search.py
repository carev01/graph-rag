from __future__ import annotations

from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_NODE_DISTANCE, EDGE_HYBRID_SEARCH_RRF)

from answer_api.scope import Scope, scope_episode_uuids
from answer_api.temporal import is_current
from graph_extract.provenance import Provenance


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


async def search_local(graphiti, driver, *, q, k=10, scope: Scope | None = None,
                       include_invalid=False, group_id, center_node_uuid=None,
                       fetch_cap: int = 450):
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
    citations = await Provenance(driver).resolve_citations([e.uuid for e in edges])
    return {
        "query": q, "count": len(edges),
        "results": [{"fact": e.fact, "fact_uuid": e.uuid,
                     "valid_at": getattr(e, "valid_at", None),
                     "invalid_at": getattr(e, "invalid_at", None),
                     "sources": citations.get(e.uuid, {}).get("sources", [])} for e in edges],
    }
