from __future__ import annotations

from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_NODE_DISTANCE, EDGE_HYBRID_SEARCH_RRF)

from graph_extract.provenance import Provenance


async def _vendor_episode_uuids(driver, vendor: str) -> set[str]:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (v:Vendor)-[:HAS_PRODUCT]->(:Product)-[:HAS_SOURCE]->(:Source)"
            "-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(e:Episodic) "
            "WHERE toLower(v.name)=toLower($v) RETURN collect(DISTINCT e.uuid) AS u", v=vendor)
        rec = await r.single()
        return set(rec["u"]) if rec else set()


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


async def search_local(graphiti, driver, *, q, k=10, vendor=None,
                       include_invalid=False, group_id, center_node_uuid=None):
    # Over-fetch so post-filters (validity, vendor scope) still leave ~k results.
    edges = await _retrieve_edges(graphiti, q, fetch_limit=max(k * 3, k),
                                  group_id=group_id, center_node_uuid=center_node_uuid)
    if not include_invalid:
        edges = [e for e in edges if getattr(e, "invalid_at", None) is None]
    if vendor:
        scope = await _vendor_episode_uuids(driver, vendor)
        edges = [e for e in edges if scope.intersection(e.episodes or [])]
    edges = edges[:k]
    citations = await Provenance(driver).resolve_citations([e.uuid for e in edges])
    return {
        "query": q, "count": len(edges),
        "results": [{"fact": e.fact, "fact_uuid": e.uuid,
                     "valid_at": getattr(e, "valid_at", None),
                     "sources": citations.get(e.uuid, [])} for e in edges],
    }
