from __future__ import annotations

from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF

from graph_extract.provenance import Provenance


async def _vendor_episode_uuids(driver, vendor: str) -> set[str]:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (v:Vendor)-[:HAS_PRODUCT]->(:Product)-[:HAS_SOURCE]->(:Source)"
            "-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(e:Episodic) "
            "WHERE toLower(v.name)=toLower($v) RETURN collect(DISTINCT e.uuid) AS u", v=vendor)
        rec = await r.single()
        return set(rec["u"]) if rec else set()


async def search_local(graphiti, driver, *, q, k=10, vendor=None,
                       include_invalid=False, group_id):
    # Over-fetch so post-filters (validity, vendor scope) still leave ~k
    # results. `SearchConfig.limit` (graphiti_core.search.search_config) is
    # the top-level fetch cap consulted by Graphiti._search; confirmed
    # against the installed graphiti-core==0.29.2 SearchConfig model.
    config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    config.limit = max(k * 3, k)
    results = await graphiti._search(q, config, group_ids=[group_id])
    edges = list(results.edges)
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
