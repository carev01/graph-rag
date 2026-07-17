from __future__ import annotations

from answer_api.search import _retrieve_edges, _vendor_episode_uuids
from graph_extract.provenance import Provenance


def _fact_status(invalid_at, expired_by_sweep) -> str:
    if invalid_at is None:
        return "current"
    if expired_by_sweep:
        return "expired"
    return "superseded"


async def _sweep_flags(driver, uuids: list[str], group_id) -> dict[str, bool]:
    """expired_by_sweep is a custom RELATES_TO property, not an EntityEdge field
    -- read it directly for the returned facts (one query)."""
    if not uuids:
        return {}
    async with driver.session() as s:
        r = await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $uuids "
            "RETURN f.uuid AS uuid, coalesce(f.expired_by_sweep, false) AS swept",
            g=group_id, uuids=uuids)
        return {rec["uuid"]: rec["swept"] async for rec in r}


async def timeline_local(graphiti, driver, *, q, limit=30, vendor=None, group_id) -> dict:
    edges = await _retrieve_edges(graphiti, q, fetch_limit=max(limit * 3, limit),
                                  group_id=group_id)
    if vendor:
        scope = await _vendor_episode_uuids(driver, vendor)
        edges = [e for e in edges if scope.intersection(e.episodes or [])]
    # ascending by valid_at; facts without valid_at sort last
    edges.sort(key=lambda e: (getattr(e, "valid_at", None) is None,
                              str(getattr(e, "valid_at", "") or "")))
    edges = edges[:limit]
    uuids = [e.uuid for e in edges]
    swept = await _sweep_flags(driver, uuids, group_id)
    citations = await Provenance(driver).resolve_citations(uuids)
    return {
        "query": q, "count": len(edges),
        "timeline": [{"fact": e.fact, "fact_uuid": e.uuid,
                      "valid_at": getattr(e, "valid_at", None),
                      "invalid_at": getattr(e, "invalid_at", None),
                      "status": _fact_status(getattr(e, "invalid_at", None),
                                             swept.get(e.uuid, False)),
                      "sources": citations.get(e.uuid, [])} for e in edges],
    }
