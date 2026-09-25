from __future__ import annotations

from datetime import datetime, timezone

from answer_api.scope import Scope, scope_episode_uuids
from answer_api.search import _retrieve_edges
from answer_api.temporal import is_current
from graph_extract.provenance import Provenance

# Sorts after any real timestamp, so a fact with no valid_at lands last.
_VALID_AT_MAX = datetime.max.replace(tzinfo=timezone.utc)


def _valid_at_sort_key(edge) -> tuple[int, datetime]:
    """Chronological sort key for an edge's valid_at.

    Order by the true UTC instant, not the stringified local wall-clock: a
    fact at 08:00+05:00 (03:00 UTC) precedes one at 05:00+00:00 (05:00 UTC),
    which a string compare of the ISO forms would get backwards. A naive
    (tz-unaware) datetime is treated as UTC so it never raises when compared
    with aware ones. A None valid_at sorts last (leading 1 vs 0).
    """
    va = getattr(edge, "valid_at", None)
    if not isinstance(va, datetime):   # None, or an unexpected non-datetime
        return (1, _VALID_AT_MAX)
    if va.tzinfo is None:
        va = va.replace(tzinfo=timezone.utc)
    return (0, va.astimezone(timezone.utc))


def _fact_status(invalid_at, expired_by_sweep) -> str:
    if is_current(invalid_at):
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


async def timeline_local(graphiti, driver, *, q, limit=30, scope: Scope | None = None,
                         group_id) -> dict:
    edges = await _retrieve_edges(graphiti, q, fetch_limit=max(limit * 3, limit),
                                  group_id=group_id)
    if scope is not None and not scope.is_empty():
        allowed = await scope_episode_uuids(driver, scope)
        edges = [e for e in edges if allowed.intersection(e.episodes or [])]
    # ascending by valid_at (true UTC instant); facts without valid_at sort last
    edges.sort(key=_valid_at_sort_key)
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
                      "sources": citations.get(e.uuid, {}).get("sources", [])} for e in edges],
    }
