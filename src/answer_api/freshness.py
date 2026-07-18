"""Freshness stamps for the /answer envelope: how current the live graph and the
thematic report layer are. Resilient — a query failure yields None, never an
error (freshness must never fail an answer)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def _max_time(driver, cypher: str, group_id: str) -> str | None:
    try:
        async with driver.session() as s:
            r = await s.run(cypher, g=group_id)
            rec = await r.single()
            return rec["c"] if rec else None
    except Exception:
        logger.warning("freshness query failed", exc_info=True)
        return None


async def freshness(driver, group_id, *, reports: bool) -> dict:
    graph_cursor_time = await _max_time(
        driver,
        "MATCH (e:Episodic {group_id:$g}) RETURN toString(max(e.created_at)) AS c",
        group_id)
    reports_as_of = None
    if reports:
        reports_as_of = await _max_time(
            driver,
            "MATCH (c:Community {group_id:$g}) RETURN toString(max(c.generated_at)) AS c",
            group_id)
    return {"graph_cursor_time": graph_cursor_time, "reports_as_of": reports_as_of}
