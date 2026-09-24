"""When a fact counts as current (BACKLOG 41).

A fact whose end date is stated in the text and lies in the FUTURE -- "ADE is
retired in September 2028" -- is still true today. Treating any non-null
invalid_at as "no longer valid" hid such facts from answers.

Filter on `invalid_at` only, never on `expired_at`: graphiti sets
`expired_at = now` on every edge whose extracted `invalid_at` is non-null,
future dates included (graphiti_core/utils/maintenance/edge_operations.py:822-823,
0.30.1), so an `expired_at` filter would hide future-dated current facts again.
"""
from __future__ import annotations

from datetime import datetime, timezone


def is_current(invalid_at: datetime | None, now: datetime | None = None) -> bool:
    """True while the fact has no end date or its end date is still ahead.

    `invalid_at` is graphiti's `EntityEdge.invalid_at` (datetime | None). There is
    deliberately no str branch: Neo4j's toString(datetime) form
    ("2026-06-26T00:28:43.493367Z[UTC]") is not ISO 8601, so parsing strings here
    would raise on real values -- convert at the query instead."""
    if invalid_at is None:
        return True
    if invalid_at.tzinfo is None:
        invalid_at = invalid_at.replace(tzinfo=timezone.utc)
    return invalid_at > (now or datetime.now(timezone.utc))
