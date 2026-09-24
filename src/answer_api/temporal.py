"""When a fact counts as current (BACKLOG 41).

A fact whose end date is stated in the text and lies in the FUTURE -- "ADE is
retired in September 2028" -- is still true today. Treating any non-null
invalid_at as "no longer valid" hid such facts from answers.
"""
from __future__ import annotations

from datetime import datetime, timezone


def is_current(invalid_at: datetime | str | None, now: datetime | None = None) -> bool:
    """True while the fact has no end date or its end date is still ahead."""
    if invalid_at is None:
        return True
    # Parse ISO format string to datetime if needed
    if isinstance(invalid_at, str):
        invalid_at = datetime.fromisoformat(invalid_at.replace('Z', '+00:00'))
    if invalid_at.tzinfo is None:
        invalid_at = invalid_at.replace(tzinfo=timezone.utc)
    return invalid_at > (now or datetime.now(timezone.utc))
