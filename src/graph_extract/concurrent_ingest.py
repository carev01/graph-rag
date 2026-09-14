"""Bounded concurrent fan-out for ingest.

Measured on the 2026-09-14 pilot re-ingest: the sum of LLM in-call time was
25,211 s against 24,933 s wall -- a ratio of 1.01, meaning effectively no overlap
across the whole run -- while four phases dispatched strictly one at a time
accounted for 55.9% of all LLM time. The provider does not serialise us: dedup
latency is flat across in-flight buckets (1231 / 1215 / 1287 / 1469 / 1082 ms from
1 to 20+ concurrent), so the headroom is real rather than assumed.

Used to fan out over ARTICLES only. Episodes within an article are consecutive
chunks of one document and share entities most heavily, so they stay sequential:
graphiti resolves entities by searching the graph as it currently stands, so two
in-flight episodes extracting a not-yet-present entity would each create it under
a different uuid. See the design spec for why partitioning is ruled out by design
invariant #4.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def run_concurrently(
    items: Sequence[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    limit: int,
) -> list[R | BaseException]:
    """Run `worker` over `items`, at most `limit` at a time.

    Returns one entry per item **in input order**. A worker exception is returned
    in that item's slot rather than raised, so one bad article cannot cancel its
    siblings -- the caller decides what a failure means. Nothing is swallowed: the
    exception object is handed back intact.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    semaphore = asyncio.Semaphore(limit)

    async def _one(item: T) -> R:
        async with semaphore:
            return await worker(item)

    return list(await asyncio.gather(
        *(_one(item) for item in items), return_exceptions=True))
