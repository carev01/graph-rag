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

The same hazard exists ACROSS articles, and this module also owns the warm-up
barrier that contains it. The A/B at concurrency 4 produced 30 duplicate
entities, every one an exact-name collision, concentrated on hub entities (mean
article span 7.6 for the duplicated names against 2.4 overall): two in-flight
articles each met an entity the graph did not yet hold and each created it. A
"cold" article -- one whose entities are not yet in the graph -- therefore runs
alone (`is_cold` below); once its entities exist, later articles resolve to
them. The barrier prevents most of those duplicates; the design spec has the
measurements.
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
    is_cold: Callable[[T], Awaitable[bool]] | None = None,
) -> list[R | BaseException]:
    """Run `worker` over `items`, at most `limit` at a time.

    Returns one entry per item **in input order**. A worker exception is returned
    in that item's slot rather than raised, so one bad article cannot cancel its
    siblings -- the caller decides what a failure means. Nothing is swallowed: the
    exception object is handed back intact.

    `is_cold` opts an item into the warm-up barrier: the item waits for every
    task launched so far to FINISH, then runs alone. That is stricter than
    "cold items are sequential among themselves", deliberately -- the hazard is
    two in-flight episodes each creating an entity neither can see yet, and a
    cold article of one source running beside a warm article of another is
    exactly the cross-source hub case invariant #4 cares about. The slot
    semantics are the same on both paths: whatever the worker raises, a
    self-raised `CancelledError` included, is returned in its slot. An
    `Exception` from `is_cold` is treated the same way -- it goes in the item's
    slot and the siblings continue. A `BaseException` escaping `is_cold` (an
    outer cancellation, `KeyboardInterrupt`) propagates, but only after every
    task launched so far has been cancelled and awaited: nothing is ever left
    running detached.

    With `is_cold=None` this is byte-for-byte the pre-warm-up helper, which is
    what makes shipping the warm-up default ON a no-op at concurrency 1.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    semaphore = asyncio.Semaphore(limit)

    async def _one(item: T) -> R:
        async with semaphore:
            return await worker(item)

    if is_cold is None:
        return list(await asyncio.gather(
            *(_one(item) for item in items), return_exceptions=True))

    results: list[R | BaseException] = [None] * len(items)  # type: ignore[list-item]
    launched: list[tuple[int, asyncio.Task[R]]] = []

    async def _drain() -> None:
        if not launched:
            return
        done = await asyncio.gather(
            *(t for _, t in launched), return_exceptions=True)
        for (idx, _), value in zip(launched, done):
            results[idx] = value
        launched.clear()

    try:
        for index, item in enumerate(items):
            try:
                cold = await is_cold(item)
            except Exception as exc:  # a predicate failure fails ITS item only
                results[index] = exc
                continue
            if cold:
                await _drain()
            # A cold item is a task too, drained at once: it still runs with
            # nothing else in flight, and its slot has exactly the warm path's
            # semantics -- gather(return_exceptions=True) hands back ANY
            # BaseException, so a worker's self-raised CancelledError lands
            # in the slot here the same as it would beside a warm sibling.
            # Awaiting the coroutine directly instead could not distinguish a
            # worker-raised CancelledError from an outer cancellation.
            launched.append((index, asyncio.ensure_future(_one(item))))
            if cold:
                await _drain()
        await _drain()
    finally:
        # Any exit -- a BaseException escaping `is_cold`, an outer cancellation
        # landing in a drain -- must leave nothing running detached: a task
        # nobody awaits keeps doing Neo4j writes and LLM calls with nobody to
        # collect its exception. A normal exit has already emptied `launched`.
        if launched:
            for _, task in launched:
                task.cancel()
            await asyncio.gather(
                *(t for _, t in launched), return_exceptions=True)
            launched.clear()
    return results
