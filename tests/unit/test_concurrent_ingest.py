"""Bounded fan-out for article ingest.

The pilot measured sum-of-LLM-in-call 25,211s against 24,933s wall -- a ratio of
1.01, so no overlap at all -- with 55.9% of LLM time in phases dispatched strictly
one at a time. The provider does not serialise us (latency flat to 20-way), so the
headroom is real. This is the primitive both ingest paths fan out with.
"""
from __future__ import annotations

import asyncio

import pytest

from graph_extract.concurrent_ingest import run_concurrently

pytestmark = pytest.mark.asyncio


async def test_results_come_back_in_input_order_not_completion_order():
    async def work(n: int) -> int:
        await asyncio.sleep(0.02 if n == 0 else 0.001)
        return n * 10

    assert await run_concurrently([0, 1, 2], work, limit=3) == [0, 10, 20]


async def test_work_actually_overlaps():
    """A slow first item must not block a later one -- the whole point."""
    started: list[int] = []

    async def work(n: int) -> int:
        started.append(n)
        await asyncio.sleep(0.05 if n == 0 else 0)
        return n

    await run_concurrently([0, 1, 2], work, limit=3)
    assert started == [0, 1, 2], "all three dispatched before the slow one finished"


async def test_limit_bounds_the_in_flight_count():
    in_flight = 0
    peak = 0

    async def work(n: int) -> int:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return n

    await run_concurrently(list(range(10)), work, limit=3)
    assert peak <= 3, f"semaphore did not bound in-flight work (peak {peak})"


async def test_limit_one_is_strictly_sequential():
    """The default. Order of execution, not just of results."""
    order: list[int] = []

    async def work(n: int) -> int:
        order.append(n)
        await asyncio.sleep(0.01 if n == 0 else 0)
        order.append(-n)
        return n

    await run_concurrently([0, 1, 2], work, limit=1)
    assert order == [0, 0, 1, -1, 2, -2]


async def test_a_failing_item_does_not_cancel_its_siblings():
    done: list[int] = []

    async def work(n: int) -> int:
        if n == 1:
            raise ValueError("boom")
        await asyncio.sleep(0.01)
        done.append(n)
        return n

    results = await run_concurrently([0, 1, 2], work, limit=3)
    assert done == [0, 2], "siblings must complete"
    assert results[0] == 0 and results[2] == 2
    assert isinstance(results[1], ValueError)


async def test_an_empty_item_list_is_fine():
    async def work(n: int) -> int:
        raise AssertionError("must not be called")

    assert await run_concurrently([], work, limit=4) == []


async def test_a_limit_below_one_is_rejected():
    async def work(n: int) -> int:
        return n

    with pytest.raises(ValueError, match="limit"):
        await run_concurrently([1], work, limit=0)


# -- Added during mutation testing (Step 5). Each pins a gap the cases above left:
#    returning sorted() survived the input-order test because [0, 10, 20] is already
#    sorted, and dropping the limit guard HUNG the limit=0 test (Semaphore(0) never
#    admits the worker) instead of failing it. And test_work_actually_overlaps passed
#    under a forced Semaphore(1): a strictly sequential run still starts items in
#    list order, so `started == [0, 1, 2]` at the end proves nothing about overlap.


async def test_later_items_start_before_the_slow_first_one_finishes():
    started: list[int] = []
    seen_when_slow_finished: list[int] | None = None

    async def work(n: int) -> int:
        nonlocal seen_when_slow_finished
        started.append(n)
        if n == 0:
            await asyncio.sleep(0.05)
            seen_when_slow_finished = list(started)
        return n

    await run_concurrently([0, 1, 2], work, limit=3)
    assert seen_when_slow_finished == [0, 1, 2], "siblings must be in flight during the slow item"


async def test_input_order_is_neither_sorted_nor_completion_order():
    async def work(n: int) -> int:
        await asyncio.sleep(0.02 if n == 0 else 0.001)
        return n * 10

    # Sorted would give [0, 10, 20]; completion order would give [20, 10, 0].
    assert await run_concurrently([2, 0, 1], work, limit=3) == [20, 0, 10]


async def test_a_limit_below_one_fails_fast_without_dispatching():
    calls = 0

    async def work(n: int) -> int:
        nonlocal calls
        calls += 1
        return n

    with pytest.raises(ValueError, match="limit"):
        await asyncio.wait_for(run_concurrently([1], work, limit=0), timeout=1)
    assert calls == 0, "the guard must fire before any worker is scheduled"
