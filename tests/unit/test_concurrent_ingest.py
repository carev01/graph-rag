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


# -- The cold-item barrier (concurrency duplicate mitigation, task 1). The A/B at
#    concurrency 4 produced 30 duplicate entities, all exact-name collisions on hub
#    entities created by two in-flight episodes that could not see each other. An
#    item `is_cold` opts into running ALONE: everything launched before it must
#    FINISH first, and nothing launches until it is done.


async def test_is_cold_none_is_byte_for_byte_the_old_helper():
    """The default path must not change. This is what lets the warm-up knob
    ship defaulted ON: at concurrency 1 it provably changes nothing."""
    log: list[str] = []

    async def worker(i: int) -> int:
        log.append(f"start-{i}")
        await asyncio.sleep(0.03 if i == 0 else 0)
        log.append(f"end-{i}")
        return i * 10

    res = await run_concurrently([0, 1, 2], worker, limit=3)
    assert res == [0, 10, 20]
    # the slow first item must still be overlapped by the others
    assert log.index("start-1") < log.index("end-0")


async def test_a_cold_item_runs_with_nothing_else_in_flight():
    """The strict barrier. A slow WARM item launched before the cold one must
    have FINISHED before the cold item starts -- start order alone is satisfied
    by a sequential implementation and proves nothing."""
    log: list[str] = []

    async def worker(name: str) -> str:
        log.append(f"start-{name}")
        await asyncio.sleep(0.05 if name == "slow-warm" else 0.01)
        log.append(f"end-{name}")
        return name

    async def is_cold(name: str) -> bool:
        return name == "cold"

    res = await run_concurrently(
        ["slow-warm", "cold", "after"], worker, limit=4, is_cold=is_cold)
    assert res == ["slow-warm", "cold", "after"]
    assert log.index("end-slow-warm") < log.index("start-cold"), log
    assert log.index("end-cold") < log.index("start-after"), log


async def test_is_cold_raising_lands_in_that_items_slot():
    async def worker(i: int) -> int:
        return i

    async def is_cold(i: int) -> bool:
        if i == 1:
            raise RuntimeError("neo4j down")
        return False

    res = await run_concurrently([0, 1, 2], worker, limit=3, is_cold=is_cold)
    assert res[0] == 0
    assert isinstance(res[1], RuntimeError)
    assert res[2] == 2


async def test_results_stay_in_input_order_with_a_cold_item_in_the_middle():
    async def worker(i: int) -> int:
        await asyncio.sleep(0.02 if i == 2 else 0)
        return i * 10

    async def is_cold(i: int) -> bool:
        return i == 1

    assert await run_concurrently(
        [0, 1, 2, 3], worker, limit=4, is_cold=is_cold) == [0, 10, 20, 30]


# -- Task 1 review fixes. The four tests above are all satisfied by a FULLY
#    SEQUENTIAL is_cold body (a barrier test is satisfied by "everything is a
#    barrier"), so nothing pinned that warm items still overlap on the path both
#    call sites use. And the cold item used to be awaited directly under
#    `except Exception`, so its slot semantics differed from a warm sibling's and
#    a BaseException out of `is_cold` walked out with launched tasks orphaned.


async def test_warm_items_still_overlap_on_either_side_of_a_cold_one():
    """Concurrency is the point of the whole branch. Snapshot at the moment the
    slow warm item FINISHES, on both sides of the barrier: a later warm item
    must already have started. A sequential loop passes every other cold-path
    test here and must fail this one."""
    started: list[str] = []
    seen_when_slow_finished: dict[str, list[str]] = {}

    async def worker(name: str) -> str:
        started.append(name)
        if name.startswith("slow"):
            await asyncio.sleep(0.05)
            seen_when_slow_finished[name] = list(started)
        return name

    async def is_cold(name: str) -> bool:
        return name == "cold"

    res = await run_concurrently(
        ["slow-a", "fast-a", "cold", "slow-b", "fast-b"], worker, limit=4, is_cold=is_cold)
    assert res == ["slow-a", "fast-a", "cold", "slow-b", "fast-b"]
    assert "fast-a" in seen_when_slow_finished["slow-a"], (
        f"warm items before the cold one must overlap: {seen_when_slow_finished}")
    assert "fast-b" in seen_when_slow_finished["slow-b"], (
        f"warm items after the cold one must overlap: {seen_when_slow_finished}")
    # and the barrier still held while they overlapped
    assert "cold" not in seen_when_slow_finished["slow-a"], seen_when_slow_finished
    assert "slow-b" not in seen_when_slow_finished["slow-a"], seen_when_slow_finished


async def test_a_cold_workers_exception_lands_in_its_slot_and_siblings_complete():
    """One bad opening article must not abort the whole source. Before the fix
    this was an untested handler: deleting it failed nothing."""
    finished: list[str] = []

    async def worker(name: str) -> str:
        if name == "cold":
            raise ValueError("boom")
        await asyncio.sleep(0.01)
        finished.append(name)
        return name

    async def is_cold(name: str) -> bool:
        return name == "cold"

    res = await run_concurrently(
        ["before", "cold", "after"], worker, limit=4, is_cold=is_cold)
    assert finished == ["before", "after"], "siblings on both sides must complete"
    assert res[0] == "before" and res[2] == "after"
    assert isinstance(res[1], ValueError)


async def test_a_cold_workers_self_raised_cancellation_lands_in_its_slot_like_a_warm_one():
    """The slot contract is identical on both paths, BaseException included:
    semantic_worker's "were cancelled" branch relies on a worker-raised
    CancelledError arriving in the slot, and the cold path used to let it
    propagate instead -- for exactly the articles the barrier singles out."""
    async def worker(name: str) -> str:
        if name == "cold":
            raise asyncio.CancelledError()
        return name

    async def is_cold(name: str) -> bool:
        return name == "cold"

    res = await run_concurrently(
        ["before", "cold", "after"], worker, limit=4, is_cold=is_cold)
    assert res[0] == "before" and res[2] == "after"
    assert isinstance(res[1], asyncio.CancelledError)


class _Interrupt(BaseException):
    """Stands in for an outer cancellation / KeyboardInterrupt escaping
    `is_cold`. A real KeyboardInterrupt would take the event loop down with
    it; this has the same `except Exception` blindness without the blast."""


async def test_a_base_exception_escaping_is_cold_leaves_no_task_running():
    """The orphan hole: the predicate blows up while a warm task is in flight.
    The exception must still propagate -- but not before that task has been
    cancelled and awaited. Orphaned, it would keep writing to Neo4j with nobody
    to collect its result; here it would simply still be asleep."""
    started: list[int] = []
    finished: list[int] = []
    cancelled: list[int] = []

    async def worker(n: int) -> int:
        started.append(n)
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            cancelled.append(n)
            raise
        finished.append(n)
        return n

    async def is_cold(n: int) -> bool:
        if n == 1:
            await asyncio.sleep(0.01)  # let item 0's task actually start
            raise _Interrupt()
        return False

    with pytest.raises(_Interrupt):
        await run_concurrently([0, 1, 2], worker, limit=4, is_cold=is_cold)
    assert started == [0], "item 0 must have been in flight when the predicate blew up"
    assert finished + cancelled == [0], (
        f"item 0 must be finished or cancelled, not left running: "
        f"finished={finished} cancelled={cancelled}")
