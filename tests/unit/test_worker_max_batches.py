"""`run_worker(max_batches=N)` must return after N batches without a signal.

Why this exists: the worker previously had no stop condition other than SIGINT/
SIGTERM, and `timeout ... uv run ...` does NOT forward SIGTERM to the wrapped
interpreter. A run that was supposed to be time-boxed kept going, looped again,
and claimed a live bootstrap-lane job nobody meant to process. A bounded mode
gives operators and tests a stop condition that does not depend on OS signal
plumbing.
"""
from __future__ import annotations

import asyncio

import pytest

from graph_sync.semantic_worker import run_worker


class _CountingStore:
    """Always reports work available, so the loop never sleeps on an empty queue."""

    def __init__(self) -> None:
        self.claims = 0

    async def reap_stale_jobs(self, lease, max_attempts):
        return None

    async def today_token_total(self):
        return 0

    async def claim_semantic_jobs(self, batch, include_bootstrap, source_ids=None):
        self.claims += 1
        return []          # no jobs -> run_worker_once returns 0, loop would poll


async def _run(max_batches, poll_seconds=0.01):
    store = _CountingStore()
    await run_worker(
        store, ingest=None, batch=1, poll_seconds=poll_seconds,
        stop_event=asyncio.Event(), budget=1, max_attempts=1,
        backoff_base=1.0, backoff_cap=1.0, lease=1.0, max_batches=max_batches)
    return store


@pytest.mark.asyncio
async def test_returns_after_exactly_n_batches():
    store = await _run(3)
    assert store.claims == 3


@pytest.mark.asyncio
async def test_one_batch_is_a_true_single_shot():
    store = await _run(1)
    assert store.claims == 1


@pytest.mark.asyncio
async def test_bounded_run_terminates_without_any_signal():
    """The whole point: no SIGINT/SIGTERM is ever delivered here. Without
    max_batches this call would never return."""
    await asyncio.wait_for(_run(2), timeout=5.0)


@pytest.mark.asyncio
async def test_none_means_unbounded_and_still_honours_the_stop_event():
    """max_batches=None must preserve the original behaviour."""
    store = _CountingStore()
    stop = asyncio.Event()

    async def _stop_soon():
        await asyncio.sleep(0.05)
        stop.set()

    asyncio.create_task(_stop_soon())
    await asyncio.wait_for(run_worker(
        store, ingest=None, batch=1, poll_seconds=0.01, stop_event=stop,
        budget=1, max_attempts=1, backoff_base=1.0, backoff_cap=1.0,
        lease=1.0, max_batches=None), timeout=5.0)
    assert store.claims >= 1
