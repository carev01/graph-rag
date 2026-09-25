"""`SEMANTIC_CLAIM_SOURCE_IDS` (the k3s bootstrap-first rehearsal, docs/deploy/k3s.md)
scopes a worker to a subset of sources by reaching `StateStore.claim_semantic_jobs`'s
`source_ids` filter. This pins that the knob actually reaches the claim call --
`run_worker_once` and `run_worker` must both forward it, not merely accept it -- the
same "configured, constructed and never passed" defect class the warm-up lock tests
guard against (test_cli_worker_wiring.py).
"""
from __future__ import annotations

import asyncio

import pytest

from graph_sync.semantic_worker import run_worker, run_worker_once

pytestmark = pytest.mark.asyncio


class _Store:
    def __init__(self) -> None:
        self.claim_calls: list[tuple[int, bool, list[str] | None]] = []

    async def reap_stale_jobs(self, *a, **k):
        return None

    async def today_token_total(self):
        return 0

    async def claim_semantic_jobs(self, batch, include_bootstrap, source_ids=None):
        self.claim_calls.append((batch, include_bootstrap, source_ids))
        return []


async def test_run_worker_once_forwards_source_ids_to_the_claim():
    store = _Store()
    await run_worker_once(
        store, ingest=None, batch=5, budget=1, max_attempts=1,
        backoff_base=1.0, backoff_cap=1.0, lease=1.0,
        source_ids=["s1", "s2"])
    assert store.claim_calls == [(5, True, ["s1", "s2"])]


async def test_run_worker_once_defaults_to_unscoped():
    """No `source_ids` given -> `None` reaches the claim, i.e. unscoped -- the
    same behaviour as before this knob existed."""
    store = _Store()
    await run_worker_once(
        store, ingest=None, batch=5, budget=1, max_attempts=1,
        backoff_base=1.0, backoff_cap=1.0, lease=1.0)
    assert store.claim_calls == [(5, True, None)]


async def test_run_worker_forwards_source_ids_through_to_run_worker_once():
    store = _Store()
    await run_worker(
        store, ingest=None, batch=5, poll_seconds=0.01,
        stop_event=asyncio.Event(), budget=1, max_attempts=1,
        backoff_base=1.0, backoff_cap=1.0, lease=1.0, max_batches=1,
        source_ids=["s1"])
    assert store.claim_calls == [(5, True, ["s1"])]
