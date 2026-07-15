import pytest

from graph_sync.semantic_worker import run_worker_once

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_worker_processes_upsert_and_remove(state_store):
    class StubIngest:
        def __init__(self):
            self.calls = []

        async def ingest_article(self, aid):
            self.calls.append(("upsert", aid))

        async def tombstone_article_episodes(self, aid):
            self.calls.append(("remove", aid))
            return 0

    await state_store.enqueue_semantic_job("w-a1", "upsert", "h")
    await state_store.enqueue_semantic_job("w-a2", "remove", None)
    stub = StubIngest()
    n = await run_worker_once(state_store, stub, batch=10)
    assert n == 2
    assert ("upsert", "w-a1") in stub.calls
    assert ("remove", "w-a2") in stub.calls
    assert await state_store.claim_semantic_jobs(10, True) == []  # both completed


async def test_worker_marks_failed_and_continues(state_store):
    class Boom:
        async def ingest_article(self, aid):
            raise RuntimeError("x")

        async def tombstone_article_episodes(self, aid):
            return 0

    await state_store.enqueue_semantic_job("w-a3", "upsert", "h")
    n = await run_worker_once(state_store, Boom(), batch=10)
    # no exception bubbled; interim worker literal is max_attempts=5, retry_delay=0,
    # so a single failure retries: job goes back to 'pending' and is immediately
    # re-claimable rather than terminal-failing.
    assert n == 1

    # drive it through the remaining attempts until it dead-letters
    for _ in range(4):
        assert await run_worker_once(state_store, Boom(), batch=10) == 1
    assert await state_store.claim_semantic_jobs(10, True) == []  # dead, not re-claimable
    assert await state_store.dead_semantic_job_count() == 1
