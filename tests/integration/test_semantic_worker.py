import pytest

from graph_extract import usage
from graph_sync.semantic_worker import run_worker_once

pytestmark = pytest.mark.asyncio(loop_scope="module")

# Shared worker-tuning kwargs used across tests: fast retry (base=0) so a
# failed job's backoff doesn't block re-claiming within a single test, and a
# generous budget so incremental behavior isn't accidentally budget-gated.
_WK = dict(batch=10, budget=5_000_000, max_attempts=5,
           backoff_base=0.0, backoff_cap=0.0, lease=1800.0)


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
    n = await run_worker_once(state_store, stub, **_WK)
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
    n = await run_worker_once(state_store, Boom(), **_WK)
    # no exception bubbled; retry_delay is ~0 (backoff_base=0), so a single
    # failure retries: job goes back to 'pending' and is immediately
    # re-claimable rather than terminal-failing.
    assert n == 1

    # drive it through the remaining attempts until it dead-letters
    for _ in range(4):
        assert await run_worker_once(state_store, Boom(), **_WK) == 1
    assert await state_store.claim_semantic_jobs(10, True) == []  # dead, not re-claimable
    assert await state_store.dead_semantic_job_count() == 1


async def test_worker_reaps_budgets_meters_and_retries(state_store):
    class FlakyIngest:
        """Fails the first call, then succeeds and reports a fixed token cost."""

        def __init__(self):
            self.calls = 0

        async def ingest_article(self, aid):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient")
            usage.get_tally().add("llm", prompt=100, completion=0)

        async def tombstone_article_episodes(self, aid):
            return 0

    before_tokens = await state_store.today_token_total()
    stub = FlakyIngest()
    await state_store.enqueue_semantic_job("w-flaky", "upsert", "h")

    # first pass: the job fails and is scheduled to retry.
    n1 = await run_worker_once(state_store, stub, **_WK)
    assert n1 == 1
    assert stub.calls == 1

    # backoff_base=0 in _WK means next_attempt_at is ~now(), so the retry is
    # immediately reclaimable without needing to force it forward in SQL.
    n2 = await run_worker_once(state_store, stub, **_WK)
    assert n2 == 1
    assert stub.calls == 2

    assert await state_store.claim_semantic_jobs(10, True) == []  # done, not re-claimable
    after_tokens = await state_store.today_token_total()
    assert after_tokens - before_tokens == 100


async def test_worker_records_tokens_on_failure(state_store):
    class BurnThenBoom:
        """Burns real tokens (as multi-call Graphiti extraction would) then
        fails on a later step (e.g. a Neo4j write timeout)."""

        async def ingest_article(self, aid):
            usage.get_tally().add("llm", prompt=100, completion=0)
            raise RuntimeError("write timeout after extraction")

        async def tombstone_article_episodes(self, aid):
            return 0

    usage.reset_tally()
    base = await state_store.today_token_total()
    await state_store.enqueue_semantic_job("w-burn-boom", "upsert", "h")

    n = await run_worker_once(state_store, BurnThenBoom(), **_WK)
    assert n == 1

    # job failed and was scheduled to retry (not silently dropped)
    remaining = await state_store.claim_semantic_jobs(10, True)
    assert len(remaining) == 1
    assert remaining[0]["article_id"] == "w-burn-boom"

    # tokens burned before the failure must still be recorded in the ledger
    assert await state_store.today_token_total() == base + 100


async def test_worker_withholds_bootstrap_over_budget(state_store):
    class RecordingIngest:
        def __init__(self):
            self.upserted = []

        async def ingest_article(self, aid):
            self.upserted.append(aid)

        async def tombstone_article_episodes(self, aid):
            return 0

    # Push today's ledger over an artificially small budget.
    budget = 10
    current = await state_store.today_token_total()
    if current < budget:
        await state_store.record_tokens(budget - current)

    await state_store.enqueue_semantic_job(
        "w-boot", "upsert", "h", lane="bootstrap")
    await state_store.enqueue_semantic_job(
        "w-incr", "upsert", "h", lane="incremental")

    stub = RecordingIngest()
    kwargs = dict(_WK)
    kwargs["budget"] = budget
    n = await run_worker_once(state_store, stub, **kwargs)

    assert n == 1
    assert stub.upserted == ["w-incr"]  # bootstrap withheld, incremental processed

    remaining = await state_store.claim_semantic_jobs(10, True)
    assert len(remaining) == 1
    assert remaining[0]["article_id"] == "w-boot"
    assert remaining[0]["lane"] == "bootstrap"
