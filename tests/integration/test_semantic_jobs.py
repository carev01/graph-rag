import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_enqueue_idempotent_pending(state_store):
    await state_store.enqueue_semantic_job("a1", "upsert", "h1")
    await state_store.enqueue_semantic_job("a1", "upsert", "h2")  # newer change, still pending
    jobs = await state_store.claim_semantic_jobs(10, True)
    assert len(jobs) == 1 and jobs[0]["content_hash"] == "h2"  # collapsed, latest wins


async def test_claim_skiplocked_disjoint(state_store):
    for i in range(4):
        await state_store.enqueue_semantic_job(f"a{i}", "upsert", "h")
    a = await state_store.claim_semantic_jobs(2, True)
    b = await state_store.claim_semantic_jobs(2, True)
    assert {j["id"] for j in a}.isdisjoint({j["id"] for j in b}) and len(a) == 2 and len(b) == 2


async def test_complete_and_fail(state_store):
    await state_store.enqueue_semantic_job("a1", "remove", None)
    [j] = await state_store.claim_semantic_jobs(10, True)
    await state_store.complete_semantic_job(j["id"])
    assert await state_store.claim_semantic_jobs(10, True) == []          # done, not re-claimable
    await state_store.enqueue_semantic_job("a2", "upsert", "h")
    [k] = await state_store.claim_semantic_jobs(10, True)
    await state_store.fail_semantic_job(k["id"], "boom")
    assert await state_store.claim_semantic_jobs(10, True) == []          # failed, not re-claimable


async def test_lane_upgrade_not_downgrade(state_store):
    await state_store.enqueue_semantic_job("a1", "upsert", "h", "bootstrap")
    await state_store.enqueue_semantic_job("a1", "upsert", "h", "incremental")  # upgrades
    [j] = await state_store.claim_semantic_jobs(10, True)
    assert j["lane"] == "incremental"


async def test_lane_incremental_not_downgraded(state_store):
    await state_store.enqueue_semantic_job("a2", "upsert", "h", "incremental")
    await state_store.enqueue_semantic_job("a2", "upsert", "h", "bootstrap")  # must NOT downgrade
    [j] = await state_store.claim_semantic_jobs(10, True)
    assert j["lane"] == "incremental"


async def test_incremental_claimed_before_bootstrap(state_store):
    await state_store.enqueue_semantic_job("b1", "upsert", "h", "bootstrap")
    await state_store.enqueue_semantic_job("i1", "upsert", "h", "incremental")
    [first] = await state_store.claim_semantic_jobs(1, True)
    assert first["article_id"] == "i1"  # incremental preempts bootstrap
    await state_store.claim_semantic_jobs(10, True)  # drain remaining (b1) so it doesn't leak


async def test_bootstrap_withheld_when_excluded(state_store):
    await state_store.enqueue_semantic_job("b1", "upsert", "h", "bootstrap")
    assert await state_store.claim_semantic_jobs(10, False) == []   # bootstrap excluded
    assert len(await state_store.claim_semantic_jobs(10, True)) == 1  # drains it


async def test_token_ledger_accumulates(state_store):
    base = await state_store.today_token_total()
    await state_store.record_tokens(100)
    await state_store.record_tokens(50)
    assert await state_store.today_token_total() == base + 150


async def test_today_token_total_zero_when_empty(state_store):
    pool = await state_store._get_pool()
    await pool.execute("DELETE FROM token_ledger WHERE day=current_date")
    assert await state_store.today_token_total() == 0
