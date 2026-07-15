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
    await state_store.complete_semantic_job(j["id"], j["claimed_at"])
    assert await state_store.claim_semantic_jobs(10, True) == []          # done, not re-claimable
    await state_store.enqueue_semantic_job("a2", "upsert", "h")
    [k] = await state_store.claim_semantic_jobs(10, True)
    await state_store.fail_semantic_job(
        k["id"], "boom", max_attempts=1, retry_delay_seconds=0, claimed_at=k["claimed_at"])
    assert await state_store.claim_semantic_jobs(10, True) == []          # dead, not re-claimable


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


async def test_fail_retries_then_dead(state_store):
    base = await state_store.dead_semantic_job_count()
    await state_store.enqueue_semantic_job("retry-a1", "upsert", "h", "incremental")
    [j] = await state_store.claim_semantic_jobs(10, True)
    await state_store.fail_semantic_job(
        j["id"], "boom", max_attempts=2, retry_delay_seconds=0, claimed_at=j["claimed_at"])
    # attempts now 1 (<2): back to pending, re-claimable (delay 0)
    [j2] = await state_store.claim_semantic_jobs(10, True)
    await state_store.fail_semantic_job(
        j2["id"], "boom2", max_attempts=2, retry_delay_seconds=0, claimed_at=j2["claimed_at"])
    # attempts now 2 (>=2): dead, not re-claimable
    assert await state_store.claim_semantic_jobs(10, True) == []
    assert await state_store.dead_semantic_job_count() == base + 1


async def test_backoff_hides_job(state_store):
    await state_store.enqueue_semantic_job("retry-a2", "upsert", "h", "incremental")
    [j] = await state_store.claim_semantic_jobs(10, True)
    await state_store.fail_semantic_job(
        j["id"], "x", max_attempts=5, retry_delay_seconds=3600, claimed_at=j["claimed_at"])
    assert await state_store.claim_semantic_jobs(10, True) == []   # next_attempt_at in the future


async def test_reaper_reclaims_stale_inprogress(state_store):
    await state_store.enqueue_semantic_job("retry-a3", "upsert", "h", "incremental")
    [j] = await state_store.claim_semantic_jobs(10, True)          # now in_progress, fresh claimed_at
    assert await state_store.reap_stale_jobs(3600, max_attempts=5) == 0  # fresh, not reaped
    # force claimed_at into the past, then reap
    async with (await state_store._get_pool()).acquire() as c:
        await c.execute(
            "UPDATE semantic_jobs SET claimed_at = now() - interval '2 hours' WHERE id=$1",
            j["id"])
    assert await state_store.reap_stale_jobs(3600, max_attempts=5) == 1
    assert len(await state_store.claim_semantic_jobs(10, True)) == 1  # re-claimable


async def test_reaper_dead_letters_at_max_attempts(state_store):
    base = await state_store.dead_semantic_job_count()
    await state_store.enqueue_semantic_job("retry-a4", "upsert", "h", "incremental")
    [j] = await state_store.claim_semantic_jobs(10, True)          # now in_progress
    # force attempts to max_attempts - 1 and claimed_at into the past, simulating a
    # process-crash-looped job that's about to exhaust its retries
    async with (await state_store._get_pool()).acquire() as c:
        await c.execute(
            "UPDATE semantic_jobs SET attempts=$2, "
            "claimed_at = now() - interval '2 hours' WHERE id=$1",
            j["id"], 2)
    assert await state_store.reap_stale_jobs(3600, max_attempts=3) == 1
    assert await state_store.dead_semantic_job_count() == base + 1
    assert await state_store.claim_semantic_jobs(10, True) == []  # dead, not re-claimable


async def test_job_status_counts(state_store):
    base = await state_store.job_status_counts()
    await state_store.enqueue_semantic_job("qs1", "upsert", "h", "incremental")
    [j] = await state_store.claim_semantic_jobs(10, True)
    await state_store.complete_semantic_job(j["id"], j["claimed_at"])
    counts = await state_store.job_status_counts()
    assert counts.get("done", 0) == base.get("done", 0) + 1


async def test_stale_worker_cannot_complete_after_reap(state_store):
    await state_store.enqueue_semantic_job("fz2", "upsert", "h", "incremental")
    [j] = await state_store.claim_semantic_jobs(10, True)          # claimed_at = T1
    # force the claim stale (older than the lease), then run the REAL reaper -- it
    # re-pends the job but, per the reaper's actual behavior, does NOT null claimed_at
    async with (await state_store._get_pool()).acquire() as c:
        await c.execute(
            "UPDATE semantic_jobs SET claimed_at = now() - interval '2 hours' WHERE id=$1",
            j["id"])
    assert await state_store.reap_stale_jobs(3600, max_attempts=5) == 1
    async with (await state_store._get_pool()).acquire() as c:
        row = await c.fetchrow(
            "SELECT status, claimed_at FROM semantic_jobs WHERE id=$1", j["id"])
    assert row["status"] == "pending"
    stale_claimed_at = row["claimed_at"]          # still the T1-derived (now stale) value
    # the ORIGINAL worker's late complete: claimed_at still MATCHES, but status is no
    # longer 'in_progress' -- the status guard must make this a no-op
    await state_store.complete_semantic_job(j["id"], stale_claimed_at)
    async with (await state_store._get_pool()).acquire() as c:
        status = await c.fetchval("SELECT status FROM semantic_jobs WHERE id=$1", j["id"])
    assert status == "pending"          # NOT flipped to done by the stale worker
    # a fresh claim re-claims it with a new claimed_at (T2); the current owner can complete it
    [j2] = await state_store.claim_semantic_jobs(10, True)
    assert j2["id"] == j["id"]
    await state_store.complete_semantic_job(j2["id"], j2["claimed_at"])
    async with (await state_store._get_pool()).acquire() as c:
        assert await c.fetchval("SELECT status FROM semantic_jobs WHERE id=$1", j["id"]) == "done"


# NOTE: this test DROPs and recreates the shared semantic_jobs table, so it must
# run LAST in this module -- every other test above relies on the module-scoped
# state_store fixture's table (and, in the reaper test's case, on leaked rows
# from earlier tests). Keep this the final function in the file.
async def test_migration_adds_columns_to_existing_table(state_store):
    pool = await state_store._get_pool()
    async with pool.acquire() as c:
        await c.execute("DROP TABLE IF EXISTS semantic_jobs CASCADE")
        await c.execute(
            "CREATE TABLE semantic_jobs (id bigserial PRIMARY KEY, article_id text NOT NULL, "
            "op text NOT NULL, content_hash text, status text NOT NULL DEFAULT 'pending', "
            "attempts int NOT NULL DEFAULT 0, last_error text, enqueued_at timestamptz DEFAULT now(), "
            "claimed_at timestamptz, updated_at timestamptz DEFAULT now())")
        await c.execute("INSERT INTO semantic_jobs (article_id, op) VALUES ('old', 'upsert')")
    await state_store.init_schema()                       # runs the ALTERs
    async with pool.acquire() as c:
        row = await c.fetchrow("SELECT lane, next_attempt_at FROM semantic_jobs WHERE article_id='old'")
    assert row["lane"] == "incremental" and row["next_attempt_at"] is not None
    # round-trip still works post-migration
    await state_store.enqueue_semantic_job("mig1", "upsert", "h", "bootstrap")
    assert any(j["article_id"] == "mig1" for j in await state_store.claim_semantic_jobs(10, True))
