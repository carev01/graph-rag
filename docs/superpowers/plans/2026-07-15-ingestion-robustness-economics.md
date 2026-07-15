# Ingestion Robustness & Economics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Sub-slice-A incremental pipeline safe at corpus scale: priority lanes (incremental preempts bootstrap), a daily token budget gating bootstrap backfill, retry/backoff, dead-lettering, and an in-progress reaper.

**Architecture:** All changes are in `graph_sync` around the `semantic_jobs` queue (`StateStore`), the enqueue call (`sync_core`), and the worker (`semantic_worker`/`cli`). The worker meters extraction tokens (read from `graph_extract.usage.get_tally()`) into a Postgres `token_ledger`. No `graph_extract` behavior change.

**Tech Stack:** Python 3.12, Postgres (asyncpg), pytest + testcontainers (Postgres), ruff, mypy.

## Global Constraints

- **Budget policy:** incremental-lane jobs are NEVER budget-gated; bootstrap-lane jobs are withheld once the day's recorded tokens reach `daily_token_budget`.
- **Lane inference:** `sync_core._apply_record` derives lane from the `res` it already holds — `BootstrapResult → 'bootstrap'`, `IncrementalResult → 'incremental'`. An incremental enqueue upgrades a pending bootstrap job; a bootstrap enqueue never downgrades an incremental one.
- **Retry keeps the job `pending`** (so A's pending-unique `ON CONFLICT` still collapses a new delta onto it); a backed-off job is hidden by `next_attempt_at > now()`. Dead-letter is `status='dead'` in place (error retained).
- **Soft budget cap:** the gate is per claim-batch (bounded one-batch overshoot), not per token — intended.
- **Migration is additive & idempotent:** `ALTER TABLE … ADD COLUMN IF NOT EXISTS` + `CREATE TABLE IF NOT EXISTS` in `init_schema`.
- **Layering:** `graph_sync → graph_extract` only (the worker reads `graph_extract.usage`).
- Secrets only in untracked `.env`; DB tests use testcontainers.

---

## File Structure

- `src/graph_sync/state_store.py` — **modify:** migration (lane, next_attempt_at, token_ledger); lane-aware `enqueue`; `claim(batch, include_bootstrap)`; retry-aware `fail`; `reap_stale_jobs`; `record_tokens`/`today_token_total`; `dead_semantic_job_count`.
- `src/graph_sync/config.py` — **modify:** budget/retry/reaper settings.
- `src/graph_sync/sync_core.py` — **modify:** pass inferred lane to `enqueue_semantic_job`.
- `src/graph_sync/semantic_worker.py` — **modify:** reaper + budget gate + token accounting + retry-aware fail; a pure `exp_backoff`.
- `src/graph_sync/cli.py` — **modify:** wire the new settings into the `worker` command.
- Tests: `tests/integration/test_semantic_jobs.py` (extend), `tests/integration/test_semantic_worker.py` (extend), `tests/unit/test_sync_core_enqueue.py` (extend), `tests/unit/test_backoff.py` (new).

---

### Task 1: Schema migration + lane-aware enqueue

**Files:**
- Modify: `src/graph_sync/state_store.py`
- Test: `tests/integration/test_semantic_jobs.py`

**Interfaces:**
- Produces: migrated `semantic_jobs` (+`lane`, +`next_attempt_at`), new `token_ledger`; `enqueue_semantic_job(article_id, op, content_hash, lane="incremental")` with lane upgrade-not-downgrade.

- [ ] **Step 1: Write failing integration tests.**

```python
async def test_lane_upgrade_not_downgrade(store):
    await store.enqueue_semantic_job("a1", "upsert", "h", "bootstrap")
    await store.enqueue_semantic_job("a1", "upsert", "h", "incremental")  # upgrades
    [j] = await store.claim_semantic_jobs(10, True)
    assert j["lane"] == "incremental"

async def test_lane_incremental_not_downgraded(store):
    await store.enqueue_semantic_job("a2", "upsert", "h", "incremental")
    await store.enqueue_semantic_job("a2", "upsert", "h", "bootstrap")  # must NOT downgrade
    [j] = await store.claim_semantic_jobs(10, True)
    assert j["lane"] == "incremental"
```
(These also depend on Task-2's `claim(batch, include_bootstrap)` signature — write both tasks' claim signature as `claim_semantic_jobs(self, batch: int, include_bootstrap: bool)` now; Task 1 may implement a minimal claim that returns `lane`, Task 2 adds the ordering/gate. Simpler: implement the full claim here per Task 2's SQL and have Task 2 only add tests — but keep the split by having Task 1 add `lane` to RETURNING and the `include_bootstrap` param, Task 2 add lane-ordering + next_attempt_at gate + budget tests.)

- [ ] **Step 2: Run, confirm fail.** `uv run --extra dev pytest tests/integration/test_semantic_jobs.py -k lane -v`.

- [ ] **Step 3: Add migration** to `_SCHEMA` in `state_store.py` — extend the `CREATE TABLE semantic_jobs` with `lane text NOT NULL DEFAULT 'incremental'` and `next_attempt_at timestamptz NOT NULL DEFAULT now()` (for fresh DBs), AND append idempotent ALTERs + the new table (for existing DBs):

```sql
ALTER TABLE semantic_jobs ADD COLUMN IF NOT EXISTS lane text NOT NULL DEFAULT 'incremental';
ALTER TABLE semantic_jobs ADD COLUMN IF NOT EXISTS next_attempt_at timestamptz NOT NULL DEFAULT now();
CREATE TABLE IF NOT EXISTS token_ledger (day date PRIMARY KEY, tokens bigint NOT NULL DEFAULT 0);
```

- [ ] **Step 4: Update `enqueue_semantic_job`** to take `lane: str = "incremental"` and set it with upgrade semantics:

```python
async def enqueue_semantic_job(self, article_id: str, op: str,
                               content_hash: str | None, lane: str = "incremental") -> None:
    pool = await self._get_pool()
    await pool.execute(
        "INSERT INTO semantic_jobs (article_id, op, content_hash, lane) VALUES ($1,$2,$3,$4) "
        "ON CONFLICT (article_id) WHERE status='pending' DO UPDATE SET "
        "op=excluded.op, content_hash=excluded.content_hash, "
        "lane=CASE WHEN excluded.lane='incremental' OR semantic_jobs.lane='incremental' "
        "THEN 'incremental' ELSE 'bootstrap' END, "
        "enqueued_at=now(), updated_at=now()",
        article_id, op, content_hash, lane)
```

- [ ] **Step 5: Update `claim_semantic_jobs`** signature to `(self, batch: int, include_bootstrap: bool)` and add `lane` to RETURNING (full ordering/gate lands in Task 2 — for now include the `include_bootstrap` filter and `next_attempt_at <= now()` guard so later tasks compose):

```python
async def claim_semantic_jobs(self, batch: int, include_bootstrap: bool) -> list[dict]:
    pool = await self._get_pool()
    rows = await pool.fetch(
        "UPDATE semantic_jobs SET status='in_progress', claimed_at=now(), updated_at=now() "
        "WHERE id IN (SELECT id FROM semantic_jobs "
        "WHERE status='pending' AND next_attempt_at <= now() "
        "AND ($2 OR lane='incremental') "
        "ORDER BY (lane='bootstrap'), next_attempt_at "
        "FOR UPDATE SKIP LOCKED LIMIT $1) "
        "RETURNING id, article_id, op, content_hash, attempts, lane",
        batch, include_bootstrap)
    return [dict(r) for r in rows]
```

- [ ] **Step 6: Fix existing callers.** The A tests call `claim_semantic_jobs(n)` — update `tests/integration/test_semantic_jobs.py` existing tests to `claim_semantic_jobs(n, True)`. (Task 5 updates the worker's call.)

- [ ] **Step 7: Run green** (new + updated A tests) + gate on `state_store.py`.

- [ ] **Step 8: Commit.** `git add src/graph_sync/state_store.py tests/integration/test_semantic_jobs.py && git commit -m "feat(sync): semantic_jobs lane + next_attempt_at migration + lane-aware enqueue"`

---

### Task 2: Lane-ordered budget claim + token ledger

**Files:**
- Modify: `src/graph_sync/state_store.py`
- Test: `tests/integration/test_semantic_jobs.py`

**Interfaces:**
- Produces: `record_tokens(delta: int) -> None` (upsert-increment today's `token_ledger`); `today_token_total() -> int`. (The claim ordering/gate was implemented in Task 1's SQL — this task adds the tests proving it, plus the ledger methods.)

- [ ] **Step 1: Write failing tests.**

```python
async def test_incremental_claimed_before_bootstrap(store):
    await store.enqueue_semantic_job("b1", "upsert", "h", "bootstrap")
    await store.enqueue_semantic_job("i1", "upsert", "h", "incremental")
    [first] = await store.claim_semantic_jobs(1, True)
    assert first["article_id"] == "i1"          # incremental preempts

async def test_bootstrap_withheld_when_excluded(store):
    await store.enqueue_semantic_job("b1", "upsert", "h", "bootstrap")
    assert await store.claim_semantic_jobs(10, False) == []   # bootstrap excluded
    assert len(await store.claim_semantic_jobs(10, True)) == 1

async def test_token_ledger_accumulates(store):
    await store.record_tokens(100)
    await store.record_tokens(50)
    assert await store.today_token_total() == 150

async def test_today_token_total_zero_when_empty(store):
    assert await store.today_token_total() == 0
```

- [ ] **Step 2: Run, confirm fail** (record_tokens/today_token_total missing; the two claim tests should already PASS if Task 1's SQL is correct — if they fail, fix Task 1's claim SQL).

- [ ] **Step 3: Add the ledger methods.**

```python
async def record_tokens(self, delta: int) -> None:
    pool = await self._get_pool()
    await pool.execute(
        "INSERT INTO token_ledger (day, tokens) VALUES (current_date, $1) "
        "ON CONFLICT (day) DO UPDATE SET tokens = token_ledger.tokens + $1", delta)

async def today_token_total(self) -> int:
    pool = await self._get_pool()
    return await pool.fetchval(
        "SELECT COALESCE((SELECT tokens FROM token_ledger WHERE day=current_date), 0)")
```

- [ ] **Step 4: Run green** + gate.

- [ ] **Step 5: Commit.** `git add src/graph_sync/state_store.py tests/integration/test_semantic_jobs.py && git commit -m "feat(sync): lane-ordered budget claim + token_ledger accounting"`

---

### Task 3: Retry/backoff, dead-letter, reaper

**Files:**
- Modify: `src/graph_sync/state_store.py`
- Test: `tests/integration/test_semantic_jobs.py`

**Interfaces:**
- Produces: `fail_semantic_job(job_id, error, *, max_attempts, retry_delay_seconds)` (retry-aware); `reap_stale_jobs(lease_seconds) -> int`; `dead_semantic_job_count() -> int`.

- [ ] **Step 1: Write failing tests.**

```python
async def test_fail_retries_then_dead(store):
    await store.enqueue_semantic_job("a1", "upsert", "h", "incremental")
    [j] = await store.claim_semantic_jobs(10, True)
    await store.fail_semantic_job(j["id"], "boom", max_attempts=2, retry_delay_seconds=0)
    # attempts now 1 (<2): back to pending, re-claimable (delay 0)
    [j2] = await store.claim_semantic_jobs(10, True)
    await store.fail_semantic_job(j2["id"], "boom2", max_attempts=2, retry_delay_seconds=0)
    # attempts now 2 (>=2): dead, not re-claimable
    assert await store.claim_semantic_jobs(10, True) == []
    assert await store.dead_semantic_job_count() == 1

async def test_backoff_hides_job(store):
    await store.enqueue_semantic_job("a2", "upsert", "h", "incremental")
    [j] = await store.claim_semantic_jobs(10, True)
    await store.fail_semantic_job(j["id"], "x", max_attempts=5, retry_delay_seconds=3600)
    assert await store.claim_semantic_jobs(10, True) == []   # next_attempt_at in the future

async def test_reaper_reclaims_stale_inprogress(store):
    await store.enqueue_semantic_job("a3", "upsert", "h", "incremental")
    [j] = await store.claim_semantic_jobs(10, True)          # now in_progress, fresh claimed_at
    assert await store.reap_stale_jobs(3600) == 0            # fresh, not reaped
    # force claimed_at into the past, then reap
    async with (await store._get_pool()).acquire() as c:
        await c.execute("UPDATE semantic_jobs SET claimed_at = now() - interval '2 hours' WHERE id=$1", j["id"])
    assert await store.reap_stale_jobs(3600) == 1
    assert len(await store.claim_semantic_jobs(10, True)) == 1  # re-claimable
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement.**

```python
async def fail_semantic_job(self, job_id: int, error: str, *,
                            max_attempts: int, retry_delay_seconds: float) -> None:
    pool = await self._get_pool()
    await pool.execute(
        "UPDATE semantic_jobs SET attempts=attempts+1, last_error=$2, "
        "status=CASE WHEN attempts+1 >= $3 THEN 'dead' ELSE 'pending' END, "
        "next_attempt_at=CASE WHEN attempts+1 >= $3 THEN next_attempt_at "
        "ELSE now() + make_interval(secs => $4) END, updated_at=now() "
        "WHERE id=$1", job_id, error, max_attempts, retry_delay_seconds)

async def reap_stale_jobs(self, lease_seconds: float) -> int:
    pool = await self._get_pool()
    res = await pool.execute(
        "UPDATE semantic_jobs SET status='pending', attempts=attempts+1, "
        "next_attempt_at=now(), updated_at=now() "
        "WHERE status='in_progress' AND claimed_at < now() - make_interval(secs => $1)",
        lease_seconds)
    return int(res.split()[-1])   # "UPDATE <n>"

async def dead_semantic_job_count(self) -> int:
    pool = await self._get_pool()
    return await pool.fetchval("SELECT count(*) FROM semantic_jobs WHERE status='dead'")
```

- [ ] **Step 4: Run green** + gate. (Note: this changes `fail_semantic_job`'s signature — the A worker calls the old form; Task 5 updates the worker. If any current test calls the old `fail_semantic_job(id, err)`, update it.)

- [ ] **Step 5: Commit.** `git add src/graph_sync/state_store.py tests/integration/test_semantic_jobs.py && git commit -m "feat(sync): retry/backoff + dead-letter + in-progress reaper"`

---

### Task 4: `sync_core` lane inference

**Files:**
- Modify: `src/graph_sync/sync_core.py`
- Test: `tests/unit/test_sync_core_enqueue.py`

**Interfaces:** Consumes `enqueue_semantic_job(..., lane)` (Task 1).

- [ ] **Step 1: Write failing unit tests** (extend the Task-A enqueue tests). A `ContentRecord` applied with a `BootstrapResult` → the recorded enqueue call has `lane="bootstrap"`; with an `IncrementalResult` → `lane="incremental"`; a tombstone (only on incremental) → `lane="incremental"`. The fake store's `enqueue_semantic_job` recorder must accept the 4th `lane` arg.

```python
async def test_bootstrap_lane_inferred():
    # drive _apply_record with a BootstrapResult
    assert store.enqueue_calls == [("<id>", "upsert", "<hash>", "bootstrap")]

async def test_incremental_lane_inferred():
    assert store.enqueue_calls == [("<id>", "upsert", "<hash>", "incremental")]
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement.** In `_apply_record`, compute `lane = "bootstrap" if isinstance(res, BootstrapResult) else "incremental"` once at the top, and pass it to every `enqueue_semantic_job(...)` call (both upsert paths and the tombstone path). `BootstrapResult`/`IncrementalResult` are already imported in `sync_core.py`.

- [ ] **Step 4: Run green** (enqueue tests + guards) + gate.

- [ ] **Step 5: Commit.** `git add src/graph_sync/sync_core.py tests/unit/test_sync_core_enqueue.py && git commit -m "feat(sync): tag semantic jobs with bootstrap/incremental lane"`

---

### Task 5: Worker — reaper + budget gate + token accounting + retry, and CLI config

**Files:**
- Modify: `src/graph_sync/semantic_worker.py`, `src/graph_sync/config.py`, `src/graph_sync/cli.py`
- Test: `tests/integration/test_semantic_worker.py`, `tests/unit/test_backoff.py`

**Interfaces:** Consumes Tasks 1-3 store methods + `graph_extract.usage.get_tally`.

- [ ] **Step 1: Add config** to `graph_sync/config.py` `Settings`: `semantic_daily_token_budget: int` (default e.g. 5_000_000), `semantic_max_attempts: int` (5), `semantic_backoff_base_seconds: float` (30), `semantic_backoff_cap_seconds: float` (3600), `semantic_reaper_lease_seconds: float` (1800). Match the file's pydantic-settings idiom.

- [ ] **Step 2: Write a failing unit test for `exp_backoff`** in `tests/unit/test_backoff.py`:

```python
from graph_sync.semantic_worker import exp_backoff
def test_exp_backoff_grows_and_caps():
    assert exp_backoff(1, base=30, cap=3600) == 30
    assert exp_backoff(2, base=30, cap=3600) == 60
    assert exp_backoff(3, base=30, cap=3600) == 120
    assert exp_backoff(20, base=30, cap=3600) == 3600   # capped
```

- [ ] **Step 3: Write a failing worker integration test** (Postgres + a stub ingest that fails on the first call, succeeds on the second, and reports a fixed token count):

```python
async def test_worker_reaps_budgets_meters_and_retries(store):
    # a stub ingest: first ingest_article raises, second succeeds; tally stub returns tokens.
    # enqueue an incremental upsert; run_worker_once with a budget config:
    #  - first pass: job fails -> pending w/ retry (delay 0 so re-claimable)
    #  - second pass: succeeds -> done, tokens recorded to token_ledger
    # also enqueue a bootstrap job and set token_ledger over budget -> bootstrap withheld.
    ...
```
(Design the stub to make `run_worker_once` observable: assert the job ends `done`, `today_token_total() > 0`, and that with the ledger over budget a bootstrap job is not processed.)

- [ ] **Step 4: Implement `exp_backoff`** (pure) and update the worker:

```python
def exp_backoff(attempts: int, *, base: float, cap: float) -> float:
    return min(base * (2 ** (attempts - 1)), cap)
```

Update `run_worker_once` to the Task-B shape:
```python
async def run_worker_once(store, ingest, *, batch, budget, max_attempts,
                          backoff_base, backoff_cap, lease) -> int:
    await store.reap_stale_jobs(lease)
    include_bootstrap = await store.today_token_total() < budget
    jobs = await store.claim_semantic_jobs(batch, include_bootstrap)
    for job in jobs:
        before = get_tally()
        t0 = before.prompt_tokens + before.completion_tokens
        try:
            if job["op"] == "upsert":
                await ingest.ingest_article(job["article_id"])
            elif job["op"] == "remove":
                await ingest.tombstone_article_episodes(job["article_id"])
            else:
                raise ValueError(f"unknown op {job['op']!r}")
            after = get_tally()
            delta = (after.prompt_tokens + after.completion_tokens) - t0
            if delta:
                await store.record_tokens(delta)
            await store.complete_semantic_job(job["id"])
        except Exception as e:
            logger.exception("semantic job %s failed", job["id"])
            await store.fail_semantic_job(
                job["id"], str(e), max_attempts=max_attempts,
                retry_delay_seconds=exp_backoff(job["attempts"] + 1, base=backoff_base, cap=backoff_cap))
    return len(jobs)
```
(`from graph_extract.usage import get_tally`.) Thread the same params through `run_worker(...)`.

- [ ] **Step 5: Update the `worker` CLI command** to read the new settings and pass them to `run_worker`. Keep `_build_worker_deps` + the finally-close intact.

- [ ] **Step 6: Run green** (backoff unit + worker integration + full non-live) + gate. Confirm no stale `claim_semantic_jobs(n)` / old `fail_semantic_job(id, err)` callers remain (grep).

- [ ] **Step 7: Commit.** `git add src/graph_sync/semantic_worker.py src/graph_sync/config.py src/graph_sync/cli.py tests/integration/test_semantic_worker.py tests/unit/test_backoff.py && git commit -m "feat(sync): worker reaper + token-budget gate + retry/backoff + CLI config"`

---

## Self-Review Notes

- **Spec coverage:** lanes → T1(enqueue)+T1/T2(claim ordering)+T4(inference); budget → T2(ledger)+T5(gate/accounting); retry/backoff/dead-letter → T3+T5; reaper → T3+T5. All §-sections covered.
- **Signature-change ripple:** `claim_semantic_jobs` gains `include_bootstrap` (T1) and `fail_semantic_job` changes shape (T3) — both have callers in the A tests + worker; each task's steps update its callers, and T5 finalizes the worker. Grep for stragglers in T5 Step 6.
- **Type consistency:** `enqueue_semantic_job(...,lane)`, `claim_semantic_jobs(batch, include_bootstrap)`, `fail_semantic_job(id, err, *, max_attempts, retry_delay_seconds)`, `reap_stale_jobs`, `record_tokens`, `today_token_total`, `dead_semantic_job_count`, `exp_backoff(attempts,*,base,cap)` used identically across tasks.
- **Order/deps:** T1 → T2 → T3 (all `state_store`); T4 needs T1's enqueue lane; T5 needs T1-T3 + config. Suggested: T1, T2, T3, T4, T5.
- **No-placeholder check:** every code step carries the actual code/SQL; tests are concrete.
