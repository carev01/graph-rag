# Pipeline Cleanup Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clear five group-A hygiene Minors: claim fencing token, dead-letter observability CLI, migration ALTER-path test, package-cycle refactor, and test/doc polish.

**Architecture:** Small, independent fixes across `graph_sync` (queue/CLI), a new neutral `docext` module, and `graph_extract` ontology. No new subsystems, no behavior change to extraction/temporal policy.

**Tech Stack:** Python 3.12, Postgres (asyncpg), Neo4j, httpx, pytest + testcontainers, ruff, mypy.

## Global Constraints

- **Layering:** `graph_sync → graph_extract` only. After Fix 4, `graph_extract` imports NOTHING from `graph_sync` (`grep -rn "graph_sync" src/graph_extract/` must be empty).
- **Fencing:** `complete`/`fail` only apply to a job still owned by the caller's claim (`AND claimed_at = $claimed_at`); a reaped+re-claimed job has a new `claimed_at`, so a stale worker's write no-ops. `record_tokens` stays unfenced (each attempt legitimately burned tokens).
- **No behavior change** to extraction, temporal policy, budget/retry logic; public ontology surface stays 9 types / 8 edges.
- Secrets only in untracked `.env`; DB tests use testcontainers; ruff/mypy clean throughout.

---

## File Structure

- `src/graph_sync/state_store.py` — claim RETURNING `claimed_at`; `complete`/`fail` gain `claimed_at`; new `job_status_counts`.
- `src/graph_sync/semantic_worker.py` — pass `job["claimed_at"]` to complete/fail.
- `src/graph_sync/cli.py` — `queue-status` command.
- `src/docext/__init__.py`, `src/docext/client.py` — **new** neutral DocExtractor client builder.
- `src/graph_sync/delta_client.py` — `make_client` delegates to the neutral builder.
- `src/graph_extract/cli.py`, `src/graph_extract/probe.py` — import from `docext.client` (drop `graph_sync` import).
- `src/graph_extract/ontology.py` — VMware docstring clarity.
- Tests: `tests/integration/test_semantic_jobs.py` (fencing, status counts, ALTER-path), `tests/integration/test_semantic_worker.py` (fencing via worker), `tests/unit/test_ontology.py` (noise markers).

---

### Task 1: Claim fencing token

**Files:**
- Modify: `src/graph_sync/state_store.py`, `src/graph_sync/semantic_worker.py`
- Test: `tests/integration/test_semantic_jobs.py`, `tests/integration/test_semantic_worker.py`

**Interfaces:**
- `claim_semantic_jobs` RETURNING adds `claimed_at` (job dicts gain `claimed_at`).
- `complete_semantic_job(job_id, claimed_at)`; `fail_semantic_job(job_id, error, *, max_attempts, retry_delay_seconds, claimed_at)`.

- [ ] **Step 1: Write the failing fencing test** (Postgres):

```python
async def test_complete_is_fenced_by_claimed_at(store):
    await store.enqueue_semantic_job("fz1", "upsert", "h", "incremental")
    [j] = await store.claim_semantic_jobs(10, True)         # claimed_at = T1
    # simulate reaper reset + re-claim -> new claimed_at (T2)
    async with (await store._get_pool()).acquire() as c:
        await c.execute("UPDATE semantic_jobs SET status='pending', claimed_at=NULL WHERE id=$1", j["id"])
    [j2] = await store.claim_semantic_jobs(10, True)        # claimed_at = T2, different
    # original worker's late complete with the STALE claimed_at must no-op
    await store.complete_semantic_job(j["id"], j["claimed_at"])
    # job is still in_progress under the T2 claim (not flipped to done)
    async with (await store._get_pool()).acquire() as c:
        status = await c.fetchval("SELECT status FROM semantic_jobs WHERE id=$1", j["id"])
    assert status == "in_progress"
    # the CURRENT owner (T2) can complete it
    await store.complete_semantic_job(j2["id"], j2["claimed_at"])
    async with (await store._get_pool()).acquire() as c:
        assert await c.fetchval("SELECT status FROM semantic_jobs WHERE id=$1", j["id"]) == "done"
```

- [ ] **Step 2: Run, confirm fail** (`complete_semantic_job` currently takes 1 arg). `uv run --extra dev pytest tests/integration/test_semantic_jobs.py -k fenced -v`.

- [ ] **Step 3: Add `claimed_at` to claim RETURNING** — in `claim_semantic_jobs`, append `, claimed_at` to the `RETURNING` list.

- [ ] **Step 4: Fence complete/fail:**

```python
async def complete_semantic_job(self, job_id: int, claimed_at) -> None:
    pool = await self._get_pool()
    await pool.execute(
        "UPDATE semantic_jobs SET status='done', updated_at=now() "
        "WHERE id=$1 AND claimed_at=$2", job_id, claimed_at)

async def fail_semantic_job(self, job_id: int, error: str, *, max_attempts: int,
                            retry_delay_seconds: float, claimed_at) -> None:
    pool = await self._get_pool()
    await pool.execute(
        "UPDATE semantic_jobs SET attempts=attempts+1, last_error=$2, "
        "status=CASE WHEN attempts+1 >= $3 THEN 'dead' ELSE 'pending' END, "
        "next_attempt_at=CASE WHEN attempts+1 >= $3 THEN next_attempt_at "
        "ELSE now() + make_interval(secs => $4) END, updated_at=now() "
        "WHERE id=$1 AND claimed_at=$5",
        job_id, error, max_attempts, retry_delay_seconds, claimed_at)
```

- [ ] **Step 5: Update the worker** (`semantic_worker.py` `run_worker_once`): pass `job["claimed_at"]` — `await store.complete_semantic_job(job["id"], job["claimed_at"])` and `await store.fail_semantic_job(job["id"], str(e), max_attempts=max_attempts, retry_delay_seconds=exp_backoff(...), claimed_at=job["claimed_at"])`.

- [ ] **Step 6: Fix other callers.** Grep `complete_semantic_job(` / `fail_semantic_job(` in `tests/` and update to pass `claimed_at` (the tests that claim a job have the job dict; those that don't can pass the row's `claimed_at` — for a job never claimed, `claimed_at` is NULL, so pass the value they read, or claim first). Keep each test's intent.

- [ ] **Step 7: Add a worker-level fencing test** in `test_semantic_worker.py`: enqueue + claim via the worker path is heavier — a store-level test (Step 1) is the core; for the worker, assert `run_worker_once` passes `claimed_at` by using a stub store recording the complete/fail args, OR rely on Step 1 + the existing worker tests still passing. Minimal: ensure existing worker tests pass with the new signature.

- [ ] **Step 8: Run green** (`test_semantic_jobs.py` + `test_semantic_worker.py` + full non-live) + gate.

- [ ] **Step 9: Commit.** `git add src/graph_sync/state_store.py src/graph_sync/semantic_worker.py tests/integration/test_semantic_jobs.py tests/integration/test_semantic_worker.py && git commit -m "fix(sync): fence complete/fail on claimed_at (no resurrecting a reaped+reclaimed job)"`

---

### Task 2: Dead-letter observability (`queue-status`)

**Files:**
- Modify: `src/graph_sync/state_store.py`, `src/graph_sync/cli.py`
- Test: `tests/integration/test_semantic_jobs.py`

**Interfaces:** `job_status_counts() -> dict[str, int]`.

- [ ] **Step 1: Write the failing test:**

```python
async def test_job_status_counts(store):
    base = await store.job_status_counts()
    await store.enqueue_semantic_job("qs1", "upsert", "h", "incremental")
    [j] = await store.claim_semantic_jobs(10, True)
    await store.complete_semantic_job(j["id"], j["claimed_at"])
    counts = await store.job_status_counts()
    assert counts.get("done", 0) == base.get("done", 0) + 1
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement `job_status_counts`:**

```python
async def job_status_counts(self) -> dict[str, int]:
    pool = await self._get_pool()
    rows = await pool.fetch("SELECT status, count(*) AS n FROM semantic_jobs GROUP BY status")
    return {r["status"]: r["n"] for r in rows}
```

- [ ] **Step 4: Add the `queue-status` CLI command** in `graph_sync/cli.py`, mirroring an existing command's build/`finally`-close pattern: build `StateStore(settings.postgres_dsn)`, `await store.init_schema()`, `_dump(await store.job_status_counts())`, close in `finally`. (Reuse the `_dump` helper if present, else `print(json.dumps(...))`.)

- [ ] **Step 5: Verify wiring** — `uv run --extra dev python -c "from typer.testing import CliRunner; from graph_sync.cli import app; print('queue-status' in CliRunner().invoke(app,['--help']).output)"` prints True.

- [ ] **Step 6: Run green + gate.**

- [ ] **Step 7: Commit.** `git add src/graph_sync/state_store.py src/graph_sync/cli.py tests/integration/test_semantic_jobs.py && git commit -m "feat(sync): queue-status CLI + job_status_counts (dead-letter visibility)"`

---

### Task 3: Migration ALTER-path test

**Files:**
- Test: `tests/integration/test_semantic_jobs.py` (test-only; no source change)

**Interfaces:** none.

- [ ] **Step 1: Write the test** — build a SECOND, isolated `StateStore` against a fresh scratch schema/DB so it doesn't collide with the module fixture (or use the module store's pool to DROP + recreate an A-era table before calling `init_schema`; whichever keeps the module fixture intact). Approach: on the module store's pool, `DROP TABLE IF EXISTS semantic_jobs CASCADE`, then create the **A-era** table (no `lane`/`next_attempt_at`), insert one `pending` row, then `await store.init_schema()`, then assert the columns exist with defaults and a lane-aware round-trip works. Restore isn't needed (init_schema recreates indexes; but be careful the DROP doesn't break other tests — prefer a SEPARATE store on a distinct schema via `SET search_path` or a temp table name). **Simplest safe approach:** create the A-era table under a temp name is not possible (init_schema hardcodes `semantic_jobs`), so run this test in a way that leaves the table in the final migrated shape (which it does — after init_schema the table has all columns), and order-independence holds because subsequent tests only need the migrated shape. Drain/rebuild as needed.

```python
async def test_migration_adds_columns_to_existing_table(store):
    pool = await store._get_pool()
    async with pool.acquire() as c:
        await c.execute("DROP TABLE IF EXISTS semantic_jobs CASCADE")
        await c.execute(
            "CREATE TABLE semantic_jobs (id bigserial PRIMARY KEY, article_id text NOT NULL, "
            "op text NOT NULL, content_hash text, status text NOT NULL DEFAULT 'pending', "
            "attempts int NOT NULL DEFAULT 0, last_error text, enqueued_at timestamptz DEFAULT now(), "
            "claimed_at timestamptz, updated_at timestamptz DEFAULT now())")
        await c.execute("INSERT INTO semantic_jobs (article_id, op) VALUES ('old', 'upsert')")
    await store.init_schema()                       # runs the ALTERs
    async with pool.acquire() as c:
        row = await c.fetchrow("SELECT lane, next_attempt_at FROM semantic_jobs WHERE article_id='old'")
    assert row["lane"] == "incremental" and row["next_attempt_at"] is not None
    # round-trip still works post-migration
    await store.enqueue_semantic_job("mig1", "upsert", "h", "bootstrap")
    assert any(j["article_id"] == "mig1" for j in await store.claim_semantic_jobs(10, True))
```
(If dropping the shared table risks other tests, mark this test to run last or isolate it — the implementer chooses the least-fragile option and documents it. The `ux_semantic_jobs_pending` index is recreated by `init_schema`'s `CREATE UNIQUE INDEX IF NOT EXISTS`, so post-migration the table is fully functional.)

- [ ] **Step 2: Run — it must genuinely exercise the ALTER branch** (the freshly-created A-era table lacks the columns until `init_schema` adds them). `uv run --extra dev pytest tests/integration/test_semantic_jobs.py -v`.

- [ ] **Step 3: Run full non-live + gate** (confirm the DROP/recreate didn't destabilize other tests in the file).

- [ ] **Step 4: Commit.** `git add tests/integration/test_semantic_jobs.py && git commit -m "test(sync): cover the existing-DB ALTER migration path"`

---

### Task 4: Package-cycle refactor (neutral DocExtractor client)

**Files:**
- Create: `src/docext/__init__.py`, `src/docext/client.py`
- Modify: `src/graph_sync/delta_client.py`, `src/graph_extract/cli.py`, `src/graph_extract/probe.py`

**Interfaces:** `docext.client.make_docext_client(*, base_url, read_key, admin_key, verify_tls, admin=False) -> httpx.AsyncClient`.

- [ ] **Step 1: Read `graph_sync/delta_client.make_client`** to copy its exact client-construction logic (headers, base_url, `verify`, the read-vs-admin key selection).

- [ ] **Step 2: Create `src/docext/client.py`** with the primitive-arg builder — the same construction, but keyed off the four fields instead of a `Settings` object:

```python
from __future__ import annotations
import httpx

def make_docext_client(*, base_url: str, read_key: str, admin_key: str,
                       verify_tls: bool, admin: bool = False) -> httpx.AsyncClient:
    key = admin_key if admin else read_key
    return httpx.AsyncClient(
        base_url=base_url,
        headers={"X-API-Key": key},
        verify=verify_tls,
        timeout=300.0,
    )
```
(This is `make_client`'s body verbatim with the four `settings.docext_*` reads replaced by the primitive args — `timeout=300.0`, header `X-API-Key`, `verify`.) Add empty `src/docext/__init__.py`.

- [ ] **Step 3: Make `graph_sync/delta_client.make_client` delegate:**

```python
from docext.client import make_docext_client

def make_client(settings: Settings, *, admin: bool = False) -> httpx.AsyncClient:
    return make_docext_client(
        base_url=settings.docext_base_url, read_key=settings.docext_read_key,
        admin_key=settings.docext_admin_key, verify_tls=settings.docext_verify_tls, admin=admin)
```
(Keep `DeltaStream`/`build_delta_params` in this module unchanged — sync-internal.)

- [ ] **Step 4: Switch `graph_extract`** — in `cli.py` and `probe.py`, replace `from graph_sync.delta_client import make_client` with `from docext.client import make_docext_client` and update the call sites to pass the four fields from the local settings (drop the `# type: ignore` — the primitive-arg builder needs no structural-compat hack).

- [ ] **Step 5: Prove the cycle is gone** — `grep -rn "graph_sync" src/graph_extract/` returns **nothing**. If `docext` needs packaging config, check `pyproject.toml`'s package discovery includes `src/docext` (it should, if it uses `src`-layout auto-discovery — verify).

- [ ] **Step 6: Run full non-live suite + gate** (the sync path and the extract path both still build working clients; `mypy src` clean, including the dropped `type: ignore`).

- [ ] **Step 7: Commit.** `git add src/docext/ src/graph_sync/delta_client.py src/graph_extract/cli.py src/graph_extract/probe.py && git commit -m "refactor: neutral docext client module (break graph_sync<->graph_extract cycle)"`

---

### Task 5: Test/doc polish

**Files:**
- Modify: `src/graph_extract/ontology.py`, `tests/unit/test_ontology.py`

- [ ] **Step 1: Extend the noise-exclusion test** — read `tests/unit/test_ontology.py`, find the assertion that `EXTRACTION_INSTRUCTIONS` mentions noise exclusions (arn/snap/install-module), and ADD assertions that it also mentions the error-code markers it actually names (`Failed` and `RequestId` — check the exact strings in `ontology.py`'s noise-exclusion block and assert those substrings, case-consistent with the instructions text).

- [ ] **Step 2: Run — confirm the new assertion passes** against the current instructions (the markers are already in the text; this just locks them). If a marker isn't literally present, assert the one that is (`...Failed`/`...RequestId`). `uv run --extra dev pytest tests/unit/test_ontology.py -v`.

- [ ] **Step 3: Clarify the VMware phrasing** in `ontology.py` — the `Platform` docstring lists `VMware vSphere` as a platform example while the Workload/canonical section also uses `VMware vSphere`. Adjust so the distinction reads clearly: Platform example → `VMware vSphere` (the hypervisor platform); ensure the Workload canonical list's VMware reference (if any) is unambiguous, or add a short parenthetical. One-line change; do NOT alter `ENTITY_TYPES`/`EDGE_TYPES`/`EDGE_TYPE_MAP` (public surface stays 9 types / 8 edges — a test already asserts this).

- [ ] **Step 4: Run `test_ontology.py` + gate** (public-surface test still green).

- [ ] **Step 5: Commit.** `git add src/graph_extract/ontology.py tests/unit/test_ontology.py && git commit -m "test/docs(extract): lock error-code noise markers + clarify VMware Platform/Workload"`

---

## Self-Review Notes

- **Spec coverage:** fencing → T1; observability → T2; ALTER test → T3; package cycle → T4; polish → T5. All five §-fixes covered.
- **Signature ripple:** `complete_semantic_job`/`fail_semantic_job` gain `claimed_at` (T1) — the worker + all test callers updated in T1 (grep in T1 Step 6). `claim_semantic_jobs` RETURNING is additive. `queue-status` and `job_status_counts` are new (T2).
- **Type consistency:** `claimed_at` threaded through claim→job dict→complete/fail; `make_docext_client(*, base_url, read_key, admin_key, verify_tls, admin=False)` used identically in delta_client + graph_extract.
- **Order/deps:** T1 and T2 both touch `state_store.py` (do T1 then T2). T3 test-only. T4 independent. T5 independent. Suggested: T1, T2, T3, T4, T5.
- **No-placeholder check:** each code step carries the actual code; the two spots marked `...` (client kwargs) are explicit instructions to copy the original `make_client`'s exact args, not vague placeholders.
