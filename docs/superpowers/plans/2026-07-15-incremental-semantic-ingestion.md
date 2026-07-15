# Incremental Semantic Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the slice-1 delta sync to the semantic graph via a durable Postgres job queue + standalone worker, and make article updates/removals temporally correct (append + supersede, never delete), fixing the `provenance.link` duplicate-edge bug.

**Architecture:** `sync_core` enqueues a `semantic_jobs` row on each content change/tombstone; a standalone `graph_sync` worker process claims jobs (SKIP LOCKED) and calls `graph_extract.ingest_driver`. Dependency direction is `graph_sync → graph_extract` only.

**Tech Stack:** Python 3.12, Postgres (asyncpg), Neo4j 5.26 (async), graphiti-core 0.29.2, FastAPI/typer, pytest + testcontainers (Postgres + Neo4j), ruff, mypy.

## Global Constraints

- **Layering is one-directional:** `graph_sync` may import `graph_extract`; `graph_extract` must NEVER import `graph_sync`.
- **Temporal policy (design-decision #3):** updates APPEND (add episodes, mark prior `superseded=true`); removals mark `removed=true`. NEVER delete episodes, NEVER call `remove_episode`. Graphiti's `add_episode` handles fact invalidation.
- **At-least-once + idempotent:** enqueue is idempotent (one `pending` job per article, latest wins); the per-chunk `content_hash` gate (`Provenance.already_ingested`) makes re-runs safe.
- **Exactly one `HAS_EPISODE` per `(article_id, chunk_index)`.**
- **State store is asyncpg Postgres** — add to the existing `_SCHEMA` `CREATE TABLE IF NOT EXISTS` block in `state_store.py`. The existing `dead_letter` table is the sync's, unrelated.
- **No rate cap / retry-backoff / dead-letter / budget** — those are Sub-slice B. A failed job is marked `failed` and left.
- Secrets only in untracked `.env`; tests use testcontainers.

---

## File Structure

- `src/graph_sync/state_store.py` — **modify:** `semantic_jobs` DDL + queue methods.
- `src/graph_sync/sync_core.py` — **modify:** enqueue in `_apply_record`.
- `src/graph_sync/semantic_worker.py` — **new:** the worker loop (`run_worker_once`, `run_worker`).
- `src/graph_sync/cli.py` — **modify:** a `worker` typer command.
- `src/graph_extract/provenance.py` — **modify:** re-keyed `link`.
- `src/graph_extract/ingest_driver.py` — **modify:** `tombstone_article_episodes`, supersede trailing chunks.
- Tests: `tests/integration/test_semantic_jobs.py` (new), `tests/unit/test_sync_core_enqueue.py` (new), `tests/integration/test_provenance_rekey.py` (new), `tests/integration/test_temporal_update.py` (new), `tests/integration/test_semantic_worker.py` (new).

---

### Task 1: `semantic_jobs` queue in `StateStore`

**Files:**
- Modify: `src/graph_sync/state_store.py`
- Test: `tests/integration/test_semantic_jobs.py`

**Interfaces:**
- Produces:
  - `async enqueue_semantic_job(article_id: str, op: str, content_hash: str | None) -> None`
  - `async claim_semantic_jobs(batch: int) -> list[dict]` — each dict has `id, article_id, op, content_hash, attempts`.
  - `async complete_semantic_job(job_id: int) -> None`
  - `async fail_semantic_job(job_id: int, error: str) -> None`

- [ ] **Step 1: Write failing integration tests** (Postgres testcontainer — mirror the existing state-store integration tests' fixture; find them with `grep -rl StateStore tests/`).

```python
async def test_enqueue_idempotent_pending(store):
    await store.enqueue_semantic_job("a1", "upsert", "h1")
    await store.enqueue_semantic_job("a1", "upsert", "h2")  # newer change, still pending
    jobs = await store.claim_semantic_jobs(10)
    assert len(jobs) == 1 and jobs[0]["content_hash"] == "h2"  # collapsed, latest wins

async def test_claim_skiplocked_disjoint(store):
    for i in range(4):
        await store.enqueue_semantic_job(f"a{i}", "upsert", "h")
    a = await store.claim_semantic_jobs(2)
    b = await store.claim_semantic_jobs(2)
    assert {j["id"] for j in a}.isdisjoint({j["id"] for j in b}) and len(a) == 2 and len(b) == 2

async def test_complete_and_fail(store):
    await store.enqueue_semantic_job("a1", "remove", None)
    [j] = await store.claim_semantic_jobs(10)
    await store.complete_semantic_job(j["id"])
    assert await store.claim_semantic_jobs(10) == []          # done, not re-claimable
    await store.enqueue_semantic_job("a2", "upsert", "h")
    [k] = await store.claim_semantic_jobs(10)
    await store.fail_semantic_job(k["id"], "boom")
    assert await store.claim_semantic_jobs(10) == []          # failed, not re-claimable
```

- [ ] **Step 2: Run, confirm fail** (`grep`/run the file): `uv run --extra dev pytest tests/integration/test_semantic_jobs.py -v` → FAIL (methods/table missing).

- [ ] **Step 3: Add the DDL** to `_SCHEMA` in `state_store.py`:

```sql
CREATE TABLE IF NOT EXISTS semantic_jobs (
  id bigserial PRIMARY KEY, article_id text NOT NULL,
  op text NOT NULL, content_hash text,
  status text NOT NULL DEFAULT 'pending', attempts int NOT NULL DEFAULT 0,
  last_error text, enqueued_at timestamptz DEFAULT now(),
  claimed_at timestamptz, updated_at timestamptz DEFAULT now());
CREATE UNIQUE INDEX IF NOT EXISTS ux_semantic_jobs_pending
  ON semantic_jobs(article_id) WHERE status='pending';
```

- [ ] **Step 4: Add the methods** to `StateStore`:

```python
async def enqueue_semantic_job(self, article_id: str, op: str, content_hash: str | None) -> None:
    pool = await self._get_pool()
    await pool.execute(
        "INSERT INTO semantic_jobs (article_id, op, content_hash) VALUES ($1,$2,$3) "
        "ON CONFLICT (article_id) WHERE status='pending' "
        "DO UPDATE SET op=excluded.op, content_hash=excluded.content_hash, "
        "enqueued_at=now(), updated_at=now()",
        article_id, op, content_hash)

async def claim_semantic_jobs(self, batch: int) -> list[dict]:
    pool = await self._get_pool()
    rows = await pool.fetch(
        "UPDATE semantic_jobs SET status='in_progress', claimed_at=now(), updated_at=now() "
        "WHERE id IN (SELECT id FROM semantic_jobs WHERE status='pending' "
        "ORDER BY enqueued_at FOR UPDATE SKIP LOCKED LIMIT $1) "
        "RETURNING id, article_id, op, content_hash, attempts", batch)
    return [dict(r) for r in rows]

async def complete_semantic_job(self, job_id: int) -> None:
    pool = await self._get_pool()
    await pool.execute(
        "UPDATE semantic_jobs SET status='done', updated_at=now() WHERE id=$1", job_id)

async def fail_semantic_job(self, job_id: int, error: str) -> None:
    pool = await self._get_pool()
    await pool.execute(
        "UPDATE semantic_jobs SET status='failed', attempts=attempts+1, "
        "last_error=$2, updated_at=now() WHERE id=$1", job_id, error)
```

- [ ] **Step 5: Run tests green.** `uv run --extra dev pytest tests/integration/test_semantic_jobs.py -v`; `ruff check` + `mypy src/graph_sync`.

- [ ] **Step 6: Commit.** `git add src/graph_sync/state_store.py tests/integration/test_semantic_jobs.py && git commit -m "feat(sync): semantic_jobs durable queue (enqueue/claim/complete/fail)"`

---

### Task 2: Enqueue trigger in `sync_core._apply_record`

**Files:**
- Modify: `src/graph_sync/sync_core.py` (`_apply_record`, `src/graph_sync/sync_core.py:58-100`)
- Test: `tests/unit/test_sync_core_enqueue.py`

**Interfaces:**
- Consumes: `StateStore.enqueue_semantic_job` (Task 1).

**Details:** enqueue BEFORE the structural write in each path, so a mid-batch failure re-applies + re-enqueues (idempotent). Tombstone → `('remove', None)`; content change (both the `apply_incomplete_article` path and the `apply_structural` path) → `('upsert', rec.content_hash)`. The unchanged-hash early-return (`existing == rec.content_hash`) must NOT enqueue.

- [ ] **Step 1: Write failing unit tests** using a fake store that records enqueue calls (extend the fake in `tests/unit/test_sync_core_guards.py` — it already has `_FakeStore`; add an `enqueue_semantic_job` recorder). Cover: an updated content record → one `('upsert', hash)`; a tombstone → one `('remove', None)`; an unchanged-hash replay → zero enqueues.

```python
async def test_content_change_enqueues_upsert():
    # build a SyncCore with a fake repo (get_content_hash -> None), fake store,
    # feed one ContentRecord via _apply_record, assert:
    assert store.enqueue_calls == [("<id>", "upsert", "<hash>")]

async def test_tombstone_enqueues_remove(): ...
async def test_unchanged_hash_does_not_enqueue(): ...
```

- [ ] **Step 2: Run, confirm fail.** `uv run --extra dev pytest tests/unit/test_sync_core_enqueue.py -v`.

- [ ] **Step 3: Add the enqueue calls** in `_apply_record`:
  - Tombstone branch: before `self._repo.tombstone_article(...)`, `await self._store.enqueue_semantic_job(rec.id, "remove", None)`.
  - After the unchanged-hash early return (so it's NOT reached on skip): in BOTH the `apply_incomplete_article` path and the `apply_structural` path, call `await self._store.enqueue_semantic_job(rec.id, "upsert", rec.content_hash)` BEFORE the repo write.

- [ ] **Step 4: Run tests green** + the existing `test_sync_core_guards.py` (ensure the fake store still satisfies SyncCore). `uv run --extra dev pytest tests/unit/test_sync_core_enqueue.py tests/unit/test_sync_core_guards.py -v`; gate.

- [ ] **Step 5: Commit.** `git add src/graph_sync/sync_core.py tests/unit/test_sync_core_enqueue.py && git commit -m "feat(sync): enqueue semantic job on content change / tombstone"`

---

### Task 3: Re-keyed `provenance.link` (fix duplicate `HAS_EPISODE`)

**Files:**
- Modify: `src/graph_extract/provenance.py` (`link`, `src/graph_extract/provenance.py:17-26`)
- Test: `tests/integration/test_provenance_rekey.py`

**Interfaces:** `link(...)` signature unchanged; behavior: exactly one `HAS_EPISODE` per `(article_id, chunk_index)`, old episode marked `superseded=true`, idempotent.

- [ ] **Step 1: Write failing integration test** (Neo4j testcontainer — use the `extract_driver` fixture pattern from `tests/integration/test_eval_quality.py`).

```python
async def test_link_rekey_supersedes_and_dedupes(extract_driver):
    from graph_extract.provenance import Provenance
    p = Provenance(extract_driver)
    async with extract_driver.session() as s:
        await s.run("CREATE (:Article {id:'a1'})")
        await s.run("CREATE (:Episodic {uuid:'e_old'})")
        await s.run("CREATE (:Episodic {uuid:'e_new'})")
    await p.link("a1", "e_old", chunk_index=0, heading_path="", token_count=1, content_hash="h1")
    await p.link("a1", "e_new", chunk_index=0, heading_path="", token_count=1, content_hash="h2")
    async with extract_driver.session() as s:
        n = (await (await s.run(
            "MATCH (:Article {id:'a1'})-[r:HAS_EPISODE {chunk_index:0}]->() RETURN count(r) AS c")).single())["c"]
        old_sup = (await (await s.run(
            "MATCH (e:Episodic {uuid:'e_old'}) RETURN e.superseded AS s")).single())["s"]
    assert n == 1 and old_sup is True
    # idempotent: re-linking the same new episode is a no-op
    await p.link("a1", "e_new", chunk_index=0, heading_path="", token_count=1, content_hash="h2")
```

- [ ] **Step 2: Run, confirm fail** (current code creates 2 edges): `uv run --extra dev pytest tests/integration/test_provenance_rekey.py -v` → FAIL (`c == 2`).

- [ ] **Step 3: Rewrite `link`** with the one-transaction re-key:

```python
async def link(self, article_id: str, episode_uuid: str, *, chunk_index: int,
               heading_path: str, token_count: int, content_hash: str) -> None:
    async with self._driver.session() as s:
        await s.run(
            "MATCH (a:Article {id:$a}) "
            "OPTIONAL MATCH (a)-[old:HAS_EPISODE {chunk_index:$i}]->(oldE:Episodic) "
            "WHERE oldE.uuid <> $u "
            "SET oldE.superseded = true DELETE old "
            "WITH a "
            "MATCH (e:Episodic {uuid:$u}) "
            "MERGE (a)-[r:HAS_EPISODE {chunk_index:$i}]->(e) "
            "SET r.heading_path=$hp, r.token_count=$tc, r.content_hash=$h",
            a=article_id, u=episode_uuid, i=chunk_index, hp=heading_path,
            tc=token_count, h=content_hash)
```

- [ ] **Step 4: Run tests green** (new + any existing provenance/ingest integration tests). Gate on `src/graph_extract/provenance.py`.

- [ ] **Step 5: Commit.** `git add src/graph_extract/provenance.py tests/integration/test_provenance_rekey.py && git commit -m "fix(extract): re-key provenance link (one HAS_EPISODE per chunk, supersede old)"`

---

### Task 4: Temporal update in `ingest_driver` (tombstone + shrink-supersede)

**Files:**
- Modify: `src/graph_extract/ingest_driver.py`
- Test: `tests/integration/test_temporal_update.py`

**Interfaces:**
- Produces: `async tombstone_article_episodes(article_id: str) -> int` (count marked `removed`).
- Consumes: the re-keyed `Provenance.link` (Task 3).

**Details:** `ingest_article` already re-ingests changed chunks (their `content_hash` differs → not `already_ingested` → new episode + re-keyed link supersedes the old). Add: (a) `tombstone_article_episodes` for the `remove` op; (b) after ingesting, supersede any `HAS_EPISODE` whose `chunk_index` is beyond the new episode count (an article that shrank leaves trailing stale episodes).

- [ ] **Step 1: Write failing integration tests.**

```python
async def test_tombstone_marks_removed_not_deleted(extract_driver, ingest_driver):
    # seed an Article with 2 HAS_EPISODE episodes, then:
    n = await ingest_driver.tombstone_article_episodes("a1")
    # assert both Episodic still exist and are removed=true, n == 2

async def test_shrink_supersedes_trailing(extract_driver, ingest_driver):
    # seed an Article with episodes at chunk_index 0,1,2; call the trailing-
    # supersede helper for new_count=1; assert chunk 1,2 episodes superseded,
    # chunk 0 untouched.
```

(Build `ingest_driver` from the existing `IngestDriver`; for the tombstone test you can seed the graph directly and only exercise the Cypher, so no live LLM is needed — keep these tests LLM-free by seeding episodes/edges by hand.)

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement.** Add to `IngestDriver`:

```python
async def tombstone_article_episodes(self, article_id: str) -> int:
    async with self._driver.session() as s:
        r = await s.run(
            "MATCH (:Article {id:$a})-[:HAS_EPISODE]->(e:Episodic) "
            "SET e.removed=true RETURN count(e) AS c", a=article_id)
        return (await r.single())["c"]

async def _supersede_trailing_episodes(self, article_id: str, new_count: int) -> None:
    async with self._driver.session() as s:
        await s.run(
            "MATCH (:Article {id:$a})-[r:HAS_EPISODE]->(e:Episodic) "
            "WHERE r.chunk_index >= $n SET e.superseded=true", a=article_id, n=new_count)
```

Call `_supersede_trailing_episodes(art.id, len(episodes))` at the end of `ingest_article` (after the loop), so a shortened article's trailing episodes are superseded.

- [ ] **Step 4: Run tests green.** Gate on `src/graph_extract/ingest_driver.py`.

- [ ] **Step 5: Commit.** `git add src/graph_extract/ingest_driver.py tests/integration/test_temporal_update.py && git commit -m "feat(extract): temporal update (tombstone episodes, supersede trailing on shrink)"`

---

### Task 5: Standalone worker + end-to-end wiring

**Files:**
- Create: `src/graph_sync/semantic_worker.py`
- Modify: `src/graph_sync/cli.py` (add `worker` command)
- Test: `tests/integration/test_semantic_worker.py`

**Interfaces:**
- Consumes: `StateStore.claim/complete/fail_semantic_job` (Task 1); `IngestDriver.ingest_article` + `tombstone_article_episodes` (Task 4).
- Produces: `async run_worker_once(store, ingest, batch: int) -> int` (returns processed count); `async run_worker(store, ingest, batch, poll_seconds, stop_event)`.

**Details:** `run_worker_once` claims a batch, dispatches each job (`upsert`→`ingest_article`, `remove`→`tombstone_article_episodes`), and marks `complete`/`fail` per job. `run_worker` loops with a poll sleep until `stop_event`. The `worker` CLI command builds a `StateStore` and an `IngestDriver` (reuse `graph_extract.cli._build_ingest_driver` or replicate its construction), then runs `run_worker`, closing resources in `finally`.

- [ ] **Step 1: Write failing integration test** (Postgres + Neo4j testcontainers). Use a **stub ingest** object exposing `ingest_article`/`tombstone_article_episodes` that records calls (so the worker test is LLM-free and fast); the real `IngestDriver` is exercised by Task 4.

```python
async def test_worker_processes_upsert_and_remove(store):
    class StubIngest:
        def __init__(self): self.calls = []
        async def ingest_article(self, aid): self.calls.append(("upsert", aid))
        async def tombstone_article_episodes(self, aid): self.calls.append(("remove", aid)); return 0
    await store.enqueue_semantic_job("a1", "upsert", "h")
    await store.enqueue_semantic_job("a2", "remove", None)
    stub = StubIngest()
    from graph_sync.semantic_worker import run_worker_once
    n = await run_worker_once(store, stub, batch=10)
    assert n == 2 and ("upsert", "a1") in stub.calls and ("remove", "a2") in stub.calls
    assert await store.claim_semantic_jobs(10) == []           # both completed

async def test_worker_marks_failed_and_continues(store):
    class Boom:
        async def ingest_article(self, aid): raise RuntimeError("x")
        async def tombstone_article_episodes(self, aid): return 0
    await store.enqueue_semantic_job("a1", "upsert", "h")
    from graph_sync.semantic_worker import run_worker_once
    await run_worker_once(store, Boom(), batch=10)
    # job is 'failed', not re-claimable, no exception bubbled
    assert await store.claim_semantic_jobs(10) == []
```

- [ ] **Step 2: Run, confirm fail** (module missing).

- [ ] **Step 3: Implement `semantic_worker.py`:**

```python
from __future__ import annotations
import asyncio, logging
logger = logging.getLogger(__name__)

async def run_worker_once(store, ingest, batch: int) -> int:
    jobs = await store.claim_semantic_jobs(batch)
    for job in jobs:
        try:
            if job["op"] == "upsert":
                await ingest.ingest_article(job["article_id"])
            elif job["op"] == "remove":
                await ingest.tombstone_article_episodes(job["article_id"])
            else:
                raise ValueError(f"unknown op {job['op']!r}")
            await store.complete_semantic_job(job["id"])
        except Exception as e:  # a poison job must not block the queue
            logger.exception("semantic job %s failed", job["id"])
            await store.fail_semantic_job(job["id"], str(e))
    return len(jobs)

async def run_worker(store, ingest, *, batch: int, poll_seconds: float,
                     stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        n = await run_worker_once(store, ingest, batch)
        if n == 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
            except asyncio.TimeoutError:
                pass
```

- [ ] **Step 4: Add the `worker` CLI command** in `graph_sync/cli.py`, mirroring `_build_sync_core`'s construct-then-`finally`-close pattern: build `StateStore` (init schema) + an `IngestDriver` (via `graph_extract.cli._build_ingest_driver` or equivalent), create a `stop_event`, install a SIGINT handler that sets it, run `run_worker`, close resources in `finally`.

- [ ] **Step 5: Run tests green** + full non-live suite + a CLI `--help` shows `worker`. Gate on the two files.

- [ ] **Step 6: Commit.** `git add src/graph_sync/semantic_worker.py src/graph_sync/cli.py tests/integration/test_semantic_worker.py && git commit -m "feat(sync): standalone semantic-ingestion worker + CLI"`

---

## Self-Review Notes

- **Spec coverage:** queue → T1; trigger → T2; provenance atomicity/re-key → T3; temporal policy (updated via T3 re-key + T4 shrink-supersede; removed via T4) → T3+T4; worker + e2e → T5. All §-sections covered.
- **Layering:** only `graph_sync/semantic_worker.py` + `cli.py` import `graph_extract`; `graph_extract` imports nothing from `graph_sync`. (Reviewer: verify no reverse import crept in.)
- **Type consistency:** queue methods (`enqueue/claim/complete/fail_semantic_job`), job dict keys (`id, article_id, op, content_hash, attempts`), `tombstone_article_episodes`, `run_worker_once/run_worker` used identically across tasks.
- **Ordering/deps:** T1 → T2; T3 → T4; {T1,T4} → T5. T3 independent of T1/T2. Suggested execution order: T1, T3, T2, T4, T5.
- **No-placeholder check:** every code step carries the actual code; tests are concrete.
