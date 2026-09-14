# Concurrent Article Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Process articles concurrently during ingest — never episodes within an article — behind a limit that defaults to today's sequential behaviour.

**Architecture:** One shared `run_concurrently` helper bounded by an `asyncio.Semaphore`, returning results in input order with exceptions in their slots. `IngestDriver.ingest_source` fans out over article ids. `run_worker_once` groups claimed jobs by `article_id` first and fans out over the *groups*: the pending-unique index means a batch holds at most one job per article today, and the grouping is defence in depth that keeps an upsert and a later remove for one article ordered if that index ever goes.

**Tech Stack:** Python 3.12, `uv`, asyncio, pydantic-settings, graphiti-core 0.30.1, pytest + pytest-asyncio.

## Global Constraints

- Concurrency is **across articles only**. Episodes within an article stay sequential.
- `ingest_article_concurrency: int = 1` — default is byte-for-byte today's call order. Values below 1 are **rejected at config validation**, not silently coerced.
- `run_worker_once` must group by `article_id` before fanning out. `ux_semantic_jobs_pending` (`state_store.py:28-29`) is UNIQUE on `(article_id) WHERE status='pending'`, `enqueue_semantic_job` collapses an upsert-then-remove into one row (`:129-136`) and `claim_semantic_jobs` selects only `status='pending'` (`:138-149`), so one claim batch holds at most one job per article today. The grouping is defence in depth that does not depend on that index staying.
- One failing article must not cancel its siblings; the failure must still surface.
- **Known behaviour change, applies even at limit=1:** today a raising article propagates out of `ingest_source`'s loop and every article after it is silently never attempted. After this change every article is attempted (`gather` creates all the tasks up front, whatever the limit) and the failure is raised at the end.
- Do not swallow exceptions. Do not change retry/backoff logic.
- CI gate, all three clean: `uv run ruff check src tests` (lints tests too: E702 no semicolons, E402 imports at top), `uv run mypy src`, `uv run --extra dev pytest -m "not live"`.
- **Never run `python -m graph_extract.cli ingest`, `theme-build`, or `answer_api.eval_router`** — all three spend real money. The paid A/B in the spec's §8 is the user's decision, not this plan's.
- The integration suite takes ~11 minutes; the full gate must be run as ONE process (`pytest -m "not live"`), because running unit and integration separately has hidden a leak three times.
- Current suite baseline: **848 passed, 15 deselected**.

---

### Task 1: The `run_concurrently` helper

**Files:**
- Create: `src/graph_extract/concurrent_ingest.py`
- Test: `tests/unit/test_concurrent_ingest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `async def run_concurrently(items: Sequence[T], worker: Callable[[T], Awaitable[R]], *, limit: int) -> list[R | BaseException]` — one result per item **in input order**; a worker exception is returned in its slot rather than raised.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_concurrent_ingest.py`:

```python
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


async def test_work_actually_overlaps():
    """A slow first item must not block a later one -- the whole point."""
    started: list[int] = []

    async def work(n: int) -> int:
        started.append(n)
        await asyncio.sleep(0.05 if n == 0 else 0)
        return n

    await run_concurrently([0, 1, 2], work, limit=3)
    assert started == [0, 1, 2], "all three dispatched before the slow one finished"


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


async def test_a_limit_below_one_is_rejected():
    async def work(n: int) -> int:
        return n

    with pytest.raises(ValueError, match="limit"):
        await run_concurrently([1], work, limit=0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_concurrent_ingest.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'graph_extract.concurrent_ingest'`

- [ ] **Step 3: Write the implementation**

Create `src/graph_extract/concurrent_ingest.py`:

```python
"""Bounded concurrent fan-out for ingest.

Measured on the 2026-09-14 pilot re-ingest: the sum of LLM in-call time was
25,211 s against 24,933 s wall -- a ratio of 1.01, meaning effectively no overlap
across the whole run -- while four phases dispatched strictly one at a time
accounted for 55.9% of all LLM time. The provider does not serialise us: dedup
latency is flat across in-flight buckets (1231 / 1215 / 1287 / 1469 / 1082 ms from
1 to 20+ concurrent), so the headroom is real rather than assumed.

Used to fan out over ARTICLES only. Episodes within an article are consecutive
chunks of one document and share entities most heavily, so they stay sequential:
graphiti resolves entities by searching the graph as it currently stands, so two
in-flight episodes extracting a not-yet-present entity would each create it under
a different uuid. See the design spec for why partitioning is ruled out by design
invariant #4.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


async def run_concurrently(
    items: Sequence[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    limit: int,
) -> list[R | BaseException]:
    """Run `worker` over `items`, at most `limit` at a time.

    Returns one entry per item **in input order**. A worker exception is returned
    in that item's slot rather than raised, so one bad article cannot cancel its
    siblings -- the caller decides what a failure means. Nothing is swallowed: the
    exception object is handed back intact.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    semaphore = asyncio.Semaphore(limit)

    async def _one(item: T) -> R:
        async with semaphore:
            return await worker(item)

    return list(await asyncio.gather(
        *(_one(item) for item in items), return_exceptions=True))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_concurrent_ingest.py -q`
Expected: `7 passed`

- [ ] **Step 5: Prove the tests discriminate**

Apply each mutation alone, run the file, restore with `git checkout src/graph_extract/concurrent_ingest.py`, and record which tests died:

| # | mutation | must kill |
|---|---|---|
| M1 | drop `return_exceptions=True` | the failing-sibling test |
| M2 | replace the semaphore body with a bare `await worker(item)` | the in-flight-bound test |
| M3 | `if limit < 1:` → `if False:` | the limit-below-one test |
| M4 | return `sorted(...)` of results instead of gather order | the input-order test |

Every mutation must kill at least one test, and every test must die under at least one mutation. If a mutation kills nothing, add the discriminating case before continuing and say so.

After the last restore, confirm `git diff --stat src/` is empty.

- [ ] **Step 6: Commit**

```bash
git add src/graph_extract/concurrent_ingest.py tests/unit/test_concurrent_ingest.py
git commit -m "feat(ingest): bounded concurrent fan-out helper

Returns results in input order with a worker's exception in its own slot, so
one failing article cannot cancel its siblings. Semaphore-bounded so a large
batch cannot open unbounded connections."
```

---

### Task 2: Config flag and `ingest_source` fan-out

**Files:**
- Modify: `src/graph_extract/config.py` (add the setting and the file's first validator)
- Modify: `src/graph_extract/ingest_driver.py` (`ingest_source`)
- Test: `tests/unit/test_ingest_source_concurrency.py`

**Interfaces:**
- Consumes: `run_concurrently(items, worker, *, limit) -> list[R | BaseException]` from Task 1.
- Produces: `ExtractSettings.ingest_article_concurrency: int = 1`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_ingest_source_concurrency.py`:

```python
"""`ingest_source` fans out over ARTICLES. Episodes stay sequential inside
`ingest_article`, because consecutive chunks of one document share entities most
heavily and graphiti resolves entities against the graph as it currently stands.
"""
from __future__ import annotations

import asyncio

import pytest

from graph_extract.config import ExtractSettings
from graph_extract.ingest_driver import IngestArticleResult, IngestDriver

pytestmark = pytest.mark.asyncio


def _settings(**kw) -> ExtractSettings:
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    base.update(kw)
    return ExtractSettings(**base)


def _driver(settings, results):
    drv = IngestDriver.__new__(IngestDriver)
    drv._s = settings
    drv._article_ids = list(results)
    drv._results = results

    async def _list(source_id, *a, **k):
        return list(results)

    async def _ingest(aid):
        return await results[aid]()

    drv.list_article_ids = _list
    drv.ingest_article = _ingest
    return drv


def test_the_default_is_sequential():
    assert _settings().ingest_article_concurrency == 1


def test_a_concurrency_below_one_is_rejected():
    with pytest.raises(ValueError):
        _settings(ingest_article_concurrency=0)


async def test_articles_overlap_when_concurrency_is_raised():
    started: list[str] = []

    def make(aid, delay):
        async def _work():
            started.append(aid)
            await asyncio.sleep(delay)
            return IngestArticleResult(article_id=aid, episodes_added=1)
        return _work

    results = {"a": make("a", 0.05), "b": make("b", 0.0), "c": make("c", 0.0)}
    drv = _driver(_settings(ingest_article_concurrency=3), results)
    out = await drv.ingest_source("s")
    assert started == ["a", "b", "c"], "all dispatched before the slow one finished"
    assert out.articles == 3 and out.episodes_added == 3


async def test_a_failing_article_does_not_prevent_its_siblings(caplog):
    """Behaviour CHANGE, and it applies at any concurrency: today a raising
    article propagates out of the loop and every article after it is silently
    never attempted."""
    done: list[str] = []

    def ok(aid):
        async def _work():
            done.append(aid)
            return IngestArticleResult(article_id=aid, episodes_added=1)
        return _work

    async def boom():
        raise ValueError("boom")

    results = {"a": ok("a"), "b": boom, "c": ok("c")}
    drv = _driver(_settings(ingest_article_concurrency=3), results)
    with pytest.raises(ValueError, match="boom"):
        await drv.ingest_source("s")
    assert done == ["a", "c"], "siblings must still have run"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_ingest_source_concurrency.py -q`
Expected: FAIL — `AttributeError: 'ExtractSettings' object has no attribute 'ingest_article_concurrency'`

- [ ] **Step 3: Add the setting and its validator**

In `src/graph_extract/config.py`, add to the imports at the top of the file:

```python
from pydantic import field_validator
```

and add the setting immediately after `valid_at_from_content_changed`:

```python
    # How many ARTICLES to ingest concurrently. Episodes within an article always
    # stay sequential: consecutive chunks of one document share entities most
    # heavily, and graphiti resolves entities by searching the graph as it
    # currently stands, so two in-flight episodes extracting a not-yet-present
    # entity would each create it under a different uuid.
    #
    # DEFAULT 1 -- byte-for-byte today's call order. Raising it is gated on the
    # A/B in the design spec's section 8: re-ingest the same 83 pilot articles and
    # compare entity count against the sequential baseline of 999.
    ingest_article_concurrency: int = 1
```

and add this validator at the end of the class body:

```python
    @field_validator("ingest_article_concurrency")
    @classmethod
    def _at_least_one(cls, v: int) -> int:
        # Rejected, not clamped: a 0 in the environment means someone believes
        # they configured something, and silently running sequentially would hide
        # that from them.
        if v < 1:
            raise ValueError(f"ingest_article_concurrency must be >= 1, got {v}")
        return v
```

- [ ] **Step 4: Fan out in `ingest_source`**

In `src/graph_extract/ingest_driver.py`, add to the imports:

```python
from graph_extract.concurrent_ingest import run_concurrently
```

and replace the body of `ingest_source` with:

```python
    async def ingest_source(self, source_id: str, limit: int | None = None) -> IngestResult:
        ids = await self.list_article_ids(source_id)
        if limit is not None:
            ids = ids[:limit]
        out = IngestResult()
        results = await run_concurrently(
            ids, self.ingest_article, limit=self._s.ingest_article_concurrency)
        failures: list[BaseException] = []
        for r in results:
            if isinstance(r, BaseException):
                failures.append(r)
                continue
            out.articles += 1
            out.episodes_added += r.episodes_added
            out.episodes_skipped += r.episodes_skipped
            out.dedup.merge(r.dedup)
        if failures:
            # Raised AFTER the batch rather than mid-loop. Today's code propagates
            # immediately and silently abandons every article after the failure;
            # this completes the ones already dispatched and still surfaces the
            # error. Deliberate change, pinned by a test.
            raise failures[0]
        return out
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_ingest_source_concurrency.py -q`
Expected: `4 passed`

- [ ] **Step 6: Run the unit suite and the linters**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest tests/unit -q`
Expected: `All checks passed!`, `Success: no issues found`, and the unit suite green. **Report the observed count; do not adjust an assertion to match a predicted number.** Four stale counts have already been caught this way in this project.

- [ ] **Step 7: Commit**

```bash
git add src/graph_extract/config.py src/graph_extract/ingest_driver.py tests/unit/test_ingest_source_concurrency.py
git commit -m "feat(ingest): fan ingest_source out over articles

Default concurrency 1, so call order is unchanged. Values below 1 are rejected
rather than clamped. A failing article no longer abandons the articles after it."
```

---

### Task 3: Worker fan-out, grouped by article

**Files:**
- Modify: `src/graph_sync/semantic_worker.py` (`run_worker_once`)
- Test: `tests/unit/test_worker_concurrency.py`

**Interfaces:**
- Consumes: `run_concurrently(items, worker, *, limit)` (Task 1), `ExtractSettings.ingest_article_concurrency` (Task 2).
- Produces: `run_worker_once(..., concurrency: int = 1)` — a new keyword-only parameter, defaulting to today's behaviour.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_worker_concurrency.py`:

```python
"""The worker fans out over ARTICLE GROUPS, never raw jobs.

Today the queue cannot produce two jobs for one article in a batch: the
`ux_semantic_jobs_pending` index is UNIQUE on (article_id) WHERE status='pending'
(state_store.py:28-29), `enqueue_semantic_job` collapses an upsert-then-remove into
one row (:129-136), and `claim_semantic_jobs` selects only status='pending'
(:138-149). The grouping is defence in depth that does not depend on that index
staying: without it, fanning out over raw jobs could apply an upsert and a later
remove out of order -- tombstoning episodes that the upsert then recreates, or the
reverse.
"""
from __future__ import annotations

import asyncio

import pytest

from graph_extract.ingest_driver import IngestArticleResult
from graph_extract.llm_timing import PromptTimings
from graph_sync.semantic_worker import run_worker_once

pytestmark = pytest.mark.asyncio


class _Store:
    def __init__(self, jobs):
        self._jobs = jobs
        self.completed: list[str] = []
        self.failed: list[str] = []

    async def reap_stale_jobs(self, *a, **k): return None
    async def today_token_total(self): return 0
    async def claim_semantic_jobs(self, batch, include_bootstrap): return self._jobs
    async def complete_semantic_job(self, jid, claimed_at): self.completed.append(jid)
    async def fail_semantic_job(self, jid, *a, **k): self.failed.append(jid)
    async def record_tokens(self, n): return None


def _job(jid, article, op="upsert"):
    return {"id": jid, "op": op, "article_id": article,
            "claimed_at": "t", "attempts": 0}


class _Ingest:
    def __init__(self, log, delays=None):
        self.log = log
        self.delays = delays or {}
        self.timings = PromptTimings()

    async def ingest_article(self, article_id):
        self.log.append(("upsert", article_id))
        await asyncio.sleep(self.delays.get(article_id, 0))
        return IngestArticleResult(article_id=article_id)

    async def tombstone_article_episodes(self, article_id):
        self.log.append(("remove", article_id))
        return 1


_KW = dict(batch=10, budget=10**9, max_attempts=3, backoff_base=1.0,
           backoff_cap=60.0, lease=300.0)


async def test_two_jobs_for_one_article_stay_in_order():
    log: list[tuple[str, str]] = []
    store = _Store([_job("j1", "a1", "upsert"), _job("j2", "a1", "remove")])
    await run_worker_once(store, _Ingest(log), concurrency=4, **_KW)
    assert log == [("upsert", "a1"), ("remove", "a1")]
    assert store.completed == ["j1", "j2"]


async def test_different_articles_overlap():
    log: list[tuple[str, str]] = []
    store = _Store([_job("j1", "slow"), _job("j2", "fast")])
    ingest = _Ingest(log, delays={"slow": 0.05})
    await run_worker_once(store, ingest, concurrency=4, **_KW)
    assert log[0] == ("upsert", "slow") and log[1] == ("upsert", "fast"), (
        "the fast article must start before the slow one finishes")


async def test_default_concurrency_is_one():
    log: list[tuple[str, str]] = []
    store = _Store([_job("j1", "slow"), _job("j2", "fast")])
    ingest = _Ingest(log, delays={"slow": 0.03})
    await run_worker_once(store, ingest, **_KW)
    assert store.completed == ["j1", "j2"]


async def test_one_failing_group_does_not_stop_another():
    log: list[tuple[str, str]] = []

    class _Boom(_Ingest):
        async def ingest_article(self, article_id):
            if article_id == "bad":
                raise ValueError("boom")
            return await super().ingest_article(article_id)

    store = _Store([_job("j1", "bad"), _job("j2", "good")])
    await run_worker_once(store, _Boom(log), concurrency=4, **_KW)
    assert store.failed == ["j1"]
    assert store.completed == ["j2"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_worker_concurrency.py -q`
Expected: FAIL — `TypeError: run_worker_once() got an unexpected keyword argument 'concurrency'`

- [ ] **Step 3: Extract the per-job body and fan out over groups**

In `src/graph_sync/semantic_worker.py`, add to the imports:

```python
from graph_extract.concurrent_ingest import run_concurrently
```

change the signature to add the parameter:

```python
async def run_worker_once(
    store, ingest, *, batch: int, budget: int, max_attempts: int,
    backoff_base: float, backoff_cap: float, lease: float,
    concurrency: int = 1,
) -> int:
```

and replace the `for job in jobs:` loop with a per-job coroutine plus a grouped fan-out, keeping the body identical:

```python
    async def _run_job(job) -> None:
        before = get_tally()
        t0 = before.prompt_tokens + before.completion_tokens
        try:
            if job["op"] == "upsert":
                res = await ingest.ingest_article(job["article_id"])
                # Observability must never fail the work it observes. A driver
                # double, or a future refactor returning None, must not raise in
                # here -- that would fail the job and poison the queue for the
                # sake of a counter. Explicit check, not a blanket try/except.
                if res is not None:
                    batch_dedup.merge(res.dedup)
                    basis_counts[res.reference_basis] += 1
            elif job["op"] == "remove":
                await ingest.tombstone_article_episodes(job["article_id"])
            else:
                raise ValueError(f"unknown op {job['op']!r}")
            await store.complete_semantic_job(job["id"], job["claimed_at"])
        except Exception as e:  # a poison job must not block the queue
            logger.exception("semantic job %s failed", job["id"])
            await store.fail_semantic_job(
                job["id"], str(e), max_attempts=max_attempts,
                retry_delay_seconds=exp_backoff(
                    job["attempts"] + 1, base=backoff_base, cap=backoff_cap),
                claimed_at=job["claimed_at"])
        finally:
            after = get_tally()
            delta = (after.prompt_tokens + after.completion_tokens) - t0
            if delta:
                await store.record_tokens(delta)

    async def _run_group(group) -> None:
        for job in group:
            await _run_job(job)

    # Grouped by article_id, NOT fanned out over raw jobs. Today the queue cannot
    # hand us two jobs for one article: `ux_semantic_jobs_pending` is UNIQUE on
    # (article_id) WHERE status='pending' (state_store.py:28-29), enqueue
    # collapses an upsert-then-remove into one row via ON CONFLICT ... SET
    # op=excluded.op (:129-136), and claim_semantic_jobs selects only
    # status='pending' (:138-149). The grouping is defence in depth that does not
    # depend on that index staying: if it ever went, running an upsert and a
    # later remove for the same article out of order would tombstone episodes
    # the upsert just created.
    groups: dict[str, list] = {}
    for job in jobs:
        groups.setdefault(job["article_id"], []).append(job)
    await run_concurrently(list(groups.values()), _run_group, limit=concurrency)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_worker_concurrency.py -q`
Expected: `4 passed`

- [ ] **Step 5: Prove the grouping discriminates**

Replace the grouped fan-out with a raw-job fan-out:

```python
    await run_concurrently(jobs, _run_job, limit=concurrency)
```

Run: `uv run --extra dev pytest tests/unit/test_worker_concurrency.py -q`
Expected: `test_two_jobs_for_one_article_stay_in_order` FAILS.

Restore the grouped version and confirm the file passes again, then confirm `git diff --stat src/` shows only your intended change.

- [ ] **Step 6: Run the FULL gate as one process**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live" -q`

This takes ~11 minutes and the Bash tool auto-backgrounds past 120s. Launch it detached to a log and poll the log — note `( cmd > log 2>&1; echo EXIT=$? >> log ) &` backgrounds the whole subshell, whereas `cmd > log 2>&1; echo ... &` backgrounds only the echo and the command dies at the tool timeout.

**Run it as ONE process.** Running unit and integration separately has hidden a process-wide patch leak three times in this project.

Expected: green. **Report the observed count.**

- [ ] **Step 7: Commit**

```bash
git add src/graph_sync/semantic_worker.py tests/unit/test_worker_concurrency.py
git commit -m "feat(worker): fan out over article groups, not raw jobs

The pending-unique index means a batch holds at most one job per article today;
grouping is defence in depth that keeps an upsert and a later remove for one
article ordered if that index ever goes, while different articles overlap.
concurrency defaults to 1."
```

---

## Verification

Against the spec's §9 success criteria:

1. **`limit=1` matches today in call order and results** — Task 1's sequential test and Task 3's default test. The one intended difference is failure handling, pinned by Task 2.
2. **A batch with two jobs for one article applies them in order** — Task 3, proven discriminating in Step 5.
3. **One failing article does not prevent siblings completing** — Task 1 and Task 2.
4. **The A/B reports entity count against 999 and wall clock against 6h55m** — **not in this plan.** It is a paid run and the user's decision.
5. **Per-article dedup attribution stays correct under concurrency** — `CURRENT_DEDUP_STATS` is a `ContextVar` and `asyncio.create_task` copies context at creation, and it is observable in the worker's batch log. Pinned by `test_concurrent_articles_keep_their_own_dedup_attribution` in `tests/unit/test_ingest_source_concurrency.py`, which runs the real `ingest_article` scope with two overlapping articles recording distinct counts through the guard's own `stats()` resolution: the claim is about our scope, not about asyncio, and a shared holder in place of the ContextVar fails it.

## Notes for the implementer

- **Do not add concurrency within an article.** Episodes of one document share entities most heavily and that is precisely the duplicate-entity hazard.
- **Do not raise the default above 1.** Adoption is gated on a paid A/B that has not run.
- If an expected test count in this plan does not match what you observe, **report the discrepancy** rather than adjusting an assertion — four stale counts have been caught that way here already.
