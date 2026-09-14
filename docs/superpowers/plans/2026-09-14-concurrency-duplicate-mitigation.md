# Concurrency Duplicate Mitigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ingest_article_concurrency > 1` safe to enable, by preventing most duplicate entities (a sequential warm-up over each source's opening articles) and deterministically repairing the rest (an exact-name merge pass).

**Architecture:** Two independent components. Warm-up adds an optional `is_cold` predicate to the existing `run_concurrently` helper and a graph-derived `WarmupGate` that both ingest call sites consult. The merge pass is a new report-first CLI command that finds `:Entity` nodes sharing a byte-identical `name` within one `group_id` and rewires every edge by hand in plain Cypher, one transaction per name group.

**Tech Stack:** Python 3.12, `uv`, asyncio, `graphiti-core` 0.30.1 (pinned, never forked), Neo4j 2026.07.1 Community (`CYPHER_25`, **no APOC**), typer, pytest + testcontainers.

**Spec:** `docs/superpowers/specs/2026-09-14-concurrency-duplicate-mitigation-design.md`
**Evidence:** `docs/superpowers/ab-concurrency-2026-09-14.md`

## Global Constraints

- **graphiti-core is pinned at 0.30.1 and is never forked.** Extend it by patching imported-by-name module references, the way `lean_edge_search`, `dedup_guard`, `contradiction_gate` and `deterministic_valid_at` do.
- **No APOC.** `apoc.refactor.mergeNodes` does not exist. All rewiring is plain Cypher. `db.create.setNodeVectorProperty` and `db.create.setRelationshipVectorProperty` ARE available and must be used for vector properties.
- **Design invariant #3:** updates append, never overwrite. No episode, fact, `HAS_EPISODE` or `MENTIONS` may be deleted or altered; they are re-created with their full property map **and their original uuid**.
- **Design invariant #4:** one corpus-wide `group_id`. Every merge `MATCH` carries `group_id` on both nodes. The merge never crosses groups.
- **Design invariant #5:** graphiti's schema is library-owned. Adding properties (`merged_from`, `merged_at`, `stale`) is allowed; renaming or restructuring its elements is not.
- **`ingest_article_concurrency` stays at its default of 1.** Nothing in this plan changes it. Raising it is the user's decision.
- **`ingest_warmup_articles` defaults to 8**, `0` is off, a negative value **raises** at settings construction (never clamped). It lives on `ExtractSettings` in `src/graph_extract/config.py` — NOT on `graph_sync.config.Settings`, which is a different, narrower object.
- **The merge is destructive and writes nothing without `--apply`.** Report mode is the default.
- **`--apply` must never run alongside ingest or the semantic worker.** graphiti saves edges with `MATCH (source:Entity {uuid}) ... MERGE (...)`; if an in-flight episode resolved an entity to a deleted loser, the edge is silently not written — no error, no log, no retry.
- **Every test must fail with its fix neutralised, proven by mutation.** Record mutations in `.superpowers/sdd/<task>-mutations.json` (mutant name, passed, failed, failing test ids). Five vacuous tests were caught during the concurrency work; a test that cannot fail is a defect. Assertions on *dispatch or start order alone* are the specific trap — a sequential implementation satisfies them identically. Assert on state captured at the moment a slow item **finishes**.
- **The CI gate must be run as ONE process:** `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live"`. Running unit and integration separately has hidden a real failure three times. Ruff lints tests too, with E7 on: no semicolons in test fakes, imports at file top.
- **Never run `python -m graph_extract.cli ingest`, `theme-build`, or `answer_api.eval_router`.** All three spend real money against a live corpus.
- **The live baseline graph (group `backup-docs`: 999 entities, 655 episodes, 655 `HAS_EPISODE`, 0 superseded) is the reference for every future comparison. Read-only Cypher against it is fine. Never write to it, and never run `--apply` against it.**

## File Structure

| file | responsibility |
|---|---|
| `src/graph_extract/concurrent_ingest.py` (modify) | add optional `is_cold` to `run_concurrently`; the strict barrier lives here so both call sites share it |
| `src/graph_extract/warmup.py` (create) | `WarmupGate` — the graph-derived cold/warm predicate and its per-run cache |
| `src/graph_extract/config.py` (modify) | `ingest_warmup_articles` + validator |
| `src/graph_extract/ingest_driver.py` (modify) | `ingest_source` passes the predicate; driver carries `warmup_gate` |
| `src/graph_sync/semantic_worker.py` (modify) | `run_worker_once`/`run_worker` accept and forward `is_cold` |
| `src/graph_sync/cli.py` (modify) | build the gate, pass the predicate |
| `src/graph_extract/merge_duplicates.py` (create) | the whole merge pass: planner (read-only) + `merge_group` (one transaction) |
| `src/graph_extract/cli.py` (modify) | `merge-duplicates` command; duplicate count in `ingest`; report in `cleanup`/`maintenance` |
| `src/theme_builder/cli.py` (modify) | refuse to run over known duplicates unless `--allow-duplicates` |
| `scripts/risk_curve.py` (create) | read-only recompute of the §1.1 risk curve, so `W` can be re-derived without another paid A/B |
| `docs/superpowers/maintenance-runbook.md` (modify) | the new entry, and the amendment to the "safe alongside ingestion" sentence |

Tasks 1–3 are warm-up and 4–8 are the merge; the two groups are independent and either could ship first. Within each group the order is a hard dependency chain.

---

### Task 1: `run_concurrently` learns a cold path

**Files:**
- Modify: `src/graph_extract/concurrent_ingest.py`
- Test: `tests/unit/test_concurrent_ingest.py` (exists — add to it)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `run_concurrently(items, worker, *, limit, is_cold: Callable[[T], Awaitable[bool]] | None = None) -> list[R | BaseException]`. Tasks 3 and 5 rely on this exact signature.

The semantics, precisely: iterate items in input order. If `is_cold` is given and `await is_cold(item)` is true, **first await every task launched so far**, then `await worker(item)` directly (nothing else in flight), then carry on. Otherwise launch `worker(item)` as a task under the semaphore. Results come back one per item in input order. An exception from `worker` **or from `is_cold`** goes in that item's slot.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_concurrent_ingest.py`:

```python
async def test_is_cold_none_is_byte_for_byte_the_old_helper():
    """The default path must not change. This is what lets the warm-up knob
    ship defaulted ON: at concurrency 1 it provably changes nothing."""
    log: list[str] = []

    async def worker(i: int) -> int:
        log.append(f"start-{i}")
        await asyncio.sleep(0.03 if i == 0 else 0)
        log.append(f"end-{i}")
        return i * 10

    res = await run_concurrently([0, 1, 2], worker, limit=3)
    assert res == [0, 10, 20]
    # the slow first item must still be overlapped by the others
    assert log.index("start-1") < log.index("end-0")


async def test_a_cold_item_runs_with_nothing_else_in_flight():
    """The strict barrier. A slow WARM item launched before the cold one must
    have FINISHED before the cold item starts -- start order alone is satisfied
    by a sequential implementation and proves nothing."""
    log: list[str] = []

    async def worker(name: str) -> str:
        log.append(f"start-{name}")
        await asyncio.sleep(0.05 if name == "slow-warm" else 0.01)
        log.append(f"end-{name}")
        return name

    async def is_cold(name: str) -> bool:
        return name == "cold"

    res = await run_concurrently(
        ["slow-warm", "cold", "after"], worker, limit=4, is_cold=is_cold)
    assert res == ["slow-warm", "cold", "after"]
    assert log.index("end-slow-warm") < log.index("start-cold"), log
    assert log.index("end-cold") < log.index("start-after"), log


async def test_is_cold_raising_lands_in_that_items_slot():
    async def worker(i: int) -> int:
        return i

    async def is_cold(i: int) -> bool:
        if i == 1:
            raise RuntimeError("neo4j down")
        return False

    res = await run_concurrently([0, 1, 2], worker, limit=3, is_cold=is_cold)
    assert res[0] == 0
    assert isinstance(res[1], RuntimeError)
    assert res[2] == 2


async def test_results_stay_in_input_order_with_a_cold_item_in_the_middle():
    async def worker(i: int) -> int:
        await asyncio.sleep(0.02 if i == 2 else 0)
        return i * 10

    async def is_cold(i: int) -> bool:
        return i == 1

    assert await run_concurrently(
        [0, 1, 2, 3], worker, limit=4, is_cold=is_cold) == [0, 10, 20, 30]
```

- [ ] **Step 2: Run them to verify they fail**

```
uv run --extra dev pytest tests/unit/test_concurrent_ingest.py -q
```
Expected: the three `is_cold` tests FAIL with `TypeError: run_concurrently() got an unexpected keyword argument 'is_cold'`. `test_is_cold_none_is_byte_for_byte_the_old_helper` should PASS already — it pins existing behaviour.

- [ ] **Step 3: Implement**

Replace the body of `run_concurrently` in `src/graph_extract/concurrent_ingest.py`:

```python
async def run_concurrently(
    items: Sequence[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    limit: int,
    is_cold: Callable[[T], Awaitable[bool]] | None = None,
) -> list[R | BaseException]:
    """Run `worker` over `items`, at most `limit` at a time.

    Returns one entry per item **in input order**. A worker exception is returned
    in that item's slot rather than raised, so one bad article cannot cancel its
    siblings -- the caller decides what a failure means. Nothing is swallowed: the
    exception object is handed back intact.

    `is_cold` opts an item into the warm-up barrier: the item waits for every
    task launched so far to FINISH, then runs alone. That is stricter than
    "cold items are sequential among themselves", deliberately -- the hazard is
    two in-flight episodes each creating an entity neither can see yet, and a
    cold article of one source running beside a warm article of another is
    exactly the cross-source hub case invariant #4 cares about. An exception
    from `is_cold` is treated like a worker exception: it goes in the item's
    slot and the siblings continue.

    With `is_cold=None` this is byte-for-byte the pre-warm-up helper, which is
    what makes shipping the warm-up default ON a no-op at concurrency 1.
    """
    if limit < 1:
        raise ValueError(f"limit must be >= 1, got {limit}")
    semaphore = asyncio.Semaphore(limit)

    async def _one(item: T) -> R:
        async with semaphore:
            return await worker(item)

    if is_cold is None:
        return list(await asyncio.gather(
            *(_one(item) for item in items), return_exceptions=True))

    results: list[R | BaseException] = [None] * len(items)  # type: ignore[list-item]
    launched: list[tuple[int, asyncio.Task[R]]] = []

    async def _drain() -> None:
        if not launched:
            return
        done = await asyncio.gather(
            *(t for _, t in launched), return_exceptions=True)
        for (idx, _), value in zip(launched, done):
            results[idx] = value
        launched.clear()

    for index, item in enumerate(items):
        try:
            cold = await is_cold(item)
        except Exception as exc:  # a predicate failure fails ITS item only
            results[index] = exc
            continue
        if cold:
            await _drain()
            try:
                results[index] = await _one(item)
            except Exception as exc:
                results[index] = exc
        else:
            launched.append((index, asyncio.ensure_future(_one(item))))
    await _drain()
    return results
```

- [ ] **Step 4: Run the tests to verify they pass**

```
uv run --extra dev pytest tests/unit/test_concurrent_ingest.py -q
```
Expected: PASS, no test in the file regressed.

- [ ] **Step 5: Prove the tests discriminate, by mutation**

Apply each mutant, run the file, restore (copy the original aside first and `cmp` it back so the restore is byte-identical):

| mutant | must fail |
|---|---|
| delete the `await _drain()` before the cold item runs | `test_a_cold_item_runs_with_nothing_else_in_flight` |
| replace the direct `await _one(item)` on the cold path with `launched.append(...)` | same test |
| re-raise instead of storing the `is_cold` exception | `test_is_cold_raising_lands_in_that_items_slot` |
| append results in completion order instead of by index | `test_results_stay_in_input_order_with_a_cold_item_in_the_middle` |

Record to `.superpowers/sdd/task-1-mutations.json`. A mutant that fails NO test means the test is vacuous — fix the test, not the record.

- [ ] **Step 6: Run the full gate as ONE process**

```
uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live"
```
Expected: green, ~11 min. Launch it as `( cmd > log 2>&1; echo EXIT=$? >> log ) &` — a bare `&` backgrounds only the last statement and the run dies at the tool timeout.

- [ ] **Step 7: Commit**

```bash
git add src/graph_extract/concurrent_ingest.py tests/unit/test_concurrent_ingest.py
git commit -m "feat(ingest): optional cold-item barrier in run_concurrently"
```

---

### Task 2: `WarmupGate` and its config knob

**Files:**
- Create: `src/graph_extract/warmup.py`
- Modify: `src/graph_extract/config.py` (add the field beside `ingest_article_concurrency`, and a validator beside `_at_least_one`)
- Test: `tests/unit/test_warmup.py` (create), `tests/unit/test_config.py` (add)

**Interfaces:**
- Consumes: nothing from Task 1 (the gate does not import the helper).
- Produces:
  ```python
  class WarmupGate:
      def __init__(self, driver: AsyncDriver, group_id: str, threshold: int) -> None
      async def is_cold_source(self, source_id: str) -> bool
      async def is_cold_article(self, article_id: str) -> bool
  ```
  Task 3 wires both methods in. `ExtractSettings.ingest_warmup_articles: int`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_warmup.py`:

```python
"""The warm-up predicate is derived from the GRAPH, not from a run counter.

That is what makes warmth persist across runs, processes and days for free: a
source ingested over three days warms once, and a quarterly re-extraction of an
already-ingested source never warms at all -- correctly, since its hub entities
already exist.
"""
from __future__ import annotations

import pytest

from graph_extract.warmup import WarmupGate


class _FakeResult:
    def __init__(self, records):
        self.records = records


class _FakeDriver:
    """Records every query so the caching claims can be asserted on."""

    def __init__(self, live_counts: dict[str, int], sources: dict[str, str] | None = None):
        self.live_counts = live_counts
        self.sources = sources or {}
        self.queries: list[tuple[str, dict]] = []

    async def execute_query(self, query, **kwargs):
        self.queries.append((query, kwargs))
        if "a.source_id AS source_id" in query:
            sid = self.sources.get(kwargs["article_id"])
            return _FakeResult([{"source_id": sid}] if sid is not None else [])
        return _FakeResult([{"live": self.live_counts.get(kwargs["source_id"], 0)}])


async def test_a_source_below_the_threshold_is_cold():
    gate = WarmupGate(_FakeDriver({"s1": 3}), "backup-docs", 8)
    assert await gate.is_cold_source("s1") is True


async def test_a_source_at_the_threshold_is_warm():
    gate = WarmupGate(_FakeDriver({"s1": 8}), "backup-docs", 8)
    assert await gate.is_cold_source("s1") is False


async def test_threshold_zero_never_touches_the_driver():
    driver = _FakeDriver({"s1": 0})
    gate = WarmupGate(driver, "backup-docs", 0)
    assert await gate.is_cold_source("s1") is False
    assert driver.queries == [], "threshold 0 is off; it must not query"


async def test_a_warm_source_is_cached_and_a_cold_one_is_not():
    """Warmth is monotonic within a run (nothing removes live episodes mid-run),
    so a warm answer can be cached. A cold source must be re-queried -- it is
    about to become warm, and that is the whole point."""
    driver = _FakeDriver({"warm": 8, "cold": 1})
    gate = WarmupGate(driver, "backup-docs", 8)
    assert await gate.is_cold_source("warm") is False
    assert await gate.is_cold_source("warm") is False
    assert len(driver.queries) == 1, "a warm source must be queried once"
    driver.queries.clear()
    assert await gate.is_cold_source("cold") is True
    assert await gate.is_cold_source("cold") is True
    assert len(driver.queries) == 2, "a cold source must be re-queried each time"


async def test_a_missing_article_is_cold():
    """sync_core.apply enqueues the semantic job BEFORE apply_structural writes
    the Article node, so a worker can claim a job in that window. Cold is the
    conservative branch: treating it as warm would disable the protection
    exactly when the source is newest."""
    gate = WarmupGate(_FakeDriver({}, sources={}), "backup-docs", 8)
    assert await gate.is_cold_article("unknown-article") is True


async def test_an_article_resolves_through_its_source():
    driver = _FakeDriver({"s1": 2}, sources={"a1": "s1"})
    gate = WarmupGate(driver, "backup-docs", 8)
    assert await gate.is_cold_article("a1") is True


async def test_a_driver_failure_propagates():
    """The caller (run_concurrently) puts it in the item's slot. Swallowing it
    here would silently disable warm-up on a Neo4j blip."""
    class _Boom:
        async def execute_query(self, *a, **k):
            raise RuntimeError("neo4j down")

    gate = WarmupGate(_Boom(), "backup-docs", 8)
    with pytest.raises(RuntimeError, match="neo4j down"):
        await gate.is_cold_source("s1")
```

Add to `tests/unit/test_config.py`:

```python
def test_warmup_default_is_eight():
    s = ExtractSettings(_env_file=None, docext_base_url="http://x",
                        docext_read_key="k", neo4j_uri="bolt://x",
                        neo4j_user="u", neo4j_password="p")
    assert s.ingest_warmup_articles == 8


def test_warmup_zero_is_accepted_as_off():
    s = ExtractSettings(_env_file=None, docext_base_url="http://x",
                        docext_read_key="k", neo4j_uri="bolt://x",
                        neo4j_user="u", neo4j_password="p",
                        ingest_warmup_articles=0)
    assert s.ingest_warmup_articles == 0


def test_a_negative_warmup_is_rejected_not_clamped():
    with pytest.raises(ValidationError):
        ExtractSettings(_env_file=None, docext_base_url="http://x",
                        docext_read_key="k", neo4j_uri="bolt://x",
                        neo4j_user="u", neo4j_password="p",
                        ingest_warmup_articles=-1)
```

- [ ] **Step 2: Run to verify they fail**

```
uv run --extra dev pytest tests/unit/test_warmup.py tests/unit/test_config.py -q
```
Expected: `test_warmup.py` fails at import (`ModuleNotFoundError: graph_extract.warmup`); the three config tests fail — note `ExtractSettings` has `extra="ignore"`, so the unknown kwarg is DROPPED rather than raising: `test_warmup_default_is_eight` fails on `AttributeError`, and `test_a_negative_warmup_is_rejected_not_clamped` fails because no `ValidationError` is raised.

- [ ] **Step 3: Add the config field and validator**

In `src/graph_extract/config.py`, beside `ingest_article_concurrency`:

```python
    # Articles at the start of a source that run one at a time, with nothing
    # else in flight, before the fan-out begins. 0 = off.
    #
    # Measured (docs/superpowers/ab-concurrency-2026-09-14.md): at N=4 the
    # pilot produced 30 duplicate entities, all exact-name, concentrated on hub
    # entities (mean article span 7.6 vs 2.4). Duplicate risk for an entity in
    # k articles scales with k-1, and per source by sort_order the first 8
    # articles carry 79.5% of that risk weight -- documentation sources open
    # with overview pages that name the product and its core concepts. The knee
    # is at 4 (71.3%); 12 buys 3 more points for 50% more sequential articles.
    ingest_warmup_articles: int = 8
```

And beside `_at_least_one`:

```python
    @field_validator("ingest_warmup_articles")
    @classmethod
    def _not_negative(cls, v: int) -> int:
        # Rejected, not clamped, for the same reason as its sibling: a -1 in
        # the environment means someone believes they configured something.
        if v < 0:
            raise ValueError(f"ingest_warmup_articles must be >= 0, got {v}")
        return v
```

- [ ] **Step 4: Create `src/graph_extract/warmup.py`**

```python
"""Is this source still in its warm-up window?

Concurrency duplicates entities when two in-flight articles both extract an
entity that is not yet in the graph. Measured at N=4 on the pilot: 30 duplicates,
every one an exact-name collision, concentrated on hub entities (mean article
span 7.6 against 2.4 for the rest) -- `AWS Backup`, `Amazon EC2`, `Microsoft
Azure`. Those are the nodes cross-vendor questions resolve through, which is what
design invariant #4 exists to protect.

The risk is front-loaded: for an entity in k articles it scales with k-1, and
636 of 999 baseline entities are single-article and carry none. Per source in
`sort_order`, the first 8 articles carry 79.5% of the total risk weight, because
a documentation source opens with overview pages naming the product and its core
concepts.

**Warmth is a property of the graph, not of a run.** That is deliberate: it makes
the answer identical for `ingest_source` (which has an order) and the semantic
worker (which claims batches spanning sources and has no notion of a start), and
it persists across runs, restarts and days with no new state. A source ingested
over three days warms once; a quarterly re-extraction of an already-ingested
source never warms, which is right -- its hubs already exist; a semantic-layer
reset drops the count to zero and it warms again, which is also right.
"""
from __future__ import annotations

from neo4j import AsyncDriver

_LIVE_ARTICLES = (
    "MATCH (a:Article {source_id:$source_id})-[r:HAS_EPISODE]->"
    "(:Episodic {group_id:$group_id}) "
    "WHERE coalesce(r.superseded, false) = false "
    "RETURN count(DISTINCT a) AS live"
)

_ARTICLE_SOURCE = "MATCH (a:Article {id:$article_id}) RETURN a.source_id AS source_id"


class WarmupGate:
    """Per-run cold/warm predicate over sources.

    `Article.source_id` carries a RANGE index (`article_source`), so the count is
    an index seek plus a small expand -- negligible against a ~300 s article.
    """

    def __init__(self, driver: AsyncDriver, group_id: str, threshold: int) -> None:
        self._driver = driver
        self._group_id = group_id
        self._threshold = threshold
        self._warm: set[str] = set()

    async def is_cold_source(self, source_id: str) -> bool:
        if self._threshold <= 0:
            return False
        if source_id in self._warm:
            return False
        r = await self._driver.execute_query(
            _LIVE_ARTICLES, source_id=source_id, group_id=self._group_id)
        live = r.records[0]["live"] if r.records else 0
        if live >= self._threshold:
            # Warmth is monotonic within a run -- nothing removes live episodes
            # mid-run -- so a warm answer is cached for the run's lifetime. A
            # cold source is deliberately NOT cached: it is about to become warm.
            self._warm.add(source_id)
            return False
        return True

    async def is_cold_article(self, article_id: str) -> bool:
        if self._threshold <= 0:
            return False
        r = await self._driver.execute_query(_ARTICLE_SOURCE, article_id=article_id)
        if not r.records or r.records[0]["source_id"] is None:
            # `sync_core.apply` enqueues the semantic job BEFORE it writes the
            # Article node, so a worker can claim a job in that window. Cold is
            # the conservative branch; calling it warm would disable the
            # protection exactly when the source is newest.
            return True
        return await self.is_cold_source(r.records[0]["source_id"])
```

- [ ] **Step 5: Run the tests to verify they pass**

```
uv run --extra dev pytest tests/unit/test_warmup.py tests/unit/test_config.py -q
```
Expected: PASS.

- [ ] **Step 6: Prove the tests discriminate, by mutation**

| mutant | must fail |
|---|---|
| remove the `threshold <= 0` short-circuit in `is_cold_source` | `test_threshold_zero_never_touches_the_driver` |
| cache cold sources too (`self._warm.add` unconditionally) | `test_a_warm_source_is_cached_and_a_cold_one_is_not` |
| drop the cache entirely | same test |
| `live > self._threshold` instead of `>=` | `test_a_source_at_the_threshold_is_warm` |
| missing article returns `False` | `test_a_missing_article_is_cold` |
| remove the `_not_negative` validator | `test_a_negative_warmup_is_rejected_not_clamped` |
| default the field to `0` | `test_warmup_default_is_eight` |

Record to `.superpowers/sdd/task-2-mutations.json`.

- [ ] **Step 7: Run the full gate as ONE process, then commit**

```bash
uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live"
git add src/graph_extract/warmup.py src/graph_extract/config.py tests/unit/test_warmup.py tests/unit/test_config.py
git commit -m "feat(ingest): graph-derived warm-up predicate + ingest_warmup_articles"
```

---

### Task 3: Wire warm-up into both ingest paths

**Files:**
- Modify: `src/graph_extract/ingest_driver.py` (`ingest_source`, and the driver gains `warmup_gate`)
- Modify: `src/graph_extract/cli.py` (`_build_ingest_driver` constructs the gate)
- Modify: `src/graph_sync/semantic_worker.py` (`run_worker_once` and `run_worker` accept/forward `is_cold`)
- Modify: `src/graph_sync/cli.py` (`_build_worker_deps` / the `worker` command pass the predicate)
- Test: `tests/unit/test_ingest_source_concurrency.py`, `tests/unit/test_worker_concurrency.py`, `tests/unit/test_cli_worker_wiring.py`

**Interfaces:**
- Consumes: `run_concurrently(..., is_cold=...)` (Task 1); `WarmupGate.is_cold_source` / `.is_cold_article` and `ExtractSettings.ingest_warmup_articles` (Task 2).
- Produces: `run_worker_once(..., is_cold: Callable[[str], Awaitable[bool]] | None = None)` and the same keyword on `run_worker`. `IngestDriver` carries `self.warmup_gate`.

A remove-only job group is never cold: tombstoning extracts nothing, so there is no entity to race over.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_ingest_source_concurrency.py`:

```python
async def test_the_first_articles_of_a_cold_source_do_not_overlap():
    """The opening articles run one at a time; once the source is warm the rest
    overlap. Asserted on FINISH-before-START, not on dispatch order -- a
    sequential implementation satisfies dispatch order identically."""
    log: list[str] = []
    live = {"n": 0}

    class _Gate:
        async def is_cold_source(self, source_id: str) -> bool:
            return live["n"] < 2

    async def fake_ingest_article(article_id: str):
        log.append(f"start-{article_id}")
        await asyncio.sleep(0.03)
        live["n"] += 1
        log.append(f"end-{article_id}")
        return IngestArticleResult(article_id=article_id)

    driver = _driver_with(fake_ingest_article, concurrency=4, gate=_Gate())
    await driver.ingest_source("s1")

    # a1 and a2 are cold: each finished before the next started
    assert log.index("end-a1") < log.index("start-a2")
    assert log.index("end-a2") < log.index("start-a3")
    # a3 and a4 are warm: they overlap
    assert log.index("start-a4") < log.index("end-a3")


async def test_warmup_is_inert_at_concurrency_one():
    """THE test that justifies shipping ingest_warmup_articles defaulted to 8.
    It must assert on ORDER, never on counts. If it is ever weakened, the
    default loses its justification (spec 3.6)."""
    orders = {}
    for threshold in (0, 8):
        log: list[str] = []

        class _Gate:
            async def is_cold_source(self, source_id: str) -> bool:
                return threshold > 0

        async def fake_ingest_article(article_id: str):
            log.append(f"start-{article_id}")
            await asyncio.sleep(0.01)
            log.append(f"end-{article_id}")
            return IngestArticleResult(article_id=article_id)

        driver = _driver_with(fake_ingest_article, concurrency=1, gate=_Gate())
        await driver.ingest_source("s1")
        orders[threshold] = list(log)
    assert orders[0] == orders[8], (
        f"warm-up must be inert at concurrency 1; got {orders}")
```

Add to `tests/unit/test_worker_concurrency.py`:

```python
async def test_a_cold_group_runs_alone_and_warm_groups_overlap():
    log: list[str] = []

    async def is_cold(article_id: str) -> bool:
        return article_id == "cold"

    class _Timed(_Ingest):
        async def ingest_article(self, article_id):
            log.append(f"start-{article_id}")
            await asyncio.sleep(0.04 if article_id == "slow" else 0.01)
            log.append(f"end-{article_id}")
            return await super().ingest_article(article_id)

    store = _Store([_job("j1", "slow"), _job("j2", "cold"), _job("j3", "warm")])
    await run_worker_once(store, _Timed(log), concurrency=4, is_cold=is_cold, **_KW)
    assert log.index("end-slow") < log.index("start-cold"), log
    assert log.index("end-cold") < log.index("start-warm"), log
    assert sorted(store.completed) == ["j1", "j2", "j3"]


async def test_a_remove_only_group_is_never_cold():
    """Tombstoning extracts nothing, so there is no entity to race over."""
    asked: list[str] = []

    async def is_cold(article_id: str) -> bool:
        asked.append(article_id)
        return True

    store = _Store([_job("j1", "a1", op="remove")])
    await run_worker_once(store, _Ingest([]), concurrency=4, is_cold=is_cold, **_KW)
    assert asked == [], "a remove-only group must not consult the predicate"
    assert store.completed == ["j1"]
```

Add to `tests/unit/test_cli_worker_wiring.py` a test in the same shape as the existing concurrency-passthrough one, asserting `run_worker` receives an `is_cold` that is not `None` when `ingest_warmup_articles=8`, and `None` when it is `0`.

- [ ] **Step 2: Run to verify they fail**

```
uv run --extra dev pytest tests/unit/test_ingest_source_concurrency.py tests/unit/test_worker_concurrency.py tests/unit/test_cli_worker_wiring.py -q
```
Expected: FAIL — `_driver_with` has no `gate` parameter; `run_worker_once` has no `is_cold` keyword.

- [ ] **Step 3: Implement `ingest_source`**

In `src/graph_extract/ingest_driver.py`, `ingest_source` — the fan-out becomes:

```python
        gate = getattr(self, "warmup_gate", None)
        is_cold = None
        if gate is not None:
            async def is_cold(_article_id: str) -> bool:
                # Warmth is a property of the SOURCE, so every article of this
                # call asks the same question; the gate caches the warm answer.
                return await gate.is_cold_source(source_id)
        results = await run_concurrently(
            ids, self.ingest_article,
            limit=self._s.ingest_article_concurrency, is_cold=is_cold)
```

Give `IngestDriver.__init__` a `warmup_gate=None` keyword stored as `self.warmup_gate`, mirroring how `timings` is carried.

- [ ] **Step 4: Implement the worker**

In `src/graph_sync/semantic_worker.py`, add `is_cold: Callable[[str], Awaitable[bool]] | None = None` to `run_worker_once` and to `run_worker` (forwarded, exactly like `concurrency`). Wrap it so a remove-only group is never cold, and pass it through:

```python
    group_is_cold = None
    if is_cold is not None:
        async def group_is_cold(group: list) -> bool:
            # Tombstoning extracts nothing, so a remove-only group cannot race
            # over an entity and never needs the barrier.
            if all(j["op"] == "remove" for j in group):
                return False
            return await is_cold(group[0]["article_id"])

    results = await run_concurrently(
        list(groups.values()), _run_group, limit=concurrency, is_cold=group_is_cold)
```

In `src/graph_sync/cli.py`, build the gate next to the existing `concurrency` passthrough and pass `is_cold=gate.is_cold_article` when `ingest_warmup_articles > 0`, else `None`. The knob is on `ExtractSettings`, so read it from `get_extract_settings()` — the same `lru_cache`d object `_build_worker_deps` hands the driver.

- [ ] **Step 5: Run the tests to verify they pass**

```
uv run --extra dev pytest tests/unit/test_ingest_source_concurrency.py tests/unit/test_worker_concurrency.py tests/unit/test_cli_worker_wiring.py -q
```

- [ ] **Step 6: Prove the tests discriminate, by mutation**

| mutant | must fail |
|---|---|
| `ingest_source` passes `is_cold=None` | `test_the_first_articles_of_a_cold_source_do_not_overlap` |
| `run_worker_once` drops the `is_cold` pass-through | `test_a_cold_group_runs_alone_and_warm_groups_overlap` |
| remove the remove-only branch | `test_a_remove_only_group_is_never_cold` |
| the cli passes `is_cold` even when the threshold is 0 | the cli wiring test |
| make the cold path launch as a task instead of awaiting | both overlap tests |

Record to `.superpowers/sdd/task-3-mutations.json`.

- [ ] **Step 7: Run the full gate as ONE process, then commit**

```bash
uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live"
git add src/graph_extract/ingest_driver.py src/graph_extract/cli.py src/graph_sync/semantic_worker.py src/graph_sync/cli.py tests/unit/
git commit -m "feat(ingest): apply the warm-up barrier on both ingest paths"
```

---

### Task 4: The duplicate planner (read-only)

**Files:**
- Create: `src/graph_extract/merge_duplicates.py`
- Test: `tests/integration/test_merge_duplicates.py` (create)

**Interfaces:**
- Consumes: nothing from Tasks 1–3.
- Produces:
  ```python
  async def count_duplicates(driver: AsyncDriver, group_id: str) -> int
  async def plan_merges(driver: AsyncDriver, group_id: str) -> dict
  ```
  `plan_merges` returns the report payload:
  ```python
  {"groups": [{"name": str, "survivor": str, "losers": list[str],
               "members": [{"uuid", "created_at", "labels", "degree", "summary_len"}],
               "name_embedding_cosine": float | None}],
   "totals": {"groups": int, "excess": int},     # excess == count_duplicates
   "near_duplicates_not_merged": [{"normalized": str, "names": list[str]}],
   "label_conflicts": [{"name": str, "labels": list[list[str]]}]}
  ```
  `apply_merges` (Task 5) returns this same payload with `merged`, `edges_moved`, `self_loops_created` added to `totals` and a `summary_dropped` list. Tasks 5, 6 and 7 all consume these two functions, and Task 7 reads `totals.excess` specifically. **This task writes nothing to the graph.**

Survivor rule: earliest `created_at`, ties broken by ascending `uuid` — a total order, so the survivor is a pure function of the group. It is the node sequential ingestion would have produced (the first extraction creates it; every later one resolves onto it) and the one most likely to be referenced from outside.

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/test_merge_duplicates.py` with a helper that seeds entities and the first planner tests:

```python
async def test_the_planner_writes_nothing(neo4j_driver):
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=3)
    before = await _snapshot(neo4j_driver)
    await plan_merges(neo4j_driver, "backup-docs")
    assert await _snapshot(neo4j_driver) == before, "report mode must not write"


async def test_the_survivor_is_the_earliest_created_at(neo4j_driver):
    """Deliberately built so the earliest node is NEITHER the smallest uuid NOR
    the highest degree -- otherwise the test cannot tell the rules apart."""
    await _seed(neo4j_driver, uuid="zzz", name="AWS Backup",
                created_at="2026-01-01T00:00:00Z", degree=1)
    await _seed(neo4j_driver, uuid="aaa", name="AWS Backup",
                created_at="2026-06-01T00:00:00Z", degree=9)
    plan = await plan_merges(neo4j_driver, "backup-docs")
    group = plan["groups"][0]
    assert group["survivor"] == "zzz"
    assert group["losers"] == ["aaa"]


async def test_ties_on_created_at_break_by_ascending_uuid(neo4j_driver):
    await _seed(neo4j_driver, uuid="bbb", name="X", created_at="2026-01-01T00:00:00Z")
    await _seed(neo4j_driver, uuid="aaa", name="X", created_at="2026-01-01T00:00:00Z")
    plan = await plan_merges(neo4j_driver, "backup-docs")
    assert plan["groups"][0]["survivor"] == "aaa"


async def test_case_variants_are_not_a_group_but_are_reported(neo4j_driver):
    """The sequential baseline holds 6 such pairs that graphiti itself left
    apart. They are not concurrency artefacts, and a pass whose purpose is to
    repair concurrency artefacts must not alter the reference graph."""
    await _seed(neo4j_driver, uuid="a", name="Access Control")
    await _seed(neo4j_driver, uuid="b", name="Access control")
    plan = await plan_merges(neo4j_driver, "backup-docs")
    assert plan["groups"] == []
    assert "Access Control" in str(plan["near_duplicates_not_merged"])


async def test_another_group_id_is_not_a_duplicate(neo4j_driver):
    await _seed(neo4j_driver, uuid="a", name="X", group_id="backup-docs")
    await _seed(neo4j_driver, uuid="b", name="X", group_id="other")
    plan = await plan_merges(neo4j_driver, "backup-docs")
    assert plan["groups"] == []
    assert plan["totals"]["excess"] == 0
    assert await count_duplicates(neo4j_driver, "backup-docs") == 0


async def test_count_duplicates_counts_excess_not_groups(neo4j_driver):
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=3)
    await _seed_duplicate_group(neo4j_driver, name="Amazon EC2", n=2)
    # 3 nodes of one name + 2 of another = 5 nodes, 2 names, 3 excess
    assert await count_duplicates(neo4j_driver, "backup-docs") == 3
```

- [ ] **Step 2: Run to verify they fail**

```
uv run --extra dev pytest tests/integration/test_merge_duplicates.py -q
```
Expected: FAIL at import — `graph_extract.merge_duplicates` does not exist. Docker must be running (testcontainers).

- [ ] **Step 3: Implement the planner**

Create `src/graph_extract/merge_duplicates.py` with the module docstring recording the evidence (30 of 30 observed duplicates were exact-name; the 6 baseline case-variant pairs are excluded deliberately; §1.2's escalation tax), and:

```python
_COUNT = (
    "MATCH (e:Entity {group_id:$group_id}) "
    "WITH e.name AS name, count(*) AS c WHERE c > 1 "
    "RETURN coalesce(sum(c - 1), 0) AS excess"
)

_GROUPS = (
    "MATCH (e:Entity {group_id:$group_id}) "
    "WITH e.name AS name, collect(e) AS nodes WHERE size(nodes) > 1 "
    "RETURN name, [n IN nodes | {uuid: n.uuid, created_at: toString(n.created_at), "
    "  labels: labels(n), labels_prop: n.labels, summary: n.summary, "
    "  name_embedding: n.name_embedding, degree: COUNT { (n)--() }}] AS members "
    "ORDER BY name"
)

_NEAR = (
    "MATCH (e:Entity {group_id:$group_id}) "
    "WITH toLower(trim(e.name)) AS norm, collect(DISTINCT e.name) AS names "
    "WHERE size(names) > 1 RETURN norm, names ORDER BY norm"
)
```

`plan_merges` runs the three queries, sorts each group's members by `(created_at, uuid)` to pick the survivor, computes the cosine between member `name_embedding`s in Python (report only — it checks the "same string, same embedder, identical vector" assumption rather than trusting it), and collects `label_conflicts` where members carry differing custom labels and the survivor is not bare `:Entity`. It must not emit `name_embedding` itself into the payload.

- [ ] **Step 4: Run the tests to verify they pass**

```
uv run --extra dev pytest tests/integration/test_merge_duplicates.py -q
```

- [ ] **Step 5: Prove the tests discriminate, by mutation**

| mutant | must fail |
|---|---|
| order members by `uuid` | `test_the_survivor_is_the_earliest_created_at` |
| order members by `degree` | same |
| drop the `uuid` tie-break | `test_ties_on_created_at_break_by_ascending_uuid` |
| group on `toLower(e.name)` | `test_case_variants_are_not_a_group_but_are_reported` |
| drop `group_id` from `_GROUPS` | `test_another_group_id_is_not_a_duplicate` |
| `count(*)` instead of `sum(c-1)` | `test_count_duplicates_counts_excess_not_groups` |
| any write statement in the planner | `test_the_planner_writes_nothing` |

Record to `.superpowers/sdd/task-4-mutations.json`.

- [ ] **Step 6: Run the full gate as ONE process, then commit**

```bash
uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live"
git add src/graph_extract/merge_duplicates.py tests/integration/test_merge_duplicates.py
git commit -m "feat(merge): read-only exact-name duplicate planner"
```

---

### Task 5: The merge transaction

**Files:**
- Modify: `src/graph_extract/merge_duplicates.py` (add `merge_group` and `apply_merges`)
- Test: `tests/integration/test_merge_duplicates.py` (add)

**Interfaces:**
- Consumes: `plan_merges` (Task 4).
- Produces: `async def apply_merges(driver: AsyncDriver, group_id: str) -> dict` — plans, then merges each group in **one explicit transaction per group**, returning the report payload plus `merged`, `edges_moved` (by type), `self_loops_created`, `summary_dropped`.

This is the destructive core. The whole safety argument rests on two mechanics; implement them exactly.

**Every edge is re-created with its full property map INCLUDING its original uuid, then the original is deleted.** The 655 baseline episodes carry 4,886 `entity_edges` references *by edge uuid*, and `Community.cited_fact_uuids`, `resolve_citations` and the sweep all address facts by uuid. A regenerated uuid orphans every one of them. There is no uniqueness constraint on `RELATES_TO.uuid`, so old and new sharing a uuid inside the transaction is legal.

**Step 6 uses `DELETE`, never `DETACH DELETE`.** This is the load-bearing safety property of the entire pass: any relationship type this plan did not enumerate makes Neo4j refuse the delete and abort the transaction, rather than silently dropping it with the loser.

- [ ] **Step 1: Write the failing tests**

The full-map conservation test is the important one — a count is never enough for a destructive pass:

```python
async def test_every_edge_survives_with_its_full_property_map_and_uuid(neo4j_driver):
    await _seed_realistic_group(neo4j_driver)  # facts both ways, MENTIONS, episodes
    before = {r["uuid"]: r for r in await _edge_snapshot(neo4j_driver)}
    await apply_merges(neo4j_driver, "backup-docs")
    after = {r["uuid"]: r for r in await _edge_snapshot(neo4j_driver)}
    assert set(before) == set(after), "every edge uuid must survive"
    for uuid, old in before.items():
        new = after[uuid]
        for key, value in old["props"].items():
            if key in ("source_node_uuid", "target_node_uuid"):
                continue
            assert new["props"][key] == value, f"{uuid}.{key} changed"


async def test_stored_endpoints_agree_with_topology_for_every_edge(neo4j_driver):
    await _seed_realistic_group(neo4j_driver)
    await apply_merges(neo4j_driver, "backup-docs")
    r = await neo4j_driver.execute_query(
        "MATCH (a:Entity)-[f:RELATES_TO]->(b:Entity) "
        "WHERE f.source_node_uuid <> a.uuid OR f.target_node_uuid <> b.uuid "
        "RETURN count(f) AS n")
    assert r.records[0]["n"] == 0


async def test_the_fact_embedding_stays_scorable(neo4j_driver):
    vec = [0.1] * 768
    await _seed_group_with_fact_embedding(neo4j_driver, vec)
    await apply_merges(neo4j_driver, "backup-docs")
    r = await neo4j_driver.execute_query(
        "MATCH ()-[f:RELATES_TO]->() "
        "RETURN vector.similarity.cosine(f.fact_embedding, $v) AS sim", v=vec)
    assert abs(r.records[0]["sim"] - 1.0) < 1e-6


async def test_a_foreign_edge_type_aborts_the_group_and_stops_the_run(neo4j_driver):
    """The safety property. DELETE (not DETACH DELETE) makes an unenumerated
    edge type fail loudly instead of vanishing with the loser."""
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=2)
    await neo4j_driver.execute_query(
        "MATCH (l:Entity {uuid:$l}) CREATE (l)-[:FOO]->(:Thing)", l=_LOSER)
    before = await _snapshot(neo4j_driver)
    with pytest.raises(Exception, match="FOO"):
        await apply_merges(neo4j_driver, "backup-docs")
    assert await _snapshot(neo4j_driver) == before, "the group must roll back whole"


async def test_applying_twice_changes_nothing_the_second_time(neo4j_driver):
    await _seed_realistic_group(neo4j_driver)
    await apply_merges(neo4j_driver, "backup-docs")
    once = await _snapshot(neo4j_driver)
    res = await apply_merges(neo4j_driver, "backup-docs")
    assert res["totals"]["merged"] == 0
    assert await _snapshot(neo4j_driver) == once, "a no-op must not re-stamp merged_at"


async def test_a_zero_duplicate_graph_is_untouched(neo4j_driver):
    await _seed(neo4j_driver, uuid="a", name="Unique")
    before = await _snapshot(neo4j_driver)
    res = await apply_merges(neo4j_driver, "backup-docs")
    assert res["totals"]["merged"] == 0
    assert await _snapshot(neo4j_driver) == before
```

Plus, from spec §7.2, one test each for: two facts `S→X` and `L→X` surviving as two edges; `L→S` becoming a self-loop counted in `self_loops_created`; `MENTIONS` rewired with an episode mentioning both keeping two edges and every `Episodic.entity_edges` uuid still resolving; `IN_COMMUNITY` rewired by `MERGE` with `c.stale = true`; `SAME_AS` rewired; label promotion applied for a bare `:Entity` survivor and NOT applied (and reported) for a typed one; empty survivor summary taking the loser's and non-empty keeping its own with the dropped text in the audit; step 0 aborting a group whose member was renamed between plan and apply while other groups still merge.

- [ ] **Step 2: Run to verify they fail**

```
uv run --extra dev pytest tests/integration/test_merge_duplicates.py -q
```
Expected: FAIL — `apply_merges` is not defined.

- [ ] **Step 3: Implement `merge_group`**

Add to `src/graph_extract/merge_duplicates.py`. Inside one explicit transaction per group (`async with driver.session() as s: async with await s.begin_transaction() as tx:`), for each loser, run steps 0–7 verbatim from spec §4.5:

- **Step 0** re-match `S` and every `L` by `uuid`, `group_id` **and** `name = $name`; if any is missing or renamed, raise (the transaction rolls back).
- **Step 1** outgoing facts, **Step 2** incoming facts — both using the Cypher in spec §4.5, which copies `properties(old)`, re-sets `source_node_uuid`/`target_node_uuid` to the new endpoints, re-sets `fact_embedding` through `db.create.setRelationshipVectorProperty`, then deletes the old edge. If the Cypher 25 parser objects to `WITH ... CALL <void procedure> ... DELETE` in one statement, split the CALL into a second statement **in the same transaction** — do not drop it.
- **Step 3** `MENTIONS`, **Step 4** `IN_COMMUNITY` (`MERGE` + `SET c.stale = true`), **Step 5** `SAME_AS` (`MERGE`).
- **Step 6** assert `COUNT { (l)--() } = 0`; if not, raise naming the surviving relationship types, then `DELETE l`.
- **Step 7** stamp `merged_from = coalesce(merged_from, []) + [loser.uuid]`, `merged_at = datetime()`, and apply the label-promotion rule.

`apply_merges` iterates groups and **stops on the first failure**, raising — unlike `ingest_source`, which continues. A failure in a destructive pass means the graph is not what the design assumed, and the operator should look before more changes.

- [ ] **Step 4: Run the tests to verify they pass**

```
uv run --extra dev pytest tests/integration/test_merge_duplicates.py -q
```

- [ ] **Step 5: Prove the tests discriminate, by mutation**

| mutant | must fail |
|---|---|
| regenerate the edge uuid instead of copying it | conservation; the `entity_edges` test |
| drop any single key from the copied property map | conservation |
| skip the endpoint `SET` | the endpoint test |
| replace the vector procedure with plain `SET` **and perturb one element** | the cosine test (the perturbation is what makes it discriminate — a plain-`SET`-only mutant may legitimately pass) |
| `DETACH DELETE` the loser | the foreign-edge test |
| auto-commit statements instead of one transaction per group | the foreign-edge rollback assertion |
| collapse meeting edges with `MERGE` on the pair | the two-facts test |
| skip `SET c.stale = true` | the `IN_COMMUNITY` test |
| drop `name = $name` from step 0's re-match | the rename test |
| re-stamp `merged_at` on a no-op | `test_applying_twice_changes_nothing_the_second_time` |

Record to `.superpowers/sdd/task-5-mutations.json`.

- [ ] **Step 6: Run the full gate as ONE process, then commit**

```bash
uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live"
git add src/graph_extract/merge_duplicates.py tests/integration/test_merge_duplicates.py
git commit -m "feat(merge): transactional exact-name entity merge, DELETE not DETACH DELETE"
```

---

### Task 6: The `merge-duplicates` CLI command

**Files:**
- Modify: `src/graph_extract/cli.py`
- Test: `tests/integration/test_merge_duplicates.py` (add a report-mode-writes-nothing test through the command)

**Interfaces:**
- Consumes: `plan_merges`, `apply_merges` (Tasks 4–5).
- Produces: the `merge-duplicates` command. Tasks 7–8 reference it by name only.

- [ ] **Step 1: Write the failing test**

```python
async def test_report_mode_is_the_default_and_writes_nothing(neo4j_driver):
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=3)
    before = await _snapshot(neo4j_driver)
    result = CliRunner().invoke(app, ["merge-duplicates"])
    assert result.exit_code == 0
    assert "AWS Backup" in result.stdout
    assert await _snapshot(neo4j_driver) == before
```

- [ ] **Step 2: Run to verify it fails**

```
uv run --extra dev pytest tests/integration/test_merge_duplicates.py -k report_mode -q
```
Expected: FAIL — `No such command 'merge-duplicates'`.

- [ ] **Step 3: Implement**

Add to `src/graph_extract/cli.py`, following the `cleanup` command's shape (`_build_driver`, `try/finally`, `_dump`):

```python
@app.command("merge-duplicates")
def merge_duplicates(
    apply: bool = typer.Option(
        False, "--apply",
        help="Actually merge. Without this the command only reports."),
) -> None:
    """Merge :Entity nodes sharing a byte-identical name within one group_id.

    Report-only by default. Concurrent ingest creates these: the A/B measured 30
    at N=4, every one an exact-name collision on a hub entity. They are not
    static -- graphiti escalates to an LLM dedup call on EVERY later mention of
    an ambiguous name, so a duplicate is a permanent tax on the hottest names in
    the corpus.
    """

    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            if apply:
                typer.echo(
                    "WARNING: --apply must not run while any ingest or semantic "
                    "worker is running. graphiti saves facts with MATCH "
                    "(source:Entity {uuid}) ... MERGE; if an in-flight episode "
                    "resolved an entity to a node this deletes, that edge is "
                    "SILENTLY not written -- no error, no log, no retry. This "
                    "command cannot check that condition for you.", err=True)
                _dump(await apply_merges(driver, settings.group_id))
            else:
                _dump(await plan_merges(driver, settings.group_id))
        finally:
            await driver.close()

    asyncio.run(_run())
```

- [ ] **Step 4: Run the test to verify it passes**

- [ ] **Step 5: Prove it discriminates**

Mutant: make the default `apply=True` → the report-mode test must fail. Mutant: drop the warning echo → add an assertion on the warning text in an `--apply` test so this is covered. Record to `.superpowers/sdd/task-6-mutations.json`.

- [ ] **Step 6: Run the full gate as ONE process, then commit**

```bash
uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live"
git add src/graph_extract/cli.py tests/integration/test_merge_duplicates.py
git commit -m "feat(cli): merge-duplicates, report-only by default"
```

---

### Task 7: Surface duplicates where they matter

**Files:**
- Modify: `src/graph_extract/cli.py` (`ingest` prints the count; `cleanup` and `maintenance` include `duplicates.report`)
- Modify: `src/theme_builder/cli.py` (`theme-build` refuses over duplicates unless `--allow-duplicates`)
- Test: `tests/integration/test_merge_duplicates.py`, `tests/integration/` theme-build tests

**Interfaces:**
- Consumes: `count_duplicates`, `plan_merges` (Task 4).
- Produces: `assert_no_duplicates(driver, group_id, *, allow) -> int` and
  `DuplicateEntitiesError` in `merge_duplicates.py`; `theme-build --allow-duplicates`.

`theme-build` refuses because community detection over a fragmented entity graph yields communities that are *wrong*, not merely stale: a duplicated `Microsoft Azure` splits one real community in two, and the strong-tier reports written over that partition are plausible and wrong in a way nothing downstream can detect. A merge *after* `theme-build` also invalidates every report citing the loser's membership, so enforcing merge → theme-build costs nothing and saves a regeneration.

`cleanup` and `maintenance` include the **report only, never the apply** — a composite command on a weekly timer must not delete nodes on its own.

The worker does **not** print the count per batch: on a million-entity graph that aggregation is seconds per batch for a number that only changes meaningfully per source.

- [ ] **Step 1: Write the failing tests**

**The guard is a function, not inline code in the command.** `theme-build` with
`--allow-duplicates` proceeds into a real community build, which costs strong-tier
LLM calls — so a test may never invoke that path. Test the guard directly instead;
the command's only job is to call it before doing any work.

Add to `src/graph_extract/merge_duplicates.py`:

```python
async def assert_no_duplicates(driver, group_id: str, *, allow: bool) -> int:
    """Raise unless the entity graph is free of exact-name duplicates.

    Community detection over a fragmented entity graph yields communities that
    are WRONG, not merely stale: a duplicated `Microsoft Azure` splits one real
    community in two, and the strong-tier reports written over that partition are
    plausible and wrong in a way nothing downstream can detect. Returns the count
    so a caller can report it.
    """
    excess = await count_duplicates(driver, group_id)
    if excess and not allow:
        raise DuplicateEntitiesError(
            f"{excess} exact-name duplicate entities in group {group_id!r}. "
            f"Run `merge-duplicates` to see them, then `merge-duplicates --apply`. "
            f"Use --allow-duplicates to proceed anyway.")
    return excess
```

```python
async def test_the_guard_raises_while_duplicates_exist(neo4j_driver):
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=2)
    with pytest.raises(DuplicateEntitiesError, match="merge-duplicates"):
        await assert_no_duplicates(neo4j_driver, "backup-docs", allow=False)


async def test_allow_duplicates_returns_the_count_instead_of_raising(neo4j_driver):
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=3)
    assert await assert_no_duplicates(neo4j_driver, "backup-docs", allow=True) == 2


async def test_the_guard_passes_on_a_clean_graph(neo4j_driver):
    await _seed(neo4j_driver, uuid="a", name="Unique")
    assert await assert_no_duplicates(neo4j_driver, "backup-docs", allow=False) == 0


def test_theme_build_calls_the_guard_before_doing_any_work(monkeypatch):
    """Hermetic: every dependency is stubbed, so no LLM call can happen. Proves
    the guard runs BEFORE the build, which is the whole point -- a guard that
    runs after the reports are generated has saved nothing."""
    calls: list[str] = []

    async def _fake_guard(driver, group_id, *, allow):
        calls.append("guard")
        raise DuplicateEntitiesError("2 exact-name duplicate entities")

    async def _fake_build(*a, **k):
        calls.append("build")
        return {}

    monkeypatch.setattr(theme_cli, "assert_no_duplicates", _fake_guard)
    monkeypatch.setattr(theme_cli, "_run_theme_build_incremental", _fake_build)
    result = CliRunner().invoke(theme_app, ["theme-build"])
    assert result.exit_code != 0
    assert calls == ["guard"], f"the build must not start; got {calls}"


async def test_cleanup_reports_duplicates_and_never_merges(neo4j_driver):
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=3)
    before = await _snapshot(neo4j_driver)
    result = CliRunner().invoke(app, ["cleanup"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["duplicates"]["totals"]["excess"] == 2
    assert "merged" not in payload["duplicates"]["totals"]
    assert await _snapshot(neo4j_driver) == before
```

- [ ] **Step 2: Run to verify they fail**
- [ ] **Step 3: Implement** — add the guard to `theme_build` before any work begins, the `duplicates` key to `cleanup`/`maintenance`'s `_dump`, and the count line at the end of `ingest`.
- [ ] **Step 4: Run to verify they pass**
- [ ] **Step 5: Mutations** — each must fail the named test:

| mutant | must fail |
|---|---|
| remove the `not allow` condition (always raise) | `test_allow_duplicates_returns_the_count_instead_of_raising` |
| invert it (raise only when `allow`) | `test_the_guard_raises_while_duplicates_exist` |
| call the guard AFTER the build in `theme_build` | `test_theme_build_calls_the_guard_before_doing_any_work` |
| make `cleanup` call `apply_merges` instead of `plan_merges` | `test_cleanup_reports_duplicates_and_never_merges` |
| drop the `duplicates` key from `cleanup`'s payload | same test |

Record to `.superpowers/sdd/task-7-mutations.json`.
- [ ] **Step 6: Full gate as ONE process, then commit**

```bash
git add src/graph_extract/cli.py src/theme_builder/cli.py tests/
git commit -m "feat: surface duplicate count in ingest/cleanup; theme-build refuses over duplicates"
```

---

### Task 8: Runbook and the risk-curve script

**Files:**
- Modify: `docs/superpowers/maintenance-runbook.md`
- Create: `scripts/risk_curve.py`

**Interfaces:** none — documentation and an operator tool.

- [ ] **Step 1: Amend the runbook's existing claim**

`docs/superpowers/maintenance-runbook.md:57` currently reads "The jobs are safe to run concurrently with ingestion (they're idempotent and ...)". Left as is, it directly contradicts the new entry on the one job where it is false. Amend it to exclude `merge-duplicates --apply` explicitly and point at the new entry.

- [ ] **Step 2: Add the runbook entry**

In this order: that `--apply` is the one job **not** safe alongside ingestion and why (silent edge loss through graphiti's `MATCH`-then-`MERGE` save — no error, no log, no retry); that the default is report-only; what the job does; and the recommended sequence `merge-duplicates` → read the report → `merge-duplicates --apply` → `cleanup` → `theme-build`.

- [ ] **Step 3: Create `scripts/risk_curve.py`**

A read-only script that recomputes the §1.1 risk curve from the graph: article spans via `(:Article)-[:HAS_EPISODE]->(:Episodic)-[:MENTIONS]->(:Entity)`, positions from `Article.sort_order`, printing cumulative share of `Σ(k-1)` whose entity first appears within the first `c` articles of each source, for `c` in 1..30. This is what lets `W` be re-derived from a future run's own spans without another paid A/B. It must write nothing.

- [ ] **Step 4: Verify the script against the baseline**

```
uv run --extra dev python scripts/risk_curve.py
```
Expected: reproduces the spec's §1.1 per-source row — `c=4 → 71.3%`, `c=8 → 79.5%`, `c=12 → 82.5%`. If it does not, the script is wrong (the spec's numbers were computed from this exact graph); fix the script, do not edit the spec.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/maintenance-runbook.md scripts/risk_curve.py
git commit -m "docs: merge-duplicates runbook entry + risk-curve recompute script"
```

---

## After the plan: the paid validation (spec §9)

**Not part of this plan and not to be run by an implementer.** When the branch is merged, the user decides whether to run: the same 83 articles at `ingest_article_concurrency=4`, `ingest_warmup_articles=8`, into an isolated group using the A/B runner's supersede-and-restore method (`.superpowers/sdd/ab-concurrent-n4.py` + `ab-revert.py`) — **never a reset of the baseline**.

| | measured (N=4, no warm-up) | to measure (N=4, W=8) |
|---|---|---|
| exact-name duplicates | 30 | predicted ~6 |
| near-duplicates (normalized) | not measured | decides whether §4.1's exclusion holds |
| wall clock | 2 h 10 m | predicted ~2 h 25 m |
| after `merge-duplicates --apply` | — | 0 duplicates |

Roughly $10 and ~2.5 h. If the residual is far above 6, the risk model is wrong for the fan-out shape and `W` should be re-derived with `scripts/risk_curve.py` from that run's own spans.

## Self-Review

**Spec coverage.** §3.1–3.6 → Tasks 1–3. §4.1, 4.3, 4.4 (planner half) → Task 4. §4.2, 4.4 (apply half), 4.5, 4.6, 4.7, 4.9, 4.10 → Task 5. §4.8 command → Task 6; §4.8 guards/surfacing → Task 7; §4.8 runbook → Task 8. §5 config → Task 2. §6 error handling → distributed across the tasks that own each row. §7.1 → Tasks 1–3; §7.2 → Tasks 4–7; §7.3 (`@live`) → **gap, deliberate**: the `@live` tests in §7.3 run against the real baseline and are opt-in; they are folded into Task 4 (planner report against the baseline) and Task 3 (warm predicate against the baseline) as `@live`-marked additions rather than given their own task, because neither is independently reviewable. §9 → the section above, explicitly out of scope for implementers. §10 rejected alternatives → no tasks, correctly.

**Placeholder scan.** Task 5 steps 3 and Task 7 step 3 describe implementation at a higher level than Tasks 1–2, deferring to the spec's verbatim Cypher rather than repeating ~120 lines of it. The spec section is named precisely at each point and the constraints that make the code correct (uuid preservation, `DELETE` not `DETACH DELETE`, the vector procedure, transaction per group) are stated in the task itself, not only by reference. Task 7's steps 2–4 are compressed because they repeat the established test-fail-implement-pass loop for three small, independent edits.

**Type consistency.** `run_concurrently(..., is_cold=)` (Task 1) matches its consumers in Task 3 and its keyword name in the worker. `WarmupGate.is_cold_source` / `.is_cold_article` (Task 2) match Task 3's wiring — `ingest_source` uses the source form, the worker the article form. `count_duplicates` / `plan_merges` (Task 4) match `apply_merges` (Task 5) and both CLI consumers (Tasks 6–7). `ingest_warmup_articles` is spelled identically in Tasks 2, 3 and the constraints block.
