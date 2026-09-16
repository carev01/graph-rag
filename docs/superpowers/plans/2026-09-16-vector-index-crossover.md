# Vector-Index Crossover Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure what a Neo4j vector index actually buys over graphiti's brute-force cosine scan, from 10k to 1M edges, plus its write cost and its recall trend — so Phase B's implementation half is decided on numbers instead of extrapolation.

**Architecture:** A single self-contained script builds synthetic graphs of increasing size in a throwaway `neo4j:2026.07.1-community` testcontainer, times three query shapes at each size (graphiti's real brute-force query, the index-backed procedure, and brute-force-with-index-present as a control), and reports latency, recall@10 at three fetch depths, and index write overhead. Pure logic (vector generation, recall arithmetic, the disk guard) is unit-tested; the query behaviour is integration-tested against a real container.

**Tech Stack:** Python 3.12, `uv`, `neo4j` async driver, `testcontainers`, pytest.

## Global Constraints

Copied from `docs/superpowers/specs/2026-09-16-vector-index-crossover-design.md`. Every task's requirements implicitly include these.

- **No changes under `src/`.** This slice is measurement only. It does not implement `SearchInterface`.
- **`SEARCH` does not exist on Neo4j 2026.07.1** — probed 2026-09-16 under both `CYPHER 5` and `CYPHER 25`. Use `db.index.vector.queryRelationships` and `db.index.vector.queryNodes`. They emit a deprecation notice and work correctly. Do not write `SEARCH` Cypher.
- **Never connect to the live Neo4j for writes.** The only permitted live access is one read-only query fetching real `fact_embedding` values as resampling seeds. The `backup-docs` group (999 entities, 655 episodes) is a read-only reference.
- **All measurement runs in a `neo4j:2026.07.1-community` testcontainer.** The image is already cached locally and `tests/integration/conftest.py` already uses that exact tag.
- Vector dimension is **768**. Similarity function is **cosine**.
- Edge sweep steps: **10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000**. Node sweep steps: **10_000, 50_000, 100_000, 250_000**.
- Recall is reported at index fetch depths **k = 10, 50, 200**, always scoring `recall@10`.
- Timing is the **median of 5 runs after one discarded warm-up run**.
- Measured points and projected points must be reported **separately and labelled**.
- The fallback to i.i.d. Gaussian vectors must be **loud** in the output — the two vector sources are not interchangeable.
- CI gate must pass as one process: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live"`.
- Ruff line length is 100. E7 rules are on: no semicolons in test fakes, imports at file top.

## File Structure

| File | Responsibility |
|---|---|
| `scripts/vector_crossover.py` (create) | The whole harness: pure helpers, fixture builders, query shapes, timing, sweep orchestration, report. Single file by spec §10 ("self-contained"), matching `scripts/risk_curve.py` and `scripts/dispatch_sim.py`. |
| `pyproject.toml` (modify, `[tool.pytest.ini_options]`) | Add `pythonpath = ["."]` so tests can import `scripts.vector_crossover`. No test imports from `scripts/` today and there is no `pythonpath` set, so without this the unit tests cannot reach the code. |
| `tests/unit/test_vector_crossover.py` (create) | Pure logic: vector pool provenance and the loud fallback, recall arithmetic, the disk guard. |
| `tests/integration/test_vector_crossover.py` (create) | Container-backed behaviour: the control holds, and brute force really returns true nearest neighbours in order. |
| `docs/superpowers/vector-crossover-2026-09-16.md` (create, Task 6) | Results and the recommendation. |

`scripts/` has no `__init__.py` and does not need one — Python 3 implicit namespace packages make `from scripts.vector_crossover import ...` work once `pythonpath = ["."]` is set. Verified 2026-09-16.

---

### Task 1: Pure helpers — vectors, recall, disk guard

Everything in this task is a pure function with no database. It is the foundation the rest reads.

**Files:**
- Create: `scripts/vector_crossover.py`
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]`, add `pythonpath`)
- Test: `tests/unit/test_vector_crossover.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `VectorPool` dataclass: `.vectors: list[list[float]]`, `.provenance: str`, `.is_fallback: bool`
  - `build_pool(n: int, dim: int, rng: random.Random, seeds: list[list[float]] | None = None, noise: float = 0.15) -> VectorPool`
  - `recall_at_10(brute_uuids: list[str], index_uuids: list[str]) -> float`
  - `rank1_agrees(brute_uuids: list[str], index_uuids: list[str]) -> bool`
  - `estimated_bytes(n_edges: int, dim: int = 768) -> int`
  - `free_bytes(path: str = "/var/lib/docker") -> int`
  - `fits(n_edges: int, free: int, headroom_bytes: int = 5 * 1024**3, dim: int = 768) -> bool`

- [ ] **Step 1: Add `pythonpath` so tests can import the script**

In `pyproject.toml`, the `[tool.pytest.ini_options]` block currently reads:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = ["live: hits the real DocExtractor cluster (opt-in)"]
addopts = "-m 'not live'"
testpaths = ["tests"]
```

Change it to:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = ["live: hits the real DocExtractor cluster (opt-in)"]
addopts = "-m 'not live'"
testpaths = ["tests"]
# `scripts/` is not a package in `packages`, so tests cannot import it without
# this. Python 3 implicit namespace packages make `scripts.vector_crossover`
# resolve with no `__init__.py`.
pythonpath = ["."]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/test_vector_crossover.py`:

```python
"""Pure helpers of the vector-index crossover harness.

The harness reports numbers that a build/no-build decision rests on, so its
failure mode is "reports something that is not what it claims". These tests
pin the claims: that a fallback to synthetic vectors is visible rather than
silent, that recall is scored against the right ground truth, and that the
disk guard refuses a step before it fills the root filesystem rather than
after.
"""
from __future__ import annotations

import random

from scripts.vector_crossover import (
    VectorPool,
    build_pool,
    estimated_bytes,
    fits,
    rank1_agrees,
    recall_at_10,
)


def test_a_pool_built_without_seeds_announces_that_it_is_a_fallback():
    """i.i.d. vectors are near-equidistant in 768 dimensions, which is the worst
    case for an index and produces recall numbers that do NOT transfer to real
    embeddings. A run that silently fell back would publish those numbers as if
    they were comparable."""
    pool = build_pool(n=5, dim=8, rng=random.Random(0), seeds=None)
    assert pool.is_fallback is True
    assert "FALLBACK" in pool.provenance
    assert len(pool.vectors) == 5
    assert all(len(v) == 8 for v in pool.vectors)


def test_a_pool_built_from_seeds_is_not_a_fallback_and_says_what_it_used():
    seeds = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    pool = build_pool(n=4, dim=4, rng=random.Random(0), seeds=seeds)
    assert pool.is_fallback is False
    assert "2" in pool.provenance, "the provenance must name how many seeds it resampled"
    assert len(pool.vectors) == 4


def test_pool_vectors_are_unit_length():
    """Cosine similarity is scale-free, but unit vectors keep the planted-distance
    fixtures in the integration tests meaningful."""
    pool = build_pool(n=3, dim=16, rng=random.Random(1), seeds=None)
    for v in pool.vectors:
        norm = sum(x * x for x in v) ** 0.5
        assert abs(norm - 1.0) < 1e-9


def test_a_pool_is_reproducible_for_a_given_seed():
    """A measurement nobody can re-run is not evidence."""
    a = build_pool(n=4, dim=8, rng=random.Random(7), seeds=None)
    b = build_pool(n=4, dim=8, rng=random.Random(7), seeds=None)
    assert a.vectors == b.vectors


def test_resampling_stays_near_its_seed():
    """Resampling exists to preserve cluster structure. A vector that wandered off
    its seed would be i.i.d. by another name, and the fallback flag would then be
    lying about which distribution was measured."""
    seeds = [[1.0, 0.0, 0.0, 0.0]]
    pool = build_pool(n=20, dim=4, rng=random.Random(3), seeds=seeds, noise=0.05)
    for v in pool.vectors:
        cosine = sum(a * b for a, b in zip(v, seeds[0]))
        assert cosine > 0.9, f"resampled vector drifted off its seed: cosine {cosine}"


def test_recall_is_scored_against_brute_force_as_ground_truth():
    brute = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    assert recall_at_10(brute, brute) == 1.0
    assert recall_at_10(brute, ["a", "b", "c", "d", "e", "x", "y", "z", "p", "q"]) == 0.5
    assert recall_at_10(brute, []) == 0.0


def test_recall_ignores_order_but_rank1_does_not():
    """recall@10 is a set measure; whether the single best neighbour was found is a
    different question and dedup cares about it, so both are reported."""
    brute = ["a", "b", "c"]
    assert recall_at_10(brute, ["c", "b", "a"]) == 1.0
    assert rank1_agrees(brute, ["c", "b", "a"]) is False
    assert rank1_agrees(brute, ["a", "c", "b"]) is True
    assert rank1_agrees([], []) is False


def test_recall_over_fetch_scores_only_the_top_ten():
    """At k=50 the index returns 50 candidates; the harness keeps the best 10 and
    scores THOSE. Scoring all 50 against a 10-item truth would report inflated
    recall that no caller would ever see."""
    brute = [f"t{i}" for i in range(10)]
    index_50 = brute[:10] + [f"x{i}" for i in range(40)]
    assert recall_at_10(brute, index_50) == 1.0
    index_50_bad = [f"x{i}" for i in range(10)] + brute
    assert recall_at_10(brute, index_50_bad) == 0.0, \
        "the true neighbours sat below rank 10, so a caller taking 10 would miss them"


def test_the_disk_guard_refuses_a_step_that_would_not_fit():
    """1M edges x 768 float64 is ~6 GB of property store before the index and the
    transaction logs. Filling the root filesystem mid-sweep would take the Docker
    daemon with it."""
    one_million = estimated_bytes(1_000_000)
    assert one_million > 6 * 1024**3, "estimate must include store overhead, not just raw floats"
    assert fits(1_000_000, free=one_million + 6 * 1024**3) is True
    assert fits(1_000_000, free=one_million) is False, "no headroom left for the index"
    assert fits(10_000, free=20 * 1024**3) is True
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_vector_crossover.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'scripts.vector_crossover'`

- [ ] **Step 4: Write the implementation**

Create `scripts/vector_crossover.py`:

```python
"""Measure what a Neo4j vector index buys over graphiti's brute-force cosine scan.

Phase B step 1 (`docs/superpowers/specs/2026-09-16-vector-index-crossover-design.md`).
Measurement only: this implements no `SearchInterface` and changes nothing under
`src/`.

Why it exists: the live graph holds 3,469 facts and the corpus projects to ~5.3M,
so every claim about the index at scale is currently a linear extrapolation three
orders of magnitude past the measured range.

There is no crossover to find -- at 3,096 facts the index already won (255 ms
brute vs 144 ms indexed). The open question is whether its curve stays FLAT while
brute force climbs at the measured 0.053 ms/fact. That needs range, not size.

`SEARCH` does not exist on Neo4j 2026.07.1. The server deprecates
`db.index.vector.queryRelationships` "in favour of SEARCH", but SEARCH is not
valid Cypher under either CYPHER 5 or CYPHER 25, and SHOW PROCEDURES offers only
the two procedures. Probed 2026-09-16. The deprecated procedures are what works.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

DIM = 768
BYTES_PER_FLOAT = 8
# Neo4j's property store costs more than the raw floats: block headers, the
# dynamic array record chain, and the copy the page cache holds. 2x is a
# deliberately conservative multiplier -- the guard's job is to refuse early.
STORE_OVERHEAD = 2.0


@dataclass(frozen=True)
class VectorPool:
    """Vectors plus WHERE THEY CAME FROM.

    Provenance travels with the data because the two sources are not
    interchangeable: resampled vectors keep the clustering of real embeddings,
    i.i.d. ones are near-equidistant in 768 dimensions, which is the worst case
    for HNSW traversal and would make the index look worse than reality.
    """

    vectors: list[list[float]]
    provenance: str
    is_fallback: bool


def _normalise(v: list[float]) -> list[float]:
    norm = sum(x * x for x in v) ** 0.5
    if norm == 0.0:
        return v
    return [x / norm for x in v]


def build_pool(
    n: int,
    dim: int,
    rng: random.Random,
    seeds: list[list[float]] | None = None,
    noise: float = 0.15,
) -> VectorPool:
    """`n` unit vectors, resampled from `seeds` with Gaussian noise when given.

    Without seeds this falls back to i.i.d. Gaussian and SAYS SO -- a silent
    fallback would publish recall numbers that do not transfer to real
    embeddings as though they did.
    """
    if seeds:
        vectors = [
            _normalise([x + rng.gauss(0.0, noise) for x in seeds[rng.randrange(len(seeds))]])
            for _ in range(n)
        ]
        return VectorPool(
            vectors,
            f"resampled from {len(seeds)} real fact_embedding values (noise sigma={noise})",
            False,
        )
    vectors = [_normalise([rng.gauss(0.0, 1.0) for _ in range(dim)]) for _ in range(n)]
    return VectorPool(
        vectors,
        "i.i.d. Gaussian -- FALLBACK, the live graph was unreachable. Latency "
        "conclusions hold; RECALL NUMBERS ARE NOT COMPARABLE to real embeddings.",
        True,
    )


def recall_at_10(brute_uuids: list[str], index_uuids: list[str]) -> float:
    """Share of brute force's true top-10 that the index's top-10 also contains.

    Both sides are cut to 10 deliberately. When the index is asked for 50
    candidates the caller still keeps 10, so scoring all 50 against a 10-item
    truth would report a recall no caller ever experiences.
    """
    truth = brute_uuids[:10]
    if not truth:
        return 0.0
    return len(set(truth) & set(index_uuids[:10])) / len(truth)


def rank1_agrees(brute_uuids: list[str], index_uuids: list[str]) -> bool:
    """Did the index find the single nearest neighbour? Dedup cares about this
    separately from set recall."""
    return bool(brute_uuids) and bool(index_uuids) and brute_uuids[0] == index_uuids[0]


def estimated_bytes(n_edges: int, dim: int = DIM) -> int:
    return int(n_edges * dim * BYTES_PER_FLOAT * STORE_OVERHEAD)


def free_bytes(path: str = "/var/lib/docker") -> int:
    """Free space where the container's store will land. Falls back to `/` when
    the Docker root is not readable from here."""
    target = path if os.path.exists(path) else "/"
    st = os.statvfs(target)
    return st.f_bavail * st.f_frsize


def fits(
    n_edges: int,
    free: int,
    headroom_bytes: int = 5 * 1024**3,
    dim: int = DIM,
) -> bool:
    """Whether a step fits with room for the HNSW index and transaction logs.

    The headroom is not politeness: filling the root filesystem mid-sweep takes
    the Docker daemon down with it.
    """
    return estimated_bytes(n_edges, dim) + headroom_bytes <= free
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_vector_crossover.py -q`
Expected: `10 passed`

- [ ] **Step 6: Run the lint and type gates**

Run: `uv run ruff check src tests scripts && uv run mypy src`
Expected: `All checks passed!` and `Success: no issues found in 77 source files`

Note: CI lints `src tests` only; `scripts` is included here as hygiene because this harness is new.

- [ ] **Step 7: Commit**

```bash
git add scripts/vector_crossover.py tests/unit/test_vector_crossover.py pyproject.toml
git commit -m "feat(scripts): vector-crossover pure helpers — vectors, recall, disk guard"
```

---

### Task 2: Fixture builder and index lifecycle

Builds the synthetic graphs the sweep measures against, in a container.

**Files:**
- Modify: `scripts/vector_crossover.py`
- Test: `tests/integration/test_vector_crossover.py`

**Interfaces:**
- Consumes: `VectorPool`, `build_pool` (Task 1)
- Produces:
  - `EDGE_INDEX = "cx_fact_vec"`, `NODE_INDEX = "cx_name_vec"` (index name constants)
  - `edge_index_ddl(name: str, dim: int) -> str`
  - `node_index_ddl(name: str, dim: int) -> str`
  - `async ensure_uuid_index(driver) -> None`
  - `async build_edge_fixture(driver, group: str, n_edges: int, pool: VectorPool, node_pool: int, batch: int = 1000) -> None`
  - `async build_node_fixture(driver, group: str, n_nodes: int, pool: VectorPool, batch: int = 1000) -> None`
  - `async create_index(driver, ddl: str) -> None` (creates and awaits)
  - `async drop_index(driver, name: str) -> None`
  - `async wipe(driver, group: str) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/integration/test_vector_crossover.py`:

```python
"""Container-backed behaviour of the crossover harness.

These run against a real neo4j:2026.07.1-community testcontainer because the
two things they check are properties of the SERVER, not of our Python: that
Neo4j does not consult a vector index from graphiti's query shape, and that a
brute-force cosine scan really does rank true nearest neighbours correctly.
The whole measurement rests on both.
"""
from __future__ import annotations

import random

import pytest

from scripts.vector_crossover import (
    EDGE_INDEX,
    build_edge_fixture,
    build_pool,
    create_index,
    edge_index_ddl,
    ensure_uuid_index,
    wipe,
)

pytestmark = pytest.mark.asyncio(loop_scope="module")

GROUP = "__crossover_test__"
DIM = 16


async def test_the_fixture_builds_the_requested_number_of_edges(extract_driver):
    await wipe(extract_driver, GROUP)
    await ensure_uuid_index(extract_driver)
    pool = build_pool(n=200, dim=DIM, rng=random.Random(0), seeds=None)
    await build_edge_fixture(extract_driver, GROUP, 200, pool, node_pool=20, batch=50,
                             rng=random.Random(100))

    r = await extract_driver.execute_query(
        "MATCH ()-[e:RELATES_TO {group_id:$g}]->() RETURN count(e) AS n", g=GROUP)
    assert r.records[0]["n"] == 200
    r = await extract_driver.execute_query(
        "MATCH ()-[e:RELATES_TO {group_id:$g}]->() "
        "WHERE e.fact_embedding IS NULL RETURN count(e) AS n", g=GROUP)
    assert r.records[0]["n"] == 0, "every edge must carry a vector or the scan measures nothing"
    await wipe(extract_driver, GROUP)


async def test_the_vector_index_comes_online_before_it_is_measured(extract_driver):
    """A query against a still-populating index returns fewer rows and would be
    timed as fast. create_index must not return until the index is ONLINE."""
    await wipe(extract_driver, GROUP)
    await ensure_uuid_index(extract_driver)
    pool = build_pool(n=300, dim=DIM, rng=random.Random(1), seeds=None)
    await build_edge_fixture(extract_driver, GROUP, 300, pool, node_pool=30, batch=100,
                             rng=random.Random(101))
    await create_index(extract_driver, edge_index_ddl(EDGE_INDEX, DIM))

    r = await extract_driver.execute_query(
        "SHOW INDEXES YIELD name, state WHERE name = $n RETURN state", n=EDGE_INDEX)
    assert r.records[0]["state"] == "ONLINE"
    await wipe(extract_driver, GROUP)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/integration/test_vector_crossover.py -q`
Expected: collection error — `ImportError: cannot import name 'EDGE_INDEX' from 'scripts.vector_crossover'`

- [ ] **Step 3: Write the implementation**

Append to `scripts/vector_crossover.py`:

```python
EDGE_INDEX = "cx_fact_vec"
NODE_INDEX = "cx_name_vec"

# Vector-index options are DDL, and Cypher does not accept parameters in an
# OPTIONS map -- the dimension has to be formatted in. int() is the guard: the
# value is never caller-controlled in this harness, and it stays un-interpolable
# if that ever changes.
def edge_index_ddl(name: str, dim: int) -> str:
    return (
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
        "FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: "
        f"{int(dim)}, `vector.similarity_function`: 'cosine'}}}}"
    )


def node_index_ddl(name: str, dim: int) -> str:
    return (
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
        "FOR (n:Entity) ON (n.name_embedding) "
        "OPTIONS {indexConfig: {`vector.dimensions`: "
        f"{int(dim)}, `vector.similarity_function`: 'cosine'}}}}"
    )


async def ensure_uuid_index(driver) -> None:
    """Without this the edge builder's MATCH-by-uuid is a label scan per batch,
    making fixture construction O(n^2) and the 1M step unreachable."""
    await driver.execute_query(
        "CREATE INDEX cx_entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)")
    await driver.execute_query("CALL db.awaitIndexes(120)")


async def wipe(driver, group: str) -> None:
    """Delete one group's fixture in bounded batches.

    Batched, not one DETACH DELETE: a million-node delete in a single transaction
    exhausts the heap, and this runs between every sweep step.

    NOT `CALL {...} IN TRANSACTIONS`, which is the usual idiom for this: it is
    illegal inside an explicit transaction, and `driver.execute_query` always
    opens one. It would fail with "Cannot use CALL { ... } IN TRANSACTIONS in an
    explicit transaction". A LIMIT loop needs no autocommit session and works the
    same on either Cypher language version.
    """
    while True:
        result = await driver.execute_query(
            "MATCH (n:Entity {group_id:$g}) WITH n LIMIT 10000 "
            "DETACH DELETE n RETURN count(n) AS n", g=group)
        if not result.records or result.records[0]["n"] == 0:
            return


async def build_node_fixture(
    driver, group: str, n_nodes: int, pool: VectorPool, batch: int = 1000
) -> None:
    for start in range(0, n_nodes, batch):
        rows = [
            {"uuid": f"n{i}", "vec": pool.vectors[i % len(pool.vectors)]}
            for i in range(start, min(start + batch, n_nodes))
        ]
        await driver.execute_query(
            "UNWIND $rows AS row "
            "CREATE (:Entity {uuid: row.uuid, group_id: $g, name_embedding: row.vec})",
            rows=rows, g=group)


async def build_edge_fixture(
    driver, group: str, n_edges: int, pool: VectorPool, node_pool: int, batch: int = 1000
) -> None:
    """`n_edges` RELATES_TO edges over a pool of `node_pool` entities.

    Facts greatly outnumber entities in the real corpus (41.8 facts/article
    against far fewer entities), so a small node pool with many edges is the
    faithful shape -- and it keeps the fixture's node count off the critical path.
    """
    for start in range(0, node_pool, batch):
        rows = [{"uuid": f"e{i}"} for i in range(start, min(start + batch, node_pool))]
        await driver.execute_query(
            "UNWIND $rows AS row CREATE (:Entity {uuid: row.uuid, group_id: $g})",
            rows=rows, g=group)
    for start in range(0, n_edges, batch):
        rows = [
            {
                "uuid": f"r{i}",
                "src": f"e{i % node_pool}",
                "dst": f"e{(i + 1) % node_pool}",
                "vec": pool.vectors[i % len(pool.vectors)],
            }
            for i in range(start, min(start + batch, n_edges))
        ]
        await driver.execute_query(
            "UNWIND $rows AS row "
            "MATCH (a:Entity {uuid: row.src, group_id: $g}), "
            "      (b:Entity {uuid: row.dst, group_id: $g}) "
            "CREATE (a)-[:RELATES_TO {uuid: row.uuid, group_id: $g, "
            "                         fact_embedding: row.vec}]->(b)",
            rows=rows, g=group)


async def create_index(driver, ddl: str) -> None:
    """Create and WAIT. A query against a populating index returns fewer rows and
    would be timed as fast -- the exact way a measurement lies."""
    await driver.execute_query(ddl)
    await driver.execute_query("CALL db.awaitIndexes(600)")


async def drop_index(driver, name: str) -> None:
    await driver.execute_query(f"DROP INDEX {name} IF EXISTS")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --extra dev pytest tests/integration/test_vector_crossover.py -q`
Expected: `2 passed` (first run pulls/starts a container; allow ~60s)

- [ ] **Step 5: Commit**

```bash
git add scripts/vector_crossover.py tests/integration/test_vector_crossover.py
git commit -m "feat(scripts): crossover fixture builder and vector-index lifecycle"
```

---

### Task 3: The three measured queries

The heart of the measurement. Contains the two tests the spec names as its correctness guards.

**Files:**
- Modify: `scripts/vector_crossover.py`
- Test: `tests/integration/test_vector_crossover.py`

**Interfaces:**
- Consumes: everything from Tasks 1-2
- Produces:
  - `BRUTE_EDGE`, `INDEX_EDGE`, `BRUTE_NODE`, `INDEX_NODE` (Cypher string constants)
  - `async time_query(driver, cypher: str, runs: int = 5, **params) -> tuple[float, list[str]]` — returns `(median_ms, uuids_from_last_run)`

- [ ] **Step 1: Write the failing tests**

Append to `tests/integration/test_vector_crossover.py`:

```python
async def test_the_index_does_not_change_graphitis_own_query(extract_driver):
    """THE CONTROL. Neo4j consults a vector index only through
    db.index.vector.query*, never from a bare vector.similarity.cosine in a WITH.
    So graphiti's own query must return identical rows with and without the index.

    If this ever fails, every latency number the harness reports is suspect: an
    apparent speed-up could be the index, cache warmth, or a changed plan, and
    nothing in the output would distinguish them.
    """
    await wipe(extract_driver, GROUP)
    await ensure_uuid_index(extract_driver)
    await drop_index(extract_driver, EDGE_INDEX)
    pool = build_pool(n=300, dim=DIM, rng=random.Random(2), seeds=None)
    await build_edge_fixture(extract_driver, GROUP, 300, pool, node_pool=30, batch=100,
                             rng=random.Random(102))

    probe = build_pool(n=1, dim=DIM, rng=random.Random(99), seeds=None).vectors[0]
    params = dict(g=GROUP, v=probe, min=-1.0, k=10)

    _, before = await time_query(extract_driver, BRUTE_EDGE, runs=1, **params)
    await create_index(extract_driver, edge_index_ddl(EDGE_INDEX, DIM))
    _, after = await time_query(extract_driver, BRUTE_EDGE, runs=1, **params)

    assert before == after, (
        "graphiti's brute-force query changed when the index appeared; the "
        "harness can no longer attribute a latency change to the index")
    assert len(before) == 10
    await drop_index(extract_driver, EDGE_INDEX)
    await wipe(extract_driver, GROUP)


async def test_brute_force_ranks_planted_neighbours_correctly(extract_driver):
    """GROUND TRUTH. Every recall number is scored against brute force, so if
    brute force itself mis-ranks, recall is measured against a wrong answer and
    would look like an index problem.

    Three vectors are planted at known decreasing cosine to the probe.
    """
    await wipe(extract_driver, GROUP)
    await ensure_uuid_index(extract_driver)
    probe = [1.0] + [0.0] * (DIM - 1)
    planted = {
        "near": [1.0, 0.05] + [0.0] * (DIM - 2),
        "mid": [1.0, 0.60] + [0.0] * (DIM - 2),
        "far": [1.0, 3.00] + [0.0] * (DIM - 2),
    }
    await extract_driver.execute_query(
        "CREATE (:Entity {uuid:'p0', group_id:$g}), (:Entity {uuid:'p1', group_id:$g})",
        g=GROUP)
    for uuid, vec in planted.items():
        norm = sum(x * x for x in vec) ** 0.5
        await extract_driver.execute_query(
            "MATCH (a:Entity {uuid:'p0', group_id:$g}), (b:Entity {uuid:'p1', group_id:$g}) "
            "CREATE (a)-[:RELATES_TO {uuid:$u, group_id:$g, fact_embedding:$v}]->(b)",
            g=GROUP, u=uuid, v=[x / norm for x in vec])

    _, uuids = await time_query(
        extract_driver, BRUTE_EDGE, runs=1, g=GROUP, v=probe, min=-1.0, k=10)
    assert uuids == ["near", "mid", "far"], (
        f"brute force mis-ranked known neighbours; got {uuids}")
    await wipe(extract_driver, GROUP)


async def test_the_index_query_agrees_with_brute_force_on_a_small_fixture(extract_driver):
    """At this size ANN has no room to be wrong, so disagreement means the index
    query is filtering or projecting differently -- a harness bug, not a recall
    finding. Catching it here stops it being reported as recall loss at 1M."""
    await wipe(extract_driver, GROUP)
    await ensure_uuid_index(extract_driver)
    await drop_index(extract_driver, EDGE_INDEX)
    pool = build_pool(n=100, dim=DIM, rng=random.Random(4), seeds=None)
    await build_edge_fixture(extract_driver, GROUP, 100, pool, node_pool=10, batch=50,
                             rng=random.Random(103))
    await create_index(extract_driver, edge_index_ddl(EDGE_INDEX, DIM))

    probe = build_pool(n=1, dim=DIM, rng=random.Random(5), seeds=None).vectors[0]
    _, brute = await time_query(
        extract_driver, BRUTE_EDGE, runs=1, g=GROUP, v=probe, min=-1.0, k=10)
    _, indexed = await time_query(
        extract_driver, INDEX_EDGE, runs=1, g=GROUP, v=probe, min=-1.0, k=10,
        idx=EDGE_INDEX)
    assert recall_at_10(brute, indexed) == 1.0, (
        f"index and brute force disagree on a 100-edge fixture: "
        f"brute={brute} index={indexed}")
    await drop_index(extract_driver, EDGE_INDEX)
    await wipe(extract_driver, GROUP)
```

Update that file's import block to add the new names:

```python
from scripts.vector_crossover import (
    BRUTE_EDGE,
    EDGE_INDEX,
    INDEX_EDGE,
    build_edge_fixture,
    build_pool,
    create_index,
    drop_index,
    edge_index_ddl,
    ensure_uuid_index,
    recall_at_10,
    time_query,
    wipe,
)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/integration/test_vector_crossover.py -q`
Expected: collection error — `ImportError: cannot import name 'BRUTE_EDGE'`

- [ ] **Step 3: Write the implementation**

Append to `scripts/vector_crossover.py` (add `import statistics` and `import time` to the import block at the top of the file):

```python
# graphiti's actual shape, matching `profile-vector-index.py` and
# `graph_queries.py`. The DISTINCT and the endpoint bindings are kept even though
# they are not the cost (reproduced standalone at 263 ms, BACKLOG 8-orig) --
# measuring a query graphiti does not run would measure nothing.
BRUTE_EDGE = """
MATCH (n:Entity)-[e:RELATES_TO {group_id:$g}]->(m:Entity)
WITH DISTINCT e, n, m, vector.similarity.cosine(e.fact_embedding, $v) AS score
WHERE score > $min
RETURN e.uuid AS uuid, score ORDER BY score DESC LIMIT $k
"""

# The deprecated procedure, deliberately. SEARCH is what 2026.07.1's deprecation
# notice names, but SEARCH is not valid Cypher on this server under either
# language version (probed 2026-09-16) -- the server deprecates a procedure in
# favour of a clause it does not ship.
INDEX_EDGE = """
CALL db.index.vector.queryRelationships($idx, $k, $v) YIELD relationship AS e, score
WHERE e.group_id = $g AND score > $min
RETURN e.uuid AS uuid, score ORDER BY score DESC
"""

BRUTE_NODE = """
MATCH (n:Entity {group_id:$g})
WITH n, vector.similarity.cosine(n.name_embedding, $v) AS score
WHERE score > $min
RETURN n.uuid AS uuid, score ORDER BY score DESC LIMIT $k
"""

INDEX_NODE = """
CALL db.index.vector.queryNodes($idx, $k, $v) YIELD node AS n, score
WHERE n.group_id = $g AND score > $min
RETURN n.uuid AS uuid, score ORDER BY score DESC
"""


async def time_query(driver, cypher: str, runs: int = 5, **params) -> tuple[float, list[str]]:
    """Median wall time in ms over `runs`, plus the uuids the last run returned.

    One warm-up run is executed and DISCARDED. Without it the first measurement
    at each step pays page-cache misses the later ones do not, which reads as the
    index winning when it is really just second.
    """
    await driver.execute_query(cypher, **params)
    times: list[float] = []
    uuids: list[str] = []
    for _ in range(runs):
        started = time.perf_counter()
        result = await driver.execute_query(cypher, **params)
        times.append((time.perf_counter() - started) * 1000.0)
        uuids = [record["uuid"] for record in result.records]
    return statistics.median(times), uuids
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/integration/test_vector_crossover.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/vector_crossover.py tests/integration/test_vector_crossover.py
git commit -m "feat(scripts): the three measured queries, with control and ground-truth tests"
```

---

### Task 4: Recall at three fetch depths, and write overhead

**Files:**
- Modify: `scripts/vector_crossover.py`
- Test: `tests/integration/test_vector_crossover.py`

**Interfaces:**
- Consumes: Tasks 1-3
- Produces:
  - `FETCH_DEPTHS = (10, 50, 200)`
  - `async measure_recall(driver, group, pool, rng, idx, brute_cypher, index_cypher, probes=20) -> dict[int, tuple[float, float]]` — maps fetch depth to `(mean_recall_at_10, rank1_agreement_rate)`
  - `async measure_insert_ms(driver, group, n, pool, node_pool, with_index, dim) -> float`

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/test_vector_crossover.py`:

```python
async def test_recall_is_reported_for_every_fetch_depth(extract_driver):
    """Over-fetching -- ask for 50, keep the best 10 -- is the cheapest recall
    lever there is, and whether it holds recall flat is what decides D2. A run
    that reported only k=10 could not answer that."""
    await wipe(extract_driver, GROUP)
    await ensure_uuid_index(extract_driver)
    await drop_index(extract_driver, EDGE_INDEX)
    pool = build_pool(n=400, dim=DIM, rng=random.Random(6), seeds=None)
    await build_edge_fixture(extract_driver, GROUP, 400, pool, node_pool=40, batch=100,
                             rng=random.Random(104))
    await create_index(extract_driver, edge_index_ddl(EDGE_INDEX, DIM))

    out = await measure_recall(
        extract_driver, GROUP, pool, random.Random(7), EDGE_INDEX,
        BRUTE_EDGE, INDEX_EDGE, probes=3)

    assert set(out) == set(FETCH_DEPTHS)
    for depth, (recall, rank1) in out.items():
        assert 0.0 <= recall <= 1.0, f"depth {depth} recall out of range: {recall}"
        assert 0.0 <= rank1 <= 1.0
    assert out[200][0] >= out[10][0] - 1e-9, (
        "asking the index for MORE candidates cannot lower recall@10; "
        "if it did, the harness is scoring the wrong slice")
    await drop_index(extract_driver, EDGE_INDEX)
    await wipe(extract_driver, GROUP)


async def test_write_overhead_is_measured_with_and_without_the_index(extract_driver):
    """The number that got the 2026-09-11 index dropped -- 'it would add write
    overhead to every fact insert on the path we are trying to speed up' -- and
    which has never actually been measured."""
    pool = build_pool(n=200, dim=DIM, rng=random.Random(8), seeds=None)
    await ensure_uuid_index(extract_driver)

    bare = await measure_insert_ms(
        extract_driver, GROUP, 200, pool, node_pool=20, with_index=False, dim=DIM)
    indexed = await measure_insert_ms(
        extract_driver, GROUP, 200, pool, node_pool=20, with_index=True, dim=DIM)

    assert bare > 0.0 and indexed > 0.0
    r = await extract_driver.execute_query(
        "MATCH ()-[e:RELATES_TO {group_id:$g}]->() RETURN count(e) AS n", g=GROUP)
    assert r.records[0]["n"] == 0, "measure_insert_ms must leave no fixture behind"
```

Add to that file's import block: `FETCH_DEPTHS`, `INDEX_EDGE`, `measure_insert_ms`, `measure_recall`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --extra dev pytest tests/integration/test_vector_crossover.py -q`
Expected: collection error — `ImportError: cannot import name 'FETCH_DEPTHS'`

- [ ] **Step 3: Write the implementation**

Append to `scripts/vector_crossover.py`:

```python
# Ask the index for more than the caller keeps. One parameter, no reindex, no
# schema change -- and if recall@10 holds flat at 50 while it declines at 10,
# decision D2 dissolves and no bounded candidate set is needed.
FETCH_DEPTHS = (10, 50, 200)


async def measure_recall(
    driver,
    group: str,
    pool: VectorPool,
    rng: random.Random,
    idx: str,
    brute_cypher: str,
    index_cypher: str,
    probes: int = 20,
) -> dict[int, tuple[float, float]]:
    """recall@10 and rank-1 agreement per fetch depth, against brute-force truth.

    Probe vectors are drawn from the fixture's own pool, so they sit in the same
    distribution as the data. Probing with vectors from elsewhere would measure
    recall on queries no real caller makes.
    """
    out: dict[int, tuple[float, float]] = {}
    probe_vectors = [pool.vectors[rng.randrange(len(pool.vectors))] for _ in range(probes)]
    truth: list[list[str]] = []
    for vector in probe_vectors:
        _, brute = await time_query(
            driver, brute_cypher, runs=1, g=group, v=vector, min=-1.0, k=10)
        truth.append(brute)
    for depth in FETCH_DEPTHS:
        recalls: list[float] = []
        rank1: list[float] = []
        for vector, brute in zip(probe_vectors, truth):
            _, indexed = await time_query(
                driver, index_cypher, runs=1, g=group, v=vector, min=-1.0, k=depth,
                idx=idx)
            recalls.append(recall_at_10(brute, indexed))
            rank1.append(1.0 if rank1_agrees(brute, indexed) else 0.0)
        out[depth] = (statistics.fmean(recalls), statistics.fmean(rank1))
    return out


async def measure_insert_ms(
    driver,
    group: str,
    n: int,
    pool: VectorPool,
    node_pool: int,
    with_index: bool,
    dim: int = DIM,
) -> float:
    """Wall time in ms to insert `n` edges, with or without the vector index.

    Builds into a scratch group and removes it, so the caller's fixture is
    untouched and successive calls do not accumulate.
    """
    scratch = f"{group}__insert_probe__"
    await wipe(driver, scratch)
    await drop_index(driver, EDGE_INDEX)
    if with_index:
        await create_index(driver, edge_index_ddl(EDGE_INDEX, dim))
    started = time.perf_counter()
    await build_edge_fixture(driver, scratch, n, pool, node_pool=node_pool,
                             rng=random.Random(2026))
    elapsed = (time.perf_counter() - started) * 1000.0
    await wipe(driver, scratch)
    await drop_index(driver, EDGE_INDEX)
    return elapsed
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/integration/test_vector_crossover.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/vector_crossover.py tests/integration/test_vector_crossover.py
git commit -m "feat(scripts): recall at three fetch depths, and index write overhead"
```

---

### Task 5: The sweep, the CLI, and the report

**Files:**
- Modify: `scripts/vector_crossover.py`
- Test: `tests/unit/test_vector_crossover.py`

**Interfaces:**
- Consumes: Tasks 1-4
- Produces:
  - `EDGE_STEPS = (10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)`
  - `NODE_STEPS = (10_000, 50_000, 100_000, 250_000)`
  - `StepResult` dataclass: `.n`, `.brute_ms`, `.index_ms`, `.control_ms`, `.recall`, `.insert_bare_ms`, `.insert_indexed_ms`
  - `format_report(steps: list[StepResult], provenance: str, measured_to: int) -> str`
  - `async fetch_seed_vectors(limit: int = 3469) -> list[list[float]] | None`
  - `main() -> int`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_vector_crossover.py`:

```python
def test_the_report_marks_measured_and_projected_rows_apart():
    """Section 1.2 of the production review marked its own extrapolated rows
    `[inference]` three orders of magnitude out. A report that presented a
    projection as a measurement would repeat the habit this project keeps paying
    for."""
    from scripts.vector_crossover import StepResult, format_report

    steps = [
        StepResult(n=10_000, brute_ms=690.0, index_ms=12.0, control_ms=688.0,
                   recall={10: (1.0, 1.0), 50: (1.0, 1.0), 200: (1.0, 1.0)},
                   insert_bare_ms=900.0, insert_indexed_ms=1400.0),
        StepResult(n=50_000, brute_ms=2810.0, index_ms=14.0, control_ms=2805.0,
                   recall={10: (0.98, 0.95), 50: (1.0, 1.0), 200: (1.0, 1.0)},
                   insert_bare_ms=4400.0, insert_indexed_ms=7000.0),
    ]
    # measured_to is 10,000, so the 50,000 row is past where measurement stopped
    # and must be labelled as a projection.
    out = format_report(steps, provenance="resampled from 3469 real values",
                        measured_to=10_000)
    assert "[measured]" in out
    assert "[inference]" in out, "a row beyond the last measured step must be labelled"
    assert "resampled from 3469 real values" in out, "provenance must travel into the report"
    assert "10,000" in out or "10000" in out


def test_the_report_leads_with_the_fallback_warning_when_vectors_are_synthetic():
    """A reader skimming for the headline number must not miss that recall came
    from i.i.d. vectors."""
    from scripts.vector_crossover import StepResult, format_report

    steps = [StepResult(n=10_000, brute_ms=690.0, index_ms=12.0, control_ms=688.0,
                        recall={10: (1.0, 1.0), 50: (1.0, 1.0), 200: (1.0, 1.0)},
                        insert_bare_ms=900.0, insert_indexed_ms=1400.0)]
    out = format_report(steps, provenance="i.i.d. Gaussian -- FALLBACK", measured_to=10_000)
    assert "FALLBACK" in out.split("\n")[0] or "FALLBACK" in out.split("\n")[1], \
        "the fallback warning must be at the top, not buried below the table"


def test_the_control_column_is_flagged_when_it_diverges():
    """The control is brute force WITH the index present; it must match brute
    force without it. A silent divergence would invalidate every latency row, so
    the report has to shout rather than print two similar numbers."""
    from scripts.vector_crossover import StepResult, format_report

    steps = [StepResult(n=10_000, brute_ms=690.0, index_ms=12.0, control_ms=120.0,
                        recall={10: (1.0, 1.0), 50: (1.0, 1.0), 200: (1.0, 1.0)},
                        insert_bare_ms=900.0, insert_indexed_ms=1400.0)]
    out = format_report(steps, provenance="resampled", measured_to=10_000)
    assert "CONTROL DIVERGED" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_vector_crossover.py -q`
Expected: FAIL — `ImportError: cannot import name 'StepResult'`

- [ ] **Step 3: Write the implementation**

Append to `scripts/vector_crossover.py` (add `import argparse` and `import asyncio` to the import block):

```python
EDGE_STEPS = (10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
NODE_STEPS = (10_000, 50_000, 100_000, 250_000)

# Brute force is linear in rows and the measured live-graph slope is
# 0.053 ms/fact with a ~160 ms intercept (production review section 1.2). Rows
# past the last measured step are projected from THIS run's own fit, not from
# that constant, and are labelled.
CONTROL_TOLERANCE = 0.25


@dataclass(frozen=True)
class StepResult:
    n: int
    brute_ms: float
    index_ms: float
    control_ms: float
    recall: dict[int, tuple[float, float]]
    insert_bare_ms: float
    insert_indexed_ms: float

    @property
    def control_diverged(self) -> bool:
        """The control must match brute force. A gap means the timing is
        measuring cache warmth or a changed plan, not the index."""
        if self.brute_ms <= 0.0:
            return False
        return abs(self.control_ms - self.brute_ms) / self.brute_ms > CONTROL_TOLERANCE


def format_report(steps: list[StepResult], provenance: str, measured_to: int) -> str:
    lines: list[str] = ["# Vector-index crossover", ""]
    if "FALLBACK" in provenance:
        lines.insert(1, f"**WARNING: {provenance}**")
    else:
        lines.append(f"vectors: {provenance}")
    diverged = [s for s in steps if s.control_diverged]
    if diverged:
        lines.append("")
        lines.append(
            "**CONTROL DIVERGED at "
            + ", ".join(f"{s.n:,}" for s in diverged)
            + " — brute force changed when the index appeared, so the latency "
            "rows below cannot be attributed to the index.**")
    lines += ["", "| rows | brute ms | index ms | control ms | r@10 k=10 | k=50 | k=200 | "
              "insert bare ms | insert indexed ms | source |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for step in steps:
        tag = "[measured]" if step.n <= measured_to else "[inference]"
        lines.append(
            f"| {step.n:,} | {step.brute_ms:.1f} | {step.index_ms:.1f} | "
            f"{step.control_ms:.1f} | {step.recall[10][0]:.3f} | "
            f"{step.recall[50][0]:.3f} | {step.recall[200][0]:.3f} | "
            f"{step.insert_bare_ms:.0f} | {step.insert_indexed_ms:.0f} | {tag} |")
    return "\n".join(lines)


async def fetch_seed_vectors(limit: int = 3469) -> list[list[float]] | None:
    """Read real fact embeddings from the live graph, READ-ONLY, as resampling
    seeds. Returns None if unreachable, which sends `build_pool` down its loud
    fallback path rather than failing the run."""
    try:
        from neo4j import AsyncGraphDatabase, RoutingControl

        from graph_extract.config import get_extract_settings
    except ImportError:
        return None
    try:
        settings = get_extract_settings()
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    except Exception:
        return None
    try:
        result = await driver.execute_query(
            "MATCH ()-[e:RELATES_TO {group_id:$g}]->() "
            "WHERE e.fact_embedding IS NOT NULL "
            "RETURN e.fact_embedding AS v LIMIT $n",
            g=settings.group_id, n=limit, routing_=RoutingControl.READ)
        return [list(record["v"]) for record in result.records] or None
    except Exception:
        return None
    finally:
        await driver.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-edges", type=int, default=max(EDGE_STEPS))
    parser.add_argument("--max-nodes", type=int, default=max(NODE_STEPS))
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--out", default="docs/superpowers/vector-crossover-2026-09-16.md")
    args = parser.parse_args()
    return asyncio.run(_run(args))


async def _run(args) -> int:
    from testcontainers.neo4j import Neo4jContainer
    from neo4j import AsyncGraphDatabase

    seeds = await fetch_seed_vectors()
    rng = random.Random(20260916)
    pool = build_pool(n=10_000, dim=DIM, rng=rng, seeds=seeds)
    print(f"vectors: {pool.provenance}", flush=True)

    steps: list[StepResult] = []
    measured_to = 0
    group = "__crossover__"
    with Neo4jContainer("neo4j:2026.07.1-community") as neo:
        driver = AsyncGraphDatabase.driver(
            neo.get_connection_url(), auth=("neo4j", neo.password))
        try:
            await ensure_uuid_index(driver)
            for n in EDGE_STEPS:
                if n > args.max_edges:
                    break
                free = free_bytes()
                if not fits(n, free):
                    print(f"stopping before {n:,}: needs "
                          f"{estimated_bytes(n) / 1024**3:.1f} GB plus headroom, "
                          f"{free / 1024**3:.1f} GB free", flush=True)
                    break
                print(f"--- {n:,} edges ---", flush=True)
                await wipe(driver, group)
                await drop_index(driver, EDGE_INDEX)
                insert_bare = await measure_insert_ms(
                    driver, group, min(n, 50_000), pool, node_pool=1_000,
                    with_index=False)
                insert_indexed = await measure_insert_ms(
                    driver, group, min(n, 50_000), pool, node_pool=1_000,
                    with_index=True)
                await build_edge_fixture(
                    driver, group, n, pool, node_pool=max(1_000, n // 50), rng=rng)
                probe = pool.vectors[rng.randrange(len(pool.vectors))]
                params = dict(g=group, v=probe, min=-1.0, k=10)
                brute_ms, _ = await time_query(driver, BRUTE_EDGE, **params)
                await create_index(driver, edge_index_ddl(EDGE_INDEX, DIM))
                index_ms, _ = await time_query(
                    driver, INDEX_EDGE, idx=EDGE_INDEX, **params)
                control_ms, _ = await time_query(driver, BRUTE_EDGE, **params)
                recall = await measure_recall(
                    driver, group, pool, rng, EDGE_INDEX, BRUTE_EDGE, INDEX_EDGE,
                    probes=args.probes)
                steps.append(StepResult(
                    n=n, brute_ms=brute_ms, index_ms=index_ms, control_ms=control_ms,
                    recall=recall, insert_bare_ms=insert_bare,
                    insert_indexed_ms=insert_indexed))
                measured_to = n
                print(f"  brute {brute_ms:.1f} ms | index {index_ms:.1f} ms | "
                      f"control {control_ms:.1f} ms | r@10 "
                      f"{recall[10][0]:.3f}/{recall[50][0]:.3f}/{recall[200][0]:.3f}",
                      flush=True)
                await drop_index(driver, EDGE_INDEX)
                await wipe(driver, group)
        finally:
            await driver.close()

    report = format_report(steps, pool.provenance, measured_to)
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_vector_crossover.py -q`
Expected: `13 passed`

- [ ] **Step 5: Run the full gate as one process**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live" -q`
Expected: `All checks passed!`, `Success: no issues found in 77 source files`, and a passing run with 7 more tests than before this plan started.

- [ ] **Step 6: Commit**

```bash
git add scripts/vector_crossover.py tests/unit/test_vector_crossover.py
git commit -m "feat(scripts): crossover sweep, cli and report"
```

---

### Task 6: Run the sweep and write the results

This task produces no code. It runs the harness and turns its output into the document the decision is read from.

**Files:**
- Create: `docs/superpowers/vector-crossover-2026-09-16.md`

**Interfaces:**
- Consumes: `scripts/vector_crossover.py` (Tasks 1-5)
- Produces: the results document.

- [ ] **Step 1: Smoke the harness at a small ceiling first**

Run: `uv run --extra dev python scripts/vector_crossover.py --max-edges 10000 --probes 3`
Expected: a `vectors:` provenance line, one `--- 10,000 edges ---` block, and a markdown table. Confirm the provenance line says `resampled from ... real fact_embedding values` — if it says `FALLBACK`, the live graph was unreachable and the recall half of this run is not comparable to real embeddings. Fix connectivity before the full run rather than publishing fallback numbers.

- [ ] **Step 2: Check free disk before committing to the full sweep**

Run: `df -h /var/lib/docker`
Expected: at least 12 GB available. The guard stops the sweep cleanly below that, but starting with headroom avoids a half-finished curve.

- [ ] **Step 3: Run the full sweep in the background**

Run: `uv run --extra dev python scripts/vector_crossover.py > /tmp/crossover.log 2>&1 &`

Expected: runs for tens of minutes. The 1M step builds ~6 GB of store. Watch with `tail -f /tmp/crossover.log`.

- [ ] **Step 4: Write the results document**

Create `docs/superpowers/vector-crossover-2026-09-16.md` containing, in this order:

1. The run's provenance line and the container's page-cache and heap settings (`CALL dbms.listConfig('server.memory') YIELD name, value`), so the numbers can be reproduced or dismissed.
2. The generated table, measured and projected rows labelled as the harness emits them.
3. **The curve shapes**: brute-force ms against rows (expected linear, compare its slope to the live graph's measured 0.053 ms/fact), and index ms against rows (the question is whether it is flat).
4. **The recall trend** across k=10/50/200, and whether over-fetching holds recall flat where k=10 declines. This is the input to decision D2.
5. **The write overhead** as a percentage of the un-indexed insert — the number that got the 2026-09-11 index dropped.
6. **A recommendation** on whether Phase B's implementation half is justified, and at what corpus size it becomes necessary.
7. Anything the run refuted. This project's pattern is that roughly one claim per measurement turns out false; record it rather than quietly dropping it.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/vector-crossover-2026-09-16.md
git commit -m "docs: vector-index crossover results"
```

---

## Self-Review

**Spec coverage.** §4 where it runs → Task 5 `_run` uses `Neo4jContainer("neo4j:2026.07.1-community")`. §5 fixture construction and resampling → Tasks 1-2. §6 the three queries with exact Cypher → Task 3, including the control. §7 the sweep and disk guard → Tasks 1 and 5. §8 recall including the k=10/50/200 over-fetch dial → Task 4. §9 write overhead → Task 4. §10 deliverables → Tasks 5-6. §11 the three named tests → the control and ground-truth tests in Task 3, the loud-fallback test in Task 1. §12 invalidation conditions → Task 6 step 4 item 1 records the container's memory configuration. §13-14 decisions and the production recall proxy → recorded in the spec, no code required.

**Placeholder scan.** No TBD/TODO. Every code step carries complete code; every run step carries an exact command and expected output.

**Type consistency.** `VectorPool` fields (`vectors`, `provenance`, `is_fallback`) are used identically in Tasks 1, 4 and 5. `measure_recall` returns `dict[int, tuple[float, float]]` in Task 4 and `StepResult.recall` consumes that type in Task 5. `time_query` returns `(float, list[str])` in Task 3 and every caller unpacks two values. `EDGE_INDEX` is defined once in Task 2 and imported by name thereafter.

**One known cost, flagged rather than hidden.** `tests/integration/test_vector_crossover.py` is a new module and `extract_neo4j` is module-scoped, so it starts its own container — roughly 20-30 s added to the CI suite. That is the price of testing server behaviour against the real server, which is the only place these two guards mean anything.
