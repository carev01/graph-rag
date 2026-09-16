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

import argparse
import asyncio
import os
import random
import statistics
import time
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
    headroom_bytes: int | None = None,
    dim: int = DIM,
) -> bool:
    """Whether a step fits with room for the HNSW index and transaction logs.

    The headroom SCALES: Neo4j's vector index keeps its own copy of the vectors
    for reranking on top of the navigation graph, so the space it needs grows
    with the data rather than sitting at some constant. A flat reserve would
    shrink to nothing in relative terms exactly as the steps got large enough to
    matter. The 5 GiB floor covers transaction logs on the small steps, where
    half the store estimate is less than the logs will be.

    The headroom is not politeness: filling the root filesystem mid-sweep takes
    the Docker daemon down with it.
    """
    estimate = estimated_bytes(n_edges, dim)
    reserve = max(5 * 1024**3, estimate // 2) if headroom_bytes is None else headroom_bytes
    return estimate + reserve <= free


def resample_one(pool: VectorPool, rng: random.Random, noise: float = 0.15) -> list[float]:
    """One FRESH vector per call, in the pool's distribution.

    The fixture needs a distinct vector per row -- indexing into a fixed pool
    makes rows byte-identical once the row count passes the pool size, and an
    HNSW index over duplicated points is not the index the sweep claims to
    measure. Drawing per row keeps 1M vectors out of memory: each is built,
    written and discarded.

    Non-fallback pools treat their vectors as SEEDS and jitter around them, so
    the result keeps real embeddings' cluster structure -- which is also the more
    faithful corpus model, since facts cluster by topic rather than scattering.
    """
    if pool.is_fallback:
        return _normalise([rng.gauss(0.0, 1.0) for _ in range(len(pool.vectors[0]))])
    seed = pool.vectors[rng.randrange(len(pool.vectors))]
    return _normalise([x + rng.gauss(0.0, noise) for x in seed])


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
    driver, group: str, n_nodes: int, pool: VectorPool, batch: int = 1000, *,
    rng: random.Random,
) -> None:
    for start in range(0, n_nodes, batch):
        rows = [
            {"uuid": f"n{i}", "vec": resample_one(pool, rng)}
            for i in range(start, min(start + batch, n_nodes))
        ]
        await driver.execute_query(
            "UNWIND $rows AS row "
            "CREATE (:Entity {uuid: row.uuid, group_id: $g, name_embedding: row.vec})",
            rows=rows, g=group)


async def build_edge_fixture(
    driver, group: str, n_edges: int, pool: VectorPool, node_pool: int, batch: int = 1000, *,
    rng: random.Random,
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
                "vec": resample_one(pool, rng),
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


async def create_index(driver, ddl: str, name: str) -> None:
    """Create and WAIT, then VERIFY.

    `CALL db.awaitIndexes(600)` returns on timeout without raising -- a build
    that outruns the timeout in the real sweep would return silently with a
    non-ONLINE index, and the timing run would measure a partially built index
    as fast. Checking `SHOW INDEXES` afterwards turns that silent lie into a
    raised error.
    """
    await driver.execute_query(ddl)
    await driver.execute_query("CALL db.awaitIndexes(600)")
    r = await driver.execute_query(
        "SHOW INDEXES YIELD name, state WHERE name = $n RETURN state", n=name)
    state = r.records[0]["state"] if r.records else "MISSING"
    if state != "ONLINE":
        raise RuntimeError(f"index {name!r} is {state}, not ONLINE after awaitIndexes")


async def drop_index(driver, name: str) -> None:
    await driver.execute_query(f"DROP INDEX {name} IF EXISTS")


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
    untouched and successive calls do not accumulate. Owns EDGE_INDEX for its
    duration and refuses to run if one already exists.
    """
    scratch = f"{group}__insert_probe__"
    await wipe(driver, scratch)
    # This function owns EDGE_INDEX for its duration and drops it on the way out.
    # Refuse rather than destroy one a caller built: a silently vanished index
    # would be timed as a scan and read as the index being worthless.
    existing = await driver.execute_query(
        "SHOW INDEXES YIELD name WHERE name = $n RETURN count(*) AS n", n=EDGE_INDEX)
    if existing.records[0]["n"]:
        raise RuntimeError(
            f"{EDGE_INDEX} already exists; measure_insert_ms would destroy it. "
            f"Drop it before measuring insert cost.")
    if with_index:
        await create_index(driver, edge_index_ddl(EDGE_INDEX, dim), EDGE_INDEX)
    started = time.perf_counter()
    await build_edge_fixture(driver, scratch, n, pool, node_pool=node_pool,
                             rng=random.Random(2026))
    elapsed = (time.perf_counter() - started) * 1000.0
    await wipe(driver, scratch)
    await drop_index(driver, EDGE_INDEX)
    return elapsed


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
    # Write overhead stays edge-only (spec section 9 is about fact inserts). Node
    # rows carry None here so the report renders a dash, never a zero.
    insert_bare_ms: float | None
    insert_indexed_ms: float | None
    insert_rows: int | None = None
    kind: str = "edge"

    @property
    def control_diverged(self) -> bool:
        """The control must match brute force. A gap means the timing is
        measuring cache warmth or a changed plan, not the index."""
        if self.brute_ms <= 0.0:
            return False
        return abs(self.control_ms - self.brute_ms) / self.brute_ms > CONTROL_TOLERANCE


KIND_HEADINGS = {
    "edge": "Edges (RELATES_TO.fact_embedding)",
    "node": "Nodes (Entity.name_embedding)",
}


def _ms_per_1k(ms: float | None, rows: int | None) -> str:
    """ms per 1,000 inserts, spec section 9's required unit -- raw ms would
    compare a fixed row count against a growing-N curve on the same line."""
    if ms is None or not rows:
        return "—"
    return f"{ms / rows * 1000:.1f}"


def _overhead_pct(bare_ms: float | None, indexed_ms: float | None) -> str:
    """Index write overhead as a percentage of the un-indexed insert, per spec
    section 9 -- computed from the two insert figures, never from raw ms."""
    if bare_ms is None or indexed_ms is None or bare_ms == 0.0:
        return "—"
    return f"{(indexed_ms - bare_ms) / bare_ms * 100:.1f}"


def format_report(
    steps: list[StepResult], provenance: str, measured_to: dict[str, int],
) -> str:
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
            + ", ".join(f"{s.n:,} ({s.kind})" for s in diverged)
            + " — brute force changed when the index appeared, so the latency "
            "rows below cannot be attributed to the index.**")

    by_kind: dict[str, list[StepResult]] = {}
    for step in steps:
        by_kind.setdefault(step.kind, []).append(step)

    for kind, kind_steps in by_kind.items():
        lines += ["", f"## {KIND_HEADINGS.get(kind, kind)}", ""]
        lines += [
            "| rows | brute ms | index ms | control ms | r@10 k=10 | k=50 | k=200 | "
            "insert bare ms/1k | insert +idx ms/1k | index write overhead % | source |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        kind_measured_to = measured_to.get(kind, 0)
        for step in kind_steps:
            tag = "[measured]" if step.n <= kind_measured_to else "[inference]"
            lines.append(
                f"| {step.n:,} | {step.brute_ms:.1f} | {step.index_ms:.1f} | "
                f"{step.control_ms:.1f} | {step.recall[10][0]:.3f} | "
                f"{step.recall[50][0]:.3f} | {step.recall[200][0]:.3f} | "
                f"{_ms_per_1k(step.insert_bare_ms, step.insert_rows)} | "
                f"{_ms_per_1k(step.insert_indexed_ms, step.insert_rows)} | "
                f"{_overhead_pct(step.insert_bare_ms, step.insert_indexed_ms)} | {tag} |")
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


async def _build_edges(driver, group: str, n: int, pool: VectorPool, rng: random.Random) -> None:
    """Adapts `build_edge_fixture` to `_sweep`'s `build` shape. Node pool grows
    with N (facts greatly outnumber entities in the real corpus)."""
    await build_edge_fixture(driver, group, n, pool, node_pool=max(1_000, n // 50), rng=rng)


async def _build_nodes(driver, group: str, n: int, pool: VectorPool, rng: random.Random) -> None:
    await build_node_fixture(driver, group, n, pool, rng=rng)


async def _sweep(
    driver,
    group: str,
    kind: str,
    steps: tuple[int, ...],
    max_n: int,
    pool: VectorPool,
    rng: random.Random,
    probes: int,
    brute_cypher: str,
    index_cypher: str,
    index_name: str,
    ddl,
    build,
    measure_inserts: bool,
) -> tuple[list[StepResult], int]:
    """One kind's sweep (edges or nodes), shared so the two sweeps do not
    duplicate the per-step body.

    Timing order per step is load-bearing: brute, THEN create the index, THEN
    index, THEN brute again as `control_ms` -- the second brute run is the guard
    against reporting page-cache warmth as the index's win. `drop_index` runs
    before every `measure_insert_ms` call because that function now raises if
    its index already exists.
    """
    results: list[StepResult] = []
    measured_to = 0
    for n in steps:
        if n > max_n:
            break
        free = free_bytes()
        if not fits(n, free):
            print(f"stopping before {n:,} {kind}s: needs "
                  f"{estimated_bytes(n) / 1024**3:.1f} GB plus headroom, "
                  f"{free / 1024**3:.1f} GB free", flush=True)
            break
        print(f"--- {n:,} {kind}s ---", flush=True)
        await wipe(driver, group)
        await drop_index(driver, index_name)

        insert_bare: float | None = None
        insert_indexed: float | None = None
        insert_rows: int | None = None
        if measure_inserts:
            insert_rows = min(n, 50_000)
            insert_bare = await measure_insert_ms(
                driver, group, insert_rows, pool, node_pool=1_000, with_index=False)
            insert_indexed = await measure_insert_ms(
                driver, group, insert_rows, pool, node_pool=1_000, with_index=True)

        await build(driver, group, n, pool, rng)
        probe = pool.vectors[rng.randrange(len(pool.vectors))]
        params = dict(g=group, v=probe, min=-1.0, k=10)
        brute_ms, _ = await time_query(driver, brute_cypher, **params)
        await create_index(driver, ddl(index_name, DIM), index_name)
        index_ms, _ = await time_query(driver, index_cypher, idx=index_name, **params)
        control_ms, _ = await time_query(driver, brute_cypher, **params)
        recall = await measure_recall(
            driver, group, pool, rng, index_name, brute_cypher, index_cypher, probes=probes)
        results.append(StepResult(
            n=n, kind=kind, brute_ms=brute_ms, index_ms=index_ms, control_ms=control_ms,
            recall=recall, insert_bare_ms=insert_bare, insert_indexed_ms=insert_indexed,
            insert_rows=insert_rows))
        measured_to = n
        print(f"  brute {brute_ms:.1f} ms | index {index_ms:.1f} ms | "
              f"control {control_ms:.1f} ms | r@10 "
              f"{recall[10][0]:.3f}/{recall[50][0]:.3f}/{recall[200][0]:.3f}",
              flush=True)
        await drop_index(driver, index_name)
        await wipe(driver, group)
    return results, measured_to


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-edges", type=int, default=max(EDGE_STEPS))
    parser.add_argument("--max-nodes", type=int, default=max(NODE_STEPS))
    parser.add_argument("--probes", type=int, default=20)
    parser.add_argument("--out", default="docs/superpowers/vector-crossover-2026-09-16.md")
    args = parser.parse_args()
    return asyncio.run(_run(args))


async def _run(args: argparse.Namespace) -> int:
    from neo4j import AsyncGraphDatabase
    from testcontainers.neo4j import Neo4jContainer

    seeds = await fetch_seed_vectors()
    rng = random.Random(20260916)
    pool = build_pool(n=10_000, dim=DIM, rng=rng, seeds=seeds)
    print(f"vectors: {pool.provenance}", flush=True)

    group = "__crossover__"
    with Neo4jContainer("neo4j:2026.07.1-community") as neo:
        driver = AsyncGraphDatabase.driver(
            neo.get_connection_url(), auth=("neo4j", neo.password))
        try:
            await ensure_uuid_index(driver)
            edge_steps, edge_measured_to = await _sweep(
                driver, group, "edge", EDGE_STEPS, args.max_edges, pool, rng, args.probes,
                BRUTE_EDGE, INDEX_EDGE, EDGE_INDEX, edge_index_ddl, _build_edges,
                measure_inserts=True)
            node_steps, node_measured_to = await _sweep(
                driver, group, "node", NODE_STEPS, args.max_nodes, pool, rng, args.probes,
                BRUTE_NODE, INDEX_NODE, NODE_INDEX, node_index_ddl, _build_nodes,
                measure_inserts=False)
        finally:
            await driver.close()

    report = format_report(
        edge_steps + node_steps, pool.provenance,
        {"edge": edge_measured_to, "node": node_measured_to})
    print("\n" + report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
