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

pytestmark = pytest.mark.asyncio(loop_scope="module")

GROUP = "__crossover_test__"
DIM = 16


async def test_the_fixture_builds_the_requested_number_of_edges(extract_driver):
    await wipe(extract_driver, GROUP)
    await ensure_uuid_index(extract_driver)
    pool = build_pool(n=200, dim=DIM, rng=random.Random(0), seeds=None)
    await build_edge_fixture(extract_driver, GROUP, 200, pool, node_pool=20, batch=50,
                              rng=random.Random(10))

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
                              rng=random.Random(20))
    await create_index(extract_driver, edge_index_ddl(EDGE_INDEX, DIM), EDGE_INDEX)

    r = await extract_driver.execute_query(
        "SHOW INDEXES YIELD name, state WHERE name = $n RETURN state", n=EDGE_INDEX)
    assert r.records[0]["state"] == "ONLINE"
    await wipe(extract_driver, GROUP)


async def test_fixture_vectors_stay_distinct_past_the_pool_size(extract_driver):
    """The sweep builds a 10,000-vector pool and goes to 1,000,000 edges. Indexing
    into the pool would make every 10,000th row byte-identical, so the index would
    be built over a fraction of the distinct points the report claims."""
    await wipe(extract_driver, GROUP)
    await ensure_uuid_index(extract_driver)
    pool = build_pool(n=5, dim=DIM, rng=random.Random(11), seeds=None)
    await build_edge_fixture(extract_driver, GROUP, 60, pool, node_pool=10, batch=20,
                              rng=random.Random(12))
    r = await extract_driver.execute_query(
        "MATCH ()-[e:RELATES_TO {group_id:$g}]->() "
        "RETURN count(DISTINCT e.fact_embedding) AS distinct, count(e) AS total", g=GROUP)
    distinct, total = r.records[0]["distinct"], r.records[0]["total"]
    assert total == 60
    assert distinct == 60, (
        f"only {distinct} distinct vectors across {total} edges from a 5-vector pool; "
        f"the fixture is reusing vectors and the index would be built over too few points")
    await wipe(extract_driver, GROUP)


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
    await create_index(extract_driver, edge_index_ddl(EDGE_INDEX, DIM), EDGE_INDEX)
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
    await create_index(extract_driver, edge_index_ddl(EDGE_INDEX, DIM), EDGE_INDEX)

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
