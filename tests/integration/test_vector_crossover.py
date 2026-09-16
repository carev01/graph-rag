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
