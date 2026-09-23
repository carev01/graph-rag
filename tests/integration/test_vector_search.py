"""Vector-index lifecycle and index-backed search on a real Neo4j testcontainer.
Hermetic: no LLM, no embedder endpoint, never the .env graph."""
from __future__ import annotations

import pytest

from graph_extract import vector_search as vs

pytestmark = pytest.mark.asyncio(loop_scope="module")

DIM = 8


async def _drop_all_vector_indexes(driver) -> None:
    r = await driver.execute_query("SHOW VECTOR INDEXES YIELD name RETURN name")
    for rec in r.records:
        await driver.execute_query(f"DROP INDEX {rec['name']} IF EXISTS")


async def test_ensure_creates_both_indexes_online_with_the_tuned_config(extract_driver):
    await _drop_all_vector_indexes(extract_driver)
    await vs.ensure_vector_indexes(extract_driver, DIM)
    status = await vs.index_status(extract_driver)
    for name in (vs.EDGE_INDEX, vs.NODE_INDEX):
        assert status[name]["state"] == "ONLINE"
        assert vs.config_mismatches(status[name]["config"], vs.expected_config(DIM)) == {}
    assert status[vs.EDGE_INDEX]["entityType"] == "RELATIONSHIP"
    assert status[vs.EDGE_INDEX]["labelsOrTypes"] == ["RELATES_TO"]
    assert status[vs.EDGE_INDEX]["properties"] == ["fact_embedding"]
    assert status[vs.NODE_INDEX]["entityType"] == "NODE"
    assert status[vs.NODE_INDEX]["labelsOrTypes"] == ["Entity"]
    assert status[vs.NODE_INDEX]["properties"] == ["name_embedding"]


async def test_ensure_is_idempotent(extract_driver):
    await vs.ensure_vector_indexes(extract_driver, DIM)
    await vs.ensure_vector_indexes(extract_driver, DIM)
    r = await extract_driver.execute_query("SHOW VECTOR INDEXES YIELD name RETURN name")
    assert sorted(x["name"] for x in r.records) == sorted([vs.EDGE_INDEX, vs.NODE_INDEX])


async def test_an_index_with_default_options_is_refused_not_rebuilt(extract_driver):
    await _drop_all_vector_indexes(extract_driver)
    await extract_driver.execute_query(
        f"CREATE VECTOR INDEX {vs.NODE_INDEX} FOR (n:Entity) ON (n.name_embedding) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {DIM}, "
        "`vector.similarity_function`: 'cosine'}}")
    with pytest.raises(vs.VectorIndexMismatch) as exc:
        await vs.ensure_vector_indexes(extract_driver, DIM)
    msg = str(exc.value)
    assert vs.NODE_INDEX in msg
    assert "vector.hnsw.m" in msg
    assert "vector-index --rebuild" in msg
    # refused, not rebuilt: the default-options index is still there
    status = await vs.index_status(extract_driver)
    assert status[vs.NODE_INDEX]["config"]["vector.hnsw.m"] == 16


async def test_a_dimension_mismatch_is_refused(extract_driver):
    await _drop_all_vector_indexes(extract_driver)
    await vs.ensure_vector_indexes(extract_driver, DIM)
    with pytest.raises(vs.VectorIndexMismatch):
        await vs.ensure_vector_indexes(extract_driver, DIM * 2)


async def test_drop_removes_both(extract_driver):
    await vs.ensure_vector_indexes(extract_driver, DIM)
    await vs.drop_vector_indexes(extract_driver)
    assert await vs.index_status(extract_driver) == {}
