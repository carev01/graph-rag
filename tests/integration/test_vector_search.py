"""Vector-index lifecycle and index-backed search on a real Neo4j testcontainer.
Hermetic: no LLM, no embedder endpoint, never the .env graph."""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import pytest
import pytest_asyncio
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.nodes import EntityNode
from graphiti_core.search import search as g_search
from graphiti_core.search import search_utils
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance import node_operations

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


G = "vs-test"


def _unit(i: int) -> list[float]:
    """Planted vectors: angle i*5 degrees in the first two dims, so cosine to
    _unit(0) strictly decreases with i -- a known exact ranking."""
    a = math.radians(i * 5)
    return [math.cos(a), math.sin(a)] + [0.0] * (DIM - 2)


class _FixedEmbedder(EmbedderClient):
    async def create(self, input_data: str | list[str] | Iterable[int]
                     | Iterable[Iterable[int]]) -> list[float]:
        return _unit(0)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [_unit(0) for _ in input_data_list]


class _NoLLM(LLMClient):
    async def _generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("the LLM must not be reached")


class _NoReranker(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        raise AssertionError("the reranker must not be reached by an RRF search")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def hermetic_graphiti(extract_neo4j):
    uri, user, password = extract_neo4j
    g = Graphiti(uri, user, password, llm_client=_NoLLM(config=None),
                 embedder=_FixedEmbedder(), cross_encoder=_NoReranker())
    await g.build_indices_and_constraints()
    yield g
    await g.close()


@pytest.fixture
def routed():
    vs.install_vector_search(enabled=True, fetch_k=200)
    vs.reset_stats()
    yield
    vs.install_vector_search(enabled=False, fetch_k=200)


async def _seed(driver, n: int = 12) -> None:
    await driver.execute_query("MATCH (n) DETACH DELETE n")
    await driver.execute_query(
        "UNWIND range(0, $n - 1) AS i "
        "CREATE (:Entity {uuid: 'n' + i, group_id: $g, name: 'entity ' + i, "
        "  name_embedding: $vecs[i], summary: '', created_at: datetime()})",
        n=n, g=G, vecs=[_unit(i) for i in range(n)])
    await driver.execute_query(
        "UNWIND range(0, $n - 2) AS i "
        "MATCH (a:Entity {uuid: 'n' + i}), (b:Entity {uuid: 'n' + (i + 1)}) "
        "CREATE (a)-[:RELATES_TO {uuid: 'f' + i, group_id: $g, name: 'REL', "
        "  fact: 'fact ' + i, fact_embedding: $vecs[i], episodes: [], "
        "  created_at: datetime()}]->(b)",
        n=n, g=G, vecs=[_unit(i) for i in range(n)])


async def test_index_path_matches_the_exact_scan_on_a_planted_ranking(
        hermetic_graphiti, extract_driver):
    await _seed(extract_driver)
    await _drop_all_vector_indexes(extract_driver)
    await vs.ensure_vector_indexes(extract_driver, DIM)
    d = hermetic_graphiti.driver
    exact_e = await vs._ORIG_EDGE(d, _unit(0), None, None, SearchFilters(), [G], 5, 0.6)
    index_e = await vs.index_edge_similarity_search(
        d, _unit(0), None, None, SearchFilters(), [G], 5, 0.6)
    assert [e.uuid for e in index_e] == [e.uuid for e in exact_e] == [
        "f0", "f1", "f2", "f3", "f4"]
    exact_n = await vs._ORIG_NODE(d, _unit(0), SearchFilters(), [G], 5, 0.6)
    index_n = await vs.index_node_similarity_search(d, _unit(0), SearchFilters(), [G], 5, 0.6)
    assert [n.uuid for n in index_n] == [n.uuid for n in exact_n] == [
        "n0", "n1", "n2", "n3", "n4"]
    # the lean projection still applies on the index path: no embedding shipped
    assert all(e.fact_embedding is None for e in index_e)


async def test_other_groups_are_filtered_out(hermetic_graphiti, extract_driver):
    await _seed(extract_driver)
    await vs.ensure_vector_indexes(extract_driver, DIM)
    out = await vs.index_node_similarity_search(
        hermetic_graphiti.driver, _unit(0), SearchFilters(), ["another-group"], 5, 0.6)
    assert out == []


async def test_retrieval_and_node_dedup_both_route_through_the_real_call_paths(
        hermetic_graphiti, extract_driver, routed):
    await _seed(extract_driver)
    await vs.ensure_vector_indexes(extract_driver, DIM)
    results = await hermetic_graphiti._search(
        "entity 0", EDGE_HYBRID_SEARCH_RRF, group_ids=[G])
    assert results.edges, "retrieval found nothing"
    assert vs.stats_snapshot()["edge"]["routed"] >= 1
    extracted = EntityNode(name="entity 0", group_id=G, labels=["Entity"], summary="")
    candidates = await node_operations._semantic_candidate_search(
        hermetic_graphiti.clients, [extracted])
    assert candidates[0], "node dedup found no candidates"
    assert candidates[0][0].uuid == "n0"
    assert vs.stats_snapshot()["node"]["routed"] >= 1


async def test_same_pair_dedup_stays_on_the_exact_scan(
        hermetic_graphiti, extract_driver, routed):
    await _seed(extract_driver)
    await vs.ensure_vector_indexes(extract_driver, DIM)
    await search_utils.edge_similarity_search(
        hermetic_graphiti.driver, _unit(0), None, None,
        SearchFilters(edge_uuids=["f0", "f1"]), [G], 5, 0.6)
    await g_search.edge_similarity_search(
        hermetic_graphiti.driver, _unit(0), None, None,
        SearchFilters(edge_uuids=["f0", "f1"]), [G], 5, 0.6)
    assert vs.stats_snapshot()["edge"]["delegated_bounded"] == 1
    assert vs.stats_snapshot()["edge"]["routed"] == 0


async def test_a_missing_index_falls_back_to_identical_results(
        hermetic_graphiti, extract_driver):
    await _seed(extract_driver)
    await vs.drop_vector_indexes(extract_driver)
    vs.reset_stats()
    d = hermetic_graphiti.driver
    out = await vs.index_node_similarity_search(d, _unit(0), SearchFilters(), [G], 5, 0.6)
    exact = await vs._ORIG_NODE(d, _unit(0), SearchFilters(), [G], 5, 0.6)
    assert [n.uuid for n in out] == [n.uuid for n in exact]
    assert vs.stats_snapshot()["node"]["fell_back"] == 1


# Enough 768-dim vectors that building the tuned index takes minutes on the
# testcontainer (120k took ~150 s), so it is reliably still POPULATING when
# queried right after CREATE. Vectors are generated server-side: seeding stays
# a few seconds.
_POPULATING_ROWS = 100_000
_POPULATING_DIM = 768


async def test_a_populating_index_is_reported_not_refused_and_searches_fall_back(
        hermetic_graphiti, extract_driver, caplog):
    await extract_driver.execute_query("MATCH (n) DETACH DELETE n")
    await _drop_all_vector_indexes(extract_driver)
    await extract_driver.execute_query(
        "UNWIND range(0, $n - 1) AS i "
        "CREATE (:Entity {uuid: 'p' + i, group_id: 'populating', name: 'p' + i, "
        "  summary: '', created_at: datetime(), "
        "  name_embedding: [x IN range(1, $dim) | rand() - 0.5]})",
        n=_POPULATING_ROWS, dim=_POPULATING_DIM)
    try:
        with caplog.at_level("WARNING", logger=vs.logger.name):
            await vs.ensure_vector_indexes(extract_driver, _POPULATING_DIM, wait_seconds=0)
        status = await vs.index_status(extract_driver)
        assert status[vs.NODE_INDEX]["state"] == "POPULATING", status
        assert any(vs.NODE_INDEX in r.getMessage() and "POPULATING" in r.getMessage()
                   for r in caplog.records), "no POPULATING warning"

        vs.reset_stats()
        await vs.index_node_similarity_search(
            hermetic_graphiti.driver, [1.0] + [0.0] * (_POPULATING_DIM - 1),
            SearchFilters(), ["populating"], 5, 0.0)
        # Neo4j blocked ~30 s waiting for the index, then refused; the wrapper
        # recognised that and served the exact scan.
        assert vs.stats_snapshot()["node"] == {
            "routed": 0, "delegated_bounded": 0, "fell_back": 1}
        status = await vs.index_status(extract_driver)
        assert status[vs.NODE_INDEX]["state"] == "POPULATING", \
            "index came ONLINE during the query: seed more rows"
    finally:
        await vs.drop_vector_indexes(extract_driver)
        deleted = 1
        while deleted:
            r = await extract_driver.execute_query(
                "MATCH (n:Entity {group_id: 'populating'}) WITH n LIMIT 20000 "
                "DETACH DELETE n RETURN count(*) AS c")
            deleted = r.records[0]["c"]
