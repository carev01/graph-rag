"""End-to-end on a real Neo4j: with contradiction detection suspended, the
invalidation search issues no query, the duplicate search still does, and the
answer path returns identical results with and without the gate.

Hermetic against every paid endpoint: the Graphiti built here carries a fixed
embedder and LLM/reranker clients that raise if reached, so what runs is
graphiti's own `search()` code against the testcontainer and nothing else."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pytest
import pytest_asyncio
from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance import edge_operations

from answer_api.search import search_local
from graph_extract.contradiction_gate import install_contradiction_gate, is_gate_installed

# The true original, not whatever is installed when this module is imported --
# another test may have gated it already.
from graphiti_core.search.search import search as _UNGATED

pytestmark = pytest.mark.asyncio(loop_scope="module")

G = "backup-docs"
_DIM = 8
# One vector for the query AND the seeded fact, so cosine similarity is exactly
# 1.0 and the search finds the fact without an embedding endpoint.
_VEC = [1.0] + [0.0] * (_DIM - 1)
_Q = "AWS Backup encrypts data with KMS"


class _FixedEmbedder(EmbedderClient):
    async def create(self, input_data: str | list[str] | Iterable[int]
                     | Iterable[Iterable[int]]) -> list[float]:
        return list(_VEC)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [list(_VEC) for _ in input_data_list]


class _NoLLM(LLMClient):
    async def _generate_response(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise AssertionError("the LLM must not be reached by a search")


class _NoReranker(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        raise AssertionError("the reranker must not be reached by an RRF search")


@pytest.fixture
def restore_search():
    original = edge_operations.search
    yield
    edge_operations.search = original


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def hermetic_graphiti(extract_neo4j):
    """A real Graphiti on the testcontainer with no reachable paid client. Builds
    graphiti's own indices (the fulltext one is required by BM25) once."""
    uri, user, password = extract_neo4j
    g = Graphiti(uri, user, password, llm_client=_NoLLM(config=None),
                 embedder=_FixedEmbedder(), cross_encoder=_NoReranker())
    await g.build_indices_and_constraints()
    yield g
    await g.close()


async def _seed_one_cited_fact(driver) -> None:
    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run(
            "CREATE (:Article {id:'art1', source_url:'https://x/art1', title:'T1'})"
            "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1', group_id:$g}) "
            "CREATE (:Entity {uuid:'e1', group_id:$g, name:'AWS Backup'})"
            "-[:RELATES_TO {group_id:$g, uuid:'f1', name:'ENCRYPTS', fact:$fact, "
            " fact_embedding:$v, episodes:['ep1'], created_at: datetime()}]->"
            "(:Entity {uuid:'e2', group_id:$g, name:'KMS'})",
            g=G, fact=_Q, v=_VEC)


async def test_the_invalidation_search_never_reaches_the_database(
        hermetic_graphiti, extract_driver, restore_search):
    """The scale claim, asserted where it matters: through graphiti's real
    `search()` with real clients against a real Neo4j, the gated invalidation
    search issues ZERO driver queries. The delegated duplicate search -- same
    clients, same counter, one line later -- DOES reach the database and finds
    the seeded fact. That second half is what makes the zero credible."""
    await _seed_one_cited_fact(extract_driver)
    edge_operations.search = _UNGATED
    install_contradiction_gate(detect_contradictions=False)

    driver = hermetic_graphiti.driver
    real_execute = driver.execute_query
    issued: list[str] = []

    async def _counting(cypher_query_, **kwargs):
        issued.append(str(cypher_query_))
        return await real_execute(cypher_query_, **kwargs)

    driver.execute_query = _counting
    try:
        out = await edge_operations.search(
            hermetic_graphiti.clients, _Q, group_ids=[G],
            config=EDGE_HYBRID_SEARCH_RRF, search_filter=SearchFilters())
        assert isinstance(out, SearchResults) and out.edges == []
        assert issued == [], "the invalidation search must not reach the driver"

        dup = await edge_operations.search(
            hermetic_graphiti.clients, _Q, group_ids=[G],
            config=EDGE_HYBRID_SEARCH_RRF, search_filter=SearchFilters(edge_uuids=["f1"]))
        assert [e.uuid for e in dup.edges] == ["f1"]
        assert len(issued) >= 1, "the duplicate search must reach the driver"
    finally:
        driver.execute_query = real_execute


async def test_build_graphiti_installs_the_gate(restore_search):
    """The wiring, proven where build_graphiti is actually exercised. It builds
    real embedder/reranker/driver clients, which is why this is not a unit test."""
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_graphiti

    edge_operations.search = _UNGATED
    assert is_gate_installed() is False
    s = get_extract_settings.__wrapped__().model_copy(
        update=dict(ingest_detect_contradictions=False))
    g = build_graphiti(s)
    try:
        assert is_gate_installed() is True, "build_graphiti did not install the gate"
    finally:
        await g.close()


async def test_local_search_is_unaffected_by_the_gate(
        hermetic_graphiti, extract_driver, restore_search):
    """The one thing this must not break, exercised through the real entry
    point: `search_local` -> `Graphiti._search` -> graphiti's `search()`. That
    search is UNFILTERED -- exactly the shape the gate skips -- so a gate bound
    to the wrong module (the defining one, or `graphiti_core.graphiti`'s copy)
    would empty the second result. The two results must be identical AND
    non-empty; equal empties would prove nothing."""
    await _seed_one_cited_fact(extract_driver)
    edge_operations.search = _UNGATED
    assert is_gate_installed() is False

    before = await search_local(hermetic_graphiti, extract_driver, q=_Q, k=10, group_id=G)
    install_contradiction_gate(detect_contradictions=False)
    assert is_gate_installed() is True
    after = await search_local(hermetic_graphiti, extract_driver, q=_Q, k=10, group_id=G)

    assert before["count"] == 1 and before["results"][0]["fact_uuid"] == "f1"
    assert before["results"][0]["sources"][0]["article_id"] == "art1"
    assert after == before, "retrieval-visible facts must be unchanged by the gate"
