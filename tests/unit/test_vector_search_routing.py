"""Routing rule, fallback and install, against a fake driver -- no database."""
from __future__ import annotations

from typing import Any

import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search import search as g_search
from graphiti_core.search import search_utils
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance import node_operations
from neo4j.exceptions import ClientError

from graph_extract import vector_search as vs

pytestmark = pytest.mark.asyncio

VEC = [1.0, 0.0, 0.0]


class _FakeDriver:
    provider = GraphProvider.NEO4J

    def __init__(self, raise_exc: Exception | None = None) -> None:
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.raise_exc = raise_exc

    async def execute_query(self, query: str, **params: Any):
        self.queries.append((query, params))
        if self.raise_exc is not None:
            raise self.raise_exc
        return [], None, None


@pytest.fixture
def originals(monkeypatch):
    """Replace the captured originals with recorders, so delegation is observed
    as a call, not inferred from reading code."""
    calls: dict[str, list[tuple[tuple, dict]]] = {"edge": [], "node": []}

    async def _edge(*args, **kwargs):
        calls["edge"].append((args, kwargs))
        return ["original-edge"]

    async def _node(*args, **kwargs):
        calls["node"].append((args, kwargs))
        return ["original-node"]

    monkeypatch.setattr(vs, "_ORIG_EDGE", _edge)
    monkeypatch.setattr(vs, "_ORIG_NODE", _node)
    vs.reset_stats()
    return calls


@pytest.mark.parametrize("kwargs", [
    {"source_node_uuid": "a", "target_node_uuid": None, "search_filter": SearchFilters()},
    {"source_node_uuid": None, "target_node_uuid": "b", "search_filter": SearchFilters()},
    {"source_node_uuid": None, "target_node_uuid": None,
     "search_filter": SearchFilters(edge_uuids=["x"])},
    {"source_node_uuid": None, "target_node_uuid": None,
     "search_filter": SearchFilters(edge_types=["USES"])},
])
async def test_bounded_edge_calls_reach_the_original_untouched(originals, kwargs):
    driver = _FakeDriver()
    out = await vs.index_edge_similarity_search(driver, VEC, group_ids=["g"], **kwargs)
    assert out == ["original-edge"]
    assert driver.queries == []
    assert len(originals["edge"]) == 1
    assert vs.stats_snapshot()["edge"]["delegated_bounded"] == 1


async def test_positional_bounded_edge_call_is_read_correctly(originals):
    """graphiti passes these positionally; binding by signature must see the filter."""
    driver = _FakeDriver()
    await vs.index_edge_similarity_search(
        driver, VEC, None, None, SearchFilters(edge_uuids=["x"]), ["g"], 10, 0.6)
    assert driver.queries == []
    assert len(originals["edge"]) == 1


async def test_unbounded_edge_call_goes_to_the_index(originals):
    driver = _FakeDriver()
    out = await vs.index_edge_similarity_search(
        driver, VEC, None, None, SearchFilters(), ["g"], 10, 0.6)
    assert out == []
    assert originals["edge"] == []
    (query, params), = driver.queries
    assert "db.index.vector.queryRelationships" in query
    assert "e.group_id IN $group_ids" in query
    assert "ORDER BY score DESC" in query
    assert params["index_name"] == vs.EDGE_INDEX
    assert params["fetch_k"] == 200
    assert params["limit"] == 10
    assert params["min_score"] == 0.6
    assert params["group_ids"] == ["g"]
    assert vs.stats_snapshot()["edge"]["routed"] == 1


async def test_fetch_k_never_below_limit(originals):
    driver = _FakeDriver()
    await vs.index_edge_similarity_search(
        driver, VEC, None, None, SearchFilters(), ["g"], 500, 0.6)
    assert driver.queries[0][1]["fetch_k"] == 500


async def test_no_group_ids_drops_the_group_predicate(originals):
    driver = _FakeDriver()
    await vs.index_node_similarity_search(driver, VEC, SearchFilters(), None, 15, 0.6)
    (query, params), = driver.queries
    assert "db.index.vector.queryNodes" in query
    assert "group_ids" not in query
    assert params["index_name"] == vs.NODE_INDEX


async def test_bounded_node_call_reaches_the_original(originals):
    driver = _FakeDriver()
    out = await vs.index_node_similarity_search(
        driver, VEC, SearchFilters(node_labels=["Entity"]), ["g"], 15, 0.6)
    assert out == ["original-node"]
    assert driver.queries == []


async def test_a_filter_field_from_a_future_graphiti_counts_as_bounded(originals):
    class _Future(SearchFilters):
        future_field: list[str] | None = None

    driver = _FakeDriver()
    await vs.index_node_similarity_search(
        driver, VEC, _Future(future_field=["x"]), ["g"], 15, 0.6)
    assert driver.queries == []
    assert len(originals["node"]) == 1


async def test_missing_index_falls_back_and_counts(originals):
    err = ClientError._hydrate_neo4j(
        code="Neo.ClientError.Procedure.ProcedureCallFailed",
        message="Failed to invoke procedure: There is no such vector schema index: x")
    driver = _FakeDriver(raise_exc=err)
    out = await vs.index_node_similarity_search(driver, VEC, SearchFilters(), ["g"], 15, 0.6)
    assert out == ["original-node"]
    assert vs.stats_snapshot()["node"]["fell_back"] == 1
    assert vs.stats_snapshot()["node"]["routed"] == 0


async def test_any_other_client_error_propagates(originals):
    err = ClientError._hydrate_neo4j(
        code="Neo.ClientError.Statement.TypeError",
        message="Vector index has a configured dimensionality of 8")
    driver = _FakeDriver(raise_exc=err)
    with pytest.raises(ClientError):
        await vs.index_node_similarity_search(driver, VEC, SearchFilters(), ["g"], 15, 0.6)
    assert originals["node"] == []


async def test_any_other_client_error_propagates_on_the_edge_path(originals):
    err = ClientError._hydrate_neo4j(
        code="Neo.ClientError.Statement.TypeError",
        message="Vector index has a configured dimensionality of 8")
    driver = _FakeDriver(raise_exc=err)
    with pytest.raises(ClientError):
        await vs.index_edge_similarity_search(
            driver, VEC, None, None, SearchFilters(), ["g"], 10, 0.6)
    assert originals["edge"] == []


def test_install_patches_every_by_name_import_and_disable_restores_identity():
    # The captured originals, NOT the current attribute: any earlier test that
    # called build_graphiti has already installed the wrappers process-wide.
    orig_edge = vs._ORIG_EDGE
    orig_node = vs._ORIG_NODE
    try:
        assert vs.install_vector_search(enabled=True, fetch_k=200) is True
        assert g_search.edge_similarity_search is vs.index_edge_similarity_search
        assert g_search.node_similarity_search is vs.index_node_similarity_search
        assert search_utils.node_similarity_search is vs.index_node_similarity_search
        assert node_operations.node_similarity_search is vs.index_node_similarity_search
        assert vs.is_vector_search_installed()
        vs.install_vector_search(enabled=True, fetch_k=200)  # idempotent
        assert vs.install_vector_search(enabled=False, fetch_k=200) is False
        assert g_search.edge_similarity_search is orig_edge
        assert g_search.node_similarity_search is orig_node
        assert search_utils.node_similarity_search is orig_node
        assert node_operations.node_similarity_search is orig_node
        assert not vs.is_vector_search_installed()
    finally:
        vs.install_vector_search(enabled=False, fetch_k=200)


def test_install_refuses_a_foreign_function(monkeypatch):
    async def _someone_elses(*a, **k):
        return []

    monkeypatch.setattr(node_operations, "node_similarity_search", _someone_elses)
    with pytest.raises(RuntimeError, match="node_operations"):
        vs.install_vector_search(enabled=True, fetch_k=200)
    assert not vs.is_vector_search_installed()


def test_fetch_k_must_be_positive():
    with pytest.raises(ValueError):
        vs.install_vector_search(enabled=True, fetch_k=0)


def test_stats_summary_names_both_kinds():
    vs.reset_stats()
    s = vs.stats_summary()
    assert "edge" in s and "node" in s and "routed=0" in s
