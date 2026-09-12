"""The candidate search shipped `fact_embedding` -- 768 floats per candidate,
20 candidates, twice per extracted fact -- and graphiti popped it on arrival.
Measured 2.8x on the query (751 -> 269 ms, identical rows)."""
from __future__ import annotations

import pytest
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.search import search_utils

from graph_extract import lean_edge_search as les


def test_the_library_query_really_does_ship_the_embedding():
    """Pins the premise. If graphiti stops using `properties(e)`, this fails and
    the whole patch should be reconsidered rather than silently kept."""
    assert "properties(e) AS attributes" in get_entity_edge_return_query(GraphProvider.NEO4J)


def test_the_lean_query_drops_properties_and_keeps_every_other_field():
    original = get_entity_edge_return_query(GraphProvider.NEO4J)
    lean = les.lean_entity_edge_return_query(GraphProvider.NEO4J)
    assert "properties(e)" not in lean
    assert "fact_embedding" not in lean
    # every non-attributes field graphiti returned is still returned
    for field in ("e.uuid AS uuid", "e.fact AS fact", "e.name AS name",
                  "e.group_id AS group_id", "e.episodes AS episodes",
                  "e.valid_at AS valid_at", "e.invalid_at AS invalid_at",
                  "e.expired_at AS expired_at", "e.reference_time AS reference_time",
                  "startNode(e).uuid AS source_node_uuid",
                  "endNode(e).uuid AS target_node_uuid"):
        assert field in original, "test premise stale"
        assert field in lean, f"lean query lost {field!r}"


def test_the_projection_covers_every_key_graphiti_pops_except_the_embedding():
    """The projection is hand-written, so it can drift from KNOWN_EDGE_KEYS. If it
    does, the guard would pass while the query still drops a field."""
    projected = {k.strip().lstrip(".") for k in
                 les._LEAN_PROJECTION.split("{")[1].split("}")[0].split(",")}
    assert projected == les.KNOWN_EDGE_KEYS - {"fact_embedding"}


def test_an_unrecognised_library_query_is_left_alone(monkeypatch, caplog):
    """A slower search is a cost; a malformed query is an outage."""
    monkeypatch.setattr(les, "get_entity_edge_return_query",
                        lambda provider: "e.uuid AS uuid, something_else AS attributes")
    out = les.lean_entity_edge_return_query(GraphProvider.NEO4J)
    assert out == "e.uuid AS uuid, something_else AS attributes"
    assert "no longer contains" in caplog.text


def test_install_patches_the_module_that_imported_it_by_name(monkeypatch):
    """search_utils does `from ... import get_entity_edge_return_query`, so the
    defining module's attribute is not what it calls."""
    monkeypatch.setattr(search_utils, "get_entity_edge_return_query",
                        get_entity_edge_return_query, raising=True)
    assert "properties(e)" in search_utils.get_entity_edge_return_query(GraphProvider.NEO4J)
    assert les.install_lean_edge_search() is True
    assert "properties(e)" not in search_utils.get_entity_edge_return_query(
        GraphProvider.NEO4J)


class _Rec(dict):
    async def single(self):
        return self


class _Session:
    def __init__(self, keys):
        self._keys = keys

    async def run(self, *a, **k):
        return _Rec(ks=self._keys)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Driver:
    def __init__(self, keys):
        self._keys = keys

    def session(self):
        return _Session(self._keys)


@pytest.mark.asyncio
async def test_guard_passes_on_the_standard_key_set():
    await les.assert_no_custom_edge_attributes(
        _Driver(sorted(les.KNOWN_EDGE_KEYS)), "g")


@pytest.mark.asyncio
async def test_guard_raises_and_names_a_custom_attribute():
    """A custom edge attribute would be dropped from search results silently --
    this project's most expensive recurring failure class."""
    with pytest.raises(ValueError, match="confidence"):
        await les.assert_no_custom_edge_attributes(
            _Driver([*les.KNOWN_EDGE_KEYS, "confidence"]), "g")
