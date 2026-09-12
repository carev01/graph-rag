"""The invalidation-candidate search is the only O(corpus) query in the ingest
path: PROFILE shows it scanning all 3,469 edges, while the duplicate search runs a
DirectedRelationshipIndexSeek over 10. It exists to feed contradiction detection,
which is suspended until the corpus has a real time axis."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance import edge_operations

from graph_extract import contradiction_gate as cg

# `SearchResults.edges` is `list[EntityEdge]` and pydantic validates it, so the
# fakes must return a real edge -- a bare string sentinel is rejected on construction.
_SENTINEL_EDGE = EntityEdge(
    source_node_uuid="s", target_node_uuid="t", name="RELATES_TO", fact="sentinel",
    group_id="g", created_at=datetime(2026, 9, 12, tzinfo=UTC))


@pytest.fixture
def restore_search():
    original = edge_operations.search
    yield
    edge_operations.search = original


def _recording():
    seen: list[SearchFilters] = []

    async def _search(clients, query, *, group_ids=None, config=None,
                      search_filter=None, **kw):
        seen.append(search_filter)
        return SearchResults(edges=[_SENTINEL_EDGE])

    return seen, _search


@pytest.mark.asyncio
async def test_the_unfiltered_invalidation_search_is_skipped(restore_search):
    seen, fake = _recording()
    edge_operations.search = fake
    assert cg.install_contradiction_gate(detect_contradictions=False) is True

    out = await edge_operations.search(
        None, "a fact", group_ids=["g"], config=None, search_filter=SearchFilters())

    assert isinstance(out, SearchResults)
    assert out.edges == []
    assert seen == [], "the underlying search must never be awaited"


@pytest.mark.asyncio
async def test_the_filtered_duplicate_search_is_delegated(restore_search):
    seen, fake = _recording()
    edge_operations.search = fake
    cg.install_contradiction_gate(detect_contradictions=False)

    out = await edge_operations.search(
        None, "a fact", group_ids=["g"], config=None,
        search_filter=SearchFilters(edge_uuids=["u1", "u2"]))

    assert out.edges == [_SENTINEL_EDGE]
    assert len(seen) == 1 and seen[0].edge_uuids == ["u1", "u2"]


@pytest.mark.asyncio
async def test_search_filter_passed_POSITIONALLY_is_still_discriminated(restore_search):
    """graphiti passes it as a keyword today, but it is the 5th positional
    parameter of `search()`. A wrapper that only reads kwargs would silently stop
    skipping if that ever changed -- and silence is the failure mode this codebase
    keeps paying for."""
    seen, fake = _recording()

    async def _positional(clients, query, group_ids, config, search_filter, **kw):
        seen.append(search_filter)
        return SearchResults(edges=[_SENTINEL_EDGE])

    edge_operations.search = _positional
    cg.install_contradiction_gate(detect_contradictions=False)

    out = await edge_operations.search(None, "a fact", ["g"], None, SearchFilters())
    assert out.edges == [] and seen == []


@pytest.mark.asyncio
async def test_enabled_delegates_everything(restore_search):
    seen, fake = _recording()
    edge_operations.search = fake
    assert cg.install_contradiction_gate(detect_contradictions=True) is False

    await edge_operations.search(None, "f", group_ids=["g"], config=None,
                                 search_filter=SearchFilters())
    await edge_operations.search(None, "f", group_ids=["g"], config=None,
                                 search_filter=SearchFilters(edge_uuids=["u"]))
    assert len(seen) == 2, "enabled must behave exactly as today"


@pytest.mark.asyncio
async def test_installing_twice_does_not_double_wrap(restore_search):
    seen, fake = _recording()
    edge_operations.search = fake
    cg.install_contradiction_gate(detect_contradictions=False)
    first = edge_operations.search
    cg.install_contradiction_gate(detect_contradictions=False)
    assert edge_operations.search is first

    await edge_operations.search(None, "f", group_ids=["g"], config=None,
                                 search_filter=SearchFilters(edge_uuids=["u"]))
    assert len(seen) == 1, "a double wrap would delegate twice or not at all"


def test_is_gate_installed_reports_state(restore_search):
    assert cg.is_gate_installed() is False
    cg.install_contradiction_gate(detect_contradictions=False)
    assert cg.is_gate_installed() is True


@pytest.mark.asyncio
async def test_re_enabling_restores_the_original_search(restore_search):
    seen, fake = _recording()
    edge_operations.search = fake

    assert cg.install_contradiction_gate(detect_contradictions=False) is True
    assert cg.is_gate_installed() is True

    assert cg.install_contradiction_gate(detect_contradictions=True) is False
    assert cg.is_gate_installed() is False
    assert edge_operations.search is fake

    out = await edge_operations.search(
        None, "a fact", group_ids=["g"], config=None, search_filter=SearchFilters())
    assert out.edges == [_SENTINEL_EDGE]
    assert len(seen) == 1, "an unfiltered search must be delegated once restored"


@pytest.mark.asyncio
async def test_a_delegated_failure_propagates(restore_search):
    async def _boom(clients, query, **kw):
        raise RuntimeError("neo4j down")

    edge_operations.search = _boom
    cg.install_contradiction_gate(detect_contradictions=False)
    with pytest.raises(RuntimeError, match="neo4j down"):
        await edge_operations.search(None, "f", group_ids=["g"], config=None,
                                     search_filter=SearchFilters(edge_uuids=["u"]))
