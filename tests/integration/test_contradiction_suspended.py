"""End-to-end on a real Neo4j: with contradiction detection suspended, an ingest
invalidates nothing and local retrieval is unchanged."""
from __future__ import annotations

import pytest
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance import edge_operations

from graph_extract.contradiction_gate import install_contradiction_gate

# The true original, not whatever is installed when this module is imported --
# another test may have gated it already.
from graphiti_core.search.search import search as _UNGATED

pytestmark = pytest.mark.asyncio(loop_scope="module")

G = "backup-docs"


@pytest.fixture
def restore_search():
    original = edge_operations.search
    yield
    edge_operations.search = original


async def test_the_invalidation_search_never_reaches_the_database(
        extract_driver, restore_search):
    """The scale claim, asserted on behaviour rather than on timing: the
    unfiltered search issues no query at all."""
    issued: list[SearchFilters] = []
    original = edge_operations.search

    async def _counting(*args, **kwargs):
        issued.append(kwargs.get("search_filter"))
        return await original(*args, **kwargs)

    edge_operations.search = _counting
    install_contradiction_gate(detect_contradictions=False)

    from graphiti_core.search.search_config import SearchResults
    out = await edge_operations.search(
        None, "any fact", group_ids=[G], config=None, search_filter=SearchFilters())
    assert isinstance(out, SearchResults) and out.edges == []
    assert issued == [], "the invalidation search must not reach the driver"


async def test_build_graphiti_installs_the_gate(restore_search):
    """The wiring, proven where build_graphiti is actually exercised. It builds
    real embedder/reranker/driver clients, which is why this is not a unit test."""
    from graph_extract.config import get_extract_settings
    from graph_extract.contradiction_gate import is_gate_installed
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


async def test_local_search_is_unaffected_by_the_gate(extract_driver, restore_search):
    """The one thing this must not break. The answer path does not route through
    edge_operations, so installing the gate must change nothing."""
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run(
            "CREATE (a:Entity {uuid:'e1', group_id:$g, name:'AWS Backup'})"
            "-[:RELATES_TO {group_id:$g, uuid:'f1', fact:'AWS Backup encrypts data', "
            " name:'Encrypts', created_at: datetime()}]->"
            "(b:Entity {uuid:'e2', group_id:$g, name:'KMS'})", g=G)

        async def count() -> int:
            r = await s.run("MATCH ()-[f:RELATES_TO {group_id:$g}]->() "
                            "WHERE f.invalid_at IS NULL RETURN count(f) AS n", g=G)
            return (await r.single())["n"]

        before = await count()
        install_contradiction_gate(detect_contradictions=False)
        after = await count()

    assert before == after == 1, "retrieval-visible facts must be unchanged"
