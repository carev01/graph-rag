"""The invalidation-candidate search is the only O(corpus) query in the ingest
path: PROFILE shows it scanning all 3,469 edges, while the duplicate search runs a
DirectedRelationshipIndexSeek over 10. It exists to feed CROSS-PAIR contradiction
detection, which is suspended until the corpus has a real time axis. Same-pair
contradiction -- `contradicted_facts` indices pointing into the delegated
duplicate-candidate list -- stays live; the last tripwire below pins that."""
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


# --- library tripwires ------------------------------------------------------
# NOT tests of our code. These pin the graphiti 0.30.1 facts the gate's safety
# rests on, so a library upgrade fails here rather than silently changing
# behaviour. They are expected to survive every mutation of our module.

def test_edge_operations_imports_search_by_name():
    """If it stopped, patching that module's attribute would silently no-op."""
    from graphiti_core.search.search import search as defining_module_search
    assert edge_operations.search is defining_module_search


def test_resolve_extracted_edges_has_exactly_two_search_call_sites():
    """The gate's discriminator is safe ONLY because there are exactly two
    `search()` calls and they differ in whether `search_filter` carries
    `edge_uuids`. Parsed with `ast` rather than matched as text: this asserts the
    structural property the gate depends on, and does not break on reformatting.

    Note this is not the `inspect.getsource` substring anti-pattern BACKLOG 16
    warns about. That one tests OUR code through its text; this pins a fact about
    a THIRD-PARTY library that has no behavioural probe -- nothing observable tells
    us graphiti grew a third call site until the gate silently mishandles it.

    Two known limits, recorded so nobody over-reads a green result:
    - It parses `resolve_extracted_edges` ONLY. A `search()` call added elsewhere
      in `edge_operations` (a new helper, say) would not trip it, and the gate
      would discriminate that call by its filter like any other.
    - `filter_has_edge_uuids` reads the `search_filter=` keyword LITERALLY: it is
      True only for an inline `SearchFilters(...)` call carrying keywords. A call
      site that passes a filter built into a variable (`search_filter=flt`) reads
      as "unfiltered" (False) and the `[True, False]` assertion could stay green
      while the second site had quietly gained an `edge_uuids` bound elsewhere.
    Neither has a behavioural probe either; they are documented, not covered."""
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(edge_operations.resolve_extracted_edges))
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "search"]
    assert len(calls) == 2, f"expected 2 search() call sites, found {len(calls)}"

    def filter_has_edge_uuids(call: ast.Call) -> bool | None:
        for kw in call.keywords:
            if kw.arg == "search_filter":
                return bool(getattr(kw.value, "keywords", []))
        return None

    assert [filter_has_edge_uuids(c) for c in calls] == [True, False], (
        "expected one filtered (duplicate) and one unfiltered (invalidation) "
        "search, in that order")


def test_no_invalidation_candidates_means_no_invalidation():
    """With the gate installed, `existing_edges` is always empty, so the
    CROSS-PAIR half of `invalidation_candidates` is always empty. This pins only
    that an empty candidate list invalidates nothing. It does NOT mean graphiti
    invalidates nothing: `invalidation_candidates` is also fed from
    `related_edges` (the delegated duplicate search) -- see the next tripwire."""
    assert edge_operations.resolve_edge_contradictions(None, []) == []


def test_same_pair_contradiction_stays_live_through_the_duplicate_candidates():
    """The residual invalidation path, pinned so it is documented rather than
    forgotten (BACKLOG 33). `resolve_extracted_edge` routes `contradicted_facts`
    indices below `len(related_edges)` into `invalidation_candidates`, and
    `related_edges` comes from the FILTERED search the gate delegates. So with
    the gate installed graphiti can still invalidate a same-pair fact, and can
    still set `invalid_at` on the NEW edge from a same-pair candidate a few
    lines later. The gate suspends the O(corpus) scan and cross-pair
    invalidation only.

    Same justification as the call-site tripwire: a third-party-library fact
    with no behavioural probe (exercising it needs an LLM reply), asserted on the
    `ast` of the library function. If graphiti ever stops routing those indices
    -- or we decide to filter `contradicted_facts` ourselves -- this fails and
    the claims around the gate must be re-examined, in either direction."""
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(edge_operations.resolve_extracted_edge))
    tree = ast.parse(src)

    def is_len_related_edges(node: ast.AST) -> bool:
        return (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "len"
                and len(node.args) == 1
                and getattr(node.args[0], "id", None) == "related_edges")

    def appends_related_edge_to_invalidation(stmt: ast.stmt) -> bool:
        # invalidation_candidates.append(related_edges[idx])
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            return False
        call = stmt.value
        target = call.func
        if not (isinstance(target, ast.Attribute) and target.attr == "append"
                and getattr(target.value, "id", None) == "invalidation_candidates"):
            return False
        return (len(call.args) == 1 and isinstance(call.args[0], ast.Subscript)
                and getattr(call.args[0].value, "id", None) == "related_edges")

    routes = []
    for loop in ast.walk(tree):
        if not (isinstance(loop, ast.For)
                and getattr(loop.iter, "id", None) == "contradicted_facts"):
            continue
        for branch in ast.walk(loop):
            # if 0 <= idx < len(related_edges): invalidation_candidates.append(related_edges[idx])
            if not (isinstance(branch, ast.If) and isinstance(branch.test, ast.Compare)):
                continue
            test = branch.test
            below_len = (len(test.comparators) == 2
                         and isinstance(test.ops[1], ast.Lt)
                         and is_len_related_edges(test.comparators[1]))
            if below_len and any(appends_related_edge_to_invalidation(s) for s in branch.body):
                routes.append(branch)

    assert len(routes) == 1, (
        "expected resolve_extracted_edge to route contradicted_facts indices below "
        f"len(related_edges) into invalidation_candidates exactly once; found {len(routes)}")
