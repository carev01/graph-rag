# Suspend Contradiction Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop graphiti issuing the O(corpus) invalidation-candidate search during ingest, removing the scale blocker that makes full-corpus ingestion impossible, and with it the phantom invalidations and the dedup index-space confusion.

> **Correction (final review, 2026-09-12):** "and with it the phantom invalidations" is only true for **cross-pair** invalidations. Same-pair contradiction stays live through `contradicted_facts` indices into `related_edges` (`edge_operations.py:769-776`), and the measured phantom invalidations were same-pair. See the spec's correction note, Verification item 2 below, and BACKLOG 33. The CI breakage this plan caused (running unit and integration as separate halves hid a process-wide leak of the gate) is fixed by an autouse fixture in `tests/unit/conftest.py`.

**Architecture:** One new module patches `graphiti_core.utils.maintenance.edge_operations.search` — the module imports that function by name, so the patch targets that module's attribute. The wrapper discriminates structurally: `search_filter.edge_uuids is None` means the invalidation search (unfiltered, scans the corpus) and returns an empty `SearchResults` without querying; anything else is the duplicate search (an index seek) and is delegated untouched. `edge_operations` has exactly two `search()` call sites and they differ precisely that way. The answer path never routes through `edge_operations`, so retrieval is unaffected by construction.

**Tech Stack:** Python 3.12, `uv`, graphiti-core 0.30.1, Neo4j 2026.07.1 Community (no APOC), pytest + pytest-asyncio, testcontainers for integration.

## Global Constraints

- graphiti-core is pinned at **0.30.1**. Do not fork it, do not edit anything under `.venv/`.
- Do **not** use `driver.search_interface`: setting it routes all twelve search methods to the supplied object and unimplemented ones raise.
- Config default is `ingest_detect_contradictions: bool = False` (suspended).
- Terminology: *suspended* = flag is `False`, invalidation search skipped. *Enabled* = flag is `True`, behaviour identical to today.
- The gate must return a well-formed `SearchResults()`, never `None`.
- Never swallow an exception from a delegated call — propagate it.
- CI gate, all three clean: `uv run ruff check src tests` (lints tests too: E702 no semicolons, E402 imports at top), `uv run mypy src`, `uv run --extra dev pytest -m "not live"`.
- Unit tests are hermetic: `ExtractSettings(_env_file=None, ...)`. Never let a test reach a live endpoint.
- Every test must fail with its fix neutralised; prove it by mutation and record the result.
- Current suite baseline: **658 unit + 141 integration**.

---

### Task 1: The contradiction gate

**Files:**
- Create: `src/graph_extract/contradiction_gate.py`
- Test: `tests/unit/test_contradiction_gate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `install_contradiction_gate(*, detect_contradictions: bool) -> bool` — returns `True` if the gate was installed (i.e. contradiction detection is now suspended), `False` if it was left enabled. Idempotent: installing twice does not double-wrap. Also `is_gate_installed() -> bool`, so callers and tests can check state without reaching for a private attribute.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_contradiction_gate.py`:

```python
"""The invalidation-candidate search is the only O(corpus) query in the ingest
path: PROFILE shows it scanning all 3,469 edges, while the duplicate search runs a
DirectedRelationshipIndexSeek over 10. It exists to feed contradiction detection,
which is suspended until the corpus has a real time axis."""
from __future__ import annotations

import pytest
from graphiti_core.search.search_config import SearchResults
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance import edge_operations

from graph_extract import contradiction_gate as cg


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
        return SearchResults(edges=["sentinel-edge"])

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

    assert out.edges == ["sentinel-edge"]
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
        return SearchResults(edges=["sentinel-edge"])

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
async def test_a_delegated_failure_propagates(restore_search):
    async def _boom(clients, query, **kw):
        raise RuntimeError("neo4j down")

    edge_operations.search = _boom
    cg.install_contradiction_gate(detect_contradictions=False)
    with pytest.raises(RuntimeError, match="neo4j down"):
        await edge_operations.search(None, "f", group_ids=["g"], config=None,
                                     search_filter=SearchFilters(edge_uuids=["u"]))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_contradiction_gate.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'graph_extract.contradiction_gate'`

- [ ] **Step 3: Write the implementation**

Create `src/graph_extract/contradiction_gate.py`:

```python
"""Suspend graphiti's contradiction detection, and the O(corpus) scan feeding it.

`resolve_extracted_edges` issues TWO searches per extracted fact, and they are not
equivalent. Measured with PROFILE on the live graph:

    duplicate candidates   SearchFilters(edge_uuids=[...])
        -> DirectedRelationshipIndexSeek, 10 rows scored. Bounded by the candidate
           list; does NOT grow with the corpus.

    invalidation candidates  SearchFilters()
        -> NodeByLabelScan + Expand(All), 3,469 rows scored. O(facts):
           scan_ms = 241 + 0.0699 x facts, i.e. 42.9 s at the projected 610k-fact
           corpus, overtaking the 1.8 s dedup LLM call at ~3,900 articles -- under
           4% of the corpus.

That scan exists solely to feed contradiction detection, which does not currently
work: every one of the 140 ingest-time invalidations measured was contradiction
driven, and the inspectable ones are refinements and near-duplicates, ordered by
the sequence DocExtractor happened to crawl the pages. There is no document
revision date in the corpus to order them properly
(docs/proposals/2026-09-12-docextractor-article-timestamps.md).

So the scan is the scale blocker AND it serves the one feature that is broken.
Suspending it removes both, plus the dedup index-space confusion: with no
invalidation candidates the dedup prompt carries ONE index range instead of two,
and all 267 observed out-of-range indices landed in the second one.

Scope: `edge_operations` is ingest-only. The answer path searches via
`graphiti.search()` / `search_utils`, so retrieval is untouched by construction
rather than by a flag someone must remember to check.

Re-enablement is NOT simply flipping the flag -- see
docs/superpowers/specs/2026-09-12-suspend-contradiction-detection-design.md section 7.
"""
from __future__ import annotations

import logging
from typing import Any

from graphiti_core.search.search_config import SearchResults
from graphiti_core.utils.maintenance import edge_operations

logger = logging.getLogger(__name__)

_MARKER = "_graph_extract_contradiction_gate"


def install_contradiction_gate(*, detect_contradictions: bool) -> bool:
    """Patch `edge_operations`' own `search` reference so the unfiltered
    invalidation search never runs.

    `edge_operations` does `from graphiti_core.search.search import search`, so the
    patch must target THAT module's attribute; patching the defining module would
    not be seen.

    Returns True when the gate is installed (contradiction detection suspended),
    False when detection is left enabled. Idempotent.
    """
    if detect_contradictions:
        return False
    if getattr(edge_operations.search, _MARKER, False):
        return True

    original = edge_operations.search

    async def gated(*args: Any, **kwargs: Any) -> Any:
        # `search_filter` is the 5th positional parameter and is passed as a
        # keyword by both call sites today. Read both, so a change in graphiti's
        # call style degrades to "still skipped" rather than silently resuming a
        # corpus scan.
        search_filter = kwargs.get("search_filter")
        if search_filter is None and len(args) >= 5:
            search_filter = args[4]
        if search_filter is not None and getattr(search_filter, "edge_uuids", None) is None:
            # The invalidation search. Empty, well-formed -- never None, which
            # would fail confusingly deep inside graphiti.
            return SearchResults()
        return await original(*args, **kwargs)

    gated.__dict__[_MARKER] = True
    edge_operations.search = gated
    logger.info(
        "contradiction detection suspended: the unfiltered invalidation-candidate "
        "search is skipped (scale blocker; see BACKLOG 6/8/30)")
    return True


def is_gate_installed() -> bool:
    """True when the invalidation-candidate search is currently being skipped."""
    return bool(getattr(edge_operations.search, _MARKER, False))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_contradiction_gate.py -q`
Expected: `7 passed`

- [ ] **Step 5: Prove the tests discriminate**

Apply each mutation alone, run the file, restore with `git checkout src/graph_extract/contradiction_gate.py`, and record which tests died:

| # | mutation | must kill |
|---|---|---|
| M1 | `if search_filter is not None and ...` → `if False:` | unfiltered-skipped, positional |
| M2 | drop the positional fallback (`if search_filter is None and len(args) >= 5:` block) | positional |
| M3 | return `None` instead of `SearchResults()` | unfiltered-skipped |
| M4 | remove the `_MARKER` idempotency check | double-wrap |
| M5 | invert the filter test to `is not None` | filtered-delegated |

Every mutation must kill at least one test, and every test must die under at least one mutation. If a mutation kills nothing, the test suite has a hole — add the discriminating case before continuing.

After the last restore, confirm `git diff --stat src/` is empty.

- [ ] **Step 6: Commit**

```bash
git add src/graph_extract/contradiction_gate.py tests/unit/test_contradiction_gate.py
git commit -m "feat(ingest): gate the O(corpus) invalidation-candidate search

The invalidation search scans every fact (PROFILE: NodeByLabelScan + Expand(All),
3,469 rows) while the duplicate search is an index seek over 10. Only the first
grows with the corpus, and it feeds contradiction detection, which is measured to
produce refinements and near-duplicates ordered by crawl order.

The gate discriminates structurally on search_filter.edge_uuids, reads it both
positionally and by keyword, and returns a well-formed empty SearchResults."
```

---

### Task 2: Configuration and wiring

**Files:**
- Modify: `src/graph_extract/config.py` (add the setting beside `dedup_retry_on_strong`)
- Modify: `src/graph_extract/graphiti_client.py` (call the installer in `build_graphiti`)
- Test: `tests/unit/test_contradiction_gate_wiring.py`

**Interfaces:**
- Consumes: `install_contradiction_gate(*, detect_contradictions: bool) -> bool` from Task 1.
- Produces: `ExtractSettings.ingest_detect_contradictions: bool = False`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_contradiction_gate_wiring.py`:

```python
"""build_graphiti must install the gate, and the default must be suspended --
the behaviour it would otherwise enable is measured to be wrong."""
from __future__ import annotations

from graph_extract.config import ExtractSettings


def _settings(**kw) -> ExtractSettings:
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    base.update(kw)
    return ExtractSettings(**base)


def test_the_default_is_suspended():
    assert _settings().ingest_detect_contradictions is False


def test_the_flag_can_be_enabled_explicitly():
    assert _settings(ingest_detect_contradictions=True).ingest_detect_contradictions is True


```

Note: this file deliberately does **not** call `build_graphiti`. That function
constructs real embedder, reranker and driver clients and is only exercised from
`tests/integration/`; a unit test calling it would be fragile and slow. The wiring
itself is proven end to end in Task 3, Step 3.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_contradiction_gate_wiring.py -q`
Expected: FAIL — `AttributeError: 'ExtractSettings' object has no attribute 'ingest_detect_contradictions'`

- [ ] **Step 3: Add the setting**

In `src/graph_extract/config.py`, immediately after the `dedup_retry_on_strong` block, add:

```python
    # graphiti issues an UNFILTERED candidate search per extracted fact to feed
    # contradiction detection. PROFILE shows it scanning every fact
    # (NodeByLabelScan + Expand(All)) while the duplicate search is an index seek:
    # scan_ms = 241 + 0.0699 x facts, so 42.9 s at the projected 610k-fact corpus
    # and past the dedup LLM call's cost at ~3,900 articles -- under 4% of it.
    #
    # DEFAULT OFF. The feature that scan serves does not work: all 140 measured
    # invalidations were contradiction-driven, and the inspectable ones are
    # refinements and near-duplicates ordered by the sequence DocExtractor crawled
    # the pages, because the corpus carries no document revision date. Re-enabling
    # is NOT a flag flip -- see the spec's section 7.
    ingest_detect_contradictions: bool = False
```

- [ ] **Step 4: Wire it into `build_graphiti`**

In `src/graph_extract/graphiti_client.py`, add to the imports beside the existing `lean_edge_search` import:

```python
from graph_extract.contradiction_gate import install_contradiction_gate
```

and inside `build_graphiti`, immediately after the existing `install_lean_edge_search()` call:

```python
    # Skips the O(corpus) invalidation-candidate search (BACKLOG 6/8/30).
    # Idempotent; safe to call per build.
    install_contradiction_gate(
        detect_contradictions=s.ingest_detect_contradictions)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_contradiction_gate_wiring.py -q`
Expected: `2 passed`

- [ ] **Step 6: Run the whole unit suite and the linters**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest tests/unit -q`
Expected: `All checks passed!`, `Success: no issues found`, and `667 passed` (658 baseline + 7 gate + 2 config).

- [ ] **Step 7: Commit**

```bash
git add src/graph_extract/config.py src/graph_extract/graphiti_client.py tests/unit/test_contradiction_gate_wiring.py
git commit -m "feat(ingest): ingest_detect_contradictions flag, default suspended

Wired into build_graphiti beside the lean edge projection. Default off because
the behaviour it enables is measured to be wrong, not merely expensive."
```

---

### Task 3: Library tripwires and the end-to-end proof

**Files:**
- Modify: `tests/unit/test_contradiction_gate.py` (append the tripwire section)
- Create: `tests/integration/test_contradiction_suspended.py`
- Modify: `docs/superpowers/BACKLOG.md`

**Interfaces:**
- Consumes: `install_contradiction_gate` (Task 1), `ExtractSettings.ingest_detect_contradictions` (Task 2).
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Write the tripwire tests**

Append to `tests/unit/test_contradiction_gate.py`:

```python
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
    us graphiti grew a third call site until the gate silently mishandles it."""
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
    """With the gate installed, existing_edges is always empty. This pins that
    graphiti then invalidates nothing, so the gate needs no separate suppression."""
    assert edge_operations.resolve_edge_contradictions(None, []) == []
```

> **Corrected in the final review:** the docstring above is wrong as written and was
> rewritten in the committed test. An empty `existing_edges` empties only the cross-pair
> half of `invalidation_candidates`; the same-pair half comes from `related_edges`. A
> further tripwire, `test_same_pair_contradiction_stays_live_through_the_duplicate_candidates`,
> pins the residual path on the library's `ast`.

- [ ] **Step 2: Run them to verify they pass against the real library**

Run: `uv run --extra dev pytest tests/unit/test_contradiction_gate.py -q`
Expected: `9 passed`. If `test_resolve_extracted_edges_has_exactly_two_search_call_sites` fails, STOP and report — the gate's safety argument no longer holds.

- [ ] **Step 3: Write the integration test**

Create `tests/integration/test_contradiction_suspended.py`:

```python
"""End-to-end on a real Neo4j: with contradiction detection suspended, an ingest
invalidates nothing and local retrieval is unchanged."""
from __future__ import annotations

import pytest
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance import edge_operations

from graph_extract.contradiction_gate import install_contradiction_gate

pytestmark = pytest.mark.asyncio(loop_scope="module")

G = "backup-docs"


# The true original, not whatever is installed when this module is imported --
# another test may have gated it already.
from graphiti_core.search.search import search as _UNGATED


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
```

- [ ] **Step 4: Run the integration tests**

Run: `uv run --extra dev pytest tests/integration/test_contradiction_suspended.py -q`
Expected: `3 passed` (a Neo4j testcontainer starts; Docker must be running).

- [ ] **Step 5: Run the full gate**

The integration suite takes ~11 minutes and the Bash tool auto-backgrounds past 120s. **Do not verify with the two halves below alone** — that is exactly what hid the gate leaking from `tests/integration/test_compat_harness.py` into the unit tripwires (final review, Critical 2). Run the real CI command, `uv run --extra dev pytest -m "not live" -q`, in one process and wait for it. The halves are kept only as a fast pre-check:

```bash
uv run ruff check src tests
uv run mypy src
uv run --extra dev pytest tests/unit -q
uv run --extra dev pytest $(ls tests/integration/test_*.py | head -21 | tr '\n' ' ') -q
uv run --extra dev pytest $(ls tests/integration/test_*.py | tail -n +22 | tr '\n' ' ') -q
```

Expected: clean, clean, `667 passed`, `72 passed`, `72 passed` (141 baseline + 3 new).

- [ ] **Step 6: Update the backlog**

In `docs/superpowers/BACKLOG.md`, replace the item 8 heading line with:

```markdown
### 8. ~~Vector scan — no vector index~~ — **RESOLVED 2026-09-12 by suspending contradiction detection**
The O(corpus) scan was the invalidation-candidate search only; the duplicate search is
already a `DirectedRelationshipIndexSeek` bounded by its candidate list. Suspending
contradiction detection removes the scan entirely — no index, no ANN, no recall trade-off.
The index question returns only if contradiction detection is re-enabled; see the spec's
section 7. Earlier entries below.

### 8-scale. Vector scan — the measurement that made it a P0
```

and mark items 29 and 32 resolved by the same change, adding one line to each:

```markdown
**Resolved 2026-09-12:** with no invalidation candidates the dedup prompt carries one index
range instead of two, so the confusion is structurally impossible rather than mitigated.
```

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_contradiction_gate.py tests/integration/test_contradiction_suspended.py docs/superpowers/BACKLOG.md
git commit -m "test(ingest): tripwires and end-to-end proof for the suspended gate

Pins the three graphiti 0.30.1 facts the gate's safety rests on (search imported
by name, exactly two call sites differing in edge_uuids, empty candidates means
no invalidation) so an upgrade fails loudly. Integration asserts the invalidation
search issues no query and that local retrieval is unchanged."
```

---

## Verification

After Task 3, confirm against the spec's success criteria:

1. **No ingest query scores more rows than its candidate list** — the invalidation search issues no query at all (Task 3 integration test).
2. ~~**Zero new `invalid_at` edges** — follows from `resolve_edge_contradictions(None, []) == []` (Task 3 tripwire), pinned rather than assumed.~~ **Withdrawn in the final review.** The tripwire asserts a true but irrelevant fact: `invalidation_candidates` is not `existing_edges`. Same-pair invalidation remains live (`edge_operations.py:769-776`, `:826-839`). Criterion 2 is not delivered by this change; what is delivered is zero *cross-pair* invalidation. BACKLOG 33.
3. **Local search unchanged** — Task 3 integration test, rewritten in the final review to call `search_local` against a real Graphiti on the testcontainer and compare non-empty results (the original compared Cypher row counts around an in-process assignment and could not fail).
4. **Per-fact search work halved** — visible as the disappearance of the unfiltered search in the item 31 per-prompt timing, on the next ingest the user chooses to run. Not verifiable in CI.
5. **`dup_in_invalidation_range` falls to zero** — same: it needs a live ingest, and `dup_beyond_range` may still be non-zero, which would be new information rather than a regression.

Criteria 4 and 5 need a paid live ingest. **Do not run one** — report that they are pending and let the user decide.

## Notes for the implementer

- Do **not** repair the 140 already-invalidated facts. That asserts they were all wrong; six were hand-checked. It is a separate, reversible decision.
- Do **not** touch `node_similarity_search`. Entities grow far slower than facts (914 vs 3,469) and it has not been measured as a blocker.
- If any tripwire in Task 3 fails on first run, stop and report rather than adjusting the tripwire to pass — it exists precisely to catch that.
