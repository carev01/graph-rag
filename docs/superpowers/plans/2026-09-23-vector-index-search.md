# Index-Backed Similarity Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace graphiti's full-scan `edge_similarity_search` / `node_similarity_search` with tuned Neo4j vector-index queries for every *unbounded* call, so retrieval and node dedup stay ~100 ms as the graph grows.

**Architecture:** A new module `graph_extract/vector_search.py` owns (a) the two vector indexes — creation, verification, rebuild — and (b) wrappers patched by name into the graphiti modules that import the two functions, routing unbounded calls to `db.index.vector.query*` and delegating everything else to graphiti's original. Wired in at `build_graphiti` (routing), `init_indices` and answer-api's lifespan (index lifecycle), and the ingest/worker reports (counters).

**Tech Stack:** Python 3.12, graphiti-core 0.30.1 (pinned, never forked), Neo4j 2026.07.1 Community (no APOC, no `SEARCH` clause), neo4j async driver, typer, pytest + testcontainers.

**Spec:** `docs/superpowers/specs/2026-09-23-vector-index-search-design.md`. Evidence: `docs/superpowers/ann-dedup-probe-2026-09-23.md`.

## Global Constraints

- graphiti-core stays pinned at 0.30.1 and unmodified; extension is by patching a module's imported-by-name reference only.
- Index names: `relates_to_fact_embedding_vec` (edges, `RELATES_TO.fact_embedding`), `entity_name_embedding_vec` (nodes, `Entity.name_embedding`).
- Index options, verbatim: `vector.dimensions` = `ExtractSettings.embed_dim`, `vector.similarity_function` = `'cosine'`, `vector.hnsw.m` = 32, `vector.hnsw.ef_construction` = 400, `vector.quantization.enabled` = false, `vector.default_search_expansion_factor` = 4.0.
- As read back from `SHOW INDEXES`, the expected config is: `vector.similarity_function: 'COSINE'`, `vector.quantization.type: 'NONE'`, numeric equality for the other four.
- Settings: `vector_search_enabled: bool = True`, `vector_search_fetch_k: int = 200`.
- Routing: to the index ONLY when unbounded — edges: `source_node_uuid is None and target_node_uuid is None` and every `SearchFilters` field (per `type(f).model_fields`) is `None`; nodes: every field `None`. Everything else calls graphiti's original with the original arguments.
- Scores need no conversion: Neo4j cosine and index scores are both (1 + cos)/2.
- Fetch depth passed to the index: `max(vector_search_fetch_k, limit)`.
- Fallback only on a `neo4j.exceptions.ClientError` whose text contains `no such vector schema index`; every other exception propagates.
- Never drop or rebuild an index automatically. Rebuild is `python -m graph_extract.cli vector-index --rebuild --yes` only.
- Use the deprecated `db.index.vector.queryNodes` / `db.index.vector.queryRelationships` procedures (`SEARCH` is not valid Cypher on this server).
- Tests are hermetic: no LLM calls, no paid endpoints, never the `.env` Neo4j. Integration tests use the `extract_neo4j` / `extract_driver` testcontainer fixtures in `tests/integration/conftest.py`.
- Lint gate lints tests too: `uv run ruff check src tests` (E702: no semicolons in test fakes; E402: imports at top). Types: `uv run mypy src`.
- CI gate, one process: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live"`.
- Every mutation-testing patch asserts `old in s` before replacing, so a mutant that fails to apply is never reported as "survived".
- Subagents must not launch long-running background jobs and wait for them.
- `install_vector_search` patches graphiti modules PROCESS-WIDE and `build_graphiti` calls it, so any test that builds graphiti leaves the wrappers installed for later tests. Tests must never assume the current attribute is graphiti's original — compare with `vs._ORIG_EDGE` / `vs._ORIG_NODE` — and a container search without the indexes simply falls back (same results).

---

## File structure

| file | responsibility |
|---|---|
| `src/graph_extract/vector_search.py` (new) | index constants, DDL, `ensure_vector_indexes`, `index_status`, `drop_vector_indexes`, `VectorIndexMismatch`; routing wrappers, `install_vector_search`, stats |
| `src/graph_extract/config.py` | two new settings |
| `src/graph_extract/graphiti_client.py` | `build_graphiti` installs routing; `init_indices(graphiti, *, embed_dim=None)` ensures indexes |
| `src/graph_extract/cli.py` | passes `embed_dim` to `init_indices`; prints vector-search stats after `ingest`; new `vector-index` command |
| `src/graph_sync/semantic_worker.py` | logs + resets vector-search stats per batch |
| `src/answer_api/app.py` | lifespan ensures indexes |
| `scripts/vector_search_acceptance.py` (new) | live acceptance: both paths on the real graph |
| `tests/unit/test_vector_search_routing.py` (new) | hermetic routing tests with a fake driver |
| `tests/integration/test_vector_search.py` (new) | lifecycle + equivalence + real call path on a testcontainer |

---

### Task 1: Settings and index lifecycle

**Files:**
- Create: `src/graph_extract/vector_search.py`
- Modify: `src/graph_extract/config.py` (after `valid_at_from_content_changed`, ~line 118)
- Test: `tests/integration/test_vector_search.py`, `tests/unit/test_extract_config.py`

**Interfaces:**
- Produces: `EDGE_INDEX: str`, `NODE_INDEX: str`, `class VectorIndexMismatch(RuntimeError)`, `expected_config(embed_dim: int) -> dict[str, Any]`, `config_mismatches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]`, `async index_status(driver) -> dict[str, dict[str, Any]]` (name → `{"state", "entityType", "labelsOrTypes", "properties", "config"}`), `async ensure_vector_indexes(driver, embed_dim: int) -> None`, `async drop_vector_indexes(driver) -> None`. `driver` is anything with `async execute_query(query, **params)` returning an object with `.records` (graphiti's `GraphDriver` or a neo4j `AsyncDriver`).
- Produces settings: `ExtractSettings.vector_search_enabled: bool = True`, `ExtractSettings.vector_search_fetch_k: int = 200`.

- [ ] **Step 1: Write the failing config test**

Append to `tests/unit/test_extract_config.py`:

```python
def test_vector_search_defaults():
    from graph_extract.config import ExtractSettings
    s = ExtractSettings(_env_file=None, neo4j_uri="bolt://x", neo4j_user="u",
                        neo4j_password="p")
    assert s.vector_search_enabled is True
    assert s.vector_search_fetch_k == 200
```

If `ExtractSettings` requires further fields to construct, copy the constructor arguments from `test_extract_settings_defaults` in the same file.

- [ ] **Step 2: Write the failing lifecycle tests**

Create `tests/integration/test_vector_search.py`:

```python
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
```

Note: `test_an_index_with_default_options_is_refused_not_rebuilt` leaves a mismatched node index; `test_a_dimension_mismatch_is_refused` drops everything first, so test order within the module does not matter for correctness of the later tests — each lifecycle test that needs a clean slate drops first.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_extract_config.py::test_vector_search_defaults tests/integration/test_vector_search.py -v`
Expected: FAIL — `AttributeError`/`ImportError` for the missing settings and `graph_extract.vector_search`.

- [ ] **Step 4: Add the settings**

In `src/graph_extract/config.py`, directly after `valid_at_from_content_changed: bool = True`:

```python
    # Index-backed similarity search (Phase B, spec 2026-09-23). Unbounded
    # edge/node similarity searches -- retrieval and node dedup -- go to tuned
    # Neo4j vector indexes instead of graphiti's full cosine scan, which grows
    # linearly (~12.5 s per call at 250k rows). False restores graphiti's own
    # functions exactly and skips index management.
    vector_search_enabled: bool = True
    # Neighbours requested from the index before the group/score filter and the
    # caller's `limit` are applied. 200 measured 97.4-99.6% of exact search's
    # dedup partners at 250k entities (ann-dedup-probe-2026-09-23.md).
    vector_search_fetch_k: int = 200
```

- [ ] **Step 5: Implement the lifecycle half of `vector_search.py`**

Create `src/graph_extract/vector_search.py`:

```python
"""Index-backed similarity search for graphiti (Phase B).

graphiti's `edge_similarity_search` / `node_similarity_search` compute cosine
against every fact / entity in the group -- 0.053 ms/fact, 0.050 ms/entity,
~12.5 s per call at 250k rows and linear beyond. This module owns two tuned
Neo4j vector indexes and routes every UNBOUNDED call to them; bounded calls
(same-pair dedup's `edge_uuids`, endpoint uuids, any filter) keep graphiti's
exact query.

The tuning is not optional. With default options the index found only ~80% of
the real dedup partners exact search finds past 50k entities; m=32,
ef_construction=400, no quantization and expansion 4 measured 97.4-99.6% at
250k (ann-dedup-probe-2026-09-23.md).

Spec: docs/superpowers/specs/2026-09-23-vector-index-search-design.md
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

EDGE_INDEX = "relates_to_fact_embedding_vec"
NODE_INDEX = "entity_name_embedding_vec"

# Changing any of these requires `vector-index --rebuild`: Neo4j cannot alter an
# existing index's options, and `ensure_vector_indexes` refuses a mismatch.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 400
SEARCH_EXPANSION_FACTOR = 4.0
# `db.awaitIndexes` returns SILENTLY on timeout; the state check after it is
# what turns a still-populating index into an error.
AWAIT_INDEXES_SECONDS = 3600

REBUILD_HINT = "python -m graph_extract.cli vector-index --rebuild --yes"


class VectorIndexMismatch(RuntimeError):
    """An existing vector index does not match the configuration this code expects."""


def _options_clause(embed_dim: int) -> str:
    # Cypher does not accept parameters in an OPTIONS map, so values are
    # formatted in; int()/float() keep them un-interpolable.
    return (
        "OPTIONS {indexConfig: {"
        f"`vector.dimensions`: {int(embed_dim)}, "
        "`vector.similarity_function`: 'cosine', "
        f"`vector.hnsw.m`: {int(HNSW_M)}, "
        f"`vector.hnsw.ef_construction`: {int(HNSW_EF_CONSTRUCTION)}, "
        "`vector.quantization.enabled`: false, "
        f"`vector.default_search_expansion_factor`: {float(SEARCH_EXPANSION_FACTOR)}"
        "}}"
    )


def _create_statements(embed_dim: int) -> list[str]:
    opts = _options_clause(embed_dim)
    return [
        f"CREATE VECTOR INDEX {EDGE_INDEX} IF NOT EXISTS "
        f"FOR ()-[e:RELATES_TO]-() ON (e.fact_embedding) {opts}",
        f"CREATE VECTOR INDEX {NODE_INDEX} IF NOT EXISTS "
        f"FOR (n:Entity) ON (n.name_embedding) {opts}",
    ]


# What each index must cover, as SHOW INDEXES reports it.
_SCHEMA: dict[str, tuple[str, list[str], list[str]]] = {
    EDGE_INDEX: ("RELATIONSHIP", ["RELATES_TO"], ["fact_embedding"]),
    NODE_INDEX: ("NODE", ["Entity"], ["name_embedding"]),
}


def expected_config(embed_dim: int) -> dict[str, Any]:
    """The indexConfig as Neo4j READS IT BACK -- which differs from what is
    written: `similarity_function` comes back upper-case, and
    `quantization.enabled: false` comes back as `quantization.type: NONE`."""
    return {
        "vector.dimensions": int(embed_dim),
        "vector.similarity_function": "COSINE",
        "vector.hnsw.m": HNSW_M,
        "vector.hnsw.ef_construction": HNSW_EF_CONSTRUCTION,
        "vector.quantization.type": "NONE",
        "vector.default_search_expansion_factor": SEARCH_EXPANSION_FACTOR,
    }


def config_mismatches(actual: Mapping[str, Any],
                      expected: Mapping[str, Any]) -> dict[str, tuple[Any, Any]]:
    """`{key: (actual, expected)}` for every expected key that differs. Numbers
    compare by value (the server may return 4 for 4.0)."""
    out: dict[str, tuple[Any, Any]] = {}
    for key, want in expected.items():
        got = actual.get(key)
        numeric = (isinstance(want, (int, float)) and not isinstance(want, bool)
                   and isinstance(got, (int, float)) and not isinstance(got, bool))
        if numeric:
            if float(got) != float(want):
                out[key] = (got, want)
        elif got != want:
            out[key] = (got, want)
    return out


async def index_status(driver) -> dict[str, dict[str, Any]]:
    """This module's vector indexes as the server reports them; absent ones are
    simply missing from the result."""
    r = await driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name, state, entityType, labelsOrTypes, "
        "properties, options WHERE name IN $names "
        "RETURN name, state, entityType, labelsOrTypes, properties, options",
        names=[EDGE_INDEX, NODE_INDEX])
    return {
        rec["name"]: {
            "state": rec["state"],
            "entityType": rec["entityType"],
            "labelsOrTypes": list(rec["labelsOrTypes"] or []),
            "properties": list(rec["properties"] or []),
            "config": dict((rec["options"] or {}).get("indexConfig") or {}),
        }
        for rec in r.records
    }


async def ensure_vector_indexes(driver, embed_dim: int) -> None:
    """Create both indexes if absent, wait, then VERIFY state, schema and config.

    Never drops or rebuilds: at corpus scale a rebuild is hours, and an index
    under construction serves nothing. A mismatch raises with the rebuild command.
    """
    for stmt in _create_statements(embed_dim):
        await driver.execute_query(stmt)
    await driver.execute_query(f"CALL db.awaitIndexes({int(AWAIT_INDEXES_SECONDS)})")
    status = await index_status(driver)
    want = expected_config(embed_dim)
    problems: list[str] = []
    for name, (entity_type, labels, props) in _SCHEMA.items():
        st = status.get(name)
        if st is None:
            problems.append(f"{name}: missing after CREATE")
            continue
        if (st["entityType"], st["labelsOrTypes"], st["properties"]) != (
                entity_type, labels, props):
            problems.append(
                f"{name}: covers {st['entityType']} {st['labelsOrTypes']} "
                f"{st['properties']}, expected {entity_type} {labels} {props}")
        diff = config_mismatches(st["config"], want)
        if diff:
            problems.append(f"{name}: config differs {diff}")
        if st["state"] != "ONLINE":
            problems.append(f"{name}: state {st['state']}, not ONLINE after awaitIndexes")
    if problems:
        raise VectorIndexMismatch(
            "vector index check failed -- " + "; ".join(problems)
            + f". Rebuild with `{REBUILD_HINT}` (searches fall back to exact "
            "scans until the new index is ONLINE).")
    logger.info("vector indexes %s and %s ONLINE with the tuned config",
                EDGE_INDEX, NODE_INDEX)


async def drop_vector_indexes(driver) -> None:
    for name in (EDGE_INDEX, NODE_INDEX):
        await driver.execute_query(f"DROP INDEX {name} IF EXISTS")
```

Note: the ONLINE check is reported alongside config problems; a still-populating index with the right config also raises (with the same rebuild hint — acceptable wording; the state is named in the message).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_extract_config.py tests/integration/test_vector_search.py -v`
Expected: all PASS.

- [ ] **Step 7: Lint, types, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/graph_extract/vector_search.py src/graph_extract/config.py tests/unit/test_extract_config.py tests/integration/test_vector_search.py
git commit -m "feat(vector-search): tuned vector indexes, created and verified, never auto-rebuilt"
```

---

### Task 2: Routing wrappers, install, stats, fallback

**Files:**
- Modify: `src/graph_extract/vector_search.py` (append)
- Test: `tests/unit/test_vector_search_routing.py` (new), `tests/integration/test_vector_search.py` (append)

**Interfaces:**
- Consumes (Task 1): `EDGE_INDEX`, `NODE_INDEX`, `ensure_vector_indexes(driver, embed_dim)`, `drop_vector_indexes(driver)`.
- Produces: `install_vector_search(*, enabled: bool, fetch_k: int) -> bool`, `is_vector_search_installed() -> bool`, `stats_snapshot() -> dict[str, dict[str, int]]` (`{"edge": {"routed", "delegated_bounded", "fell_back"}, "node": {...}}`), `reset_stats() -> None`, `stats_summary() -> str`, and the wrappers `index_edge_similarity_search`, `index_node_similarity_search` (same signatures as graphiti's).

- [ ] **Step 1: Write the failing unit tests**

Create `tests/unit/test_vector_search_routing.py`:

```python
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
```

`test_install_refuses_a_foreign_function` must leave no module patched — the refusal raises before any `setattr`, because `install_vector_search` validates all four targets before patching any (see Step 3). `monkeypatch` restores `node_operations`.

- [ ] **Step 2: Run the unit tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_vector_search_routing.py -v`
Expected: FAIL — `AttributeError: module 'graph_extract.vector_search' has no attribute 'index_edge_similarity_search'`.

- [ ] **Step 3: Implement routing**

Append to `src/graph_extract/vector_search.py` (and add the imports to the top of the file, below `from typing import Any`):

```python
import inspect
from types import ModuleType

from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.nodes import EntityNode, get_entity_node_from_record
from graphiti_core.search import search as _g_search
from graphiti_core.search import search_utils
from graphiti_core.utils.maintenance import node_operations
from neo4j.exceptions import ClientError
```

```python
# ---------------------------------------------------------------- routing ----

# Captured ONCE, from the defining module, before anything is patched. These
# are what `enabled=False` restores and what bounded calls delegate to.
_ORIG_EDGE = search_utils.edge_similarity_search
_ORIG_NODE = search_utils.node_similarity_search
_EDGE_SIG = inspect.signature(_ORIG_EDGE)
_NODE_SIG = inspect.signature(_ORIG_NODE)

_MISSING_INDEX = "no such vector schema index"
_fetch_k = 200
_warned: set[str] = set()
_stats: dict[str, dict[str, int]] = {
    kind: {"routed": 0, "delegated_bounded": 0, "fell_back": 0} for kind in ("edge", "node")
}


def stats_snapshot() -> dict[str, dict[str, int]]:
    return {kind: dict(counts) for kind, counts in _stats.items()}


def reset_stats() -> None:
    for counts in _stats.values():
        for key in counts:
            counts[key] = 0


def stats_summary() -> str:
    return " | ".join(
        f"{kind} routed={c['routed']} delegated_bounded={c['delegated_bounded']} "
        f"fell_back={c['fell_back']}" for kind, c in _stats.items())


def _is_unbounded(search_filter: Any) -> bool:
    """Every filter field None. Iterates the model's DECLARED fields, so a field a
    future graphiti adds counts as a filter and delegates instead of being
    silently dropped."""
    if search_filter is None:
        return True
    return all(getattr(search_filter, name) is None
               for name in type(search_filter).model_fields)


def _is_missing_index(exc: ClientError) -> bool:
    return _MISSING_INDEX in f"{exc.message} {exc}"


def _warn_once(index: str) -> None:
    if index not in _warned:
        _warned.add(index)
        logger.warning(
            "vector index %s is missing; similarity search is falling back to "
            "graphiti's full scan (correct but slow). Restart a process that runs "
            "ensure_vector_indexes, or `%s`.", index, REBUILD_HINT)


def _group_clause(var: str, group_ids: list[str] | None) -> str:
    return f"{var}.group_id IN $group_ids AND " if group_ids is not None else ""


async def index_edge_similarity_search(*args: Any, **kwargs: Any) -> list[EntityEdge]:
    """Drop-in for graphiti's `edge_similarity_search` (same signature)."""
    a = _EDGE_SIG.bind(*args, **kwargs)
    a.apply_defaults()
    p = a.arguments
    if not (p["source_node_uuid"] is None and p["target_node_uuid"] is None
            and _is_unbounded(p["search_filter"])):
        _stats["edge"]["delegated_bounded"] += 1
        return await _ORIG_EDGE(*args, **kwargs)
    driver, limit = p["driver"], p["limit"]
    query = (
        "CALL db.index.vector.queryRelationships($index_name, $fetch_k, $search_vector) "
        "YIELD relationship AS e, score "
        f"WHERE {_group_clause('e', p['group_ids'])}score > $min_score "
        "MATCH (n:Entity)-[e]->(m:Entity) "
        "WITH e, n, m, score "
        # Resolved at CALL time, so lean_edge_search's patch of this name applies.
        "RETURN " + search_utils.get_entity_edge_return_query(driver.provider)
        + " ORDER BY score DESC LIMIT $limit")
    try:
        records, _, _ = await driver.execute_query(
            query, index_name=EDGE_INDEX, fetch_k=max(_fetch_k, int(limit)),
            search_vector=p["search_vector"], group_ids=p["group_ids"],
            min_score=p["min_score"], limit=int(limit), routing_="r")
    except ClientError as exc:
        if not _is_missing_index(exc):
            raise
        _warn_once(EDGE_INDEX)
        _stats["edge"]["fell_back"] += 1
        return await _ORIG_EDGE(*args, **kwargs)
    _stats["edge"]["routed"] += 1
    return [get_entity_edge_from_record(r, driver.provider) for r in records]


async def index_node_similarity_search(*args: Any, **kwargs: Any) -> list[EntityNode]:
    """Drop-in for graphiti's `node_similarity_search` (same signature)."""
    a = _NODE_SIG.bind(*args, **kwargs)
    a.apply_defaults()
    p = a.arguments
    if not _is_unbounded(p["search_filter"]):
        _stats["node"]["delegated_bounded"] += 1
        return await _ORIG_NODE(*args, **kwargs)
    driver, limit = p["driver"], p["limit"]
    query = (
        "CALL db.index.vector.queryNodes($index_name, $fetch_k, $search_vector) "
        "YIELD node AS n, score "
        f"WHERE {_group_clause('n', p['group_ids'])}score > $min_score "
        "WITH n, score "
        "RETURN " + search_utils.get_entity_node_return_query(driver.provider)
        + " ORDER BY score DESC LIMIT $limit")
    try:
        records, _, _ = await driver.execute_query(
            query, index_name=NODE_INDEX, fetch_k=max(_fetch_k, int(limit)),
            search_vector=p["search_vector"], group_ids=p["group_ids"],
            min_score=p["min_score"], limit=int(limit), routing_="r")
    except ClientError as exc:
        if not _is_missing_index(exc):
            raise
        _warn_once(NODE_INDEX)
        _stats["node"]["fell_back"] += 1
        return await _ORIG_NODE(*args, **kwargs)
    _stats["node"]["routed"] += 1
    return [get_entity_node_from_record(r, driver.provider) for r in records]


# Every module that imported the functions BY NAME. Patching the defining
# module alone would be invisible to these.
_TARGETS: list[tuple[ModuleType, str, str]] = [
    (_g_search, "edge_similarity_search", "edge"),
    (_g_search, "node_similarity_search", "node"),
    (search_utils, "node_similarity_search", "node"),  # hybrid_node_search
    (node_operations, "node_similarity_search", "node"),  # node dedup
]


def _pair(kind: str) -> tuple[Any, Any]:
    if kind == "edge":
        return _ORIG_EDGE, index_edge_similarity_search
    return _ORIG_NODE, index_node_similarity_search


def install_vector_search(*, enabled: bool, fetch_k: int) -> bool:
    """Route unbounded similarity searches to the vector indexes (enabled) or
    restore graphiti's originals exactly (disabled). Idempotent.

    Validates ALL targets before patching ANY: each must currently be graphiti's
    original or this module's wrapper. A graphiti upgrade that moves or re-wraps
    one fails loudly here rather than leaving a silent full-scan path."""
    global _fetch_k
    if fetch_k < 1:
        raise ValueError(f"vector_search_fetch_k must be >= 1, got {fetch_k}")
    for module, name, kind in _TARGETS:
        current = getattr(module, name, None)
        original, wrapper = _pair(kind)
        if current is not original and current is not wrapper:
            raise RuntimeError(
                f"{module.__name__}.{name} is neither graphiti's original nor "
                "graph_extract.vector_search's wrapper; refusing to patch")
    _fetch_k = fetch_k
    for module, name, kind in _TARGETS:
        original, wrapper = _pair(kind)
        setattr(module, name, wrapper if enabled else original)
    return enabled


def is_vector_search_installed() -> bool:
    return all(getattr(module, name) is _pair(kind)[1] for module, name, kind in _TARGETS)
```

Note on the `originals` unit fixture: it monkeypatches `vs._ORIG_EDGE`/`_ORIG_NODE`, which the wrappers read at call time (module globals) — that is what makes delegation observable. `install_vector_search`'s identity check also reads them, so the install tests do NOT use that fixture.

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_vector_search_routing.py -v`
Expected: all PASS.

- [ ] **Step 5: Write the failing integration tests (equivalence, real call path, fallback)**

Append to `tests/integration/test_vector_search.py` (merge the imports into the file's import block at the top — E402):

```python
import math
from collections.abc import Iterable
from typing import Any

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
```

```python
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
```

Note on `test_same_pair_dedup_stays_on_the_exact_scan`: `search_utils.edge_similarity_search` is NOT patched (only `search_utils.node_similarity_search` is — see `_TARGETS`), so the first call runs graphiti's original directly and touches no counter; the second goes through the patched `search.edge_similarity_search` and must be counted as delegated. Expect exactly 1.

If `EntityNode(...)` rejects the arguments shown, check its required fields in `graphiti_core/nodes.py` and supply them; `name_embedding` must NOT be set (`_semantic_candidate_search` embeds the name itself via the fixed embedder).

- [ ] **Step 6: Run the integration tests**

Run: `uv run --extra dev pytest tests/integration/test_vector_search.py -v`
Expected: all PASS.

- [ ] **Step 7: Mutation-test the routing (record results in the report, do not commit mutants)**

For each mutant: copy `src/graph_extract/vector_search.py`, apply the edit with a script that does `assert old in s` then `s.replace(old, new, 1)`, run `uv run --extra dev pytest tests/unit/test_vector_search_routing.py tests/integration/test_vector_search.py -q`, confirm at least one test FAILS, restore the file. Mutants:

1. `p["source_node_uuid"] is None and p["target_node_uuid"] is None` → `p["source_node_uuid"] is None`
2. `return all(getattr(search_filter, name) is None` → `return all(getattr(search_filter, name) is None or True`
3. `fetch_k=max(_fetch_k, int(limit))` (edge occurrence) → `fetch_k=int(limit)`
4. `if not _is_missing_index(exc):\n            raise` (first occurrence) → `if False:\n            raise`
5. `f"WHERE {_group_clause('e', p['group_ids'])}score > $min_score "` → `"WHERE score > $min_score "`
6. `(node_operations, "node_similarity_search", "node"),  # node dedup` → `` (remove the line)
7. `if current is not original and current is not wrapper:` → `if False:`
8. `+ " ORDER BY score DESC LIMIT $limit")` (first occurrence, the edge query) → `+ " LIMIT $limit")`

Any survivor: add a test that kills it, then re-run.

- [ ] **Step 8: Lint, types, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/graph_extract/vector_search.py tests/unit/test_vector_search_routing.py tests/integration/test_vector_search.py
git commit -m "feat(vector-search): route unbounded similarity searches to the vector indexes"
```

---

### Task 3: Wiring — build, startup, reports, CLI

**Files:**
- Modify: `src/graph_extract/graphiti_client.py` (`build_graphiti` ~line 206-217; `init_indices` ~line 277)
- Modify: `src/graph_extract/cli.py` (`_build_ingest_driver` ~line 94; `ingest` report ~line 273; new `vector-index` command)
- Modify: `src/graph_sync/semantic_worker.py` (batch summary ~line 190-201)
- Modify: `src/answer_api/app.py` (`_lifespan`, after the driver/graphiti are built, before `app.state` assignment)
- Modify tests: `tests/unit/test_cli_ingest_wiring.py:49`, `tests/unit/test_cli_dedup_wiring.py:62`, `tests/unit/test_extract_cli.py:176,197` — their `init_indices` fakes must accept `**kwargs`
- Test: `tests/unit/test_vector_search_wiring.py` (new)

**Interfaces:**
- Consumes (Tasks 1-2): `install_vector_search(*, enabled, fetch_k)`, `ensure_vector_indexes(driver, embed_dim)`, `drop_vector_indexes(driver)`, `index_status(driver)`, `expected_config(embed_dim)`, `config_mismatches(...)`, `stats_summary()`, `reset_stats()`.
- Produces: `init_indices(graphiti, *, embed_dim: int | None = None) -> None`; typer command `vector-index [--rebuild] [--yes]`.

- [ ] **Step 1: Write the failing wiring tests**

Create `tests/unit/test_vector_search_wiring.py`:

```python
"""The routing and the index lifecycle are wired where the spec says."""
from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from graph_extract import cli
from graph_extract import graphiti_client as gc
from graph_extract import vector_search as vs
from graph_extract.config import ExtractSettings


def _settings(**over: Any) -> ExtractSettings:
    base: dict[str, Any] = dict(_env_file=None, neo4j_uri="bolt://x", neo4j_user="u",
                                neo4j_password="p")
    base.update(over)
    return ExtractSettings(**base)


def test_build_graphiti_installs_routing_per_the_setting(monkeypatch):
    seen: list[tuple[bool, int]] = []
    monkeypatch.setattr(gc, "install_vector_search",
                        lambda *, enabled, fetch_k: seen.append((enabled, fetch_k)) or enabled)
    s = _settings(vector_search_enabled=False, vector_search_fetch_k=77,
                  embed_base_url="http://e", llm_base_url="http://l")
    g = gc.build_graphiti(s)
    assert seen == [(False, 77)]
    del g


@pytest.mark.asyncio
async def test_init_indices_ensures_vector_indexes_only_when_given_a_dim(monkeypatch):
    calls: list[int] = []

    async def _ensure(driver, embed_dim):
        calls.append(embed_dim)

    class _G:
        driver = object()

        async def build_indices_and_constraints(self):
            return None

    monkeypatch.setattr(gc, "ensure_vector_indexes", _ensure)
    await gc.init_indices(_G())
    assert calls == []
    await gc.init_indices(_G(), embed_dim=768)
    assert calls == [768]


def test_vector_index_rebuild_refuses_without_yes(monkeypatch):
    monkeypatch.setattr(cli, "get_extract_settings", lambda: _settings())
    result = CliRunner().invoke(cli.app, ["vector-index", "--rebuild"])
    assert result.exit_code != 0
    assert "--yes" in result.output
```

If `build_graphiti` needs more settings than shown to construct offline, add them to `_settings(...)` from `tests/unit/test_extract_config.py`. If `cli` does not import `get_extract_settings` by that name, patch the name it does use.

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/unit/test_vector_search_wiring.py -v`
Expected: FAIL (`install_vector_search` / `ensure_vector_indexes` not attributes of `graphiti_client`; no `vector-index` command).

- [ ] **Step 3: Wire `graphiti_client.py`**

Add to the imports: `from graph_extract.vector_search import ensure_vector_indexes, install_vector_search`.

In `build_graphiti`, after the `install_deterministic_valid_at(...)` line:

```python
    # Unbounded edge/node similarity searches -> tuned vector indexes (Phase B).
    # Idempotent; enabled=False restores graphiti's own functions exactly.
    install_vector_search(enabled=s.vector_search_enabled,
                          fetch_k=s.vector_search_fetch_k)
```

Replace `init_indices`:

```python
async def init_indices(graphiti: Graphiti, *, embed_dim: int | None = None) -> None:
    """graphiti's own indices, plus -- when `embed_dim` is given -- the tuned
    vector indexes, created if absent and VERIFIED (a mismatch raises; nothing
    is ever rebuilt automatically)."""
    await graphiti.build_indices_and_constraints()
    if embed_dim is not None:
        await ensure_vector_indexes(graphiti.driver, embed_dim)
```

- [ ] **Step 4: Wire `cli.py`**

In `_build_ingest_driver`, replace `await init_indices(graphiti)` (the strong-tier one, ~line 94) with:

```python
        await init_indices(
            graphiti,
            embed_dim=settings.embed_dim if settings.vector_search_enabled else None)
```

Leave the cheap-tier `await init_indices(cheap_graphiti)` unchanged (same database; the strong call already ensured the vector indexes).

In `ingest`, directly after `typer.echo(ingest_driver.timings.report())`:

```python
            # Phase B: proves the index path engaged (routed > 0) or shows it
            # falling back to the full scan.
            typer.echo(f"vector search: {vector_search.stats_summary()}")
```

with `from graph_extract import vector_search` added to the imports.

Add the command (next to the other `@app.command` definitions):

```python
@app.command("vector-index")
def vector_index(
    rebuild: bool = typer.Option(False, "--rebuild",
                                 help="drop and recreate both vector indexes"),
    yes: bool = typer.Option(False, "--yes", help="confirm --rebuild"),
) -> None:
    """Show the vector indexes against the expected config, or rebuild them.

    Rebuild drops both; similarity searches fall back to graphiti's full scan
    until the new indexes are ONLINE -- hours at corpus scale."""
    if rebuild and not yes:
        typer.echo("--rebuild drops both vector indexes; searches fall back to full "
                   "scans until they are rebuilt. Re-run with --yes to confirm.")
        raise typer.Exit(code=2)

    async def _run() -> None:
        settings = get_extract_settings()
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        try:
            if rebuild:
                await vector_search.drop_vector_indexes(driver)
                await vector_search.ensure_vector_indexes(driver, settings.embed_dim)
            status = await vector_search.index_status(driver)
            want = vector_search.expected_config(settings.embed_dim)
            _dump({name: {**st, "mismatches": vector_search.config_mismatches(
                st["config"], want)} for name, st in status.items()})
        finally:
            await driver.close()

    asyncio.run(_run())
```

Use the imports `cli.py` already has for `asyncio`, `AsyncGraphDatabase` and `get_extract_settings`; add any that are missing.

Update the three test fakes so they accept the new keyword: in `tests/unit/test_cli_ingest_wiring.py` change `async def _init_indices(graphiti):` to `async def _init_indices(graphiti, **_kwargs):`; in `tests/unit/test_cli_dedup_wiring.py` and `tests/unit/test_extract_cli.py`, give the functions assigned to `init_indices` a `**_kwargs` parameter the same way.

- [ ] **Step 5: Wire the worker**

In `src/graph_sync/semantic_worker.py`, inside `if jobs:` directly after the `logger.info("semantic batch: jobs=%d ...` call:

```python
        # Phase B: per-batch proof the index path engaged; reset so each batch
        # reports its own counts.
        logger.info("semantic batch vector search: %s", vector_search.stats_summary())
        vector_search.reset_stats()
```

with `from graph_extract import vector_search` at the top.

- [ ] **Step 6: Wire answer-api**

In `src/answer_api/app.py` `_lifespan`, directly after the `driver = await _build_driver(settings)` guard block (graphiti and driver both built), add:

```python
    # Verify (and on a fresh database create) the tuned vector indexes before
    # serving: a mismatch refuses to start rather than serving full scans.
    if settings.vector_search_enabled:
        try:
            await ensure_vector_indexes(graphiti.driver, settings.embed_dim)
        except Exception:
            await graphiti.close()
            await driver.close()
            raise
```

with `from graph_extract.vector_search import ensure_vector_indexes` in the imports. If a unit test exercises `_lifespan` with fakes, monkeypatch `answer_api.app.ensure_vector_indexes` there with an async no-op.

- [ ] **Step 7: Run the full gate**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live"`
Expected: all green. Any test now failing because a real `build_graphiti` + container search fell back is acceptable ONLY if it passes; a failure means the wiring changed behaviour — investigate, do not loosen the test.

- [ ] **Step 8: Commit**

```bash
git add src/graph_extract/graphiti_client.py src/graph_extract/cli.py src/graph_sync/semantic_worker.py src/answer_api/app.py tests/unit/test_vector_search_wiring.py tests/unit/test_cli_ingest_wiring.py tests/unit/test_cli_dedup_wiring.py tests/unit/test_extract_cli.py
git commit -m "feat(vector-search): wire routing, index lifecycle, reports and vector-index CLI"
```

---

### Task 4: Live acceptance harness

The controller — not a subagent — runs this against the live graph; the subagent writes and unit-tests it only.

**Files:**
- Create: `scripts/vector_search_acceptance.py`
- Test: `tests/unit/test_vector_search_acceptance.py`

**Interfaces:**
- Consumes: `build_graphiti`, `build_embedder` (`graph_extract.graphiti_client`), `get_extract_settings`, `vector_search.ensure_vector_indexes`, `vector_search._ORIG_EDGE/_ORIG_NODE`, `index_edge_similarity_search`, `index_node_similarity_search`, `stats_snapshot`, `reset_stats`.
- Produces: `overlap_at(a: list[str], b: list[str], k: int) -> float` and a printed report.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_vector_search_acceptance.py`:

```python
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "vsa", Path(__file__).parents[2] / "scripts" / "vector_search_acceptance.py")
vsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vsa)


def test_overlap_at_counts_shared_uuids_in_the_top_k():
    assert vsa.overlap_at(["a", "b", "c"], ["c", "b", "x"], 3) == 2 / 3
    assert vsa.overlap_at(["a", "b"], ["a", "b"], 2) == 1.0


def test_overlap_of_two_empty_lists_is_perfect_not_zero():
    """Both paths agreeing on 'nothing above the threshold' is agreement."""
    assert vsa.overlap_at([], [], 10) == 1.0


def test_overlap_when_only_one_side_is_empty_is_zero():
    assert vsa.overlap_at(["a"], [], 10) == 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/unit/test_vector_search_acceptance.py -v`
Expected: FAIL (file not found).

- [ ] **Step 3: Implement the harness**

Create `scripts/vector_search_acceptance.py`:

```python
"""Live acceptance for index-backed similarity search (spec §7).

Against the .env graph: creates/verifies the two vector indexes (the ONLY write,
schema, approved 2026-09-23), then runs the same searches through graphiti's
exact scan and through the index and reports agreement and latency.
  - retrieval: the 29 router golden questions, edge search, top 10
  - dedup:     up to 200 live entity names, node search, top 15, min 0.6
Free: the embedder is local and no LLM is called.

    uv run python scripts/vector_search_acceptance.py
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

from graphiti_core.search.search_filters import SearchFilters
from neo4j import RoutingControl

from graph_extract import vector_search as vs
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_embedder, build_graphiti

GOLDEN = Path("src/answer_api/router_golden.json")
EDGE_K, NODE_K, MIN_SCORE, MAX_NAMES = 10, 15, 0.6, 200


def overlap_at(a: list[str], b: list[str], k: int) -> float:
    a, b = a[:k], b[:k]
    if not a and not b:
        return 1.0
    return len(set(a) & set(b)) / max(len(a), len(b))


async def _timed(coro) -> tuple[float, list]:
    t = time.perf_counter()
    out = await coro
    return (time.perf_counter() - t) * 1000.0, out


async def main() -> None:
    s = get_extract_settings()
    graphiti = build_graphiti(s)
    embedder = build_embedder(s)
    d = graphiti.driver
    try:
        await vs.ensure_vector_indexes(d, s.embed_dim)
        vs.reset_stats()
        questions = [q["question"] for q in json.loads(GOLDEN.read_text())]
        qvecs = await embedder.create_batch(questions)
        e_ov, e_ms_x, e_ms_i = [], [], []
        for v in qvecs:
            tx, ex = await _timed(vs._ORIG_EDGE(d, v, None, None, SearchFilters(),
                                                [s.group_id], EDGE_K, MIN_SCORE))
            ti, ei = await _timed(vs.index_edge_similarity_search(
                d, v, None, None, SearchFilters(), [s.group_id], EDGE_K, MIN_SCORE))
            e_ov.append(overlap_at([x.uuid for x in ex], [x.uuid for x in ei], EDGE_K))
            e_ms_x.append(tx)
            e_ms_i.append(ti)
        r = await d.execute_query(
            "MATCH (n:Entity {group_id:$g}) RETURN n.name AS name ORDER BY n.uuid LIMIT $k",
            g=s.group_id, k=MAX_NAMES, routing_=RoutingControl.READ)
        names = [rec["name"].replace("\n", " ") for rec in r.records]
        nvecs = await embedder.create_batch(names)
        n_ov, n_ms_x, n_ms_i = [], [], []
        for v in nvecs:
            tx, nx = await _timed(vs._ORIG_NODE(d, v, SearchFilters(), [s.group_id],
                                                NODE_K, MIN_SCORE))
            ti, ni = await _timed(vs.index_node_similarity_search(
                d, v, SearchFilters(), [s.group_id], NODE_K, MIN_SCORE))
            n_ov.append(overlap_at([x.uuid for x in nx], [x.uuid for x in ni], NODE_K))
            n_ms_x.append(tx)
            n_ms_i.append(ti)
        stats = vs.stats_snapshot()
        print(f"retrieval: {len(e_ov)} questions, top-{EDGE_K} overlap mean "
              f"{statistics.fmean(e_ov):.3f} min {min(e_ov):.3f}; median ms exact "
              f"{statistics.median(e_ms_x):.0f} index {statistics.median(e_ms_i):.0f}")
        print(f"dedup:     {len(n_ov)} names, top-{NODE_K} overlap mean "
              f"{statistics.fmean(n_ov):.3f} min {min(n_ov):.3f}; median ms exact "
              f"{statistics.median(n_ms_x):.0f} index {statistics.median(n_ms_i):.0f}")
        print(f"counters:  {stats}")
        if stats["edge"]["routed"] != len(e_ov) or stats["node"]["routed"] != len(n_ov):
            raise SystemExit("index path did not engage on every call -- see counters")
    finally:
        await embedder.client.close()
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_vector_search_acceptance.py -v`
Expected: PASS.

- [ ] **Step 5: Lint, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add scripts/vector_search_acceptance.py tests/unit/test_vector_search_acceptance.py
git commit -m "feat(scripts): live acceptance for index-backed similarity search"
```

- [ ] **Step 6 (controller only): run acceptance on the live graph and record it**

Run: `uv run python scripts/vector_search_acceptance.py`. Write the output into `docs/superpowers/vector-index-search-2026-09-23.md` (what was built, the acceptance numbers, counters, what is not established — recall at scale is the probe's evidence). Update `CLAUDE.md` "What's built" and the `graph_extract/` module list with `vector_search.py`, and mark Phase B done in `docs/superpowers/BACKLOG.md` (item 8 lineage). Commit.
