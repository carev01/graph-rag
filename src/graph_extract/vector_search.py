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

While an index is POPULATING (a fresh database, or after `vector-index
--rebuild`), Neo4j 2026.07.1 does NOT answer from it and does NOT fail fast:
`db.index.vector.queryNodes` / `queryRelationships` block ~30 s server-side
waiting for the index, then raise `ProcedureCallFailed` ("Expected index to
come online within a reasonable time"). The wrappers treat that like a missing
index and fall back to graphiti's exact scan -- so during a rebuild every
unbounded similarity search costs ~30 s of waiting PLUS the exact scan, until
the index is ONLINE (measured on a testcontainer, fix report 2026-09-23 F1).

Every fallback also emits an ERROR line from graphiti's own
`Neo4jDriver.execute_query`, which logs any failing query at ERROR with its
Cypher before re-raising; this module's WARNING is emitted once per index and
reason, but that ERROR line appears on every fallback call. graphiti's logger
is deliberately left alone -- use the `fell_back` counter to size the problem.

Spec: docs/superpowers/specs/2026-09-23-vector-index-search-design.md
"""
from __future__ import annotations

import inspect
import logging
import math
import time
from collections.abc import Mapping
from types import ModuleType
from typing import Any

from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.nodes import EntityNode, get_entity_node_from_record
from graphiti_core.search import search as _g_search
from graphiti_core.search import search_utils
from graphiti_core.utils.maintenance import node_operations
from neo4j.exceptions import ClientError

logger = logging.getLogger(__name__)

EDGE_INDEX = "relates_to_fact_embedding_vec"
NODE_INDEX = "entity_name_embedding_vec"

# Changing any of these requires `vector-index --rebuild`: Neo4j cannot alter an
# existing index's options, and `ensure_vector_indexes` refuses a mismatch.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 400
SEARCH_EXPANSION_FACTOR = 4.0
# How long a service start waits for THIS module's two indexes (never all of
# the database's: `db.awaitIndexes` would also wait on unrelated indexes). An
# index still POPULATING after it is logged and the start continues -- searches
# fall back to the exact scan (see the module docstring for the cost).
DEFAULT_STARTUP_WAIT_SECONDS = 60.0
# `vector-index --rebuild` waits this long and then reports the final state.
REBUILD_WAIT_SECONDS = 3600.0
# What `db.awaitIndex` raises when its timeout passes (unlike `db.awaitIndexes`,
# which returns silently). Anything else it raises propagates.
_AWAIT_TIMED_OUT = "Neo.ClientError.Procedure.ProcedureTimedOut"

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
        if numeric and got is not None:
            if float(got) != float(want):
                out[key] = (got, want)
        elif got != want:
            out[key] = (got, want)
    return out


async def index_status(driver) -> dict[str, dict[str, Any]]:
    """This module's vector indexes as the server reports them; absent ones are
    simply missing from the result."""
    r = await driver.execute_query(
        "SHOW VECTOR INDEXES YIELD name, state, populationPercent, entityType, "
        "labelsOrTypes, properties, options WHERE name IN $names "
        "RETURN name, state, populationPercent, entityType, labelsOrTypes, "
        "properties, options",
        names=[EDGE_INDEX, NODE_INDEX])
    return {
        rec["name"]: {
            "state": rec["state"],
            "populationPercent": rec["populationPercent"],
            "entityType": rec["entityType"],
            "labelsOrTypes": list(rec["labelsOrTypes"] or []),
            "properties": list(rec["properties"] or []),
            "config": dict((rec["options"] or {}).get("indexConfig") or {}),
        }
        for rec in r.records
    }


async def _await_our_indexes(driver, wait_seconds: float) -> None:
    """`db.awaitIndex` on each of this module's indexes, sharing one budget.

    A timeout is not an error here -- the state check after it decides. Any
    other error propagates unless the index it concerns is FAILED (then the
    state check raises with the rebuild hint instead)."""
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    for name in (EDGE_INDEX, NODE_INDEX):
        remaining = max(0, math.ceil(deadline - time.monotonic()))
        try:
            await driver.execute_query("CALL db.awaitIndex($name, $timeout)",
                                       name=name, timeout=remaining)
        except ClientError as exc:
            if exc.code == _AWAIT_TIMED_OUT:
                continue
            st = (await index_status(driver)).get(name)
            if st is None or st["state"] != "FAILED":
                raise


async def ensure_vector_indexes(driver, embed_dim: int, *,
                                wait_seconds: float = DEFAULT_STARTUP_WAIT_SECONDS) -> None:
    """Create both indexes if absent, wait up to `wait_seconds` for THEM, then
    VERIFY schema, config and state.

    Never drops or rebuilds: at corpus scale a rebuild is hours. A schema or
    config mismatch, or a FAILED index, raises with the rebuild command. An
    index still POPULATING after the wait is healthy but unfinished: it is
    logged at WARNING with its progress and the call returns -- telling an
    operator to rebuild it would throw away the work in flight.
    """
    for stmt in _create_statements(embed_dim):
        await driver.execute_query(stmt)
    await _await_our_indexes(driver, wait_seconds)
    status = await index_status(driver)
    want = expected_config(embed_dim)
    problems: list[str] = []
    populating: list[tuple[str, Any]] = []
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
        if st["state"] == "POPULATING":
            populating.append((name, st.get("populationPercent")))
        elif st["state"] != "ONLINE":
            problems.append(f"{name}: state {st['state']}")
    if problems:
        raise VectorIndexMismatch(
            "vector index check failed -- " + "; ".join(problems)
            + f". Rebuild with `{REBUILD_HINT}` (until the new index is ONLINE, "
            "each similarity search waits ~30 s on it, then falls back to the "
            "exact scan).")
    for name, pct in populating:
        logger.warning(
            "vector index %s is POPULATING (%s%% built) after a %.0f s wait; it "
            "is healthy -- let it finish. Until it is ONLINE each similarity "
            "search against it waits ~30 s, then falls back to the exact scan. "
            "Check progress with `python -m graph_extract.cli vector-index`.",
            name, _pct(pct), float(wait_seconds))
    if not populating:
        logger.info("vector indexes %s and %s ONLINE with the tuned config",
                    EDGE_INDEX, NODE_INDEX)


def _pct(value: Any) -> str:
    return "?" if value is None else f"{float(value):.1f}"


async def drop_vector_indexes(driver) -> None:
    for name in (EDGE_INDEX, NODE_INDEX):
        await driver.execute_query(f"DROP INDEX {name} IF EXISTS")


# ---------------------------------------------------------------- routing ----

# Captured ONCE, from the defining module, before anything is patched. These
# are what `enabled=False` restores and what bounded calls delegate to.
_ORIG_EDGE = search_utils.edge_similarity_search
_ORIG_NODE = search_utils.node_similarity_search
_EDGE_SIG = inspect.signature(_ORIG_EDGE)
_NODE_SIG = inspect.signature(_ORIG_NODE)

# The two ClientError messages that mean "this index cannot answer right now";
# any other error propagates. Both measured on Neo4j 2026.07.1.
_MISSING_INDEX = "no such vector schema index"
_NOT_ONLINE = "Expected index to come online within a reasonable time"
_fetch_k = 200
_warned: set[tuple[str, str]] = set()
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


def _index_unavailable(exc: ClientError) -> bool:
    """The index is absent, or exists but is not ONLINE yet (POPULATING): in
    both cases graphiti's exact scan gives the correct answer."""
    text = f"{exc.message} {exc}"
    return _MISSING_INDEX in text or _NOT_ONLINE in text


def _warn_once(index: str, exc: ClientError) -> None:
    """One WARNING per index and reason for the process's lifetime.

    Not the only log line a fallback produces: graphiti's
    `Neo4jDriver.execute_query` logs EVERY failing query at ERROR (with the
    Cypher) before re-raising, so each fallback call also emits an ERROR from
    `graphiti_core.driver.neo4j_driver`. That logger is not ours to silence;
    the `fell_back` counter says how many there were."""
    not_online = _NOT_ONLINE in f"{exc.message} {exc}"
    key = (index, "not_online" if not_online else "missing")
    if key in _warned:
        return
    _warned.add(key)
    if not_online:
        logger.warning(
            "vector index %s is not ONLINE yet (POPULATING); each similarity search "
            "waits ~30 s on it in Neo4j, then falls back to graphiti's exact scan, "
            "until it is ONLINE. Progress: `python -m graph_extract.cli "
            "vector-index`.", index)
    else:
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
        if not _index_unavailable(exc):
            raise
        _warn_once(EDGE_INDEX, exc)
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
        if not _index_unavailable(exc):
            raise
        _warn_once(NODE_INDEX, exc)
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
