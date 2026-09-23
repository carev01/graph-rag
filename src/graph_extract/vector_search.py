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
