# Index-backed similarity search (Phase B) — design

**Date:** 2026-09-23
**Status:** Design approved 2026-09-23, including creation of the two vector indexes on the
live `backup-docs` graph (schema only).
**Evidence:** `ann-dedup-probe-2026-09-23.md` (recall on real dedup pairs, tuned index),
`vector-crossover-2026-09-16.md` (latency; its recall figures are withdrawn, §0).
**Decision D2 (taken 2026-09-23):** ANN everywhere — one tuned vector index per embedding,
used by retrieval and node dedup alike, fetch depth 200.

---

## 1. Why

graphiti's `edge_similarity_search` and `node_similarity_search` are exact cosine scans
over the whole group. Measured at 0.053 ms/fact and 0.050 ms/entity, they reach ~12.5 s per
call at 250k rows and grow linearly. Every `/search/local`, `/timeline` and DRIFT follow-up
runs the edge scan; every extracted entity at ingest runs the node scan (~a dozen per
episode). Both would dominate at corpus scale (~5.3M facts).

A tuned vector index answers the same queries in 73–114 ms at 250k and finds 97.4–99.6% of
the real duplicate partners exact search finds (probe, pessimistic backgrounds).

## 2. Approach

Patch by name, the pattern already used by `lean_edge_search`, `contradiction_gate` and
`deterministic_valid_at`. Rejected: graphiti's `driver.search_interface` hook (all 14
methods route through it — fulltext, BFS, rerankers, communities — so adopting it means
reimplementing all of them); forking graphiti (invariant: the library stays pinned and
unmodified).

## 3. Routing — `src/graph_extract/vector_search.py` (new)

`install_vector_search(enabled: bool, fetch_k: int) -> bool` replaces the two functions in
every module that imported them by name:

| module | names replaced |
|---|---|
| `graphiti_core.search.search` | `edge_similarity_search`, `node_similarity_search` |
| `graphiti_core.search.search_utils` | `node_similarity_search` (used by `hybrid_node_search`) |
| `graphiti_core.utils.maintenance.node_operations` | `node_similarity_search` |

The originals are captured once, before the first patch, so re-installation is idempotent
and `enabled=False` restores them exactly. Called from `build_graphiti`, beside the other
installers, with `ExtractSettings.vector_search_enabled` and `vector_search_fetch_k`.

**Routing rule.** A call goes to the index **only if it is unbounded**:

- edges: `source_node_uuid is None`, `target_node_uuid is None`, and `search_filter` has
  every field `None`;
- nodes: `search_filter` has every field `None`.

Everything else — same-pair dedup's `SearchFilters(edge_uuids=[...])`, date, label or
property filters, endpoint-bound searches — is delegated to graphiti's original function
with the original arguments. Bounded behaviour therefore cannot change. "Every field None"
is checked against `SearchFilters.model_fields`, so a field added by a future graphiti
release counts as a filter and delegates rather than being silently ignored.

**Index query (edges):**

```cypher
CALL db.index.vector.queryRelationships($idx, $fetch_k, $search_vector)
YIELD relationship AS e, score
WHERE e.group_id IN $group_ids AND score > $min_score
MATCH (n:Entity)-[e]->(m:Entity)
WITH e, n, m, score ORDER BY score DESC LIMIT $limit
RETURN <graphiti's edge return query, as patched by lean_edge_search>
```

**Nodes:** the same with `db.index.vector.queryNodes`, `YIELD node AS n` and graphiti's
node return query. When `group_ids is None` the group predicate is omitted, matching the
original.

- Scores need no conversion: `vector.similarity.cosine` and the index both return
  (1 + cos)/2 on Neo4j (probe §"Two findings"), so graphiti's `min_score` means the same
  thing on both paths.
- Rows are parsed with graphiti's own `get_entity_edge_from_record` /
  `get_entity_node_from_record`, so callers receive identical types.
- `fetch_k = max(vector_search_fetch_k, limit)`; the filter is applied after the index
  returns, so a group or score filter can leave fewer than `limit` rows. That is the
  accepted ANN trade (D2); at one corpus-wide `group_id` (invariant #4) the group filter
  removes nothing in production.

**Fallback.** If the index query raises because the index does not exist or is not ONLINE,
the call is delegated to the original function and logged at WARNING once per process per
index and reason. Any other exception propagates. Answers stay correct, only slower, and the
event is counted.

*Amended 2026-09-23 (final-review F1, measured on `neo4j:2026.07.1-community`):* the two
cases differ in cost. A missing index fails immediately (`There is no such vector schema
index`). A **POPULATING** index does not fail fast: `queryNodes` / `queryRelationships`
**block ~30 s** server-side waiting for it, then raise `ProcedureCallFailed` "Expected index
to come online within a reasonable time" (if it comes ONLINE inside that window the query
is answered from it). So while an index populates, every unbounded similarity search costs
~30 s plus the exact scan. Each fallback also produces an ERROR line from graphiti's
`Neo4jDriver.execute_query`, which logs every failing query before re-raising; that logger
is left alone, the `fell_back` counter sizes the event.

**Other backends.** The wrappers delegate to graphiti's original (counted as
`delegated_backend`) when the driver has a `search_interface` or its provider is not
`GraphProvider.NEO4J`: the indexes and the Cypher are Neo4j's.

**Counters.** A module-level `VectorSearchStats` (`routed`, `delegated_bounded`,
`delegated_backend`, `fell_back`, per kind edge/node) with `snapshot()` and `reset()`. `graph_extract.cli ingest`
prints it with the dedup summary; `run_worker_once` logs it per batch next to
`reference_basis`. A run can then prove the index path engaged (`routed > 0`) — the
cost-awareness rule in `CLAUDE.md`.

## 4. Index lifecycle

Module constants (not settings — changing any of them requires a rebuild):

```python
EDGE_INDEX = "relates_to_fact_embedding_vec"   # (:Entity)-[:RELATES_TO {fact_embedding}]
NODE_INDEX = "entity_name_embedding_vec"       # (:Entity {name_embedding})
INDEX_OPTIONS = {
    "vector.dimensions": <ExtractSettings.embed_dim>,
    "vector.similarity_function": "cosine",
    "vector.hnsw.m": 32,
    "vector.hnsw.ef_construction": 400,
    "vector.quantization.enabled": False,
    "vector.default_search_expansion_factor": 4.0,
}
```

`async ensure_vector_indexes(driver, embed_dim, *, wait_seconds=60.0) -> None`:

1. `CREATE VECTOR INDEX … IF NOT EXISTS` for both, with the options above.
2. `CALL db.awaitIndex(<name>, <remaining>)` for **these two indexes only**, sharing one
   `wait_seconds` budget (amended 2026-09-23, F1: the original `db.awaitIndexes(3600)`
   waited on every index in the database). `awaitIndex` raises `ProcedureTimedOut` on
   timeout, which is expected; any other error propagates unless that index is FAILED.
3. Read `SHOW VECTOR INDEXES` and compare each index's schema and stored `indexConfig` with
   the expected values (as read back: `vector.quantization.type: NONE`,
   `vector.similarity_function: COSINE`, numeric equality for the rest).
4. Schema or config mismatch, or state `FAILED` → **raise** `VectorIndexMismatch` naming the
   index, the problem and the rebuild command. State `POPULATING` → log a WARNING naming the
   index and its `populationPercent` (searches wait ~30 s on it, then fall back, until it is
   ONLINE) and **return**: a populating index is healthy, and telling an operator to rebuild
   it would discard the work in flight. Never drop or rebuild automatically: at corpus scale
   a rebuild is hours.

Called from `init_indices` (so `graph_extract.cli ingest`, the worker path and `probe`
get it) and from answer-api's `_lifespan` after the driver is built, each passing
`vector_index_startup_wait_seconds`. Skipped when `vector_search_enabled` is False.

**Rebuild command:** `python -m graph_extract.cli vector-index --rebuild` drops both indexes
and recreates them via `ensure_vector_indexes` with a 3600 s wait, then reports the final
state; `vector-index` alone prints each index's state, `populationPercent` and config
against the expected. `--rebuild` requires `--yes`, since until the new index is ONLINE
each search waits ~30 s on it and then falls back to the exact scan.

## 5. Configuration (`ExtractSettings`)

| setting | default | meaning |
|---|---|---|
| `vector_search_enabled` | `True` | kill switch; `False` restores graphiti's functions and skips index management |
| `vector_search_fetch_k` | `200` | neighbours requested from the index before filtering |
| `vector_index_startup_wait_seconds` | `60.0` | how long a service start waits for the two indexes; one still POPULATING after it is logged and the start proceeds |

## 6. Testing

Hermetic unit tests (fakes) and Neo4j testcontainer integration tests; no LLM calls.

1. **Equivalence:** on a fixture with planted nearest neighbours, the index path returns the
   same uuids in the same order as the original for both kinds (small N: HNSW is exact
   there).
2. **Routing:** bounded calls (`edge_uuids`, endpoint uuids, a date filter, `node_labels`)
   reach the original — asserted with a call counter on the captured originals, not by
   reading code; unbounded calls do not.
3. **Real call path:** graphiti's `_semantic_candidate_search` (node dedup) and
   `Graphiti._search` with `EDGE_HYBRID_SEARCH_RRF` (retrieval) both increment `routed`.
4. **Lifecycle:** first call creates both indexes ONLINE with the expected config; a
   second call is a no-op; an index created with default options raises
   `VectorIndexMismatch`; a missing index at query time falls back and counts it; a
   POPULATING index is warned about, not refused, and a query against it falls back; a
   FAILED index raises.
5. **Kill switch:** `install_vector_search(False, …)` leaves every patched module
   attribute identical (`is`) to graphiti's original.
6. **Unknown filter field:** a `SearchFilters` subclass / instance with an extra non-None
   field delegates.
7. **Mutation testing** per the project's standard practice, `assert old in s` on every
   mutant.

The CI gate is unchanged: `ruff check src tests`, `mypy src`, `pytest -m "not live"`.

## 7. Acceptance on the live graph

Free; the only write is creating the two indexes (approved).

1. Run `ensure_vector_indexes` against `backup-docs`; confirm both ONLINE with the
   expected config.
2. **Retrieval:** for each of the 29 golden questions, `_retrieve_edges` with the index on
   and with `vector_search_enabled=False`; report top-k overlap and latency per path.
3. **Dedup:** ~200 entity names from the live graph through `node_similarity_search` both
   ways; report top-15 overlap.
4. Record the counters from both runs, proving each path engaged.

At 3.5k facts and 999 entities both paths should agree almost exactly and latency will
barely differ. This step proves the wiring on real data; recall at scale is the probe's
evidence, not this step's.

## 8. Out of scope

- The `SEARCH` clause (not valid Cypher on 2026.07.1; the deprecated procedures are used
  and emit a deprecation notice).
- `community_similarity_search` — global search uses its own code.
- Fulltext/BM25 — already index-backed.
- Tuning beyond the probed point (m 32 / ef_construction 400 / expansion 4), index build
  time and memory at corpus scale, and write overhead of the tuned index — measured during
  the bootstrap via the counters and Neo4j metrics, not here.

## 9. What would invalidate this

- A graphiti upgrade that renames, moves or re-imports the patched functions. Mitigation:
  the installer asserts each target attribute exists and is either graphiti's original
  (identity-checked against `search_utils`' definition) or this module's own wrapper before
  patching, and raises otherwise — a loud failure at startup, not a silent bypass.
- A graphiti caller that passes filters positionally in a way the routing rule misreads.
  Mitigation: the wrappers take graphiti's exact signatures and bind arguments by name via
  `inspect.signature(original).bind`.
