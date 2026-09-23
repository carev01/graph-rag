# Index-backed similarity search — results

**Date:** 2026-09-23. **Branch:** `vector-index-search`. **Spec:**
`specs/2026-09-23-vector-index-search-design.md`. **Plan:** `plans/2026-09-23-vector-index-search.md`.
**Evidence it rests on:** `ann-dedup-probe-2026-09-23.md` (recall at scale).

## What was built

- `graph_extract/vector_search.py` — two tuned vector indexes
  (`relates_to_fact_embedding_vec`, `entity_name_embedding_vec`: m=32,
  ef_construction=400, no quantization, search expansion 4), created and **verified**
  at startup, never rebuilt automatically; and wrappers patched by name into
  `graphiti_core.search.search`, `search_utils` and `node_operations` that send every
  **unbounded** similarity search to the index and delegate everything bounded
  (same-pair dedup's `edge_uuids`, endpoint uuids, any filter) to graphiti's original.
- A missing index falls back to the exact scan with a WARNING; any other error
  propagates.
- Counters (`routed` / `delegated_bounded` / `fell_back`, per edge/node) printed by
  `ingest` and logged per worker batch.
- `python -m graph_extract.cli vector-index` (status against the expected config) and
  `vector-index --rebuild --yes`.
- Settings: `VECTOR_SEARCH_ENABLED` (default on; off restores graphiti exactly) and
  `VECTOR_SEARCH_FETCH_K` (default 200).

## Live acceptance on `backup-docs`

Indexes created on the live graph (schema only, approved 2026-09-23). Census before
and after: 999 entities, 655 episodes, 655 `HAS_EPISODE`, 3,590 facts, 0 superseded —
unchanged; the only difference is two `ONLINE` vector indexes.

| path | calls | top-k agreement with exact scan | median ms exact | median ms index |
|---|---:|---:|---:|---:|
| retrieval (29 golden questions, top 10) | 29 | **1.000** (min 1.000) | 572 | 369 |
| node dedup (200 live entity names, top 15, min 0.6) | 200 | **1.000** (min 1.000) | 1,081 | 1,085 |

Counters: edge `routed=29`, node `routed=200`, `fell_back=0`, `delegated_bounded=0` —
every call provably took the index path.

**What this establishes:** the wiring is correct on real data and real embeddings —
identical results, the index engaged on every call. **What it does not:** recall at
scale. At 999 entities and 3.6k facts HNSW is effectively exact and the scan is cheap;
recall and latency at 250k rows are the probe's evidence, not this run's.

## A finding on the way: node search ships the embedding

Dedup costs ~1.08 s on **both** paths at 999 entities — the scan is not the cost.
graphiti's node return query ends `properties(n) AS attributes`, which carries the
768-float `name_embedding` for every candidate — the same waste `lean_edge_search`
removed for edges. Measured on the live graph, index path, 30 queries:

| projection | median |
|---|---:|
| graphiti's (`properties(n)`) | 1,006 ms |
| without `properties(n)` | 358 ms (**2.8×**) |

Not fixed here: entity nodes can carry ontology attributes that live in
`properties(n)`, so a lean node projection needs the same custom-attribute guard as
`lean_edge_search`. Recorded as BACKLOG 39. Every node-dedup call pays it (~a dozen per
episode).

## Verification

Four tasks, each reviewed (spec + quality); Task 3 took one fix round (process-wide
patch registry in `tests/unit/conftest.py`; `probe` wired per spec §4). Mutation testing:
8/8 routing mutants and 3/3 wiring mutants killed. Full gate green.
