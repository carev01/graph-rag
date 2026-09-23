# ANN for node dedup — does the index find real duplicates at scale?

**Date:** 2026-09-23. **Cost:** $0 (local embedder, throwaway Neo4j container; the live
graph read once, read-only). **Scripts:** `scripts/spikes/ann_dedup/` (throwaway, not CI).
**Feeds:** decision D2 of `production-readiness-review-2026-09-13.md`, Phase B.

## Question

graphiti's node dedup (`node_operations.py:439`) is **cosine only**: top 15 by
`name_embedding`, score > 0.6. An entity whose true duplicate is not in that list becomes a
duplicate node. The crossover sweep's recall@10 over random queries is the wrong metric for
this, and its fixture turned out to be near-isotropic (correction in
`vector-crossover-2026-09-16.md` §0). This probe measures the right thing directly: **of the
real duplicates exact search finds, how many does the vector index also find?**

## Method

- **Pairs:** 418 `(extracted name, existing name)` pairs that solar-pro4 judged duplicates
  in the captured `dedupe_nodes.nodes` calls, identical names excluded (graphiti resolves
  those deterministically before the search). 218 distinct targets. Embedded with the
  production embedder, which reproduces the stored `name_embedding`s exactly (cosine 1.0).
- **Background:** the 999 live entity vectors (exact twins of targets removed) plus
  synthetic vectors jittered around them, calibrated to real geometry: **tight** —
  cos-to-seed ≈ 0.81, the real nearest-neighbour median; **loose** — ≈ 0.60. Both are
  pessimistic: every synthetic entity crowds an existing AWS/Azure concept, where a real
  corpus adds 38 other vendors' concepts elsewhere in the space.
- **Truth:** exact cosine in numpy — the computation graphiti's brute query performs —
  top 15 above the threshold. **Index:** `db.index.vector.queryNodes` at fetch depth
  k = 15 / 50 / 200, filtered and cut to the same top 15.
- **Metric:** of the pairs exact search finds, the share the index also finds.
- The index configuration is read back from `SHOW INDEXES` on every run, so the variable
  under test is proven engaged rather than assumed.

## Two findings on the way

1. **Neo4j cosine is (1 + cos) / 2.** `vector.similarity.cosine` and the index score
   both (raw 0.071 → 0.5357). graphiti's `min_score = 0.6` is raw cosine **> 0.2**. Index
   and brute share the scale, so a replacement needs no conversion.
2. **The default index is compressed.** Unspecified options give
   `vector.quantization.type: SCALAR`, `vector.hnsw.m: 16`, `ef_construction: 100`,
   `default_search_expansion_factor: 1.5`. Disabling quantization *alone* resets the
   expansion factor to **1.0**.

## Results — share of exactly-found duplicates the index also finds

**Tight background** (worst case):

| entities | exact finds | default k=15 / 50 / 200 | quantization off only | **tuned** k=15 / 50 / 200 | tuned ms (k=200) |
|---:|---:|---:|---:|---:|---:|
| 10k | 217 | 0.968 / 1.000 / 1.000 | 0.940 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 19 |
| 50k | 199 | 0.960 / 0.995 / 1.000 | 0.905 / 0.995 / 1.000 | 0.995 / 1.000 / 1.000 | 39 |
| 100k | 197 | 0.741 / 0.772 / 0.812 | 0.721 / 0.766 / 0.787 | 0.934 / 0.944 / **0.985** | 48 |
| 250k | 195 | 0.703 / 0.759 / 0.795 | 0.672 / 0.749 / 0.779 | 0.913 / 0.949 / **0.974** | 73 |

**Loose background:**

| entities | exact finds | default k=15 / 50 / 200 | **tuned** k=15 / 50 / 200 | tuned ms (k=200) |
|---:|---:|---:|---:|---:|
| 10k | 234 | 0.962 / 0.991 / 1.000 | 0.996 / 1.000 / 1.000 | 18 |
| 50k | 227 | 0.960 / 0.991 / 1.000 | 0.996 / 1.000 / 1.000 | 42 |
| 100k | 226 | 0.881 / 0.920 / 0.965 | 0.973 / 0.996 / **0.996** | 81 |
| 250k | 223 | 0.865 / 0.901 / 0.951 | 0.960 / 0.991 / **0.996** | 114 |

*Tuned* = `vector.hnsw.m: 32, vector.hnsw.ef_construction: 400,
vector.quantization.enabled: false, vector.default_search_expansion_factor: 4.0`.
Exact search at 250k entities costs ~12.5 s per query (0.050 ms/entity, review §1.2).

## Reading

- **The default index is not safe for dedup past ~50k entities**: a cliff between 50k
  and 100k that over-fetch barely moves (0.81 at k=200).
- **Tuning removes most of it.** Tuned, k=200: 0.974 in the worst-case background at 250k,
  0.996 in the looser one, at 73–114 ms against ~12.5 s. The gain comes from index
  construction and search breadth; quantization alone does nothing.
- **graphiti's own top-15 limit loses far more than the index.** Exact search already
  misses 40–47% of the 368 searchable partners at every size (it keeps 15 candidates, and
  a crowded neighbourhood pushes the partner out). The tuned index adds a further 2.6% of
  what exact search finds in the worst case, 0.4% in the looser one. Those misses are
  partly recovered in practice, because graphiti shows the dedup LLM the *union* of every
  extracted entity's candidates for the episode.

## What this does not establish

- Real corpus geometry at 250k entities — both backgrounds are synthetic and pessimistic.
- Behaviour past 250k, or of HNSW parameters above these (m 32 / ef_construction 400 /
  expansion 4 is one point, not an optimum).
- Index build time and memory at these settings; write overhead was measured once, on the
  default index (+23.9% on insert at 10k).
- Edge (fact) retrieval under the tuned configuration — this probe is node dedup only.
