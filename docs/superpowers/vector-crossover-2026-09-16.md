# Vector-index crossover — results

**Date:** sweep run 2026-09-16/17; written up 2026-09-22.
**Spec:** `specs/2026-09-16-vector-index-crossover-design.md`. **Harness:** `scripts/vector_crossover.py`.
**Setup:** throwaway `neo4j:2026.07.1-community` testcontainer, default heap and page cache,
default HNSW parameters. Vectors resampled from the live graph's 3,469 real
`fact_embedding` values (Gaussian noise σ=0.15), so recall figures are directional only
(spec §12). No production code, no paid calls, the live graph untouched.

**A note on provenance.** The harness writes its report to this path; that file was lost
before it was committed (the working tree moved between branches and the run's `/tmp` log
did not survive a reboot). The tables below are recovered from the session transcript of
the run. Figures marked *derived* were computed from the printed summary (ms/row × rows,
brute ÷ speedup) rather than copied from the harness's own table.

## 0. CORRECTION (2026-09-23) — the recall figures below are not evidence

Two defects found by the follow-up dedup probe. **The latency conclusions stand; the
recall conclusions in §1, §2 and §5 do not.**

1. **The fixture was near-isotropic, not "resampled real embeddings".** `resample_one`
   adds N(0, 0.15) noise *per dimension* to a 768-d unit vector — a noise vector of norm
   ≈ 0.15·√768 ≈ 4.2, four times the signal. Measured: cos(seed, resampled) median
   **0.231**, against **0.466** for a random pair of real entity names and **0.806** for a
   real nearest neighbour. The sweep measured HNSW on data far more uniform than real
   embeddings — the very case `VectorPool`'s docstring warns makes an index look worse
   than reality.
2. **Neo4j cosine is normalised.** Both `vector.similarity.cosine` and the vector-index
   score return **(1 + cos) / 2** (probed: raw cosine 0.071 scores 0.5357). graphiti's
   `min_score = 0.6` is therefore **raw cosine > 0.2**. The two queries in this harness
   share the scale, so this did not bias the comparison, but any threshold read off these
   numbers must be converted.

Recall for the path where it matters, node dedup, is measured directly on real duplicate
pairs in `ann-dedup-probe-2026-09-23.md`.

## 1. Edges — `RELATES_TO.fact_embedding`

| rows | brute ms | index ms | speedup | control dev | r@10, k=10 | k=50 | k=200 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 571 *derived* | 20 *derived* | 28x | 4.3% | 0.660 | 0.890 | 0.980 |
| 50,000 | 2,665 *derived* | 24 *derived* | 110x | 1.5% | 0.575 | 0.755 | 0.935 |
| 100,000 | 5,910 *derived* | 33 *derived* | 181x | 0.2% | 0.530 | 0.780 | 0.920 |
| 250,000 | 15,000 *derived* | 52 *derived* | **288x** | 0.3% | **0.400** | **0.605** | **0.835** |

Brute-force slope **0.0602 ms/row**, against **0.053 ms/fact** measured independently on
the live graph (production review §1.2): the fixture reproduces production's scan cost.
The index grew 2.6x while rows grew 25x.

The sweep reached 250k, not the planned 1M (a first attempt died in `wipe()` on
`MemoryPoolOutOfMemoryError`, fixed in `0b04283`; the 500k and 1M steps were never run).
250k already settles the latency question; it leaves the recall trend with four points.

The control (graphiti's brute-force query re-run with the index present) stays within
0.2–4.3% of brute force at every step: Neo4j does not consult a vector index except through
the procedures, and no speedup here came from cache warmth.

## 2. Nodes — `Entity.name_embedding`

| rows | brute ms | index ms | control ms | r@10, k=10 | k=50 | k=200 |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 617 | 22 | 711 | 0.660 | 0.890 | 0.980 |
| 50,000 | 3,487 | 24 | 3,485 | 0.500 | 0.750 | 0.915 |
| 100,000 | 7,614 | 34 | 7,622 | 0.515 | 0.750 | 0.920 |

Same shape as edges. The 250k node step was not run.

## 3. Write overhead (B4)

Measured on the 10k smoke step only: **2,964 → 3,672 ms per 1k inserts, +23.9%**. The
per-step figures from the full sweep were not recovered. The number that caused the
2026-09-11 index to be dropped is therefore now measured once — about a quarter on the
*insert*, which is a small fraction of an episode's ~38 s wall — and not as a trend.

## 4. Projection

| facts | brute force per query | index | basis |
|---:|---:|---:|---|
| 250k | 15 s | 52 ms | measured |
| 870k (pilot + Veeam) | ~52 s | ~73–104 ms | *inference* |
| 5.3M (full corpus) | ~319 s | ~73–104 ms | *inference* |

Brute force is linear throughout the measured range; the index-latency band is a
log-scaling guess and should be treated as such.

## 5. Reading

**Latency: Phase B's implementation half is justified.** At the full corpus the scan
graphiti runs on every `/search/local`, `/timeline` and DRIFT follow-up would take minutes,
and the node-dedup scan at ingest is the same shape. The index keeps both under ~100 ms.

**Recall — WITHDRAWN, see §0.** *As originally written:* it degrades with N, and over-fetch slows the decline but does not stop it.
This is the question spec §8 existed to answer, and the answer is the unfavourable one:

- k=10 falls 0.66 → 0.40 over 25x more rows.
- k=200 (ask for 200, keep the best 10) falls 0.98 → 0.835.
- Nodes flatten between 50k and 100k (0.915 → 0.920 at k=200), edges do not.

So the spec's "if recall@10 at k=50 is flat in N, D2 dissolves" branch did **not**
happen. D2 has to be decided, and per spec §13 the decision is not symmetric:

- **Retrieval** has no bounded-candidate alternative (it exists to match differently
  worded text), and is backed by BM25 and the reranker, so an ANN miss costs one of
  several candidates rather than the answer. ANN with over-fetch is the only viable
  option there.
- **Node dedup** turns every missed neighbour into a duplicate entity that the exact-name
  merge pass will not catch when the names differ. That is where recall matters.

**Levers not yet measured:** HNSW construction parameters (`vector.hnsw.m`,
`vector.hnsw.ef_construction`) were left at defaults, and k=10 recall of 0.66 at only 10k
rows suggests the defaults are conservative. Larger over-fetch (k=500+) is one parameter.
Both are cheap to sweep with this harness before designing anything bigger.

## 6. Caveats

Synthetic resampled vectors, one container on a shared host, default index parameters,
one machine (spec §12). Latency conclusions are robust to all four; recall conclusions are
a trend, not an absolute.
