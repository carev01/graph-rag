# Ingestion throughput — where the 9 minutes actually goes

**Date:** 2026-09-12
**Corpus:** 3,096 facts, 914 entities, remote Neo4j (`alpcirag01`)
**Reproduce:** `.superpowers/sdd/profile-concurrency.py`, `profile-dedup-search.py`
**Cost:** zero — searches and embeddings only, no extraction LLM

Ground truth to beat: **17 articles in 2h34m ≈ 9 minutes per article**, cheap tier.

## 1. Our own loops are sequential; graphiti's concurrency is innermost only

`IngestDriver.ingest_source` processes **articles one at a time**
(`for aid in ids: await self.ingest_article(aid)`), and `ingest_article` processes
**chunks one at a time** (`for e in episodes: await add_text_episode(...)`).

graphiti's `semaphore_gather` (`SEMAPHORE_LIMIT`, default **20**, env-settable) applies only
*within* one episode's extracted edges. So the outer two levels of the work are serial by
construction, and the parallelism that exists is the narrowest one.

## 2. But the graph saturates at ~2x, so concurrency is not the jackpot

Concurrent `EDGE_HYBRID_SEARCH_RRF` calls against the live graph:

| concurrency | wall ms | ms/search | speedup |
|---|---|---|---|
| 1 | 1,476 | 1,476 | 1.0x |
| 2 | 3,076 | 1,538 | 1.0x |
| 4 | 3,028 | 757 | **1.9x** |
| 8 | 6,338 | 792 | 1.9x |
| 16 | 14,440 | 903 | 1.6x |

Throughput plateaus at ~1.9x by concurrency 4 and **degrades** by 16. Parallelising the
outer loops therefore buys roughly 2x on the search half — not the order of magnitude the
9-minute article needs. (Medians of 3; the concurrency-2 row is within noise of 1.)

## 3. The dedup cycle, decomposed

Per extracted fact: two `search()` calls plus one LLM call. Each `search()` internally runs
BM25 and cosine **concurrently**, so its floor is the slower of the two.

| component | median |
|---|---|
| `edge_fulltext_search` (BM25) | 785 ms |
| `edge_similarity_search` (cosine) | **1,258 ms** |
| → `search()` floor = max(the two) | ~1,258 ms |
| dedup LLM call (`solar-pro4`) | 1,837 ms |

## 4. The finding that matters: the vector maths is 20% of the vector search

The raw cosine scan over all 3,096 facts is **253 ms**. The `edge_similarity_search` that
wraps it is **1,258 ms**. **~1,000 ms — 80% — is the query around the maths.**

It is not payload: `get_entity_edge_return_query` documents that `fact_embedding` is *not*
returned. The difference is shape. graphiti runs

```
MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
WITH DISTINCT e, n, m, vector.similarity.cosine(e.fact_embedding, $v) AS score
```

binding and materialising **both endpoint nodes** for every candidate edge and applying
`DISTINCT` over the `(e, n, m)` triple. My 253 ms scan touched the relationship alone.

This is why the earlier vector-index experiment bought so little: **the scan was never the
bottleneck.** Indexing optimises the 20%.

## 5. What this rules in and out

**Ruled out — measured, not argued:**
- *Vector index.* 0% as graphiti's query is written, 1.8x rewritten, and it optimises the
  253 ms, not the 1,000 ms. (`dedup-cost-profile-2026-09-11.md`)
- *More concurrency alone.* The graph saturates at ~2x and degrades past 8.

**Ruled in, in order of expected value:**
1. **A leaner similarity query** via graphiti's pluggable `SearchInterface` — the same hook
   item 8 identified, but aimed at the *shape* (drop the endpoint binding and the
   `DISTINCT`) rather than at adding an index. Targets the 1,000 ms, needs no fork, and has
   no correctness hazard: same inputs, same outputs, fewer materialised nodes. **Measure the
   candidate query standalone before writing any integration.**
2. **Fewer searches.** Two `search()` calls per fact, the unfiltered one 2.6x the filtered
   one. Whether both are needed per fact, or can be shared across an episode's edges, is
   unexamined.
3. **Concurrent chunks/articles — with a stated hazard.** Worth ~2x at best, and it carries
   a real risk: graphiti's dedup reads existing edges, so two episodes extracting
   concurrently cannot see each other's writes and would both miss the duplicate. Given
   Slice B just established that dedup already silently drops candidates, adding races to
   the same mechanism is the wrong trade until the others are exhausted.

## 6. Honest limits of this profile

- Single-run medians on a shared remote database; another tenant's load would move them.
- The LLM half (1,837 ms) was measured with 3 calls on a synthetic 18-candidate prompt, not
  sampled from real traffic.
- No end-to-end re-measurement has been done. **The 9-minute article remains the only number
  that matters**, and any change must be re-validated against it, not against these
  components — `semaphore_gather` overlaps work in ways component timings do not capture.
