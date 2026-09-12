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

## 4. CORRECTED — the cost is the payload, not the query shape

**My first hypothesis in this document was wrong and I am replacing it.** I wrote that the
~1,000 ms was graphiti binding both endpoint nodes and applying `DISTINCT` over `(e,n,m)`.
Reproducing that exact shape standalone costs **263 ms** — indistinguishable from the raw
253 ms scan. The shape is not the cost.

Isolating each layer with the same Cypher:

| layer | median | delta |
|---|---|---|
| raw neo4j session, `uuid + score` only | 253 ms | — |
| + graphiti's full RETURN field list | 269 ms | +16 ms |
| + `properties(e) AS attributes` | **751 ms** | **+482 ms** |
| graphiti `driver.execute_query` wrapper | 389 ms | +136 ms over raw |
| full `edge_similarity_search` | **1,342 ms** | +953 ms over the wrapper |

**`properties(e)` returns every property on the relationship — including
`fact_embedding`, 768 floats.** Confirmed by inspecting the returned dict:

```
attributes keys: ['reference_time', 'fact_embedding', 'fact', 'group_id',
                  'source_node_uuid', 'name', 'created_at', 'target_node_uuid',
                  'uuid', 'episodes']
fact_embedding present: True | floats: 768
```

Dropping it alone is **2.8x** on that query (751 ms → 269 ms, identical rows). For 20
candidates that is 15,360 floats serialised from a remote database and parsed into Python —
**twice per extracted fact, ~67 facts per article.**

Note the code comment at `edge_db_queries.py:190` states `fact_embedding` "is not returned
by default and must be manually loaded using `load_fact_embedding()`". That is **false for
the Neo4j path**, where `properties(e)` sweeps it in. The comment is what made me look
elsewhere first.

The remaining ~590 ms sits between the driver wrapper and the returned objects — filter
construction and pydantic `EntityEdge` building for 20 edges. Measured as a block, not yet
decomposed; do not attribute it further without measuring.

## 5. What this rules in and out

**Ruled out — measured, not argued:**
- *Vector index.* 0% as graphiti's query is written, 1.8x rewritten, and it optimises the
  253 ms, not the 1,000 ms. (`dedup-cost-profile-2026-09-11.md`)
- *More concurrency alone.* The graph saturates at ~2x and degrades past 8.

**Ruled in, in order of expected value:**
1. **Stop shipping `fact_embedding` in the candidate search**, via graphiti's pluggable
   `SearchInterface` — the same hook item 8 identified, but aimed at the *payload*.
   **Measured 2.8x on the query, identical rows**, no fork, and no correctness hazard: the
   dedup path never reads `attributes.fact_embedding`. Verify that last claim in graphiti's
   code before shipping — if some caller does read it, `load_fact_embedding()` exists for
   exactly that.
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
