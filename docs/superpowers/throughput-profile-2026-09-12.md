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


---

# Shipped 2026-09-12 — the lean edge projection

`graph_extract/lean_edge_search.py`, wired into `build_graphiti`.

| `edge_similarity_search` | median |
|---|---|
| before | 1,072 ms |
| after | **407 ms** |
| speedup | **2.63x** |
| identical results | yes |
| `fact_embedding` on returned edges | `None` before AND after |

That last row is the safety proof measured rather than argued: graphiti reads the vector
from a top-level record key the search never returns, so search-derived edges never carried
it. `properties(e)` shipped 768 floats into a dict whose next statement popped them.

**Implementation note.** Cypher has no "map without key" — `map.drop`, `map.remove`, map
subtraction and APOC were all probed and are unavailable on 2026.07.1 Community, and
`e {.*}` re-includes the vector. Hence an explicit key projection, patched onto
`search_utils`' own reference (it imports the function by name).

Two guards, because an explicit projection is exactly the shape that drops data silently:
if graphiti's query stops containing `properties(e) AS attributes` the patch returns the
original unchanged and logs; and `assert_no_custom_edge_attributes` fails loudly naming any
`:RELATES_TO` property outside the projection. A test also pins the premise, so if the
library stops shipping the embedding the patch is reconsidered rather than silently kept.

## CORRECTED — BM25 is NOT the ceiling; the patch fixed it too

I wrote that BM25 would cap the gain at 785 ms. Wrong: `edge_fulltext_search` uses the same
`get_entity_edge_return_query` (`search_utils.py:271`), so patching that reference fixed
**both** halves. Re-measured:

| | before | after |
|---|---|---|
| BM25 | 782 ms | **291 ms** (2.7x) |
| cosine | 914 ms | **402 ms** (2.3x) |
| `search()` floor = max of the two | 914 ms | **402 ms** |

Serial per-fact estimate: 2 searches + 1 LLM call, **~3,665 → ~2,641 ms (~28%)**. The
**1,837 ms dedup LLM call is now ~70% of the cycle** and is the next target.

Revised order: **the dedup LLM call** (~70% of the remaining cycle), then **fewer than two
searches per fact**. Concurrency stays last — it races graphiti's dedup reads, and Slice B established
those already drop candidates silently.

**Still unmeasured end to end.** The 9-minute article remains the only number that matters.


---

# Second win — skip the dedup LLM call when there is nothing to dedup against

`resolve_extracted_edge` issues its LLM call **unconditionally**
(`edge_operations.py:726`); the `if related_edges or existing_edges` guards above and below
it cover only a debug log and the contradiction block. So when BOTH candidate lists are
empty graphiti asks a model to pick duplicates out of an empty list — and then discards the
answer: `duplicate_fact_ids` filters every index against `len(related_edges) == 0`, and
contradictions are skipped entirely.

The reply therefore **cannot** affect the outcome, and the only correct answer is the empty
one. `dedup_guard` already recovers N and M from the prompt, so it now short-circuits that
case and returns `{"duplicate_facts": [], "contradicted_facts": []}` without a call. This is
**exact, not an approximation** — and it saves the full ~1,837 ms each time it fires.

How often it fires is unknown and deliberately not guessed: the new `no_candidate_skips`
counter is printed per article and per run by `ingest`, so the next run measures it.

---

# Validated end to end — 2026-09-12

Second cheap-tier ingest on the same source, with both optimisations live. Raw log:
`.superpowers/sdd/throughput-ab-ingest.log`.

**Metric: seconds per dedup call, not per article.** The baseline run's "9 min/article"
is contaminated — some of its 17 articles were already-ingested skips costing only a fetch
and a chunking pass. Normalising by dedup calls removes that. Both figures charge *all*
extraction work to dedup, so they are comparable to each other and to nothing else.

| | baseline | after |
|---|---|---|
| dedup calls | 1,145 | 560 |
| wall clock | 9,270 s | 3,093 s |
| **s per dedup call** | **8.10** | **5.52** |
| speedup | | **1.47x (-32%)** |

The component profile predicted **-28%**; the live run delivered **-32%**. That agreement
is the reason to trust the profile, not the other way round.

**Conservatively biased against the new code:** the graph grew from 2,173 to 3,096+ facts
between runs, so every candidate search scanned more data.

## The short-circuit fired ZERO times — recorded as measured-ineffective

`no_candidate_skips = 0` across all 560 calls. The zero-candidate short-circuit is provably
correct and **worthless on this corpus**: hybrid search with BM25 almost always matches
something, so a fact with no candidates at all essentially does not occur. **All 32% came
from the lean projection.**

Kept because it costs nothing and will matter on a sparse cold-start graph, but it must not
be counted as a win, and no variant that assumes empty candidate lists are common should be
proposed again — that is now measured and refuted.

## Slice B reconfirmed on independent data

| | run 1 | run 2 | total |
|---|---|---|---|
| out-of-range duplicate indices | 154 | 113 | **267** |
| inside the invalidation range | 154 | 113 | **267 (100%)** |
| beyond range / negative | 0 | 0 | **0** |
| `parse_failures` | 0 | 0 | 0 |
| invalid-call rate | 7.4% | 9.5% | — |

Two independent runs, 267 of 267. The index-space-confusion diagnosis is as solid as this
kind of evidence gets. The rate moved within the variation expected across different
articles.

## Where this leaves the goal

5.52 s/call x ~67 calls ≈ **6 minutes per article**. Better than 9, nowhere near enough for
105k articles. Per-call cost is close to exhausted: the dedup LLM call (~1,837 ms) is now
the largest single component and graphiti issues exactly one per extracted fact.

The next lever is **call volume, not call latency** — and every option there changes
behaviour, so it needs measurement rather than a confident patch.
