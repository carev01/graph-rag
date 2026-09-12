# Where the 9 minutes per article goes — dedup cost profile

**Date:** 2026-09-11
**Corpus at measurement:** 3,096 facts, 914 entities, **0 vector indexes**
**Reproduce:** `.superpowers/sdd/profile-dedup-search.py`, `.superpowers/sdd/profile-dedup-scale.py`
(the search half costs nothing; the LLM half is 3 cheap-tier calls)

## First, a correction

I wrote that the dedup call takes "~8s". That number was total elapsed time divided by
dedup calls (9,270s / 1,145), which charges **all** extraction work — chunking, entity
extraction, edge extraction, node dedup, summaries, embedding, writes — to dedup alone.
The measured dedup cycle is **~4.4s**, not 8.1s. The conclusion that dedup dominates
survives; the arithmetic behind it did not.

## The cycle, per extracted fact

graphiti runs **two** `EDGE_HYBRID_SEARCH_RRF` searches per extracted fact — duplicate
candidates then invalidation candidates — and **one** dedup LLM call.

| component | median | share |
|---|---|---|
| embed the query | 35 ms | 1% |
| **search: invalidation candidates** (unfiltered) | **1,852 ms** | **42%** |
| search: duplicate candidates (filtered by `edge_uuids`) | 716 ms | 16% |
| dedup LLM call (`solar-pro4`, 18 candidates) | 1,837 ms | 42% |
| **total per fact** | **~4.4 s** | |

**Search is the larger half, not the LLM.** That inverts the intuition the backlog was
carrying, and it is the whole point of profiling before building.

## The unfiltered search is the expensive one — and it is the one that grows

The duplicate search is constrained to a small `edge_uuids` set. The invalidation search is
not. Same query, same index situation, **2.6x the latency**. Constraining the candidate set
is what makes the difference, which is exactly what a vector index would do structurally.

Scan cost against corpus size (`vector.similarity.cosine`, no index):

| facts scanned | median | per fact |
|---|---|---|
| 387 | 170 ms | 440 µs |
| 774 | 198 ms | 256 µs |
| 1,548 | 331 ms | 213 µs |
| 3,096 | 567 ms | 183 µs |

Per-fact cost falls as N grows, so a fixed overhead (~110–130 ms) dominates at small N with
a linear term on top. **Caveat on the slope:** this subset test filters with `f.uuid IN $u`,
and that filter's own cost grows with the list — the clean unfiltered scan over all 3,096
facts is **253 ms**, not 567 ms. So the direction (cost grows with corpus) is measured; the
coefficient is contaminated and should not be extrapolated precisely.

What is safe to say: at **3,096 facts** a single scan is 253 ms. The target corpus is
105k articles. Our 17-article sample produced ~920 new facts, i.e. **~54 facts per
article**, which projects to **millions of facts** — two to three orders of magnitude more
than what was measured. A scan with no index cannot absorb that.

## What this does and does not license

**Established:** search is ~58% of the dedup cycle; the unfiltered search is 2.6x the
filtered one; scan cost rises with fact count; there is no vector index on this graph.

**Not established:** a clean per-article attribution. graphiti wraps these searches in
`semaphore_gather`, so they overlap and the wall-clock per article is *less* than
components × facts. The ground truth remains the measured **9 minutes per article**; the
component numbers explain its magnitude, they do not decompose it exactly. Anyone quoting a
"dedup is X% of ingestion" figure needs a concurrency-aware measurement first.

**Also not established:** that indexing fixes it. An index changes the scan, not the two
round trips, the RRF merge, or the 1.8s LLM call. A realistic ceiling on the search half is
what an index-backed search measures — which is the next measurement, not an assumption.

## The lever, already scoped

BACKLOG 8: graphiti's driver exposes a pluggable `SearchInterface`
(`driver/search_interface/search_interface.py`), so an index-backed
`edge_similarity_search` / `node_similarity_search` can be supplied without patching
library internals — the same wrapper pattern `graphiti_client.py` already uses for
`chat.completions.create` and `generate_response`. No fork.

Order of work, cheapest evidence first:

1. Create a vector index on `RELATES_TO.fact_embedding` (and the entity equivalent) and
   re-run `profile-dedup-search.py`. Costs nothing, and measures the real ceiling.
2. Only if the index alone does not reach it, supply the `SearchInterface` override.
3. Re-measure a small ingest end to end, because the 9-minute figure is the only number
   that actually matters.

---

# The vector-index experiment — 2026-09-12

I raised item 8 to P1 on the theory that the missing vector index was the lever for the
9-minute article. **The measurement refutes that for the current corpus.** Reproduce with
`.superpowers/sdd/profile-vector-index.py`.

## 1. The index is not used by graphiti's query at all

Neo4j only consults a vector index through `db.index.vector.query*` (now `SEARCH`).
graphiti's `edge_similarity_search` emits a bare
`vector.similarity.cosine(e.fact_embedding, $v)` inside a `WITH` — a brute-force scan —
so the planner never touches an index.

| graphiti's actual query | median |
|---|---|
| before the index existed | 254.6 ms |
| after creating it (ONLINE, 100% populated) | 254.4 ms |
| **change** | **0%** |

Creating the index and changing nothing else is worth exactly nothing. Had I shipped "add
a vector index" as the fix, it would have measured as a no-op and looked like bad luck.

## 2. Even a correct index-backed query is only 1.8x here

| query | median |
|---|---|
| brute force (what graphiti runs) | 255 ms |
| `db.index.vector.queryRelationships` | **144 ms** |

**1.8x**, not the order of magnitude the 9-minute article needs. At 3,096 facts the fixed
overhead (~110–130 ms, measured earlier) dominates and ANN has little room to win. The
crossover grows with corpus size — brute force is linear in facts, ANN is not — so this
result says *"not the lever today"*, **not** *"never the lever"*. At millions of facts the
same brute-force scan projects to tens of seconds and the index becomes unavoidable.

## 3. ANN is approximate, and dedup depends on exact candidates

At `k=1` the index returned the **rank-2** fact, not the true nearest. At `k=3` it returned
all three in exactly the right order, and the top-3 uuids and scores match brute force
precisely — so the earlier "top-1 disagrees" line in my first run was an artifact of asking
the index for a single neighbour, not a semantic difference. My initial guess that it was an
endpoint-label or group filtering difference was **wrong**: all 3,096 edges are in the
group and 0 have non-`:Entity` endpoints.

But the recall miss at `k=1` is the real point. graphiti asks for `limit=10` and feeds those
candidates to dedup; a neighbour the index fails to surface is a **missed dedup** — the
exact defect Slice B exists to make visible. Trading exact retrieval for 1.8x is a bad deal
at this scale.

## 4. The procedure we would have to call is deprecated

Neo4j 2026.07.1 warns: `db.index.vector.queryRelationships is deprecated. It is replaced by
SEARCH.` Any override written today should target `SEARCH`, not the procedure.

## What this changes

**Item 8 stays real but is NOT the fix for ingestion throughput.** The missing index is a
genuine corpus-scale landmine; it is not why an article takes 9 minutes today. I raised it
to P1 on a hypothesis and the measurement did not support it — the honest move is to put it
back down and stop treating it as the throughput answer.

**The index was dropped after measuring.** Leaving it would add write overhead to every
fact insert during ingestion — the very path we are trying to speed up — in exchange for a
benefit no query currently claims.

**Where the throughput fix must come from instead:** something structural — fewer dedup
calls, batched candidate retrieval, or real concurrency across articles — not a storage
tweak. ~4.4 s per fact is *two* searches plus an LLM call, and no single component is a
majority. That is the shape of a problem you fix by doing less work, not by doing the same
work faster.
