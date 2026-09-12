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
