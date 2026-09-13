# The ingestion scale wall — measured

**Date:** 2026-09-12. Read-only in effect: synthetic edges in an isolated
`group_id`, deleted in a `finally` (verified 0 remaining). No LLM calls.
**Reproduce:** `.superpowers/sdd/profile-scale-curve.py`.

## The question

The un-indexed cosine scan is **O(facts)**. At pilot scale it measured as a
non-issue — 253 ms against a 1,837 ms LLM call — which is why the vector index
(BACKLOG 8) was measured at only 1.8x and demoted to P2. But "not the lever today"
and "not the lever at 105k articles" are different claims, and only the first was
measured.

## The curve

Synthetic `:RELATES_TO` edges with random 768-d vectors, scanned with the same
query shape graphiti runs:

| synthetic | total facts | scan ms | µs/fact |
|---|---|---|---|
| 5,000 | 8,469 | 768 | 90.7 |
| 10,000 | 13,469 | 1,131 | 84.0 |
| 20,000 | 23,469 | 1,881 | 80.2 |
| 40,000 | 43,469 | 3,279 | 75.4 |

Cleanly linear:

```
scan_ms = 241 + 0.0699 x facts
```

## The projection

| corpus | facts | one scan |
|---|---|---|
| **today** | 3,469 | **0.5 s** |
| crossover with the dedup LLM call | **22,840** | 1.8 s |
| ~17k articles | 100,000 | 7.2 s |
| **full corpus (~105k articles)** | **610,000** | **42.9 s** |

**CORRECTED 2026-09-13.** This said 5.8 facts/article, from 3,469 facts / 593 articles.
**593 is the STRUCTURAL `:Article` count; only 83 of them have episodes** — the rest were
mapped by graph-sync and never extracted. The real figures are **41.8 facts and 7.9 episodes
per ingested article**, ~7x what this doc assumed, and the error runs in the dangerous
direction: the crossover is at **~546 articles, not ~3,900**, and we are at 83, not 593.
Every per-article projection below is wrong by that factor; the per-FACT curve is unaffected
because it was measured against fact counts directly.

**graphiti runs two of these searches per extracted fact.** Past the crossover the
scan is the whole cost and it keeps growing linearly, so full-corpus ingestion on
this shape is not slow, it is impossible: 610k facts x ~43 s of scanning each.

## Correcting the demotion

Yesterday I measured the index at 1.8x and put BACKLOG 8 back to P2. That reading
was right for the graph in front of me and wrong for the roadmap:

- brute force is **linear** in facts — 42.9 s at corpus scale;
- an HNSW index is roughly **logarithmic** — it should stay in the low hundreds of
  ms across the same range.

The 1.8x at 3,096 facts is not the index's value; it is the value of an index on a
corpus small enough not to need one. **The index is not a scale-out nicety. It is
the wall, and it arrives at ~4% of the corpus.**

## What still has to be solved with it

- **Neo4j only uses a vector index via `db.index.vector.query*` / `SEARCH`.**
  graphiti emits a bare `vector.similarity.cosine` in a `WITH`, so creating the
  index changes nothing on its own — measured at exactly 0%
  (`dedup-cost-profile-2026-09-11.md`). The query must be replaced, which is what
  `driver.search_interface` is for. Note it is **all-or-nothing**: setting it routes
  every search method to your object, and unimplemented ones raise.
- **ANN is approximate.** At `k=1` the index returned the rank-2 fact. graphiti asks
  for `limit=10` and feeds those to dedup, so a missed neighbour is a missed dedup —
  the defect Slice B exists to expose. Recall must be measured against brute force
  before this is trusted, not assumed.
- `db.index.vector.queryRelationships` is **deprecated** on 2026.07.1 in favour of
  `SEARCH`; write the override against `SEARCH`.

## The other scale blockers, in order

1. **This one.** Nothing else matters until the O(N) term is gone.
2. **Concurrency.** Articles and chunks are processed strictly sequentially by
   `IngestDriver`; graphiti's 20-wide `semaphore_gather` applies only within one
   episode. Measured DB saturation at ~1.9x — but that was measured on brute-force
   scans, which is the thing being removed, so it must be re-measured afterwards.
3. **The token budget** (BACKLOG 11): `semantic_daily_token_budget` = 5M/day at
   ~200k tokens/article is ~25 articles/day — 11 years for the corpus. A decision,
   not a bug.
4. **Python-side full scans** (BACKLOG 9): `_vendor_episode_uuids` collects every
   episode uuid of a vendor into a set per request; `shortlist_communities` loads
   every community embedding and does cosine in Python. Both are fine at pilot scale
   and wrong at corpus scale.
