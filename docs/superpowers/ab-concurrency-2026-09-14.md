# The concurrency A/B — results

**Date:** 2026-09-14. The same 83 pilot articles at `ingest_article_concurrency=4`,
against the sequential baseline in `pilot-reingest-2026-09-14.md`.
Raw log: `scratchpad/ab-n4.log`. Runner: `.superpowers/sdd/ab-concurrent-n4.py`.

## Headline

| | sequential | concurrent (N=4) | delta |
|---|---|---|---|
| entities | 999 | **1027** | +28 (+2.8%) |
| **exact-name duplicates** | **0** | **30** | — |
| episodes | 655 | 655 | 0 |
| wall clock | 6 h 55 m | **2 h 10 m** | **3.20x** |
| per episode | 38.0 s | 11.9 s | 3.20x |
| LLM in-call total | 25,211 s | 22,525 s | -11% |
| overlap ratio (in-call / wall) | 1.01 | **2.90** | — |

**The speedup is real and the duplication is real.** Neither dominates the other;
the decision is what to do about the second.

## Method note — the baseline was NOT destroyed

§8 prescribed resetting the semantic layer first. That would have destroyed the only
copy of the 999-entity baseline, rebuildable only by another $10 / 7 h sequential run.
It was not necessary.

A first attempt to sidestep the reset by running into a separate `group_id` **skipped
all 83 articles and cost $0**: `already_ingested` (`provenance.py:11-15`) gates on
`(:Article)-[r:HAS_EPISODE]->()` by `chunk_index` + `content_hash` and does **not**
filter by `group_id`. The replay gate is structural, so group isolation cannot bypass
it. (graphiti's *dedup* search IS group-scoped — `node_operations.py:439-446`, the only
search in node resolution — so the isolation itself was sound.)

The gate also requires `coalesce(r.superseded,false)=false`, and `provenance.link`
already supersedes the prior edge on re-ingest. So the run: superseded the baseline's
655 `HAS_EPISODE` edges to open the gate, appended into an isolated group, then
restored. Verified clean beforehand (655 edges, 0 superseded, 0 removed) and verified
identical after (999 / 655 / 655 / 0). Revert: `.superpowers/sdd/ab-revert.py`.

## Duplication is measured, not bounded

§8 warned that excess entities is an *upper bound* — concurrency also changes
graphiti's `previous_episodes` extraction context. That caveat turned out to be
testable directly: **exact-name collisions within one group are duplicates by
definition.**

- baseline: 999 entities / 999 distinct names → **0 duplicates**
- concurrent: 1027 entities / 997 distinct names → **30 duplicates**

The bound (+28) and the direct count (30) agree, so the extraction-context effect did
not inflate the total. It is nonetheless large and roughly symmetric: **280 names
appeared only in the concurrent run and 282 only in the baseline** — about 28% of the
name set differs run to run. That is LLM extraction nondeterminism, present in any two
runs, and it cancels rather than accumulating.

## Duplication concentrates on hub entities, as predicted

Expected duplicates scale with `(k-1)` for an entity in k articles, so the risk was
predicted to target the most-connected entities. It did:

| | entities | avg article span | median | max |
|---|---|---|---|---|
| duplicated | 27 | **7.6** | 2.0 | 55 |
| not duplicated | 972 | 2.4 | 1.0 | 42 |

The duplicated names are exactly the ones a backup-documentation corpus is asked about:
**AWS Backup (3x), Amazon EC2 (3x), Azure Backup (3x)**, Amazon DynamoDB, Amazon EBS,
Amazon RDS, Microsoft Azure, Azure Files, Azure Data Lake Storage — plus a run of Azure
region names (Japan East, Korea Central, Germany North …) that co-occur in region-listing
articles.

This is the outcome design invariant #4 exists to prevent: a duplicate `Microsoft Azure`
fragments precisely the cross-vendor unification the single corpus-wide `group_id` is for.

## Why 3.20x and not 4x — the "provider does not serialise us" claim was too broad

The spec's §1 evidence for headroom was latency-by-in-flight for
`dedupe_edges.resolve_edge`, measured flat. **That prompt is still flat** (1327 / 1255 /
1361 / 1429 / 1261 ms across buckets 1, 2-4, 5-9, 10-19, 20+). The claim does not
generalise to the others:

| prompt | 1 in flight | 10-19 in flight | degradation |
|---|---|---|---|
| `extract_nodes.extract_summaries_batch` | 7,569 ms | 13,670 ms | **1.81x** |
| `extract_edges.edge` | 7,673 ms | 11,553 ms | **1.51x** |
| `dedupe_edges.resolve_edge` | 1,327 ms | 1,429 ms | 1.08x (flat) |

Max in-flight reached **36**, not the ~12 predicted from `N x max_coroutines` — so N=4
already pushes the two slowest prompts into their degraded regime. **N=8 should not be
assumed to give 6.4x**; the evidence points to diminishing returns above N≈4.

`contradicted_same_pair` was 270 over 2,246 dedup calls (12.0%), consistent with the
sequential run's 339 / 2,863 (11.8%) — BACKLOG 33 is unchanged by concurrency.

## What this means for the decision

**Every one of the 30 duplicates is an exact-name collision.** That makes them
repairable deterministically — same name, same group, no LLM, no semantic judgement —
rather than needing graphiti's LLM-driven `dedupe_nodes_bulk`. Merging entities still
rewrites `RELATES_TO` edges, which is a destructive write path (§10), but the matching
criterion is exact.

Three options, not mutually exclusive:

1. **Sequential warm-up per source.** 61.5% of duplicate risk weight sits in the opening
   10% of articles, because hub entities appear early. Costs ~10% of the speedup.
   Reduces duplication; does not eliminate it.
2. **Post-hoc exact-name merge.** Repairs 100% of what was observed here, deterministically.
   Needs care around edge rewriting.
3. **Lower N.** The model puts N=2 at roughly half the duplication for ~1.9x.

Warm-up and the merge pass are complementary: the first prevents most, the second
repairs the remainder.

## Correction to §2 of the design spec

§2 rejected a sequential warm-up on the grounds that "the risk does not decay" —
citing a steady ~1.53 entities created per episode and the last decile of entities
appearing 87% of the way through. That measured entity **creation rate**, which is
indeed flat. Duplicate risk is not proportional to creation rate: it scales with
`(k-1)`, and the flat creation rate is dominated by the 636 single-article entities
(63.7%) whose risk is exactly zero. Weighted by actual risk, hub entities appear at
median article **1-6 of 83**. The warm-up idea was killed on the wrong measurement.
