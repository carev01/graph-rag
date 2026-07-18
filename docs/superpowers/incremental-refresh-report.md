# Incremental Community Refresh — Demonstration Report

**Slice:** Phase 3 hardening — incremental `theme-build`: regenerate community
reports only for new/changed communities instead of LLM-rebuilding every one.
**Date:** 2026-07-18
**Status:** Implemented, reviewed, merged locally. Full non-live suite green;
`@live` no-change smoke green; demonstrated end-to-end on the live `backup-docs`
graph.

## What it does

`theme-build` is now incremental by default (`--full` forces the old rebuild):

1. **Detect** (GDS Leiden, cheap, no LLM) — now **seeded** (`randomSeed`,
   `concurrency: 1`) so an unchanged graph reproduces the same partition.
2. **Touched set** — entities/facts with `created_at` after the last refresh
   watermark (`max(corpus_cursor)`).
3. **Match** fresh↔persisted communities by member-set **Jaccard ≥ 0.5**
   (stable ids carry across refreshes; absorbs Leiden wobble).
4. **Classify** — new or touched → **dirty** (regenerate); matched + untouched →
   **clean** (reuse the report).
5. **Regenerate** reports (GLM + embed) for dirty only; **reuse** report text +
   embedding + `generated_at` for clean.
6. **Writeback** — atomic: write the fresh set under stable ids, delete dissolved
   communities, `corpus_cursor` = new watermark.

The cost win is entirely at step 5: detection + graph writes still run fully
(cheap); the LLM only touches new/changed communities.

## Live run on `backup-docs` (68 communities, 3 levels)

| Scenario | regenerated | reused | time | note |
|---|---|---|---|---|
| **No change** | **0** | **68** | **1.4 s** | the headline — zero GLM calls on an unchanged graph |
| Touch 1 entity (in 2 hierarchical communities) | **2** | 66 | 50 s | only the affected leaf + its parent regenerate |
| First run after seeding baseline | 1 | 67 | 84 s | reconciles the one report that had been skipped |

**The guarantee holds:** a re-run with no new content regenerated **zero** reports
in **1.4 seconds** — versus the old full rebuild's ~68 GLM reports over ~20
minutes. Touching a single entity regenerated exactly the two communities
containing it (a leaf and its parent), reusing the other 66.

**Determinism (the enabling fix):** two consecutive detections now produce
byte-identical partitions (same 68 communities, same member sets, same
content-hash ids) — without the seed, Leiden's re-run wobble would drop
communities below the Jaccard threshold and falsely regenerate them.

## Bugs found & fixed during the live bring-up

1. **Unseeded Leiden** — GDS Leiden ran with a random seed, so re-detecting an
   unchanged graph produced a different partition each time → communities wobble
   below τ → falsely "new" → regenerated. Fixed by seeding
   (`randomSeed: 42, concurrency: 1`; the latter is required for GDS determinism).
   Verified: identical partitions across runs.
2. **Watermark too low** — `corpus_cursor` was `max(Episodic.created_at)`, but
   dirty-detection compares **Entity/RELATES_TO** `created_at`, and graphiti
   writes entities/facts *slightly after* their episode (`13:13:25` vs `13:12:53`
   here). So those newer nodes tested as "touched" on every run → every community
   dirty → a no-change run regenerated all ~68 reports. Fixed: the watermark now
   maxes `created_at` across Episodic + Entity + RELATES_TO. Regression test added.
   Post-fix a no-change run regenerates 0 in 1.4 s.

## Verification summary

- **Unit — `match_communities`/`classify`:** exact/wobble match, below-τ→new,
  level isolation, greedy 1:1 split/merge, cold-start all-dirty, touched→dirty.
- **Integration (testcontainer):** `touched_entities` (created_at above/below
  cursor, fact endpoints), `load_persisted`, `prev_corpus_cursor`, the
  comprehensive-watermark regression, `write_communities_incremental` (reuse/dirty/
  dissolve/stable-id/atomic), and the CLI end-to-end asserting the report
  generator is called **once** for a touched community and **zero** times on a
  no-change run.
- **`@live`:** seeded-detection determinism; a no-change incremental run on
  `backup-docs` regenerates 0, reuses 68, in seconds.
- Full non-live suite + `ruff check src tests` + mypy clean.

## Design-decision alignment

Reports remain fact-cited; the LLM never writes a URL (unchanged `report.py`;
reused reports were generated under design #2). `theme-builder` stays the only
writer of `:Community`; the layer stays derived & disposable (`--full` rebuilds it
from scratch, and did so here to establish the seeded baseline).

## Follow-ups (deferred, per spec §10)

- Removal-driven dirtiness (integrate the weekly tombstone sweep so silent
  deletions mark communities dirty — the `created_at` signal catches adds/updates,
  not tombstone-only removals).
- Optimal split/merge handling (currently best-Jaccard-wins).
- graph-sync explicit dirty-marking (Approach B) if the `created_at` signal proves
  insufficient at scale.
- Scale-optimizing the O(F×P) Jaccard match and the watermark aggregation scan.
