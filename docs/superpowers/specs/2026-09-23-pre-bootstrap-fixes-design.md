# Pre-bootstrap fixes — design

**Date:** 2026-09-23. **Approved:** 2026-09-23 (the user approved D4's suppression, the
live revert and D6's paid A/B; the future-`invalid_at` fix is BACKLOG 41, a defect).
**Evidence:** `docs/superpowers/pre-bootstrap-decisions-2026-09-23.md`.

## 1. Suppress same-pair contradiction invalidation at ingest (D4, BACKLOG 33)

**Why.** 0 of 14 live same-pair invalidations were genuine; they removed core facts
from answers. The cross-pair path is already off (`contradiction_gate`).

**Where.** graphiti consumes `contradicted_facts` in two places inside
`resolve_extracted_edge` (0.30.1): `resolve_edge_contradictions` expires the OLD edge,
and an inline block (`edge_operations.py:826-838`) can expire the NEW edge when a
candidate has a later `valid_at`. The inline block cannot be patched separately, so the
only point that neutralises both is the model's reply. `dedup_guard` already wraps that
exact call (`dedupe_edges.resolve_edge`) and already counts `contradicted_same_pair`.

**What.** `install_dedup_guard(..., suppress_contradictions: bool)`: on every return path
of the dedup branch (clean, out-of-range, retried, and parse-failure), AFTER all counting,
if suppression is on, return a copy of the reply with `contradicted_facts = []` and add
the number of removed entries to a new `DedupIndexStats.contradictions_suppressed`
counter. `duplicate_facts` is never touched — the module's rule against rewriting
duplicate indices stands (a wrong merge is invisible; a dropped contradiction is exactly
the intent here). Setting `ExtractSettings.ingest_same_pair_contradictions: bool = False`
(False = suppressed); both `install_dedup_guard` call sites in `cli._build_ingest_driver`
pass `suppress_contradictions=not settings.ingest_same_pair_contradictions`.

**Unaffected.** An end date stated in the text (`invalid_at` from extraction) still
applies — that is the extractor dating the fact, not a contradiction. Invariant #3 is
carried by updates: superseded episodes and the weekly sweep.

## 2. A future `invalid_at` means "still current" (BACKLOG 41)

`answer_api`: a fact is current iff `invalid_at is None or invalid_at > now` (UTC;
naive datetimes treated as UTC, as `timeline._sort_key` already does). One helper,
`answer_api.temporal.is_current(invalid_at, now) -> bool`, used by
`search_local`'s filter (`search.py:38`) and `timeline._fact_status` ("current" for a
future end date). `now` is injectable for tests.

## 3. Chunk packing (D6, BACKLOG 42)

`episode_builder.build_episodes(..., pack_target_tokens: int = 0)`: after the existing
split-then-merge-tiny, greedily merge CONSECUTIVE chunks while the running total stays
`<= min(pack_target_tokens, max_chunk_tokens)`. 0 = off (today's behaviour exactly).
Text is joined with "\n" as `_merge_tiny` does; token counts add; content is preserved
(concatenation equals the unpacked concatenation modulo the join newlines). Setting
`ExtractSettings.pack_target_tokens: int = 0`, passed by `IngestDriver` and `probe`.
Default stays 0 until the A/B decides. Note: changing it on an ingested article changes its
chunk layout, so the article's episodes are re-keyed on its next re-ingest — set it once,
before the bootstrap.

## 4. The D6 A/B harness (`scripts/chunk_ab.py`)

Paid, authorised, cap $5. Two arms, each in its OWN throwaway Neo4j testcontainer (the
live graph is never written; separate containers because the per-chunk `HAS_EPISODE` gate
would otherwise skip the second arm's already-linked articles):

- **A (today):** production settings.
- **B (pack 1200):** `pack_target_tokens=1200`, `cheap_max_chunk_tokens=1200`.

Same article list for both: ~30 articles from the D6 sample, biased to those with
≥3 episodes today, spread across vendors. Each container is seeded with the articles'
`:Article {id, source_url}` and a `:Chapter` via `IN_CHAPTER`; graphiti indices and the
vector indexes are created by `init_indices`. Articles run sequentially through
`IngestDriver.ingest_article` (the chunking variable does not depend on the concurrency
path; this bypasses `ingest_source`'s warm-up and sibling spreading, which is recorded
in the results).

**Proof the variable is engaged (CLAUDE.md cost rule):** a 2-article smoke per arm first;
refuse to continue unless arm B added FEWER episodes than arm A for the smoke articles.
**Spend cap:** abort an arm when the process usage tally exceeds $5 total.

**Measured per arm:** episodes, facts (`RELATES_TO`), entities, dedup counters (incl.
`contradictions_suppressed`), structured-output/invalid-index counts, wall clock,
$ (usage tally at the solar-pro4/gpt-5-mini rates), and a per-article fact dump for a
hand-judged sample. Output: a JSON results file plus the dumps.

**Decision rule** (from the D6 recommendation): adopt 1200 if facts/article and entities
hold within noise and the hand-judged sample shows no systematic loss; otherwise try 900.
