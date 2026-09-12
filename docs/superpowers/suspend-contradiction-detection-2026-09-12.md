# Suspend contradiction detection — slice report

**Date:** 2026-09-12. **Branch:** `suspend-contradiction-detection`.
**Spec:** `specs/2026-09-12-suspend-contradiction-detection-design.md`.
**Plan:** `plans/2026-09-12-suspend-contradiction-detection.md`.
**Backlog:** resolves 8 (scale), 29 and 32 (index space); opens **33**; item 6 stays open.

This report was written at the end of the final whole-branch review, not at the end of
implementation, because the review found the branch's central quality claim to be false.
It separates what was built, what was measured, what was claimed, and which success
criteria stand.

## What was built

One module, `src/graph_extract/contradiction_gate.py`, wired into `build_graphiti` behind
`ExtractSettings.ingest_detect_contradictions` (default `False` = suspended).

`resolve_extracted_edges` issues two `search()` calls per extracted fact. The gate patches
`edge_operations`' own imported-by-name `search` reference and discriminates structurally on
`search_filter.edge_uuids`:

| search | filter | what the gate does |
|---|---|---|
| duplicate candidates | `SearchFilters(edge_uuids=[...])` | delegated unchanged |
| invalidation candidates | `SearchFilters()` | returns an empty `SearchResults()`, no query |

Reads `search_filter` by keyword and positionally, returns a well-formed empty result rather
than `None`, propagates delegated exceptions, is idempotent, and un-installs itself when
called with `detect_contradictions=True`.

## What is measured

- **The scan is O(facts):** `scan_ms = 241 + 0.0699 × facts`, from synthetic edges on the
  live graph (`scale-wall-2026-09-12.md`). 0.5 s today, 42.9 s at the projected 610k-fact
  corpus, overtaking the dedup LLM call at ~3,900 articles — under 4% of the corpus.
- **Only the unfiltered search scans.** `PROFILE`: the duplicate search is a
  `DirectedRelationshipIndexSeek` over its candidate list (10 rows); the invalidation search
  is `NodeByLabelScan + Expand(All)` over every fact (3,469 rows).
- **All 140 ingest-time invalidations were contradiction-driven**, and the six hand-inspected
  ones — selected as same endpoints and same relation name — were refinements and
  near-duplicates ordered by crawl order (`invalidation-measurement-2026-09-12.md`).

## What is claimed, and what the review found

The branch claimed that with the invalidation search skipped, "nothing is invalidated".
**That is false**, and it was false in the spec (§4, §9), the plan (Verification 2), a
tripwire docstring, the module docstring, and BACKLOG 6/8/29/30/32.

`resolve_extracted_edge` fills `invalidation_candidates` from **both** candidate lists
(`edge_operations.py:769-776`): `contradicted_facts` indices below `len(related_edges)`
select from the **duplicate** candidates — the filtered search the gate delegates. graphiti's
dedup prompt invites exactly that ("idx values from EITHER list"; its worked example returns
a contradiction on a same-pair refinement). A second path (`edge_operations.py:826-839`)
sets `invalid_at` on the *new* edge when a same-pair candidate has a later `valid_at`.

So: the gate suspends the O(corpus) scan and **cross-pair** invalidation. **Same-pair
invalidation remains live.** And the six of six bad invalidations that justified the change
on quality grounds were all same-pair — they came from the path that stays open, not the
one suspended. **The scale fix is real. The quality fix was claimed and is not delivered.**
Recorded as BACKLOG 33; its share of invalidations is unmeasured and can only be measured
forward, on a paid run.

Per the review's judgement, the claims were corrected and the gate was **not** extended to
filter `contradicted_facts` — that is a second semantic patch into graphiti and gets its own
design cycle.

## Success criteria — status

| # | criterion | status |
|---|---|---|
| 1 | No ingest query scores more rows than its candidate list | **Delivered.** The invalidation search issues no query at all; asserted through graphiti's real `search()` against a testcontainer with a driver-level query counter, with the delegated duplicate search proving the counter live. `PROFILE` evidence for the remaining search is in the spec. |
| 2 | An ingest produces zero new `invalid_at` edges | **Claimed and not delivered; withdrawn.** Not a property of this design — same-pair invalidation is live (BACKLOG 33). The original tripwire asserted `resolve_edge_contradictions(None, []) == []`, a true fact about the wrong list. What is delivered: no `invalid_at` is ever set from a candidate on a *different* entity pair. |
| 3 | Local search results identical with and without the gate | **Claimed and not verified until this fix pass; now delivered.** The original test counted Cypher rows around an in-process assignment and could not fail. It now calls `answer_api.search.search_local` through a real `Graphiti` on the testcontainer (fixed embedder, LLM/reranker that raise if reached) before and after installing the gate and asserts equal, **non-empty** results with a resolved citation. Killed by a mutation that also binds the gate to `graphiti_core.graphiti.search`. |
| 4 | Per-fact search work halved (item 31 per-prompt timing) | **Pending a paid run.** Not verifiable in CI. |
| 5 | `dup_in_invalidation_range` falls to zero | **Pending a paid run.** `dup_beyond_range` may still be non-zero and would be new information, not a regression. |

## Verification

Tripwires added or corrected in `tests/unit/test_contradiction_gate.py`:

- `test_same_pair_contradiction_stays_live_through_the_duplicate_candidates` — pins, on the
  library's `ast`, that `contradicted_facts` indices below `len(related_edges)` are routed
  into `invalidation_candidates`. Killed by two mutations of the library source (routing
  line removed; routing `existing_edges` instead).
- `test_no_invalidation_candidates_means_no_invalidation` — docstring corrected to what the
  assertion actually pins.
- The call-site tripwire's docstring now records its two limits: it parses
  `resolve_extracted_edges` only, and reads the `search_filter=` keyword literally.

`tests/integration/test_contradiction_suspended.py` rewritten so both behavioural tests can
fail (see criteria 1 and 3). Mutation results: A (gate also binds the answer path's copy of
`search`) kills the answer-path test; B (gate delegates everything) kills the driver-path test.

**The CI breakage.** `build_graphiti` installs the gate process-wide and nothing uninstalls
it. `tests/integration/test_compat_harness.py` calls it in a non-live test, and the CI command
runs integration before unit in one process, so two unit tripwires failed
(`2 failed, 813 passed` on the branch tip). The plan's verification step ran the two halves
separately, which is why it was missed. Fixed with an autouse fixture in
`tests/unit/conftest.py` that pins `edge_operations.search` to the defining module's function
around every unit test.

Full CI command, one process, run in the foreground for this report:
see `.superpowers/sdd/final-review-fixes-report.md` for the observed output and counts.

## Not done here, deliberately

- No extension of the gate to same-pair contradiction (BACKLOG 33 — own design cycle).
- No repair of the 140 already-invalidated facts (spec §8).
- No paid run: criteria 4 and 5 remain pending until the user chooses to run an ingest.
