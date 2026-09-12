# Suspend contradiction detection — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Date:** 2026-09-12
**Status:** Approved design — implemented; claims corrected in the final whole-branch review (see §4, §9)
**Backlog:** resolves the P0 scale blocker in item 8; supersedes item 32; acts on items 6, 29, 30; opens item 33

> **Correction (final review, 2026-09-12).** The original text of §4 and §9 claimed that with
> the gate installed "nothing is invalidated". That is false. `resolve_extracted_edge`
> (`edge_operations.py:769-776`) fills `invalidation_candidates` from **both** candidate
> lists: `contradicted_facts` indices below `len(related_edges)` select edges from the
> **filtered duplicate search the gate delegates**. The gate suspends the O(corpus) scan and
> **cross-pair** invalidation; **same-pair** contradiction remains live. The measured bad
> invalidations (BACKLOG 6: six of six inspected, all same-endpoint, same-relation-name)
> come from the path that stays open. See BACKLOG 33. The sections below are corrected in
> place; the scale analysis in §1–§3 stands.

---

## 1. The problem

Ingestion cannot reach corpus scale, and the reason is one query.

`resolve_extracted_edges` issues **two** searches per extracted fact. They look alike and
are not:

| search | `search_filter` | Neo4j plan | rows scored |
|---|---|---|---|
| duplicate candidates (`edge_operations.py:394`) | `SearchFilters(edge_uuids=[...])` | `DirectedRelationshipIndexSeek` | **10** — the candidate list |
| invalidation candidates (`edge_operations.py:409`) | `SearchFilters()` | `NodeByLabelScan` → `Expand(All)` | **3,469** — the whole corpus |

Measured with `PROFILE` on the live graph. The duplicate search is **already index-backed**
and bounded by node-pair degree; it does not grow with the corpus. The invalidation search
is O(facts):

```
scan_ms = 241 + 0.0699 x facts
```

| corpus | facts | one scan |
|---|---|---|
| today | 3,469 | 0.5 s |
| crossover with the 1,837 ms dedup LLM call | 22,840 | 1.8 s |
| full corpus (~105k articles) | 610,000 | **42.9 s** |

At 5.8 facts/article the crossover is **~3,900 articles — under 4% of the corpus**. Past it
the scan is the entire cost and keeps growing. Full-corpus ingestion on this shape is not
slow, it is impossible.

**That O(N) scan exists solely to feed contradiction detection, and contradiction detection
does not currently work.** Measured the same day (`invalidation-measurement-2026-09-12.md`):
all 140 ingest-time invalidations are contradiction-driven, and the ones inspectable are
refinements and near-duplicates rather than change events — *"supports Azure database for
PostgreSQL"* invalidated by *"supports Azure Database for PostgreSQL **Flexible Server**"*.
Which fact survives is decided by the order DocExtractor crawled the pages, because
`valid_at` for present-tense facts is the crawl timestamp and no document revision date
exists in the corpus.

It is also actively degrading answers: `answer_api/search.py:38` drops every edge with
`invalid_at` set, so **4% of the graph's facts are already excluded from local answers** on
this basis, and every ingest adds more.

## 2. The decision

**Suspend contradiction detection until there is a real time axis.** Off is better than
wrong: design invariant #3 exists to stop losing history, and we are losing it by crawl
order.

Rejected alternatives, with reasons:

- **A vector index / ANN.** Would keep invalidation working at scale, but ANN is approximate
  — at `k=1` the index returned the rank-2 fact — and dedup depends on exact candidates, so
  a missed neighbour is a permanent duplicate. Exactness was set as non-negotiable, and this
  design achieves the same scale goal without trading it away.
- **Bounding the invalidation candidates** (e.g. to edges sharing an endpoint). Attractive,
  but it is a **semantic change to graphiti's contradiction model**, not an optimisation:
  graphiti deliberately excludes same-pair edges from the invalidation list
  (`edge_operations.py:419-429`, "keep it only in duplicate candidates") — but note it
  then lets the LLM mark a *duplicate* candidate as contradicted too
  (`contradicted_facts` "from EITHER list", `edge_operations.py:769-776`), so same-pair
  contradiction lives in the duplicate branch, not the invalidation one. Designing that narrowing
  requires timestamp semantics we do not have and a measurement of how many genuine
  contradictions this corpus contains — currently unmeasurable, since none of the 140
  observed are genuine. Deferred to its own design cycle. See §7.
- **Suspending the invalidation only, keeping the search.** Leaves the O(N) scan — the
  entire scale blocker — in place for no benefit.

## 3. Architecture

One new module, following the pattern proven twice in this codebase
(`lean_edge_search`, `dedup_guard`): patch a module's imported-by-name reference rather than
reimplement library internals.

```
src/graph_extract/contradiction_gate.py
    install_contradiction_gate() -> bool
```

It wraps `graphiti_core.utils.maintenance.edge_operations.search`. That module imports the
function by name (`edge_operations.py:40`), so the patch must be applied to **that module's
attribute**; patching the defining module would not be seen.

The discriminator is **structural, not heuristic**:

```
search_filter.edge_uuids is None   ->  invalidation search  ->  return SearchResults(), no query
search_filter.edge_uuids is not None -> duplicate search    ->  delegate unchanged
```

`edge_operations` contains **exactly two** `search()` call sites (394, 409) and they differ
in precisely this way, so the rule cannot misfire. `SearchResults` has `default_factory=list`
on every field, so `SearchResults()` is a valid empty result.

**Scope containment:** the answer path does not route through `edge_operations` at all — its
searches go through `graphiti.search()` / `search_utils` — so retrieval is untouched by
construction, not by a flag someone must remember to check.

### 3.1 Why `driver.search_interface` is NOT used

It is graphiti's sanctioned extension point and it is the wrong tool here. Setting it routes
**all twelve** search methods to the supplied object, and unimplemented ones raise — so
using it to change one query means reimplementing eleven others from library internals,
which then drift on every upgrade. The two-call-site patch touches strictly less.

### 3.2 Configuration

```python
ingest_detect_contradictions: bool = False
```

Default **off**, because the behaviour it enables is measured to be wrong. When `True`,
both searches delegate and behaviour is byte-for-byte what it is today.

The flag exists so the change is reversible and A/B-testable — **not** as the
re-enablement plan (§7).

**Terminology, because the polarity is easy to misread:** *suspended* means
`ingest_detect_contradictions is False`, and the invalidation search is skipped. *Enabled*
means the flag is `True` and graphiti behaves exactly as it does today. This spec uses only
those two words.

## 4. What follows without additional code

With `existing_edges` empty for every fact:

1. **Cross-pair invalidation is gone.** `existing_edges` contributes nothing to
   `invalidation_candidates`, so no fact on a *different* entity pair can be invalidated.
   **Same-pair invalidation is NOT gone** (corrected in the final review; the original text
   here said "nothing is invalidated"): `contradicted_facts` indices in `0..n-1` still
   select from `related_edges` (`edge_operations.py:769-776`), the duplicate candidates
   the gate delegates, and those reach `resolve_edge_contradictions` unchanged. A same-pair
   candidate with a later `valid_at` can also mark the *new* edge invalid on arrival
   (`edge_operations.py:826-839`). The tripwire
   `test_same_pair_contradiction_stays_live_through_the_duplicate_candidates` pins this.
   Its share of the 140 measured invalidations is unmeasured; the six hand-inspected ones
   were all same-pair (BACKLOG 33).
2. The dedup prompt carries **one populated index space instead of two**, so the confusion
   behind all **267** observed out-of-range indices (BACKLOG 29) becomes structurally
   impossible rather than merely less likely.
3. Where `related_edges` is also empty, graphiti's early return (`edge_operations.py:653`)
   skips the dedup LLM call entirely.
4. Per-fact search work halves, and what remains is the index-seek-bounded duplicate search.

One change removes the scale wall and the index-space defect. It removes the
**cross-pair** phantom invalidations; the same-pair ones — the kind actually observed —
are not addressed by this design and need their own (see §7, BACKLOG 33).

## 5. Error handling

| case | behaviour |
|---|---|
| Invalidation search while suspended | Return `SearchResults()` — a well-formed empty result, never `None`, which would fail confusingly inside graphiti. |
| Duplicate search | Delegate unchanged, including all arguments and exceptions. |
| `ingest_detect_contradictions = True` | Delegate both; no behavioural difference from today. |
| graphiti stops importing `search` by name | Tripwire test fails (§6). The patch would silently no-op otherwise. |
| `edge_operations` gains a third `search()` call site | Tripwire test fails, because the discriminator's safety rests on there being exactly two. |
| Delegated search raises | Propagate. A dead search must fail the episode loudly, as today. |

Exceptions are never swallowed, per the standing rule in this codebase: a failure coerced
into a legitimate-looking value has been the most expensive recurring defect here.

## 6. Testing

**Unit (hermetic):**
- While suspended, an unfiltered edge search from `edge_operations` returns an empty
  `SearchResults` and the underlying `search` is never awaited.
- A filtered search is delegated with its arguments intact.
- With `ingest_detect_contradictions=True`, both are delegated.
- The returned object is a real `SearchResults` with empty `edges` — asserted on the type,
  not on truthiness.
- `install_contradiction_gate()` patches `edge_operations`' own attribute.

**Tripwire (library pins, expected to fail on a graphiti upgrade rather than on our code):**
- `edge_operations` imports `search` by name.
- `edge_operations` contains exactly two `search(` call sites, one filtered by `edge_uuids`
  and one not.
- `resolve_edge_contradictions([])` returns `[]` — pins only that an empty *cross-pair*
  list invalidates nothing.
- `resolve_extracted_edge` routes `contradicted_facts` indices below `len(related_edges)`
  into `invalidation_candidates` — pins that the same-pair path is live (added in the final
  review; the design originally assumed it was not).

**Integration (real Neo4j, no paid endpoint):**
- The gated invalidation search issues **zero** driver queries through graphiti's real
  `search()`; the delegated duplicate search, same clients and counter, does reach the
  database and returns the seeded fact.
- The answer path is unaffected — `search_local` returns the same, **non-empty** result
  before and after the gate is installed.
- Not covered hermetically: "an ingest while suspended produces zero new `invalid_at`
  edges" was originally listed here and is **not a property of this design** — see §4
  item 1. A duplicate-detection-still-works ingest needs an LLM and is a paid run.

**Discrimination:** every test must fail with its fix neutralised, proven by mutation and
recorded, per this project's standing practice.

## 7. The re-enablement path — recorded so the reasoning is not lost

**This is not a revert, and the flag is not the plan.**

When DocExtractor supplies `content_changed_at`
(`docs/proposals/2026-09-12-docextractor-article-timestamps.md`), turning
`ingest_detect_contradictions` back on as-is would restore three things at once: the O(N)
scan, the two-index-space dedup prompt, and a contradiction judgment measured to be
unreliable.

**Timestamps are necessary but not sufficient.** A real time axis fixes the *ordering* of two
facts. It does not stop the LLM asserting that a refinement contradicts the thing it refines
— those pairs still fire whenever the two facts come from different revisions. Arguably that
is worse: a wrong invalidation carrying a credible date is harder to spot than one keyed on
crawl order.

Re-enablement therefore needs, as its own design cycle:

1. A **bounded** invalidation candidate set, so O(N) never returns — accounting for
   graphiti's deliberate exclusion of same-pair edges (§2).
2. A measurement of **how many genuine contradictions this corpus actually contains**.
   Today the answer appears to be close to zero, which would itself be a finding: a vendor
   documentation corpus may simply not contradict itself often, in which case the feature is
   not worth its cost at any scale.
3. A **re-ingest**. Existing facts carry crawl-derived `valid_at` values that cannot be
   retrofitted with document dates. This is a natural moment for it, since the scale work
   should by then make a full ingest feasible.

**Carried forward untouched:** the lean edge projection, the dedup guard and its counters,
per-prompt timing, and the duplicate search. None depend on invalidation.

## 8. Out of scope

- **The 140 already-invalidated facts stay invalidated.** Repairing them asserts they were
  all wrong; six were hand-checked. Un-invalidating is reversible and deserves its own
  evidence.
- **Node-side similarity search.** `node_similarity_search` is also unfiltered in places
  (`node_operations.py:439`), but entities grow far slower than facts (914 vs 3,469 today)
  and it has not been measured as a blocker. Not addressed here; worth measuring separately.
- **Concurrency across articles and chunks.** The next scale lever after this one, and it
  must be re-measured afterwards: the ~1.9x DB saturation ceiling was measured on
  brute-force scans, which this change removes.

## 9. Success criteria

1. No query in the ingest path scores more rows than the candidate list it was given —
   verified by `PROFILE`, not by timing.
2. ~~An ingest produces zero new `invalid_at` edges.~~ **Withdrawn in the final review** —
   not a property of this design (§4 item 1). What the design delivers is: no
   `invalid_at` is ever set from a candidate on a *different* entity pair. Same-pair
   invalidations can still occur and their rate is unmeasured (BACKLOG 33).
3. Local search results are identical whether or not the gate is installed — asserted on a
   real, non-empty `search_local` result (the original test compared row counts around an
   in-process assignment and could not fail; replaced in the final review).
4. Per-fact search work is halved, visible in the item 31 per-prompt timing on the next run.
5. `dup_in_invalidation_range` falls to **zero**, because that index range no longer exists.
   Note this is deliberately narrower than "out-of-range indices fall to zero":
   `dup_beyond_range` (a hallucinated index above the candidate count) remains possible and
   is still counted. Across 267 observed indices it was 0, so any non-zero value after this
   change is new information worth investigating, not noise.
