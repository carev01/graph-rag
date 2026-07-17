# `/timeline` — Open Follow-ups

Forward-looking only. The `/timeline` slice (bi-temporal retrieval endpoint)
merged clean; the final whole-branch review returned **Ready to merge** with no
Critical/Important findings. The items below are the three Minor review notes
plus the spec's deferred work. None blocks anything.

## Minor review findings (from final whole-branch review, 2026-07-16)

1. **Cross-timezone string ordering in `timeline_local` sort key**
   (`src/answer_api/timeline.py`, sort key `(valid_at is None, str(valid_at or ""))`).
   The sort stringifies `valid_at` (a `datetime`) and compares lexicographically.
   Lexicographic == chronological **only when all offsets are identical**. Graphiti
   normalizes to UTC (`+00:00`) today, so this cannot misorder on current data —
   hence Minor. Concrete latent failure: an edge at `08:24:39+05:00` (03:24 UTC)
   would sort *after* `05:00:00+00:00` (05:00 UTC) despite being earlier.
   **Fix:** sort by the raw `datetime` with a `datetime.min/max` sentinel for
   `None`, instead of a stringified key. Removes the reliance on UTC normalization.

2. **No lower bound on `limit` (parity nit shared with `search_local`)**
   (`src/answer_api/app.py` `/timeline`, and `/search/local`, `/answer`).
   `limit=-1` → `fetch_limit=max(-3,-1)=-1` → `edges[:-1]` silently drops the last
   edge instead of erroring. Pre-existing pattern (`search_local`'s `[:k]` behaves
   the same), so it's a parity nit, not a regression. **Fix:** add `Query(ge=1)`
   bounds to `limit`/`k` on all three endpoints in one pass.

3. **`group_id` asymmetry: `_sweep_flags` vs `resolve_citations`** (not a bug today)
   (`src/answer_api/timeline.py` `_sweep_flags` filters `RELATES_TO {group_id:$g}`;
   `src/graph_extract/provenance.py` `resolve_citations` matches `RELATES_TO`
   without a group filter). The uuids are already group-scoped by `_retrieve_edges`
   and graphiti sets `group_id` on every `RELATES_TO`, so both resolve the same set.
   Worth a one-line note only: a future edge missing `group_id` would classify
   `superseded` (sweep row absent) while still getting citations. **Fix (if ever):**
   make the two group_id filters consistent.

## Deferred capability (from the design spec §8)

- **LLM narrative timeline** — a synthesized "here's how X evolved over time" over
  the ordered facts, reusing the GLM-5.2 synthesis layer + design-decision #2
  markers (LLM emits `[N]`, deterministic resolver expands to sources). The natural
  next add on top of `/timeline`. Retrieval-first `/timeline` is the substrate.
- **Time-window filtering** (`?since=&until=`), a `superseded_by` fact-to-fact
  linkage (graphiti gives no direct pointer), and richer grouping (by entity/period).

## Honest data caveat (carried from the design + demonstration report)

Today's invalidations are largely **event-time supersession within document
content** (e.g. a CloudTrail event pair hours apart) plus article `reference_time`,
not documentation evolving over months. The `/timeline` **mechanism works today**
(demonstrated end-to-end on the live graph); its "how did vendor X's treatment
change?" value **grows with accumulated incremental history**. See
[`timeline-report.md`](timeline-report.md).
