# Temporal Coherence — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Make design decision #3 (*updates append, they don't overwrite*) coherent —
one liveness rule shared by the sweep, search and timeline, and citations that still
resolve for superseded facts.
**Date:** 2026-09-04
**Status:** Approved design — ready for implementation planning

Arises from `docs/superpowers/project-review-2026-09-04.md`, which found the temporal
policy implemented two contradictory ways and silently wrong on the live graph.

---

## 1. The problem

The system overloads `HAS_EPISODE` **linkage** to mean two different things — *"this
fact is citable"* and *"this fact is still supported by current content"*. They must
diverge, and today they cannot.

**Defect A — superseded facts lose their citations.**
`Provenance.link` (`src/graph_extract/provenance.py:23-25`) runs
`SET oldE.superseded = true DELETE old` when a chunk is re-keyed. `resolve_citations`
reaches an article via `(a:Article)-[he:HAS_EPISODE]->(e:Episodic)`, so once that edge
is gone a superseded fact resolves to **no sources**. `/timeline` can still surface the
fact but cannot cite it — breaking design decision #2 for exactly the historical
answers the temporal graph exists to give. The plan (§6.4) wants both old and new
citable.

**Defect B — a shrunk article's facts never expire.**
`IngestDriver._supersede_trailing_episodes` (`ingest_driver.py:111-114`) marks dropped
trailing chunks `superseded=true` but **keeps** their `HAS_EPISODE` edge. The sweep
(`staleness_sweep.py`) keys liveness on linkage and, by its own docstring, *never*
reads `superseded`. So those episodes look alive, and facts from content that no longer
exists stay current indefinitely. The sweep exists precisely to catch this case and is
defeated by it.

The two paths are each internally consistent and mutually contradictory: `link` deletes
the edge and relies on the sweep ignoring `superseded`; `_supersede_trailing_episodes`
relies on `superseded` meaning something the sweep refuses to consult.

**Not in this slice (§7):** 43 facts (3.3%) carry a spurious `invalid_at`, and the
ontology's `Vendor`/`Product` labels collide with structural labels.

## 2. The rule

Split the two meanings onto two mechanisms:

| Question | Answered by |
|---|---|
| *Can we show where this fact came from?* | the **existence** of the `HAS_EPISODE` edge — permanent |
| *Is this fact still supported by current content?* | the **`superseded` flag on that edge** — mutable |

**Liveness (one definition, one module).** A supporting episode is **alive** iff all
three hold:

1. the `HAS_EPISODE` edge is not `superseded` (absent property ⇒ alive),
2. the `Episodic` node is not `removed`,
3. the `Article` it hangs from is not `removed`.

A fact is **live** iff at least one supporting episode is alive.

**Citability is unconditional.** `resolve_citations` traverses `HAS_EPISODE`
irrespective of `superseded`, so a fact that was ever true remains attributable to the
document version that asserted it.

## 3. Components

### 3.1 `src/graph_extract/episode_liveness.py` (new)

The single home for the rule. Exports the Cypher predicate as a constant so callers
compose it rather than restating it:

- `ALIVE_EPISODE_PREDICATE: str` — a Cypher boolean over bound `he` (the `HAS_EPISODE`
  relationship), `e` (`:Episodic`) and `a` (`:Article`).
- `LIVE_FACT_EXISTS: str` — the `EXISTS { … }` form used to filter facts.

Both use `coalesce(x, false)` so a missing property means alive, which is what makes
the existing 161 unflagged edges behave correctly without a backfill.

### 3.2 `Provenance.link` — mark, don't delete

Replace `SET oldE.superseded = true DELETE old` with a flag on the relationship:

```
OPTIONAL MATCH (a)-[old:HAS_EPISODE {chunk_index:$i}]->(oldE:Episodic)
WHERE oldE.uuid <> $u
SET old.superseded = true, oldE.superseded = true
```

The edge survives, so the old episode stays citable; the flag makes it dead for
liveness. The `oldE.superseded` node flag is kept for backward compatibility with
existing tests and diagnostics, but **no liveness decision reads the node flag** — the
edge flag is authoritative, because one episode can in principle be referenced by more
than one article.

### 3.3 `_supersede_trailing_episodes` — same mechanism

Set `r.superseded = true` on the edge (not only `e.superseded` on the node), so a
shrunk article's dropped chunks die by exactly the mechanism `link` uses. This is what
makes Defect B fixed *by construction* rather than by a second rule.

### 3.4 `staleness_sweep` — consume the shared predicate

Rewrite `_SWEEP` to use `ALIVE_EPISODE_PREDICATE`. Its docstring currently argues at
length that liveness must never read `superseded` *because `link` deletes the edge* —
that reasoning is invalidated by §3.2 and must be replaced, not left as a trap.

The sweep's other invariants are unchanged: only facts with `invalid_at IS NULL` and a
non-empty `episodes` list are candidates; find-and-`SET` stay in one statement; expired
facts are stamped `expired_by_sweep = true`.

### 3.5 Retrieval — consumes the result, does not re-derive the rule

**Search and timeline need no change**, and that is the point rather than an omission.

`search_local` filters on `e.invalid_at` in Python over the edges graphiti returns
(`search.py:38`); it never runs our Cypher. `timeline._fact_status` likewise reads
`invalid_at` plus `expired_by_sweep`. Both are *consumers* of a decision the sweep has
already made.

So the architecture is: **the rule is defined once (§3.1), applied by the sweep (§3.4),
and materialised into `invalid_at`.** Search and timeline read that materialised result.
Pushing the predicate into retrieval as well would mean two implementations of liveness
evaluated at different times against the same graph — reintroducing exactly the class of
divergence this slice exists to remove.

One consequence is accepted deliberately: between an ingest that supersedes an episode
and the next sweep run, a dead fact still has `invalid_at IS NULL` and can be retrieved.
That staleness window is the existing weekly-batch design (`CLAUDE.md`, decision #3), not
a defect introduced here. Closing it in real time is a separate change with its own
cost, and is out of scope.

## 4. Migration

None required. The live graph holds 161 `HAS_EPISODE` edges, none superseded, and
today's re-ingest was clean, so there is no history to preserve. The `coalesce(…, false)`
treatment of a missing property (§3.1) is what makes existing edges correct by default —
this is a behavioural change with no data migration, deliberately.

## 5. Proof: one real update through the queue

The acceptance test is the scenario the review found has never run — `semantic_jobs`
holds 593 rows and zero completions, and every ingest so far bypassed the queue via
`cli ingest`.

A `@live` test that:

1. ingests a synthetic article of three chunks,
2. re-ingests a modified version: chunk 1 unchanged, chunk 2 **edited**, chunk 3
   **dropped**,
3. asserts, in order:
   - the superseded chunk-2 episode **keeps** its `HAS_EPISODE` edge and its facts still
     `resolve_citations` to the article's URL (Defect A),
   - after `sweep_stale_facts`, the dropped chunk-3 facts are expired and stamped
     `expired_by_sweep` (Defect B),
   - the unchanged chunk-1 facts are untouched,
   - `search_local` returns the new chunk-2 facts and not the superseded ones,
   - `/timeline` reports the chunk-2 change with citations resolving for **both** the
     old and new fact.

Driven through `graph_sync.cli worker` against `semantic_jobs`, not through
`cli ingest`, so the queue path is exercised for the first time.

**Risk, stated plainly.** The queue has genuinely never run: `semantic_jobs` holds 593
rows, all `pending`, and `token_ledger` is empty. So step 5 depends on a code path with
no execution history, and it may surface its own unrelated defects — which would balloon
this slice.

**Mitigation, decided in advance rather than improvised:** the five assertions are the
deliverable; the queue is the preferred *vehicle*. If the worker proves broken, the
implementer records precisely what failed as a finding for a follow-up slice, then runs
the same five assertions through `cli ingest` and ships. Proving the temporal rule is
correct must not be held hostage to fixing an unrelated subsystem. The worker's daily
budget default admits roughly 25 bootstrap articles/day (§7), so the test article must
be injected as a targeted job rather than by draining the existing 593-row backlog.

## 6. Error handling and edge cases

| Case | Behaviour |
|---|---|
| Edge predates this change (no `superseded` property) | Alive. `coalesce(he.superseded, false)`. |
| Shrink then restore (chunk dropped, later returns) | Re-ingest re-keys the same `chunk_index`; the restored edge is written without the flag, so the episode is alive again. This is the case the old sweep docstring worried about, and the edge flag handles it correctly. |
| Episode referenced by two articles | Liveness is per-edge, so it dies only for the article that superseded it. This is why the edge flag, not the node flag, is authoritative. |
| Article tombstoned (`removed=true`) | All its episodes are dead by clause 3, regardless of edge flags. Unchanged. |
| Fact already `invalid_at` (graphiti or a prior sweep) | Not a sweep candidate. Unchanged. |
| A fact with an empty `episodes` list | Not a candidate; cannot be judged. Unchanged. |

## 7. Deferred to their own slices

- **The 43 phantom invalidations.** Facts carry `invalid_at` equal to their episode's
  reference time with no genuine successor — e.g. *"AWS Backup supports Amazon FSx for
  NetApp ONTAP file systems"*, a current capability, invalid 23h53m after its `valid_at`.
  This is graphiti contradiction-detection misfiring on complementary statements, not a
  liveness problem: a different root cause needing prompt/ontology work. It is also why
  the old eval scored 0.20 on timeline grounding.
- **Ontology/structural label collision.** `MATCH (v:Vendor)` returns 22, not 2 — 20 are
  extracted `:Entity` nodes wearing the structural label. Latent rather than broken today
  (our structural queries anchor on the `Vendor→Product→Source→Article` chain), but any
  bare label match is wrong.
- The `semantic_jobs` daily budget default, which admits roughly 25 bootstrap
  articles/day.

## 8. Testing

- **Unit:** the predicate composes into valid Cypher; missing property ⇒ alive; each of
  the three clauses independently kills liveness.
- **Integration (testcontainer):** build a small graph by hand covering every §6 row —
  re-keyed chunk, dropped trailing chunk, restored chunk, two-article episode, tombstoned
  article — and assert the sweep expires exactly the facts with no alive supporting
  episode, then that search and timeline agree with that outcome. Agreement between the
  three is the property that was missing; §3.5 is what guarantees it cannot drift again.
- **`@live`:** the §5 end-to-end update through the queue.
- Full non-live suite, `uv run ruff check src tests`, `uv run mypy src` clean.

## 9. Acceptance criteria

1. `Provenance.link` no longer deletes `HAS_EPISODE`; re-keyed edges are flagged instead.
2. `_supersede_trailing_episodes` flags the same edge property, so shrink and re-key
   share one mechanism.
3. Exactly one definition of episode liveness exists in `src/`, in
   `episode_liveness.py`, and the staleness sweep is its only evaluator; search and
   timeline consume the materialised `invalid_at` rather than re-deriving the rule.
4. `resolve_citations` resolves a superseded fact to its source URL.
5. The staleness sweep expires facts from a shrunk article's dropped chunks.
6. The §5 `@live` update runs through `semantic_jobs` and all five assertions hold.
7. The sweep's docstring no longer claims liveness must ignore `superseded`.
8. Full non-live suite, ruff and mypy clean.
