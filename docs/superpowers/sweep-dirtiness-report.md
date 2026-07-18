# Tombstone-Sweep Dirtiness — Demonstration Report

**Slice:** Phase 3 hardening — close the one gap in incremental community
refresh: facts silently expired by the weekly staleness sweep must mark their
communities dirty so incremental `theme-build` regenerates the now-stale
report instead of reusing it.
**Date:** 2026-07-18
**Status:** Implemented, tested, full non-live suite + ruff + mypy clean.

## The gap

Incremental `theme-build` derives its dirty signal from `created_at`: a
community is dirty if any member entity, or the endpoint of any fact touching
a member, has a `created_at` after the stored `corpus_cursor` watermark
(see `docs/superpowers/incremental-refresh-report.md`).

The weekly staleness sweep (`graph_extract.staleness_sweep.sweep_stale_facts`)
expires a fact whose supporting episodes are all superseded/removed with a
single atomic Cypher `SET f.invalid_at = datetime(), f.expired_by_sweep =
true` — per the temporal policy, it never deletes the edge and it creates no
new episode, entity, or fact. That means a swept fact carries **no new
`created_at` anywhere in the graph**. The existing dirty signal is blind to
it: the community containing the swept fact's endpoints tests as clean and
the incremental run reuses its stale, pre-sweep report indefinitely.

Graphiti's own fact invalidations are not the same case: they always set
`invalid_at` to a contradicting episode's reference time, but that
invalidation always accompanies a *new* episode, so it's already caught by
the `created_at` signal. The sweep is the only path that invalidates a fact
with no accompanying `created_at`, and it is the only writer of the
`expired_by_sweep` marker — which is what makes it safe to key off that
marker specifically rather than off bare `invalid_at`.

## The fix

Two symmetric extensions, both merged in `3fbe364` (Task 1 of this slice):

1. **`theme_builder.incremental.touched_entities`** (`src/theme_builder/incremental.py`) —
   a third query, alongside the existing `created_at` ones, adds the endpoints
   of any fact where `expired_by_sweep = true AND invalid_at > prev_cursor` to
   the touched set:
   ```cypher
   MATCH (a:Entity {group_id:$g})-[f:RELATES_TO {group_id:$g}]->(b:Entity {group_id:$g})
   WHERE f.expired_by_sweep = true AND f.invalid_at > datetime($c)
   RETURN a.uuid AS a, b.uuid AS b
   ```
   This makes a community dirty the first time its member fact is swept.

2. **`theme_builder.cli._corpus_cursor`** (`src/theme_builder/cli.py`) — a
   fourth `UNION ALL` branch widens the watermark to also cover swept
   `invalid_at` values:
   ```cypher
   UNION ALL MATCH ()-[f:RELATES_TO {group_id:$g}]->()
             WHERE f.expired_by_sweep = true RETURN f.invalid_at AS t
   ```
   This makes the *next* stored `corpus_cursor` advance past the sweep, so the
   same swept fact does not test as `invalid_at > prev_cursor` again.

Both branches are required together. `touched_entities` alone would mark the
community dirty on every run forever (`invalid_at > prev_cursor` stays true
since the cursor never moves past it). `_corpus_cursor` alone would never
observe the sweep in the first place, since nothing feeds it into the touched
set. Together they produce **dirty exactly once**: the run that first sees a
newly swept fact regenerates the community; every run after that sees a
watermark that already covers the sweep's `invalid_at`, so the fact stops
looking touched.

Zero swept facts anywhere in the graph means both new query branches return
nothing, so behavior is byte-for-byte identical to before this slice —
verified by every pre-existing incremental test staying green, including the
`@live` no-change smoke.

## End-to-end proof

`tests/integration/test_incremental_cli.py::test_swept_community_regenerates_once`
seeds two persisted communities, A={e1,e2} and B={e3,e4}, both with
`corpus_cursor = 2026-03-01` and no entity `created_at` after that date (so
neither would be dirty under the old `created_at`-only signal). It then adds
one `RELATES_TO` fact between A's members, `e1`→`e2`, marked
`expired_by_sweep: true, invalid_at: 2026-06-01` — i.e. swept *after* A's
cursor, with no new `created_at` anywhere.

- **Run 1** (`r1 = await cli._run_theme_build_incremental(...)`): the report
  generator is called exactly once (`calls["n"] == 1`) — only A, whose fact
  was swept. `r1["reports_regenerated"] == 1` and `r1["reports_reused"] == 1`
  (B, untouched, is reused). The persisted `corpus_cursor` for A is rewritten
  to the new global watermark, which now covers the sweep's `2026-06-01`
  `invalid_at`.
- **Run 2** (`r2 = await cli._run_theme_build_incremental(...)`, same graph,
  no further change): the report generator call count is still `calls["n"]
  == 1` — zero additional calls. `r2["reports_regenerated"] == 0` and
  `r2["reports_reused"] == 2` (both A and B now reused).

That is the dirty-once guarantee demonstrated directly: a sweep-expired fact
with no `created_at` signal still forces exactly one regeneration of its
community, and the very next run is clean.

```
$ uv run --extra dev pytest tests/integration/test_incremental_cli.py::test_swept_community_regenerates_once -q
1 passed, 3 warnings in 23.27s
```

## Full-suite gate

```
$ uv run --extra dev pytest -m "not live" -q
448 passed, 12 deselected, 25 warnings in 510.52s

$ uv run ruff check src tests
All checks passed!

$ uv run mypy src
Success: no issues found in 59 source files
```

## Design-decision alignment

Reports remain fact-cited; the report LLM is not touched by this slice — a
swept community's regenerated report goes through the same unchanged
`report.py` synthesis path, so the "LLM never writes a URL" invariant is
unaffected. `theme-builder` stays the only writer of `:Community`; the sweep
stays the only writer of `expired_by_sweep`/its `invalid_at` on
`RELATES_TO`. The two layers remain fully decoupled, communicating only
through graph timestamps and markers — the same pattern as the existing
`created_at` dirty signal.

## Deferred (per spec §9)

- **An `@live` sweep demonstration** — showing this working against the real
  `backup-docs` graph would require deliberately expiring a real live fact
  (running `sweep_stale_facts` against production data, or fabricating a
  fact whose supporting episodes are all superseded/removed) purely to
  exercise this test. The integration end-to-end test above already proves
  the mechanism against a real Neo4j instance (testcontainer), so the live
  demo is deferred rather than run against production state.
- graph-sync/sweep explicit entity dirty-marking (Approach B), if the
  marker-based signal ever proves insufficient at scale.
- Regenerated-report content handling of invalidated facts — the report
  layer already carries per-fact `valid_at`/`invalid_at`; unchanged here.
