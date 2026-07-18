# Tombstone-Sweep Dirtiness — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Phase 3 hardening — close the one gap in incremental community refresh:
facts silently expired by the weekly staleness sweep must mark their communities
dirty so the incremental `theme-build` regenerates the now-stale reports.
**Date:** 2026-07-18
**Status:** Approved design — ready for implementation planning

Incremental refresh derives its dirty signal from `created_at` (adds/updates). The
staleness sweep expires facts by setting `invalid_at` + `expired_by_sweep=true`
without creating any new episode/entity/fact — so a swept community currently tests
as clean and reuses a stale report. This slice keys off the sweep's own markers to
close that gap, symmetrically with the `created_at` mechanism.

---

## 1. Scope

**In:** extend `theme_builder.incremental.touched_entities` (add sweep-expired
fact endpoints to the touched set) and `theme_builder.cli._corpus_cursor` (widen
the watermark to cover sweep `invalid_at`). No change to the sweep itself.

**Out (deferred):** graph-sync/sweep explicit entity marking (Approach B);
report-content changes for invalidated facts (the report layer already carries
`valid_at`/`invalid_at` per fact — out of scope here); split/merge and O(F×P)
match scaling (unchanged from the incremental slice).

## 2. Grounding (verified)

- `graph_extract.staleness_sweep.sweep_stale_facts` expires a fact with a SINGLE
  atomic Cypher `SET f.invalid_at = datetime(), f.expired_by_sweep = true` — so an
  expired fact carries a wall-clock `invalid_at` (the sweep run time) **and** the
  `expired_by_sweep=true` marker. The fact edge and its endpoint entities are NOT
  deleted (temporal policy: mark, never delete).
- `theme_builder.incremental.touched_entities(driver, group_id, prev_cursor) ->
  set[str] | None`: today unions {entities `created_at > prev_cursor`} ∪ {endpoints
  of facts `created_at > prev_cursor`}; `prev_cursor is None` → `None` (all dirty).
- `theme_builder.cli._corpus_cursor(driver, group_id) -> str | None`: today
  `max(created_at)` over `Episodic ∪ Entity ∪ RELATES_TO` via a `CALL {…UNION ALL…}`
  subquery, `toString(max(t))`.
- Graphiti's own invalidations set `invalid_at` to the contradicting episode's
  reference_time (a temporal, possibly historical date) — NOT a wall-clock
  "invalidated-at". But a graphiti invalidation always accompanies a new episode
  (`created_at > prev_cursor`), so it is already caught by the `created_at` signal.
  The sweep is the only path that invalidates with no new `created_at`, and it is
  the only writer of `expired_by_sweep`.

## 3. Architecture & data flow

```
weekly:  sweep_stale_facts -> SET f.invalid_at=now, f.expired_by_sweep=true   (no new created_at)
nightly: theme-build (incremental)
   prev_cursor = max(:Community.corpus_cursor)
   touched = {created_at > prev_cursor entities/fact-endpoints}               (existing)
           ∪ {endpoints of facts where expired_by_sweep=true AND invalid_at > prev_cursor}   (NEW)
   -> community containing a swept fact's endpoint is DIRTY -> regenerated
   new watermark = max created_at over Episodic/Entity/RELATES_TO
                 ∪ max invalid_at over RELATES_TO where expired_by_sweep=true    (NEW)
   -> stored corpus_cursor now covers the sweep's invalid_at
      -> next no-change run: invalid_at !> prev_cursor -> swept fact NOT touched -> CLEAN
```

The sweep and theme-build stay fully decoupled — they communicate only through
graph timestamps/markers, exactly like the `created_at` signal.

## 4. Components

### 4.1 `touched_entities` (extend, `incremental.py`)
After the existing entity + fact `created_at` queries, add a third within the same
session:
```cypher
MATCH (a:Entity {group_id:$g})-[f:RELATES_TO {group_id:$g}]->(b:Entity {group_id:$g})
WHERE f.expired_by_sweep = true AND f.invalid_at > datetime($c)
RETURN a.uuid AS a, b.uuid AS b
```
Add both endpoints to the touched set. (`prev_cursor is None` still short-circuits
to `None` before any query — swept facts are irrelevant on a cold start.)

### 4.2 `_corpus_cursor` (extend, `cli.py`)
Add a fourth `UNION ALL` branch to the watermark subquery so the stored cursor
advances past sweep expirations:
```cypher
UNION ALL MATCH ()-[f:RELATES_TO {group_id:$g}]->()
          WHERE f.expired_by_sweep = true RETURN f.invalid_at AS t
```
`toString(max(t))` over the combined stream is unchanged.

## 5. Design-decision alignment

- Reports stay fact-cited; the report LLM never writes a URL (regeneration reuses
  the unchanged `report.py`). `theme-builder` stays the only writer of
  `:Community`. The sweep remains the only writer of `expired_by_sweep`/its
  `invalid_at`. No new coupling — timestamp/marker-driven, like `created_at`.

## 6. Error handling & edge cases

| Case | Behavior |
|---|---|
| No swept facts | Both new queries return nothing → behavior identical to today; the current live baseline (0 swept facts) is unaffected. |
| Swept fact, `invalid_at > prev_cursor` | Endpoints touched → community dirty → regenerated once; the widened watermark then covers that `invalid_at` → subsequent no-change runs leave it clean (dirty-once). |
| Graphiti invalidation (historical `invalid_at`, no `expired_by_sweep`) | NOT matched by the new signal (keyed on `expired_by_sweep=true`); already caught by `created_at` on the accompanying new episode. No double-count, no false dirty. |
| Swept fact `invalid_at ≤ prev_cursor` | Already accounted in a prior refresh → not touched. |
| Cold start (`prev_cursor is None`) | `touched=None` → all dirty; the sweep signal is moot. |
| Swept fact's endpoint entity still a community member | It is (the sweep doesn't change graph topology) → dirty correctly; the regenerated report reflects the post-sweep fact set. |

## 7. Testing

- **Integration (Neo4j testcontainer) — `touched_entities`:** seed a fact with
  `expired_by_sweep=true, invalid_at > cursor` → its endpoints appear in the touched
  set; a fact with `invalid_at > cursor` but NO `expired_by_sweep` (graphiti-style)
  → NOT added by the sweep branch; a swept fact with `invalid_at ≤ cursor` → NOT
  added.
- **Integration — `_corpus_cursor`:** seed a swept fact whose `invalid_at` is later
  than every `created_at` → the watermark equals that `invalid_at`.
- **Integration — end-to-end (extend `test_incremental_cli.py`):** seed a persisted
  community + entities/fact; mark the fact `expired_by_sweep=true, invalid_at >
  the community's corpus_cursor` → an incremental run regenerates that community
  (report counter = 1); the watermark advances; a second run (no further change) →
  regenerates 0 (dirty-once).
- Full non-live suite + `ruff check src tests` + mypy clean. (The staleness sweep's
  own tests are unchanged.)

## 8. Acceptance criteria

1. A fact expired by the sweep (`expired_by_sweep=true`, `invalid_at > prev_cursor`)
   marks its community dirty → regenerated on the next incremental `theme-build`.
2. After regeneration the watermark covers the sweep `invalid_at`, so the swept
   community regenerates **exactly once** — subsequent no-change runs leave it clean.
3. Graphiti's own invalidations are not double-counted; a historically-`invalid_at`
   fact without the `expired_by_sweep` marker never falsely marks a community dirty.
4. Zero swept facts → behavior unchanged.
5. Integration + end-to-end tests green; full non-live suite + ruff/mypy clean.

## 9. Deferred

- graph-sync/sweep explicit entity dirty-marking (Approach B) if the marker signal
  proves insufficient.
- Regenerated-report content handling of invalidated facts (report layer already
  carries per-fact `valid_at`/`invalid_at`).
- An `@live` sweep demonstration (would require expiring a real live fact); covered
  by the integration end-to-end here.
