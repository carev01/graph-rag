# Maintenance Jobs — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Pipeline hardening — Sub-slice C (maintenance jobs). Completes the ingestion-side hardening arc.
**Date:** 2026-07-15
**Status:** Approved design — ready for implementation planning

Builds on Sub-slice A (incremental ingestion), B (robustness & economics), and the cleanup pass. Delivers the two deferred maintenance jobs — the residual-staleness sweep and structural↔semantic `SAME_AS` reconciliation — and folds in two Minors carried from earlier reviews: **#6** (the sweep must not expire live facts whose supporting episode is wrongly flagged `superseded`) and **#7** (the `should_distinct` metric's silent-merge blind spot). After C, the ingestion side is production-complete.

---

## 1. Problem & scope

Two gaps remain in keeping the temporal graph honest over time:
- **Silent staleness:** Graphiti invalidates facts it sees contradicted, but it cannot detect a fact whose supporting documentation was silently *deleted* (a `removed` tombstone, or an article that shrank so a chunk-episode was detached). Those facts linger as apparently-valid. A periodic Cypher sweep must expire facts whose support is all gone.
- **Structural↔semantic disconnect:** the deterministic structural layer (`:Vendor`/`:Product` from the DocExtractor catalog) and Graphiti's extracted `:Entity` nodes describe the same real-world vendors/products but are unlinked, so vendor-scoped queries can't bridge from the authoritative structural node to the semantic facts.

**In scope (C):** a `sweep` (fact-expiry) job, a `reconcile` (`SAME_AS`) job, and a `dedup_report_v2` silent-merge enhancement. All are batch/CLI-invoked; scheduling is an ops note.
**Out of scope:** the community layer / theme-builder (Phase 3); retrieval (Phase 2); the Region-magnet re-extraction (rides a future extraction change).

## 2. Grounding (verified against the live graph)

- Facts are `RELATES_TO` edges carrying an `episodes` list (supporting Episodic uuids), `valid_at`, `created_at`, and graphiti's nullable `invalid_at` (2073 facts, 785 already invalidated by graphiti).
- Provenance: `(:Article)-[:HAS_EPISODE {chunk_index}]->(:Episodic)`. On update the re-key **deletes** the old chunk's `HAS_EPISODE` edge and marks the old episode `superseded=true`; `tombstone_article_episodes` marks episodes `removed=true` (edges kept). So a *truly* stale episode has **no `HAS_EPISODE` edge**; a *removed*-article episode still has one but is flagged `removed`.
- Structural `:Vendor` (`AWS`, `Microsoft`) / `:Product` exist as non-`:Entity` nodes; semantic `:Entity:Vendor` includes `Amazon Web Services` (same as structural `AWS`, different name) plus noise (`vaults`, `Chris Hendon`). 0 `SAME_AS` edges today.

## 3. Component 1 — residual-staleness sweep (folds in #6)

**File:** `src/graph_extract/staleness_sweep.py` (new): `async def sweep_stale_facts(driver, group_id) -> dict`.

**"Dead episode" is defined by `HAS_EPISODE` linkage, NOT the `superseded` flag** — this is the #6 fix. An episode is **dead** iff:
- it is `removed=true`, OR
- it has **no** `HAS_EPISODE` edge from a non-removed `:Article`.

A superseded episode is dead because its edge was deleted; a shrink-restore episode wrongly flagged `superseded` is **alive** because it still holds its `HAS_EPISODE` edge — so its facts are never wrongly expired. The `superseded` flag is not consulted.

**A fact is expired iff all its supporting episodes are dead** (and no supporting episode is alive). Expiry = set graphiti's `invalid_at = datetime()` **and** a custom audit flag `expired_by_sweep = true`; only facts with `invalid_at IS NULL` are candidates (never re-expire, never touch graphiti's own invalidations). Sketch:

```cypher
MATCH (src)-[f:RELATES_TO {group_id:$g}]->(dst)
WHERE f.invalid_at IS NULL
WITH f, f.episodes AS eps
CALL {
  WITH eps
  UNWIND eps AS epu
  MATCH (e:Episodic {uuid: epu})
  OPTIONAL MATCH (a:Article)-[:HAS_EPISODE]->(e)
    WHERE coalesce(a.removed, false) = false
  WITH e, count(a) AS live_links
  RETURN sum(CASE WHEN coalesce(e.removed,false)=false AND live_links>0 THEN 1 ELSE 0 END) AS alive
}
WITH f WHERE eps IS NOT NULL AND size(eps) > 0 AND alive = 0
SET f.invalid_at = datetime(), f.expired_by_sweep = true
RETURN count(f) AS expired
```
(Facts with an empty/null `episodes` list are left alone — no support to reason about.) Returns `{scanned, expired}` plus a small sample of expired fact uuids for audit.

**CLI:** a `sweep` command (build driver, run, `_dump` the result, close in `finally`). Weekly cadence is an ops note (cron/scheduler out of scope).

## 4. Component 2 — structural↔semantic `SAME_AS` reconciliation

**Files:** `src/graph_extract/vendor_aliases.py` (new, data) + `src/graph_extract/reconcile.py` (new): `async def reconcile_same_as(driver, group_id) -> dict`.

Design-decision #5: **link, never merge.** Create `(:Vendor|:Product)-[:SAME_AS]->(:Entity)` between a structural node and the semantic entity that denotes the same thing.

**Matching (conservative, curated):** `vendor_aliases.py` holds a normalized alias map — canonical structural name → set of accepted semantic surface forms — e.g. `"AWS" -> {"aws", "amazon web services", "amazon"}`, `"Microsoft" -> {"microsoft", "azure", "microsoft azure"}`. Reconciliation, per structural `:Vendor`/`:Product` node:
- normalize its name; gather the accepted forms (its own normalized name + any alias-map forms);
- find semantic `:Entity` nodes (same `group_id`, typed `Vendor`/`Product`) whose normalized name is in that accepted set;
- `MERGE` a `SAME_AS` edge to each match (idempotent — re-running adds nothing).

Only confident, listed matches link; noise entities (`vaults`, `Chris Hendon`) never match. Returns `{structural_scanned, linked, unmatched_structural}` (the unmatched list surfaces vendors/products needing an alias-map entry). `SAME_AS` is created by this job only; it is additive and never deletes.

**Note:** the alias map is a small static data file (like `quality_labels`/`region_names`), dated, extended per-vendor as the corpus widens. Product reconciliation is best-effort (product names are messier); vendors are the reliable anchor.

## 5. Component 3 — `dedup_report_v2` silent-merge detection (#7)

**File:** `src/graph_extract/eval.py` (`dedup_report_v2`).

The current `should_distinct` `merged` state is name-set-identity based, so it misses a *silent* merge where one vendor's mentions attached to the other vendor's node without a separate node ever being created. Add a **`silent_merge_suspects`** field: for each `SHOULD_DISTINCT` pair, if a node matching **either** member has cross-vendor episode support (via the existing `_cross_vendor_names` set), record it as a suspected silent merge (name + which pair). Deterministic (no LLM), reuses `_cross_vendor_names`. This complements `suspect_false_merge` (which is corpus-wide) with a labelled-pair-scoped signal, closing the blind spot the T1 review of the quality-followups slice flagged.

## 6. Testing

- **Sweep** (Neo4j testcontainer, LLM-free — seed by hand): a fact whose only episode has no `HAS_EPISODE` edge → expired (`invalid_at` set, `expired_by_sweep=true`); a fact whose episode is `removed=true` → expired; a fact with a **live** episode (has `HAS_EPISODE`) but the episode carries a stale `superseded=true` flag → **NOT** expired (the #6 case); a fact already `invalid_at` → untouched; a fact with a live and a dead episode → not expired (any alive keeps it).
- **Reconcile** (Neo4j testcontainer): seed structural `:Vendor {name:'AWS'}` + semantic `:Entity:Vendor {name:'Amazon Web Services'}` → a `SAME_AS` is created via the alias map; a noise `:Entity:Vendor {name:'vaults'}` gets no link; re-running is idempotent (no duplicate `SAME_AS`); an unmatched structural node appears in `unmatched_structural`.
- **Silent-merge** (Neo4j testcontainer, extends `test_eval_quality.py`): a node matching a `SHOULD_DISTINCT` member with cross-vendor episode support appears in `silent_merge_suspects`; a cleanly-distinct pair yields none.
- **Aliases** (unit): `vendor_aliases` normalization + KEEP-guard (no alias form collides with a genuine *different* vendor).
- Full non-live suite + ruff/mypy clean.

## 7. Acceptance criteria

1. `sweep_stale_facts` expires exactly the facts whose supporting episodes are all dead (removed, or no `HAS_EPISODE`), sets `invalid_at`+`expired_by_sweep`, never re-expires or touches graphiti's own `invalid_at`, and **never expires a fact kept alive by a live-but-`superseded`-flagged episode** (#6).
2. `reconcile_same_as` links structural `:Vendor`/`:Product` to the correct semantic `:Entity` via `SAME_AS` using the alias map, is idempotent, never links noise, and reports unmatched structural nodes.
3. `dedup_report_v2` reports `silent_merge_suspects` for labelled distinct pairs with cross-vendor episode support (#7).
4. `sweep` and `reconcile` CLI commands exist and run; unit + integration tests green (Neo4j testcontainer); ruff/mypy clean.

## 8. Deferred

- Cron/scheduler wiring for the weekly sweep + periodic reconcile (ops; the CLI commands ship here).
- Auto-expanding the alias map (fuzzy/embedding matching) — only if the curated map proves insufficient as vendors are added.
- Splitting a *confirmed* silent merge (design forbids splitting false-merges; the metric only *surfaces* them).
- Community layer (Phase 3), retrieval (Phase 2), Region-magnet re-extraction.
