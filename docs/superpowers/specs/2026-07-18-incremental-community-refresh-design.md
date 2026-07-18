# Incremental Community Refresh — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Phase 3 hardening — make `theme-build` incremental: regenerate community
reports only for new/changed communities instead of LLM-rebuilding every community
on every run.
**Date:** 2026-07-18
**Status:** Approved design — ready for implementation planning

Today `theme-build` is full-rebuild only: it deletes and rewrites all `:Community`
nodes and LLM-regenerates every report on every run, regardless of what changed.
That won't scale as ingestion grows (the report LLM is the dominant cost). This
slice adds a timestamp-derived incremental path (Approach A): re-run detection
(cheap), Jaccard-match fresh communities to persisted ones for stable identity, and
LLM-regenerate reports **only** for new or touched communities.

---

## 1. Scope

**In:** a new `theme_builder/incremental.py` (touched-set, persisted-community
load, Jaccard matching, dirty/clean classification); a modified `writeback.py`
(carry per-community `embedding`/`generated_at` instead of stamping uniformly); an
incremental orchestration path in `cli.py` with a `--full` backstop flag; one
config field. Reuses `detect.py` (unchanged) and `report.py` (unchanged —
generate a report per dirty community).

**Out (deferred):** removal-driven dirtiness (tombstone-sweep integration; the
`created_at` signal catches adds/updates, not tombstone-only removals); optimal
community split/merge handling; graph-sync explicit dirty-marking (Approach B) if
the timestamp signal proves insufficient; scale-optimizing the O(F×P) Jaccard
match.

## 2. Grounding (verified)

- `Entity`, `Episodic`, and `RELATES_TO` all carry a wall-clock `created_at`
  (populated on every node/edge: 555 entities, 348 episodes). This is the reliable
  "written since" signal — dirty-detection lives entirely in theme-builder, no
  graph-sync change.
- `:Community` today: content-hash `community_id` (`sha1(f"{level}:"+sorted(members))[:16]`),
  `level`, `title`, `summary`, `full_report`, `rating`, `rating_explanation`,
  `tags`, `cited_fact_uuids`, `embedding`, `member_count`, `generated_at`
  (datetime), `corpus_cursor`; membership via `(:Community)<-[:IN_COMMUNITY]-(:Entity)`;
  hierarchy via `PARENT_OF`.
- **GDS Leiden is stochastic** — re-runs don't reproduce identical member sets even
  on unchanged regions, so exact content-hash matching across refreshes is fragile.
  Jaccard (fuzzy) matching is required to recognize a community as "the same" across
  refreshes.
- `detect.py`: `detect_communities(driver, group_id, *, min_community_size,
  max_levels) -> list[Community]`; `Community{community_id, level, member_uuids,
  parent_id}`; `_community_id(level, member_uuids)` (content hash).
- `report.py`: `generate_report(client, model, context) -> CommunityReport | None`;
  `CommunityReport{title, summary, full_report, rating, rating_explanation, tags,
  cited_fact_uuids}`.
- `writeback.py`: `write_communities(driver, embedder, group_id, communities,
  reports, corpus_cursor)` — atomic delete-all + write-all; computes embeddings +
  `generated_at=datetime()` uniformly.
- `cli.py`: `_run_theme_build(settings, *, driver)`; `_corpus_cursor` = max
  `Episodic.created_at`.

## 3. Architecture & flow

```
theme-build            (default: incremental; --full forces today's rebuild)
  1. DETECT (full, cheap, no LLM): detect_communities(...) -> fresh communities
  2. TOUCHED SET: prev_cursor = max(c.corpus_cursor) over persisted :Community
       touched = {e:Entity | e.created_at > prev_cursor}
               ∪ {endpoints of ()-[f:RELATES_TO]->() | f.created_at > prev_cursor}
       prev_cursor is None (no persisted communities) -> treat all fresh as new
  3. LOAD persisted: [{community_id, level, members, title, summary, full_report,
       rating, rating_explanation, tags, cited_fact_uuids, embedding, generated_at}]
  4. MATCH fresh->persisted per level by Jaccard(members) >= tau, greedy 1:1:
       matched fresh inherits the persisted (stable) community_id;
       unmatched fresh -> new birth id = _community_id(level, members)
  5. CLASSIFY each fresh community:
       unmatched (new)                         -> DIRTY
       matched AND members ∩ touched != {}     -> DIRTY
       matched AND no touched member           -> CLEAN
  6. REPORTS: generate_report(...) (+embed) for DIRTY only;
       CLEAN reuse matched persisted report fields + embedding + generated_at
  7. WRITEBACK (atomic): write fresh set under stable ids
       DIRTY: new report, generated_at=now, fresh embedding
       CLEAN: reused report/embedding/generated_at
       ALL:  corpus_cursor = new watermark (max Episodic.created_at)
       rebuild IN_COMMUNITY/PARENT_OF from the fresh partition;
       delete persisted communities absent from the fresh set (dissolved)
  return {communities_detected, reports_regenerated, reports_reused,
          communities_dissolved, by_level, corpus_cursor}
```

The cost win is entirely at step 6: detection + graph writes run fully (cheap); the
LLM only touches new/changed communities. Cold start (no persisted) → all new → all
dirty → equivalent to a full build.

## 4. Components (`theme_builder/incremental.py`)

### 4.1 `touched_entities`
`async def touched_entities(driver, group_id, prev_cursor: str | None) -> set[str] | None`
- `prev_cursor is None` → return a sentinel meaning "everything is new" (the caller
  then treats every fresh community as dirty; in practice all are also unmatched).
  Concretely: return `None` to signal "all new" (distinct from an empty set).
- Else: `MATCH (e:Entity {group_id:$g}) WHERE e.created_at > datetime($c) RETURN
  collect(e.uuid)` ∪ `MATCH (a:Entity {group_id:$g})-[f:RELATES_TO {group_id:$g}]->
  (b:Entity {group_id:$g}) WHERE f.created_at > datetime($c) RETURN
  collect(a.uuid)+collect(b.uuid)`. Union into a `set[str]`.

### 4.2 `load_persisted`
`async def load_persisted(driver, group_id) -> list[PersistedCommunity]`
- `PersistedCommunity` dataclass: `community_id, level, members: set[str], title,
  summary, full_report, rating, rating_explanation, tags, cited_fact_uuids,
  embedding, generated_at`.
- Query: `MATCH (c:Community {group_id:$g}) OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:Entity)
  WITH c, collect(e.uuid) AS members RETURN c{.*}, members`.

### 4.3 `match_communities` (pure)
`def match_communities(fresh: list[Community], persisted: list[PersistedCommunity],
*, tau: float) -> dict[int, PersistedCommunity | None]`
- For each `(fresh_idx, persisted)` pair **at the same level**, compute
  `jaccard = |A∩B| / |A∪B|`; keep pairs with `jaccard >= tau`. Sort all candidate
  pairs by jaccard desc; greedily assign 1:1 (each fresh and each persisted used at
  most once). Returns `{fresh_idx: matched PersistedCommunity or None}`.

### 4.4 `classify` (pure)
`def classify(fresh: list[Community], matches, touched: set[str] | None) ->
tuple[list[int], list[int]]` returning `(dirty_idxs, clean_idxs)`:
- `touched is None` (cold start) → all dirty.
- fresh `i` unmatched → dirty. Matched but `set(fresh[i].member_uuids) & touched` →
  dirty. Matched with no touched member → clean.

### 4.5 Orchestration (`cli.py::_run_theme_build`, incremental path)
1. `communities = await detect_communities(...)`.
2. `persisted = await load_persisted(...)`; `prev_cursor = max(p.generated-cursor)` —
   i.e. the max stored `corpus_cursor` over persisted communities (query it).
3. `touched = await touched_entities(driver, group_id, prev_cursor)`.
4. `matches = match_communities(communities, persisted, tau=settings.theme_refresh_jaccard_tau)`.
5. `dirty, clean = classify(communities, matches, touched)`.
6. Assign each fresh community its **stable id**: matched → `matches[i].community_id`;
   unmatched → its detect content-hash id.
7. For `dirty`: fetch context (existing `_fetch_members`/`_fetch_facts` +
   `assemble_context`) and `generate_report`; embed. For `clean`: reuse the matched
   persisted report fields + embedding + `generated_at`.
8. `new_cursor = await _corpus_cursor(...)`; `write_communities_incremental(driver,
   group_id, entries, corpus_cursor=new_cursor)` where each entry carries the stable
   id, member list, report fields, embedding, and generated_at.
- A `--full` flag routes to the existing full-rebuild path unchanged.

### 4.6 `writeback.py`
Add `write_communities_incremental(driver, group_id, entries, *, corpus_cursor)`
(or generalize `write_communities` to take pre-computed embeddings + per-community
`generated_at`). `entries`: `[{community_id, level, member_uuids, parent_id, title,
summary, full_report, rating, rating_explanation, tags, cited_fact_uuids, embedding,
generated_at}]`. Atomic single tx: `DETACH DELETE` all `:Community {group_id}` then
write every entry (using its carried `embedding`/`generated_at`, `corpus_cursor` =
the run watermark), rebuild `IN_COMMUNITY`/`PARENT_OF`. (Delete-all + write-all
stays — cheap graph writes; dissolved communities vanish by not being re-written.)

## 5. Config additions (`ExtractSettings`)

```
theme_refresh_jaccard_tau: float = 0.5   # min member-set Jaccard to treat a fresh community as the same as a persisted one
```
`--full` is a CLI flag on `theme-build` (not a setting); incremental is the default.

## 6. Design-decision alignment

- Reports remain fact-cited; the LLM never writes a URL (unchanged `report.py`).
  Reused reports were themselves generated under design #2. `theme-builder` remains
  the only writer of `:Community`. The layer stays derived & disposable — `--full`
  rebuilds it from scratch at any time.

## 7. Error handling & edge cases

| Case | Behavior |
|---|---|
| No persisted communities (cold start) / `prev_cursor` None | `touched=None` → all dirty → full generation. |
| Leiden wobble (members shuffle, no data change) | Absorbed by Jaccard; matched + untouched → reused. |
| Community split (1 persisted → 2 fresh) | Higher-Jaccard fresh inherits id + reuses; the other → new → dirty. |
| Community merge (2 persisted → 1 fresh) | Matches higher-Jaccard persisted; the other dissolves (deleted). |
| Entity leaves a community, rest untouched | Community may be clean with a slightly stale report; corrected on next `--full` or when a remaining member is touched. |
| Removal via tombstone sweep (no new episode) | NOT caught by `created_at`; covered by the weekly sweep + periodic `--full`. |
| A dirty community's report generation fails | Logged + skipped (existing behavior); it is simply not written this run. |

## 8. Testing

- **Unit — `match_communities`** (pure): exact-overlap match; wobble (small member
  diff) still matches; below-τ no match; split (1→2, only best inherits); merge
  (2→1, best wins, other unmatched); greedy 1:1 (a persisted used once); level
  isolation (never matches across levels).
- **Unit — `classify`** (pure): `touched=None` → all dirty; unmatched → dirty;
  matched+touched → dirty; matched+untouched → clean.
- **Integration (Neo4j testcontainer):** `touched_entities` (seed entities/facts
  with `created_at` above/below cursor); `load_persisted` (members via
  IN_COMMUNITY); `write_communities_incremental` (clean entry keeps its old
  embedding + `generated_at`; dirty gets new; dissolved persisted community deleted;
  stable id preserved; corpus_cursor bumped).
- **Integration — end-to-end incremental** (fake `detect_communities` + a
  report-generation counter): seed a persisted layer, run incremental with one
  community's member touched → assert the report generator is invoked **only** for
  the touched/new communities, others reused; a no-change run → generator invoked
  **zero** times.
- **`@live`:** on `backup-docs`, inject a synthetic new fact among an existing
  community's members, run incremental, assert only the affected community's
  `generated_at` advances (others unchanged) and stable ids persist; write
  `docs/superpowers/incremental-refresh-report.md`.
- Full non-live suite + `ruff check src tests` + mypy clean.

## 9. Acceptance criteria

1. A `theme-build` re-run with **no new content regenerates zero reports** (the
   report LLM is never called): every community reused, `generated_at` unchanged,
   `corpus_cursor` bumped to the new watermark.
2. After new content, **only** communities containing a touched entity (or entirely
   new communities) are LLM-regenerated; untouched communities are reused.
3. Community identity is **stable**: an untouched community keeps its
   `community_id` across refreshes (Jaccard-matched), tolerating Leiden re-run
   wobble.
4. Dissolved communities (no fresh match) are deleted; `IN_COMMUNITY`/`PARENT_OF`
   reflect the fresh partition. Writeback is atomic.
5. `--full` still performs today's full rebuild (backstop). Cold start works.
6. Reports remain fact-cited; no LLM authors a URL (design #2).
7. Unit + integration tests green; `@live` incremental demonstration passes; full
   non-live suite + ruff/mypy clean.

## 10. Deferred

- Removal-driven dirtiness (integrate the weekly tombstone sweep so silent
  deletions mark communities dirty).
- Optimal split/merge handling (currently best-Jaccard-wins; the loser regenerates
  or dissolves).
- graph-sync explicit dirty-marking (Approach B) if the `created_at` signal proves
  insufficient in practice.
- Scale-optimizing the O(F×P) per-level Jaccard match (fine at hundreds of
  communities; revisit at thousands).
- A `member_hash` fast-path for exact-unchanged detection (Jaccard covers it).
