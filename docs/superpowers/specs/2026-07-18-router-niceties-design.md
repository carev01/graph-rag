# `/answer` Router Niceties — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Phase 4, slice 3 — three deferred `/answer` router enhancements: a
**freshness block**, **richer citations** (vendor/product/section + valid_at/
invalid_at), and a **global→local fallback** arc.
**Date:** 2026-07-18
**Status:** Approved design — ready for implementation planning

Builds on the merged `/answer` router (classify → dispatch → uniform
`{mode, answer, citations, routing}` envelope). Two skipped §11 items stay
deferred: a single-product→local heuristic (the cheap LLM already routes these
correctly) and a classifier confidence score (the `routing.via` field is a
sufficient coarse signal).

---

## 1. Scope

**In:**
- `answer_api/freshness.py` — a `freshness(driver, group_id, *, reports)` helper;
  the router stamps `freshness` onto the envelope.
- `graph_extract/provenance.py` — enrich `resolve_citations` (structural
  vendor/product + `HAS_EPISODE.heading_path` section + fact `valid_at`/
  `invalid_at`); update its 4 callers.
- `answer_api/router.py` — the global→local fallback arc + freshness wiring.

**Out (still deferred):** single-product→local heuristic; classifier confidence
score; caching the freshness/`graph_cursor_time` aggregation; MCP exposure
(Phase 5).

## 2. Grounding (verified)

- `:Community` nodes carry `generated_at` (datetime, set at write-back) and
  `corpus_cursor`; there are ~66 of them (cheap to aggregate).
- The structural graph exists: `(v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(:Source)-[:HAS_ARTICLE]->(a:Article)-[he:HAS_EPISODE]->(e:Episodic)`;
  `he.heading_path` is set by `Provenance.link` (→ section).
- `RELATES_TO` fact edges carry `valid_at`/`invalid_at`.
- `Provenance.resolve_citations(fact_uuids) -> {uuid: [ {url,title,article_id} ]}`
  currently; direct callers: `search.search_local`, `timeline.timeline_local`,
  `global_search.global_search` (reduce), `drift._synthesize`. `answer_local`
  builds its citations from `search_local` results (indirect).
- Router: `answer_router` classifies → `_dispatch` → local-empty→drift fallback →
  `_normalize` into `{mode, query, answer, citations, routing}`. `global_search`
  returns `communities_used: []` on an empty shortlist (its refusal path).

## 3. Freshness block

New `answer_api/freshness.py`:
`async def freshness(driver, group_id, *, reports: bool) -> dict`
- `graph_cursor_time` (always): `MATCH (e:Episodic {group_id:$g}) RETURN
  toString(max(e.created_at)) AS c` — the live-retrieval "graph as-of".
- `reports_as_of` (only when `reports=True`): `MATCH (c:Community {group_id:$g})
  RETURN toString(max(c.generated_at)) AS c` — when the thematic layer was last
  built. `None` when `reports=False`.
- Returns `{"graph_cursor_time": <str|None>, "reports_as_of": <str|None>}`.
- **Resilient:** any query error is caught and yields `None` for that field —
  freshness never blocks or fails the answer.

Wiring: `answer_router` computes `reports = final_mode in ("global", "drift")`
(the FINAL, post-fallback mode) and adds `env["freshness"] = await
freshness(driver, group_id, reports=reports)` after `_normalize`. `freshness` is a
module-attr seam for tests.

## 4. Richer citations

Single enrichment point — `Provenance.resolve_citations` — so every mode gains
richer citations consistently.

### 4.1 Enriched query
```cypher
MATCH ()-[f:RELATES_TO]->() WHERE f.uuid IN $uuids
OPTIONAL MATCH (a:Article)-[he:HAS_EPISODE]->(e:Episodic) WHERE e.uuid IN f.episodes
OPTIONAL MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(:Source)-[:HAS_ARTICLE]->(a)
WITH f.uuid AS uuid, toString(f.valid_at) AS valid_at, toString(f.invalid_at) AS invalid_at,
     collect(DISTINCT CASE WHEN a IS NULL THEN NULL ELSE
       {url:a.source_url, title:a.title, article_id:a.id,
        section:he.heading_path, vendor:v.name, product:p.name} END) AS raw
RETURN uuid, valid_at, invalid_at, [x IN raw WHERE x IS NOT NULL] AS sources
```

### 4.2 New return shape (the one breaking change)
`resolve_citations(fact_uuids) -> {uuid: {"valid_at": str|None, "invalid_at":
str|None, "sources": [ {url, title, article_id, vendor, product, section} ]}}`.
`vendor`/`product`/`section` are `None` when the structural parents or heading are
absent (best-effort). `resolve_chain` (separate method) is untouched.

### 4.3 Caller updates
- **`search.search_local`:** `sources = citations.get(uuid, {}).get("sources", [])`;
  results keep their own edge-derived `valid_at`/`invalid_at`.
- **`timeline.timeline_local`:** same source extraction; timeline entries already
  carry `valid_at`/`invalid_at` from the edge.
- **`global_search` reduce & `drift._synthesize`:** each citation becomes
  `{marker, fact_uuid, valid_at, invalid_at, sources}` — reading
  `resolved.get(uuid, {})`.
- **`synthesize.answer_local`:** citation keeps `fact`, gains `valid_at`/
  `invalid_at` (from the `search_local` result; `invalid_at` is `None` for local,
  which excludes invalid facts by default), and its `sources` are already enriched
  via `search_local`.
- **`router._render_timeline`:** its citations gain `valid_at`/`invalid_at` (the
  timeline entries already carry both from the edge); sources are enriched via the
  entries' resolved sources. Keeps the timeline mode's citations uniform with the
  others.

### 4.4 Resulting citation shape (all modes, uniform)
`{marker, fact_uuid, valid_at, invalid_at, sources:[{url, title, article_id,
vendor, product, section}]}` (`answer_local` also keeps `fact`). Additive for
consumers — `sources[0]["url"]` still resolves.

**Design #2:** every new field comes from graph traversal (structural layer + edge
properties), never an LLM.

## 5. global→local fallback

In `answer_router`, alongside the existing local-empty→drift arc:
- After dispatching `global`, if `not raw.get("communities_used")` (empty shortlist
  → global's refusal), escalate to `local`: re-dispatch, set `mode="local"`,
  `routing.fallback_from="global"`.
- **Terminal, no chaining:** each dispatch escalates at most once. (drift would not
  help without communities — it degrades to local anyway — so global→local is the
  right terminal, not global→drift.) The two arcs are independent one-shots:
  `local(retrieved==0)→drift` and `global(no communities_used)→local`.

## 6. Error handling

| Condition | Behavior |
|---|---|
| Freshness query error/timeout | That field → `None`; the answer is unaffected. |
| Structural parents / heading missing for a source | `vendor`/`product`/`section` → `None` (best-effort). |
| A fact with no resolvable article | `sources: []`, `valid_at`/`invalid_at` still returned from the edge. |
| global escalates to local, and local is also empty | Return local's refusal (no further chain — one escalation per dispatch). |

## 7. Testing

- **Unit — `freshness`** (fake driver returning canned rows): `graph_cursor_time`
  always present; `reports_as_of` present only when `reports=True`, else `None`; a
  raising driver → both/relevant fields `None` (no exception escapes).
- **Unit — router** (monkeypatched modes + freshness): the envelope carries a
  `freshness` block; `reports_as_of` populated for global/drift, `None` for local/
  timeline (using the FINAL mode after any fallback); `global` with empty
  `communities_used` → local dispatched, `routing.fallback_from="global"`,
  `mode="local"`; a global with communities stays global.
- **Integration (Neo4j testcontainer) — enriched `resolve_citations`:** seed
  `Vendor→Product→Source→Article-[HAS_EPISODE {heading_path}]->Episodic` +
  `RELATES_TO {valid_at, invalid_at, episodes}`; assert the returned dict has
  `valid_at`/`invalid_at` and each source carries `vendor`/`product`/`section`;
  a fact whose article lacks structural parents → those fields `None` but `url`
  present. Update the existing `resolve_citations`/`search_local`/`timeline`
  integration tests for the new `{uuid: {…, sources}}` shape.
- **App (`test_answer_api_app.py`):** the `/answer` envelope includes `freshness`;
  existing endpoint tests still pass (citation shape is additive).
- **`@live` smoke + demonstration:** a `/answer` run on `backup-docs` showing the
  `freshness` block, a citation with `vendor`/`product`/`section` populated, and a
  `valid_at`/`invalid_at` on a cited fact → `docs/superpowers/router-niceties-report.md`.
- Full non-live suite + `ruff check src tests` + mypy clean.

## 8. Acceptance criteria

1. Every `/answer` response carries `freshness: {graph_cursor_time, reports_as_of}`;
   `reports_as_of` is non-null only for community-based modes (global/drift, by
   final mode), `graph_cursor_time` reflects the latest episode; a freshness query
   failure degrades to `null`, never an error.
2. `resolve_citations` returns `{uuid: {valid_at, invalid_at, sources}}` with each
   source carrying `vendor`/`product`/`section` (best-effort `None`); all four
   modes and the `/answer` envelope surface the richer citation shape; existing
   `sources[0]["url"]` access still works.
3. A `global` classification that shortlists no communities escalates to `local`
   (`routing.fallback_from="global"`); each dispatch escalates at most once.
4. No LLM authors any new field (design #2); the router still authors no prose/URL.
5. Unit + integration + app tests green; `@live` smoke passes; a demonstration
   report is written; full non-live suite + ruff/mypy clean.

## 9. Deferred

- Single-product→local heuristic; classifier confidence score.
- Caching the `graph_cursor_time` aggregation (a `max(Episodic.created_at)` scan)
  if it proves slow at scale.
- Richer citation provenance beyond vendor/product/section (e.g. the superseded
  article version's URL for temporal citations).
- MCP exposure (Phase 5).
