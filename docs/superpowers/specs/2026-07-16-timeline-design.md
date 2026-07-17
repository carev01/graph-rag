# Temporal Timeline (`/timeline`) — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Phase 2 retrieval — `/timeline`: a bi-temporal endpoint returning facts about a topic ordered chronologically, with their validity intervals and supersession status.
**Date:** 2026-07-16
**Status:** Approved design — ready for implementation planning

The endpoint that exposes the append-not-overwrite temporal model (design-decision #3). Retrieval-first (no LLM, like `/search/local`): hybrid-retrieve relevant fact edges **including invalidated ones**, annotate each with its status + validity interval, order by `valid_at`, and resolve citations deterministically. Shows what was true when, what's current, and what was superseded/expired.

---

## 1. Scope

**In:** `answer_api.timeline_local` (temporal retrieval core); a shared `_retrieve_edges` helper refactored out of `search_local`; a `_fact_status` classifier; a `GET /timeline` FastAPI endpoint; optional vendor scoping. **Out:** an LLM narrative over the timeline (a later add reusing the synthesis layer); explicit time-window filtering; a `superseded_by` fact-to-fact linkage (graphiti gives no direct pointer); global/drift.

## 2. Grounding (verified against the live graph)

- **1936/1975 facts carry `valid_at`; 797 carry `invalid_at`** (65 from our staleness sweep = `expired_by_sweep=true`; 732 graphiti-invalidated).
- **Honest caveat on today's data:** the graphiti invalidations are largely **event-time supersession within document content** (e.g. a CloudTrail event `valid=2019-01-10T08:24 → invalid=2019-01-10T13:45`, hours apart) plus article `reference_time` — **not** documentation evolving over months. That kind of change accrues as the incremental pipeline runs over time. So `/timeline` **works today** on the event-time/supersession data and its "how vendor X evolved" value **grows with accumulated history** — the mechanism is the deliverable now.
- `EntityEdge` carries `valid_at`, `invalid_at`, `episodes`, `uuid`, `fact`; our custom `expired_by_sweep` on the fact edge distinguishes sweep-expiry from graphiti contradiction. `search_local` already retrieves via `EDGE_HYBRID_SEARCH_RRF` and resolves citations.

## 3. Architecture & flow

```
GET /timeline?q=<topic>&vendor=<optional>&limit=30
  1. _retrieve_edges(graphiti, q, fetch_limit=max(limit*3, limit), group_id)  RRF edges (shared w/ search_local)
  2. [optional] vendor-scope filter (structural, same as search_local)
  3. classify each edge: _fact_status(edge) -> current | superseded | expired
       current    = invalid_at IS NULL
       expired    = expired_by_sweep truthy               (staleness sweep: source docs gone)
       superseded = invalid_at set, not expired_by_sweep  (graphiti: a later/contradicting fact)
     (NO validity filter -- timeline inherently includes history)
  4. sort chronologically ASCENDING by valid_at (None valid_at sorts last); truncate to `limit`
  5. resolve citations (deterministic, design-decision #2)
  6. return { query, count, timeline: [ {fact, fact_uuid, valid_at, invalid_at, status, sources} ] }
```

No LLM at query time; citations are graph traversal.

## 4. Components

### 4.1 `_retrieve_edges` refactor (`answer_api/search.py`)
Extract the shared retrieval from `search_local`:
```python
async def _retrieve_edges(graphiti, q, *, fetch_limit, group_id) -> list:
    config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    config.limit = fetch_limit
    results = await graphiti._search(q, config, group_ids=[group_id])
    return list(results.edges)
```
`search_local` calls it with `fetch_limit=max(k*3, k)` then applies its validity/vendor filters + truncation (behaviour unchanged — covered by existing tests). `_vendor_episode_uuids` is already shared.

### 4.2 `_fact_status` (pure)
`answer_api/timeline.py`: `_fact_status(edge) -> str`:
```python
def _fact_status(edge) -> str:
    if getattr(edge, "invalid_at", None) is None:
        return "current"
    if getattr(edge, "expired_by_sweep", None):   # attr may be absent on EntityEdge -> read via ...
        return "expired"
    return "superseded"
```
Note: `expired_by_sweep` is a custom property we SET on the `RELATES_TO` edge; graphiti's `EntityEdge` model may not surface it as an attribute. **The retrieval must read it** — either include it when loading edges, or a small Cypher lookup of `expired_by_sweep` for the returned uuids (batch, one query) that `timeline_local` merges in. The implementer verifies whether `EntityEdge` exposes `expired_by_sweep`; if not, do the batch Cypher lookup (`MATCH ()-[f:RELATES_TO] WHERE f.uuid IN $uuids RETURN f.uuid, f.expired_by_sweep`). Pure classification given the flag.

### 4.3 `timeline_local` (`answer_api/timeline.py`)
`async def timeline_local(graphiti, driver, *, q, limit=30, vendor=None, group_id) -> dict`:
1. `edges = await _retrieve_edges(graphiti, q, fetch_limit=max(limit*3, limit), group_id=group_id)`.
2. If `vendor`: keep edges whose `episodes` intersect the vendor's structural episode set (`_vendor_episode_uuids`).
3. Determine `expired_by_sweep` per edge (attr or batch Cypher).
4. Sort edges ascending by `valid_at` (a `None` `valid_at` sorts last — key `(valid_at is None, valid_at)`), truncate to `limit`.
5. `resolve_citations` for the kept uuids.
6. Return `{"query": q, "count": len(edges), "timeline": [{"fact": e.fact, "fact_uuid": e.uuid, "valid_at": ..., "invalid_at": ..., "status": _fact_status(e), "sources": citations.get(e.uuid, [])} for e in edges]}`.

### 4.4 `GET /timeline` endpoint (`answer_api/app.py`)
Query params `q` (required → 422), `limit: int = 30`, `vendor: str | None = None`. Calls module-attr `timeline_mod.timeline_local(app.state.graphiti, app.state.driver, q=q, limit=limit, vendor=vendor, group_id=app.state.settings.group_id)` (injectable for tests, like `/search/local` and `/answer`). No new lifespan resources (no synth client needed — retrieval-first). `/search/local`, `/answer`, `/health` unchanged.

## 5. Design-decision alignment

- **#3 (temporal policy):** timeline consumes the append-not-overwrite result — it surfaces superseded/expired facts rather than hiding them; `status` uses the model's own vocabulary (current/superseded/expired). No deletion; read-only.
- **#2 (citations):** sources resolved by graph traversal (`resolve_citations`), never authored by an LLM (there is no LLM here).
- **#4 (vendor scoping):** through the structural layer, reusing `_vendor_episode_uuids`.

## 6. Testing

- **Unit — `_fact_status`** (pure, stubbed edges): invalid_at None → current; expired_by_sweep true → expired; invalid_at set + no sweep → superseded.
- **Unit — `timeline_local` ordering/status** with a **stub `_retrieve_edges`** returning canned edges (mixed valid_at incl. a None, mixed status) + a real Neo4j `extract_driver` for citations: assert ascending `valid_at` order (None last), correct `status` per fact, citations resolved, `limit` truncation. If `expired_by_sweep` needs the Cypher lookup, seed those facts.
- **Refactor safety:** the existing `search_local` tests stay green (behaviour unchanged) after `_retrieve_edges` extraction.
- **FastAPI `/timeline` TestClient** (stub `timeline_local`): 200 with `{query,count,timeline}`; missing `q` → 422; `/search/local`+`/answer`+`/health` still 200.
- **`@live` smoke:** `timeline_local` against the real graph for a query with known invalidated facts returns them in `valid_at` order with correct `current`/`superseded`/`expired` status and resolved sources.
- **Controller demonstration + short report:** run `/timeline` on a topic with a real supersession chain (e.g. an AWS CloudTrail event sequence), show the chronological facts with intervals + status, and note honestly that doc-evolution value scales with accumulated incremental history.
- Full non-live suite + ruff/mypy clean.

## 7. Acceptance criteria

1. `GET /timeline?q=...` returns facts ordered ascending by `valid_at`, each with `status` (current/superseded/expired), `valid_at`/`invalid_at`, and deterministically-resolved `sources` — no LLM at query time.
2. `_fact_status` correctly distinguishes current / graphiti-superseded / sweep-expired (via the `expired_by_sweep` flag, read correctly whether or not `EntityEdge` surfaces it).
3. Invalidated facts ARE included (timeline shows history); `None` `valid_at` sorts last; `limit` respected; optional `vendor` scopes via the structural layer.
4. `_retrieve_edges` is shared with `search_local` with no behaviour change (existing tests green).
5. Unit + FastAPI tests green; `@live` smoke passes; a demonstration report is written; ruff/mypy clean.

## 8. Deferred

- **LLM narrative timeline** (a synthesized "here's how X evolved" over the ordered facts, reusing the GLM synthesis layer + design-decision #2 markers) — the natural next add.
- **Time-window filtering** (`?since=&until=`), `superseded_by` fact linkage, and richer grouping (by entity/period).
- Verbose-query rewrite (the other deferred retrieval item); global/drift/router (Phase 3/4).
