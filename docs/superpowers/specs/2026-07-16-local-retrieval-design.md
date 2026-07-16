# Local Retrieval + Citation Precision — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Phase 2 (retrieval core) — first sub-slice: `/search/local` + citation resolver + golden-set harness.
**Date:** 2026-07-16
**Status:** Approved design — ready for implementation planning

The first retrieval slice: turn the semantic graph into **cited, ranked facts** for a query. Retrieval-first (no synthesis LLM) and fully deterministic at query time. Meets the Phase-2 exit criterion: "local questions answered with correct URLs at target citation precision." Timeline, global/DRIFT/router, and MCP exposure are later slices.

---

## 1. Scope

**In:** a new `answer_api` FastAPI package with `GET /search/local`; a batch citation resolver; optional vendor scoping via the structural layer; a validity filter; a golden-set harness measuring citation precision. **Out:** synthesis LLM (retrieval-first — the endpoint returns ranked cited facts, not prose), `/timeline`, `/search/global`, `/search/drift`, the `/answer` router, MCP.

## 2. Grounding (verified against the code + live graph)

- **Graphiti hybrid search** returns fact edges: `graphiti._search(query, config, group_ids=[...])` → `SearchResults.edges: list[EntityEdge]`. Each `EntityEdge` carries `uuid`, `fact`, **`episodes` (list of supporting Episodic uuids)**, `valid_at`, `invalid_at`, `group_id`.
- **Recipe:** `EDGE_HYBRID_SEARCH_RRF` (from `graphiti_core.search.search_config_recipes`) — full-text + semantic over edges fused with **RRF reranking, no cross-encoder** → backend-agnostic (avoids the tiktoken-True/False reranker that may be random on non-OpenAI backends; see slice-2 followups).
- **Citation chain (design-decision #2):** `provenance.resolve_chain(fact_uuid)` already does `fact → episodes → (:Article)-[:HAS_EPISODE]-> → source_url`, returning `{url, title, article_id, fact}`. Query-time never runs an LLM; citations are pure graph traversal.
- **Structural path (design-decision #4):** `(:Vendor)-[:HAS_PRODUCT]->(:Product)-[:HAS_SOURCE]->(:Source)-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(:Episodic)` — the basis for vendor-scoped queries (one corpus-wide group; scope through structure).
- **One embedding space:** the query is embedded with the same TEI/Jina model as extraction (already guaranteed), so query↔fact similarity is valid.

## 3. Architecture

```
GET /search/local?q=<query>&k=10&vendor=<optional>&include_invalid=false
  1. graphiti._search(q, EDGE_HYBRID_SEARCH_RRF, group_ids=["backup-docs"], limit=k*over)  → ranked EntityEdges
  2. validity filter: drop edges with invalid_at set unless include_invalid=true   (current-state by default)
  3. [optional] vendor scope: keep edges whose episodes intersect the vendor's structural episodes
  4. take top-k; batch-resolve citations for their uuids (one Cypher query)
  5. return: { query, count, results: [ { fact, fact_uuid, valid_at, sources: [ {url,title,article_id} ] } ] }
```

No LLM at query time. The response is ranked cited facts; a synthesis layer can consume this later.

## 4. Components

### 4.1 `answer_api` package (FastAPI)
- `answer_api/app.py` — `create_app()` + a lifespan that builds one `Graphiti` (via `graph_extract.graphiti_client.build_graphiti`) and one Neo4j driver, closed on shutdown; `GET /health`.
- `answer_api/search.py` — `async def search_local(graphiti, driver, q, k, vendor, include_invalid) -> dict` (the core, testable independent of FastAPI).
- `answer_api/config.py` — reuse `graph_extract.config.ExtractSettings` (Neo4j + embedder + group_id) — no new config surface.
- `answer_api/cli.py` or uvicorn entry — run the service.

### 4.2 Retrieval
`search_local` calls `graphiti._search(q, EDGE_HYBRID_SEARCH_RRF, group_ids=[settings.group_id], ...)` with a limit somewhat above `k` (to survive the validity/vendor post-filters), then post-filters and truncates to `k`. If `_search`'s exact signature differs, adapt — the requirement is the RRF edge recipe (no cross-encoder) and getting `EntityEdge`s back. **Validate** during implementation that the RRF path runs cleanly on the Azure/TEI backend and returns sensible ordering (a query with an obvious answer ranks the right fact near the top).

### 4.3 Citation resolver (batch)
Add `provenance.resolve_citations(fact_uuids: list[str]) -> dict[str, list[dict]]` — one query resolving all uuids at once (no N+1):
```cypher
MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $uuids
UNWIND f.episodes AS epu
MATCH (a:Article)-[:HAS_EPISODE]->(:Episodic {uuid: epu})
WITH f.uuid AS uuid, collect(DISTINCT {url:a.source_url, title:a.title, article_id:a.id}) AS sources
RETURN uuid, sources
```
Reuse the existing per-fact `resolve_chain` where a single lookup is enough. A fact whose episodes have no resolvable Article yields `sources: []` (surfaced, not dropped — a dangling-citation signal).

### 4.4 Vendor scoping (optional, structural)
When `vendor` is given, resolve the vendor's supporting episode uuids via the structural path, and keep only fact edges whose `episodes` intersect that set:
```cypher
MATCH (v:Vendor)-[:HAS_PRODUCT]->(:Product)-[:HAS_SOURCE]->(:Source)
      -[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(e:Episodic)
WHERE toLower(v.name) = toLower($vendor)
RETURN collect(DISTINCT e.uuid) AS episode_uuids
```
Default (no `vendor`) is cross-vendor — unscoped. (Matching a vendor name uses the structural `:Vendor` node names, e.g. `AWS`, `Microsoft`.)

### 4.5 Validity filter
Default `include_invalid=false` → drop edges with `invalid_at` set (current-state answers). `include_invalid=true` returns historical facts too (a stepping-stone toward the deferred `/timeline`). `valid_at` is returned per result.

## 5. Golden-set harness

- `answer_api/golden.py` + `answer_api/golden_questions.json` — ~12–15 questions grounded in the pilot articles, each: `{ question, expected_article_ids: [...], vendor?: str }` (article-id, not URL string, is the stable key; the harness maps to URLs via the citations).
- `answer_api/eval_golden.py` (script): run each question through `search_local`, and compute **citation precision@k** = fraction of questions for which an expected article-id appears among the returned citations' `article_id`s (a hit), plus the rank of the first hit (MRR-style). Emit `docs/superpowers/local-retrieval-golden-report.md` with per-question hit/miss + the aggregate.
- The golden questions are authored from known pilot content (e.g. "What mode does AWS Backup Vault Lock require?" → the Vault Lock article; "How long does Azure Backup soft delete retain items?" → the soft-delete article). Grounded by inspecting the current graph's articles.

## 6. Testing

- **Unit (Neo4j testcontainer, LLM-free):** `resolve_citations` batch (seeded facts/episodes/articles → correct grouped sources; a fact with an unresolvable episode → `[]`); the vendor-scope episode query (seeded structural chain); `search_local`'s post-filters (validity + vendor + truncate-to-k) with a **stubbed `graphiti` search** returning canned `EntityEdge`-like objects, so the deterministic logic is tested without embeddings.
- **Live smoke (`@live`):** `search_local` against the real compose graph for a known query returns the expected fact near the top with a resolvable citation.
- **FastAPI:** a `TestClient` test of `/search/local` (stubbed search) asserting the response shape + `/health`.
- The golden harness is a report-producing script (needs the live graph + embeddings), not CI.
- Full non-live suite + ruff/mypy clean.

## 7. Acceptance criteria

1. `GET /search/local?q=...` returns ranked facts each with resolved `sources` (url/title/article_id) via pure graph traversal — no LLM at query time.
2. Retrieval uses the **RRF edge recipe (no cross-encoder)** and is validated to run + rank sensibly on the Azure/TEI backend.
3. Invalidated facts are excluded by default; `include_invalid=true` includes them; `vendor=` scopes via the structural layer.
4. The golden harness reports **citation precision@k** over ~12–15 questions with a per-question breakdown; the headline number is recorded (the Phase-2 exit signal).
5. Unit + FastAPI tests green (testcontainer); the live smoke passes against the real graph; ruff/mypy clean.

## 8. Deferred

- **Synthesis layer** (LLM writes prose citing fact-IDs → resolver expands to URLs) — a later slice; this slice returns the ranked cited facts it would consume.
- `/timeline` (bi-temporal), `/search/global` (community map-reduce — needs Phase 3), `/search/drift`, the `/answer` router — later Phase 2/3/4 slices.
- MCP server wrapper + Copilot exposure (Phase 5).
- Cross-encoder reranking (only if RRF proves insufficient on the golden set — then a local BGE cross-encoder / embedding reranker, per the slice-2 followup).
