# Global (Map-Reduce) Search — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Phase 3, slice 2 — `/search/global`: dynamic-selection map-reduce over
the community-report layer for cross-vendor thematic questions ("how do all these
vendors treat backup of workload X?"), with fact-ID → URL citations.
**Date:** 2026-07-17
**Status:** Approved design — ready for implementation planning

Consumes the `:Community` layer built in slice 1. MS-GraphRAG-style: shortlist
reports by embedding + rating, MAP each to query-relevant key points (carrying
fact IDs), REDUCE across the map outputs into one cited answer. The full
map-reduce is built now (not a single-shot synthesis) so the scale-out path is
already in place.

---

## 1. Scope

**In:** `answer_api/global_search.py` (shortlist + map + reduce + citation
resolve); a `GET /search/global` endpoint; config for the map tier, shortlist
size, default level, relevance floor. Reuses `synthesize._finalize_answer`/
`_URL_RE`/`_synthesis_client_and_model`, `Provenance.resolve_citations`, and the
shared embedder (`graphiti_client.build_embedder`).

**Out (later slices):** hierarchical descent into high-scoring communities'
children; auto level-selection by query breadth; wiring `/search/global` into the
`/answer` router (Phase 4); DRIFT; a vector index on `:Community.embedding`
(brute-force cosine is fine at this scale).

## 2. Grounding (verified)

- The `:Community` layer is populated on `backup-docs`: ~66 communities, levels
  0/1/2, each with `embedding` (shared TEI/Jina space, `embed_dim`), `rating`
  (0–10), `title`, `summary`, `full_report` (JSON findings), `cited_fact_uuids`,
  `member_count`. `PARENT_OF`/`IN_COMMUNITY` edges. Level 1 has ~19 communities.
- Reused, confirmed present: `synthesize._finalize_answer(raw, marker_map) ->
  (text, cited: list[int])` (strips URLs, keeps ordered-unique `[N]` markers in
  `marker_map`), `synthesize._URL_RE`/`_MARKER_RE`,
  `synthesize._synthesis_client_and_model(settings) -> (AsyncOpenAI, str)` (GLM
  via judge_*), `Provenance.resolve_citations(fact_uuids: list[str]) ->
  dict[str, list[dict]]`, `graphiti_client.build_embedder(s) -> OpenAIEmbedder`
  (`.create_batch(texts)`), the `answer_api` FastAPI lifespan
  (`app.state.{settings,graphiti,driver,synth_client,synth_model}`).

## 3. Architecture & flow

```
GET /search/global?q=&level=1&k=10
  1. embed q with the shared embedder (same space as :Community.embedding)     [decision #4]
  2. shortlist: cosine(q, c.embedding) over :Community {group_id, level}; rank by
     similarity (+ small rating boost); take top-k -> [CommunityHit]
  3. MAP (parallel, asyncio.gather): per hit, map-tier LLM -> JSON
       {relevance:0-10, key_points:[str], fact_ids:[str] ⊆ hit.cited_fact_uuids}
     validate fact_ids (drop hallucinated); drop hits with relevance < floor or bad JSON
  4. REDUCE: marker_map = number the UNION of surviving fact_ids [1..N];
     render per-community key_points + their [N] markers; synthesis-tier LLM writes
     a themed/cross-vendor answer citing [N] only (never a URL)                 [decision #2]
  5. resolve: _finalize_answer(raw, marker_map) -> (answer, cited[]);
     Provenance.resolve_citations(all fact_uuids) -> sources; build citations[]
  return { query, answer, citations:[{marker,fact_uuid,sources}],
           communities_used:[{community_id,title,relevance}] }
```

Zero-shortlist or zero-surviving-map → fixed refusal (no LLM). The only LLM calls
are the map fan-out (map tier) and one reduce (synthesis tier).

## 4. Components (`answer_api/global_search.py`)

### 4.1 Shortlist — `shortlist_communities`
`async def shortlist_communities(driver, embedder, q, *, level, k, group_id) -> list[CommunityHit]`
- `CommunityHit{community_id, title, summary, level, rating, cited_fact_uuids, similarity}`.
- Embed `q` via `embedder.create_batch([q])[0]`. Fetch `:Community {group_id,
  level}` rows (community_id, title, summary, rating, cited_fact_uuids, embedding).
  Compute cosine similarity in Python (brute-force; ~19 rows at level 1). Rank by
  `similarity + rating_boost*rating/10` (small boost, e.g. 0.1) desc; take top-k.
  Skip communities with no `cited_fact_uuids` (nothing to cite).

### 4.2 Map — `_map_client_and_model` + `map_report`
- `_map_client_and_model(settings) -> (AsyncOpenAI, str)`: resolves the map tier
  from `map_llm_*`, falling back to the synthesis/judge tier (GLM-5.2).
- `async def map_report(client, model, q, hit: CommunityHit) -> MapResult | None`:
  prompt the map LLM with `q` + the hit's `title/summary/full_report`; ask for JSON
  `{"relevance": 0-10, "key_points": [str], "fact_ids": [uuid,...]}` where fact_ids
  MUST come from the report's cited facts. Robust JSON parse (fenced/chatty, one
  retry — reuse the `_extract_json` pattern from `theme_builder.report`). Keep only
  `fact_ids ⊆ hit.cited_fact_uuids`. Return `None` if unparseable or
  `relevance < global_map_relevance_min`. `MapResult{community_id, title,
  relevance, key_points, fact_ids}`.

### 4.3 Reduce + resolve — `global_search`
`async def global_search(driver, embedder, map_client, map_model, synth_client, synth_model, *, q, level, k, group_id) -> dict`
1. `hits = shortlist_communities(...)`; if empty → refusal dict.
2. `maps = [m for m in await asyncio.gather(*[map_report(...) for h in hits]) if m]`
   (a map call that raises is caught → contributes None). If empty → refusal.
3. `marker_map`: enumerate the ordered-unique union of `fact_ids` across `maps`,
   `{i: {"fact_uuid": uuid}}` starting at 1.
4. Render reduce context: per map result, `COMMUNITY "{title}" (relevance {r}):`
   then its `key_points`, then `Supporting facts: [i] [j] …` (the markers for its
   fact_ids). Reduce prompt: synthesize a cross-vendor answer to `q` using ONLY
   these findings, cite the `[N]` markers, never write a URL, refuse if nothing is
   relevant. `max_tokens` generous (GLM reasoning headroom — the slice-1 lesson).
5. `answer, cited = _finalize_answer(raw, marker_map)`;
   `srcs = await Provenance(driver).resolve_citations([marker_map[m]["fact_uuid"] for m in cited])`;
   `citations = [{marker:m, fact_uuid:u, sources:srcs.get(u,[])} for m,u …]`.
6. Return `{query:q, answer, citations, communities_used:[{community_id,title,
   relevance} for m in maps]}`.

### 4.4 Endpoint — `GET /search/global`
`answer_api/app.py`: `q: str` (required → 422), `level: int | None = Query(None, ge=0)`,
`k: int | None = Query(None, ge=1)` — the handler falls back to
`settings.global_default_level` / `settings.global_shortlist_k` when a param is
omitted (Query defaults can't read settings, so the fallback lives in the handler).
Calls module-attr `global_mod.global_search(...)`
(injectable for tests, like `/answer`/`/timeline`). The lifespan additionally
builds the shared embedder and the map client/model, stored on `app.state`
(closed on shutdown alongside `synth_client`). `/health`, `/search/local`,
`/answer`, `/timeline` unchanged.

## 5. Config additions (`ExtractSettings`)

```
map_llm_base_url: str = ""      # map tier; defaults to judge/synthesis when empty
map_llm_model: str = ""
map_llm_api_key: str = ""
global_shortlist_k: int = 10
global_default_level: int = 1
global_map_relevance_min: int = 2
```

## 6. Design-decision alignment

- **#2 citations are traversal, never LLM:** both map and reduce emit only fact
  UUIDs / `[N]` markers; map fact_ids are validated ⊆ the report's real
  `cited_fact_uuids`; reduce markers are validated against `marker_map`; `_URL_RE`
  strips any URL the reduce LLM writes; `Provenance` expands markers → source URLs.
  The chain is community → map fact_id (∈ report cited facts) → episode → article →
  source_url. No LLM authors a URL.
- **#4 one embedding space:** the query is embedded with the SAME shared embedder
  as `:Community.embedding` (and entities/facts), so cosine is meaningful.
- Reuses the `/answer` citation machinery (`_finalize_answer`, `Provenance`) —
  global answers are cited the same, verifiable way as local ones.

## 7. Error handling

- A map call that errors or returns unparseable JSON → that community contributes
  nothing (logged, skipped); the fan-out `gather` uses `return_exceptions` so one
  failure never aborts the batch.
- Zero shortlisted communities, or zero surviving map results → a fixed refusal
  (`"I don't have enough thematic coverage to answer that."`), no LLM spend.
- Hallucinated fact_ids dropped at map; invalid `[N]` markers dropped at reduce.
- Reduce truncation avoided with adequate `max_tokens` (GLM reasoning headroom).

## 8. Testing

- **Unit — `shortlist_communities`** (stub embedder + seeded rows or a fake fetch):
  ranks by cosine (+rating boost), respects `level` filter and `k`, skips
  cite-less communities.
- **Unit — `map_report`:** valid JSON → MapResult with fact_ids ⊆ report's;
  hallucinated fact_ids dropped; relevance < floor → None; bad JSON → one retry → None.
- **Unit — reduce citation:** given canned map results, `global_search` (with fake
  map/reduce clients) numbers facts, the reduce answer's `[N]` markers resolve to
  the right `fact_uuid`s, a model-written URL is stripped, an invalid marker dropped.
- **Integration (Neo4j testcontainer, fake LLM clients):** seed a small
  `:Community` layer + entities/facts + provenance; `global_search` returns a
  cross-community answer whose `citations[].sources` resolve to real URLs; empty
  shortlist → refusal.
- **`@live` smoke + controller demonstration:** `/search/global` on `backup-docs`
  for "compare how AWS Backup and Azure Backup handle <workload>"; show the cited,
  cross-vendor answer with resolving URLs; write `docs/superpowers/global-search-report.md`.
- Full non-live suite + ruff/mypy clean.

## 9. Acceptance criteria

1. `GET /search/global?q=…` returns a synthesized cross-vendor answer with
   `citations` (fact_uuid → resolvable source URLs) and `communities_used`; no URL
   authored by an LLM; every cited fact_uuid is a real fact of a shortlisted
   community.
2. Shortlist ranks communities by query-embedding cosine (+ rating) at the chosen
   level; `level`/`k` are params; the map fan-out runs in parallel and tolerates
   per-report failures.
3. Map tier defaults to GLM-5.2 and is config-overridable; reduce uses the
   synthesis tier; both avoid truncation with adequate token headroom.
4. Zero-coverage → a fixed refusal with no LLM spend.
5. Unit + integration tests green; `@live` smoke passes; a demonstration report is
   written; ruff/mypy clean.

## 10. Deferred

- Hierarchical descent (map high-scoring communities' `PARENT_OF` children);
  auto level selection by query breadth; a `:Community.embedding` vector index for
  scale; `/answer` router integration (Phase 4); DRIFT; per-key-point (vs
  per-community) fact-marker granularity.
