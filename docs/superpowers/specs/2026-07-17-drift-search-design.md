# DRIFT Search `/search/drift` — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Phase 4, slice 1 — `/search/drift`: the broad→deep DRIFT motion
(primer → follow-up loop → synthesis) for questions that are broad but expect
concrete, sourced detail. Fact-ID → URL citations.
**Date:** 2026-07-17
**Status:** Approved design — ready for implementation planning

DRIFT "enters through the theme, drills into the facts": it primes on the
`:Community` report layer (slice built in Phase 3) to draft a preliminary answer
and targeted follow-up queries, executes each follow-up as a graph-biased local
fact search, then synthesizes one cited answer. The `/answer` router that unifies
this with `/search/local`, `/search/global`, and `/timeline` is the **next**
slice — out of scope here.

---

## 1. Scope

**In:** `answer_api/drift.py` (primer + follow-up loop + synthesis + citation
resolve); a `GET /search/drift` endpoint; an optional `center_node_uuid` on
`search_local` (node-distance reranking); config for primer level/size, follow-up
budget/k, and iteration count. Reuses `global_search.shortlist_communities` /
`CommunityHit` / `_extract_json`, `synthesize._finalize_answer` /
`_synthesis_client_and_model`, `answer_local` (degrade path),
`Provenance.resolve_citations`, and the shared embedder (already on `app.state`).

**Out (later slices):** the `/answer` router + uniform `mode` response contract
(Phase 4 slice 2); iterations > 2 / adaptive depth; cross-encoder reranking;
caching primer shortlists; MCP exposure (Phase 5).

## 2. Grounding (verified)

- graphiti-core 0.29.2 ships `EDGE_HYBRID_SEARCH_NODE_DISTANCE`
  (`edge_config.reranker == EdgeReranker.node_distance`), and `Graphiti._search`
  accepts `center_node_uuid` — so center-node (graph-distance) reranking of edges
  is available. Today's `search_local` uses `EDGE_HYBRID_SEARCH_RRF`.
- `:Community` layer populated on `backup-docs`: ~66 reports across levels 0/1/2,
  each with `embedding` (shared space), `rating`, `title`, `summary`,
  `full_report`, `cited_fact_uuids`; membership via `(:Community)<-[:IN_COMMUNITY]-(:Entity)`.
  Level 1 has ~19 communities.
- Reused, confirmed present: `global_search.shortlist_communities(driver, embedder,
  q, *, level, k, group_id) -> list[CommunityHit]`, `global_search.CommunityHit`
  (`community_id, title, summary, level, rating, cited_fact_uuids, full_report,
  similarity`), `global_search._extract_json(raw) -> dict | None`;
  `synthesize._finalize_answer(raw, marker_map) -> (text, cited: list[int])`,
  `synthesize._synthesis_client_and_model(settings) -> (AsyncOpenAI, str)` (GLM via
  judge_*), `synthesize.answer_local(graphiti, driver, synth_client, synth_model,
  *, q, k, vendor, group_id) -> dict`; `search.search_local(graphiti, driver, *, q,
  k, vendor, include_invalid, group_id) -> {query,count,results:[{fact,fact_uuid,
  valid_at,sources}]}`; `Provenance.resolve_citations(fact_uuids) -> {uuid:
  [{url,title,article_id}]}`; app lifespan `app.state.{settings,graphiti,driver,
  synth_client,synth_model,embedder,map_client,map_model}`.

## 3. Architecture & flow

```
GET /search/drift?q=&level=1&iterations=1
  1. PRIMER  (1 strong LLM call, synthesis tier / GLM)
     hits = shortlist_communities(q, level, k=drift_primer_k)          [reuse]
     if not hits: return answer_local(q) + {degraded:"no-primer-communities"}
     strong model, given q + each hit's {community_id,title,summary}, returns JSON:
        { "preliminary_answer": str,
          "follow_ups": [ {"query": str, "community_id": str|null, "relevance": 0-10} ] }
     validate community_id ∈ hit ids (else null); keep top drift_max_followups by relevance
     (bad/empty JSON, one retry -> fall back to a single follow-up = q, community_id=null)

  2. FOLLOW-UP LOOP  (no LLM in retrieval; rounds = clamp(iterations,1,2))
     round 1: for each follow-up f:
         center = _top_member_entity(f.community_id) if f.community_id else None
         r = search_local(q=f.query, k=drift_followup_k, center_node_uuid=center)
         accumulate r["results"] into facts, dedup by fact_uuid
     if rounds == 2 and facts non-empty:
         1 strong LLM call: given q + accumulated facts, draft refined follow-ups
         (same JSON shape, minus preliminary_answer); execute round 2 the same way

  3. SYNTHESIS  (1 strong LLM call, synthesis tier / GLM)
     if not facts: return fixed refusal (NO synthesis call)
     marker_map = {i: fact_i} over the deduped fact union, i from 1
     strong model merges primer preliminary_answer + numbered facts; cites [N] only,
     never a URL; refuses if nothing relevant                            [decision #2]
     answer, cited = _finalize_answer(raw, marker_map)                   [decision #2]
     srcs = Provenance.resolve_citations([marker_map[m].fact_uuid for m in cited])

  return { query, answer, citations:[{marker,fact_uuid,sources}],
           follow_ups:[{query,community_id,iteration}],
           communities_used:[{community_id,title}] }
```

**LLM cost:** 2 strong calls (iterations=1) or 3 (iterations=2), all on the
synthesis tier — no new model tier. Follow-up retrieval has no LLM. Latency
target (plan §11): 15–45s.

## 4. Components (`answer_api/drift.py`)

### 4.1 Primer — `_primer`
`async def _primer(embedder, synth_client, synth_model, driver, *, q, level, k,
max_followups, group_id) -> tuple[str, list[FollowUp], list[CommunityHit]]`
- `hits = await shortlist_communities(driver, embedder, q, level=level, k=k, group_id=group_id)`.
  If empty → signal the degrade path (return sentinel; caller runs `answer_local`).
- Prompt the synthesis LLM with `q` + a numbered list of `hits`
  (`community_id`, `title`, `summary`) for JSON `{preliminary_answer, follow_ups:
  [{query, community_id, relevance}]}`. Robust parse (`_extract_json`, one retry).
- Validate each `community_id ∈ {h.community_id}` (else `None`); sort follow-ups by
  `relevance` desc; keep top `max_followups`. On unparseable JSON after retry →
  `preliminary_answer=""`, `follow_ups=[FollowUp(query=q, community_id=None)]`.
- `FollowUp` dataclass: `query: str`, `community_id: str | None`, `iteration: int`.

### 4.2 Center-node resolution — `_top_member_entity`
`async def _top_member_entity(driver, group_id, community_id) -> str | None`
```cypher
MATCH (c:Community {group_id:$g, community_id:$cid})<-[:IN_COMMUNITY]-(e:Entity)
OPTIONAL MATCH (e)-[r:RELATES_TO {group_id:$g}]-()
WITH e, count(r) AS deg ORDER BY deg DESC LIMIT 1
RETURN e.uuid AS uuid
```
Returns the top-degree member's `uuid`, or `None` (unknown/blank community, or a
member-less community). Best-effort — a `None` center just means a plain local search.

### 4.3 Follow-up execution — `_run_followup`
`async def _run_followup(graphiti, driver, fu: FollowUp, *, k, group_id) -> list[dict]`
- `center = await _top_member_entity(driver, group_id, fu.community_id) if fu.community_id else None`.
- `res = await search_local(graphiti, driver, q=fu.query, k=k, center_node_uuid=center, group_id=group_id)`.
- Return `res["results"]` (`[{fact, fact_uuid, valid_at, sources}]`).

### 4.4 Refinement (iteration 2) — `_refine_followups`
`async def _refine_followups(synth_client, synth_model, *, q, facts, max_followups, hit_ids) -> list[FollowUp]`
- One strong LLM call: given `q` + the accumulated numbered facts, return JSON
  `{follow_ups:[{query, community_id, relevance}]}` (community_id ∈ `hit_ids` or
  null). Same validation/budget as the primer. Only invoked when `rounds == 2` and
  `facts` is non-empty. Follow-ups tagged `iteration=2`.

### 4.5 Synthesis + resolve — `_synthesize`
`async def _synthesize(synth_client, synth_model, driver, *, q, preliminary_answer, facts) -> tuple[str, list[dict]]`
- Dedup `facts` by `fact_uuid` preserving first-seen order; `marker_map = {i: fact}`
  from 1. Render numbered facts; reduce prompt merges `preliminary_answer` (as a
  draft to refine, not a source of truth) + facts; cite `[N]` only, never a URL,
  refuse if nothing relevant. Generous `max_tokens` (GLM reasoning headroom).
- `answer, cited = _finalize_answer(raw, marker_map)`; resolve
  `[marker_map[m]["fact_uuid"] for m in cited]` → `citations`.

### 4.6 Orchestrator — `drift_search`
`async def drift_search(graphiti, driver, embedder, synth_client, synth_model, *,
q, level, iterations, group_id) -> dict`
Ties 4.1–4.5 together per §3; owns the degrade/refusal branches (§7) and the
`follow_ups` / `communities_used` metadata. `rounds = max(1, min(iterations, 2))`.

### 4.7 `search_local` extension (`answer_api/search.py`)
Add `center_node_uuid: str | None = None` to `search_local` and `_retrieve_edges`.
In `_retrieve_edges`, when `center_node_uuid` is set, deep-copy
`EDGE_HYBRID_SEARCH_NODE_DISTANCE` (else today's `EDGE_HYBRID_SEARCH_RRF`), set
`config.limit = fetch_limit`, and pass `center_node_uuid` to `graphiti._search`.
Backward-compatible: existing callers omit it and hit the unchanged RRF path.

### 4.8 Endpoint — `GET /search/drift` (`answer_api/app.py`)
`q: str` (required → 422), `level: int | None = Query(None, ge=0)`,
`iterations: int | None = Query(None, ge=1, le=2)`. Handler falls back to
`settings.drift_primer_level` / `settings.drift_iterations` when a param is None.
Calls module-attr `drift_mod.drift_search(...)` (injectable seam, like the others),
using `app.state.{graphiti,driver,embedder,synth_client,synth_model,settings}`.
`/health`, `/search/local`, `/answer`, `/timeline`, `/search/global` unchanged.

## 5. Config additions (`ExtractSettings`)

```
drift_primer_level: int = 1      # community level the primer shortlists at
drift_primer_k: int = 5          # reports shortlisted for the primer
drift_max_followups: int = 4     # follow-ups kept per round (relevance-budgeted; plan 3-6)
drift_followup_k: int = 8        # local-search k per follow-up
drift_iterations: int = 1        # follow-up rounds; clamped to [1,2]
```

## 6. Design-decision alignment

- **#2 citations are traversal, never LLM:** the only cited output is the synthesis
  answer, which emits `[N]` markers validated against `marker_map`; `_URL_RE`
  strips any URL; `Provenance` expands fact UUIDs → source URLs. The primer's
  `preliminary_answer` and follow-up *queries* never carry citations and never
  reach the user un-synthesized. Chain: community (primer) → follow-up local search
  → fact_uuid → episode → article → source_url. No LLM authors a URL.
- **#4 one embedding space:** the primer embeds `q` with the SAME shared embedder as
  `:Community.embedding`; follow-up local search reuses the same Graphiti hybrid
  retrieval over the one group.
- Reuses the `/answer` + `/search/global` citation machinery — DRIFT answers are
  cited the same verifiable way.

## 7. Error handling & degradation

| Condition | Behavior |
|---|---|
| Empty primer shortlist | Degrade to `answer_local(q)`; response includes `"degraded":"no-primer-communities"`. Always a live-fact answer. |
| Primer JSON unparseable (after 1 retry) | Fall back to one follow-up = `q`, `community_id=None`; proceed. |
| Follow-up `community_id` blank/unknown, or member-less community | Plain local search (no center node). Center-node biasing is best-effort. |
| A follow-up's `search_local` raises | Logged, contributes no facts; loop continues (never aborts the batch). |
| Zero facts across all rounds | Fixed refusal (`_REFUSAL`), NO synthesis LLM call. |
| `iterations` out of range | Clamped to `[1,2]`; refinement skipped when rounds=1. |

## 8. Testing

- **Unit (`drift.py`, fake LLM/embedder clients):** primer JSON parse + relevance
  budget (keep top `drift_max_followups`, drop invalid `community_id`);
  primer-bad-JSON → single `q` follow-up fallback; marker numbering over the
  deduped fact union; synthesis citation resolution + URL strip + invalid-marker
  drop; `rounds` clamp; iteration-2 refinement only when rounds=2 and facts exist.
- **Integration (Neo4j testcontainer, module-scoped — `MATCH (n) DETACH DELETE n`
  per test):** seed `:Community` + `IN_COMMUNITY` members + facts/episodes/articles;
  full `drift_search` returns a cited answer whose `citations[].sources` resolve to
  real URLs; `_top_member_entity` returns the top-degree member; empty-shortlist →
  `answer_local` degrade; zero-facts → refusal.
- **`search.py` regression (integration):** `search_local(center_node_uuid=<seeded
  entity>)` runs the node-distance recipe and still resolves citations; existing
  RRF callers unaffected (no signature break).
- **App (`test_answer_api_app.py`):** `GET /search/drift` returns the
  `{query,answer,citations,follow_ups,communities_used}` shape; missing `q` → 422;
  `iterations=3` → 422 (Query `le=2`); pre-existing endpoints unchanged.
- **`@live` smoke + demonstration:** a broad question on `backup-docs`
  ("What should I consider when planning long-term backup retention across cloud
  vendors?"); show the primer follow-ups, center-node biasing, and cross-vendor
  cited synthesis → `docs/superpowers/drift-report.md`.
- Full non-live suite + `ruff check src tests` + mypy clean.

## 9. Acceptance criteria

1. `GET /search/drift?q=…` returns a synthesized answer with `citations`
   (fact_uuid → resolvable source URLs), the executed `follow_ups`, and
   `communities_used`; no URL authored by an LLM; every cited fact_uuid is a real
   retrieved fact.
2. The primer shortlists community reports at `level`, drafts a preliminary answer
   + relevance-budgeted follow-ups (≤ `drift_max_followups`), each optionally
   tagged with a shortlisted `community_id`.
3. A follow-up tagged with a community biases its local search by that community's
   top-degree member entity as the Graphiti center node (node-distance recipe);
   untagged/unknown follow-ups run plain local search.
4. `iterations` ∈ {1,2}; iteration 2 drafts refined follow-ups from round-1 facts.
5. Degradation paths hold: empty shortlist → `answer_local`; zero facts → refusal
   with no synthesis spend; per-follow-up failures never abort the request.
6. Unit + integration + app tests green; `@live` smoke passes; a demonstration
   report is written; full non-live suite + ruff/mypy clean.

## 10. Deferred

- `/answer` router + uniform `mode` contract (Phase 4 slice 2); iterations > 2 /
  adaptive depth by query breadth; cross-encoder reranking for follow-ups; caching
  primer shortlists across requests; per-follow-up (vs merged) evidence attribution.
