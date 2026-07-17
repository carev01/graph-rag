# Theme-Builder (Community Layer, Slice 1) — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Phase 3, slice 1 — the **theme-builder** batch service (Layer C): GDS
hierarchical Leiden community detection over the `backup-docs` entity graph +
LLM-written MS-GraphRAG-style community reports with per-finding fact-ID
citations + write-back of a `:Community` subgraph to Neo4j.
**Date:** 2026-07-17
**Status:** Approved design — ready for implementation planning

The derived, disposable thematic layer that later powers global (map-reduce)
search and DRIFT entry. This slice builds the layer as a **full rebuild**;
incremental refresh and global search are separate later slices.

---

## 1. Scope

**In:** a `theme_builder` package + a `theme-build` CLI command that (1) detects
hierarchical communities via GDS Leiden, (2) assembles per-community report
context under a token budget, (3) generates a fact-cited report per community
with the strong/synthesis LLM, (4) writes back a `:Community` subgraph. Reuses
the Neo4j driver, the single shared embedder (TEI/Jina), and the synthesis LLM
config. Full rebuild (delete + rewrite the group's `:Community` subgraph).

**Out (later slices):** global map-reduce `/search/global`; incremental refresh
(dirty-marking, Jaccard-stable community IDs — this slice uses full rebuild +
deterministic content-hash IDs); DRIFT; structural vendor attribution beyond
model-inferred `tags`; Graphiti's own `build_communities`.

## 2. Grounding (verified against the live graph)

- **GDS 2.13.11** available; `gds.graph.project`, `gds.leiden.stream/write`
  present. Entity graph: **555 `:Entity` / 1975 `RELATES_TO`** in `backup-docs`.
- `:Entity` carries `uuid`, `name`, `labels` (type is the non-`Entity` label),
  `summary` (551/555 non-empty), `name_embedding`. `RELATES_TO` carries `uuid`,
  `name` (edge type), `fact` (text), `valid_at`, `invalid_at` (on invalidated
  edges), `episodes`, `group_id`. **No `weight`** — Leiden edge weight is the
  count of parallel `RELATES_TO` between an entity pair. 1178 current / 1975 total.
- No `:Community` nodes yet (clean slate). The existing `Provenance` resolver
  (`graph_extract/provenance.py`) already turns fact UUIDs → source URLs; the
  synthesis LLM config (`_synthesis_client_and_model`, GLM-5.2 via judge_*) and
  the shared embedder (`build_graphiti`'s embedder path) already exist.

## 3. Architecture & flow

```
theme-build (CLI, full rebuild):
  1. detect: project (:Entity {g})–[RELATES_TO {g}]→(:Entity {g}) into GDS,
     weight = parallel-edge count; gds.leiden.stream includeIntermediateCommunities=true;
     drop dust (< min_community_size); derive PARENT_OF across adjacent levels;
     community_id = sha1(level + sorted member uuids).            [needs GDS]
  2. for each community (any order — reports come from member facts, not children):
       context  = assemble(members ranked by intra-community degree; intra-community
                  facts current-first by recency; labelled with fact UUIDs; token budget)
       report   = GLM-5.2(MS-GraphRAG prompt, context) -> JSON; 1 retry on bad JSON
       cited    = validate report.fact_ids ⊆ community fact UUIDs  (drop hallucinated)
  3. writeback: DELETE the group's :Community subgraph; write :Community nodes +
     IN_COMMUNITY + PARENT_OF; embed title+summary (shared embedder);
     stamp generated_at + best-effort corpus_cursor.
```

No LLM at detection or write-back; the only LLM call is report generation.
Citations are graph traversal (design decision #2): the model emits fact UUIDs,
a deterministic validator keeps the real ones, and global search later resolves
them to URLs.

## 4. Components

### 4.1 `theme_builder/detect.py` — hierarchical community detection
`async def detect_communities(driver, group_id, *, min_community_size, max_levels) -> list[Community]`
where `Community` is a dataclass `{community_id: str, level: int, member_uuids:
list[str], parent_id: str | None}`.
- Project a named GDS graph via `gds.graph.project.cypher` (node query:
  `:Entity {group_id}` → id; relationship query: aggregate `RELATES_TO
  {group_id}` between pairs into `weight = count(*)`, UNDIRECTED). A fixed graph
  name (e.g. `theme-<group_id>`); **drop it in a `finally`** (and pre-drop if it
  exists) so reruns are clean.
- `gds.leiden.stream(graph, {includeIntermediateCommunities: true, relationshipWeightProperty:'weight'})`
  yields `nodeId, communityId, intermediateCommunityIds[]`. Map nodeId → entity
  uuid. The `intermediateCommunityIds` array gives the level hierarchy
  (index 0 = finest). Cap at `max_levels`.
- Per level, group entities by community label → member sets. **Drop dust**
  (member count < `min_community_size`) at each level.
- **Deterministic id:** `community_id = sha1(f"{level}:" + ",".join(sorted(member_uuids)))[:16]`.
- **Parent derivation:** a level-`k` community's parent is the level-`k+1`
  community that contains (a superset of) its members (Leiden's intermediate
  levels are nested, so every member shares one level-`k+1` label). `parent_id`
  is that community's id; the top level → `None`. A surviving child's parent is
  always ≥ the child in size, so it can't have been dust-dropped — but defensively,
  if a derived `parent_id` refers to a dropped community, set `parent_id = None`.

### 4.2 `theme_builder/context.py` — report context assembly (pure)
`def assemble_context(members: list[EntityRow], facts: list[FactRow], *, top_entities: int, token_budget: int) -> ContextResult`
returning `{text: str, fact_uuids: set[str]}`.
- `EntityRow{uuid,name,type,summary,degree}`; `FactRow{uuid,fact,valid_at,invalid_at,name}`.
- Rank members by intra-community `degree` desc, take `top_entities`; render
  `- {name} ({type}): {summary}`.
- Order facts **current-first** (`invalid_at IS NULL`) then by `valid_at` desc;
  render each as `[{uuid}] {fact}  (valid {valid_at}{, invalid <invalid_at> if set})`
  and accumulate until `token_budget` (approx by chars/4). `fact_uuids` = the set
  actually included (the citable universe for validation).
- The fetch of `members`/`facts` (Neo4j) lives in the orchestrator; `context` is
  a pure function over the fetched rows so it is unit-testable without a DB.

### 4.3 `theme_builder/report.py` — LLM report generation + citation validation
- `_REPORT_PROMPT`: an adapted MS-GraphRAG community-report instruction — "You are
  analyzing a community of related entities from backup-product documentation.
  Using ONLY the numbered FACTS (each tagged `[uuid]`), write an analytical
  report. Cite the fact `[uuid]`s that support each finding. Do NOT write URLs."
  Output JSON schema: `{title, summary, full_report:[{finding, fact_ids:[...]}],
  rating (0-10 number), rating_explanation, tags:[...]}`.
- `async def generate_report(client, model, context: ContextResult, *, group_id) -> CommunityReport | None`:
  call the LLM (temperature 0, ample `max_tokens` — reasoning models truncate at
  low caps, see `answer_api/synthesize.py`), parse JSON, **one retry** on parse
  failure, return `None` (skip) if still bad. Then **validate**: keep only
  `fact_ids ⊆ context.fact_uuids` (drop hallucinated); `cited_fact_uuids` = the
  union across findings. Strip any URL the model emitted (reuse the
  `synthesize._URL_RE` pattern) — defense in depth for #2.
- `CommunityReport{title, summary, full_report(json str), rating, rating_explanation, tags, cited_fact_uuids}`.

### 4.4 `theme_builder/writeback.py` — persist the `:Community` subgraph
`async def write_communities(driver, embedder, group_id, communities, reports, corpus_cursor) -> dict`
- Delete first: `MATCH (c:Community {group_id:$g}) DETACH DELETE c` (derived &
  disposable — full rebuild).
- For each community with a report: embed `title + "\n" + summary` via the shared
  embedder; `MERGE (c:Community {community_id, group_id}) SET c += {level, title,
  summary, full_report, rating, rating_explanation, tags, cited_fact_uuids,
  embedding, member_count, generated_at: datetime(), corpus_cursor}`.
- `UNWIND member_uuids: MATCH (e:Entity {uuid}) MERGE (e)-[:IN_COMMUNITY]->(c)`.
- After all nodes exist: for each community with `parent_id`,
  `MATCH (c),(p:Community {community_id:parent_id, group_id}) MERGE (p)-[:PARENT_OF]->(c)`.
- Return counts (`communities`, `by_level`, `facts_cited`, `reports_written`,
  `reports_skipped`).

### 4.5 `theme_builder/cli.py` — `theme-build`
Wires it up: build driver + shared embedder + report client/model (from config);
`detect_communities`; per community fetch members+facts (Cypher) → `assemble_context`
→ `generate_report` → collect; `write_communities`. Echo per-level community
counts, reports written/skipped, total cited facts. Reuse the build-or-cleanup
resource guard pattern from `graph_extract/cli.py`.

## 5. Data model (matches plan §8.3)

`(:Community {community_id, group_id, level:int, title, summary, full_report,
rating:float, rating_explanation, tags:[str], cited_fact_uuids:[str],
embedding:[float], generated_at:datetime, corpus_cursor, member_count:int})`;
`(:Entity)-[:IN_COMMUNITY]->(:Community)`; `(:Community)-[:PARENT_OF]->(:Community)`.
`cited_fact_uuids` is an array property (Neo4j can't point an edge at an edge);
global search resolves each via `Provenance` → episode → article → URL.

## 6. Config additions (`ExtractSettings`)

```
report_llm_base_url: str = ""     # defaults to judge_base_url when empty
report_llm_model: str = ""        # defaults to judge_model
report_llm_api_key: str = ""      # defaults to judge_api_key (GLM-5.2 synthesis tier)
leiden_min_community_size: int = 3
leiden_max_levels: int = 3
report_token_budget: int = 12000
report_top_entities: int = 30
```
A `_report_client_and_model(settings)` helper resolves the report tier, falling
back to the judge/synthesis config when the `report_*` fields are empty — so the
default is GLM-5.2, overridable via `.env`.

## 7. Design-decision alignment

- **#2 (citations are traversal, never LLM):** the report LLM emits fact UUIDs;
  a deterministic validator drops any not in the community's real fact set; the
  model never writes a URL (prompt + `_URL_RE` strip). Global search resolves the
  validated UUIDs to URLs. The chain is community → cited_fact_uuid → episode →
  article → source_url.
- **#4 (one group / one embedding space):** communities are per `group_id`;
  `title+summary` embeddings use the SAME shared embedder as entities/facts/
  queries — so community-vs-query similarity is comparable in the next slice.
- **Derived & disposable:** a rebuild deletes and rewrites the `:Community`
  subgraph; the theme-builder is the only writer. Graphiti's schema is untouched
  (`:Community` is our node label, linked via `IN_COMMUNITY` — never merged into
  Graphiti nodes, design decision #5).

## 8. Error handling

- GDS projection dropped in a `finally` (and pre-dropped if a stale graph of the
  same name exists).
- A community whose report generation fails (bad JSON after one retry, or an LLM
  error) is logged and **skipped** — the build continues and reports
  `reports_skipped`. A partial rebuild is acceptable and visible.
- Hallucinated fact IDs are dropped by validation; a finding left with zero valid
  fact_ids is kept in `full_report` (its text is still useful) but contributes
  nothing to `cited_fact_uuids`.
- Empty graph / zero communities → writeback deletes stale communities and writes
  none; exits cleanly.

## 9. Testing

- **Unit — `context.assemble_context`:** ranking (degree desc), current-first fact
  order, token-budget truncation, fact-UUID labelling, `fact_uuids` == included set.
- **Unit — `report` validation:** hallucinated fact_ids dropped; URL stripped;
  bad JSON → one retry → None; `cited_fact_uuids` is the union of valid ids.
- **Unit — `detect` helpers:** deterministic `community_id` (same members →
  same id; different order → same id), dust-drop threshold, parent derivation
  from nested intermediate labels (with stubbed Leiden rows — no GDS).
- **Integration (Neo4j testcontainer, NO GDS needed):** `writeback` — delete +
  write, `:Community` props, `IN_COMMUNITY`/`PARENT_OF` edges, embedding present;
  and the members/facts fetch Cypher against a seeded entity graph.
- **`@live` smoke (real GDS-enabled Neo4j):** `detect_communities` on `backup-docs`
  returns ≥1 multi-level community with ≥`min_community_size` members and correct
  parent nesting.
- **Controller demonstration:** run `theme-build` on `backup-docs`; show a real
  community report (title/summary/findings) whose `cited_fact_uuids` resolve to
  real source URLs via `Provenance`; write `docs/superpowers/theme-builder-report.md`.
- Full non-live suite + ruff/mypy clean.

## 10. Acceptance criteria

1. `theme-build` produces a hierarchical `:Community` subgraph over `backup-docs`
   with `IN_COMMUNITY` membership and `PARENT_OF` levels; a rerun is a clean full
   rebuild (old communities deleted).
2. Each community has an LLM report (`title, summary, full_report, rating,
   rating_explanation, tags`) and `cited_fact_uuids` that are all real fact UUIDs
   of that community (no hallucinations; no URLs authored by the LLM).
3. Community `title+summary` embeddings use the shared embedder; `generated_at`
   stamped; report model defaults to GLM-5.2 and is config-overridable.
4. GDS projection is always cleaned up; report failures skip a community without
   aborting the build.
5. Unit + integration tests green; `@live` detection smoke passes; a demonstration
   report is written showing citations resolving to URLs; ruff/mypy clean.

## 11. Deferred

- Global map-reduce `/search/global` (next slice) — consumes this layer.
- Incremental refresh: dirty-marking from graph-sync, Jaccard-stable community IDs,
  regenerate only dirty/new reports, `corpus_cursor`-driven staleness.
- Roll-up reports (level ≥ 1 from child summaries) — unnecessary at 555 entities;
  revisit when parent communities exceed the token budget.
- Structural vendor attribution (via `SAME_AS`) in reports beyond model-inferred
  `tags`; DRIFT; report-quality eval harness.
