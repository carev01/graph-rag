# Temporal GraphRAG over DocExtractor — Architecture & Implementation Plan

**Scope:** Build a temporally-aware GraphRAG system over the DocExtractor corpus (40 vendors, 79 products, 280 sources, ~105k articles of backup-industry documentation), using Graphiti as the knowledge-graph substrate, a Microsoft-GraphRAG-style community/report layer for cross-corpus thematic questions, a DRIFT-capable query router with URL-level citation traceability, and exposure to corporate users through a Microsoft Copilot agent.

**Status:** Planning document — no code. Written 2026-07-12 against the DocExtractor Client Usage Guide (live deployment `docextractor.k3s.home.lan`).

---

## 1. Objectives and design constraints

The system must satisfy five requirements simultaneously, and each one drives a specific architectural choice:

1. **Frequent incremental updates.** DocExtractor re-extracts sources continuously (the sample data shows daily runs with 5–200 changed articles per source). Microsoft GraphRAG's batch reindex model is a poor fit; Graphiti's episodic, incremental ingestion is the right substrate. The DocExtractor delta feed (`GET /api/articles/delta`) with its gap-free cursor semantics is the single ingestion channel.

2. **Temporal awareness.** You want to answer "how did vendor X's treatment of workload Y change?" Graphiti's bi-temporal model (`valid_at` / `invalid_at` on fact edges, `reference_time` on episodes) provides this, but only if updates are ingested as *new episodes* rather than destructive rewrites — see §4.4, which is the most consequential design decision in this plan.

3. **Cross-corpus thematic summarization.** "How do all these vendors treat backup of Kubernetes workloads?" is a query-focused-summarization (QFS) problem. Graphiti's built-in `build_communities` produces single-level Leiden communities whose summaries are collations of member entity summaries — useful, but not the analytical, claim-bearing reports that make MS GraphRAG's global search work. You will build a derived **community-report layer** (hierarchical Leiden + LLM-written reports) on top of the Graphiti entity graph, refreshed on a cadence.

4. **URL-level citation traceability.** Every answer must trace back to `source_url`. This dictates a **provenance chain that is never LLM-mediated**: fact edge → supporting episode(s) → article ID → `source_url`. The chain is maintained structurally in the graph so citation resolution is a deterministic traversal, not a generation step.

5. **Copilot exposure.** Corporate users consume the system through Microsoft Copilot. The retrieval layer is therefore packaged as an **MCP server (streamable HTTP)** onboarded into Copilot Studio, with an OpenAPI custom-connector path as fallback.

Two hard constraints shape everything else:

- **Cost.** ~105k articles is roughly 250–400M tokens of content. Graphiti's extraction pipeline makes several LLM calls per episode (entity extraction, entity dedup/resolution, edge extraction, edge dedup, temporal resolution). Naive full ingestion could consume 1–2B+ LLM tokens. The plan mitigates this with a two-layer graph (structural metadata is written deterministically for free; only content semantics go through LLM extraction), a small/cheap extraction model, and a phased vendor-by-vendor rollout.
- **Network topology.** DocExtractor lives on a private K3s cluster with an internal CA (`rootca.home.lan`); Copilot lives in the Microsoft cloud. The ingestion side stays entirely on-prem; only the retrieval API needs controlled public exposure (§8.3).

---

## 2. System overview

```
                        DocExtractor (K3s, on-prem)
              delta feed (NDJSON, cursor)   webhooks (extraction_complete)
                          │                        │
                          ▼                        ▼
        ┌──────────────────────────────────────────────────────┐
        │ (A) INGESTION SERVICE  "graph-sync"                   │
        │  bootstrap + incremental sync · cursor store          │
        │  chunker · episode mapper · tombstone handler         │
        │  work queue + rate limiter (LLM budget)               │
        └──────────────┬───────────────────────┬────────────────┘
          deterministic│Cypher       Graphiti  │add_episode (LLM)
                       ▼                       ▼
        ┌──────────────────────────────────────────────────────┐
        │ (B) KNOWLEDGE GRAPH — Neo4j (single instance)         │
        │  B1 Structural layer (no LLM): Vendor→Product→Source  │
        │     →Chapter→Article nodes, TOC edges, source_url,    │
        │     content_hash, versions                            │
        │  B2 Graphiti semantic layer: Episodic nodes,          │
        │     Entity nodes (custom types), RELATES_TO fact      │
        │     edges with valid_at/invalid_at, MENTIONS edges    │
        │  B1↔B2 link: (:Article)-[:HAS_EPISODE]->(:Episodic)   │
        └──────────────┬───────────────────────▲────────────────┘
         read entity   │                       │ write back
         subgraph      ▼                       │
        ┌──────────────────────────────────────────────────────┐
        │ (C) COMMUNITY-REPORT LAYER  "theme-builder"           │
        │  hierarchical Leiden (Neo4j GDS) · dirty-marking      │
        │  MS-GraphRAG-style report prompt · report embeddings  │
        │  :Community nodes (level, parent, members, report,    │
        │  provenance appendix) · scheduled/triggered refresh   │
        └──────────────────────────▲───────────────────────────┘
                                   │ read
        ┌──────────────────────────┴───────────────────────────┐
        │ (D) RETRIEVAL SERVICE  "answer-api" (FastAPI)         │
        │  /search/local   → Graphiti hybrid search + citations │
        │  /search/global  → dynamic community map-reduce       │
        │  /search/drift   → primer → follow-ups → synthesis    │
        │  /timeline       → temporal fact history + versions   │
        │  /answer         → router over the above              │
        │  Citation resolver: edge→episode→article→source_url   │
        └──────────────────────────▲───────────────────────────┘
                                   │ MCP (streamable HTTP)
        ┌──────────────────────────┴───────────────────────────┐
        │ (E) EXPOSURE — APIM/reverse proxy (public CA, OAuth)  │
        │  → Copilot Studio agent (MCP onboarding wizard)       │
        │  → published to Teams / M365 Copilot                  │
        └──────────────────────────────────────────────────────┘
```

This matches your sketch with three refinements worth calling out up front:

- **The graph has two sub-layers.** Everything DocExtractor already gives you as structured metadata (vendor, product, source, chapter hierarchy, URLs, hashes, ordering) is written to Neo4j **deterministically with plain Cypher, at zero LLM cost**. Graphiti's LLM pipeline is spent only on what it's uniquely good at: extracting semantic entities and temporal facts from the article *content*. This cuts extraction cost dramatically and makes provenance exact rather than extracted.
- **Citation resolution is a graph traversal, not part of generation.** The LLM in the retrieval layer only ever cites *fact IDs / report IDs* it was given; a deterministic resolver expands those to URLs. This is what makes "trace back to the original source" reliable.
- **The community layer is derived and disposable.** It can be rebuilt from B2 at any time; nothing downstream writes into it except the theme-builder. That keeps the refresh problem tractable.

**Recommended stack:** Python 3.12; `graphiti-core` on **Neo4j 5.x + GDS 2.x** (FalkorDB is supported by Graphiti but has no GDS — you would lose in-database hierarchical Leiden and have to export to `graspologic`; Neptune has the same issue. Choose Neo4j unless there is a strong operational reason not to); FastAPI for graph-sync and answer-api; Postgres or Redis for the sync cursor/state; an LLM provider with a cheap-tier model for extraction and a strong-tier model for reports/synthesis; a single embedding model used consistently everywhere (entity names, fact edges, community reports, and query embedding must share one space).

---

## 3. Layer A — Ingestion service ("graph-sync")

A single long-running service (or K8s deployment in the same cluster as DocExtractor, which sidesteps the internal-CA issue) with three responsibilities: consume the delta feed correctly, translate records into graph writes, and meter the LLM budget.

### 3.1 Sync protocol (bootstrap + incremental)

Follow the guide's Pattern 1/2 exactly; the details that matter for correctness:

1. **Bootstrap** with no `since`, sharded by `vendor_id` (Pattern 3) so vendors can be ingested in priority order and in parallel workers. Store the `bootstrap_start.next_since` watermark *immediately* on first line; only promote it to the live cursor after the terminal `cursor` control record arrives. On a dropped stream, resume with `bootstrap_after=<highest applied article id>` and keep the original watermark.
2. **Incremental** pulls are triggered by the `extraction_complete` webhook (HMAC-verified, `X-DocExtractor-Signature`) with a debounce (e.g., coalesce webhooks arriving within 5 minutes — extraction runs for different sources often cluster), plus a 30-minute polling fallback so a missed webhook self-heals. Always pull with the *stored* cursor, never the webhook's watermark.
3. **Cursor discipline:** the cursor is opaque; persist atomically (Postgres row or Redis with AOF) and advance only on a clean terminal control record. Because the feed is idempotent and gap-free, at-least-once processing with idempotent graph writes is the consistency model — every write below must be an upsert keyed on stable IDs.
4. **Change gating on `content_hash`.** Store the last-processed `content_hash` per article (in the `:Article` node). Skip records whose hash is unchanged (bootstrap replays, overlap after resume). Note the guide's point that VLM enrichment runs surface as `updated` deltas with a changed hash — these *should* be reprocessed, since image descriptions add indexable content.

### 3.2 What each delta record becomes

For an `added`/`updated` content record, graph-sync performs, in order:

1. **Structural upsert (Cypher, no LLM):** `MERGE` the `(:Vendor)`, `(:Product)`, `(:Source)`, `(:Chapter)` (from `top_level_chapter` / `parent_chapter`), and `(:Article {id})` nodes with all metadata (`title`, `source_url`, `topic_key`, `content_hash`, `estimated_tokens`, `sort_order`, `extracted_at`, `run_id`). Wire `(:Vendor)-[:HAS_PRODUCT]->(:Product)-[:HAS_SOURCE]->(:Source)-[:HAS_ARTICLE]->(:Article)` and chapter membership. Periodically (weekly, or on `extraction_complete` for a source) refresh the full TOC tree for the source from `GET /api/articles/toc/{source_id}` to capture sibling/ordering edges and non-leaf navigation nodes.
2. **Chunking:** split `content_markdown` into episode-sized chunks (see §3.3).
3. **Semantic ingestion:** enqueue one Graphiti `add_episode` job per chunk (see §4).
4. **Linking:** after each episode is created, `MERGE (:Article {id})-[:HAS_EPISODE]->(:Episodic {uuid})` with properties `{chunk_index, content_hash}`. This edge *is* the provenance backbone.

For a `removed` tombstone: look up the article's episodes via `HAS_EPISODE`, apply the deletion/invalidation policy of §4.4, then flag the `:Article` node `removed: true, removed_at: …` (keep the node — tombstoned articles remain useful for temporal queries: "vendor X removed its OpenShift restore guidance in July 2026"). Remember tombstones can arrive with `run_id: null` (out-of-band vendor/product/source deletion) — handle batches of thousands gracefully.

### 3.3 Chunking strategy

Graphiti extraction quality degrades on very large episodes; docs articles range from ~450 to ~8,000 tokens (`estimated_tokens` is pre-computed for you). Policy:

- Split on markdown heading boundaries (`##`/`###`), greedily packing sections into chunks of **≤ ~1,500–2,000 tokens** with a 1-heading overlap of context (prepend the article title and chapter path to every chunk: `"[AWS Backup › Backup vaults › Vault Lock] …"` — this anchors entity extraction to the right product context and dramatically improves cross-vendor disambiguation).
- Append each image's VLM `description` (when non-null) to the chunk containing the image reference, prefixed `"Figure: …"` — the guide notes descriptions are also injected as captions into the markdown; verify at implementation time and avoid double-inclusion by checking for the caption text.
- Articles ≤ 2,000 tokens become a single episode. Record `chunk_index` and the heading path on the `HAS_EPISODE` edge so citations can later point at the section, not just the page.

### 3.4 Work queue and budget control

`add_episode` calls are the expensive resource. Put chunks on a durable queue (Redis streams / RQ / Celery — anything with retry + dead-letter) with: per-provider concurrency limits (Graphiti's own `SEMAPHORE_LIMIT` guidance), token-budget accounting per day, priority lanes (incremental updates preempt bootstrap backfill so freshness never waits behind the initial load), and a dead-letter queue for chunks that repeatedly fail extraction (malformed markdown, provider errors) with alerting. Failed chunks must not advance the article's stored `content_hash`, so retries re-derive from a consistent state.

---

## 4. Layer B2 — Graphiti semantic graph design

### 4.1 group_id strategy: one corpus-wide group

Graphiti namespaces graphs by `group_id`, and entity resolution/deduplication happens *within* a group. Cross-corpus questions ("how do all vendors treat S3 backup?") require that `Amazon S3` extracted from Veeam docs and from AWS docs resolve to the **same entity node** — otherwise communities fragment along vendor lines and global search degrades into per-vendor silos. Therefore: **a single `group_id` (e.g., `backup-docs`) for the whole corpus.**

The costs of this choice are (a) a larger dedup search space per `add_episode` (slower, slightly costlier) and (b) no cheap per-vendor isolation. Mitigations: custom entity types (below) narrow the dedup candidate sets; vendor scoping at query time is done through the structural layer (filter facts to episodes whose article belongs to vendor X) rather than through groups. Do **not** use one group per vendor — you would be permanently unable to build cross-vendor communities.

### 4.2 Custom entity and edge ontology

Define a small, closed set of Pydantic entity types passed to `add_episode`. A deliberately compact ontology keeps extraction consistent across 40 vendors' differing terminology. Suggested starting set:

| Entity type | Examples | Notes |
|---|---|---|
| `Vendor` | Veeam, Dell, Acronis | Will be reconciled with structural layer (§4.3) |
| `Product` | AWS Backup, PowerProtect Data Manager, Cyber Protect Cloud | Ditto; include `version` attribute |
| `Workload` | VMware vSphere, Kubernetes, Microsoft 365, SQL Server, Amazon S3, Oracle | The pivot of your canonical cross-corpus question |
| `Capability` | immutability, instant recovery, CDP, dedup, air gap, agentless backup | Features/mechanisms |
| `Concept` | RPO, RTO, 3-2-1 rule, backup window, retention policy | Domain concepts |
| `Platform` | Windows, Linux, Azure, AWS, GCP, Hyper-V | Infrastructure context |
| `Requirement` | licenses, ports, minimum versions, permissions | High value for how-to answers |

Edge (fact) types with an `edge_type_map`, e.g.: `Product SUPPORTS Workload`, `Product PROVIDES Capability`, `Capability APPLIES_TO Workload`, `Product REQUIRES Requirement`, `Product INTEGRATES_WITH Platform/Product`, `Product LIMITS Workload` (restrictions/unsupported scenarios — explicitly valuable, docs are full of "X is not supported when Y"). Unmapped pairs fall back to Graphiti's generic `RELATES_TO`, which is fine. Iterate the ontology after Phase 1 evaluation on real extractions; Graphiti tolerates additive schema evolution.

Also pass `excluded_entity_types` or extraction instructions to suppress noise classes you'll otherwise drown in: UI element names, button labels, generic doc-navigation entities ("this guide", "the following table").

### 4.3 Reconciling structural and semantic layers

The structural layer already has exact `Vendor`/`Product` nodes; Graphiti will *also* extract vendor/product entities from text. Don't fight this — link them. A nightly reconciliation job matches Graphiti `Entity` nodes of type Vendor/Product to structural nodes (exact/fuzzy name match + co-occurrence via episodes) and writes `(:Entity)-[:SAME_AS]->(:Vendor|:Product)`. This gives query-time filters a bridge in both directions and gives the community layer clean vendor attribution for reports. Keep it as a linking edge rather than merging nodes — Graphiti owns its entity lifecycle and merging externally-created data into its nodes invites upgrade pain.

### 4.4 Update semantics: the temporal policy (key decision)

Three options exist when an article changes; pick per change-type:

- **Option 1 — remove & re-add:** delete old episodes (`remove_episode` cascades to solely-derived nodes/edges), add new ones. Graph always reflects "current docs" only. *Loses the temporal history you explicitly want.* Rejected as the default.
- **Option 2 — append & invalidate (recommended for `updated`):** add the new chunks as new episodes with `reference_time = extracted_at` (or `last_updated_at` when present, which is the true document-change time). Graphiti's temporal resolution invalidates contradicted prior facts (`invalid_at` set), which is exactly the "how did this change" record. Mark superseded episodes `superseded: true, superseded_at: …` (custom property via Cypher) and *exclude superseded episodes from default retrieval* while keeping them for `/timeline` queries.
- **Residual-staleness control:** Graphiti only invalidates facts that are *contradicted* by new text. A silently deleted claim (paragraph removed, nothing contradicts it) survives as an apparently-valid fact. Mitigation: a weekly "sweep" job finds fact edges whose *only* supporting episodes are all superseded/removed and sets `expired_at` on them (they remain temporally queryable but drop out of current-truth retrieval). This is a pure-Cypher job, no LLM.
- **For `removed` tombstones:** do not `remove_episode` (that destroys history); instead mark all the article's episodes `removed: true` and run the same expiry sweep over their solely-supported facts, with `invalid_at = removed_at`.

This policy gives you: current-truth answers by default (facts not expired/invalidated, episodes not superseded), and a genuine temporal record (all facts with their validity intervals + DocExtractor's own `ArticleVersion` history via `GET /api/articles/{id}/versions/...` for verbatim before/after diffs when a user drills in).

### 4.5 Provenance chain (the citation backbone)

End-to-end: `RELATES_TO` fact edge → `episodes` (Graphiti stores supporting episode UUIDs on edges, and `MENTIONS` edges link episodes to entities) → `(:Episodic)<-[:HAS_EPISODE]-(:Article)` → `article.source_url` (+ `title`, `vendor`, `product`, heading path from the `HAS_EPISODE` edge). Every retrieval response carries this resolved chain. Two rules keep it trustworthy: graph-sync is the only writer of `HAS_EPISODE`, and answer-api never lets the LLM emit URLs — only fact/report IDs, resolved afterward.

---

## 5. Layer C — Community-report layer ("theme-builder")

A scheduled batch service that derives the thematic layer from the Graphiti entity graph. It is intentionally *not* Graphiti's `build_communities`: that produces one flat Leiden level with summaries collated from entity summaries. You need hierarchy (for level-selected global search and DRIFT entry) and analytical reports (findings, claims, cross-vendor comparisons) in the MS GraphRAG style. You *may* still call Graphiti's `build_communities` for its incremental label-propagation membership as a cheap freshness signal, but the report layer below is the system of record for QFS.

### 5.1 Build pipeline (per refresh cycle)

1. **Project the entity subgraph.** Cypher → GDS in-memory graph: nodes = Graphiti `Entity` nodes (exclude `Episodic` and structural nodes), relationships = `RELATES_TO` fact edges that are current (no `expired_at`/`invalid_at`), weighted by `w = ln(1 + supporting_episode_count) + ln(1 + distinct_source_count)` so cross-vendor, well-attested relationships bind communities more strongly than single-page mentions.
2. **Hierarchical Leiden** via `gds.leiden` with `includeIntermediateCommunities: true` (typ. 3 levels; tune resolution/gamma so level-0 leaf communities average ~10–40 entities). Discard singleton/dust communities below a size threshold.
3. **Stable community identity.** Match new communities to previous build by maximum Jaccard overlap of member sets (≥ 0.5 → same `community_id`, carried forward; else new ID). Stability matters because reports are cached, cited, and embedded — thrashing IDs invalidates everything downstream.
4. **Report generation** for new or dirty communities only (see 5.2), using an adapted MS GraphRAG community-report prompt. Context assembly per community, under a token budget (~12–16k): member entities ranked by weighted degree (name, type, entity summary); top fact edges by weight (the `fact` text, with validity dates — instruct the model that facts carry timestamps and vendor attribution); for level ≥ 1, the child communities' report summaries instead of raw facts (roll-up). Output schema (JSON): `title`, `summary` (1 paragraph), `full_report` (findings with per-finding references to fact IDs), `rating` (0–10 importance), `rating_explanation`, `tags` (e.g., workload names, vendors covered). **Require the model to reference fact IDs per finding** — this is what lets global answers cite URLs (finding → fact → episode → article → URL) instead of hand-waving at "the community".
5. **Write-back.** `(:Community {community_id, level, title, summary, full_report, rating, tags, embedding, generated_at, member_count, corpus_cursor})` with `[:IN_COMMUNITY]` from member entities, `[:PARENT_OF]` hierarchy edges, and `[:CITES_FACT]` edges to the fact edges referenced by findings (materialize as edge-to-node references via fact UUID properties, since Neo4j can't point edges at edges — store cited fact UUIDs as an array property and/or link to the fact's endpoint entities). Embed `title + summary` with the same embedding model used everywhere else. Stamp the delta-feed cursor at build time (`corpus_cursor`) so every report is auditable: "this thematic view reflects the corpus as of cursor X / time T."

### 5.2 Incremental refresh strategy

- **Dirty marking:** graph-sync appends every touched entity UUID (created, re-mentioned, or with invalidated edges) to a `dirty_entities` set (Redis/Postgres) as it ingests. A community is dirty if it contains a dirty entity or its membership changed in step 3; a parent is dirty if any child is dirty.
- **Triggering:** run nightly by default; run early if `|dirty_entities| / |entities| > ~2%` or after large deltas (the `extraction_complete` payload's added/updated/removed counts give you this signal for free), with a debounce so a burst of extraction runs produces one rebuild.
- **What actually re-runs:** Leiden re-runs globally (it's cheap — seconds to minutes at this scale, ~100k–500k entities expected); report generation re-runs only for dirty/new communities (the expensive part). Expect steady-state nightly report regeneration to touch 1–5% of communities. Full-corpus report regeneration is reserved for ontology/prompt changes.
- **Staleness contract:** thematic answers lag the corpus by at most one refresh cycle; answer-api includes each report's `generated_at`/`corpus_cursor` in responses so the agent can say "themes as of last night." Local/DRIFT fact retrieval is always live.

---

## 6. Layer D — Retrieval service ("answer-api")

A FastAPI service exposing four retrieval modes plus a router, all returning a **uniform response contract**:

```json
{
  "answer_markdown": "...",
  "mode": "local | global | drift | timeline",
  "citations": [
    {"n": 1, "fact": "Veeam B&R 12 supports immutable backups to S3 Object Lock",
     "valid_at": "2025-11-02", "invalid_at": null,
     "article_id": "…", "title": "Immutability", "section": "Backup vaults › Vault Lock",
     "vendor": "Veeam", "product": "Backup & Replication",
     "source_url": "https://…"}
  ],
  "communities_used": [{"community_id": "…", "title": "…", "generated_at": "…"}],
  "freshness": {"graph_cursor_time": "…", "reports_as_of": "…"}
}
```

The synthesis LLM is instructed to cite by `[n]` markers referring to fact/report IDs it was given in context; a deterministic resolver builds the `citations` array by traversing the provenance chain. The LLM never fabricates a URL because it never writes one.

### 6.1 Local search (specific/factual)

Graphiti hybrid search (`graphiti.search` / `_search` with a combined recipe: semantic + BM25 + graph reranking, RRF or cross-encoder) over current facts (filter out `expired_at`/`invalid_at`/superseded), with optional scoping: vendor/product filters applied via the structural bridge (episode → article → vendor), and `SearchFilters` on entity/edge types when the router detects them ("what are the *requirements* for…"). Top-K facts + their 1-hop entity summaries go to the synthesis model. Latency target: sub-second retrieval (Graphiti retrieval has no LLM in the loop), 3–8s end-to-end with synthesis.

### 6.2 Global search (thematic/cross-corpus)

Dynamic-selection map-reduce over community reports (MS GraphRAG 1.x style, cheaper than rating every report): shortlist reports by embedding similarity + `rating` at a chosen level (default level 1; level 0 for narrow themes, top level for "state of the industry"), optionally descend into high-scoring communities' children; **map** — a cheap LLM scores each shortlisted report's relevance and extracts key points *with the fact IDs the report's findings already carry*; **reduce** — the strong model synthesizes across map outputs, organized by theme and vendor, citing fact IDs. This is how "how do all vendors treat XYZ workload" gets both breadth (reports) and verifiable citations (facts→URLs).

### 6.3 DRIFT search (broad → deep)

Three phases. **Primer:** embed the query against community-report embeddings, take top-K reports, have the strong model draft a preliminary answer plus 3–6 targeted follow-up queries (with a relevance-scored budget). **Follow-up loop (1–2 iterations):** each follow-up runs *local* search (§6.1); when the follow-up originated from a specific community, bias retrieval by using that community's top member entities as Graphiti center nodes for graph-distance reranking — this is the "enter through the theme, drill into the facts" motion. **Synthesis:** merge primer + follow-up evidence, rank claims, cite fact IDs. This mode is the default for questions that are broad but expect concrete, sourced detail — most real user questions in this domain.

### 6.4 Timeline queries (temporal)

For "how did X change": Cypher over fact edges touching the resolved entities, returning facts *including* invalidated ones ordered by `valid_at`/`invalid_at`; group into change events; optionally enrich a specific transition with the verbatim diff from DocExtractor's `versions/{id}/diff` endpoint (answer-api may call DocExtractor read-only for this). Output: a change narrative with per-interval citations (the superseded article version's `source_url` still resolves — cite both old and new).

### 6.5 Router

Two-level routing. **Inside answer-api:** a cheap LLM classifier (few-shot: local vs global vs drift vs timeline, plus extracted scope entities/vendors/date-ranges) behind `/answer`, with heuristics as guardrails (explicit "compare all vendors" → global/drift; contains "changed/used to/history/since" → timeline; names a single product + specific noun → local) and a fallback chain (empty local results → escalate to drift). **In Copilot Studio:** also expose the four modes as *separate MCP tools* with sharp descriptions and let the agent's generative orchestrator pick — in practice the orchestrator handles coarse routing well, and `/answer` remains the safe default tool. Ship both; measure which routing the agent actually needs.

---

## 7. Layer E — Microsoft Copilot integration

### 7.1 Recommended path: MCP server → Copilot Studio agent

Copilot Studio natively onboards MCP servers (Tools → Add tool → Model Context Protocol; requires generative orchestration), auto-imports every tool the server publishes, and tracks tool changes dynamically. MCP servers ride Power Platform connector infrastructure, so DLP policies, VNet integration, and standard auth options apply. Declarative agents for M365 Copilot also support MCP (GA), via the Microsoft 365 Agents Toolkit. Notes that constrain implementation: Copilot Studio supports the **streamable HTTP** transport only (SSE transport is no longer supported), and the custom-connector route requires an **OpenAPI v2 (Swagger)** spec with the `x-ms-agentic-protocol: mcp-streamable-1.0` extension if you go that way instead of the onboarding wizard.

So: wrap answer-api's four modes as MCP tools (`answer_question` as the router/default, `compare_across_vendors` → global, `investigate_topic` → drift, `get_change_history` → timeline, plus a `get_article` convenience tool returning title+URL+summary for a cited article). Tool descriptions are the routing surface for Copilot's orchestrator — write them as decision rules, not feature lists. Keep responses well under connector payload ceilings: cap answers (~2–4k tokens), cap citations (~15, deduped by URL), and have tools return markdown with inline `[n]` links so answers render cleanly in Teams/Copilot chat without adaptive-card work in v1.

**Agent build:** a Copilot Studio custom agent ("Backup Docs Analyst") with instructions covering persona, when to use which tool, citation etiquette ("always show sources; never answer product-capability questions without citations"), and honesty about freshness (surface the `freshness` block when using thematic answers). Publish to Teams first (fastest to pilot); promote to a declarative agent in M365 Copilot chat once stable. Suggested prompts seeded with the real use cases ("How do the major vendors handle immutable backups for S3?", "What changed in Dell PPDM's Kubernetes guidance recently?").

### 7.2 Alternatives (documented, not recommended as primary)

- **REST/OpenAPI custom connector** calling answer-api directly (no MCP): works, but you lose dynamic tool sync and take on OpenAPI-v2 conversion friction. Keep as fallback if MCP onboarding hits tenant policy issues.
- **Microsoft Graph connector** pushing articles/reports into the M365 semantic index: gives native Copilot grounding but forfeits graph reasoning, routing, temporal answers, and controlled citations — it would reduce the system to vanilla RAG. Could be a *complement* later (index community reports so Copilot's built-in search can discover the agent's domain), never the primary path.

### 7.3 Network and auth topology

Copilot (cloud) must reach the MCP endpoint over public HTTPS with a publicly-trusted certificate — the K3s internal CA (`rootca.home.lan`) will not do. Ingestion stays fully private; only answer-api's MCP surface is exposed. Options, in order of preference:

1. **Azure API Management (or Front Door) as the public face**, with private connectivity back to the cluster (S2S VPN / ExpressRoute / self-hosted APIM gateway on the K3s cluster). APIM validates **Entra ID OAuth2** tokens (Copilot Studio's MCP wizard supports OAuth 2.0, including manual Entra configuration), enforces rate limits, and gives audit logs. This is the enterprise-defensible answer.
2. **Cloudflare Tunnel / equivalent** exposing answer-api with OAuth or API-key auth configured in the MCP wizard — fine for the pilot phase, cheaper, less governance.

Either way answer-api itself also enforces auth (defense in depth) and is read-only by construction. Per-user authorization is not needed in v1 (the corpus is public vendor documentation); if it ever is, Entra token claims flow through APIM to answer-api.

---

## 8. Data model reference

### 8.1 Structural layer (written by graph-sync, Cypher only)

| Node | Key properties |
|---|---|
| `:Vendor` | `id` (DocExtractor UUID), `name`, `website` |
| `:Product` | `id`, `name`, `version`, `vendor_id` |
| `:Source` | `id`, `name`, `base_url`, `source_type`, `platform`, `last_extracted_at` |
| `:Chapter` | `source_id + title` composite key, `level` |
| `:Article` | `id`, `title`, `source_url`, `topic_key`, `content_hash`, `estimated_tokens`, `sort_order`, `extracted_at`, `removed`, `removed_at` |

Relationships: `HAS_PRODUCT`, `HAS_SOURCE`, `HAS_ARTICLE`, `HAS_CHAPTER`, `IN_CHAPTER`, `NEXT` (sibling order via `sort_order`), `HAS_EPISODE` (→ Graphiti `:Episodic`, props: `chunk_index`, `heading_path`, `content_hash`).

### 8.2 Graphiti layer (written by graphiti-core)

Graphiti-managed: `:Episodic` (+ custom props added by graph-sync: `superseded`, `removed`), `:Entity` (with custom-type labels/attributes), `RELATES_TO` fact edges (`fact`, `valid_at`, `invalid_at`, `episodes[]`, embeddings; + swept `expired_at`), `MENTIONS` episodic edges. Custom bridge: `(:Entity)-[:SAME_AS]->(:Vendor|:Product)`. Treat Graphiti's schema as owned by the library — add properties, never rename or restructure its elements.

### 8.3 Community layer (written by theme-builder)

`:Community {community_id, level, title, summary, full_report, rating, rating_explanation, tags[], cited_fact_uuids[], embedding, generated_at, corpus_cursor, member_count}`; `(:Entity)-[:IN_COMMUNITY]->(:Community)`, `(:Community)-[:PARENT_OF]->(:Community)`.

### 8.4 Delta record → graph mapping cheat sheet

| Delta field | Destination |
|---|---|
| `id` | `:Article.id`; key for `HAS_EPISODE`, tombstones, hash gating |
| `topic_key` / `source_url` | `:Article` props; `source_url` is the citation target |
| `vendor` / `product` | Structural `MERGE`; also prefixed into chunk text for extraction context |
| `content_markdown` (+ `images[].description`) | Chunked → Graphiti episodes |
| `content_hash` | Change gate; stored on `:Article` and `HAS_EPISODE` |
| `estimated_tokens` | Chunk planning |
| `parent_chapter` / `top_level_chapter` / `sort_order` | Chapter nodes, `NEXT` ordering, heading path |
| `last_updated_at` (else `extracted_at`) | Graphiti `reference_time` |
| `run_id`, `seq` | Audit properties on `:Article` |
| tombstone `removed_at` | §4.4 removal policy; `invalid_at` for swept facts |

---

## 9. Scale, cost, and performance planning

**Corpus math (order of magnitude):** ~105k articles × ~2.5k avg tokens ≈ 260M content tokens → ~150–200k episodes after chunking. Graphiti extraction overhead is roughly 3–6× content tokens across its multi-call pipeline, i.e., **~0.8–1.5B LLM tokens for a full bootstrap**. Consequences:

- Use a **cheap, fast model for extraction** (mini/haiku tier) and reserve the strong model for community reports, map-reduce, and synthesis. Benchmark extraction quality on ~200 articles across 3 vendors before committing (Phase 1 gate).
- **Phase the bootstrap by vendor** in business-priority order; the system is useful long before all 40 vendors are in. Budget expectation at mini-tier pricing: low-thousands of dollars for full bootstrap, tens of dollars/day steady-state (daily deltas look like hundreds of changed articles corpus-wide). Validate with the Phase 1 measurement rather than trusting these priors.
- Embedding cost is comparatively negligible; pick the embedding model **once** (re-embedding the whole graph later is painful — Graphiti stores embeddings on nodes and edges).
- Steady-state latencies: local ≈ 3–8s; global ≈ 10–30s (map fan-out); drift ≈ 15–45s. Copilot tolerates this but set user expectations in agent instructions ("deep investigations take up to a minute"). Neo4j sizing: this graph (order 10⁶ nodes, 10⁷ edges with embeddings) fits comfortably on a single well-provisioned instance (32–64 GB RAM); enable native vector indexes and full-text indexes per Graphiti requirements; GDS needs headroom for the Leiden projection.

---

## 10. Operations, evaluation, security

**Monitoring.** Sync health: cursor lag (time since last clean cursor advance), queue depth, dead-letter count, per-vendor bootstrap progress; reconcile weekly against `GET /api/dashboard/sources` (article counts per source vs `:Article` counts per source — drift means missed deltas). Graph health: entity/edge growth, dedup rate, fraction of expired facts. Theme layer: dirty-community backlog, report age distribution. Retrieval: per-mode latency/error rates, router decision distribution, citation-resolution failures (a resolved citation with a dead URL or missing article is a red flag).

**Evaluation (build in Phase 1, run continuously).** A golden set of ~60–100 questions spanning the four modes with expected source URLs, curated with a domain expert. Metrics: citation precision (does the cited URL actually support the claim — spot-check with an LLM judge + human sampling), answer faithfulness/groundedness (RAGAS-style), routing accuracy, and *temporal correctness* on a handful of known doc changes (seed these by watching real deltas during Phase 1). Re-run the suite after every ontology, prompt, or model change, and after community rebuilds (report regressions are otherwise invisible).

**Security.** (a) *Prompt injection via scraped docs*: vendor documentation is third-party content flowing into extraction and synthesis prompts; treat it as data — extraction prompts should instruct the model to ignore instructions inside content, and answer-api's synthesis prompt must never grant the retrieved text authority over behavior. Low real-world risk for vendor docs, non-zero for a system exposed to a whole company. (b) *Credentials*: read-only DocExtractor API key in a secret store; answer-api has no write path to DocExtractor. (c) *Exposure*: only the MCP surface is public, OAuth-protected, rate-limited (§7.3). (d) *Copyright posture*: answers synthesize and cite; the agent links out to vendor docs rather than reproducing pages.

---

## 11. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Bootstrap LLM cost blows past budget | Project stalls mid-ingestion | Phase 1 cost measurement gate; cheap extraction model; vendor-priority phasing; per-day token budget in the queue |
| Entity dedup fragments cross-vendor entities ("K8s" vs "Kubernetes" vs "OpenShift") | Communities and global answers silo by vendor | Compact ontology; chunk-context prefixing; Phase 1 dedup audit; targeted `SAME_AS`/merge fix-ups; extraction instructions with canonical names for the top ~50 workloads |
| Stale facts survive silent doc deletions | Wrong "current" answers | Expiry sweep over superseded-only-supported facts (§4.4); freshness metadata in every answer |
| Community churn between builds | Cached reports/citations invalidated, inconsistent thematic answers | Jaccard-stable community IDs; regenerate only dirty reports; `corpus_cursor` stamping |
| Graphiti API/schema evolution (fast-moving project) | Upgrade breakage | Pin versions; never restructure Graphiti-owned elements, only annotate; integration tests over the provenance chain |
| Copilot payload/latency limits truncate answers | Poor UX in Teams | Response caps; citation dedup; "answer + expandable details" tool design; measure in pilot |
| Tenant Power Platform DLP blocks the MCP connector | No user access | Engage the Power Platform admin early (Phase 0); MCP rides connector governance, so approval is an admin action, not a workaround |
| Single-group dedup slows ingestion at scale | Throughput ceiling | Entity-type filters narrow candidates; monitor add_episode latency as graph grows; Graphiti concurrency tuning |

---

## 12. Phased implementation roadmap

**Phase 0 — Foundations (≈1 week).** Provision Neo4j + GDS, queue/state store, LLM/embedding accounts; deploy skeleton graph-sync in-cluster; verify DocExtractor connectivity, key, TLS; register webhook; write the golden-question seed list; open the Power Platform/Entra conversation. *Exit:* delta feed streams end-to-end into a log; webhook deliveries verified.

**Phase 1 — Ingestion MVP on 2–3 vendors (≈2–3 weeks).** Structural layer complete (all vendors — it's free); chunker; Graphiti ingestion with v1 ontology for pilot vendors (pick ones with contrasting doc styles, e.g., AWS + Dell + Acronis ≈ 2–3k articles); update/tombstone policy; `HAS_EPISODE` provenance; hash gating; measure cost/quality, audit dedup. *Exit:* incremental deltas flow within minutes of extraction; measured $/article; dedup audit passed; go/no-go on extraction model and full-corpus budget.

**Phase 2 — Retrieval core (≈2 weeks, overlaps 1).** answer-api with local search + citation resolver + timeline; golden-set harness for local questions. *Exit:* local questions answered with correct URLs at target citation precision.

**Phase 3 — Community layer (≈2–3 weeks).** GDS hierarchical Leiden; report prompt + generation with fact-ID references; write-back; dirty-marking + scheduled refresh; global search (map-reduce). *Exit:* "compare all pilot vendors on workload X" produces a coherent, fact-cited thematic answer; nightly refresh touches only dirty communities.

**Phase 4 — DRIFT + router (≈1–2 weeks).** Primer/follow-up/synthesis loop; `/answer` router; fallback chains; full golden-set across modes. *Exit:* routing accuracy target met; drift beats both pure-local and pure-global on broad questions in side-by-side review.

**Phase 5 — Copilot exposure (≈2 weeks, can overlap 4).** MCP server wrapper; public exposure via APIM/tunnel with OAuth; Copilot Studio agent, instructions, tool descriptions, suggested prompts; pilot with a small user group in Teams. *Exit:* pilot users get cited answers in Teams; DLP/auth signed off.

**Phase 6 — Scale-out & hardening (ongoing).** Bootstrap remaining vendors in priority waves under the daily token budget; monitoring dashboards + alerts; expiry sweep + reconciliation jobs in cron; ontology v2 informed by real queries; evaluate promotion to an M365 declarative agent; revisit adaptive cards / richer citation UX.

---

## 13. Decisions to lock before Phase 1

1. **Graph DB:** Neo4j (recommended, for GDS) vs FalkorDB (+ external graspologic clustering). 
2. **LLM provider/models:** extraction model (cheap tier), report+synthesis model (strong tier), single embedding model — all fixed before bootstrap.
3. **Vendor priority order** for phased ingestion (business input).
4. **Refresh cadence & dirty threshold** for the community layer (start: nightly, 2%).
5. **Exposure route:** APIM + private link (enterprise) vs tunnel (pilot) — and who owns the Entra app registration.
6. **Ontology v1 sign-off** — the entity/edge tables in §4.2, reviewed by a domain expert against ~20 sample articles from different vendors.

---

*Prepared from the DocExtractor Client Usage Guide (2026-07-12), Graphiti/Zep documentation, and Microsoft Copilot Studio MCP documentation.*
