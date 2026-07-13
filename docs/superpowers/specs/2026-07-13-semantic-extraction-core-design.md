# Semantic Extraction Core — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** 2a of N — Semantic extraction core (Graphiti + local models), a quality/cost gate
**Date:** 2026-07-13
**Status:** Approved design — ready for implementation planning

Parent plan: [`../../../graphrag-docextractor-plan.md`](../../../graphrag-docextractor-plan.md) (Layer B2, §3.3–3.4, §4, §9, §12 Phase 1).
Builds on slice 1: [`2026-07-12-ingestion-foundation-design.md`](2026-07-12-ingestion-foundation-design.md) (structural layer).

---

## 1. Scope

Build the **first buildable slice of the Graphiti semantic layer**: wire `graphiti-core` to the local models, chunk pilot-source articles, run `add_episode` with a custom ontology, link the `HAS_EPISODE` provenance edge, and **measure cost + extraction quality on real data** to produce a defensible **GO/NO-GO verdict on Graphiti + `gpt-oss-20b`** (the parent plan's Phase-1 gate, §9/§12).

This is a thin vertical ("extraction core"), not the full semantic pipeline. It runs as a CLI spike over two pilot sources. It deliberately excludes the durable queue, budget metering, the temporal update policy, the staleness sweep, and reconciliation — see §10.

### Pilot corpus

| Source | id | Platform | Articles |
|---|---|---|---|
| AWS Backup — Developer Guide | `21632f3b-5a4c-4c93-9f00-6701d0e9f677` | docusaurus | 146 |
| Azure Backup — service documentation | `6da00d8b-7eee-40b7-ab67-a574465bca78` | docfx (MS Learn) | 442 |

~588 articles, median ~2k tokens/article (Azure max ~11k). AWS + Azure are both native cloud-backup offerings, chosen so shared concepts (vaults, immutability, cross-region, VM/blob/object workloads) stress-test cross-vendor **entity dedup**.

## 2. Environment (established, verified)

- **LLM (extraction):** `http://srv-llm.home.lan:8080/v1` — llama-server, model `gpt-oss-20b`, OpenAI-compatible. Verified: returns clean JSON on demand, reports `usage` incl. `cached_tokens`. A *reasoning* model (harmony format, analysis/final channels).
- **Embeddings:** `http://srv-llm.home.lan:8082/v1` (and native `/embed`) — TEI serving `jinaai/jina-embeddings-v5-text-nano-retrieval`, **768-dim**, max input 8192 tok, client batch ≤32.
- **Chunker:** `http://srv-llm.home.lan:8084` — Chonkie, `POST /v1/chunk/neural` (model `mirth/chonky_modernbert_base_1`, ModernBERT ~8k context) → `{chunks:[{text,start_index,end_index,token_count}]}`. Verified: splits on semantic boundaries, returns token counts.
- **Graph DB:** the same Neo4j 5.x as slice 1 (local Docker for the spike). Graphiti writes `:Episodic`/`:Entity`/`RELATES_TO` into it, linked to slice-1 `:Article` via `HAS_EPISODE`. No GDS this slice.
- **Community/hosted models:** available for later; reused here only as an optional stronger **LLM-judge** in the eval.
- **Language/tooling:** Python 3.12, `uv`, ruff, mypy, pytest. Async where it pays.

## 3. Architecture (Approach A: library + CLI, real per-episode `add_episode`)

New `graph_extract` package beside slice-1's `graph_sync`, run as a CLI.

```
   DocExtractor (content)      Neo4j (slice-1 structural + Graphiti semantic)
          │                                    │
   ┌──────┼────────────────────────────────────────────────────────────┐
   │ graph_extract (CLI)                                                 │
   │  ingest_driver ─► content_fetch ─► chonkie_client ─► episode_builder │
   │        │           (DocExtractor    (neural chunk    (pure: prefix + │
   │        │            by article id)   HTTP)            guards -> episodes)│
   │        │                                    │                        │
   │        └─► graphiti_client.add_episode ◄─────┘                       │
   │        └─► provenance: MERGE (:Article)-[:HAS_EPISODE]->(:Episodic)  │
   │  eval ─► dedup report + fact-quality judge + provenance + cost       │
   └─────────────────────────────────────────────────────────────────────┘
```

| Unit | Responsibility | Kind |
|---|---|---|
| `config` | LLM/embedder/chonkie base_urls + models, `group_id`, chunk ceilings, concurrency, judge model. Extends slice-1 settings. | pydantic-settings |
| `content_fetch` | `GET /api/articles/{id}` → `content_markdown` + `images[].description` + `last_updated_at`/`extracted_at`. Reuses slice-1 client. | thin I/O |
| `chonkie_client` | `POST /v1/chunk/neural` → semantic chunks. | thin I/O |
| `episode_builder` | **Pure:** `(article meta, chonkie chunks) → ordered episode bodies` with disambiguation prefix, oversize re-split, tiny-chunk merge, and `{chunk_index, heading_path, token_count, content_hash}`. | pure, unit-tested |
| `graphiti_client` | Builds one configured `Graphiti` (local LLM client, TEI embedder, Neo4j driver, entity/edge types); owns `add_episode`; wraps clients to capture token `usage` + parse-fail counts. | integration |
| `provenance` | Sole writer of `(:Article)-[:HAS_EPISODE]->(:Episodic)` with chunk metadata; idempotency gate. | neo4j driver |
| `ingest_driver` | Orchestrate list→fetch→chunk→add_episode (bounded concurrency)→link. Idempotent/resumable. | integration |
| `eval` | dedup / fact-quality / provenance / cost reports + viability verdict. | integration |
| `cli` | `probe`, `ingest [--source-id]`, `eval …`. | — |

**Key seam:** `episode_builder` is the pure, fiddly-logic core (prefixing, size guards). Everything else is a thin adapter over a real service — and the services *are* what's under test, so their verification is integration/live.

## 4. Chunking (Chonkie neural)

Chonkie owns *where* to split (learned semantic boundaries — better than heading heuristics, esp. for docfx). `episode_builder` turns spans into episodes:

1. **Long-article pre-split.** Chonkie's neural model is ModernBERT (~8k ctx); an ~11k-token Azure article may exceed it. Pre-split very long articles on top-level headings before the neural call (or confirm Chonkie windows internally at impl time).
2. **Neural chunk** each (pre-split) piece → semantic chunks with `token_count`.
3. **Prefix** each chunk with `[Product › chapter path]` (chapter path from the slice-1 `:Article`'s `IN_CHAPTER`), for cross-vendor disambiguation. Prefix is applied *after* chunking (never fed to the chunker).
4. **Guards:** oversize chunk (> `max_chunk_tokens`, ~1,800) → re-split (recursive endpoint or token-cap fallback); trivially tiny adjacent chunks (< `min_chunk_tokens`) → merge.
5. **Figures:** `images[].description` are (per the DocExtractor guide) injected as captions into `content_markdown`, so the chunker already sees them — verify at impl time and avoid double-inclusion.

Chunk-token distribution and % oversize-resplit are **eval metrics** — chunk granularity is a quality lever (finer vs coarser episodes) we measure.

## 5. Graphiti + local-model wiring

- **LLM client — grammar-constrained `OpenAIClient`** (structured outputs / json-schema `response_format`), `base_url=…:8080/v1`, `model=gpt-oss-20b`, `small_model=gpt-oss-20b`, `api_key="not-needed"`. Chosen over `OpenAIGenericClient` (JSON-mode+retry) to get guaranteed-valid JSON — the production-desirable mode — and remove parse-failures as a confound. **Gated by a Step-0 mode probe** (§7): the one real unknown is whether llama-server constrains only gpt-oss's *final* channel (leaving reasoning free) or clobbers reasoning. If the probe shows reasoning-channel conflict or quality loss, fall back to `OpenAIGenericClient` (a one-line client swap). Client swap is cheap; the probe spends ~30 min to decide on evidence.
- **Embedder — TEI/Jina, `embedding_dim=768`**, `base_url=…:8082/v1`, `embedding_model=jinaai/jina-embeddings-v5-text-nano-retrieval`. One embedding space for the whole graph (entity names, fact edges, later reports+queries). Batch ≤32. One task-mode used consistently (query/passage prefix tuning deferred).
- **Neo4j vector index dim = 768** (Graphiti `build_indices_and_constraints()`), matching TEI. Same DB as slice 1.
- **`group_id = "backup-docs"`** — single corpus-wide group (plan §4.1) so AWS/Azure entities can resolve to shared nodes; this is what makes the dedup test valid.
- **Concurrency:** `gpt-oss-20b` on one GPU → conservative `SEMAPHORE_LIMIT` (~2–4); eval reports wall-clock. Slow-but-correct is fine for a spike.
- **Version pinning:** pin exact `graphiti-core`; verify client/embedder class names + config shapes against it at impl start (API moves fast — plan §11).

## 6. Ontology v1 (locked)

Compact, closed set passed as Pydantic `entity_types` + `edge_type_map`. Graphiti evolves schema additively; adding types later is cheap, un-muddling after extraction is not.

**Entity types (7):** `Vendor`, `Product` (attr `version?`), `Workload`, `Capability`, `Platform`, `Concept`, `Requirement`.

**Edge types (mapped; unmapped → generic `RELATES_TO`):** `Product SUPPORTS Workload` · `Product PROVIDES Capability` · `Capability APPLIES_TO Workload` · `Product INTEGRATES_WITH Platform` · `Product LIMITS Workload` · `Product REQUIRES Requirement`. Concept relations fall to generic `RELATES_TO`.

**Noise suppression + canonicalization** (extraction-prompt instructions):
- Suppress doc-nav/UI junk: "this guide", "the following table", "Note", "Important", button/menu/breadcrumb labels.
- Canonical names for the top ~20 workloads/capabilities/concepts so AWS and Azure text converges: `Kubernetes` (not "K8s"), `Amazon S3` (not "S3 bucket"), `Azure Blob Storage`, `immutability` (not "WORM"/"immutable backups"), `RPO`/`RTO`/`3-2-1 rule`. This canonical list is the single biggest lever on dedup and is part of the locked design.

**Type-confusion watch:** 7 types is real surface for a 20B to mislabel (`soft delete` Concept-vs-Capability, `cross-region` Platform-vs-Capability). The eval reports per-type precision; a consistently muddled type is revisited.

## 7. Provenance chain & content flow

Per pilot article:
1. `ingest_driver` lists the source's `:Article` nodes from Neo4j (**prerequisite:** slice-1 `bootstrap` has populated both pilot sources).
2. `content_fetch` gets markdown (+ figure descriptions, `last_updated_at`/`extracted_at`).
3. Pre-split → neural chunk → `episode_builder` episodes.
4. `graphiti.add_episode(name, episode_body, source=text, reference_time, group_id, entity_types, edge_type_map)`. `reference_time` = `last_updated_at` else `extracted_at`.
5. `provenance`: `MERGE (:Article {id})-[:HAS_EPISODE]->(:Episodic {uuid})` with `{chunk_index, heading_path, token_count, content_hash}`.

**Chain:** `RELATES_TO` fact → its `episodes[]` (Graphiti stores supporting episode UUIDs on the edge) → `(:Episodic)<-[:HAS_EPISODE]-(:Article)` → `article.source_url` + title + product + heading path. Deterministic traversal, verified by an eval query.

**Invariants (from slice-1 spec):** `graph_extract` is the sole writer of `HAS_EPISODE`; Graphiti's schema is library-owned (add the edge, read `:Episodic` UUIDs, never restructure its nodes).

**Idempotency/resumability (spike-grade):** stable episode name `{article_id}:{chunk_index}:{content_hash8}`; skip a chunk whose `(article_id, chunk_index, content_hash)` already has a `HAS_EPISODE` episode — a re-run resumes rather than re-extracts (a 588-article local run takes hours and may be interrupted). First-extraction only — re-extraction of *changed* articles (the append/invalidate policy) is slice 2b.

### Step-0 mode probe (first implementation step)

Run a Graphiti-representative extraction schema through ~10 real chunks in **both** `OpenAIClient` (grammar) and `OpenAIGenericClient` (JSON-mode). Record: valid-JSON rate, whether gpt-oss reasoning survives under the grammar, extraction quality on a couple of hand-checked chunks, latency. **Lock the LLM client on this evidence.** Default is grammar `OpenAIClient` unless the probe shows the reasoning-channel conflict.

## 8. Eval & viability verdict

`eval` subcommand, five parts:
1. **Instrumented usage:** wrap Graphiti's clients to accumulate token `usage` per call-type (entity/dedup/edge/temporal/embed) + parse-fail/retry counts (structured-output reliability).
2. **Dedup report (headline):** for should-merge canon concepts (`immutability`, `cross-region copy`, `Kubernetes`, `RPO`) → node count each resolved to (1 = clean) + which vendors' episodes support each; for should-stay-distinct pairs (`Amazon S3` vs `Azure Blob Storage`, `AWS Backup` vs `Azure Backup`) → confirm no collapse; corpus totals (entities/type, edges/type, % cross-vendor entities).
3. **Fact-quality judging:** sample N `RELATES_TO` facts; score faithfulness (fact vs its supporting episode text) with a stronger hosted judge if configured else gpt-oss; **plus a mandatory human spot-check of ~20–30 facts against source URLs** (the real gate). Precision estimate.
4. **Provenance check:** walk sampled facts to `source_url`; assert the chain resolves.
5. **Cost/throughput:** tokens/article + extrapolated full-corpus (~105k), wall-clock, chunk-token distribution, % oversize re-split, effective concurrency.

**Verdict document** `docs/superpowers/slice-2a-viability.md`: GO / GO-WITH-CHANGES / NO-GO with the numbers + a slice-2b recommendation. Documented fallbacks: structured-output unreliable → grammar client → bigger model; dedup fragments → tighter canon list / `SAME_AS` fix-ups / center-node reranking; quality poor → ontology/prompt iteration, chunk-granularity lever; too slow → raise concurrency/batch, accept as background ingestion.

## 9. Testing & project layout

- **Pure unit:** `episode_builder` (prefix, oversize re-split, tiny merge, metadata) — fixtures.
- **Thin adapters:** `chonkie_client`, `content_fetch` — recorded-response (MockTransport) + live smoke.
- **Integration:** `graphiti_client` config + `provenance` vs a real Neo4j testcontainer; a `@live` end-to-end ingesting a couple of real chunks through Graphiti (hits srv-llm) asserting entities/edges/`HAS_EPISODE` + provenance resolves.
- **Idempotency:** re-run skips already-extracted chunks (integration).
- Step-0 probe + full eval are **manual report-producing scripts**, not CI.

```
src/graph_extract/
  config.py  content_fetch.py  chonkie_client.py  episode_builder.py
  graphiti_client.py  provenance.py  ingest_driver.py  eval.py  cli.py
tests/{unit,integration,e2e}/ ...
```
New deps: `graphiti-core` (pinned), `openai`. Chonkie is remote HTTP — no local dep.

## 10. Success criteria — slice is "done" when

1. Step-0 mode probe yields a client-mode decision with evidence (valid-JSON rate, reasoning-survival, latency).
2. AWS + Azure Backup articles ingest end-to-end; every episode links to its `:Article` via `HAS_EPISODE`; re-run is idempotent (skips extracted chunks).
3. Provenance chain resolves fact→episode→article→`source_url` on a sample.
4. Eval produces all four reports (dedup, fact-quality incl. human spot-check, cost/throughput, structured-output reliability).
5. Viability verdict written (GO / GO-WITH-CHANGES / NO-GO) with numbers + slice-2b recommendation.

## 11. Deferred-items ledger

Tracked so nothing set aside is lost (→ committed `docs/superpowers/slice-2-followups.md`).

**Deferred to slice 2b (semantic pipeline hardening):**
- Append/invalidate **temporal update policy** for re-extracted changed articles (Graphiti `reference_time` new-episode ingestion; mark superseded episodes; exclude from default retrieval).
- **Residual-staleness sweep** (expire facts whose only supporting episodes are superseded/removed).
- Structural↔semantic **`SAME_AS` reconciliation** (link Graphiti `Vendor`/`Product` entities to slice-1 `:Vendor`/`:Product`).
- **Durable work queue** (Redis/RQ/Celery) + dead-letter for chunks that repeatedly fail extraction.
- **Token-budget metering** + priority lanes (incremental preempts bootstrap backfill).
- **Webhook/delta-driven incremental** semantic ingestion, wired to the slice-1 sync (currently first-extraction only, run manually).

**Deferred to later phases:**
- Full-corpus rollout (40 vendors, phased, business-priority order).
- **Community-report layer** (theme-builder, hierarchical Leiden, Neo4j GDS, MS-GraphRAG-style reports).
- Retrieval / answer-api (local/global/drift/timeline + citation resolver); MCP server + Copilot Studio.

**Deferred refinements (revisit after the eval):**
- Per-chunk `heading_path` derived from markdown positions (using article chapter-path for now).
- Jina query/passage task-prefix tuning (one mode now).
- `excluded_entity_types` tuning + **ontology v2** from eval learnings (v1 already includes `Concept`/`Requirement`; further types/edges as the corpus widens).
- Concurrency/throughput tuning.

**Carried from slice 1 (still open):** see [`../slice-1-followups.md`](../slice-1-followups.md) — mypy/ruff gate + CI, real webhook cluster→host delivery verification, catalog fan-out semaphore, malformed-line dead-lettering, etc. Referenced, not duplicated.
