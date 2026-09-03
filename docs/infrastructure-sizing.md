# Architecture & Infrastructure Sizing

**Date:** 2026-07-18. Grounded in the live `backup-docs` graph (a 39-article pilot)
extrapolated to the full DocExtractor corpus (~105k articles / 40 vendors / 79
products / 280 sources).

---

## 1. The stack — five services around one Neo4j graph

Data flows **A → B → C → D → E**:

- **A · `graph-sync` (FastAPI, ingestion).** Consumes DocExtractor's
  `GET /api/articles/delta` NDJSON feed (bootstrap + incremental), chunks article
  markdown, writes the **structural** graph (plain Cypher) and drives **semantic**
  extraction. Keeps an opaque sync cursor + processing state in **Postgres**.
  Triggered by the HMAC-verified `extraction_complete` webhook with a polling
  fallback. Read-only against DocExtractor.
- **B · Neo4j graph (5.26 + GDS 2.13).** Two sub-layers, one database:
  - **B1 structural** — `Vendor→Product→Source→Article`, written deterministically
    (no LLM). Cheap, exact provenance.
  - **B2 semantic** — written by `graphiti-core`: `Episodic` (chunk) + `Entity`
    nodes, `RELATES_TO` bi-temporal fact edges (each with a 768-d embedding),
    linked to B1 via `(:Article)-[:HAS_EPISODE]->(:Episodic)`. One corpus-wide
    `group_id="backup-docs"`, one shared embedding space.
- **C · `theme-builder` (batch).** GDS hierarchical Leiden over the entity graph +
  GLM-5.2 fact-cited community reports → the `:Community` layer. **Incremental** by
  default (regenerates only changed communities); `--full` rebuilds. Derived &
  disposable.
- **D · `answer-api` (FastAPI, retrieval).** Four modes — `/search/local` (Graphiti
  hybrid), `/search/global` (community map-reduce), `/search/drift` (primer→
  follow-up→synthesis), `/timeline` (temporal) — behind the `/answer` router
  (classify → dispatch → uniform `{mode, answer, citations, routing, freshness}`).
  Citations resolve by graph traversal to source URLs; the LLM never writes a URL.
- **E · Copilot exposure (Phase 5, not built).** `answer-api` as an MCP server →
  Copilot Studio.

**External dependencies (not the graph):** the **embedder** (TEI/Jina
`jina-embeddings-v5-text-nano-retrieval`, 768-d, at `srv-llm.home.lan:8082`); a
**cheap LLM tier** (ling, OpenRouter) for extraction routing + the router
classifier; a **strong LLM tier** (GLM-5.2 via the judge endpoint) for community
reports, map-reduce, DRIFT/local synthesis, and eval judging.

## 2. The graph model (what Neo4j stores)

| Node | Role | Heavy properties |
|---|---|---|
| `Vendor/Product/Source` | structural spine (40/79/280) | names |
| `Article` | one DocExtractor article (~105k) | `id, source_url, title, content_hash` |
| `Episodic` | a markdown chunk of an article | **`content`** (the chunk text), `created_at` |
| `Entity` | an extracted concept | **`name_embedding`** (768-d), `summary` |
| `Community` | a theme (derived) | **`embedding`** (768-d), `full_report` (LLM text), `cited_fact_uuids` |

| Relationship | Role |
|---|---|
| `HAS_PRODUCT/HAS_SOURCE/HAS_ARTICLE` | structural |
| `HAS_EPISODE` | Article → Episodic (provenance) |
| `RELATES_TO` | **fact edge** (bi-temporal `valid_at`/`invalid_at`), **`fact_embedding`** (768-d) |
| `IN_COMMUNITY / PARENT_OF` | community membership + hierarchy |

The three **768-d float32 embeddings** (entity names, facts, community reports = ~3
KB each) plus **native vector + full-text indexes** over them are the dominant
storage and RAM driver. GDS needs a Leiden **projection** in heap during
`theme-build`.

## 3. Current footprint (39 articles, 2 vendors)

| | count |
|---|---|
| Articles ingested (with episodes) | **39** |
| Episodics | 348 (≈ 8.9 / article) |
| Entities (all with `name_embedding`) | 555 (≈ 14 / article) |
| `RELATES_TO` facts (all with `fact_embedding`) | 1,975 (≈ 51 / article) |
| Communities | 68 |
| All relationships | 8,574 |
| **Neo4j store on disk** | **25 MB** (≈ 0.64 MB / ingested article) |
| Container RAM (idle) | 1.25 GB |

## 4. Extrapolation to the full corpus (~105k articles)

Scaling factor from the pilot: **105,000 / 39 ≈ 2,700×**. Two adjustments matter:
**entities and communities dedup** (shared concepts recur across vendors → grow
*sublinearly*), while **episodes and facts grow ~linearly** with content.

| Element | Naive ×2700 | Realistic (with dedup) |
|---|---|---|
| Episodics | ~940k | **~0.9–1.0 M** |
| Entities | ~1.5 M | **~200–400 k** (heavy cross-vendor dedup) |
| `RELATES_TO` facts | ~5.3 M | **~3–5 M** |
| Communities | ~180 k | **~15–30 k** |
| Nodes total | | **~1.2–1.5 M** (order 10⁶ — matches the plan) |
| Relationships total | | **~5–9 M** |

**Disk** (dominated by embeddings + their vector indexes + episode content text):

| Component | Estimate |
|---|---|
| Fact embeddings (3–5 M × 3 KB) | 9–15 GB raw |
| Entity + community embeddings | ~1 GB raw |
| Vector (HNSW) indexes over the above (~1.3–1.7×) | +14–27 GB |
| Episode `content` text (~0.95 M × ~5 KB) | ~4–6 GB |
| Full-text (BM25) indexes | ~2–5 GB |
| Fact/entity properties, rel store, structural, overhead | ~8–15 GB |
| **Total** | **~40–70 GB steady; provision ~120 GB** (checkpoints/compaction transiently ~2× the working set; leave headroom) |

**RAM:** Neo4j wants the vector + full-text indexes resident in page cache for fast
retrieval, plus heap for queries and the GDS Leiden projection:

- Index working set (vectors + fulltext): **~25–40 GB** → wants to live in page cache.
- Heap: **8–16 GB** (query execution + the Leiden projection over ~200–400 k entity
  nodes during nightly `theme-build`).
- **Recommendation: 64 GB RAM** for comfortable full-corpus performance; **32 GB is
  a tight floor** (retrieval works but vector search spills from cache and
  `theme-build` gets memory-pressured). This matches the plan's §11 estimate
  (32–64 GB, single well-provisioned instance).

**CPU / disk type:**
- **NVMe/SSD is required** — vector search and Graphiti's hybrid retrieval are
  random-read heavy; spinning disk will not perform.
- Retrieval is not CPU-bound. **GDS Leiden (nightly `theme-build`) is the CPU
  hotspot**, and we pinned it to `concurrency:1` for deterministic re-detection
  (needed by incremental refresh). At ~300 k entity nodes the single-threaded
  Leiden pass will run **tens of minutes** nightly — acceptable for a batch job,
  but budget 8–16 vCPU on the box so retrieval isn't starved while it runs. (If
  nightly Leiden time becomes a problem, revisit the determinism/concurrency
  trade-off.)

> **Correction (2026-09-02).** graphiti-core 0.29.2 creates **no vector indexes** and
> scores similarity with brute-force `vector.similarity.cosine` scans in Cypher
> (`search_ops.py:148-160`). The vector-index line in the disk table above is
> therefore not what the current code produces, and at 3-5 M facts every hybrid
> search would scan every fact embedding. This is a scaling blocker for broad
> ingestion, tracked as its own slice; see
> `docs/superpowers/specs/2026-09-02-neo4j-compat-check-design.md` section 9.

## 5. Sizing is proportional — you don't have to jump to full corpus

Resources scale with **ingested** articles, not the 105k target. A phased,
priority-vendor rollout (the plan's approach) needs proportionally less:

| Ingestion | Articles | Approx. Neo4j RAM (comfortable) | Approx. disk (steady) |
|---|---|---|---|
| Pilot (today) | 39 | 2 GB | 25 MB |
| ~5 vendors | ~13 k | 8–16 GB | ~6–10 GB |
| ~15 vendors | ~40 k | 24–32 GB | ~18–28 GB |
| Full corpus | ~105 k | **64 GB** | **~40–70 GB (provision ~120 GB)** |

## 6. The real cost is LLM tokens, not the database

Neo4j at full scale is a **single mid-sized box** (64 GB / 8–16 vCPU / ~120 GB
NVMe). The dominant *operational* cost of broadening ingestion is **LLM extraction
tokens**: the plan estimates a full bootstrap at **~260 M content tokens → 0.8–1.5 B
tokens** through Graphiti's multi-call extraction pipeline. Controls already built:
the **hybrid extraction router** (cheap `ling` tier for prose, strong tier only for
dense tables — ~23× cheaper on the common case), phased vendor-by-vendor bootstrap,
and a Phase-1 cost/quality gate over ~200 articles. Plan the LLM budget and a
metered work-queue before a full bootstrap — that, not the graph DB, is what scales
the bill.

## 7. Companion infrastructure (for a production-grade stage)

- **Postgres** (graph-sync cursor/state): tiny — MBs; a small managed instance or a
  shared one is fine.
- **Embedder** (TEI/Jina, 768-d): a GPU (or a well-provisioned CPU) endpoint;
  throughput here gates bootstrap speed (every entity name, fact, report, and query
  is embedded). One shared embedding model — re-embedding later is painful.
- **LLM tiers:** the cheap tier (extraction/classify) and the strong GLM tier
  (reports/synthesis/judge). Both are external API calls today.
- **The four `answer-api` / `graph-sync` FastAPI services**: light — a couple of
  small app containers; they hold no state (state is in Neo4j/Postgres).
- For production: TLS/OAuth at the edge (Phase 5 APIM/tunnel), backups of the Neo4j
  volume (the semantic layer is expensive to rebuild — the community layer is
  cheap/disposable, the entity/fact layer is not), and the monitoring hooks (router
  decision distribution, cursor lag, citation-resolution failures) still on the C
  backlog.

## 8. Bottom line

- **Neo4j: one instance, 64 GB RAM / 8–16 vCPU / ~120 GB NVMe** for the full 105k
  corpus; scale down proportionally for a phased rollout (32 GB covers ~15 vendors).
- **NVMe mandatory; SSD-class latency is what vector search needs.**
- **The graph DB is the cheap part** — the LLM extraction budget (~1 B tokens for a
  full bootstrap) is the real number to plan, and the hybrid router + phased
  rollout are the levers.
