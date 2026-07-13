# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

**Planning stage — no code yet.** The repository currently contains two documents and nothing else:

- `graphrag-docextractor-plan.md` — the architecture & implementation plan. This is the authoritative design spec; read it before writing any code.
- `CLIENT-USAGE-GUIDE.md` — the DocExtractor REST API contract this system consumes (the upstream data source). Read it before touching ingestion.

When implementation begins, this file should be updated with real build/lint/test commands and the actual module layout.

## What this project is

A **temporal GraphRAG system** over the DocExtractor corpus (~105k articles of backup-industry vendor documentation: 40 vendors, 79 products, 280 sources). It ingests documentation incrementally, builds a temporally-aware knowledge graph, derives a thematic community-report layer, and exposes cited answers to corporate users via a Microsoft Copilot agent.

The upstream, **DocExtractor**, is a live deployment at `https://docextractor.k3s.home.lan` (private K3s cluster, internal CA `rootca.home.lan`, read-only API key auth). This project is the *downstream consumer* — it never writes to DocExtractor.

## Architecture (five layers)

The system is five services around a single Neo4j graph. Data flows A → B → C → D → E:

- **A — `graph-sync` (ingestion, FastAPI):** consumes DocExtractor's `GET /api/articles/delta` NDJSON feed (bootstrap + incremental), chunks article markdown, and drives graph writes. Triggered by the `extraction_complete` webhook (HMAC-verified) with a polling fallback.
- **B — Neo4j graph (5.x + GDS 2.x):** two sub-layers in one database.
  - **B1 structural** (Vendor→Product→Source→Chapter→Article) written with **plain Cypher, zero LLM cost**.
  - **B2 semantic** written by `graphiti-core`: Episodic + Entity nodes, `RELATES_TO` bi-temporal fact edges. Linked to B1 via `(:Article)-[:HAS_EPISODE]->(:Episodic)`.
- **C — `theme-builder` (community layer, batch):** hierarchical Leiden (Neo4j GDS) + LLM-written MS-GraphRAG-style reports over the entity graph. Derived and disposable — rebuildable from B2 at any time.
- **D — `answer-api` (retrieval, FastAPI):** four modes — `/search/local` (Graphiti hybrid), `/search/global` (community map-reduce), `/search/drift` (primer→follow-up→synthesis), `/timeline` (temporal) — behind an `/answer` router.
- **E — Copilot exposure:** `answer-api` wrapped as an **MCP server (streamable HTTP)**, exposed publicly via APIM/tunnel with OAuth, onboarded into Copilot Studio.

**Recommended stack:** Python 3.12, `graphiti-core` on Neo4j 5.x + GDS 2.x, FastAPI, Postgres or Redis for the sync cursor/state, a cheap LLM tier for extraction + a strong tier for reports/synthesis, and **one embedding model used everywhere** (entity names, facts, reports, and queries must share one embedding space — chosen once, re-embedding later is painful).

## Design decisions that constrain all implementation

These are the non-obvious invariants. Violating them breaks the system's core guarantees — check `graphrag-docextractor-plan.md` §4 for full rationale before changing any of them:

1. **Structural metadata is written deterministically, never via LLM.** Only article *content* goes through Graphiti's extraction pipeline. This is both a cost control and what makes provenance exact rather than extracted.

2. **Citation resolution is a graph traversal, never LLM generation.** The provenance chain is `fact edge → supporting episode(s) → article → source_url`. The synthesis LLM only ever emits fact/report *IDs*; a deterministic resolver expands them to URLs. **The LLM must never write a URL.** `graph-sync` is the only writer of `HAS_EPISODE` edges.

3. **Updates append, they don't overwrite (temporal policy).** On an `updated` article, add new episodes; let Graphiti invalidate contradicted facts (`invalid_at`) and mark superseded episodes rather than deleting. Deleting history defeats the whole "how did vendor X's treatment change over time?" purpose. `removed` tombstones mark episodes `removed: true` — they do **not** call `remove_episode`. A weekly Cypher-only "sweep" expires facts whose only supporting episodes are all superseded/removed (handles silent doc deletions Graphiti can't detect).

4. **One corpus-wide `group_id` (e.g. `backup-docs`).** Entity dedup happens within a group, and cross-vendor questions require `Amazon S3` from Veeam docs and AWS docs to resolve to the *same* entity node. Per-vendor groups would permanently silo cross-vendor communities. Vendor scoping at query time is done through the structural layer instead.

5. **Treat Graphiti's schema as library-owned.** Add custom properties (`superseded`, `removed`) to its nodes; never rename or restructure its elements. Link structural Vendor/Product nodes to Graphiti's extracted entities via `SAME_AS` edges rather than merging nodes.

## Sync correctness rules (from the DocExtractor contract)

The delta feed is gap-free and idempotent, but only if the consumer follows these — see `CLIENT-USAGE-GUIDE.md` §6:

- **Cursor is opaque.** Store `next_since` as-is; never decode/modify. **Advance the stored cursor only after receiving the terminal `{"control":"cursor",...}` line** — a truncated stream lacks it. On a dropped stream, resume with the same cursor (incremental) or `?bootstrap_after=<highest applied id>` (bootstrap), keeping the *original* bootstrap watermark.
- **All graph writes are idempotent upserts keyed on stable IDs** (at-least-once processing).
- **Gate on `content_hash`** (SHA-256 of served markdown, stored on `:Article`) to skip unchanged replays. Note: VLM image-enrichment runs surface as `updated` deltas with a changed hash and *should* be reprocessed.
- **Always pull with your own stored cursor**, never the webhook's `watermark` (informational only) — a missed webhook self-heals on the next pull.
- Tombstones can arrive with `run_id: null` (out-of-band vendor/product/source deletion) in batches of thousands — handle gracefully.

## Cost awareness

Full bootstrap is ~260M content tokens → ~0.8–1.5B LLM tokens through Graphiti's multi-call extraction pipeline. Use the cheap model tier for extraction, phase the bootstrap vendor-by-vendor in priority order, and meter the `add_episode` work queue against a daily token budget (incremental updates preempt bootstrap backfill). Phase 1 gates full-corpus rollout on a measured cost/quality benchmark over ~200 articles.

## Roadmap

Implementation is phased 0→6 (foundations → ingestion MVP on 2–3 vendors → retrieval core → community layer → DRIFT+router → Copilot exposure → scale-out). See `graphrag-docextractor-plan.md` §12. Six decisions are meant to be locked before Phase 1 (graph DB, models, vendor priority, refresh cadence, exposure route, ontology v1 sign-off) — §13.
