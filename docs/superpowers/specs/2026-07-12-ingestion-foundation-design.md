# Ingestion Foundation — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** 1 of N — Ingestion foundation (structural layer + delta sync, no LLM)
**Date:** 2026-07-12
**Status:** Approved design — ready for implementation planning

Parent plan: [`../../../graphrag-docextractor-plan.md`](../../../graphrag-docextractor-plan.md) (Layer A).
Upstream API contract: [`../../../CLIENT-USAGE-GUIDE.md`](../../../CLIENT-USAGE-GUIDE.md).

---

## 1. Scope

Build `graph-sync` — the ingestion service (plan Layer A) — limited to the **deterministic, zero-LLM** part: consume DocExtractor's delta feed and write the **B1 structural layer** (Vendor→Product→Source→Chapter→Article) into Neo4j, driven end-to-end by both the webhook push path and a polling fallback.

This is the backbone the rest of the system hangs off. It is independently valuable and fully testable without any model spend.

### Explicitly out of scope (later slices)

Graphiti episodes / `add_episode`, chunking, any LLM or embedding call, the `HAS_EPISODE` provenance edge, the community layer, retrieval, and MCP/Copilot exposure. `:NEXT` sibling edges remain deferred (reading order is derivable from `sort_order`). **Authoritative chapter hierarchy IS in scope** for this slice via the TOC-refresh pass (§5.5).

## 2. Environment (established, not assumed)

- **DocExtractor is live and reachable** from the dev machine at `https://docextractor.k3s.home.lan` (internal CA `rootca.home.lan`; clients use the CA cert or `verify=false`). Verified: `/api/health` → ok; 40 vendors, 284 sources, 59 never-extracted, 1 run active.
- **Keys available:** a read-only key for sync, and an admin key for webhook registration + triggering test extractions. Keys live in env/secret store, never committed.
- **Dev machine** is `srv-openclaw.home.lan` (172.16.255.190) on the **same `home.lan` LAN** as the cluster, so DocExtractor can POST webhooks back directly (no tunnel expected; confirmed at registration time).
- **Neo4j 5.x and Postgres 16 run locally in Docker** (compose).
- **Language/tooling:** Python 3.12, `uv`, ruff, mypy, pytest. Async throughout.

## 3. Architecture (Approach A: one service, two triggers, shared sync core)

`graph-sync` is a single FastAPI service composed of small, single-purpose, independently testable units.

```
                    DocExtractor (live)                        Postgres            Neo4j
                          │                                       │                  │
   ┌──────────────────────┼───────────────────────────────────────────────────────────┐
   │ graph-sync (FastAPI)  │                                                            │
   │  triggers ─────────►  sync_core ─────────► mapper ─────────► neo4j_repo            │
   │   • POST /webhooks/docextractor (HMAC)     (record→          (idempotent Cypher    │
   │   • background poll loop (fallback)         graph ops)        upserts)             │
   │   • CLI: bootstrap / sync-once / register-webhook                                  │
   │        └──► delta_client (NDJSON stream, cursor discipline)                        │
   │        └──► toc (per-source TOC tree fetch) ──► toc_mapper (tree→chapter ops)       │
   │        └──► state_store (cursor, bootstrap progress, webhook dedup) ──► Postgres   │
   │        └──► catalog (source→product→vendor id + metadata cache) ──► DocExtractor   │
   └────────────────────────────────────────────────────────────────────────────────────┘
```

| Unit | Single responsibility | Depends on |
|---|---|---|
| `models` | Pydantic types for delta records (content/tombstone/control) and the parsed cursor. Validation boundary. | — |
| `delta_client` | Streams `GET /api/articles/delta` as typed records; enforces cursor rules (surfaces "clean finish" only on the terminal `cursor` control line); bootstrap resume via `bootstrap_after`. Pure HTTP I/O. | httpx, models |
| `catalog` | Resolves `source_id → product_id → vendor_id` and supplies metadata absent from the delta; in-memory cache refreshed on miss and on `extraction_complete`. | httpx |
| `mapper` | **Pure function:** `(delta record, catalog snapshot) → ordered list of graph ops`. Receives an already-resolved in-memory catalog lookup and performs no I/O itself — the bulk of correctness lives here and is unit-tested with plain calls. | models |
| `toc` | Fetches `GET /api/articles/toc/{source_id}` (JSON tree). Thin HTTP I/O. | httpx |
| `toc_mapper` | **Pure function:** TOC tree → ordered chapter ops (chapter nodes keyed by `toc_entry_id`, hierarchy edges, `IN_CHAPTER` rewiring by `article_id`, prune list). No I/O — unit-tested with recorded TOC fixtures. | models |
| `neo4j_repo` | Executes graph ops (article + chapter) as idempotent `MERGE` Cypher in a transaction; owns schema/constraints; single writer of chapter structure. | neo4j driver |
| `state_store` | Cursor get/set (atomic), bootstrap progress, webhook dedup; single-flight advisory lock. | Postgres (asyncpg/SQLAlchemy) |
| `sync_core` | Orchestrates read cursor → stream → map → write → advance-on-clean-terminal; also invokes the per-source TOC-refresh pass (§5.5). Idempotent, single-flight. | all above |
| `webhook` | FastAPI route: verify HMAC, dedup, debounce, enqueue a `sync_core` pass, return 200 fast. | sync_core, state_store |
| `poll_loop` | Background task invoking `sync_core` every 30 min as the missed-webhook safety net. | sync_core |
| `app` / `cli` | FastAPI wiring + thin ops CLI. | all |

**Key seam:** `mapper` is pure and holds the interesting logic; `delta_client` / `neo4j_repo` / `state_store` / `catalog` are thin I/O adapters.

## 4. Data model

### 4.1 Neo4j — structural nodes (B1 only)

| Node | Key properties |
|---|---|
| `:Vendor` | `id` (UUID), `name`, `website` |
| `:Product` | `id`, `name`, `version`, `vendor_id` |
| `:Source` | `id`, `name`, `base_url`, `source_type`, `platform`, `last_extracted_at` |
| `:Chapter` | `id` (`toc_entry_id` UUID), `source_id`, `title`, `url`, `level`, `sort_order` |
| `:Article` | `id`, `title`, `source_url`, `topic_key`, `content_hash`, `estimated_tokens`, `sort_order`, `last_updated_at`, `run_id`, `seq`, `removed`, `removed_at` |

Relationships: `(:Vendor)-[:HAS_PRODUCT]->(:Product)-[:HAS_SOURCE]->(:Source)-[:HAS_ARTICLE]->(:Article)`; `(:Source)-[:HAS_CHAPTER]->(:Chapter)`; chapter nesting `(:Chapter)-[:HAS_CHAPTER]->(:Chapter)`; `(:Article)-[:IN_CHAPTER]->(:Chapter)` (to the article's immediate parent chapter). Chapter structure and `IN_CHAPTER` are written **only** by the TOC-refresh pass (§5.5).

Constraints/indexes: uniqueness on `Vendor/Product/Source/Article/Chapter.id`; index on `Article.source_id` and `Chapter.source_id` (reconciliation + per-source TOC rewiring/pruning). All writes are `MERGE`, so replays are idempotent.

**⚠️1 — Vendor/Product IDs are not in the delta feed.** The delta gives `vendor`/`product` as *name strings* but the authoritative `source_id`. Resolve deterministically `source_id → product_id → vendor_id` via `/api/sources`, `/api/products`, `/api/vendors` (40+79+284 rows). The `catalog` cache loads these at startup, refreshes on cache miss and on `extraction_complete`, and supplies delta-absent metadata (website, product version, base_url, platform). No per-article detail calls.

**⚠️2 — Chapters come from the TOC endpoint, not the delta.** The delta carries only title-strings (`parent_chapter`, `top_level_chapter`) — no IDs, no hierarchy, no `toc_entry_id` — so they are **not used to build chapter nodes**. Instead, authoritative chapter structure is built by the TOC-refresh pass (§5.5) from `GET /api/articles/toc/{source_id}`, which returns the full tree with `id` (`toc_entry_id`), `title`, `url`, `level`, `sort_order`, `is_article`, `article_id`, and `children[]`. Non-leaf navigation entries become `:Chapter` nodes keyed by `toc_entry_id`; each leaf's `article_id` links its `:Article` to the immediate parent chapter via `IN_CHAPTER`. This makes the delta-side article upsert simpler (it writes no chapter linkage; it still stores the article's own `sort_order`). Sibling `:NEXT` edges remain deferred — reading order is derivable from `sort_order`.

**⚠️3 — deliberately dropped in this slice:** `content_markdown` and `images` (become Graphiti episodes in slice 2; storing ~260M tokens of markdown as node properties is waste) and `extracted_at`/`created_at` (article-detail-only; not worth 105k detail calls). `content_hash` is retained for change-gating.

### 4.2 Postgres — sync state

- `sync_cursor` — global incremental watermark (opaque `next_since`); updated atomically, only on a clean terminal `cursor` line.
- `bootstrap_progress` — per shard (`vendor_id`/`source_id`/`global`): original first-attempt watermark, highest applied article id, status. Enables `bootstrap_after` resume.
- `webhook_delivery` — signature/delivery dedup + per-source debounce timestamps.
- `dead_letter` — unparseable records (raw line + context) for alerting/inspection.

**⚠️ Sharded-bootstrap cursor rule:** each sharded bootstrap returns its own `next_since`. The cursor wraps a *global* outbox `seq`, so global incremental starts from the **minimum watermark across all shards**; anything newer is re-served and absorbed by idempotent upserts.

## 5. Sync protocol & correctness

### 5.1 Bootstrap (resumable)

1. Stream `GET /api/articles/delta` (optionally `?vendor_id=`/`?source_id=` to shard). Read leading `bootstrap_start.next_since` and **persist it immediately** — this is the watermark to promote, not one a resume recomputes.
2. Apply each `added` record (structural upsert); track highest applied article `id`.
3. Stream ends **without** terminal `cursor` line → truncated. Resume with `?bootstrap_after=<highest applied id>`, **keeping the original watermark** (ignore the resumed `bootstrap_start`). Repeat until clean terminal.
4. On clean terminal, mark shard complete. When all shards complete, set global `sync_cursor` to the **min watermark across shards** and switch to incremental.
5. Run the **TOC-refresh pass (§5.5) for each source** touched during bootstrap, so chapter structure is authoritative before the slice is considered synced.

### 5.2 Incremental (steady state)

1. Read stored cursor. Stream `?since=<cursor>`.
2. Apply `added`/`updated` (upsert) and `removed` (tombstone) in order.
3. **Advance the stored cursor only when the terminal `cursor` line arrives.** Truncation → cursor unchanged → next run retries the range (idempotent). `count:0` is the no-op happy path.
4. **Never** use the webhook's `watermark` as the cursor — always the stored one.

### 5.3 Correctness invariants (each gets a test)

- **Cursor is opaque** — stored/passed verbatim; decoded only in a debug helper.
- **content_hash gating** — before an `added`/`updated` write, compare the record's `content_hash` to `:Article.content_hash` in Neo4j. Unchanged → skip. Changed → upsert props + new hash. VLM-enrichment `updated` records (changed hash) correctly pass the gate.
- **At-least-once + idempotent** — every write is a keyed `MERGE`; reprocessing is a no-op or in-place update. This is the entire consistency model.
- **Tombstones** — `removed` → `MERGE (:Article {id}) SET removed=true, removed_at=…`; **keep the node**. Handle `run_id:null` batches (out-of-band vendor/product/source deletion), potentially thousands, in-order, no special-casing.
- **Single-flight** — a Postgres advisory lock ensures one `sync_core` at a time. A nudge arriving mid-sync sets a dirty flag; the active run does one more pass rather than racing on the cursor.

### 5.4 Trigger cadence

- **Webhook** (`extraction_complete`, HMAC-verified) is the primary nudge, with a **5-min debounce** coalescing clustered per-source runs.
- **Poll loop** every **30 min** is the safety net.
- Both funnel through the single-flight `sync_core`.

`added`/`updated` are treated identically in this slice (idempotent structural upserts); the distinction only matters in slice 2 (append-episode temporal policy). `content_hash` gating is what makes them safe here.

### 5.5 TOC-refresh pass (authoritative chapters)

Chapter structure is derived from the source's navigation tree, not the per-article delta, so it is refreshed **per source** rather than per record.

**When it runs:**
- After bootstrap of a source (§5.1 step 5).
- On `extraction_complete` for a source — the webhook payload names the `source_id`, and page add/remove reshapes the TOC — so the pass runs for exactly that source after the article deltas are applied.
- (Belt-and-suspenders) a periodic full refresh across all sources on the 30-min poll cadence is **not** required; a per-source refresh keyed off extraction is sufficient. A manual `refresh-toc <source_id>` CLI exists for repair.

**What it does** (per source, single Neo4j transaction):
1. `toc` fetches `GET /api/articles/toc/{source_id}`.
2. `toc_mapper` (pure) flattens the tree → ordered chapter ops: `MERGE (:Chapter {id: toc_entry_id})` with `title/url/level/sort_order/source_id`; `(:Source)-[:HAS_CHAPTER]->(:Chapter)` for roots and `(:Chapter)-[:HAS_CHAPTER]->(:Chapter)` for nesting; `(:Article {id: article_id})-[:IN_CHAPTER]->(:Chapter)` for each leaf's immediate parent.
3. **Rewire + prune:** `IN_CHAPTER` for the source's articles is replaced to match the current tree, and `:Chapter` nodes for that `source_id` that are absent from the current tree are detached and removed. Chapters are structural navigation — safe to rebuild from the authoritative tree each pass (unlike articles, which are tombstoned, never deleted).

**Ordering vs. article ingestion:** `IN_CHAPTER` links by `article_id`; if the TOC references an article not yet upserted (rare timing skew), the `MERGE (:Article {id})` creates a stub that the article delta later fills in — no lost edge. The pass therefore runs *after* the delta batch for the source, but is self-healing if ordering slips.

## 6. Webhook path (end-to-end)

**Registration** (one-time CLI `register-webhook`, admin key): `POST /api/webhooks` with `events:"extraction_complete"`, our endpoint URL, and a generated HMAC `secret` stored in Postgres/env. Endpoint: `http(s)://srv-openclaw.home.lan:<port>/webhooks/docextractor`. If DocExtractor enforces HTTPS/internal-CA trust on delivery, terminate TLS with a cert from that CA (or fall back to a tunnel); the accepted scheme is confirmed at registration time.

**Receipt** (`POST /webhooks/docextractor`):
1. Compute `sha256=HMAC(secret, raw_body)`, `compare_digest` vs `X-DocExtractor-Signature`. Mismatch → 401, no work.
2. Dedup on delivery via `webhook_delivery` (DocExtractor retries 3× with backoff).
3. Debounce within the 5-min window per source.
4. Enqueue a single-flight `sync_core` pass; return **200 fast** and pull asynchronously.

Validated live: trigger an extraction (admin key) on a small source, watch the signed POST arrive and drive a sync.

## 7. Error handling

| Failure | Behavior |
|---|---|
| Stream drops mid-bootstrap | Resume via `bootstrap_after`, original watermark kept. |
| Stream drops mid-incremental | Cursor not advanced; next run retries; idempotent upserts absorb overlap. |
| Neo4j write fails on a record | Fail the whole batch → cursor stays put → full retry. Never advance past an unpersisted record. |
| DocExtractor 5xx / timeout | Exponential backoff with cap; poll loop is the backstop. |
| DocExtractor 401 (bad/expired key) | Fail loudly, alertable; no silent partial sync. |
| Catalog cache miss (unknown `source_id`) | Refresh catalog once; if still unknown, write the article with `source_id` + `catalog_incomplete=true` and log — never drop the record. |
| Malformed/unparseable NDJSON line | Route to `dead_letter` with raw line + context; continue the stream but **fail the batch** (do not advance cursor past a dropped line). Alert on dead-letter growth. |
| TOC fetch fails / TOC-refresh errors for a source | Article deltas already applied and cursor advancement are **independent** of the TOC pass — a failed TOC refresh is logged and retried (next extraction or manual `refresh-toc`), never blocks or reverts article sync. Chapter structure is eventually-consistent; articles are not. |
| Webhook HMAC fail | 401, dropped, logged. |
| Duplicate webhook delivery | Deduped, 200, no work. |
| Two syncs race (webhook + poll) | Single-flight lock; later one sets dirty flag; active run does one more pass. |

Guiding rule: **the cursor advances only on fully-applied, cleanly-terminated batches.** Any doubt → don't advance → retry. Correctness over throughput.

**Observability (lightweight):** structured logs + a `/status` endpoint exposing cursor lag (time since last clean advance), last-run outcome, per-shard bootstrap progress, dead-letter count, and the reconciliation signal — `:Article` counts per source vs `GET /api/dashboard/sources` (drift = missed deltas). Plus `/health`.

## 8. Testing strategy (TDD)

The pure `mapper` seam keeps most correctness infrastructure-free.

- **Unit (pure, the bulk):** `mapper` record→graph-ops for every case (added, updated, tombstone, `run_id:null`, unknown source); `toc_mapper` tree→chapter-ops (nesting, leaf→article `IN_CHAPTER`, prune list, article-stub-on-missing); `delta_client` cursor discipline (terminal detection, truncation, bootstrap resume with retained watermark); HMAC verify/reject; content_hash gate skip/pass; min-watermark computation. Fed by **recorded NDJSON + TOC-JSON fixtures** captured from the live feed and checked in.
- **Integration (testcontainers):** `neo4j_repo` idempotency (apply a batch twice → identical graph), constraints, tombstone-keeps-node, TOC-refresh rewire+prune (chapter removed from tree → detached; re-running the pass is a no-op); `state_store` atomic cursor advance + single-flight lock; webhook dedup.
- **End-to-end (`@live`, opt-in):** bootstrap AWS Backup (146 articles) → 146 `:Article` nodes wired to correct Source/Product/Vendor, and chapter hierarchy matching the live TOC (spot-check known nesting, e.g. *Backup vaults › Vault Lock*); reconcile vs `/api/dashboard/sources`; register webhook, trigger extraction (admin key), assert signed POST arrives and drives sync + TOC refresh. Excluded from the default CI lane.

## 9. Project layout

```
graph-rag/
  pyproject.toml            # uv; ruff + mypy + pytest
  docker-compose.yml        # neo4j 5.x + postgres 16
  .env.example              # keys, URLs, secrets (never real values)
  src/graph_sync/
    models.py  delta_client.py  catalog.py  mapper.py
    toc.py  toc_mapper.py
    neo4j_repo.py  state_store.py  sync_core.py
    webhook.py  poll_loop.py  app.py  cli.py  config.py
  tests/
    unit/  integration/  e2e/  fixtures/*.ndjson
  docs/superpowers/specs/2026-07-12-ingestion-foundation-design.md
```

Config via `pydantic-settings` (env). Async throughout (`httpx.AsyncClient`, async Neo4j driver, asyncpg/SQLAlchemy). Default test lane: `pytest -m "not live"`.

## 10. Success criteria — slice is "done" when

1. Bootstrap ingests a chosen source/vendor into the structural graph; node/edge counts reconcile against DocExtractor's dashboard, and chapter hierarchy (from the TOC pass) matches the live TOC tree for the source.
2. Incremental sync applies added/updated/removed correctly; re-running with the same cursor is a clean no-op (`count:0`), proving idempotency.
3. Bootstrap resume works after a forced mid-stream drop, with no missed or duplicated articles.
4. A real `extraction_complete` webhook is received, HMAC-verified, deduped, and drives a sync end-to-end.
5. `content_hash` gating measurably skips unchanged records on replay.
6. TOC-refresh is authoritative and self-correcting: re-running the pass is a no-op, and a chapter removed from the rebuilt tree is detached from the graph (no orphan `:Chapter`).
7. Full unit + integration suite green; `@live` E2E passes against the cluster.
