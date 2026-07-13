# DocExtractor — Downstream Client Usage Guide

> **Audience:** Engineers building a downstream consumer (e.g. a GraphRAG pipeline)
> that pulls documentation data from DocExtractor's REST API.
>
> **Deployment referenced:** live K3s cluster at
> `https://docextractor.k3s.home.lan` — 40 vendors, 79 products, 280 sources,
> ~105,000 articles (as of 2026-07-12).

---

## Table of Contents

1. [What DocExtractor Gives You](#1-what-docextractor-gives-you)
2. [Connection & Authentication](#2-connection--authentication)
3. [Data Hierarchy: Vendors → Products → Sources → Articles](#3-data-hierarchy-vendors--products--sources--articles)
4. [Reading Articles](#4-reading-articles)
5. [Table of Contents](#5-table-of-contents)
6. [The Delta Feed — Bootstrap + Incremental Sync](#6-the-delta-feed--bootstrap--incremental-sync)
7. [Webhook Integration](#7-webhook-integration)
8. [Images & VLM Descriptions](#8-images--vlm-descriptions)
9. [Extraction Runs](#9-extraction-runs)
10. [Export](#10-export)
11. [Dashboard & Monitoring](#11-dashboard--monitoring)
12. [Reference: Live Deployment Sample Data](#12-reference-live-deployment-sample-data)
13. [Client Implementation Patterns](#13-client-implementation-patterns)

---

## 1. What DocExtractor Gives You

DocExtractor scrapes product documentation from vendor websites (20+ platform
profiles: Docusaurus, MkDocs, Sphinx, Zendesk, Confluence, Zoomin, and more),
stores articles as Markdown with full metadata in PostgreSQL, and exposes a
REST API for downstream consumption.

For a GraphRAG pipeline, the key resources are:

| What | How | Notes |
|------|-----|------|
| Full corpus snapshot | `GET /api/articles/delta` (no `since`) | Every current, visible, non-removed article |
| Incremental changes | `GET /api/articles/delta?since=<cursor>` | Added / updated / removed since last pull |
| Single article | `GET /api/articles/{id}` | Full markdown + images + chapter metadata |
| TOC structure | `GET /api/articles/toc/{source_id}` | Hierarchical navigation tree with article IDs |
| Search | `GET /api/articles?q=...` | Full-text search with facets |
| Change notifications | `POST /api/webhooks` + `extraction_complete` | Pull nudge — not the data channel |
| Health | `GET /api/health` | No auth required |

---

## 2. Connection & Authentication

### Base URL

```
https://docextractor.k3s.home.lan
```

All API endpoints are under `/api/*`. Swagger UI is at `/docs`.

### Auth

Auth is **enabled** on this deployment. Every `/api/` request must include one of:

```
X-API-Key: dxk_...        # API key (recommended for service-to-service)
Authorization: Bearer ...  # JWT (for interactive sessions)
```

For a downstream indexer, use a **read-only API key**. The key used in this
guide is:

```
dxk_XX…PUfE   (read_only role)
```

### RBAC

| Method | Minimum Role |
|--------|-------------|
| GET, HEAD | `read_only` |
| POST, PUT, PATCH, DELETE | `read_write` |
| Admin endpoints (users, jobs, auth realms) | `admin` |

Read-only keys can list, search, read articles, pull the delta feed, browse
TOCs, and list extraction runs — everything a GraphRAG consumer needs.

### Quick health check

```bash
curl -sk https://docextractor.k3s.home.lan/api/health
# {"status":"ok","version":"0.1.0"}
```

### Verify your API key

```bash
curl -sk -H "X-API-Key: dxk_XX…PUfE" \
  https://docextractor.k3s.home.lan/api/auth/status
# {"auth_enabled":true,"needs_bootstrap":false}
```

> **TLS note:** The cluster uses an internal CA (`rootca.home.lan`). External
> clients need either the CA cert in their trust store or `--insecure` /
> `verify=false` in the HTTP client. The samples below use `-k` for brevity.

---

## 3. Data Hierarchy: Vendors → Products → Sources → Articles

```
Vendor (e.g. "AWS")
  └── Product (e.g. "AWS Backup")
        └── Source (e.g. "Developer Guide" — a doc site or PDF)
              ├── ExtractionRun (each scrape pass)
              ├── TOCEntry tree (navigation structure)
              └── Article (one per page, with markdown + images)
                    └── ArticleVersion (historical snapshots)
```

### List vendors

```bash
curl -sk -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/vendors?limit=10"
```

**Response (live):**

```json
{
  "vendors": [
    {"id": "128b5dee-...", "name": "AWS",       "website": "https://aws.amazon.com",  "created_at": "2026-06-25T21:59:04Z"},
    {"id": "4d13da93-...", "name": "Acronis",   "website": "https://acronis.com",     "created_at": "2026-06-21T17:07:05Z"},
    {"id": "62d129d0-...", "name": "Afi.ai",    "website": "https://afi.ai",          "created_at": "2026-06-21T17:16:40Z"},
    {"id": "af9ab075-...", "name": "Arcserve",  "website": "https://arcserve.com",    "created_at": "2026-06-19T12:38:07Z"},
    {"id": "c791aaff-...", "name": "AvePoint",  "website": "https://www.avepoint.com","created_at": "2026-06-28T17:51:06Z"}
    // ... 40 total
  ],
  "total": 40
}
```

### List products (optionally filtered by vendor)

```bash
curl -sk -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/products?limit=10"
```

**Response (live):**

```json
{
  "products": [
    {"id": "5afa4158-...", "vendor_id": "128b5dee-...", "name": "AWS Backup",      "version": null},
    {"id": "8fe3cce5-...", "vendor_id": "af9ab075-...", "name": "Arcserve Backup",  "version": "19.0"},
    {"id": "f9895808-...", "vendor_id": "af9ab075-...", "name": "Arcserve UDP",     "version": "10.0"},
    {"id": "4df43db1-...", "vendor_id": "fcaacf7a-...", "name": "Avamar",           "version": "19.12"}
    // ... 79 total
  ],
  "total": 79
}
```

### List sources

```bash
curl -sk -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/sources?limit=5"
```

**Response (live):**

```json
{
  "sources": [
    {
      "id": "21632f3b-5a4c-4c93-9f00-6701d0e9f677",
      "product_id": "5afa4158-7413-46c4-9cc9-5b1b4bbe4377",
      "source_type": "web",
      "name": "Developer Guide",
      "base_url": "https://docs.aws.amazon.com/aws-backup/latest/devguide/",
      "status": "completed",
      "platform": "docusaurus",
      "last_extracted_at": "2026-07-12T16:00:53Z",
      "article_count": null
    },
    {
      "id": "26def9f7-a9d9-44cb-a4f1-6b5e77c60335",
      "product_id": "418c07bc-...",
      "source_type": "web",
      "name": "Cyber Protect Cloud Console",
      "base_url": "https://www.acronis.com/...",
      "status": "completed",
      "platform": "zoomin",
      "last_extracted_at": "2026-07-11T20:21:19Z"
    }
    // ... 280 total
  ]
}
```

**Key source fields:**

| Field | Description |
|-------|-------------|
| `id` | UUID — use this to filter articles, TOC, delta, and extraction runs |
| `product_id` | Parent product |
| `source_type` | `"web"` or `"pdf"` |
| `name` | Human-readable source name |
| `base_url` | Root URL being scraped |
| `url_template` | Optional — contains `{version}` placeholder for versioned docs |
| `status` | `completed`, `running`, `pending`, `failed`, `paused` |
| `platform` | Detected platform profile (`docusaurus`, `zoomin`, `mkdocs`, etc.) |
| `last_extracted_at` | Timestamp of last successful extraction |

---

## 4. Reading Articles

### Search/list articles

```bash
curl -sk -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/articles?source_id=21632f3b-...&limit=3"
```

**Response (live — AWS Backup Developer Guide):**

```json
{
  "articles": [
    {
      "id": "2c92266f-f84c-49c2-9e89-3b4376ec9043",
      "source_id": "21632f3b-5a4c-4c93-9f00-6701d0e9f677",
      "toc_entry_id": "6d93201e-6728-48cc-9f94-4cba18b5d2a5",
      "title": "What is AWS Backup?",
      "source_url": "https://docs.aws.amazon.com/aws-backup/latest/devguide/whatisbackup.html",
      "last_updated_at": null,
      "sort_order": 0,
      "estimated_tokens": 4075,
      "content_size_bytes": 16308,
      "created_at": "2026-06-25T22:00:00Z",
      "extracted_at": "2026-07-12T16:00:17Z",
      "search_rank": null,
      "change_status": null
    }
    // ... 146 total in this source
  ],
  "total": 146,
  "next_cursor": "eyJpZCI6IjQ0NDhhYzEwLTM4ZmUtNDQ2ZS04MWJiLWIwZTQzOWI2Y2FhMyIsIm8iOjJ9",
  "has_more": true,
  "limit": 3,
  "facets": null
}
```

**Query parameters:**

| Param | Description |
|-------|-------------|
| `source_id` | Filter to one source |
| `q` | Full-text search (title + content, GIN-indexed) |
| `from` / `to` | Date range on `extracted_at` (ISO datetime) |
| `status` | Change status vs. latest run: `new`, `updated`, `unchanged` |
| `cursor` | Opaque cursor for forward pagination (echo back `next_cursor`) |
| `limit` | Page size (1–200, default 50) |

When `q` is set, results are ranked by `ts_rank` and facets include per-status
and per-month counts.

### Get a single article (full content)

```bash
curl -sk -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/articles/2c92266f-f84c-49c2-9e89-3b4376ec9043"
```

**Response (live — truncated for display):**

```json
{
  "id": "2c92266f-f84c-49c2-9e89-3b4376ec9043",
  "source_id": "21632f3b-5a4c-4c93-9f00-6701d0e9f677",
  "toc_entry_id": "6d93201e-6728-48cc-9f94-4cba18b5d2a5",
  "title": "What is AWS Backup?",
  "source_url": "https://docs.aws.amazon.com/aws-backup/latest/devguide/whatisbackup.html",
  "last_updated_at": null,
  "sort_order": 0,
  "estimated_tokens": 4075,
  "content_size_bytes": 16308,
  "created_at": "2026-06-25T22:00:00Z",
  "extracted_at": "2026-07-12T16:00:17Z",
  "content_markdown": "# What is AWS Backup?\n\nAWS Backup is a fully-managed service...",
  "images": [],
  "vendor": {"id": "128b5dee-...", "name": "AWS"},
  "product": {"id": "5afa4158-...", "name": "AWS Backup"},
  "parent_chapter": null,
  "top_level_chapter": {"id": "6d93201e-...", "title": "What is AWS Backup?"}
}
```

**Key article fields for GraphRAG:**

| Field | GraphRAG Use |
|-------|-------------|
| `id` | Stable node identifier (UUID) |
| `topic_key` | Natural key — the canonical source URL (with `{version}` for versioned sources) |
| `source_url` | Resolved URL for this specific version |
| `content_markdown` | The document body — chunk this for your vector store |
| `content_hash` | SHA-256 of served markdown — dedup / change detection |
| `estimated_tokens` | Pre-computed token count for budgeting chunk splits |
| `vendor` / `product` | Entity nodes in the graph |
| `parent_chapter` / `top_level_chapter` | Chapter hierarchy for structural relationships |
| `sort_order` | Reading order within the source |
| `images[].description` | VLM-generated text description of images (see [§8](#8-images--vlm-descriptions)) |

### Article versions (history)

Each extraction run that detected a content change creates an
`ArticleVersion` snapshot of the previous content:

```bash
# List versions
curl -sk -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/articles/{id}/versions"

# Get a specific version's content
curl -sk -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/articles/{id}/versions/{version_id}"

# Diff a version against the next or current content
curl -sk -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/articles/{id}/versions/{version_id}/diff?against=current"
```

---

## 5. Table of Contents

The TOC endpoint returns the navigational tree for a source, with article IDs
linked to leaf entries:

```bash
curl -sk -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/articles/toc/21632f3b-..."
```

**Response (live — abbreviated):**

```
- What is AWS Backup?              → article:2c92266f-...  (level=0, sort=0)
  - AWS Backup feature availability → article:85ca0827-... (level=1, sort=1)
- How it works                     → article:4448ac10-...  (level=0, sort=2)
  - Working with supported AWS services → article:a5cfa426-... (level=1)
- Getting started                  → article:f1932f08-...  (level=0, sort=6)
- Backup plans                     → article:014e2e10-...  (level=0, sort=7)
  - Create a backup plan           → article:d42e9c68-...  (level=1, sort=8)
    - Backup plan options          → article:554ebdbf-...  (level=2, sort=9)
    - Delete a backup plan         → article:39daf856-...  (level=2, sort=11)
- Backup vaults                    → article:6e1dc8cb-...  (level=0, sort=18)
  - Vault access policies          → article:011dc3fa-...  (level=1, sort=26)
  - Vault Lock                     → article:dad69d7c-...  (level=1, sort=27)
```

Each entry has: `id`, `title`, `url`, `level`, `sort_order`, `is_article`,
`article_id` (null for non-leaf navigation nodes), and `children[]`.

For a GraphRAG pipeline, the TOC provides:
- **Structural edges** (parent → child, siblings by `sort_order`)
- **Chapter membership** for each article (via `toc_entry_id` on the article)
- **Hierarchy depth** (`level`)

---

## 6. The Delta Feed — Bootstrap + Incremental Sync

> **This is the primary integration point for a downstream GraphRAG pipeline.**

`GET /api/articles/delta` streams article changes as **JSON Lines**
(`application/x-ndjson`) — one JSON object per line. It is the reliable,
pull-based data channel. The `extraction_complete` webhook (see [§7](#7-webhook-integration))
is the "wake up and pull" nudge.

### Endpoint

```
GET /api/articles/delta
```

**Query parameters:**

| Param | Description |
|-------|-------------|
| `since` | Opaque cursor from a prior pull's control record. **Omit for full bootstrap.** |
| `source_id` | Optional — restrict to one source |
| `vendor_id` | Optional — restrict to one vendor (sharding) |
| `bootstrap_after` | Optional, **bootstrap only** — resume a dropped snapshot after this article ID (ignored when `since` is set) |

### Two modes

| Mode | Trigger | What you get |
|------|---------|-------------|
| **Bootstrap** | `since` omitted | Every current, visible, non-removed article as `change_type: "added"` (with `seq: null`) |
| **Incremental** | `since` present | `added` / `updated` content records + `removed` tombstones since that watermark |

### Control records

The feed always bookends with control records:

**Bootstrap — first line:**
```json
{"control":"bootstrap_start","next_since":"eyJzZXEiOjExNTQ0LCJ2IjoxfQ=="}
```

**Terminal line (both modes):**
```json
{"control":"cursor","next_since":"eyJzZXEiOjExNTQ0LCJ2IjoxfQ==","count":146}
```

> **Critical:** A truncated stream lacks the terminal control record. Only
> advance your stored cursor when you receive a clean `cursor` line. If the
> stream drops, retry with the same cursor (incremental) or `bootstrap_after`
> (bootstrap).

### Record shapes

#### Content record (`added` / `updated`)

```json
{
  "seq": 4811,
  "change_type": "updated",
  "id": "57e94224-4561-4320-8d77-263200ca7716",
  "topic_key": "https://www.dell.com/support/manuals/.../preface?guid=...",
  "source_id": "29408b48-1665-4b85-8f46-d40935417786",
  "vendor": "Dell",
  "product": "PowerProtect Data Manager",
  "title": "Preface",
  "source_url": "https://www.dell.com/support/manuals/.../preface?guid=...",
  "last_updated_at": null,
  "content_hash": "a2c83de484d8a174db246e3d21f6cb2ae7afdbd27b360cd8a54a5810b8e2da8a",
  "estimated_tokens": 3750,
  "parent_chapter": null,
  "top_level_chapter": "Preface",
  "sort_order": 1,
  "run_id": "d760391a-2976-4b80-b88f-2473febfd8f8",
  "content_markdown": "# Preface\n\nAs part of an effort to improve...",
  "images": [
    {
      "url": "/media/57e94224-.../3359e18c2790.png",
      "alt": "QR code",
      "description": "A QR code, a square matrix barcode composed of black modules on a white background, with three large square alignment patterns in the corners.",
      "kind": "other"
    }
  ]
}
```

> In **bootstrap** mode, `seq` is `null` (snapshot rows are not change events).

#### Tombstone record (`removed`)

```json
{
  "seq": 327,
  "change_type": "removed",
  "id": "93c0c79f-2cd6-4d2d-9dc5-348c952dc9b5",
  "topic_key": "https://www.dell.com/support/manuals/.../edit-or-delete-a-protection-rule?guid=...",
  "source_id": "8f39da35-419a-4a97-b56b-e8b061bf88be",
  "removed_at": "2026-07-11T18:41:55.170422Z",
  "run_id": "3de0b5a2-b595-4203-bbf9-256754429ef1"
}
```

**Tombstones carry the original article `id`** even after the article row is
hard-deleted — the `article_id` column in the outbox is a plain UUID with no
foreign key, so a later hard delete never nulls it. Process `removed` records
by dropping the corresponding node from your graph.

> **Out-of-band deletions** (removing a source, product, or vendor in the UI)
> emit `removed` tombstones for every one of its live articles. These
> tombstones have `"run_id": null` since they're not part of an extraction run.

### The cursor

The `next_since` cursor is an opaque, base64-encoded JSON object:

```
eyJzZXEiOjExNTQ0LCJ2IjoxfQ==  →  {"seq":11544,"v":1}
```

It wraps the `content_changes.id` watermark (a BIGSERIAL). Don't decode or
modify it — store it as-is and pass it back as `?since=<cursor>`.

### Gap-free ordering guarantee

The delta feed is **gap-free even when extraction runs overlap**. Here's how:

1. Every mutation (add/update/remove) appends a row to the `content_changes`
   outbox **in the same transaction** as the article change.
2. Each extraction run commits a `run_start` sentinel row before processing
   any article, creating a committed floor in `id` space.
3. The feed serves only rows below a **"safe ceiling"** — the lowest `id`
   belonging to any still-active run. This means every served row provably
   belongs to a terminal (committed or rolled-back) run.
4. Sentinels (`change_type: "run_start"`) are skipped by the stream but
   `last` advances past them — they're permanent and below the ceiling.

This means a consumer never misses a change, even if two extraction runs
overlap across the same article set.

### `content_hash` — the served-content fingerprint

Each record's `content_hash` is the **SHA-256 of the served
`content_markdown`** — not the internal raw-scrape fingerprint. This means:

- It changes whenever the served content changes (including when image
  descriptions are injected as captions by a later enrichment run).
- Consumers can safely de-dup and change-detect on it.
- Enrichment updates surface as `updated` deltas with a changed hash —
  re-index just those.

---

### Full bootstrap — live sample

```bash
curl -sN -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/articles/delta?source_id=21632f3b-..." \
  > snapshot.ndjson
```

**First line (control):**
```json
{"control":"bootstrap_start","next_since":"eyJzZXEiOjExNTQ0LCJ2IjoxfQ=="}
```

**Sample data records (truncated):**
```json
{"seq":null,"change_type":"added","id":"010d05f5-33b0-473f-b725-e671d3e16d37","topic_key":"https://docs.aws.amazon.com/aws-backup/latest/devguide/s3-backups.html","source_id":"21632f3b-...","vendor":"AWS","product":"AWS Backup","title":"Amazon S3 backups","source_url":"https://docs.aws.amazon.com/.../s3-backups.html","last_updated_at":null,"content_hash":"8f291e32...","estimated_tokens":7625,"parent_chapter":"Backup creation by resource type","top_level_chapter":"Backup creation, maintenance, and restore","sort_order":47,"run_id":"4e11cfe7-...","content_markdown":"Amazon S3 backups\n=================\n\nOverview\n...","images":[]}

{"seq":null,"change_type":"added","id":"011dc3fa-62ea-4832-8588-e7b815e8a380","topic_key":"https://docs.aws.amazon.com/aws-backup/latest/devguide/create-a-vault-access-policy.html","source_id":"21632f3b-...","vendor":"AWS","product":"AWS Backup","title":"Vault access policies","source_url":"https://...","content_hash":"c27a8bd6...","estimated_tokens":1688,"parent_chapter":"Backup vaults","top_level_chapter":"Backup vaults","sort_order":26,"run_id":"599e19e1-...","content_markdown":"Vault access policies\n=====\n...","images":[]}

{"seq":null,"change_type":"added","id":"014e2e10-b57a-4f90-8dbc-538daa377fe5","topic_key":"https://docs.aws.amazon.com/aws-backup/latest/devguide/about-backup-plans.html","source_id":"21632f3b-...","vendor":"AWS","product":"AWS Backup","title":"Backup plans","source_url":"https://...","content_hash":"e6add4cf...","estimated_tokens":452,"parent_chapter":null,"top_level_chapter":"Backup plans","sort_order":7,"run_id":"599e19e1-...","content_markdown":"Backup plans\n====\n...","images":[]}
```

**Last line (control):**
```json
{"control":"cursor","next_since":"eyJzZXEiOjExNTQ0LCJ2IjoxfQ==","count":146}
```

This source (AWS Backup Developer Guide) has 146 articles. The full corpus
across all 280 sources is ~105,000 articles.

---

### Incremental delta — live sample

```bash
CURSOR="eyJzZXEiOjExNTQ0LCJ2IjoxfQ=="

curl -sN -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/articles/delta?since=$CURSOR"
```

**When there are no changes since the cursor:**
```json
{"control":"cursor","next_since":"eyJzZXEiOjExNTQ0LCJ2IjoxfQ==","count":0}
```

**When there are changes (using an older cursor `since=seq:1`):**

```json
{"seq":2,"change_type":"updated","id":"57e94224-4561-4320-8d77-263200ca7716","topic_key":"https://www.dell.com/.../preface?guid=...","source_id":"29408b48-...","vendor":"Dell","product":"PowerProtect Data Manager","title":"Preface","source_url":"https://www.dell.com/.../preface?guid=...","last_updated_at":null,"content_hash":"a2c83de4...","estimated_tokens":3750,"parent_chapter":null,"top_level_chapter":"Preface","sort_order":1,"run_id":"d760391a-...","content_markdown":"# Preface\n...","images":[{"url":"/media/57e94224-.../3359e18c2790.png","alt":"QR code","description":"A QR code, a square matrix barcode composed of black modules on a white background, with three large square alignment patterns in the corners.","kind":"other"}]}

{"seq":3,"change_type":"updated","id":"6b48b506-8c58-4736-a2ad-ae88c05343f5","topic_key":"https://www.dell.com/.../supported-internet-protocol-versions?guid=...","source_id":"29408b48-...","vendor":"Dell","product":"PowerProtect Data Manager","title":"Supported Internet Protocol versions","source_url":"https://...","content_hash":"27bf3fa8...","estimated_tokens":2105,"parent_chapter":"Getting started","top_level_chapter":"Getting started","sort_order":4,"run_id":"d760391a-...","content_markdown":"# Supported Internet Protocol versions\n...","images":[]}

{"seq":4,"change_type":"updated","id":"b51bdc39-a397-4516-9eca-33723de5d7e6","topic_key":"https://www.dell.com/.../license-types?guid=...","source_id":"29408b48-...","vendor":"Dell","product":"PowerProtect Data Manager","title":"License types","source_url":"https://...","content_hash":"9edbfb5b...","estimated_tokens":530,"parent_chapter":"License considerations","top_level_chapter":"System maintenance","sort_order":45,"run_id":"d760391a-...","content_markdown":"# License types\n...","images":[]}
```

**Tombstones from the same stream:**

```json
{"seq":327,"change_type":"removed","id":"93c0c79f-2cd6-4d2d-9dc5-348c952dc9b5","topic_key":"https://www.dell.com/.../edit-or-delete-a-protection-rule?guid=...","source_id":"8f39da35-...","removed_at":"2026-07-11T18:41:55.170422Z","run_id":"3de0b5a2-..."}

{"seq":328,"change_type":"removed","id":"5b32f85d-0d67-4e86-ac61-e8ad138176ca","topic_key":"https://www.dell.com/.../ppdmk8sdiag-diagnostics-tool?guid=...","source_id":"8f39da35-...","removed_at":"2026-07-11T18:41:55.170422Z","run_id":"3de0b5a2-..."}

{"seq":329,"change_type":"removed","id":"f8c76242-fb4f-43c3-b74a-3a347727857a","topic_key":"https://www.dell.com/.../restore-considerations-for-openshift-environments?guid=...","source_id":"8f39da35-...","removed_at":"2026-07-11T18:41:55.170422Z","run_id":"3de0b5a2-..."}
```

---

### Resuming a dropped bootstrap

For large corpora, the bootstrap snapshot can be thousands of records. If the
connection drops mid-stream, resume instead of restarting:

1. On the **first** attempt, read `next_since` from the leading
   `bootstrap_start` line and store it immediately. Apply `added` records,
   tracking the highest article `id` you've applied.
2. If the stream ends **without** the terminal `cursor` line, resume:
   `GET /api/articles/delta?bootstrap_after=<highest id applied>`.
   **Keep your originally-stored `next_since`** — ignore the resumed stream's
   own `bootstrap_start` value (it's recomputed at resume time).
3. Repeat until you receive the terminal `cursor` line. Only then begin
   incremental pulls with `?since=<stored next_since>`.

```bash
# First attempt
curl -sN -H "X-API-Key: $KEY" \
  "$URL/api/articles/delta" > snapshot.ndjson

# If dropped — extract the watermark from the FIRST line:
CURSOR=$(head -1 snapshot.ndjson | jq -r 'select(.control=="bootstrap_start").next_since')
# And the last article id you applied:
LAST_ID=$(tail -1 snapshot.ndjson | jq -r 'select(.change_type=="added").id')

# Resume:
curl -sN -H "X-API-Key: $KEY" \
  "$URL/api/articles/delta?bootstrap_after=$LAST_ID" >> snapshot.ndjson
```

Anchoring incremental to the **first** attempt's watermark is what keeps
resume correct: an update to an already-emitted article that lands between
your original start and the resume falls below that watermark's successors
and is replayed by your first incremental pull (idempotent upserts absorb
overlap).

### Onboarding a new consumer late

You do **not** need to re-extract anything. The bootstrap snapshot is a
**direct scan of the live article table**, not a replay of the change outbox.
It returns the entire current corpus regardless of when each article was
extracted or whether the consumer existed at extraction time.

Two guarantees make this safe:

- **No gap between snapshot and first delta.** The bootstrap's `next_since` is
  pinned to the current outbox watermark but never past the safe ceiling of
  any still-running extraction — so a change committed *while you were
  streaming* is re-served by your first incremental pull.
- **Enrichment updates are visible.** Because `content_hash` is the SHA-256
  of the served markdown, a later image-description run surfaces affected
  pages as `updated` deltas with a changed hash.

### Sharding

Add `?source_id=…` or `?vendor_id=…` to either bootstrap or incremental calls
to scope the feed. Both are RBAC-scoped to the API key's visible vendors.

---

## 7. Webhook Integration

The delta feed is pull-based. Webhooks are the push notification that tells
you *when* to pull.

### Setup

```bash
curl -sk -X POST -H "X-API-Key: $ADMIN_KEY" -H "Content-Type: application/json" \
  "https://docextractor.k3s.home.lan/api/webhooks" \
  -d '{
    "url": "https://my-graphrag-indexer.internal/api/docextractor/webhook",
    "events": "extraction_complete,new_page,updated_page,removed_page",
    "secret": "shared-hmac-secret",
    "is_active": true
  }'
```

### Events

| Event | When | Payload |
|-------|------|---------|
| `new_page` | A page was scraped for the first time | Per-page |
| `updated_page` | A page's content changed since last run | Per-page |
| `removed_page` | A page dropped out of the rebuilt TOC | Per-page |
| `extraction_complete` | Full extraction run finished | Summary with delta counts |

### `extraction_complete` payload

```json
{
  "event": "extraction_complete",
  "timestamp": "2026-07-12T16:00:53Z",
  "run_id": "4e11cfe7-dfe5-45f9-a502-c56cf3610292",
  "source_id": "21632f3b-5a4c-4c93-9f00-6701d0e9f677",
  "source_name": "Developer Guide",
  "vendor_name": "AWS",
  "product_name": "AWS Backup",
  "delta": {
    "added": 0,
    "updated": 13,
    "removed": 0,
    "watermark": "eyJzZXEiOjExNTQ0LCJ2IjoxfQ=="
  }
}
```

The `watermark` is informational. **Always pull with your own stored cursor**,
not the webhook's watermark — a missed delivery self-heals on the next pull.

### Signature verification

Every webhook POST includes:

```
X-DocExtractor-Signature: sha256=<HMAC-SHA256 of the body using the webhook's secret>
```

Verify on receipt:

```python
import hmac, hashlib

expected = "sha256=" + hmac.new(
    secret.encode(), request_body, hashlib.sha256
).hexdigest()
hmac.compare_digest(expected, request.headers["X-DocExtractor-Signature"])
```

Retries: 3 attempts with exponential backoff (0s, 5s, 15s). Delivery records
are auditable via `GET /api/webhooks/{id}/deliveries`.

---

## 8. Images & VLM Descriptions

Articles can contain images. When VLM image descriptions are enabled
(`DOCEXTRACTOR_IMAGE_VLM_ENABLED=true`), meaningful images (screenshots,
diagrams — skipping icons/spacers) are described by a vision model, and the
descriptions are:

1. Injected as captions into the article markdown
2. Exposed as structured fields on the `images[]` array

```json
"images": [
  {
    "url": "/media/57e94224-.../3359e18c2790.png",
    "alt": "QR code",
    "description": "A QR code, a square matrix barcode composed of black modules on a white background, with three large square alignment patterns in the corners.",
    "kind": "other"
  }
]
```

| Field | Description |
|-------|-------------|
| `url` | Path to the locally-stored image (relative to the DocExtractor host) |
| `alt` | Original `alt` text from the source HTML |
| `description` | VLM-generated text description (or `null` if not described) |
| `kind` | VLM classification: `screenshot`, `diagram`, `chart`, `photo`, `other` (or `null`) |

**For a text-only GraphRAG pipeline:** the `description` field lets you
"see" visual content without processing images. Index it alongside the
article markdown so queries about diagrams or screenshots match.

Descriptions are **cached by image content hash** — an image is described
once, ever. Running an enrichment pass on a source doesn't re-describe
images that were already processed.

When `description` / `kind` are `null`, the image either wasn't processed
(feature off, decorative, or not yet enriched). A later enrichment run
surfaces these as `updated` deltas with a changed `content_hash`.

---

## 9. Extraction Runs

Extraction runs are the scrape passes. Each run:

1. Creates a `run_start` sentinel in the change outbox
2. Discovers the TOC
3. Fetches every article (incrementally — only changed pages)
4. Creates/updates articles, writing to the change outbox
5. Optionally enriches images (VLM descriptions)
6. Commits a `run_start` sentinel, finishes, fires `extraction_complete`

### List runs

```bash
curl -sk -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/extraction/runs?limit=3"
```

**Response (live):**

```json
{
  "runs": [
    {
      "id": "7cbc7223-3ec3-4951-a837-c53447eebdaa",
      "source_id": "b2fab6e5-...",
      "source_name": "CommCell Console",
      "product_name": "CommCell Console",
      "vendor_name": "Commvault",
      "status": "running",
      "trigger": "manual",
      "kind": "extract",
      "current_phase": "toc_discovery",
      "articles_extracted": 0,
      "articles_total": 0,
      "articles_updated": 0,
      "articles_unchanged": 0,
      "started_at": "2026-07-12T16:00:53Z",
      "completed_at": null,
      "heartbeat_at": "2026-07-12T16:07:38Z"
    },
    {
      "id": "4e11cfe7-dfe5-45f9-a502-c56cf3610292",
      "source_id": "21632f3b-...",
      "source_name": "Developer Guide",
      "product_name": "AWS Backup",
      "vendor_name": "AWS",
      "status": "completed",
      "trigger": "manual",
      "kind": "extract",
      "current_phase": "image_enrich",
      "articles_total": 146,
      "articles_updated": 13,
      "articles_unchanged": 133,
      "started_at": "2026-07-12T16:00:13Z",
      "completed_at": "2026-07-12T16:00:53Z"
    }
  ]
}
```

Run statuses: `pending`, `running`, `paused`, `completed`, `failed`,
`cancelled`.

A downstream consumer normally doesn't need to trigger or manage runs — just
listen for `extraction_complete` and pull the delta. But the runs endpoint is
useful for debugging and monitoring freshness.

---

## 10. Export

If you prefer a bulk file download over streaming the delta feed:

```bash
curl -sk -X POST -H "X-API-Key: $RW_KEY" -H "Content-Type: application/json" \
  "https://docextractor.k3s.home.lan/api/export" \
  -d '{
    "source_id": "21632f3b-...",
    "format": "markdown",
    "split_max_tokens": 100000
  }'
```

This creates an export job (processed by the worker). Poll
`GET /api/export/jobs/{id}` for status, then download via
`GET /api/export/download/{export_id}` (zip) or
`GET /api/export/download/{export_id}/{filename}` (single file).

Split options: `split_max_articles`, `split_max_bytes`, `split_max_tokens`.
Articles are never split across files.

> For programmatic GraphRAG ingestion, prefer the delta feed over exports —
> it's streaming, incremental, and carries the change semantics exports lack.

---

## 11. Dashboard & Monitoring

### Source health

```bash
curl -sk -H "X-API-Key: dxk_XX…PUfE" \
  "https://docextractor.k3s.home.lan/api/dashboard/sources"
```

**Response (live):**

```json
{
  "summary": {
    "total": 280,
    "never_extracted": 0,
    "stale": 0,
    "failing": 0,
    "running": 1
  },
  "sources": [
    {
      "id": "21632f3b-...",
      "name": "Developer Guide",
      "vendor_name": "AWS",
      "product_name": "AWS Backup",
      "status": "completed",
      "last_extracted_at": "2026-07-12T16:00:53Z",
      "age_seconds": 345,
      "article_count": 146,
      "last_run_status": "completed",
      "last_run_new": 0,
      "last_run_updated": 13,
      "last_run_unchanged": 133
    }
    // ...
  ]
}
```

Useful for monitoring ingestion freshness and detecting stale or failing
sources in your pipeline.

---

## 12. Reference: Live Deployment Sample Data

### Corpus snapshot (2026-07-12)

| Metric | Value |
|--------|-------|
| Vendors | 40 |
| Products | 79 |
| Sources | 280 |
| Total articles | ~105,464 |
| Never extracted | 0 |
| Failing | 0 |
| Running | 1 |

### Sample vendors

| Vendor | Website |
|--------|---------|
| AWS | https://aws.amazon.com |
| Acronis | https://acronis.com |
| Afi.ai | https://afi.ai |
| Arcserve | https://arcserve.com |
| AvePoint | https://www.avepoint.com |
| Barracuda | https://barracuda.com |
| Commvault | — |
| Dell | — |

### Sample sources with article counts

| Source | Vendor | Product | Articles | Last Run |
|--------|--------|---------|----------|----------|
| Developer Guide | AWS | AWS Backup | 146 | 2026-07-12 (13 updated) |
| Cyber Protect Cloud Console | Acronis | Cyber Protect Cloud | 1,320 | 2026-07-11 (11 updated) |
| Cyber Protect On-premises | Acronis | Cyber Protect On-premises | 680 | 2026-07-11 (19 updated) |
| User Guide | Afi.ai | Afi.ai | 210 | 2026-07-11 (5 new, 205 updated) |

### Sample article (with VLM image description)

From the incremental delta, Dell PowerProtect Data Manager "Preface" article:

```json
{
  "seq": 2,
  "change_type": "updated",
  "id": "57e94224-4561-4320-8d77-263200ca7716",
  "vendor": "Dell",
  "product": "PowerProtect Data Manager",
  "title": "Preface",
  "content_hash": "a2c83de484d8a174db246e3d21f6cb2ae7afdbd27b360cd8a54a5810b8e2da8a",
  "estimated_tokens": 3750,
  "parent_chapter": null,
  "top_level_chapter": "Preface",
  "sort_order": 1,
  "content_markdown": "Preface\n-------\n\nAs part of an effort to improve...",
  "images": [
    {
      "url": "/media/57e94224-4561-4320-8d77-263200ca7716/3359e18c2790.png",
      "alt": "QR code",
      "description": "A QR code, a square matrix barcode composed of black modules on a white background, with three large square alignment patterns in the corners.",
      "kind": "other"
    }
  ]
}
```

### Sample tombstones

Three articles removed from a Dell PowerProtect Data Manager Kubernetes
User Guide source (page dropped from rebuilt TOC):

```json
{"seq":327,"change_type":"removed","id":"93c0c79f-...","topic_key":"https://www.dell.com/.../edit-or-delete-a-protection-rule","source_id":"8f39da35-...","removed_at":"2026-07-11T18:41:55Z","run_id":"3de0b5a2-..."}
{"seq":328,"change_type":"removed","id":"5b32f85d-...","topic_key":"https://www.dell.com/.../ppdmk8sdiag-diagnostics-tool","source_id":"8f39da35-...","removed_at":"2026-07-11T18:41:55Z","run_id":"3de0b5a2-..."}
{"seq":329,"change_type":"removed","id":"f8c76242-...","topic_key":"https://www.dell.com/.../restore-considerations-for-openshift-environments","source_id":"8f39da35-...","removed_at":"2026-07-11T18:41:55Z","run_id":"3de0b5a2-..."}
```

---

## 13. Client Implementation Patterns

### Pattern 1: Bootstrap + incremental sync (recommended)

```python
import httpx, json

BASE = "https://docextractor.k3s.home.lan"
KEY = "dxk_XX…PUfE"
HEADERS = {"X-API-Key": KEY}
CURSOR_FILE = "/var/data/docextractor_cursor.txt"

def pull_delta(since=None, source_id=None):
    """Pull the delta feed and yield records."""
    params = {}
    if since:
        params["since"] = since
    if source_id:
        params["source_id"] = source_id
    with httpx.stream("GET", f"{BASE}/api/articles/delta",
                      headers=HEADERS, params=params, timeout=300,
                      verify=False) as resp:
        for line in resp.iter_lines():
            if line:
                yield json.loads(line)

def sync():
    # 1. Bootstrap (first run) or incremental (subsequent)
    cursor = read_cursor()  # None on first run

    records = pull_delta(since=cursor)
    for record in records:
        if record.get("control"):
            if record["control"] in ("cursor", "bootstrap_start"):
                new_cursor = record["next_since"]
            continue

        if record["change_type"] in ("added", "updated"):
            upsert_in_graphrag(record)
        elif record["change_type"] == "removed":
            delete_from_graphrag(record["id"])

    # 2. Only persist cursor on clean finish
    if new_cursor:
        write_cursor(new_cursor)

def upsert_in_graphrag(record):
    """Create/update a node in your graph + chunks in your vector store."""
    # Node: record["id"] with properties from the record
    # Edges: vendor → product → source → article
    #        parent_chapter → article, top_level_chapter → article
    # Chunks: split record["content_markdown"] by token budget
    #         (use record["estimated_tokens"] for budgeting)
    # Include record["images[].description"] as additional chunks
    pass

def delete_from_graphrag(article_id):
    """Remove the node and its chunks."""
    pass
```

### Pattern 2: Webhook-triggered sync

```python
# Flask / FastAPI endpoint
@app.post("/webhook/docextractor")
def webhook():
    # 1. Verify HMAC signature
    # 2. Check event == "extraction_complete"
    # 3. Pull delta with YOUR stored cursor (not the webhook's watermark)
    sync()
    return {"status": "ok"}
```

### Pattern 3: Sharded bootstrap (for large corpora)

```python
# Pull each vendor separately — parallelizable
VENDORS = ["128b5dee-...", "4d13da93-...", ...]  # from GET /api/vendors

for vendor_id in VENDORS:
    cursor = None
    while True:
        records = pull_delta(since=cursor, vendor_id=vendor_id)
        for record in records:
            process(record)
        if received_terminal_cursor:
            break
```

### Pattern 4: Polling fallback (no webhook)

If you can't receive webhooks, poll on a schedule:

```python
import schedule

schedule.every(30).minutes.do(sync)  # pull incremental delta
```

The delta feed is idempotent — pulling with a stale cursor when nothing
changed returns `count: 0` and costs almost nothing.

---

### Field reference cheat sheet

| Field | In delta? | In article detail? | Description |
|-------|-----------|-------------------|-------------|
| `seq` | ✅ (null in bootstrap) | ❌ | Outbox sequence number |
| `change_type` | ✅ | ❌ | `added` / `updated` / `removed` |
| `id` | ✅ | ✅ | Article UUID (stable across versions) |
| `topic_key` | ✅ | ❌ | Canonical source URL (with `{version}` for versioned sources) |
| `source_id` | ✅ | ✅ | Parent source UUID |
| `vendor` | ✅ | ✅ (`{id, name}`) | Vendor name |
| `product` | ✅ | ✅ (`{id, name}`) | Product name |
| `title` | ✅ | ✅ | Article title |
| `source_url` | ✅ | ✅ | Resolved URL for this version |
| `last_updated_at` | ✅ | ✅ | From source page (often null) |
| `content_hash` | ✅ | ❌ | SHA-256 of served markdown |
| `estimated_tokens` | ✅ | ✅ | Token count for chunking |
| `parent_chapter` | ✅ | ✅ (`{id, title}`) | One level up in TOC |
| `top_level_chapter` | ✅ | ✅ (`{id, title}`) | Root of TOC subtree |
| `sort_order` | ✅ | ✅ | Reading order within source |
| `run_id` | ✅ | ❌ | Extraction run that produced this change |
| `content_markdown` | ✅ | ✅ | Full article body |
| `images` | ✅ | ✅ | `[{url, alt, description, kind}]` |
| `removed_at` | tombstone only | ❌ | When the article was removed |
| `content_size_bytes` | ❌ | ✅ | Byte size of markdown |
| `created_at` | ❌ | ✅ | When article was first scraped |
| `extracted_at` | ❌ | ✅ | When article was last scraped |

---

*Document generated 2026-07-12 from DocExtractor codebase and live K3s
deployment data.*