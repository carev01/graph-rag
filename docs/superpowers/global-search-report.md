# Global (Map-Reduce) Search `/search/global` — Demonstration Report

**Slice:** Phase 3, slice 2 — global map-reduce search over the community-report
layer for cross-vendor thematic questions ("compare all vendors on workload X"),
with fact-ID → URL citations.
**Date:** 2026-07-17
**Status:** Implemented, reviewed, merged locally. Full non-live suite green
(380 passed); `@live` smoke green; demonstrated end-to-end on the live
`backup-docs` graph.

## What it does

`GET /search/global?q=&level=&k=` answers a cross-vendor thematic question by
map-reducing over the `:Community` report layer built in slice 1:

1. **Embed** the query with the shared TEI/Jina embedder — the SAME space as
   `:Community.embedding` (design decision #4).
2. **Shortlist** `:Community {group_id, level}` by cosine similarity + a small
   rating boost (`sim + 0.1·rating/10`), top-k, skipping reports with no
   `cited_fact_uuids` (nothing citable).
3. **MAP** (parallel `asyncio.gather`): each shortlisted report → the map-tier
   LLM (GLM-5.2 by default, config-overridable via `map_llm_*`) returns
   `{relevance, key_points, fact_ids}`; `fact_ids` are validated ⊆ the report's
   real `cited_fact_uuids` (hallucinations dropped); reports below the relevance
   floor or with unparseable JSON are dropped. A map call that raises is logged
   and skipped — never aborts the batch.
4. **REDUCE**: number the ordered-unique union of surviving `fact_ids` into `[N]`
   markers; the synthesis-tier LLM (GLM-5.2) writes a themed, cross-vendor answer
   citing `[N]` only. `_finalize_answer` strips any URL the model writes and keeps
   only markers present in the marker map.
5. **Resolve**: `Provenance.resolve_citations` expands each cited `fact_uuid` to
   its source URLs by graph traversal (`fact → episodes → article → source_url`).

Zero shortlisted communities, or zero surviving map results → a fixed refusal
with **no LLM spend**. The only LLM calls are the map fan-out (map tier) and one
reduce (synthesis tier).

## Live run on `backup-docs`

The `:Community` layer at query time: 66 reports across 3 Leiden levels
(35 / 19 / 12 at levels 0 / 1 / 2), all embedded in the shared space; level 1
(the default query level) has 19 communities.

**Query:** *"Compare how AWS Backup and Azure Backup handle backup retention and
lifecycle policies."*

Result (one representative run — the map relevance filter is LLM-judged, so the
exact community/citation counts vary slightly between runs):

```
communities_used:            4   (relevance 3–9, shortlisted at level 1)
citations:                  24   (every one a real fact of a shortlisted community)
total resolved sources:     24   (every citation resolves to ≥1 source URL)
distinct source hosts:      docs.aws.amazon.com, learn.microsoft.com   ← cross-vendor
URL authored by the LLM:    none  ("http" not in answer)
```

The answer is organized **by theme, then by vendor** — exactly the cross-vendor
synthesis goal:

> **Theme 1: Retention Policy Configuration**
> *AWS Backup:* Enforces minimum retention ranges for daily/weekly/monthly/yearly
> retention [13]…[18]; modifying policies aligns existing recovery points to new
> retention values [7]…[12]; manages recovery-point lifecycle via incremental
> backup chains and a garbage collector that updates expiry times.
>
> **Theme 2: Immutability and Deletion Protection**
> *AWS Backup:* Uses Vault Lock to enforce WORM immutability [1]…[6].
> *Azure Backup:* Immutable vaults prevent deletion; immutability blocks
> operations that reduce cloud/online retention or delete valid backup items.
>
> **Theme 3: Soft Delete** — *Azure Backup:* soft-delete retention up to 180 days;
> deleted recovery points retained 14 days at no cost. (*AWS:* no finding in the
> shortlisted communities — the model says so rather than inventing one.)
>
> **Theme 4: Long-Term Retention (Archive Tier)** — *Azure Backup:* archive tier
> for LTR; only monthly/yearly recovery points for Azure VMs; points must age in
> the standard tier before archiving; not supported on ZRS vaults; …

`communities_used` for this run:

| community | relevance |
|---|---|
| Backup Retention and Recovery Point Lifecycle Analysis | 9 |
| Azure Backup: Archive Tier, Recovery Point Management, Soft Delete, Instant Restore, Encryption | 7 |
| Backup Vault Security, Governance, and Recovery Features | 6 |
| Azure Backup Archive Tier Capabilities and Limitations | 6 |

## Citations resolve to real, cross-vendor source URLs

Every `[N]` marker traces to a concrete source (design decision #2, resolver
chain `community cited_fact_uuid → episode → article → source_url` — no
LLM-authored URLs):

| marker → fact_uuid | resolves to |
|---|---|
| `[28]` → `291fef70-…` | *Overview* — learn.microsoft.com/azure/backup/archive-tier-support |
| `[19]` → `229f6997-…` | *Back up encrypted Azure VMs* — learn.microsoft.com/azure/backup/backup-azure-vms-encryption |
| `[12]` → `cded2586-…` | *Overview of soft delete* — learn.microsoft.com/azure/backup/secure-by-default |
| (AWS-side markers) → … | docs.aws.amazon.com/aws-backup/… (Vault Lock, retention, lifecycle) |

The point: a thematic, cross-vendor answer built on community reports cites
concrete, verifiable AWS **and** Azure documentation URLs — not a hand-wave at
"the community."

## Design-decision alignment (verified)

- **#2 citations are graph traversal, never LLM:** map `fact_ids` are validated ⊆
  the report's real `cited_fact_uuids`; reduce emits only `[N]` markers validated
  against the marker map; `_finalize_answer`'s `_URL_RE` strips any URL the reduce
  model writes; `Provenance.resolve_citations` expands markers → source URLs.
  Demonstrated: `"http" not in answer`, all 24 citations resolve.
- **#4 one embedding space:** the query is embedded with the SAME shared TEI/Jina
  embedder as `:Community.embedding` (and entities/facts), so shortlist cosine is
  meaningful.
- Reuses the `/answer` citation machinery (`_finalize_answer`, `Provenance`) —
  global answers are cited the same verifiable way as local ones.

## Verification summary

- **Unit (`tests/unit/test_global_rank.py`, `test_global_map.py`):** cosine
  (zero-vector safe), ranking (rating boost, top-k, cite-less skip), level filter;
  map JSON parse + retry, relevance floor, fact_id ⊆ report validation, tier
  fallback to the judge/synthesis (GLM) tier.
- **Integration (`tests/integration/test_global_search.py`, Neo4j testcontainer +
  fake LLM clients):** shortlist level-scoping; full end-to-end map→reduce→resolve
  (citations resolve to seeded URLs, model-written URL stripped, invalid marker
  dropped); empty-shortlist → refusal with **no LLM call**.
- **App (`tests/unit/test_answer_api_app.py`):** `GET /search/global` returns the
  `{query, answer, citations, communities_used}` shape; missing `q` → 422;
  lifespan builds the shared embedder + map client and closes the map client on
  shutdown (embedder shares TEI, not closed); all pre-existing endpoints
  (`/health`, `/search/local`, `/answer`, `/timeline`) unchanged — 17 app tests.
- **`@live` smoke (`tests/integration/test_global_search_live.py`):**
  `/search/global` on `backup-docs` returns a non-empty answer, `communities_used`,
  no URL in the answer, and every citation resolving to ≥1 source.
- Full non-live suite: **380 passed, 7 `@live` deselected**; ruff/mypy clean.

## Review findings addressed

- **Task 4 — `gather` exception narrowing:** the reduce/orchestrator looped over
  `asyncio.gather(return_exceptions=True)` results with `isinstance(m, Exception)`,
  which (a) failed the mandatory mypy gate and (b) let a non-`Exception`
  `BaseException` (e.g. `asyncio.CancelledError`, common on client disconnect in a
  FastAPI service) slip into `results` and crash on `.fact_ids`. Fixed to
  `isinstance(m, BaseException)` — mypy clean and the batch-failure guarantee holds.
- Plan test defect caught pre-implementation: the integration test asserted
  `sources[0]["source_url"]`, but `Provenance.resolve_citations` returns sources
  keyed `{url, title, article_id}`; corrected to `["url"]` before dispatch.

## Follow-ups (deferred, per spec §10)

- Hierarchical descent into high-scoring communities' `PARENT_OF` children; auto
  level-selection by query breadth; a `:Community.embedding` vector index for
  scale (brute-force cosine is fine at 19–66 rows).
- Wiring `/search/global` into the `/answer` router (Phase 4) and DRIFT.
- Per-key-point (vs per-community) fact-marker granularity.
- Minor test-coverage gap (final-review triage): the "all map results
  filtered/failed → refusal" branch has no dedicated test (the empty-shortlist
  refusal path is covered).
