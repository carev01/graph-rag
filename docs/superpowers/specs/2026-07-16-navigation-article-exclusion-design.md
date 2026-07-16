# Navigation-Article Exclusion + Blog-Fact Cleanup — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Exclude navigation/link-farm articles from semantic extraction + deterministically expire the junk facts already extracted.
**Date:** 2026-07-16
**Status:** Approved design — ready for implementation planning

Fixes the primary cause of the Azure soft-delete retrieval miss (see `local-retrieval-golden-report.md`): the pilot's *"Blogs, videos, tutorials, and other resources"* article — a navigation/link-farm index page — produced **86 facts (~4.4% of all 1,975)**, including **33 pure-noise `Blog 'X' was authored with <person>` facts** that pollute retrieval on generic queries. Two mechanisms: **prevent** (skip such articles at ingestion) and **clean** (expire the already-extracted junk, reusing existing maintenance machinery, no re-extraction).

---

## 1. Scope

**In:** a pure `is_navigation_article(title)` predicate; an ingest-time skip guard; a `tombstone_navigation_articles` cleanup wired into `maintenance` before `sweep`; retrieval re-validation. **Out:** re-extraction (the cleanup is deterministic); verbose/compound-query robustness (the *secondary* cause — the real soft-delete facts ARE retrievable with focused phrasing — deferred to a query-rewrite/synthesis slice); a fact-text noise filter (the tombstone-then-sweep path handles it structurally).

## 2. Prevention — `article_filter` + ingest guard

**`src/graph_extract/article_filter.py` (new, pure):** `is_navigation_article(title: str) -> bool`. Conservative — only clear index/changelog/link-farm pages, never real content. Patterns (grounded against all 451 corpus titles — matches exactly 5: `Blogs, videos, tutorials, and other resources`, `Archived release notes`, `Release notes MABS`, `What's New in MABS`, `What's new in Azure Backup`; misses `Overview of soft delete`, `Encryption in Azure Backup`, `Vault Lock`, `FAQ-Soft Delete`, `AWS Backup feature availability`, `Requester tasks`):
- contains `blogs, videos` (link-farm index),
- ends with `and other resources`,
- `archived release notes` / `release notes <suffix>` / bare `release notes`,
- `what's new` (changelog).
Case-insensitive; single source of truth for "is this a navigation page".

**Ingest guard:** in `ingest_driver.ingest_article`, immediately after `art = await content_fetch.fetch_article(...)`, if `is_navigation_article(art.title)` → return the empty `IngestArticleResult` (episodes_added=0) **without** chunking/extraction. Add a `skipped_navigation: bool = False` field to `IngestArticleResult` set True on skip (visibility in the pilot/worker logs + cost). Enforced at the single ingestion chokepoint, so bootstrap (`run_pilot`) and the incremental worker both honor it — no LLM tokens spent on nav pages.

## 3. Cleanup — tombstone-then-sweep (reuses A + C machinery)

**`graph_cleanup.tombstone_navigation_articles(driver, group_id) -> dict`:** find already-extracted `:Article`s (those with `HAS_EPISODE` edges) whose title matches `is_navigation_article`, and mark their episodes `removed=true` (the same `SET e.removed=true` mechanism as `tombstone_article_episodes`). Returns `{scanned, tombstoned_articles, tombstoned_episodes, titles}` for audit.

**Then the existing `sweep_stale_facts` (Sub-slice C) expires the junk facts** — a fact is expired only when **all** its supporting episodes are dead. This is the crucial property: a `Blog '…' authored with…` fact is supported *only* by the nav article's episodes → all dead → expired; a real fact that happens to also be supported by a nav episode but *also* by a real article's episode → still alive → kept. No new deletion logic; temporally correct (mark, never delete; `invalid_at`+`expired_by_sweep`).

**Wire into `maintenance`:** insert `tombstone_navigation_articles` **before** `sweep_stale_facts` in the `maintenance` command's composition (prune → retype → **nav-tombstone** → sweep → reconcile), so one `maintenance` run cleans nav facts. The combined audit dict gains a `navigation` key.

## 4. Retrieval validation (closes the loop)

Controller-run, no re-extraction: run `maintenance` on the current graph → confirm the ~86 nav facts (incl. the 33 blog-author facts) are now `invalid_at`-set (excluded by `/search/local`'s default validity filter) → re-run the Azure soft-delete golden query and the full golden harness → record the citation-precision delta. Expectation: the blog junk no longer crowds the top-k; the soft-delete query (and generic queries broadly) improve.

## 5. Testing

- **Unit (`article_filter`):** `is_navigation_article` True for the 5 nav titles; False for the real-content KEEP titles (Overview/Encryption/Vault Lock/FAQ/feature-availability/tasks) — the exact grounded set.
- **Integration (Neo4j testcontainer):** `tombstone_navigation_articles` on a seeded graph — a nav-title article's episodes become `removed=true`, a real article's untouched; returns the right audit counts. The ingest skip is covered by a focused test (stub the fetched article title → `ingest_article` returns `skipped_navigation=True`, episodes_added=0, no extraction attempted) OR, since `ingest_article` needs the full pipeline, a unit test of the guard predicate path is acceptable — assert the skip returns early (the `@live` ingest path already exercises the rest).
- **`maintenance` composition:** the nav-tombstone step runs before sweep and its audit key appears (extend the existing maintenance/cli check).
- Full non-live suite + ruff/mypy clean. The retrieval re-validation (§4) is controller-run, reported.

## 6. Acceptance criteria

1. `is_navigation_article` flags the 5 corpus navigation/changelog/link-farm titles and no real-content title.
2. `ingest_article` skips a navigation article without extraction (`skipped_navigation=True`, 0 episodes, 0 LLM tokens).
3. `tombstone_navigation_articles` marks the nav articles' episodes `removed`; a following `sweep` expires the facts supported *only* by them (the blog-author junk) while keeping facts co-supported by real articles.
4. `maintenance` composes the nav-tombstone before sweep; running it on the current graph expires the ~86 nav facts.
5. Retrieval re-validation shows the blog junk gone from `/search/local` results; the golden citation-precision delta is recorded.
6. Unit + integration tests green (testcontainer); ruff/mypy clean.

## 7. Deferred

- **Verbose/compound-query robustness** (secondary cause) — a query-rewrite or synthesis-layer concern for the next retrieval slice; focused queries already retrieve soft-delete facts.
- Broader content-quality heuristics (marketing/preamble paragraphs inside otherwise-real articles) — out of scope; this targets whole navigation *pages*.
- Re-extraction of the full corpus with the ingest guard active — happens naturally on the next bootstrap; not needed to clean the current graph.
