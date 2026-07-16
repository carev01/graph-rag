# Navigation-Article Exclusion + Blog-Fact Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skip navigation/link-farm articles at ingestion (prevention) and deterministically expire the junk facts already extracted from them (cleanup, reusing tombstone-then-sweep), then re-validate retrieval.

**Architecture:** `graph_extract` only. A pure `article_filter.is_navigation_article` predicate; an ingest-time skip guard; a `tombstone_navigation_articles` cleanup wired into `maintenance` before `sweep`. No re-extraction.

**Tech Stack:** Python 3.12, Neo4j (async), pytest + testcontainers, ruff, mypy.

## Global Constraints

- **Conservative predicate:** `is_navigation_article` must flag ONLY clear index/changelog/link-farm pages, never real content. Grounded against all 451 corpus titles — matches exactly 5 (`Blogs, videos, tutorials, and other resources`, `Archived release notes`, `Release notes MABS`, `What's New in MABS`, `What's new in Azure Backup`); must miss `Overview of soft delete`, `Encryption in Azure Backup`, `Vault Lock`, `FAQ-Soft Delete`, `AWS Backup feature availability`, `Requester tasks`.
- **Cleanup marks, never deletes** (temporal policy #3): tombstone marks episodes `removed=true`; the existing `sweep` expires facts whose episodes are ALL dead (keeps facts co-supported by a real article).
- **Single ingestion chokepoint:** the skip lives in `ingest_article`, so bootstrap + the incremental worker both honor it.
- `article_filter` is the single source of truth for "is this a navigation page" (used by the ingest guard AND the tombstone cleanup).
- Tests pass (testcontainer); ruff/mypy clean; no re-extraction.

---

## File Structure

- `src/graph_extract/article_filter.py` — **new:** `is_navigation_article(title)`.
- `src/graph_extract/ingest_driver.py` — **modify:** `IngestArticleResult.skipped_navigation`; skip guard in `ingest_article`.
- `src/graph_extract/graph_cleanup.py` — **modify:** `tombstone_navigation_articles`.
- `src/graph_extract/cli.py` — **modify:** wire nav-tombstone into `maintenance`.
- Tests: `tests/unit/test_article_filter.py`, `tests/integration/test_ingest_driver.py` (skip), `tests/integration/test_tombstone_navigation.py` (new).

---

### Task 1: `article_filter` predicate + ingest skip guard

**Files:**
- Create: `src/graph_extract/article_filter.py`
- Modify: `src/graph_extract/ingest_driver.py`
- Test: `tests/unit/test_article_filter.py`

**Interfaces:** `is_navigation_article(title: str) -> bool`; `IngestArticleResult` gains `skipped_navigation: bool = False`.

- [ ] **Step 1: Write failing unit tests** in `tests/unit/test_article_filter.py`:

```python
import pytest
from graph_extract.article_filter import is_navigation_article

NAV = [
    "Blogs, videos, tutorials, and other resources",
    "Archived release notes",
    "Release notes MABS",
    "What's New in MABS",
    "What's new in Azure Backup",
]
KEEP = [
    "Overview of soft delete", "Encryption in Azure Backup", "Vault Lock",
    "FAQ-Soft Delete", "AWS Backup feature availability", "Requester tasks",
    "Cross-Region backup", "Amazon S3 backups",
]

@pytest.mark.parametrize("t", NAV)
def test_navigation_flagged(t): assert is_navigation_article(t) is True

@pytest.mark.parametrize("t", KEEP)
def test_content_kept(t): assert is_navigation_article(t) is False
```

- [ ] **Step 2: Run, confirm fail** (module missing). `uv run --extra dev pytest tests/unit/test_article_filter.py -v`.

- [ ] **Step 3: Implement `article_filter.py`:**

```python
"""Pure classifier for navigation / index / link-farm / changelog articles.

Single source of truth for "this article page is not durable knowledge
content" -- used to skip such pages at ingestion and to clean already-
extracted junk facts. Conservative: only clear non-content pages.
"""
from __future__ import annotations

import re

_NAV = re.compile(
    r"blogs?,?\s*videos"          # "Blogs, videos, tutorials, ..."
    r"|and other resources$"      # link-farm index tail
    r"|\brelease notes\b"         # "Archived release notes", "Release notes MABS"
    r"|what'?s new",              # changelog pages
    re.IGNORECASE,
)


def is_navigation_article(title: str) -> bool:
    """True for navigation/index/link-farm/changelog pages that carry no
    durable backup-domain knowledge (their extracted facts are retrieval noise)."""
    return bool(title and _NAV.search(title))
```

- [ ] **Step 4: Run predicate tests green.** Gate on `article_filter.py`.

- [ ] **Step 5: Add `skipped_navigation` + the ingest guard.** In `ingest_driver.py`: add `skipped_navigation: bool = False` to the `IngestArticleResult` dataclass. In `ingest_article`, right after `art = await content_fetch.fetch_article(self._docext, article_id)`:

```python
        from graph_extract.article_filter import is_navigation_article  # or top-level import
        if is_navigation_article(art.title or ""):
            res.skipped_navigation = True
            return res      # navigation page: no chunking, no extraction, 0 tokens
```
(Prefer a top-level `from graph_extract.article_filter import is_navigation_article` import at the module head, matching the file's import style.)

- [ ] **Step 6: Test the skip.** In `tests/integration/test_ingest_driver.py` (or a focused new test), verify the guard returns early WITHOUT extraction. Since `ingest_article` needs docext + graphiti, the cleanest LLM-free test builds an `IngestDriver` with a **stub docext** whose `fetch_article` returns an object with `.title = "Archived release notes"`, `.id`, `.content_markdown`, and asserts `res.skipped_navigation is True and res.episodes_added == 0` and that graphiti's `add_episode` was never called (a stub graphiti recording calls). If the existing `test_ingest_driver.py` is `@live`, add a small standalone `tests/unit/test_ingest_skip.py` with the stubs instead. Choose the LLM-free path.

- [ ] **Step 7: Run tests + full non-live suite green.** Gate on `article_filter.py` + `ingest_driver.py`.

- [ ] **Step 8: Commit.** `git add src/graph_extract/article_filter.py src/graph_extract/ingest_driver.py tests/unit/test_article_filter.py tests/unit/test_ingest_skip.py 2>/dev/null tests/integration/test_ingest_driver.py 2>/dev/null; git commit -m "feat(extract): is_navigation_article + skip nav/link-farm pages at ingestion"`

---

### Task 2: `tombstone_navigation_articles` + wire into `maintenance`

**Files:**
- Modify: `src/graph_extract/graph_cleanup.py`, `src/graph_extract/cli.py`
- Test: `tests/integration/test_tombstone_navigation.py`

**Interfaces:** `async def tombstone_navigation_articles(driver, group_id) -> dict` → `{"scanned", "tombstoned_articles", "tombstoned_episodes", "titles"}`. Consumes `is_navigation_article` (Task 1).

- [ ] **Step 1: Write failing integration test** (Neo4j testcontainer, `extract_driver`). Seed: a nav-title Article (`title:'Archived release notes'`) with 2 HAS_EPISODE→Episodic(group_id set), a real Article (`title:'Vault Lock'`) with 1 episode. After `tombstone_navigation_articles(driver, "backup-docs")`: the nav article's episodes are `removed=true`, the real article's episode is untouched; returns `tombstoned_articles==1, tombstoned_episodes==2, titles==['Archived release notes']`.

```python
async def test_tombstone_navigation_articles(extract_driver):
    from graph_extract.graph_cleanup import tombstone_navigation_articles
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (a:Article {id:'nav1', title:'Archived release notes'}) "
                    "CREATE (a)-[:HAS_EPISODE]->(:Episodic {uuid:'ne1', group_id:$g}) "
                    "CREATE (a)-[:HAS_EPISODE]->(:Episodic {uuid:'ne2', group_id:$g})", g=g)
        await s.run("CREATE (b:Article {id:'real1', title:'Vault Lock'}) "
                    "CREATE (b)-[:HAS_EPISODE]->(:Episodic {uuid:'re1', group_id:$g})", g=g)
    res = await tombstone_navigation_articles(extract_driver, g)
    assert res["tombstoned_articles"] == 1 and res["tombstoned_episodes"] == 2
    assert res["titles"] == ["Archived release notes"]
    async with extract_driver.session() as s:
        removed = {r["u"]: r["rm"] async for r in await s.run(
            "MATCH (:Article)-[:HAS_EPISODE]->(e:Episodic) RETURN e.uuid AS u, e.removed AS rm")}
    assert removed["ne1"] is True and removed["ne2"] is True and removed.get("re1") is not True
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement `tombstone_navigation_articles`.** Fetch extracted articles (id+title of those with group-scoped episodes), filter titles in Python via `is_navigation_article`, then mark their episodes removed in one batched query:

```python
from graph_extract.article_filter import is_navigation_article

async def tombstone_navigation_articles(driver: AsyncDriver, group_id: str) -> dict:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (a:Article)-[:HAS_EPISODE]->(e:Episodic {group_id:$g}) "
            "RETURN a.id AS id, a.title AS title, count(e) AS eps", g=group_id)
        rows = [dict(rec) async for rec in r]
        nav = [row for row in rows if is_navigation_article(row["title"] or "")]
        ids = [row["id"] for row in nav]
        episodes = 0
        if ids:
            rr = await s.run(
                "MATCH (a:Article)-[:HAS_EPISODE]->(e:Episodic {group_id:$g}) "
                "WHERE a.id IN $ids SET e.removed=true RETURN count(e) AS c",
                g=group_id, ids=ids)
            episodes = (await rr.single())["c"]
    return {"scanned": len(rows), "tombstoned_articles": len(ids),
            "tombstoned_episodes": episodes, "titles": sorted(row["title"] for row in nav)}
```
(Verify `AsyncDriver` is already imported in `graph_cleanup.py`; the file already has `prune_noise_entities`/`retype_region_entities` in this style.)

- [ ] **Step 4: Run the test green.** Gate on `graph_cleanup.py`.

- [ ] **Step 5: Wire into `maintenance`.** In `cli.py`'s `maintenance` command, import `tombstone_navigation_articles`, and insert it BEFORE `sweep_stale_facts` in the composition, adding its result to the dumped audit under `"navigation"`:
```python
            prune = await prune_noise_entities(driver, settings.group_id)
            retype = await retype_region_entities(driver, settings.group_id)
            navigation = await tombstone_navigation_articles(driver, settings.group_id)
            sweep = await sweep_stale_facts(driver, settings.group_id)
            reconcile = await reconcile_same_as(driver, settings.group_id)
            _dump({"prune": prune, "retype": retype, "navigation": navigation,
                   "sweep": sweep, "reconcile": reconcile})
```
(nav-tombstone before sweep so the sweep expires the now-unsupported nav facts in the same run.)

- [ ] **Step 6: Verify `maintenance` still registers + composes** (`uv run --extra dev python -c "..."` --help check) + full non-live suite green. Gate.

- [ ] **Step 7: Commit.** `git add src/graph_extract/graph_cleanup.py src/graph_extract/cli.py tests/integration/test_tombstone_navigation.py && git commit -m "feat(extract): tombstone_navigation_articles + wire into maintenance (before sweep)"`

---

### Task 3: Cleanup run + retrieval re-validation (controller-run)

**Files:**
- Modify: `docs/superpowers/local-retrieval-golden-report.md` (append a "navigation-cleanup" section) OR create `docs/superpowers/navigation-cleanup-report.md`.

Controller-run against the current graph (no re-extraction).

- [ ] **Step 1: Run `maintenance` on the current graph.** `uv run --extra dev python -m graph_extract.cli maintenance` — capture the `navigation` audit (expect the "Blogs, videos, tutorials, and other resources" article tombstoned, its episodes marked removed) and the `sweep.expired` count (expect ~86 nav facts, incl. the 33 blog-author facts, now expired). Confirm via a query that `Blog '…' authored with…` facts now have `invalid_at` set.

- [ ] **Step 2: Confirm the junk is gone from retrieval.** Re-run the queries that previously surfaced blog facts (the Azure soft-delete query + a couple generic ones) through `search_local` — the `Blog '…'` facts should no longer appear (validity filter excludes expired facts).

- [ ] **Step 3: Re-run the golden harness.** `uv run --extra dev python -m answer_api.eval_golden --k 10` — record the new citation-precision@k + per-question deltas vs the 0.733 baseline (esp. the Azure soft-delete question; note any other movement).

- [ ] **Step 4: Write the report** — a short "navigation-cleanup" section (append to `local-retrieval-golden-report.md` or a new `navigation-cleanup-report.md`): the maintenance `navigation`+`sweep` audit, before/after retrieval on the soft-delete + generic queries, and the golden citation-precision delta. Honest: if soft-delete still misses due to the *secondary* (verbose-query) cause, say so and reaffirm that's the deferred query-rewrite concern. Commit.

- [ ] **Step 5: Commit.** `git add docs/superpowers/*.md && git commit -m "docs: navigation-cleanup retrieval re-validation (before/after golden precision)"`

---

## Self-Review Notes

- **Spec coverage:** predicate + ingest skip → T1; tombstone cleanup + maintenance wiring → T2; run + re-validate + report → T3. All §-sections covered.
- **Deps:** T1 → T2 (tombstone uses `is_navigation_article`); T3 uses T2 (+ the answer_api golden harness). Order T1, T2, T3.
- **Reuse:** cleanup = tombstone (mark removed) + the existing `sweep` (expire all-dead-episode facts) — no new deletion logic; the sweep's "all episodes dead" property keeps co-supported real facts.
- **Type consistency:** `is_navigation_article(title)->bool`; `tombstone_navigation_articles(driver,group_id)->{scanned,tombstoned_articles,tombstoned_episodes,titles}`; `IngestArticleResult.skipped_navigation`.
- **No-placeholder check:** each code step carries real code + exact commands; the ingest-skip test is explicitly the LLM-free stub path.
