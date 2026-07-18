# Tombstone-Sweep Dirtiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make facts silently expired by the weekly staleness sweep mark their communities dirty, so the incremental `theme-build` regenerates the now-stale reports.

**Architecture:** Two small query extensions, both keyed on the sweep's own `expired_by_sweep=true` marker: `touched_entities` gains a branch that adds the endpoints of sweep-expired facts (`invalid_at > prev_cursor`) to the touched set, and `_corpus_cursor` gains a `UNION ALL` branch that maxes over sweep `invalid_at` so the watermark advances past expirations (the swept community regenerates exactly once). Symmetric with the existing `created_at` signal; the sweep and theme-build stay decoupled.

**Tech Stack:** Python 3.12, Neo4j 5.26 Cypher, pytest (Neo4j testcontainer).

## Global Constraints

- **Key the new dirty signal on `expired_by_sweep = true`** (the sweep's marker), plus `invalid_at > prev_cursor`. Do NOT key on `invalid_at` alone — graphiti's contradiction-invalidations set `invalid_at` to a historical reference_time and are already covered by `created_at` on their new episode; keying broadly would double-count and mis-fire.
- The sweep (`graph_extract/staleness_sweep.py`) is UNCHANGED. `theme-builder` stays the only writer of `:Community`.
- **Dirty-once:** the widened watermark must cover sweep `invalid_at`, so a swept community regenerates on the first incremental run after the sweep and is clean on subsequent no-change runs.
- No new config. Zero swept facts → behavior identical to today (the current live baseline has none).
- Run with `uv`: `uv run --extra dev pytest …`, `uv run ruff check src tests` (CI gate — lints tests; **no semicolons in fakes (E702), imports at file top (E402)**), `uv run mypy src`. Integration tests share a **module-scoped** Neo4j testcontainer — wipe with `MATCH (n) DETACH DELETE n` before seeding.

---

## File Structure

- **Modify** `src/theme_builder/incremental.py` — add the sweep branch to `touched_entities`.
- **Modify** `src/theme_builder/cli.py` — add the sweep `UNION ALL` branch to `_corpus_cursor`.
- **Tests:** `tests/integration/test_incremental_queries.py` (extend — touched sweep signal), `tests/integration/test_incremental_cli.py` (extend — `_corpus_cursor` sweep watermark + the end-to-end dirty-once).

---

### Task 1: Sweep dirty signal — `touched_entities` + `_corpus_cursor`

**Files:**
- Modify: `src/theme_builder/incremental.py`, `src/theme_builder/cli.py`
- Test: `tests/integration/test_incremental_queries.py` (extend), `tests/integration/test_incremental_cli.py` (extend)

**Interfaces:**
- Consumes: existing `touched_entities(driver, group_id, prev_cursor) -> set[str] | None`, `_corpus_cursor(driver, group_id) -> str | None`.
- Produces: `touched_entities` also includes endpoints of facts `expired_by_sweep=true AND invalid_at > prev_cursor`; `_corpus_cursor` also maxes over `invalid_at` of `expired_by_sweep=true` facts.

- [ ] **Step 1: Write the failing tests**

Append to `tests/integration/test_incremental_queries.py`:
```python
async def test_touched_includes_swept_fact_endpoints(extract_driver):
    from theme_builder.incremental import touched_entities
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        # swept fact: expired_by_sweep + invalid_at AFTER cursor -> endpoints touched
        await s.run("CREATE (a:Entity {group_id:$g, uuid:'sa', created_at: datetime('2026-01-01')})"
                    "-[:RELATES_TO {group_id:$g, uuid:'fs', created_at: datetime('2026-01-01'), "
                    "expired_by_sweep: true, invalid_at: datetime('2026-02-01')}]->"
                    "(b:Entity {group_id:$g, uuid:'sb', created_at: datetime('2026-01-01')})", g=g)
        # graphiti-style invalidation: invalid_at after cursor but NO expired_by_sweep -> NOT touched
        await s.run("CREATE (a:Entity {group_id:$g, uuid:'ga', created_at: datetime('2026-01-01')})"
                    "-[:RELATES_TO {group_id:$g, uuid:'fg', created_at: datetime('2026-01-01'), "
                    "invalid_at: datetime('2026-02-01')}]->"
                    "(b:Entity {group_id:$g, uuid:'gb', created_at: datetime('2026-01-01')})", g=g)
        # swept fact but invalid_at BEFORE cursor -> already accounted -> NOT touched
        await s.run("CREATE (a:Entity {group_id:$g, uuid:'oa', created_at: datetime('2026-01-01')})"
                    "-[:RELATES_TO {group_id:$g, uuid:'fo', created_at: datetime('2026-01-01'), "
                    "expired_by_sweep: true, invalid_at: datetime('2026-01-10')}]->"
                    "(b:Entity {group_id:$g, uuid:'ob', created_at: datetime('2026-01-01')})", g=g)
    out = await touched_entities(extract_driver, g, "2026-01-15T00:00:00Z")
    assert out == {"sa", "sb"}
```

Append to `tests/integration/test_incremental_cli.py`:
```python
async def test_corpus_cursor_includes_sweep_invalid_at(extract_driver):
    import theme_builder.cli as cli
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (:Episodic {group_id:$g, uuid:'ep', created_at: datetime('2026-01-01')})", g=G)
        await s.run("CREATE (a:Entity {group_id:$g, uuid:'e1', created_at: datetime('2026-01-02')})"
                    "-[:RELATES_TO {group_id:$g, uuid:'f1', created_at: datetime('2026-01-03'), "
                    "expired_by_sweep: true, invalid_at: datetime('2026-05-01')}]->"
                    "(b:Entity {group_id:$g, uuid:'e2', created_at: datetime('2026-01-02')})", g=G)
    cur = await cli._corpus_cursor(extract_driver, G)
    assert cur.startswith("2026-05-01")     # sweep invalid_at dominates the created_ats
```
(`G = "backup-docs"` is already defined at the top of `test_incremental_cli.py`.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/integration/test_incremental_queries.py::test_touched_includes_swept_fact_endpoints tests/integration/test_incremental_cli.py::test_corpus_cursor_includes_sweep_invalid_at -q`
Expected: FAIL (`touched` is missing `sa`/`sb`; the cursor is `2026-01-03`, not `2026-05-01`).

- [ ] **Step 3: Implement**

In `src/theme_builder/incremental.py`, extend `touched_entities` — after the existing fact `created_at` loop (before `return touched`), add:
```python
        # Facts silently expired by the staleness sweep carry no new created_at:
        # the sweep sets invalid_at=now + expired_by_sweep=true (never deletes the
        # edge). Key the dirty signal on those markers so the community regenerates.
        r = await s.run(
            "MATCH (a:Entity {group_id:$g})-[f:RELATES_TO {group_id:$g}]->"
            "(b:Entity {group_id:$g}) "
            "WHERE f.expired_by_sweep = true AND f.invalid_at > datetime($c) "
            "RETURN a.uuid AS a, b.uuid AS b", g=group_id, c=prev_cursor)
        async for rec in r:
            touched.add(rec["a"])
            touched.add(rec["b"])
```

In `src/theme_builder/cli.py`, extend `_corpus_cursor`'s subquery with a fourth branch (so the watermark advances past sweep expirations — this is what makes a swept community dirty exactly once):
```python
        r = await s.run(
            "CALL { MATCH (e:Episodic {group_id:$g}) RETURN e.created_at AS t "
            "UNION ALL MATCH (n:Entity {group_id:$g}) RETURN n.created_at AS t "
            "UNION ALL MATCH ()-[f:RELATES_TO {group_id:$g}]->() RETURN f.created_at AS t "
            "UNION ALL MATCH ()-[f:RELATES_TO {group_id:$g}]->() "
            "WHERE f.expired_by_sweep = true RETURN f.invalid_at AS t } "
            "RETURN toString(max(t)) AS c",
            g=group_id)
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/integration/test_incremental_queries.py tests/integration/test_incremental_cli.py -q`
Expected: PASS (the two new tests + all existing incremental query/CLI tests). `uv run ruff check src/theme_builder/incremental.py src/theme_builder/cli.py tests/integration/test_incremental_queries.py tests/integration/test_incremental_cli.py` + `uv run mypy src/theme_builder/incremental.py src/theme_builder/cli.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/theme_builder/incremental.py src/theme_builder/cli.py tests/integration/test_incremental_queries.py tests/integration/test_incremental_cli.py
git commit -m "feat(theme-builder): sweep-expired facts mark communities dirty (+ watermark)"
```

---

### Task 2: End-to-end dirty-once + full-suite gate + demonstration

**Files:**
- Test: `tests/integration/test_incremental_cli.py` (extend)
- Create: `docs/superpowers/sweep-dirtiness-report.md`

**Interfaces:**
- Consumes: the Task-1 `touched_entities`/`_corpus_cursor` extensions; the existing `_patch`, `_settings`, `_seed_persisted_and_entities`, `_FakeEmbedder` helpers in `test_incremental_cli.py`.

- [ ] **Step 1: Write the failing end-to-end test** (append to `tests/integration/test_incremental_cli.py`)

```python
async def test_swept_community_regenerates_once(extract_driver, monkeypatch):
    import theme_builder.cli as cli
    calls = _patch(monkeypatch)
    # persisted A {e1,e2}, B {e3,e4}, corpus_cursor 2026-03-01; nothing created after it
    await _seed_persisted_and_entities(extract_driver, e_a_created="2026-01-01")
    # a fact among A's members is silently swept AFTER the community's corpus_cursor
    async with extract_driver.session() as s:
        await s.run("MATCH (a:Entity {uuid:'e1', group_id:$g}), (b:Entity {uuid:'e2', group_id:$g}) "
                    "CREATE (a)-[:RELATES_TO {group_id:$g, uuid:'fx', created_at: datetime('2026-01-01'), "
                    "expired_by_sweep: true, invalid_at: datetime('2026-06-01')}]->(b)", g=G)
    r1 = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)
    assert calls["n"] == 1                       # only community A regenerated (its fact was swept)
    assert r1["reports_regenerated"] == 1 and r1["reports_reused"] == 1
    # second run: the watermark now covers the sweep invalid_at -> A clean -> zero regen
    r2 = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)
    assert calls["n"] == 1                       # unchanged: no further regeneration
    assert r2["reports_regenerated"] == 0 and r2["reports_reused"] == 2
```

- [ ] **Step 2: Run to verify it fails (before Task 1 is applied) or passes (after)**

Run: `uv run --extra dev pytest tests/integration/test_incremental_cli.py::test_swept_community_regenerates_once -q`
Expected: PASS (Task 1 is already implemented). If Task 1 were absent, r1 would show `reports_regenerated == 0` (the sweep isn't detected) — this test is the integration proof that the two Task-1 extensions work together.

- [ ] **Step 3: Full non-live suite + CI lint gate**

Run: `uv run --extra dev pytest -m "not live" -q` → all pass.
Run: `uv run ruff check src tests` → clean. `uv run mypy src` → clean.
Fix any fallout before continuing.

- [ ] **Step 4: Demonstration report**

Write `docs/superpowers/sweep-dirtiness-report.md` documenting: the gap (sweep expires facts with no new `created_at` → previously reused a stale report); the fix (dirty signal + watermark keyed on `expired_by_sweep`); and the end-to-end proof (a swept community regenerates once, then is clean on the next run — the dirty-once guarantee), citing the `test_swept_community_regenerates_once` counts. Note the `@live` sweep demo is deferred (per spec §9) since it would require expiring a real live fact.

- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_incremental_cli.py docs/superpowers/sweep-dirtiness-report.md
git commit -m "test(theme-builder): sweep dirty-once end-to-end + demonstration"
```

---

## Notes for the implementer

- The whole slice is two `WHERE f.expired_by_sweep = true` query branches. Both are required together: the `touched_entities` branch makes a swept community dirty; the `_corpus_cursor` branch advances the stored watermark past the sweep `invalid_at` so it is dirty **exactly once** (without it, `invalid_at > prev_cursor` stays true forever and the community regenerates every run).
- Precision: keying on `expired_by_sweep=true` (not bare `invalid_at`) is deliberate — graphiti's own invalidations carry a historical `invalid_at` and are already caught by `created_at` on their new episode.
- Zero swept facts → both branches return nothing → behavior is byte-for-byte today's; existing incremental tests (incl. the live no-change smoke) stay green.
- Test seeds: one statement per line, no semicolons (E702); imports at file top (E402). Wipe the module-scoped container per test.
