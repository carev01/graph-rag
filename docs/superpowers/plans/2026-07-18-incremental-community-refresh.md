# Incremental Community Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `theme-build` incremental — regenerate community reports only for new/changed communities (via `created_at` timestamps + Jaccard matching), instead of LLM-rebuilding every community every run.

**Architecture:** A new `theme_builder/incremental.py` (touched-set from `created_at`, persisted-community load, pure Jaccard match + dirty/clean classify); a new atomic `write_communities_incremental` that carries per-community `embedding`/`generated_at`; an incremental orchestration path in `cli.py` (default) with a `--full` backstop. Detection (`detect.py`) and report generation (`report.py`) are reused unchanged; the LLM only touches new/dirty communities.

**Tech Stack:** Python 3.12, Neo4j 5.26 (Cypher over the entity + community graph), typer CLI, pytest (unit for pure logic; Neo4j testcontainer for graph I/O; `@live` for GDS detection).

## Global Constraints

- **Design decision #2:** reports stay fact-cited; the report LLM never writes a URL (unchanged `report.py`). Reused reports were generated under #2. `theme-builder` remains the only writer of `:Community`; the layer stays derived & disposable (`--full` rebuilds from scratch).
- **Dirty signal = `created_at`:** a community is dirty if it is new (no Jaccard match) or any member is *touched* — an Entity with `created_at > prev_cursor`, or an endpoint of a `RELATES_TO` fact with `created_at > prev_cursor`. `prev_cursor` = `max(:Community.corpus_cursor)`; `None` (no prior layer) ⇒ everything dirty.
- **Stable identity:** fresh Leiden communities are matched to persisted ones by member-set **Jaccard ≥ τ** (default 0.5), greedy 1:1 within a level; a match **inherits the persisted `community_id`**, an unmatched fresh community keeps its content-hash birth id. `detect`'s content-hash `parent_id`s must be **remapped** to stable ids before writeback.
- **Writeback is atomic** (single `execute_write`): delete-all `:Community {group_id}` then write the fresh set (dissolved communities vanish by omission); dirty entries carry fresh `embedding`/`generated_at=now`, clean entries carry the reused prior ones; all get `corpus_cursor` = the run watermark.
- **`--full` flag** on `theme-build` forces today's full rebuild (`_run_theme_build`); incremental is the default.
- Run with `uv`: `uv run --extra dev pytest …`, `uv run ruff check src tests` (CI gate — lints tests; **no semicolons in fakes (E702), imports at file top (E402)**), `uv run mypy src`. Hermetic settings: `ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")`. Integration tests share a **module-scoped** Neo4j testcontainer — wipe with `MATCH (n) DETACH DELETE n` before seeding.

---

## File Structure

- **Modify** `src/graph_extract/config.py` — `theme_refresh_jaccard_tau`.
- **Create** `src/theme_builder/incremental.py` — `PersistedCommunity`, `_jaccard`, `match_communities`, `classify` (pure); `touched_entities`, `load_persisted`, `prev_corpus_cursor` (Neo4j).
- **Modify** `src/theme_builder/writeback.py` — add `write_communities_incremental`.
- **Modify** `src/theme_builder/cli.py` — `_run_theme_build_incremental` + `--full` flag.
- **Tests:** `tests/unit/test_config.py` (extend), `tests/unit/test_incremental_match.py`, `tests/integration/test_incremental_queries.py`, `tests/integration/test_incremental_writeback.py`, `tests/integration/test_incremental_cli.py`, `tests/integration/test_incremental_live.py`.

---

### Task 1: Config field

**Files:**
- Modify: `src/graph_extract/config.py`
- Test: `tests/unit/test_config.py` (extend)

**Interfaces:**
- Produces on `ExtractSettings`: `theme_refresh_jaccard_tau: float`.

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_config.py`)

```python
def test_theme_refresh_tau_default():
    from graph_extract.config import ExtractSettings
    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    assert s.theme_refresh_jaccard_tau == 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_config.py::test_theme_refresh_tau_default -q`
Expected: FAIL (field missing).

- [ ] **Step 3: Implement** — in `src/graph_extract/config.py`, inside `class ExtractSettings`, after the theme-builder / leiden fields block, add:

```python
    # --- incremental community refresh (design: incremental-community-refresh) ---
    theme_refresh_jaccard_tau: float = 0.5   # min member-set Jaccard to treat a fresh community as the same as a persisted one
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_config.py -q`
Expected: PASS. ruff + mypy clean on `src/graph_extract/config.py`.

- [ ] **Step 5: Commit**

```bash
git add src/graph_extract/config.py tests/unit/test_config.py
git commit -m "feat(config): theme_refresh_jaccard_tau"
```

---

### Task 2: `incremental.py` — pure matching & classification

**Files:**
- Create: `src/theme_builder/incremental.py`
- Test: `tests/unit/test_incremental_match.py`

**Interfaces:**
- Consumes: `theme_builder.detect.Community` (`community_id, level, member_uuids, parent_id`).
- Produces:
  - `@dataclass PersistedCommunity: community_id: str; level: int; members: set[str]; title: str; summary: str; full_report: str; rating: float; rating_explanation: str; tags: list[str]; cited_fact_uuids: list[str]; embedding: list[float]; generated_at`
  - `match_communities(fresh: list[Community], persisted: list[PersistedCommunity], *, tau: float) -> dict[int, PersistedCommunity | None]`
  - `classify(fresh: list[Community], matches, touched: set[str] | None) -> tuple[list[int], list[int]]` → `(dirty_idxs, clean_idxs)`

- [ ] **Step 1: Write the failing test** (`tests/unit/test_incremental_match.py`)

```python
from theme_builder.detect import Community
from theme_builder.incremental import PersistedCommunity, match_communities, classify


def _fresh(cid, level, members):
    return Community(community_id=cid, level=level, member_uuids=list(members), parent_id=None)


def _pers(cid, level, members):
    return PersistedCommunity(community_id=cid, level=level, members=set(members),
                              title="t", summary="s", full_report="[]", rating=1.0,
                              rating_explanation="", tags=[], cited_fact_uuids=[],
                              embedding=[0.0], generated_at="old")


def test_exact_and_wobble_match():
    fresh = [_fresh("h1", 1, ["a", "b", "c", "d"])]     # 3/4 overlap with p1 -> J=0.6
    persisted = [_pers("stable1", 1, ["a", "b", "c", "e"])]
    m = match_communities(fresh, persisted, tau=0.5)
    assert m[0] is not None and m[0].community_id == "stable1"


def test_below_threshold_is_new():
    fresh = [_fresh("h1", 1, ["a", "b", "x", "y"])]     # 2/6 overlap -> J=0.33 < 0.5
    persisted = [_pers("stable1", 1, ["a", "b", "z", "w"])]
    assert match_communities(fresh, persisted, tau=0.5)[0] is None


def test_level_isolation():
    fresh = [_fresh("h1", 1, ["a", "b"])]
    persisted = [_pers("stable1", 0, ["a", "b"])]        # same members, different level
    assert match_communities(fresh, persisted, tau=0.5)[0] is None


def test_greedy_one_to_one():
    # two fresh both overlap one persisted; only the best-Jaccard fresh claims it
    fresh = [_fresh("h1", 1, ["a", "b", "c"]),           # J with p1 = 3/3 = 1.0
             _fresh("h2", 1, ["a", "b", "z"])]           # J with p1 = 2/4 = 0.5
    persisted = [_pers("stable1", 1, ["a", "b", "c"])]
    m = match_communities(fresh, persisted, tau=0.5)
    assert m[0] is not None and m[0].community_id == "stable1"
    assert m[1] is None                                  # persisted already claimed


def test_classify_dirty_and_clean():
    fresh = [_fresh("h1", 1, ["a", "b"]), _fresh("h2", 1, ["c", "d"]), _fresh("hnew", 1, ["x"])]
    p = _pers("s1", 1, ["a", "b"])
    q = _pers("s2", 1, ["c", "d"])
    matches = {0: p, 1: q, 2: None}
    dirty, clean = classify(fresh, matches, touched={"a"})   # community 0 has touched 'a'
    assert dirty == [0, 2]     # matched+touched, and the new one
    assert clean == [1]        # matched, untouched


def test_classify_cold_start_all_dirty():
    fresh = [_fresh("h1", 1, ["a"]), _fresh("h2", 1, ["b"])]
    dirty, clean = classify(fresh, {0: None, 1: None}, touched=None)
    assert dirty == [0, 1] and clean == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_incremental_match.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — create `src/theme_builder/incremental.py`:

```python
"""Incremental community refresh: decide which communities actually changed since
the last theme-build (via created_at timestamps) and match fresh Leiden communities
to persisted ones (Jaccard) so their stable ids carry across refreshes and only
new/changed communities are LLM-regenerated."""
from __future__ import annotations

from dataclasses import dataclass

from neo4j import AsyncDriver

from theme_builder.detect import Community


@dataclass
class PersistedCommunity:
    community_id: str
    level: int
    members: set[str]
    title: str
    summary: str
    full_report: str
    rating: float
    rating_explanation: str
    tags: list[str]
    cited_fact_uuids: list[str]
    embedding: list[float]
    generated_at: object          # neo4j DateTime; passed back unchanged on reuse


def _jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def match_communities(fresh: list[Community], persisted: list[PersistedCommunity],
                      *, tau: float) -> dict[int, "PersistedCommunity | None"]:
    """Greedy 1:1 Jaccard match of fresh->persisted within the same level. Each
    fresh community maps to its best-overlapping persisted community (>= tau) or
    None (new); a persisted community is claimed by at most one fresh."""
    cands: list[tuple[float, int, int]] = []
    for fi, fc in enumerate(fresh):
        fmembers = set(fc.member_uuids)
        for pi, pc in enumerate(persisted):
            if pc.level != fc.level:
                continue
            j = _jaccard(fmembers, pc.members)
            if j >= tau:
                cands.append((j, fi, pi))
    cands.sort(key=lambda t: t[0], reverse=True)
    out: dict[int, "PersistedCommunity | None"] = {i: None for i in range(len(fresh))}
    used_fresh: set[int] = set()
    used_persisted: set[int] = set()
    for _j, fi, pi in cands:
        if fi in used_fresh or pi in used_persisted:
            continue
        out[fi] = persisted[pi]
        used_fresh.add(fi)
        used_persisted.add(pi)
    return out


def classify(fresh: list[Community], matches: dict[int, "PersistedCommunity | None"],
             touched: set[str] | None) -> tuple[list[int], list[int]]:
    """Split fresh community indexes into (dirty, clean). touched=None -> all dirty
    (cold start). A matched community is clean iff none of its members were touched."""
    dirty: list[int] = []
    clean: list[int] = []
    for i, fc in enumerate(fresh):
        if touched is None or matches[i] is None:
            dirty.append(i)
        elif set(fc.member_uuids) & touched:
            dirty.append(i)
        else:
            clean.append(i)
    return dirty, clean
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_incremental_match.py -q`
Expected: PASS (6 tests). `uv run ruff check src/theme_builder/incremental.py tests/unit/test_incremental_match.py` + `uv run mypy src/theme_builder/incremental.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/theme_builder/incremental.py tests/unit/test_incremental_match.py
git commit -m "feat(theme-builder): Jaccard community matching + dirty/clean classify"
```

---

### Task 3: `incremental.py` — graph queries

**Files:**
- Modify: `src/theme_builder/incremental.py`
- Test: `tests/integration/test_incremental_queries.py`

**Interfaces:**
- Produces:
  - `async touched_entities(driver, group_id, prev_cursor: str | None) -> set[str] | None`
  - `async load_persisted(driver, group_id) -> list[PersistedCommunity]`
  - `async prev_corpus_cursor(driver, group_id) -> str | None`

- [ ] **Step 1: Write the failing test** (`tests/integration/test_incremental_queries.py`)

```python
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

G = "backup-docs"


async def test_touched_entities_none_cursor_returns_none(extract_driver):
    from theme_builder.incremental import touched_entities
    out = await touched_entities(extract_driver, G, None)
    assert out is None


async def test_touched_entities_from_created_at(extract_driver):
    from theme_builder.incremental import touched_entities
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        # e_new created after cursor; e_old before; a new fact touches e_a and e_b
        await s.run("CREATE (:Entity {group_id:$g, uuid:'e_new', created_at: datetime('2026-02-01')})", g=G)
        await s.run("CREATE (:Entity {group_id:$g, uuid:'e_old', created_at: datetime('2026-01-01')})", g=G)
        await s.run("CREATE (a:Entity {group_id:$g, uuid:'e_a', created_at: datetime('2026-01-01')}), "
                    "(b:Entity {group_id:$g, uuid:'e_b', created_at: datetime('2026-01-01')}) "
                    "CREATE (a)-[:RELATES_TO {group_id:$g, uuid:'f1', created_at: datetime('2026-02-01')}]->(b)", g=G)
    out = await touched_entities(extract_driver, G, "2026-01-15T00:00:00Z")
    assert out == {"e_new", "e_a", "e_b"}      # e_old excluded; fact endpoints included


async def test_load_persisted_and_prev_cursor(extract_driver):
    from theme_builder.incremental import load_persisted, prev_corpus_cursor
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (c:Community {group_id:$g, community_id:'s1', level:1, title:'T', "
                    "summary:'S', full_report:'[]', rating:7.0, rating_explanation:'why', "
                    "tags:['x'], cited_fact_uuids:['fa'], embedding:[1.0,2.0], "
                    "generated_at: datetime('2026-03-01'), corpus_cursor:'2026-03-01T00:00:00Z', member_count:2}) "
                    "CREATE (e1:Entity {group_id:$g, uuid:'e1'})-[:IN_COMMUNITY]->(c) "
                    "CREATE (e2:Entity {group_id:$g, uuid:'e2'})-[:IN_COMMUNITY]->(c)", g=G)
    persisted = await load_persisted(extract_driver, G)
    assert len(persisted) == 1
    p = persisted[0]
    assert p.community_id == "s1" and p.level == 1
    assert p.members == {"e1", "e2"}
    assert p.title == "T" and p.rating == 7.0 and p.cited_fact_uuids == ["fa"]
    assert p.embedding == [1.0, 2.0]
    assert (await prev_corpus_cursor(extract_driver, G)) == "2026-03-01T00:00:00Z"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/integration/test_incremental_queries.py -q`
Expected: FAIL (functions missing).

- [ ] **Step 3: Implement** — append to `src/theme_builder/incremental.py`:

```python
async def touched_entities(driver: AsyncDriver, group_id: str,
                           prev_cursor: str | None) -> set[str] | None:
    """Entity uuids whose data changed since prev_cursor: entities created after it,
    plus both endpoints of RELATES_TO facts created after it. prev_cursor=None
    (no prior layer) -> None, meaning 'treat everything as new'."""
    if prev_cursor is None:
        return None
    touched: set[str] = set()
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) WHERE e.created_at > datetime($c) "
            "RETURN e.uuid AS uuid", g=group_id, c=prev_cursor)
        touched |= {rec["uuid"] async for rec in r}
        r = await s.run(
            "MATCH (a:Entity {group_id:$g})-[f:RELATES_TO {group_id:$g}]->"
            "(b:Entity {group_id:$g}) WHERE f.created_at > datetime($c) "
            "RETURN a.uuid AS a, b.uuid AS b", g=group_id, c=prev_cursor)
        async for rec in r:
            touched.add(rec["a"])
            touched.add(rec["b"])
    return touched


async def load_persisted(driver: AsyncDriver, group_id: str) -> list[PersistedCommunity]:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (c:Community {group_id:$g}) "
            "OPTIONAL MATCH (c)<-[:IN_COMMUNITY]-(e:Entity) "
            "WITH c, collect(e.uuid) AS members "
            "RETURN c.community_id AS community_id, c.level AS level, members, "
            "c.title AS title, coalesce(c.summary,'') AS summary, "
            "coalesce(c.full_report,'[]') AS full_report, coalesce(c.rating,0.0) AS rating, "
            "coalesce(c.rating_explanation,'') AS rating_explanation, "
            "coalesce(c.tags,[]) AS tags, coalesce(c.cited_fact_uuids,[]) AS cited_fact_uuids, "
            "c.embedding AS embedding, c.generated_at AS generated_at", g=group_id)
        return [PersistedCommunity(
            community_id=x["community_id"], level=x["level"], members=set(x["members"]),
            title=x["title"], summary=x["summary"], full_report=x["full_report"],
            rating=x["rating"], rating_explanation=x["rating_explanation"], tags=x["tags"],
            cited_fact_uuids=x["cited_fact_uuids"], embedding=x["embedding"],
            generated_at=x["generated_at"]) async for x in r]


async def prev_corpus_cursor(driver: AsyncDriver, group_id: str) -> str | None:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (c:Community {group_id:$g}) RETURN toString(max(c.corpus_cursor)) AS c",
            g=group_id)
        rec = await r.single()
        return rec["c"] if rec else None
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/integration/test_incremental_queries.py -q`
Expected: PASS. ruff + mypy clean on the module.

- [ ] **Step 5: Commit**

```bash
git add src/theme_builder/incremental.py tests/integration/test_incremental_queries.py
git commit -m "feat(theme-builder): touched-entity set + persisted-community load"
```

---

### Task 4: `write_communities_incremental`

**Files:**
- Modify: `src/theme_builder/writeback.py`
- Test: `tests/integration/test_incremental_writeback.py`

**Interfaces:**
- Produces: `async write_communities_incremental(driver, group_id, entries: list[dict], *, corpus_cursor: str | None) -> dict`. Each `entry`: `{community_id, level, member_uuids, parent_id, title, summary, full_report, rating, rating_explanation, tags, cited_fact_uuids, embedding, generated_at}`. Returns `{reports_written, by_level}`.

- [ ] **Step 1: Write the failing test** (`tests/integration/test_incremental_writeback.py`)

```python
import pytest
from datetime import datetime, timezone

pytestmark = pytest.mark.asyncio(loop_scope="module")

G = "backup-docs"


def _entry(cid, members, ga, parent=None, level=1, emb=None):
    return {"community_id": cid, "level": level, "member_uuids": list(members),
            "parent_id": parent, "title": f"T-{cid}", "summary": "S", "full_report": "[]",
            "rating": 5.0, "rating_explanation": "", "tags": [], "cited_fact_uuids": [],
            "embedding": emb or [0.1], "generated_at": ga}


async def test_incremental_writeback_reuse_dirty_and_dissolve(extract_driver):
    from theme_builder.writeback import write_communities_incremental
    old = datetime(2026, 3, 1, tzinfo=timezone.utc)
    new = datetime(2026, 4, 1, tzinfo=timezone.utc)
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        # a stale community that will be DISSOLVED (absent from entries)
        await s.run("CREATE (:Community {group_id:$g, community_id:'gone', level:1})", g=G)
        await s.run("CREATE (:Entity {group_id:$g, uuid:'m1'}), (:Entity {group_id:$g, uuid:'m2'})", g=G)
    entries = [
        _entry("clean1", ["m1"], old, emb=[9.0]),          # reused: old ga + old emb
        _entry("dirty1", ["m2"], new, emb=[7.0], parent="clean1"),
    ]
    res = await write_communities_incremental(extract_driver, G, entries, corpus_cursor="2026-04-01T00:00:00Z")
    assert res["reports_written"] == 2
    async with extract_driver.session() as s:
        r = await s.run("MATCH (c:Community {group_id:$g}) RETURN c.community_id AS cid, "
                        "toString(c.generated_at) AS ga, c.embedding AS emb, "
                        "toString(c.corpus_cursor) AS cur ORDER BY cid", g=G)
        rows = {x["cid"]: x async for x in r}
        assert set(rows) == {"clean1", "dirty1"}           # 'gone' dissolved
        assert rows["clean1"]["ga"].startswith("2026-03-01")   # reused old generated_at
        assert rows["clean1"]["emb"] == [9.0]
        assert rows["dirty1"]["ga"].startswith("2026-04-01")   # fresh generated_at
        assert rows["clean1"]["cur"].startswith("2026-04-01")  # run watermark on all
        # IN_COMMUNITY + PARENT_OF rebuilt
        r = await s.run("MATCH (:Entity {uuid:'m1'})-[:IN_COMMUNITY]->(c) RETURN c.community_id AS cid", )
        assert (await r.single())["cid"] == "clean1"
        r = await s.run("MATCH (:Community {community_id:'clean1'})-[:PARENT_OF]->(c) RETURN c.community_id AS cid")
        assert (await r.single())["cid"] == "dirty1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/integration/test_incremental_writeback.py -q`
Expected: FAIL (function missing).

- [ ] **Step 3: Implement** — append to `src/theme_builder/writeback.py`:

```python
async def write_communities_incremental(driver: AsyncDriver, group_id: str,
                                        entries: list[dict], *,
                                        corpus_cursor: str | None) -> dict:
    """Atomic rewrite of the :Community layer from complete entries (each carrying
    its stable id, members, report fields, embedding, and generated_at). Clean
    entries carry their prior embedding/generated_at; dirty ones carry fresh values.
    Communities absent from `entries` are dropped (dissolved)."""
    by_level: dict[int, int] = {}
    for e in entries:
        by_level[e["level"]] = by_level.get(e["level"], 0) + 1

    async def _rebuild(tx):
        await tx.run("MATCH (c:Community {group_id:$g}) DETACH DELETE c", g=group_id)
        for e in entries:
            await tx.run(
                "MERGE (c:Community {community_id:$cid, group_id:$g}) "
                "SET c += {level:$level, title:$title, summary:$summary, "
                "full_report:$full_report, rating:$rating, rating_explanation:$re, "
                "tags:$tags, cited_fact_uuids:$cited, embedding:$emb, "
                "member_count:$mc, generated_at:$ga, corpus_cursor:$cur}",
                cid=e["community_id"], g=group_id, level=e["level"], title=e["title"],
                summary=e["summary"], full_report=e["full_report"], rating=e["rating"],
                re=e["rating_explanation"], tags=e["tags"], cited=e["cited_fact_uuids"],
                emb=e["embedding"], mc=len(e["member_uuids"]), ga=e["generated_at"],
                cur=corpus_cursor)
            await tx.run(
                "MATCH (c:Community {community_id:$cid, group_id:$g}) "
                "UNWIND $members AS mu MATCH (ent:Entity {uuid:mu, group_id:$g}) "
                "MERGE (ent)-[:IN_COMMUNITY]->(c)",
                cid=e["community_id"], g=group_id, members=e["member_uuids"])
        for e in entries:
            if e.get("parent_id"):
                await tx.run(
                    "MATCH (p:Community {community_id:$pid, group_id:$g}), "
                    "(c:Community {community_id:$cid, group_id:$g}) "
                    "MERGE (p)-[:PARENT_OF]->(c)",
                    pid=e["parent_id"], cid=e["community_id"], g=group_id)

    async with driver.session() as s:
        await s.execute_write(_rebuild)
    return {"reports_written": len(entries), "by_level": by_level}
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/integration/test_incremental_writeback.py -q`
Expected: PASS. ruff + mypy clean on `src/theme_builder/writeback.py`.

- [ ] **Step 5: Commit**

```bash
git add src/theme_builder/writeback.py tests/integration/test_incremental_writeback.py
git commit -m "feat(theme-builder): atomic incremental writeback (reuse/dirty/dissolve)"
```

---

### Task 5: CLI incremental orchestration + `--full` flag

**Files:**
- Modify: `src/theme_builder/cli.py`
- Test: `tests/integration/test_incremental_cli.py`

**Interfaces:**
- Consumes: `detect_communities`, `load_persisted`, `touched_entities`, `prev_corpus_cursor`, `match_communities`, `classify`, `write_communities_incremental`, `generate_report`, `build_embedder`, `_report_client_and_model`, `_fetch_members`, `_fetch_facts`, `assemble_context`, `_corpus_cursor`.
- Produces: `async _run_theme_build_incremental(settings, *, driver) -> dict`; `theme-build --full` routes to the existing `_run_theme_build`.

- [ ] **Step 1: Write the failing test** (`tests/integration/test_incremental_cli.py`)

```python
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

G = "backup-docs"


class _FakeEmbedder:
    class _Client:
        async def close(self):
            pass
    def __init__(self):
        self.client = _FakeEmbedder._Client()
    async def create_batch(self, texts):
        return [[0.5] for _ in texts]


class _FakeClient:
    async def close(self):
        pass


def _settings():
    from graph_extract.config import ExtractSettings
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


async def _seed_persisted_and_entities(driver, *, e_a_created):
    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        # persisted A {e1,e2}, B {e3,e4}, corpus_cursor T0
        for cid, members in (("sA", ["e1", "e2"]), ("sB", ["e3", "e4"])):
            await s.run("CREATE (c:Community {group_id:$g, community_id:$cid, level:1, title:'old', "
                        "summary:'old', full_report:'[]', rating:1.0, rating_explanation:'', tags:[], "
                        "cited_fact_uuids:[], embedding:[0.0], generated_at: datetime('2026-03-01'), "
                        "corpus_cursor:'2026-03-01T00:00:00Z', member_count:2})", g=G, cid=cid)
            for mu in members:
                await s.run("MATCH (c:Community {community_id:$cid, group_id:$g}) "
                            "MERGE (e:Entity {uuid:$mu, group_id:$g}) MERGE (e)-[:IN_COMMUNITY]->(c)",
                            cid=cid, g=G, mu=mu)
        # entity created_at: e1 varies (touch A or not), others old; plus an Episodic for _corpus_cursor
        await s.run("MATCH (e:Entity {group_id:$g}) SET e.created_at = datetime('2026-01-01')", g=G)
        await s.run("MATCH (e:Entity {uuid:'e1', group_id:$g}) SET e.created_at = datetime($t)", g=G, t=e_a_created)
        await s.run("CREATE (:Episodic {group_id:$g, uuid:'ep1', created_at: datetime('2026-04-01')})", g=G)


def _patch(monkeypatch):
    import theme_builder.cli as cli
    from theme_builder.detect import Community
    from theme_builder.report import CommunityReport
    calls = {"n": 0}

    async def _fake_detect(driver, group_id, *, min_community_size, max_levels):
        return [Community("hA", 1, ["e1", "e2"], None), Community("hB", 1, ["e3", "e4"], None)]

    async def _fake_generate(client, model, ctx):
        calls["n"] += 1
        return CommunityReport(title="NEW", summary="NEW", full_report="[]", rating=9.0,
                               rating_explanation="", tags=[], cited_fact_uuids=[])

    monkeypatch.setattr(cli, "detect_communities", _fake_detect)
    monkeypatch.setattr(cli, "generate_report", _fake_generate)
    monkeypatch.setattr(cli, "build_embedder", lambda s: _FakeEmbedder())
    monkeypatch.setattr(cli, "_report_client_and_model", lambda s: (_FakeClient(), "m"))
    # context assembly is irrelevant (generate_report is faked); stub it so the
    # real _fetch_members/assemble_context path can't crash on unnamed seed entities.
    monkeypatch.setattr(cli, "assemble_context", lambda *a, **k: None)
    return calls


async def test_incremental_regenerates_only_touched(extract_driver, monkeypatch):
    import theme_builder.cli as cli
    calls = _patch(monkeypatch)
    await _seed_persisted_and_entities(extract_driver, e_a_created="2026-04-15")   # e1 touched (> T0)
    res = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)
    assert calls["n"] == 1                    # only community A regenerated
    assert res["reports_regenerated"] == 1 and res["reports_reused"] == 1
    async with extract_driver.session() as s:
        r = await s.run("MATCH (c:Community {group_id:$g, community_id:'sA'}) RETURN c.title AS t, toString(c.generated_at) AS ga", g=G)
        a = await r.single()
        assert a["t"] == "NEW" and a["ga"].startswith("2026-04")     # A regenerated, ga advanced
        r = await s.run("MATCH (c:Community {group_id:$g, community_id:'sB'}) RETURN c.title AS t, toString(c.generated_at) AS ga", g=G)
        b = await r.single()
        assert b["t"] == "old" and b["ga"].startswith("2026-03")     # B reused, ga unchanged (stable id kept)


async def test_incremental_no_change_regenerates_nothing(extract_driver, monkeypatch):
    import theme_builder.cli as cli
    calls = _patch(monkeypatch)
    await _seed_persisted_and_entities(extract_driver, e_a_created="2026-01-01")   # nothing after T0
    res = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)
    assert calls["n"] == 0                    # THE headline: zero LLM report calls
    assert res["reports_reused"] == 2 and res["reports_regenerated"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/integration/test_incremental_cli.py -q`
Expected: FAIL (`_run_theme_build_incremental` missing).

- [ ] **Step 3: Implement** — in `src/theme_builder/cli.py`:

Add imports near the top:
```python
from datetime import datetime, timezone

from theme_builder.incremental import (
    classify, load_persisted, match_communities, prev_corpus_cursor, touched_entities)
from theme_builder.writeback import write_communities, write_communities_incremental
```
(the existing `from theme_builder.writeback import write_communities` line becomes the combined import above.)

Add the orchestrator (after `_run_theme_build`):
```python
async def _run_theme_build_incremental(settings: ExtractSettings, *, driver: AsyncDriver) -> dict:
    communities = await detect_communities(
        driver, settings.group_id,
        min_community_size=settings.leiden_min_community_size,
        max_levels=settings.leiden_max_levels)
    persisted = await load_persisted(driver, settings.group_id)
    prev_cursor = await prev_corpus_cursor(driver, settings.group_id)
    touched = await touched_entities(driver, settings.group_id, prev_cursor)
    matches = match_communities(communities, persisted,
                                tau=settings.theme_refresh_jaccard_tau)
    dirty, clean = classify(communities, matches, touched)

    # stable id per fresh community + content-hash -> stable id map (to remap parents)
    stable_ids = {i: (matches[i].community_id if matches[i] is not None else c.community_id)
                  for i, c in enumerate(communities)}
    id_map = {communities[i].community_id: stable_ids[i] for i in range(len(communities))}

    client, model = _report_client_and_model(settings)
    embedder = build_embedder(settings)
    entries: list[dict] = []
    regenerated = 0
    reused = 0
    skipped = 0
    now = datetime.now(timezone.utc)
    try:
        for i, c in enumerate(communities):
            base = {"community_id": stable_ids[i], "level": c.level,
                    "member_uuids": c.member_uuids,
                    "parent_id": id_map.get(c.parent_id) if c.parent_id else None}
            if i in clean:
                p = matches[i]
                assert p is not None
                entries.append({**base, "title": p.title, "summary": p.summary,
                                "full_report": p.full_report, "rating": p.rating,
                                "rating_explanation": p.rating_explanation, "tags": p.tags,
                                "cited_fact_uuids": p.cited_fact_uuids,
                                "embedding": p.embedding, "generated_at": p.generated_at})
                reused += 1
                continue
            try:
                members = await _fetch_members(driver, settings.group_id, c.member_uuids)
                facts = await _fetch_facts(driver, settings.group_id, c.member_uuids)
                ctx = assemble_context(members, facts, top_entities=settings.report_top_entities,
                                       token_budget=settings.report_token_budget)
                rep = await generate_report(client, model, ctx)
            except Exception:
                logger.exception("theme-build: community %s errored; skipping", stable_ids[i])
                rep = None
            if rep is None:
                skipped += 1
                continue
            emb = (await embedder.create_batch([f"{rep.title}\n{rep.summary}"]))[0]
            entries.append({**base, "title": rep.title, "summary": rep.summary,
                            "full_report": rep.full_report, "rating": rep.rating,
                            "rating_explanation": rep.rating_explanation, "tags": rep.tags,
                            "cited_fact_uuids": rep.cited_fact_uuids,
                            "embedding": emb, "generated_at": now})
            regenerated += 1
        new_cursor = await _corpus_cursor(driver, settings.group_id)
        res = await write_communities_incremental(driver, settings.group_id, entries,
                                                  corpus_cursor=new_cursor)
        matched_persisted = sum(1 for i in matches if matches[i] is not None)
        res.update({"communities_detected": len(communities),
                    "reports_regenerated": regenerated, "reports_reused": reused,
                    "reports_skipped": skipped,
                    "communities_dissolved": len(persisted) - matched_persisted})
        return res
    finally:
        await client.close()
        await embedder.client.close()
```

Change the `theme-build` command to add the `--full` flag and route:
```python
@app.command("theme-build")
def theme_build(full: bool = typer.Option(
        False, "--full", help="Full rebuild (regenerate every report) instead of incremental.")) -> None:
    """Refresh the community-report layer (incremental by default; --full rebuilds all)."""
    async def _main() -> None:
        settings = get_extract_settings()
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        try:
            runner = _run_theme_build if full else _run_theme_build_incremental
            res = await runner(settings, driver=driver)
            typer.echo(json.dumps(res, indent=2, default=str))
        finally:
            await driver.close()

    asyncio.run(_main())
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/integration/test_incremental_cli.py -q`
Expected: PASS (2 tests — only-touched regenerates; no-change → 0 LLM calls). `uv run ruff check src/theme_builder/cli.py tests/integration/test_incremental_cli.py` + `uv run mypy src/theme_builder/cli.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/theme_builder/cli.py tests/integration/test_incremental_cli.py
git commit -m "feat(theme-builder): incremental theme-build orchestration + --full flag"
```

---

### Task 6: Full-suite gate + `@live` incremental demonstration

**Files:**
- Create: `tests/integration/test_incremental_live.py`, `docs/superpowers/incremental-refresh-report.md`

- [ ] **Step 1: Full non-live suite + CI lint gate**

Run: `uv run --extra dev pytest -m "not live" -q` → all pass.
Run: `uv run ruff check src tests` → clean. `uv run mypy src` → clean.
Fix any fallout before continuing.

- [ ] **Step 2: `@live` incremental demonstration** (`tests/integration/test_incremental_live.py`) — runs the real GDS detection + real report tier on `backup-docs`

```python
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_incremental_no_change_regenerates_nothing_live(live_extract_driver):
    # Assumes a populated :Community layer exists on backup-docs. A second
    # incremental run with no new content must regenerate zero reports.
    import theme_builder.cli as cli
    from graph_extract.config import get_extract_settings
    s = get_extract_settings()
    res = await cli._run_theme_build_incremental(s, driver=live_extract_driver)
    assert res["reports_regenerated"] == 0, res
    assert res["reports_reused"] > 0, res
```

Run: `uv run --extra dev pytest -m live tests/integration/test_incremental_live.py -q`
Expected: PASS (the existing `backup-docs` community layer is reused wholesale — zero regeneration).

- [ ] **Step 3: Controller demonstration**

Run `theme-build` (incremental) on `backup-docs` twice — once as-is (no change → 0 regenerated), and once after touching one community (e.g. bump a member entity's `created_at`) to show only that community regenerates while the rest are reused with unchanged `generated_at` and stable ids. Write `docs/superpowers/incremental-refresh-report.md` with the run stats (`reports_regenerated`/`reports_reused`/`communities_dissolved`) and the before/after `generated_at` on a reused vs a regenerated community.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_incremental_live.py docs/superpowers/incremental-refresh-report.md
git commit -m "test(incremental): live no-change smoke + demonstration report"
```

---

## Notes for the implementer

- `match_communities`/`classify`/`_jaccard` are pure — unit-tested with no DB. The graph functions (`touched_entities`/`load_persisted`/`prev_corpus_cursor`) and the orchestrator are integration-tested against a Neo4j testcontainer; the CLI test monkeypatches `detect_communities` (avoids GDS, which the testcontainer lacks), `generate_report` (a call counter), `build_embedder`, and `_report_client_and_model`.
- **Parent remapping is essential:** `detect` emits content-hash `parent_id`s, but matched communities inherit *stable* ids. The orchestrator builds `id_map` (content-hash → stable) and remaps each entry's `parent_id` before writeback — otherwise `PARENT_OF` links to nonexistent ids.
- The cost win is only at report generation: detection + graph writes still run fully. The headline test asserts a no-change run makes **zero** `generate_report` calls.
- `generated_at` is a real value per entry (dirty → `datetime.now(timezone.utc)`; clean → the persisted `generated_at` passed straight back); `corpus_cursor` is the run watermark on all communities.
- Test fakes: one statement per line, no semicolons (E702); imports at file top (E402). Integration tests wipe the module-scoped container per test.
- `--full` preserves today's exact behavior (`_run_theme_build` + `write_communities` untouched).
