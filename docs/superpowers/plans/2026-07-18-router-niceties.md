# `/answer` Router Niceties Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three deferred `/answer` router enhancements — a freshness block, richer citations (vendor/product/section + valid_at/invalid_at), and a global→local fallback arc.

**Architecture:** A small `answer_api/freshness.py` helper the router stamps onto the envelope; a single enrichment of `Provenance.resolve_citations` (structural vendor/product + heading_path section + fact validity) with its four callers + the timeline render updated to the new return shape; a second fallback arc in `answer_router`.

**Tech Stack:** Python 3.12, FastAPI (`answer_api`), Neo4j 5.26 (Cypher over the structural + fact graph), pytest (unit + Neo4j testcontainer; `@live` for the real stack).

## Global Constraints

- **Design decision #2 — citations are graph traversal, never LLM:** every new citation field (vendor/product/section/valid_at/invalid_at) comes from graph traversal (structural layer + edge properties); no LLM authors any of it. The router still authors no prose and no URL.
- **`resolve_citations` new return shape:** `{uuid: {"valid_at": str|None, "invalid_at": str|None, "sources": [ {url, title, article_id, vendor, product, section} ]}}`. A fact uuid absent from the graph stays ABSENT from the dict (callers rely on `.get(uuid, {})`). vendor/product/section are `None` when structural parents / heading are missing (best-effort).
- **Freshness:** `{graph_cursor_time, reports_as_of}`; `graph_cursor_time` = latest `Episodic.created_at`; `reports_as_of` = latest `Community.generated_at`, non-null only when `reports=True` (global/drift by FINAL mode); a query failure yields `None`, never an error.
- **Fallback:** two independent one-shot arcs — `local`(retrieved==0)→drift and `global`(no `communities_used`)→local. Each dispatch escalates at most once (mutually exclusive `if/elif`); no chaining.
- Run with `uv`: `uv run --extra dev pytest …`, `uv run ruff check src tests` (CI gate — lints tests; **no semicolons in fakes (E702), imports at file top (E402)**), `uv run mypy src`. Integration tests share a **module-scoped** Neo4j testcontainer — wipe with `MATCH (n) DETACH DELETE n` before seeding.

---

## File Structure

- **Create** `src/answer_api/freshness.py` — the freshness helper.
- **Modify** `src/graph_extract/provenance.py` — enrich `resolve_citations` (new shape).
- **Modify** `src/answer_api/search.py`, `timeline.py`, `global_search.py`, `drift.py`, `synthesize.py`, `router.py` — update the citation-building sites for the new shape + carry the new fields.
- **Modify** `src/answer_api/router.py` — global→local fallback + freshness stamping.
- **Tests:** `tests/unit/test_freshness.py`, `tests/integration/test_resolve_citations.py` (rewrite for new shape + enrichment), `tests/unit/test_router_dispatch.py` (extend), `tests/integration/test_router_niceties_live.py`.

---

### Task 1: Freshness helper

**Files:**
- Create: `src/answer_api/freshness.py`
- Test: `tests/unit/test_freshness.py`

**Interfaces:**
- Produces: `async freshness(driver, group_id, *, reports: bool) -> dict` returning `{"graph_cursor_time": str|None, "reports_as_of": str|None}`.

- [ ] **Step 1: Write the failing test** (`tests/unit/test_freshness.py`)

```python
import pytest

pytestmark = pytest.mark.asyncio


class _FakeResult:
    def __init__(self, value):
        self._value = value
    async def single(self):
        return {"c": self._value}


class _FakeSession:
    def __init__(self, value):
        self._value = value
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False
    async def run(self, cypher, **kw):
        return _FakeResult(self._value)


class _FakeDriver:
    def __init__(self, values):
        self._values = list(values)
    def session(self):
        return _FakeSession(self._values.pop(0))


async def test_freshness_with_reports():
    from answer_api.freshness import freshness
    d = _FakeDriver(["2026-07-18T00:00:00Z", "2026-07-17T00:00:00Z"])
    out = await freshness(d, "backup-docs", reports=True)
    assert out["graph_cursor_time"] == "2026-07-18T00:00:00Z"
    assert out["reports_as_of"] == "2026-07-17T00:00:00Z"


async def test_freshness_without_reports_skips_community_query():
    from answer_api.freshness import freshness
    d = _FakeDriver(["2026-07-18T00:00:00Z"])   # only ONE value -> only one query allowed
    out = await freshness(d, "backup-docs", reports=False)
    assert out["graph_cursor_time"] == "2026-07-18T00:00:00Z"
    assert out["reports_as_of"] is None


async def test_freshness_resilient_on_error():
    from answer_api.freshness import freshness

    class _Boom:
        def session(self):
            raise RuntimeError("neo4j down")

    out = await freshness(_Boom(), "g", reports=True)
    assert out == {"graph_cursor_time": None, "reports_as_of": None}
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_freshness.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — create `src/answer_api/freshness.py`:

```python
"""Freshness stamps for the /answer envelope: how current the live graph and the
thematic report layer are. Resilient — a query failure yields None, never an
error (freshness must never fail an answer)."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


async def _max_time(driver, cypher: str, group_id: str) -> str | None:
    try:
        async with driver.session() as s:
            r = await s.run(cypher, g=group_id)
            rec = await r.single()
            return rec["c"] if rec else None
    except Exception:
        logger.warning("freshness query failed", exc_info=True)
        return None


async def freshness(driver, group_id, *, reports: bool) -> dict:
    graph_cursor_time = await _max_time(
        driver,
        "MATCH (e:Episodic {group_id:$g}) RETURN toString(max(e.created_at)) AS c",
        group_id)
    reports_as_of = None
    if reports:
        reports_as_of = await _max_time(
            driver,
            "MATCH (c:Community {group_id:$g}) RETURN toString(max(c.generated_at)) AS c",
            group_id)
    return {"graph_cursor_time": graph_cursor_time, "reports_as_of": reports_as_of}
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_freshness.py -q`
Expected: PASS (3 tests). `uv run ruff check src/answer_api/freshness.py tests/unit/test_freshness.py` + `uv run mypy src/answer_api/freshness.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/freshness.py tests/unit/test_freshness.py
git commit -m "feat(answer-api): freshness helper (graph_cursor_time + reports_as_of)"
```

---

### Task 2: Richer citations — enrich `resolve_citations` + all callers

**Files:**
- Modify: `src/graph_extract/provenance.py`, `src/answer_api/search.py`, `src/answer_api/timeline.py`, `src/answer_api/global_search.py`, `src/answer_api/drift.py`, `src/answer_api/synthesize.py`, `src/answer_api/router.py`
- Test: `tests/integration/test_resolve_citations.py` (rewrite)

**Interfaces:**
- Consumes: nothing new.
- Produces: `resolve_citations(fact_uuids) -> {uuid: {"valid_at": str|None, "invalid_at": str|None, "sources": [ {url, title, article_id, vendor, product, section} ]}}`. All mode citations become `{marker, fact_uuid, valid_at, invalid_at, sources:[…enriched]}` (`answer_local` also keeps `fact`).

This is an atomic cross-cutting change: the resolver's return shape changes, so every caller is updated in the same commit (the suite must be green at commit).

- [ ] **Step 1: Rewrite the resolver test** (`tests/integration/test_resolve_citations.py`, replace the whole file)

```python
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

GROUP = "backup-docs"


async def test_resolve_citations_batch(extract_driver):
    from graph_extract.provenance import Provenance
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (a:Article {id:'art1', source_url:'https://x/1', title:'T1'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1'})")
        await s.run("CREATE (:Episodic {uuid:'ep_dangling'})")   # no Article
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1']}]->(y:Entity)", g=GROUP)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f2', episodes:['ep_dangling']}]->(y:Entity)", g=GROUP)
    out = await Provenance(extract_driver).resolve_citations(["f1", "f2", "f_absent"])
    assert out["f1"]["sources"] == [
        {"url": "https://x/1", "title": "T1", "article_id": "art1",
         "section": None, "vendor": None, "product": None}]
    assert out["f1"]["valid_at"] is None and out["f1"]["invalid_at"] is None
    assert out["f2"] == {"valid_at": None, "invalid_at": None, "sources": []}
    assert "f_absent" not in out          # absent fact stays absent (caller contract)


async def test_resolve_citations_enriched(extract_driver):
    from graph_extract.provenance import Provenance
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run(
            "CREATE (v:Vendor {name:'Veeam'})-[:HAS_PRODUCT]->(p:Product {name:'B&R'})"
            "-[:HAS_SOURCE]->(src:Source)-[:HAS_ARTICLE]->"
            "(a:Article {id:'art1', source_url:'https://x/1', title:'Vault Lock'})"
            "-[:HAS_EPISODE {heading_path:'Backup > Vault Lock'}]->(:Episodic {uuid:'ep1'})")
        await s.run(
            "CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1'], "
            "valid_at: datetime('2023-01-01'), invalid_at: null}]->(y:Entity)", g=GROUP)
    out = await Provenance(extract_driver).resolve_citations(["f1"])
    assert out["f1"]["valid_at"].startswith("2023-01-01")
    assert out["f1"]["invalid_at"] is None
    src = out["f1"]["sources"][0]
    assert src["url"] == "https://x/1"
    assert src["vendor"] == "Veeam"
    assert src["product"] == "B&R"
    assert src["section"] == "Backup > Vault Lock"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/integration/test_resolve_citations.py -q`
Expected: FAIL (old shape returns a list, not the `{valid_at, invalid_at, sources}` dict).

- [ ] **Step 3: Implement the resolver** — in `src/graph_extract/provenance.py`, replace `resolve_citations`:

```python
    async def resolve_citations(self, fact_uuids: list[str]) -> dict[str, dict]:
        async with self._driver.session() as s:
            r = await s.run(
                "MATCH ()-[f:RELATES_TO]->() WHERE f.uuid IN $uuids "
                "OPTIONAL MATCH (a:Article)-[he:HAS_EPISODE]->(e:Episodic) "
                "  WHERE e.uuid IN f.episodes "
                "OPTIONAL MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->"
                "  (:Source)-[:HAS_ARTICLE]->(a) "
                "WITH f.uuid AS uuid, toString(f.valid_at) AS valid_at, "
                "     toString(f.invalid_at) AS invalid_at, "
                "     collect(DISTINCT CASE WHEN a IS NULL THEN NULL ELSE "
                "       {url:a.source_url, title:a.title, article_id:a.id, "
                "        section:he.heading_path, vendor:v.name, product:p.name} END) AS raw "
                "RETURN uuid, valid_at, invalid_at, [x IN raw WHERE x IS NOT NULL] AS sources",
                uuids=fact_uuids)
            return {rec["uuid"]: {"valid_at": rec["valid_at"],
                                  "invalid_at": rec["invalid_at"],
                                  "sources": rec["sources"]} async for rec in r}
```

- [ ] **Step 4: Update the callers**

**`src/answer_api/search.py`** — in `search_local`, change the results list comprehension so `sources` reads the new shape, and add `invalid_at`:
```python
    return {
        "query": q, "count": len(edges),
        "results": [{"fact": e.fact, "fact_uuid": e.uuid,
                     "valid_at": getattr(e, "valid_at", None),
                     "invalid_at": getattr(e, "invalid_at", None),
                     "sources": citations.get(e.uuid, {}).get("sources", [])} for e in edges],
    }
```

**`src/answer_api/timeline.py`** — in `timeline_local`, change the `sources` field:
```python
                      "sources": citations.get(e.uuid, {}).get("sources", [])} for e in edges],
```

**`src/answer_api/global_search.py`** — in the reduce (`global_search`), rebuild `citations`:
```python
    resolved = await Provenance(driver).resolve_citations(
        [marker_map[m]["fact_uuid"] for m in cited])
    citations = [{"marker": m, "fact_uuid": marker_map[m]["fact_uuid"],
                  "valid_at": resolved.get(marker_map[m]["fact_uuid"], {}).get("valid_at"),
                  "invalid_at": resolved.get(marker_map[m]["fact_uuid"], {}).get("invalid_at"),
                  "sources": resolved.get(marker_map[m]["fact_uuid"], {}).get("sources", [])}
                 for m in cited]
```

**`src/answer_api/drift.py`** — in `_synthesize`, rebuild `citations` the same way:
```python
    resolved = await Provenance(driver).resolve_citations(
        [marker_map[m]["fact_uuid"] for m in cited])
    citations = [{"marker": m, "fact_uuid": marker_map[m]["fact_uuid"],
                  "valid_at": resolved.get(marker_map[m]["fact_uuid"], {}).get("valid_at"),
                  "invalid_at": resolved.get(marker_map[m]["fact_uuid"], {}).get("invalid_at"),
                  "sources": resolved.get(marker_map[m]["fact_uuid"], {}).get("sources", [])}
                 for m in cited]
    return answer, citations
```

**`src/answer_api/synthesize.py`** — in `answer_local`, add validity to the citation (sources are already enriched via `search_local`'s results):
```python
    citations = [{"marker": m, "fact": marker_map[m]["fact"],
                  "fact_uuid": marker_map[m]["fact_uuid"],
                  "valid_at": marker_map[m].get("valid_at"),
                  "invalid_at": marker_map[m].get("invalid_at"),
                  "sources": marker_map[m]["sources"]} for m in cited]
```

**`src/answer_api/router.py`** — in `_render_timeline`, add validity to each citation:
```python
        citations.append({"marker": i, "fact_uuid": e["fact_uuid"],
                          "valid_at": e.get("valid_at"), "invalid_at": e.get("invalid_at"),
                          "sources": e.get("sources", [])})
```

- [ ] **Step 5: Run the resolver test + the affected regression suite**

Run: `uv run --extra dev pytest tests/integration/test_resolve_citations.py tests/integration/test_search_local.py tests/integration/test_timeline_local.py tests/integration/test_global_search.py tests/integration/test_drift.py tests/unit/test_router_normalize.py tests/unit/test_router_dispatch.py -q`
Expected: PASS. The other integration tests access sources by key (`sources[0]["url"]`/`["article_id"]`) so remain green; if any asserts an exact source-dict equality, add the three new keys (`section`/`vendor`/`product`: `None`) to that expected dict.
Then `uv run ruff check src/graph_extract/provenance.py src/answer_api/ tests/integration/test_resolve_citations.py` + `uv run mypy src/graph_extract/provenance.py src/answer_api` clean.

- [ ] **Step 6: Commit**

```bash
git add src/graph_extract/provenance.py src/answer_api/search.py src/answer_api/timeline.py src/answer_api/global_search.py src/answer_api/drift.py src/answer_api/synthesize.py src/answer_api/router.py tests/integration/test_resolve_citations.py
git commit -m "feat(citations): enrich resolve_citations with vendor/product/section + fact validity"
```

---

### Task 3: Router — global→local fallback + freshness stamping

**Files:**
- Modify: `src/answer_api/router.py`
- Test: `tests/unit/test_router_dispatch.py` (extend)

**Interfaces:**
- Consumes: `freshness.freshness`, `_dispatch`, `classify`, `_normalize`.
- Produces: `answer_router` gains the global→local fallback arc and stamps `env["freshness"]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_router_dispatch.py`)

Add `import answer_api.freshness as freshness_mod` to the file's TOP import block
(alongside the existing `import answer_api.global_search as global_mod` etc. — not
mid-file; ruff E402). Then add a fake freshness + an empty-communities global fake
near the other fakes:
```python
async def _fake_freshness(driver, group_id, *, reports):
    return {"graph_cursor_time": "2026-07-18T00:00:00Z",
            "reports_as_of": "2026-07-17T00:00:00Z" if reports else None}


async def _fake_global_empty(*a, **k):
    return {"query": k["q"], "answer": "refusal", "citations": [], "communities_used": []}
```
Extend the autouse `_patch_modes` fixture with:
```python
    monkeypatch.setattr(freshness_mod, "freshness", _fake_freshness)
```
Add the tests:
```python
async def test_envelope_carries_freshness_reports_for_drift():
    env = await _route("drift")
    assert env["freshness"]["graph_cursor_time"] == "2026-07-18T00:00:00Z"
    assert env["freshness"]["reports_as_of"] == "2026-07-17T00:00:00Z"


async def test_envelope_freshness_reports_none_for_local():
    env = await _route("local")
    assert env["freshness"]["reports_as_of"] is None      # local isn't community-based


async def test_global_empty_escalates_to_local(monkeypatch):
    monkeypatch.setattr(global_mod, "global_search", _fake_global_empty)
    env = await _route("global")
    assert env["mode"] == "local"
    assert env["routing"]["fallback_from"] == "global"
    assert env["citations"][0]["fact_uuid"] == "f1"       # local fake's citation
    assert env["freshness"]["reports_as_of"] is None      # final mode is local


async def test_global_with_communities_stays_global():
    env = await _route("global")
    assert env["mode"] == "global"
    assert env["routing"]["fallback_from"] is None
    assert env["freshness"]["reports_as_of"] == "2026-07-17T00:00:00Z"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_router_dispatch.py -k "freshness or global_empty or stays_global" -q`
Expected: FAIL (no `freshness` key; global-empty doesn't escalate).

- [ ] **Step 3: Implement** — in `src/answer_api/router.py`:

Add the import to the top block:
```python
from answer_api import freshness as freshness_mod
```
Replace the fallback + return in `answer_router` (after the initial `raw = await _dispatch(...)`):
```python
    fallback_from: str | None = None
    if mode == "local" and raw.get("retrieved") == 0:
        fallback_from = "local"
        mode = "drift"
        raw = await _dispatch("drift", graphiti, driver, embedder, synth_client,
                              synth_model, map_client, map_model, q=q, vendor=vendor,
                              settings=settings)
    elif mode == "global" and not raw.get("communities_used"):
        fallback_from = "global"
        mode = "local"
        raw = await _dispatch("local", graphiti, driver, embedder, synth_client,
                              synth_model, map_client, map_model, q=q, vendor=vendor,
                              settings=settings)
    env = _normalize(mode, via, fallback_from, raw, q)
    env["freshness"] = await freshness_mod.freshness(
        driver, settings.group_id, reports=mode in ("global", "drift"))
    return env
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_router_dispatch.py -q`
Expected: PASS (all router dispatch tests, incl. the 4 new). `uv run ruff check src/answer_api/router.py tests/unit/test_router_dispatch.py` + `uv run mypy src/answer_api/router.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/router.py tests/unit/test_router_dispatch.py
git commit -m "feat(router): global->local fallback + freshness stamping on the envelope"
```

---

### Task 4: Full-suite gate + `@live` smoke + demonstration

**Files:**
- Create: `tests/integration/test_router_niceties_live.py`, `docs/superpowers/router-niceties-report.md`

- [ ] **Step 1: Full non-live suite + CI lint gate**

Run: `uv run --extra dev pytest -m "not live" -q` → all pass.
Run: `uv run ruff check src tests` → clean. `uv run mypy src` → clean.
Fix any fallout (e.g. an integration test that asserted an exact source dict — add `section`/`vendor`/`product` keys) before continuing.

- [ ] **Step 2: `@live` smoke** (`tests/integration/test_router_niceties_live.py`)

```python
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_router_niceties_live(live_extract_driver):
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_embedder, build_graphiti
    from answer_api.synthesize import _synthesis_client_and_model
    from answer_api.global_search import _map_client_and_model
    from answer_api.router import answer_router, _cheap_classify_client
    s = get_extract_settings()
    graphiti = build_graphiti(s)
    emb = build_embedder(s)
    sc, sm = _synthesis_client_and_model(s)
    mc, mm = _map_client_and_model(s)
    cc, cmodel = _cheap_classify_client(s)
    try:
        # cross-vendor -> global (heuristic); has communities -> reports_as_of set
        env = await answer_router(graphiti, live_extract_driver, emb, sc, sm, mc, mm,
                                  cc, cmodel, q="Compare how AWS Backup and Azure Backup handle retention",
                                  mode_override=None, vendor=None, settings=s)
        assert env["freshness"]["graph_cursor_time"]          # live graph watermark present
        assert env["freshness"]["reports_as_of"]              # global is community-based
        assert "http" not in env["answer"]
        # at least one citation carries a resolvable source with structural attribution
        srcs = [src for c in env["citations"] for src in c["sources"]]
        assert srcs, "expected >=1 resolved source"
        assert any(src.get("vendor") or src.get("product") for src in srcs)
    finally:
        await graphiti.close()
        await sc.close()
        await mc.close()
        if cc is not None:
            await cc.close()
```

Run: `uv run --extra dev pytest -m live tests/integration/test_router_niceties_live.py -q`
Expected: PASS.

- [ ] **Step 3: Controller demonstration**

Call `answer_router` on `backup-docs` and write `docs/superpowers/router-niceties-report.md` showing: the `freshness` block (graph_cursor_time + reports_as_of for a global/drift query, reports_as_of null for a local one), a citation with `vendor`/`product`/`section` populated + `valid_at`/`invalid_at`, and a global→local fallback case (a query that shortlists no communities) with `routing.fallback_from="global"`.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_router_niceties_live.py docs/superpowers/router-niceties-report.md
git commit -m "test(router-niceties): live smoke + demonstration report"
```

---

## Notes for the implementer

- Task 2 is atomic: the resolver's return shape changes, so all citation-building sites update in one commit. Every consumer that reads `sources[0]["url"]`/`["article_id"]` still works (sources stays a list of dicts, just with more keys); only the internal `resolve_citations` return shape (`{uuid: {…, sources}}`) forces the caller edits.
- The "absent uuid stays absent" contract is preserved by `MATCH ()-[f:RELATES_TO]->() WHERE f.uuid IN $uuids` (no edge → no row → absent); callers use `.get(uuid, {})`.
- Freshness must never fail an answer: `_max_time` catches everything and returns `None`. The router stamps freshness using the FINAL (post-fallback) mode to decide `reports`.
- `freshness`/`answer_router` are module-attr seams; unit tests monkeypatch `freshness_mod.freshness` and the mode functions.
- Test fakes: one statement per line, no semicolons (E702); imports at file top (E402). Integration tests wipe the module-scoped container per test.
- Design #2: all new citation fields are graph-derived; no LLM authors them.
