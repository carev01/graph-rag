# Local Retrieval + Citation Precision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A deterministic `/search/local` FastAPI endpoint that hybrid-retrieves fact edges (RRF, no cross-encoder), filters by validity/vendor, resolves citations by graph traversal, and a golden-set harness measuring citation precision@k.

**Architecture:** New `answer_api` package. `search_local` (pure, testable) calls Graphiti's `EDGE_HYBRID_SEARCH_RRF` recipe → `EntityEdge`s → post-filters → batch citation resolve. FastAPI wraps it. No LLM at query time.

**Tech Stack:** Python 3.12, FastAPI, graphiti-core 0.29.2, Neo4j (async), TEI/Jina embeddings, pytest + testcontainers, ruff, mypy.

## Global Constraints

- **No LLM at query time.** Retrieval + graph-traversal citation only (design-decision #2: citations are traversal, never LLM; the LLM never writes a URL — there is no LLM here at all).
- **RRF edge recipe, no cross-encoder** (`EDGE_HYBRID_SEARCH_RRF`) — backend-agnostic; the cross-encoder reranker may be random on non-OpenAI backends.
- **One corpus-wide group** (`settings.group_id`); vendor scoping is via the STRUCTURAL layer (design-decision #4), not per-group.
- **Reuse** `graph_extract.config.ExtractSettings`, `graph_extract.graphiti_client.build_graphiti`, `graph_extract.provenance` — no duplicate config/graph plumbing.
- New `src/answer_api` package MUST be added to `pyproject.toml` `[tool.hatch.build.targets.wheel] packages` (like `src/docext`), then `uv sync --extra dev`.
- Secrets only in untracked `.env`; DB tests use testcontainers; ruff/mypy clean.

---

## File Structure

- `src/graph_extract/provenance.py` — add `resolve_citations` (batch).
- `src/answer_api/__init__.py`, `src/answer_api/search.py` (`search_local`), `src/answer_api/app.py` (`create_app`), `src/answer_api/golden.py` (precision calc), `src/answer_api/eval_golden.py` (script), `src/answer_api/golden_questions.json`.
- `pyproject.toml` — packages += `src/answer_api`.
- Tests: `tests/integration/test_resolve_citations.py`, `tests/integration/test_search_local.py`, `tests/unit/test_answer_api_app.py`, `tests/unit/test_golden.py`.
- Report (T4): `docs/superpowers/local-retrieval-golden-report.md`.

---

### Task 1: Batch citation resolver

**Files:**
- Modify: `src/graph_extract/provenance.py`
- Test: `tests/integration/test_resolve_citations.py`

**Interfaces:** `async def resolve_citations(self, fact_uuids: list[str]) -> dict[str, list[dict]]` — `{fact_uuid: [{url,title,article_id}, ...]}`; a fact with no resolvable Article → `[]`; a fact_uuid absent from the graph → not a key (or `[]` — pick and test).

- [ ] **Step 1: Write failing integration test** (Neo4j testcontainer, `extract_driver` fixture — see `tests/integration/test_provenance_rekey.py`). Seed: an Article with `source_url`/`title`/`id`, a HAS_EPISODE→Episodic, and a `RELATES_TO` fact whose `episodes` list includes that episode; plus a fact whose episode has no Article (dangling).

```python
async def test_resolve_citations_batch(extract_driver):
    from graph_extract.provenance import Provenance
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (a:Article {id:'art1', source_url:'https://x/1', title:'T1'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1'})")
        await s.run("CREATE (:Episodic {uuid:'ep_dangling'})")   # no Article
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1']}]->(y:Entity)", g=g)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f2', episodes:['ep_dangling']}]->(y:Entity)", g=g)
    prov = Provenance(extract_driver)
    out = await prov.resolve_citations(["f1", "f2"])
    assert out["f1"] == [{"url": "https://x/1", "title": "T1", "article_id": "art1"}]
    assert out["f2"] == []       # dangling -> empty, surfaced not dropped
```

- [ ] **Step 2: Run, confirm fail.** `uv run --extra dev pytest tests/integration/test_resolve_citations.py -v`.

- [ ] **Step 3: Implement `resolve_citations`** (one query; left-join so dangling facts return `[]`):

```python
async def resolve_citations(self, fact_uuids: list[str]) -> dict[str, list[dict]]:
    async with self._driver.session() as s:
        r = await s.run(
            "MATCH ()-[f:RELATES_TO]->() WHERE f.uuid IN $uuids "
            "OPTIONAL MATCH (a:Article)-[:HAS_EPISODE]->(e:Episodic) "
            "  WHERE e.uuid IN f.episodes "
            "WITH f.uuid AS uuid, "
            "     collect(DISTINCT CASE WHEN a IS NULL THEN NULL ELSE "
            "       {url:a.source_url, title:a.title, article_id:a.id} END) AS raw "
            "RETURN uuid, [x IN raw WHERE x IS NOT NULL] AS sources",
            uuids=fact_uuids)
        return {rec["uuid"]: rec["sources"] async for rec in r}
```
(Group not filtered here — fact uuids are unique; matches the existing `resolve_chain` style. Verify the `OPTIONAL MATCH ... WHERE e.uuid IN f.episodes` cardinality on the container; adjust to an `UNWIND f.episodes` form if cleaner.)

- [ ] **Step 4: Run tests green** + gate on `provenance.py`.

- [ ] **Step 5: Commit.** `git add src/graph_extract/provenance.py tests/integration/test_resolve_citations.py && git commit -m "feat(extract): batch resolve_citations (fact uuids -> grouped source citations)"`

---

### Task 2: `answer_api.search_local` (retrieval core)

**Files:**
- Create: `src/answer_api/__init__.py`, `src/answer_api/search.py`
- Modify: `pyproject.toml` (packages += `src/answer_api`)
- Test: `tests/integration/test_search_local.py`

**Interfaces:** `async def search_local(graphiti, driver, *, q, k=10, vendor=None, include_invalid=False, group_id) -> dict` → `{"query", "count", "results": [{"fact","fact_uuid","valid_at","sources":[...]}]}`.

- [ ] **Step 1: Add `src/answer_api` to pyproject + sync.** Edit `pyproject.toml` `packages` to include `"src/answer_api"`; create empty `src/answer_api/__init__.py`; `uv sync --extra dev`; confirm `uv run --extra dev python -c "import answer_api; print('ok')"`.

- [ ] **Step 2: Write failing tests** in `tests/integration/test_search_local.py`. Use a **stub graphiti** (a fake with `_search` returning a `SearchResults`-like object) for the deterministic post-filter logic, plus a real Neo4j `extract_driver` for citation resolution. Seed the same article/episode/fact graph as Task 1, and make the stub return edges referencing those fact uuids.

```python
class _Edge:
    def __init__(self, uuid, fact, episodes, valid_at=None, invalid_at=None):
        self.uuid, self.fact, self.episodes = uuid, fact, episodes
        self.valid_at, self.invalid_at = valid_at, invalid_at

class _Results:
    def __init__(self, edges): self.edges = edges

class _StubGraphiti:
    def __init__(self, edges): self._edges = edges
    async def _search(self, query, config, group_ids=None, **kw):
        return _Results(self._edges)

async def test_search_local_returns_cited_facts(extract_driver):
    # seed art1/ep1/f1 (valid) and f_invalid (invalid_at set) as in Task 1
    from answer_api.search import search_local
    g = _StubGraphiti([_Edge("f1","AWS Backup Vault Lock requires compliance mode",["ep1"]),
                       _Edge("f_invalid","stale fact",["ep1"], invalid_at="2020-01-01")])
    out = await search_local(g, extract_driver, q="vault lock", k=10, group_id="backup-docs")
    facts = {r["fact_uuid"] for r in out["results"]}
    assert "f1" in facts and "f_invalid" not in facts       # invalid excluded by default
    assert out["results"][0]["sources"][0]["article_id"] == "art1"

async def test_search_local_include_invalid(extract_driver):
    # include_invalid=True -> f_invalid present
    ...

async def test_search_local_vendor_scope(extract_driver):
    # seed a structural Vendor 'AWS' chain to ep1; vendor='AWS' keeps f1, vendor='Microsoft' drops it
    ...
```

- [ ] **Step 3: Run, confirm fail** (module missing).

- [ ] **Step 4: Implement `search.py`:**

```python
from __future__ import annotations
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graph_extract.provenance import Provenance

async def _vendor_episode_uuids(driver, vendor: str) -> set[str]:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (v:Vendor)-[:HAS_PRODUCT]->(:Product)-[:HAS_SOURCE]->(:Source)"
            "-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(e:Episodic) "
            "WHERE toLower(v.name)=toLower($v) RETURN collect(DISTINCT e.uuid) AS u", v=vendor)
        rec = await r.single()
        return set(rec["u"]) if rec else set()

async def search_local(graphiti, driver, *, q, k=10, vendor=None,
                       include_invalid=False, group_id):
    # over-fetch so post-filters still leave ~k
    config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    config.limit = max(k * 3, k)                      # verify the limit field name/mechanism vs installed graphiti
    results = await graphiti._search(q, config, group_ids=[group_id])
    edges = list(results.edges)
    if not include_invalid:
        edges = [e for e in edges if getattr(e, "invalid_at", None) is None]
    if vendor:
        scope = await _vendor_episode_uuids(driver, vendor)
        edges = [e for e in edges if scope.intersection(e.episodes or [])]
    edges = edges[:k]
    citations = await Provenance(driver).resolve_citations([e.uuid for e in edges])
    return {
        "query": q, "count": len(edges),
        "results": [{"fact": e.fact, "fact_uuid": e.uuid,
                     "valid_at": getattr(e, "valid_at", None),
                     "sources": citations.get(e.uuid, [])} for e in edges],
    }
```
(If `config.limit` isn't the right knob in the installed graphiti version, set the per-edge-search limit the recipe exposes; the requirement is capping fetch above k. Validate on the container/live.)

- [ ] **Step 5: Run tests green** + gate on `answer_api/search.py`.

- [ ] **Step 6: Add a `@live` smoke test** (marked `live`, deselected in the normal suite) hitting the real compose graph: `search_local(build_graphiti(settings), driver, q="What does AWS Backup Vault Lock require?", k=5, group_id=settings.group_id)` returns ≥1 result with a non-empty `sources`. (Controller runs this in T4; here just author it.)

- [ ] **Step 7: Commit.** `git add pyproject.toml src/answer_api/__init__.py src/answer_api/search.py tests/integration/test_search_local.py && git commit -m "feat(answer-api): search_local (RRF retrieve + validity/vendor filter + citations)"`

---

### Task 3: FastAPI app + `/search/local`

**Files:**
- Create: `src/answer_api/app.py`
- Test: `tests/unit/test_answer_api_app.py`

**Interfaces:** `create_app()` → FastAPI; `GET /health`; `GET /search/local?q=&k=&vendor=&include_invalid=`.

- [ ] **Step 1: Write failing FastAPI tests** (TestClient, mirror `tests/unit/test_app.py`). Inject a **stub `search_local`** (via a dependency override or a module-level seam) so the endpoint test needs no Neo4j: assert `/health` → 200; `/search/local?q=x` → 200 with the `{query,count,results}` shape; missing `q` → 422.

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement `app.py`.** `create_app()` builds a FastAPI with a lifespan that constructs one `Graphiti` (`build_graphiti(get_extract_settings())`) + one Neo4j `AsyncDriver`, stores them on `app.state`, closes both on shutdown (mirror `graph_sync/app.py`'s lifespan + `graph_extract/cli.py`'s build/close). `GET /search/local` reads `q`/`k`/`vendor`/`include_invalid` query params, calls `search_local(app.state.graphiti, app.state.driver, q=..., group_id=settings.group_id)`, returns the dict. `GET /health` → `{"status":"ok"}`. Make `search_local` injectable (module attribute or FastAPI dependency) so the test can stub it without a graph.

- [ ] **Step 4: Run tests green** + confirm the app imports and routes register. Gate.

- [ ] **Step 5: Commit.** `git add src/answer_api/app.py tests/unit/test_answer_api_app.py && git commit -m "feat(answer-api): FastAPI app + GET /search/local + /health"`

---

### Task 4: Golden-set harness + citation-precision report

**Files:**
- Create: `src/answer_api/golden.py`, `src/answer_api/golden_questions.json`, `src/answer_api/eval_golden.py`
- Test: `tests/unit/test_golden.py`
- Report (controller): `docs/superpowers/local-retrieval-golden-report.md`

**Interfaces:** `precision_at_k(results, expected_article_ids) -> bool` (hit if any expected id ∈ returned citations' article_ids); `first_hit_rank(...) -> int | None`; an aggregate over questions.

- [ ] **Step 1: Author `golden_questions.json`** — ~12–15 questions grounded in the CURRENT graph's pilot articles. Controller/implementer first inspects available articles:
```bash
uv run --extra dev python -c "
import asyncio; from neo4j import AsyncGraphDatabase as G; from graph_extract.config import get_extract_settings as S
async def m():
    s=S.__wrapped__(); d=G.driver(s.neo4j_uri,auth=(s.neo4j_user,s.neo4j_password))
    async with d.session() as x:
        r=await x.run(\"MATCH (a:Article) WHERE a.removed IS NULL OR a.removed=false RETURN a.id AS id, a.title AS t ORDER BY t\")
        [print(y['id'], '|', y['t']) async for y in x.run('MATCH (a:Article) RETURN a.id AS id, a.title AS t ORDER BY t')]
    await d.close()
asyncio.run(m())"
```
  Each entry: `{"question": "...", "expected_article_ids": ["<id>"], "vendor": null}`. Author them so the answer is unambiguously in one known article (e.g. Vault Lock, soft delete, cross-region, encryption, S3 restore).

- [ ] **Step 2: Write `golden.py` + a unit test** (`tests/unit/test_golden.py`, no graph) for the pure precision functions:
```python
def precision_at_k(results: list[dict], expected_article_ids: list[str]) -> bool:
    got = {s["article_id"] for r in results for s in r["sources"]}
    return any(e in got for e in expected_article_ids)

def first_hit_rank(results, expected_article_ids):
    for i, r in enumerate(results, 1):
        if any(s["article_id"] in expected_article_ids for s in r["sources"]):
            return i
    return None
```
Test with canned `results` dicts (hit, miss, rank).

- [ ] **Step 3: Write `eval_golden.py`** (script): load the questions, for each call `search_local(build_graphiti(settings), driver, q=question, k=..., vendor=..., group_id=...)`, compute `precision_at_k` + `first_hit_rank`, print per-question hit/miss + aggregate citation-precision@k and mean-first-hit-rank. Build/close graphiti+driver in a `finally`.

- [ ] **Step 4: Run the harness (controller, live) + write the report.** `uv run --extra dev python -m answer_api.eval_golden`. Capture the aggregate + per-question table into `docs/superpowers/local-retrieval-golden-report.md` with the headline **citation precision@k** (the Phase-2 exit signal) and a note on any misses (query phrasing vs RRF ranking). Also run the T2 `@live` smoke to confirm end-to-end.

- [ ] **Step 5: Commit.** `git add src/answer_api/golden.py src/answer_api/golden_questions.json src/answer_api/eval_golden.py tests/unit/test_golden.py docs/superpowers/local-retrieval-golden-report.md && git commit -m "feat(answer-api): golden-set harness + citation-precision report"`

---

## Self-Review Notes

- **Spec coverage:** batch resolver → T1; `search_local` (retrieval+filters+citations) → T2; FastAPI endpoint → T3; golden harness + precision report → T4. All §-components covered.
- **Deps:** T1 → T2 (search uses resolve_citations); T3 wraps T2; T4 uses T2. Order T1, T2, T3, T4.
- **Reranker risk:** handled by using `EDGE_HYBRID_SEARCH_RRF` (no cross-encoder); T2/T4 validate ranking on the real backend.
- **Type consistency:** `resolve_citations(list[str])->dict[str,list[dict]]`; `search_local(...group_id=...)->{query,count,results}`; `precision_at_k`/`first_hit_rank`.
- **Packaging:** `src/answer_api` added to pyproject + `uv sync` in T2 Step 1 (before any answer_api import runs).
- **No-placeholder check:** code steps carry real code; the two `...` test bodies (include_invalid, vendor_scope) explicitly mirror the shown test's pattern; the golden authoring step gives the exact inspection command.
