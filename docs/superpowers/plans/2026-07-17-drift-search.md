# DRIFT Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `GET /search/drift` — the broad→deep DRIFT motion (primer over community reports → graph-biased local follow-up loop → cited synthesis) for questions that are broad but expect concrete, sourced answers.

**Architecture:** Reuse the Phase-3 pieces: `shortlist_communities` (primer), `search_local` (follow-ups, extended with an optional center node for graph-distance reranking), `_finalize_answer` + `Provenance` (citations). New `answer_api/drift.py` orchestrates primer → follow-up loop (1–2 iterations) → synthesis; all LLM calls run on the synthesis tier (GLM). No LLM authors a URL.

**Tech Stack:** Python 3.12, FastAPI (`answer_api`), Neo4j 5.26, graphiti-core 0.29.2 (`EDGE_HYBRID_SEARCH_NODE_DISTANCE` recipe + `_search(center_node_uuid=…)`), OpenAI-compatible LLM (GLM-5.2 via `judge_*`), pytest (+ Neo4j testcontainer; live LLM/embed are `@live`).

## Global Constraints

- **Design decision #2 — citations are graph traversal, never LLM:** the only cited output is the synthesis answer; it emits `[N]` markers validated against `marker_map`; `_finalize_answer` (`synthesize.py`) strips any URL; `Provenance.resolve_citations` (returns sources keyed `{url,title,article_id}`) expands fact UUIDs → source URLs. The primer's `preliminary_answer` and follow-up *queries* never carry citations. No LLM authors a URL.
- **Design decision #4 — one embedding space:** the primer embeds `q` with the SAME shared embedder as `:Community.embedding` (via `shortlist_communities`).
- All DRIFT LLM calls use the synthesis tier (`synthesize._synthesis_client_and_model`, GLM via `judge_*`) with generous `max_tokens` (GLM reasoning truncates to empty at tiny caps — slice-1/2 lesson).
- Zero facts after all follow-ups → fixed refusal, NO synthesis LLM call. Empty primer shortlist → degrade to `answer_local(q)` (+ a `degraded` flag), no DRIFT synthesis.
- `iterations` clamped to `[1,2]`; iteration 2 (refinement) only runs when 2 rounds AND round-1 facts exist.
- `center_node_uuid` on `search_local` is best-effort: an untagged / unknown / member-less community runs plain local search. Extending `search_local` must be backward-compatible (existing callers pass nothing).
- Run with `uv`: `uv run --extra dev pytest …`, `uv run ruff check src tests` (the CI gate — lint applies to test files too; write test fakes with **one statement per line, no semicolons**, or E702 fails CI), `uv run mypy <files>`. Hermetic settings: `ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")`.
- Integration tests share a **module-scoped** Neo4j testcontainer (`extract_driver` fixture) — begin each test body with `await s.run("MATCH (n) DETACH DELETE n")` before seeding.

---

## File Structure

- **Modify** `src/graph_extract/config.py` — `drift_*` settings.
- **Modify** `src/answer_api/search.py` — optional `center_node_uuid` + node-distance recipe.
- **Create** `src/answer_api/drift.py` — `FollowUp`, `_parse_followups`, `_primer`, `_top_member_entity`, `_run_followup`, `_refine_followups`, `_dedup_facts`, `_synthesize`, `drift_search`, prompts, `_REFUSAL`.
- **Modify** `src/answer_api/app.py` — `GET /search/drift` endpoint.
- **Tests:** `tests/unit/test_config.py` (extend), `tests/integration/test_search_center_node.py`, `tests/unit/test_drift_parse.py`, `tests/integration/test_drift.py`, `tests/unit/test_answer_api_app.py` (extend), `tests/integration/test_drift_live.py`.

---

### Task 1: Config fields

**Files:**
- Modify: `src/graph_extract/config.py`
- Test: `tests/unit/test_config.py` (extend)

**Interfaces:**
- Produces on `ExtractSettings`: `drift_primer_level: int`, `drift_primer_k: int`, `drift_max_followups: int`, `drift_followup_k: int`, `drift_iterations: int`.

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_config.py`)

```python
def test_drift_defaults():
    from graph_extract.config import ExtractSettings
    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    assert s.drift_primer_level == 1
    assert s.drift_primer_k == 5
    assert s.drift_max_followups == 4
    assert s.drift_followup_k == 8
    assert s.drift_iterations == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_config.py::test_drift_defaults -q`
Expected: FAIL (fields do not exist).

- [ ] **Step 3: Implement** — in `src/graph_extract/config.py`, inside `class ExtractSettings`, after the global-search fields block, add:

```python
    # --- DRIFT search (design: drift-search) ---
    drift_primer_level: int = 1      # community level the primer shortlists at
    drift_primer_k: int = 5          # reports shortlisted for the primer
    drift_max_followups: int = 4     # follow-ups kept per round (relevance-budgeted)
    drift_followup_k: int = 8        # local-search k per follow-up
    drift_iterations: int = 1        # follow-up rounds; clamped to [1,2] at call time
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_config.py -q`
Expected: PASS. `uv run ruff check src/graph_extract/config.py` + `uv run mypy src/graph_extract/config.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/graph_extract/config.py tests/unit/test_config.py
git commit -m "feat(config): DRIFT search settings"
```

---

### Task 2: `search_local` center-node extension

**Files:**
- Modify: `src/answer_api/search.py`
- Test: `tests/integration/test_search_center_node.py`

**Interfaces:**
- Consumes: existing `search_local`, `_retrieve_edges`.
- Produces: `search_local(..., center_node_uuid: str | None = None)` and `_retrieve_edges(..., center_node_uuid=None)`; when `center_node_uuid` is set, uses `EDGE_HYBRID_SEARCH_NODE_DISTANCE` and passes `center_node_uuid` to `graphiti._search`; else unchanged RRF path.

- [ ] **Step 1: Write the failing test** (`tests/integration/test_search_center_node.py`)

```python
import pytest
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_NODE_DISTANCE, EDGE_HYBRID_SEARCH_RRF)

pytestmark = pytest.mark.asyncio(loop_scope="module")

GROUP_ID = "backup-docs"


class _Edge:
    def __init__(self, uuid, fact, episodes):
        self.uuid = uuid
        self.fact = fact
        self.episodes = episodes
        self.valid_at = None
        self.invalid_at = None


class _Results:
    def __init__(self, edges):
        self.edges = edges


class _CapGraphiti:
    def __init__(self, edges):
        self._edges = edges
        self.last_kw = None
        self.last_config = None

    async def _search(self, query, config, group_ids=None, **kw):
        self.last_kw = kw
        self.last_config = config
        return _Results(self._edges)


async def _seed(driver):
    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (a:Article {id:'art1', source_url:'https://x/art1', title:'T'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1'})")
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1']}]->(y:Entity)",
                    g=GROUP_ID)


async def test_center_node_switches_recipe_and_passes_through(extract_driver):
    from answer_api.search import search_local
    await _seed(extract_driver)
    g = _CapGraphiti([_Edge("f1", "AWS Backup fact", ["ep1"])])
    out = await search_local(g, extract_driver, q="q", k=10, group_id=GROUP_ID,
                             center_node_uuid="center-entity")
    assert g.last_kw.get("center_node_uuid") == "center-entity"
    assert g.last_config.edge_config.reranker == EDGE_HYBRID_SEARCH_NODE_DISTANCE.edge_config.reranker
    assert out["results"][0]["sources"][0]["url"] == "https://x/art1"   # still resolves


async def test_no_center_node_keeps_rrf(extract_driver):
    from answer_api.search import search_local
    await _seed(extract_driver)
    g = _CapGraphiti([_Edge("f1", "AWS Backup fact", ["ep1"])])
    await search_local(g, extract_driver, q="q", k=10, group_id=GROUP_ID)
    assert g.last_kw.get("center_node_uuid") is None
    assert g.last_config.edge_config.reranker == EDGE_HYBRID_SEARCH_RRF.edge_config.reranker
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/integration/test_search_center_node.py -q`
Expected: FAIL (`search_local` rejects `center_node_uuid`).

- [ ] **Step 3: Implement** — edit `src/answer_api/search.py`:

Change the recipe import:
```python
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_NODE_DISTANCE, EDGE_HYBRID_SEARCH_RRF)
```
Replace `_retrieve_edges` and `search_local` signatures/bodies:
```python
async def _retrieve_edges(graphiti, q, *, fetch_limit, group_id,
                          center_node_uuid=None) -> list:
    # SearchConfig.limit is the top-level fetch cap consulted by Graphiti._search
    # (graphiti-core==0.29.2). A center node switches edge reranking from RRF to
    # graph-distance (node_distance) so retrieval is biased toward that node.
    recipe = EDGE_HYBRID_SEARCH_NODE_DISTANCE if center_node_uuid else EDGE_HYBRID_SEARCH_RRF
    config = recipe.model_copy(deep=True)
    config.limit = fetch_limit
    results = await graphiti._search(q, config, group_ids=[group_id],
                                     center_node_uuid=center_node_uuid)
    return list(results.edges)


async def search_local(graphiti, driver, *, q, k=10, vendor=None,
                       include_invalid=False, group_id, center_node_uuid=None):
    # Over-fetch so post-filters (validity, vendor scope) still leave ~k results.
    edges = await _retrieve_edges(graphiti, q, fetch_limit=max(k * 3, k),
                                  group_id=group_id, center_node_uuid=center_node_uuid)
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

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/integration/test_search_center_node.py tests/integration/test_search_local.py -q`
Expected: PASS (new tests + existing search_local regressions). ruff + `mypy src/answer_api/search.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/search.py tests/integration/test_search_center_node.py
git commit -m "feat(search): optional center-node (node-distance) reranking in search_local"
```

---

### Task 3: Primer — `FollowUp`, `_parse_followups`, `_primer`

**Files:**
- Create: `src/answer_api/drift.py`
- Test: `tests/unit/test_drift_parse.py`, `tests/integration/test_drift.py` (primer part)

**Interfaces:**
- Consumes: `global_search.shortlist_communities`, `global_search.CommunityHit`, `global_search._extract_json`.
- Produces:
  - `@dataclass FollowUp: query: str; community_id: str | None; iteration: int = 1`
  - `_parse_followups(obj: dict, hit_ids: set[str], max_followups: int, iteration: int) -> list[FollowUp]`
  - `async _primer(embedder, synth_client, synth_model, driver, *, q, level, k, max_followups, group_id) -> tuple[str, list[FollowUp], list[CommunityHit]] | None` (None = degrade signal)

- [ ] **Step 1: Write the failing unit test** (`tests/unit/test_drift_parse.py`)

```python
from answer_api.drift import FollowUp, _parse_followups


def test_parse_orders_by_relevance_and_budgets():
    obj = {"follow_ups": [
        {"query": "a", "community_id": "c1", "relevance": 3},
        {"query": "b", "community_id": "c2", "relevance": 9},
        {"query": "c", "community_id": None, "relevance": 7},
    ]}
    fus = _parse_followups(obj, {"c1", "c2"}, max_followups=2, iteration=1)
    assert [f.query for f in fus] == ["b", "c"]          # top-2 by relevance
    assert all(f.iteration == 1 for f in fus)


def test_parse_drops_invalid_community_and_blank_query():
    obj = {"follow_ups": [
        {"query": "a", "community_id": "unknown", "relevance": 5},
        {"query": "", "community_id": "c1", "relevance": 9},
        {"query": "keep", "community_id": "c1", "relevance": 1},
    ]}
    fus = _parse_followups(obj, {"c1"}, max_followups=5, iteration=2)
    assert [f.query for f in fus] == ["a", "keep"]       # blank-query dropped
    a = next(f for f in fus if f.query == "a")
    assert a.community_id is None                         # unknown id -> None
    assert a.iteration == 2


def test_parse_empty_or_garbage():
    assert _parse_followups({}, set(), 4, 1) == []
    assert _parse_followups({"follow_ups": ["notadict", 3]}, set(), 4, 1) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_drift_parse.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — create `src/answer_api/drift.py`:

```python
"""DRIFT search: broad->deep. Primer on the :Community report layer drafts a
preliminary answer + follow-up queries; each follow-up runs a graph-biased local
fact search; a final synthesis merges the evidence into one cited answer.
Citations are graph traversal (design decision #2): the LLM emits only [N]
markers / fact UUIDs, never a URL."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from answer_api.search import search_local
from answer_api.synthesize import _finalize_answer, answer_local
from answer_api.global_search import shortlist_communities, _extract_json
from graph_extract.provenance import Provenance

logger = logging.getLogger(__name__)

_REFUSAL = "I don't have enough information to answer that from the available sources."


@dataclass
class FollowUp:
    query: str
    community_id: str | None
    iteration: int = 1


def _parse_followups(obj: dict, hit_ids: set[str], max_followups: int,
                     iteration: int) -> list[FollowUp]:
    scored: list[tuple[float, FollowUp]] = []
    for item in obj.get("follow_ups") or []:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        cid = item.get("community_id")
        cid = cid if (isinstance(cid, str) and cid in hit_ids) else None
        try:
            rel = float(item.get("relevance", 0) or 0)
        except (TypeError, ValueError):
            rel = 0.0
        scored.append((rel, FollowUp(query=query, community_id=cid, iteration=iteration)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [fu for _, fu in scored[:max_followups]]


_PRIMER_PROMPT = (
    "You are starting a broad investigation of a QUESTION about backup products. "
    "Use the COMMUNITY themes below as leads (not as the final answer). Draft a "
    "brief preliminary answer and 3-6 targeted follow-up questions that would "
    "retrieve concrete supporting facts. Tag each follow-up with the community_id "
    "it draws from, or null. Respond with ONLY JSON: {\"preliminary_answer\": str, "
    "\"follow_ups\": [{\"query\": str, \"community_id\": str|null, "
    "\"relevance\": 0-10}]}. Do NOT write URLs.\n\n"
    "QUESTION: {q}\n\nCOMMUNITIES:\n{blocks}"
)


async def _primer(embedder, synth_client, synth_model, driver, *, q, level, k,
                  max_followups, group_id):
    hits = await shortlist_communities(driver, embedder, q, level=level, k=k,
                                       group_id=group_id)
    if not hits:
        return None
    blocks = "\n".join(f'- {h.community_id} "{h.title}": {h.summary}' for h in hits)
    obj: dict | None = None
    for _ in range(2):
        resp = await synth_client.chat.completions.create(
            model=synth_model, temperature=0, max_tokens=2000,
            messages=[{"role": "user",
                       "content": _PRIMER_PROMPT.format(q=q, blocks=blocks)}])
        obj = _extract_json(resp.choices[0].message.content or "")
        if obj is not None:
            break
    hit_ids = {h.community_id for h in hits}
    if obj is None:
        return "", [FollowUp(query=q, community_id=None, iteration=1)], hits
    preliminary = str(obj.get("preliminary_answer", "")).strip()
    fus = _parse_followups(obj, hit_ids, max_followups, iteration=1)
    if not fus:
        fus = [FollowUp(query=q, community_id=None, iteration=1)]
    return preliminary, fus, hits
```

- [ ] **Step 4: Run the unit test**

Run: `uv run --extra dev pytest tests/unit/test_drift_parse.py -q`
Expected: PASS (3 tests). ruff + `mypy src/answer_api/drift.py` clean.

- [ ] **Step 5: Add the `_primer` integration test** (`tests/integration/test_drift.py`)

```python
import json
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

GROUP_ID = "backup-docs"


class _FakeEmbedder:
    def __init__(self, vec):
        self._vec = vec
    async def create_batch(self, texts):
        return [self._vec for _ in texts]


class _FakeLLM:
    def __init__(self, contents):
        self._c = list(contents)
        self.chat = self
        self.completions = self
    async def create(self, **kw):
        c = self._c.pop(0)
        return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": c})()})()]})


async def _seed_community(driver, cid="c1", emb=(1.0, 0.0)):
    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (:Community {group_id:$g, level:1, community_id:$cid, title:'S3', "
                    "summary:'s', rating:8.0, cited_fact_uuids:['f1'], full_report:'[]', "
                    "embedding:$e})", g=GROUP_ID, cid=cid, e=list(emb))


async def test_primer_shortlists_and_budgets(extract_driver):
    from answer_api.drift import _primer
    await _seed_community(extract_driver)
    payload = json.dumps({"preliminary_answer": "draft",
                          "follow_ups": [{"query": "how retained", "community_id": "c1", "relevance": 9},
                                         {"query": "extra", "community_id": None, "relevance": 1}]})
    out = await _primer(_FakeEmbedder([1.0, 0.0]), _FakeLLM([payload]), "m", extract_driver,
                        q="retention", level=1, k=5, max_followups=1, group_id=GROUP_ID)
    assert out is not None
    preliminary, fus, hits = out
    assert preliminary == "draft"
    assert [f.query for f in fus] == ["how retained"]     # budgeted to 1
    assert fus[0].community_id == "c1"
    assert hits[0].community_id == "c1"


async def test_primer_empty_shortlist_signals_degrade(extract_driver):
    from answer_api.drift import _primer
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")   # no communities
    out = await _primer(_FakeEmbedder([1.0, 0.0]), _FakeLLM([]), "m", extract_driver,
                        q="q", level=1, k=5, max_followups=4, group_id=GROUP_ID)
    assert out is None


async def test_primer_bad_json_falls_back_to_single_query(extract_driver):
    from answer_api.drift import _primer
    await _seed_community(extract_driver)
    out = await _primer(_FakeEmbedder([1.0, 0.0]), _FakeLLM(["nope", "still nope"]), "m",
                        extract_driver, q="original q", level=1, k=5, max_followups=4,
                        group_id=GROUP_ID)
    assert out is not None
    preliminary, fus, hits = out
    assert preliminary == ""                              # unparseable -> empty draft
    assert [(f.query, f.community_id) for f in fus] == [("original q", None)]   # single fallback
```

- [ ] **Step 6: Run + commit**

Run: `uv run --extra dev pytest tests/unit/test_drift_parse.py tests/integration/test_drift.py -q`
Expected: PASS. ruff + mypy clean.

```bash
git add src/answer_api/drift.py tests/unit/test_drift_parse.py tests/integration/test_drift.py
git commit -m "feat(drift): primer (shortlist + preliminary answer + budgeted follow-ups)"
```

---

### Task 4: Follow-up execution — `_top_member_entity`, `_run_followup`, `_refine_followups`

**Files:**
- Modify: `src/answer_api/drift.py`
- Test: `tests/integration/test_drift.py` (extend), `tests/unit/test_drift_parse.py` (extend)

**Interfaces:**
- Consumes: `FollowUp`, `_parse_followups`, `_extract_json`, `search_local`.
- Produces:
  - `async _top_member_entity(driver, group_id, community_id) -> str | None`
  - `async _run_followup(graphiti, driver, fu: FollowUp, *, k, group_id) -> list[dict]`
  - `async _refine_followups(synth_client, synth_model, *, q, facts, max_followups, hit_ids) -> list[FollowUp]`

- [ ] **Step 1: Write the failing integration test** (append to `tests/integration/test_drift.py`)

```python
async def test_top_member_entity_picks_highest_degree(extract_driver):
    from answer_api.drift import _top_member_entity
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (c:Community {group_id:$g, level:1, community_id:'c1'})", g=GROUP_ID)
        # hub has 2 RELATES_TO, leaf has 1; both members of c1
        await s.run(
            "MATCH (c:Community {community_id:'c1'}) "
            "CREATE (hub:Entity {uuid:'hub'})-[:IN_COMMUNITY]->(c) "
            "CREATE (leaf:Entity {uuid:'leaf'})-[:IN_COMMUNITY]->(c) "
            "CREATE (o1:Entity)-[:RELATES_TO {group_id:$g, uuid:'r1'}]->(hub) "
            "CREATE (hub)-[:RELATES_TO {group_id:$g, uuid:'r2'}]->(o2:Entity) "
            "CREATE (leaf)-[:RELATES_TO {group_id:$g, uuid:'r3'}]->(o3:Entity)", g=GROUP_ID)
    assert await _top_member_entity(extract_driver, GROUP_ID, "c1") == "hub"
    assert await _top_member_entity(extract_driver, GROUP_ID, "nope") is None


async def test_run_followup_biases_by_center_node(extract_driver):
    from answer_api.drift import _run_followup, FollowUp

    class _Edge:
        def __init__(self):
            self.uuid = "f1"
            self.fact = "fact"
            self.episodes = ["ep1"]
            self.valid_at = None
            self.invalid_at = None

    class _Results:
        def __init__(self, edges):
            self.edges = edges

    class _CapGraphiti:
        def __init__(self):
            self.last_center = "unset"
        async def _search(self, query, config, group_ids=None, **kw):
            self.last_center = kw.get("center_node_uuid")
            return _Results([_Edge()])

    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (a:Article {id:'art1', source_url:'https://x/art1', title:'T'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1'})")
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1']}]->(y:Entity)", g=GROUP_ID)
        await s.run("CREATE (c:Community {group_id:$g, level:1, community_id:'c1'}) "
                    "CREATE (m:Entity {uuid:'m1'})-[:IN_COMMUNITY]->(c) "
                    "CREATE (m)-[:RELATES_TO {group_id:$g, uuid:'r1'}]->(:Entity)", g=GROUP_ID)
    g = _CapGraphiti()
    rows = await _run_followup(g, extract_driver, FollowUp("q", "c1", 1), k=8, group_id=GROUP_ID)
    assert g.last_center == "m1"                    # community's top member used as center
    assert rows[0]["fact_uuid"] == "f1"

    g2 = _CapGraphiti()
    await _run_followup(g2, extract_driver, FollowUp("q", None, 1), k=8, group_id=GROUP_ID)
    assert g2.last_center is None                   # untagged -> plain local
```

Also add the refinement unit test to `tests/unit/test_drift_parse.py`. **Move all
imports to the top of the file** (ruff E402 forbids mid-file imports): the file's
top import block should read
```python
import json
import pytest
from answer_api.drift import FollowUp, _parse_followups, _refine_followups
```
(merging Task 3's `from answer_api.drift import FollowUp, _parse_followups`). Then
append the `_FakeLLM` class + the two async tests below:

```python
class _FakeLLM:
    def __init__(self, contents):
        self._c = list(contents)
        self.chat = self
        self.completions = self
    async def create(self, **kw):
        c = self._c.pop(0)
        return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": c})()})()]})


@pytest.mark.asyncio
async def test_refine_followups_parses_and_tags_iteration_2():
    payload = json.dumps({"follow_ups": [{"query": "deeper", "community_id": "c1", "relevance": 8}]})
    fus = await _refine_followups(_FakeLLM([payload]), "m", q="q",
                                  facts=[{"fact": "f", "fact_uuid": "u1"}],
                                  max_followups=4, hit_ids={"c1"})
    assert [f.query for f in fus] == ["deeper"]
    assert fus[0].iteration == 2


@pytest.mark.asyncio
async def test_refine_followups_bad_json_returns_empty():
    fus = await _refine_followups(_FakeLLM(["nope", "still nope"]), "m", q="q",
                                  facts=[{"fact": "f", "fact_uuid": "u1"}],
                                  max_followups=4, hit_ids=set())
    assert fus == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/integration/test_drift.py -k "top_member or biases" tests/unit/test_drift_parse.py -k refine -q`
Expected: FAIL (symbols missing).

- [ ] **Step 3: Implement** — append to `src/answer_api/drift.py`:

```python
async def _top_member_entity(driver, group_id, community_id) -> str | None:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (c:Community {group_id:$g, community_id:$cid})<-[:IN_COMMUNITY]-(e:Entity) "
            "OPTIONAL MATCH (e)-[rel:RELATES_TO {group_id:$g}]-() "
            "WITH e, count(rel) AS deg ORDER BY deg DESC LIMIT 1 "
            "RETURN e.uuid AS uuid", g=group_id, cid=community_id)
        rec = await r.single()
        return rec["uuid"] if rec else None


async def _run_followup(graphiti, driver, fu: FollowUp, *, k, group_id) -> list[dict]:
    center = (await _top_member_entity(driver, group_id, fu.community_id)
              if fu.community_id else None)
    res = await search_local(graphiti, driver, q=fu.query, k=k,
                             center_node_uuid=center, group_id=group_id)
    return res["results"]


_REFINE_PROMPT = (
    "You are investigating a QUESTION and have gathered these FACTS. Draft up to "
    "{n} refined follow-up questions that fill the biggest remaining gaps. Tag each "
    "with a community_id from the list, or null. Respond with ONLY JSON: "
    "{{\"follow_ups\": [{{\"query\": str, \"community_id\": str|null, "
    "\"relevance\": 0-10}}]}}. Do NOT write URLs.\n\n"
    "QUESTION: {q}\n\nCOMMUNITY IDS: {cids}\n\nFACTS:\n{facts}"
)


async def _refine_followups(synth_client, synth_model, *, q, facts, max_followups,
                            hit_ids) -> list[FollowUp]:
    facts_block = "\n".join(f"- {f['fact']}" for f in facts)
    obj: dict | None = None
    for _ in range(2):
        resp = await synth_client.chat.completions.create(
            model=synth_model, temperature=0, max_tokens=1500,
            messages=[{"role": "user", "content": _REFINE_PROMPT.format(
                n=max_followups, q=q, cids=sorted(hit_ids), facts=facts_block)}])
        obj = _extract_json(resp.choices[0].message.content or "")
        if obj is not None:
            break
    if obj is None:
        return []
    return _parse_followups(obj, set(hit_ids), max_followups, iteration=2)
```

Note the doubled braces `{{…}}` in `_REFINE_PROMPT` — the literal JSON schema must survive `.format()` (only `{n}`, `{q}`, `{cids}`, `{facts}` are real fields).

- [ ] **Step 4: Run to verify pass**

Run: `uv run --extra dev pytest tests/integration/test_drift.py tests/unit/test_drift_parse.py -q`
Expected: PASS. ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/drift.py tests/integration/test_drift.py tests/unit/test_drift_parse.py
git commit -m "feat(drift): center-node resolution, follow-up execution, iteration-2 refinement"
```

---

### Task 5: Synthesis + orchestrator — `_dedup_facts`, `_synthesize`, `drift_search`

**Files:**
- Modify: `src/answer_api/drift.py`
- Test: `tests/unit/test_drift_parse.py` (extend), `tests/integration/test_drift.py` (extend)

**Interfaces:**
- Consumes: `_primer`, `_run_followup`, `_refine_followups`, `_finalize_answer`, `Provenance`, `answer_local`.
- Produces:
  - `_dedup_facts(facts: list[dict]) -> list[dict]` (first-seen order, keyed on `fact_uuid`)
  - `async _synthesize(synth_client, synth_model, driver, *, q, preliminary_answer, facts) -> tuple[str, list[dict]]`
  - `async drift_search(graphiti, driver, embedder, synth_client, synth_model, *, q, level, iterations, primer_k, max_followups, followup_k, group_id) -> dict`

- [ ] **Step 1: Write the failing unit test** (append to `tests/unit/test_drift_parse.py`)

```python
def test_dedup_facts_preserves_first_seen_order():
    from answer_api.drift import _dedup_facts
    facts = [{"fact_uuid": "a", "fact": "A"}, {"fact_uuid": "b", "fact": "B"},
             {"fact_uuid": "a", "fact": "A2"}]
    out = _dedup_facts(facts)
    assert [f["fact_uuid"] for f in out] == ["a", "b"]
    assert out[0]["fact"] == "A"        # first occurrence kept
```

- [ ] **Step 2: Write the failing integration test** (append to `tests/integration/test_drift.py`)

```python
async def _seed_fact_provenance(driver, cid="c1"):
    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (a:Article {id:'art1', source_url:'https://x/art1', title:'T'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1', group_id:$g})", g=GROUP_ID)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1'], "
                    "fact:'AWS Backup supports S3'}]->(y:Entity)", g=GROUP_ID)
        await s.run("CREATE (c:Community {group_id:$g, level:1, community_id:$cid, title:'S3', "
                    "summary:'s', rating:8.0, cited_fact_uuids:['f1'], full_report:'[]', "
                    "embedding:[1.0,0.0]}) "
                    "CREATE (m:Entity {uuid:'m1'})-[:IN_COMMUNITY]->(c) "
                    "CREATE (m)-[:RELATES_TO {group_id:$g, uuid:'r1'}]->(:Entity)", g=GROUP_ID, cid=cid)


class _Edge:
    def __init__(self, uuid, fact):
        self.uuid = uuid
        self.fact = fact
        self.episodes = ["ep1"]
        self.valid_at = None
        self.invalid_at = None


class _Results:
    def __init__(self, edges):
        self.edges = edges


class _FactGraphiti:               # returns fact f1 for any follow-up search
    async def _search(self, query, config, group_ids=None, **kw):
        return _Results([_Edge("f1", "AWS Backup supports S3")])


async def test_drift_search_end_to_end(extract_driver):
    from answer_api.drift import drift_search
    await _seed_fact_provenance(extract_driver)
    primer = json.dumps({"preliminary_answer": "S3 is supported",
                         "follow_ups": [{"query": "how", "community_id": "c1", "relevance": 9}]})
    synth = "AWS Backup supports S3 [1]. See https://evil/x [9]."   # bad URL + invalid marker
    llm = _FakeLLM([primer, synth])
    res = await drift_search(_FactGraphiti(), extract_driver, _FakeEmbedder([1.0, 0.0]),
                             llm, "m", q="s3 retention", level=1, iterations=1,
                             primer_k=5, max_followups=4, followup_k=8, group_id=GROUP_ID)
    assert res["citations"][0]["fact_uuid"] == "f1"
    assert res["citations"][0]["sources"][0]["url"] == "https://x/art1"
    assert "http" not in res["answer"]                         # URL stripped
    assert [c["marker"] for c in res["citations"]] == [1]      # invalid [9] dropped
    assert res["follow_ups"][0]["community_id"] == "c1"
    assert res["communities_used"][0]["community_id"] == "c1"


async def test_drift_search_empty_shortlist_degrades_to_local(extract_driver):
    from answer_api.drift import drift_search
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")   # no communities
    # answer_local runs its own search_local over the (empty) graph -> refusal-shaped local answer
    llm = _FakeLLM([])   # no primer call expected (shortlist empty short-circuits before LLM)
    res = await drift_search(_FactGraphiti(), extract_driver, _FakeEmbedder([1.0, 0.0]),
                             llm, "m", q="q", level=1, iterations=1,
                             primer_k=5, max_followups=4, followup_k=8, group_id=GROUP_ID)
    assert res["degraded"] == "no-primer-communities"


async def test_drift_search_zero_facts_refuses(extract_driver):
    from answer_api.drift import drift_search, _REFUSAL

    class _NoFacts:
        async def _search(self, query, config, group_ids=None, **kw):
            return _Results([])

    await _seed_fact_provenance(extract_driver)
    primer = json.dumps({"preliminary_answer": "d",
                         "follow_ups": [{"query": "how", "community_id": "c1", "relevance": 9}]})

    class _Boom:
        def __init__(self, contents):
            self._c = list(contents)
            self.chat = self
            self.completions = self
        async def create(self, **kw):
            if not self._c:
                raise AssertionError("no synthesis call when zero facts")
            return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": self._c.pop(0)})()})()]})

    res = await drift_search(_NoFacts(), extract_driver, _FakeEmbedder([1.0, 0.0]),
                             _Boom([primer]), "m", q="q", level=1, iterations=1,
                             primer_k=5, max_followups=4, followup_k=8, group_id=GROUP_ID)
    assert res["answer"] == _REFUSAL and res["citations"] == []


async def test_drift_search_iterations_two_runs_refinement(extract_driver):
    from answer_api.drift import drift_search
    await _seed_fact_provenance(extract_driver)
    primer = json.dumps({"preliminary_answer": "d",
                         "follow_ups": [{"query": "round1", "community_id": "c1", "relevance": 9}]})
    refine = json.dumps({"follow_ups": [{"query": "round2", "community_id": "c1", "relevance": 9}]})
    synth = "Answer [1]."
    # 3 LLM calls expected for iterations=2: primer, refine, synth
    llm = _FakeLLM([primer, refine, synth])
    res = await drift_search(_FactGraphiti(), extract_driver, _FakeEmbedder([1.0, 0.0]),
                             llm, "m", q="q", level=1, iterations=2,
                             primer_k=5, max_followups=4, followup_k=8, group_id=GROUP_ID)
    iters = {f["iteration"] for f in res["follow_ups"]}
    assert iters == {1, 2}                                   # both rounds executed
    assert res["citations"][0]["fact_uuid"] == "f1"


async def test_drift_search_one_followup_raises_does_not_abort(extract_driver):
    from answer_api.drift import drift_search

    class _FlakyGraphiti:   # first follow-up raises, second returns a fact
        def __init__(self):
            self._n = 0
        async def _search(self, query, config, group_ids=None, **kw):
            self._n += 1
            if self._n == 1:
                raise RuntimeError("boom")
            return _Results([_Edge("f1", "AWS Backup supports S3")])

    await _seed_fact_provenance(extract_driver)
    primer = json.dumps({"preliminary_answer": "d", "follow_ups": [
        {"query": "a", "community_id": "c1", "relevance": 9},
        {"query": "b", "community_id": None, "relevance": 8}]})
    synth = "Answer [1]."
    res = await drift_search(_FlakyGraphiti(), extract_driver, _FakeEmbedder([1.0, 0.0]),
                             _FakeLLM([primer, synth]), "m", q="q", level=1, iterations=1,
                             primer_k=5, max_followups=4, followup_k=8, group_id=GROUP_ID)
    assert res["citations"][0]["fact_uuid"] == "f1"          # survived the raising follow-up
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run --extra dev pytest tests/integration/test_drift.py -k "end_to_end or degrades or zero_facts" tests/unit/test_drift_parse.py -k dedup -q`
Expected: FAIL (`drift_search`/`_dedup_facts` missing).

- [ ] **Step 4: Implement** — append to `src/answer_api/drift.py`:

```python
_SYNTH_PROMPT = (
    "Answer the QUESTION using ONLY the numbered FACTS, refining the DRAFT where "
    "the facts support it. Cite every claim inline with its [N] marker. Do NOT use "
    "outside knowledge. Do NOT write any URL. If the facts do not answer the "
    "question, reply exactly: \"{refusal}\"\n\n"
    "QUESTION: {q}\n\nDRAFT: {draft}\n\nFACTS:\n{facts}\n\nAnswer:"
)


def _dedup_facts(facts: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for f in facts:
        u = f["fact_uuid"]
        if u not in seen:
            seen.add(u)
            out.append(f)
    return out


async def _synthesize(synth_client, synth_model, driver, *, q, preliminary_answer,
                      facts) -> tuple[str, list[dict]]:
    facts = _dedup_facts(facts)
    marker_map = {i: f for i, f in enumerate(facts, 1)}
    facts_block = "\n".join(f"[{i}] {f['fact']}" for i, f in marker_map.items())
    resp = await synth_client.chat.completions.create(
        model=synth_model, temperature=0, max_tokens=3000,
        messages=[{"role": "user", "content": _SYNTH_PROMPT.format(
            refusal=_REFUSAL, q=q, draft=preliminary_answer or "(none)", facts=facts_block)}])
    answer, cited = _finalize_answer(resp.choices[0].message.content or "", marker_map)
    resolved = await Provenance(driver).resolve_citations(
        [marker_map[m]["fact_uuid"] for m in cited])
    citations = [{"marker": m, "fact_uuid": marker_map[m]["fact_uuid"],
                  "sources": resolved.get(marker_map[m]["fact_uuid"], [])} for m in cited]
    return answer, citations


async def drift_search(graphiti, driver, embedder, synth_client, synth_model, *,
                       q, level, iterations, primer_k, max_followups, followup_k,
                       group_id) -> dict:
    rounds = max(1, min(iterations, 2))
    primed = await _primer(embedder, synth_client, synth_model, driver, q=q,
                           level=level, k=primer_k, max_followups=max_followups,
                           group_id=group_id)
    if primed is None:
        res = await answer_local(graphiti, driver, synth_client, synth_model,
                                 q=q, group_id=group_id)
        res["degraded"] = "no-primer-communities"
        return res
    preliminary, followups, hits = primed
    hit_ids = {h.community_id for h in hits}
    facts: list[dict] = []
    executed: list[FollowUp] = []
    for fu in followups:
        try:
            facts.extend(await _run_followup(graphiti, driver, fu, k=followup_k,
                                             group_id=group_id))
        except Exception:
            logger.warning("drift follow-up failed: %s", fu.query, exc_info=True)
        executed.append(fu)
    if rounds == 2 and _dedup_facts(facts):
        refined = await _refine_followups(synth_client, synth_model, q=q,
                                          facts=_dedup_facts(facts),
                                          max_followups=max_followups, hit_ids=hit_ids)
        for fu in refined:
            try:
                facts.extend(await _run_followup(graphiti, driver, fu, k=followup_k,
                                                 group_id=group_id))
            except Exception:
                logger.warning("drift follow-up failed: %s", fu.query, exc_info=True)
            executed.append(fu)
    follow_ups_meta = [{"query": fu.query, "community_id": fu.community_id,
                        "iteration": fu.iteration} for fu in executed]
    communities_used = [{"community_id": h.community_id, "title": h.title} for h in hits]
    if not _dedup_facts(facts):
        return {"query": q, "answer": _REFUSAL, "citations": [],
                "follow_ups": follow_ups_meta, "communities_used": communities_used}
    answer, citations = await _synthesize(synth_client, synth_model, driver, q=q,
                                          preliminary_answer=preliminary, facts=facts)
    return {"query": q, "answer": answer, "citations": citations,
            "follow_ups": follow_ups_meta, "communities_used": communities_used}
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run --extra dev pytest tests/integration/test_drift.py tests/unit/test_drift_parse.py -q`
Expected: PASS. `uv run ruff check src/answer_api/drift.py tests/integration/test_drift.py tests/unit/test_drift_parse.py` + `uv run mypy src/answer_api/drift.py` clean.

- [ ] **Step 6: Commit**

```bash
git add src/answer_api/drift.py tests/integration/test_drift.py tests/unit/test_drift_parse.py
git commit -m "feat(drift): synthesis + orchestrator (degrade, refusal, iteration loop)"
```

---

### Task 6: Endpoint — `GET /search/drift`

**Files:**
- Modify: `src/answer_api/app.py`
- Test: `tests/unit/test_answer_api_app.py` (extend)

**Interfaces:**
- Consumes: `drift.drift_search`.
- Produces: `GET /search/drift?q=&level=&iterations=`.

- [ ] **Step 1: Write the failing test** (extend `tests/unit/test_answer_api_app.py`)

Add a stub coroutine near the other fakes:
```python
async def _fake_drift_search(graphiti, driver, embedder, synth_client, synth_model, *,
                             q, level, iterations, primer_k, max_followups, followup_k, group_id):
    return {"query": q, "answer": "cross-vendor DRIFT answer [1].",
            "citations": [{"marker": 1, "fact_uuid": "f1",
                           "sources": [{"url": "https://x/art1", "title": "T", "article_id": "art1"}]}],
            "follow_ups": [{"query": "how retained", "community_id": "c1", "iteration": 1}],
            "communities_used": [{"community_id": "c1", "title": "S3"}]}
```
Add to the `_stub_deps` autouse fixture body:
```python
    import answer_api.drift as drift_mod
    monkeypatch.setattr(drift_mod, "drift_search", _fake_drift_search)
```
Add the tests:
```python
async def test_drift_returns_stubbed_answer():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/drift", params={"q": "plan retention across vendors"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"query", "answer", "citations", "follow_ups", "communities_used"}
    assert body["citations"][0]["fact_uuid"] == "f1"


async def test_drift_requires_q():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/drift")
    assert resp.status_code == 422


async def test_drift_iterations_over_two_is_422():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/drift", params={"q": "x", "iterations": 3})
    assert resp.status_code == 422
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_answer_api_app.py -k drift -q`
Expected: FAIL (route missing).

- [ ] **Step 3: Implement** — in `src/answer_api/app.py`:

Add the import near the other module imports:
```python
from answer_api import drift as drift_mod
```
Add the endpoint inside `create_app` (after `/search/global`):
```python
    @app.get("/search/drift")
    async def search_drift(
        q: str, level: int | None = Query(None, ge=0),
        iterations: int | None = Query(None, ge=1, le=2)
    ) -> dict[str, Any]:
        st = app.state
        s = st.settings
        return await drift_mod.drift_search(
            st.graphiti, st.driver, st.embedder, st.synth_client, st.synth_model,
            q=q,
            level=s.drift_primer_level if level is None else level,
            iterations=s.drift_iterations if iterations is None else iterations,
            primer_k=s.drift_primer_k, max_followups=s.drift_max_followups,
            followup_k=s.drift_followup_k, group_id=s.group_id)
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_answer_api_app.py -q`
Expected: PASS (all app tests, incl. the 3 new + `/health`,`/search/local`,`/answer`,`/timeline`,`/search/global` regressions). ruff + `mypy src/answer_api/app.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/app.py tests/unit/test_answer_api_app.py
git commit -m "feat(answer-api): GET /search/drift endpoint"
```

---

### Task 7: Full-suite gate + `@live` smoke + demonstration

**Files:**
- Create: `tests/integration/test_drift_live.py`, `docs/superpowers/drift-report.md`

- [ ] **Step 1: Full non-live suite + CI lint gate**

Run: `uv run --extra dev pytest -m "not live" -q` → all pass.
Run: `uv run ruff check src tests` → clean. `uv run mypy src/answer_api src/graph_extract` → clean.
Fix any fallout before continuing.

- [ ] **Step 2: `@live` smoke** (`tests/integration/test_drift_live.py`)

```python
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_drift_search_live(live_extract_driver):
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_embedder, build_graphiti
    from answer_api.synthesize import _synthesis_client_and_model
    from answer_api.drift import drift_search
    s = get_extract_settings()
    graphiti = build_graphiti(s)
    emb = build_embedder(s)
    sc, sm = _synthesis_client_and_model(s)
    try:
        res = await drift_search(
            graphiti, live_extract_driver, emb, sc, sm,
            q="What should I consider when planning long-term backup retention across cloud vendors?",
            level=s.drift_primer_level, iterations=s.drift_iterations,
            primer_k=s.drift_primer_k, max_followups=s.drift_max_followups,
            followup_k=s.drift_followup_k, group_id=s.group_id)
        assert res["answer"]
        assert "http" not in res["answer"]                    # no LLM-authored URL
        for c in res.get("citations", []):
            assert c["sources"], "cited fact must resolve to >=1 source"
    finally:
        await graphiti.close()
        await sc.close()
```

Run: `uv run --extra dev pytest -m live tests/integration/test_drift_live.py -q`
Expected: PASS.

- [ ] **Step 3: Controller demonstration**

Call `drift_search` directly (or via the running API) on `backup-docs` with a broad question; write `docs/superpowers/drift-report.md` showing: the primer-drafted `follow_ups` (with their `community_id` tags), the center-node biasing, the cross-vendor cited synthesis, and that `citations[].sources` resolve to real source URLs (no URL authored by the LLM).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_drift_live.py docs/superpowers/drift-report.md
git commit -m "test(drift): live smoke + demonstration report"
```

---

## Notes for the implementer

- `drift_search`/`_primer`/`_run_followup`/`_refine_followups`/`_synthesize` are module-level names so tests substitute fakes; only the `@live` smoke + demonstration need real infra. The endpoint calls `drift_mod.drift_search` (module-attr seam).
- Decision #2: only the synthesis answer is cited; its `[N]` markers are validated against `marker_map` and `_finalize_answer` strips URLs. Never surface the primer draft or a follow-up query as if it were a sourced answer.
- Follow-up retrieval reuses `search_local` unchanged except for the optional `center_node_uuid`; every existing `search_local` caller is unaffected (default `None`).
- GLM reasoning headroom: primer/refine/synth use generous `max_tokens` (2000/1500/3000); a tiny cap truncates GLM output to empty.
- Test fakes: **one statement per line, no semicolons** (CI `ruff check src tests` flags E702). Fakes for the LLM client expose `.chat.completions.create`; the embedder exposes `.create_batch`; the graphiti stub exposes `async _search(self, query, config, group_ids=None, **kw)`.
