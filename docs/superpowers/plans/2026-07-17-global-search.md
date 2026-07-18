# Global Map-Reduce Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `GET /search/global` — a map-reduce over the `:Community` report layer that answers cross-vendor thematic questions with fact-ID → URL citations.

**Architecture:** Embed the query (shared embedder), shortlist community reports by cosine + rating at a level, MAP each (parallel, per-report relevance + key points + validated fact IDs), REDUCE across map outputs into one cited answer (GLM synthesis), resolve fact IDs → URLs via the existing `Provenance` resolver. New `answer_api/global_search.py`; reuses `synthesize` + `Provenance` citation machinery.

**Tech Stack:** Python 3.12, FastAPI (`answer_api`), Neo4j 5.26, graphiti-core embedder (TEI/Jina), OpenAI-compatible LLM (GLM‑5.2 via judge config), pytest (+ Neo4j testcontainer; live LLM/embed are `@live`).

## Global Constraints

- **Design decision #2 — citations are graph traversal, never LLM:** map emits fact UUIDs that must be a subset of the report's real `cited_fact_uuids` (drop hallucinated); reduce emits only `[N]` markers validated against the marker map; `_URL_RE` strips any URL the reduce LLM writes; `Provenance.resolve_citations` expands fact UUIDs → source URLs. No LLM authors a URL.
- **Design decision #4 — one embedding space:** the query is embedded with the SAME shared embedder (`graphiti_client.build_embedder`, `embed_model`/`embed_dim`) as `:Community.embedding`.
- Map tier defaults to the synthesis/judge tier (GLM‑5.2) and is config-overridable (`map_llm_*`); reduce uses the synthesis tier (`_synthesis_client_and_model`).
- Zero shortlisted / zero surviving-map → a fixed refusal string, NO LLM spend.
- Reduce (and map) LLM calls use generous `max_tokens` (GLM is a reasoning model — a small cap truncates the output to empty; slice-1 used 8000 for reports).
- Run via `uv` (`uv run --extra dev pytest ...`, `uv run ruff check ...`, `uv run mypy ...`). Hermetic settings in tests: `ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")`.

---

## File Structure

- **Create** `src/answer_api/global_search.py` — dataclasses `CommunityHit`/`MapResult`; `_cosine`/`_rank_hits` (pure); `shortlist_communities`; `_map_client_and_model`/`_extract_json`/`map_report`; `global_search`; `_REDUCE_PROMPT`/`_MAP_PROMPT`/`_REFUSAL`.
- **Modify** `src/graph_extract/config.py` — `map_llm_*`, `global_*` fields.
- **Modify** `src/answer_api/app.py` — lifespan builds the shared embedder + map client; `GET /search/global` endpoint.
- **Tests:** `tests/unit/test_global_rank.py`, `tests/unit/test_global_map.py`, `tests/integration/test_global_search.py`, `tests/unit/test_answer_api_app.py` (extend), `tests/unit/test_config.py` (extend), plus an `@live` smoke in `tests/integration/test_global_search_live.py`.

---

### Task 1: Config fields

**Files:**
- Modify: `src/graph_extract/config.py`
- Test: `tests/unit/test_config.py` (extend)

**Interfaces:**
- Produces on `ExtractSettings`: `map_llm_base_url: str`, `map_llm_model: str`, `map_llm_api_key: str`, `global_shortlist_k: int`, `global_default_level: int`, `global_map_relevance_min: int`.

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_config.py`)

```python
def test_global_search_defaults():
    from graph_extract.config import ExtractSettings
    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    assert ExtractSettings.model_fields["map_llm_base_url"].default == ""
    assert s.global_shortlist_k == 10
    assert s.global_default_level == 1
    assert s.global_map_relevance_min == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_config.py::test_global_search_defaults -q`
Expected: FAIL (fields do not exist).

- [ ] **Step 3: Implement** — in `src/graph_extract/config.py`, inside `class ExtractSettings`, after the theme-builder fields, add:

```python
    # --- global (map-reduce) search (design: global-search) ---
    map_llm_base_url: str = ""   # map tier; defaults to the judge/synthesis tier when empty
    map_llm_model: str = ""
    map_llm_api_key: str = ""
    global_shortlist_k: int = 10
    global_default_level: int = 1
    global_map_relevance_min: int = 2
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_config.py -q`
Expected: PASS. `uv run ruff check src/graph_extract/config.py` + `uv run mypy src/graph_extract/config.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/graph_extract/config.py tests/unit/test_config.py
git commit -m "feat(config): global-search map tier + shortlist settings"
```

---

### Task 2: Shortlist — cosine ranking + `shortlist_communities`

**Files:**
- Create: `src/answer_api/global_search.py`
- Test: `tests/unit/test_global_rank.py`, `tests/integration/test_global_search.py` (shortlist part)

**Interfaces:**
- Produces:
  - `@dataclass CommunityHit: community_id: str; title: str; summary: str; level: int; rating: float; cited_fact_uuids: list[str]; full_report: str; similarity: float`
  - `_cosine(a: list[float], b: list[float]) -> float`
  - `_rank_hits(query_vec: list[float], rows: list[dict], *, k: int, rating_boost: float) -> list[CommunityHit]` (pure; `rows` have keys community_id/title/summary/level/rating/cited_fact_uuids/full_report/embedding)
  - `async shortlist_communities(driver, embedder, q: str, *, level: int, k: int, group_id: str, rating_boost: float = 0.1) -> list[CommunityHit]`

- [ ] **Step 1: Write the failing unit test** (pure ranking — no DB, no embedder)

```python
# tests/unit/test_global_rank.py
from answer_api.global_search import _cosine, _rank_hits


def test_cosine_basic():
    assert _cosine([1, 0], [1, 0]) == 1.0
    assert abs(_cosine([1, 0], [0, 1])) < 1e-9
    assert _cosine([0, 0], [1, 1]) == 0.0            # zero vector -> 0, no div error


def _row(cid, emb, rating=5.0, cited=("f1",)):
    return dict(community_id=cid, title=cid, summary="s", level=1, rating=rating,
                cited_fact_uuids=list(cited), full_report="[]", embedding=emb)


def test_rank_by_cosine_then_rating_boost_and_k():
    q = [1.0, 0.0]
    rows = [_row("a", [1.0, 0.0]), _row("b", [0.9, 0.1]), _row("c", [0.0, 1.0])]
    hits = _rank_hits(q, rows, k=2, rating_boost=0.1)
    assert [h.community_id for h in hits] == ["a", "b"]     # closest two, c dropped by k
    assert hits[0].similarity > hits[1].similarity


def test_rank_skips_cite_less_communities():
    q = [1.0, 0.0]
    rows = [_row("a", [1.0, 0.0], cited=[]), _row("b", [0.5, 0.5])]
    hits = _rank_hits(q, rows, k=5, rating_boost=0.1)
    assert [h.community_id for h in hits] == ["b"]          # 'a' has no cited facts
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_global_rank.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** the ranking core in `src/answer_api/global_search.py`:

```python
# src/answer_api/global_search.py
"""Global map-reduce search over the :Community report layer. Shortlist reports
by query-embedding cosine + rating, MAP each to query-relevant key points (with
validated fact IDs), REDUCE into one cited answer. Citations are graph traversal
(design decision #2): the LLM only ever emits fact UUIDs / [N] markers."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.provenance import Provenance
from graph_extract.usage import instrument
from answer_api.synthesize import _finalize_answer

logger = logging.getLogger(__name__)

_REFUSAL = "I don't have enough thematic coverage to answer that from the community reports."
_URL_RE = re.compile(r"https?://[^\s\[\]]+", re.IGNORECASE)   # local copy (answer_api-independent)


@dataclass
class CommunityHit:
    community_id: str
    title: str
    summary: str
    level: int
    rating: float
    cited_fact_uuids: list[str]
    full_report: str
    similarity: float


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _rank_hits(query_vec: list[float], rows: list[dict], *, k: int,
               rating_boost: float) -> list[CommunityHit]:
    scored: list[tuple[float, CommunityHit]] = []
    for r in rows:
        if not r.get("cited_fact_uuids"):
            continue                        # nothing citable -> useless for a cited answer
        sim = _cosine(query_vec, r["embedding"])
        score = sim + rating_boost * (r.get("rating") or 0.0) / 10.0
        scored.append((score, CommunityHit(
            community_id=r["community_id"], title=r["title"], summary=r["summary"],
            level=r["level"], rating=r.get("rating") or 0.0,
            cited_fact_uuids=list(r["cited_fact_uuids"]), full_report=r["full_report"],
            similarity=sim)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [h for _, h in scored[:k]]
```

- [ ] **Step 4: Run to verify the unit test passes**

Run: `uv run --extra dev pytest tests/unit/test_global_rank.py -q`
Expected: PASS (3 tests). ruff + `mypy src/answer_api/global_search.py` clean.

- [ ] **Step 5: Add `shortlist_communities` + its integration test**

Append to `src/answer_api/global_search.py`:

```python
async def shortlist_communities(driver, embedder, q: str, *, level: int, k: int,
                                group_id: str, rating_boost: float = 0.1) -> list[CommunityHit]:
    query_vec = (await embedder.create_batch([q]))[0]
    async with driver.session() as s:
        r = await s.run(
            "MATCH (c:Community {group_id:$g, level:$lvl}) "
            "RETURN c.community_id AS community_id, c.title AS title, "
            "coalesce(c.summary,'') AS summary, c.level AS level, "
            "coalesce(c.rating,0.0) AS rating, "
            "coalesce(c.cited_fact_uuids,[]) AS cited_fact_uuids, "
            "coalesce(c.full_report,'[]') AS full_report, c.embedding AS embedding",
            g=group_id, lvl=level)
        rows = [dict(rec) async for rec in r if rec["embedding"]]
    return _rank_hits(query_vec, rows, k=k, rating_boost=rating_boost)
```

Test (Neo4j testcontainer + fake embedder — no live embed):

```python
# tests/integration/test_global_search.py
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


class _FakeEmbedder:
    def __init__(self, vec): self._vec = vec
    async def create_batch(self, texts): return [self._vec for _ in texts]


async def test_shortlist_ranks_and_filters_by_level(extract_driver):
    from answer_api.global_search import shortlist_communities
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Community {group_id:$g, level:1, community_id:'near', title:'Near', "
                    "summary:'x', rating:6.0, cited_fact_uuids:['f1'], full_report:'[]', embedding:[1.0,0.0]})", g=g)
        await s.run("CREATE (:Community {group_id:$g, level:1, community_id:'far', title:'Far', "
                    "summary:'x', rating:9.0, cited_fact_uuids:['f2'], full_report:'[]', embedding:[0.0,1.0]})", g=g)
        await s.run("CREATE (:Community {group_id:$g, level:0, community_id:'wronglvl', title:'L0', "
                    "summary:'x', rating:9.0, cited_fact_uuids:['f3'], full_report:'[]', embedding:[1.0,0.0]})", g=g)
    hits = await shortlist_communities(extract_driver, _FakeEmbedder([1.0, 0.0]), "q",
                                       level=1, k=5, group_id=g)
    assert [h.community_id for h in hits] == ["near", "far"]   # level-1 only, near first
```

- [ ] **Step 6: Run + commit**

Run: `uv run --extra dev pytest tests/unit/test_global_rank.py tests/integration/test_global_search.py -q`
Expected: PASS. ruff + mypy clean.

```bash
git add src/answer_api/global_search.py tests/unit/test_global_rank.py tests/integration/test_global_search.py
git commit -m "feat(global): community shortlist (cosine + rating, level-scoped)"
```

---

### Task 3: Map — `map_report` + `_map_client_and_model`

**Files:**
- Modify: `src/answer_api/global_search.py`
- Test: `tests/unit/test_global_map.py`

**Interfaces:**
- Consumes: `CommunityHit` (Task 2), `ExtractSettings`.
- Produces:
  - `@dataclass MapResult: community_id: str; title: str; relevance: int; key_points: list[str]; fact_ids: list[str]`
  - `_map_client_and_model(settings) -> tuple[AsyncOpenAI, str]`
  - `_extract_json(raw: str) -> dict | None`
  - `async map_report(client, model, q: str, hit: CommunityHit, *, relevance_min: int) -> MapResult | None`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_global_map.py
import json
import pytest
from answer_api.global_search import CommunityHit, map_report, _map_client_and_model
from graph_extract.config import ExtractSettings

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


class _FakeClient:
    def __init__(self, contents):
        self._c = list(contents); self.chat = self; self.completions = self
    async def create(self, **kw):
        c = self._c.pop(0)
        return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": c})()})()]})


def _hit(cited=("f1", "f2")):
    return CommunityHit("c1", "T", "sum", 1, 7.0, list(cited), "[]", 0.9)


@pytest.mark.asyncio
async def test_map_keeps_only_real_fact_ids_above_floor():
    payload = json.dumps({"relevance": 8, "key_points": ["kp1", "kp2"],
                          "fact_ids": ["f1", "f9-halluc"]})
    m = await map_report(_FakeClient([payload]), "mm", "q", _hit(), relevance_min=2)
    assert m is not None and m.fact_ids == ["f1"] and m.relevance == 8
    assert m.key_points == ["kp1", "kp2"] and m.community_id == "c1"


@pytest.mark.asyncio
async def test_map_low_relevance_dropped():
    payload = json.dumps({"relevance": 1, "key_points": ["x"], "fact_ids": ["f1"]})
    assert await map_report(_FakeClient([payload]), "mm", "q", _hit(), relevance_min=2) is None


@pytest.mark.asyncio
async def test_map_bad_json_retries_then_none():
    assert await map_report(_FakeClient(["nope", "still nope"]), "mm", "q", _hit(),
                            relevance_min=2) is None


def test_map_tier_defaults_to_judge():
    s = ExtractSettings(_env_file=None, judge_base_url="http://glm", judge_model="glm-5.2:cloud",
                        judge_api_key="jk", **_MIN)
    _, model = _map_client_and_model(s)
    assert model == "glm-5.2:cloud"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_global_map.py -q`
Expected: FAIL (map symbols missing).

- [ ] **Step 3: Implement** — append to `src/answer_api/global_search.py`:

```python
@dataclass
class MapResult:
    community_id: str
    title: str
    relevance: int
    key_points: list[str]
    fact_ids: list[str]


_MAP_PROMPT = (
    "You are assessing one COMMUNITY report for relevance to a QUESTION. Using ONLY "
    "the report, respond with JSON: {\"relevance\": 0-10 (how useful for the question), "
    "\"key_points\": [short strings relevant to the question], \"fact_ids\": [the fact "
    "uuids from the report that support those points]}. fact_ids MUST be uuids that "
    "appear in the report. Do NOT write URLs.\n\n"
    "QUESTION: {q}\n\nCOMMUNITY \"{title}\": {summary}\nFINDINGS: {full_report}"
)


def _map_client_and_model(settings: ExtractSettings) -> tuple[AsyncOpenAI, str]:
    base = settings.map_llm_base_url or settings.judge_base_url
    model = settings.map_llm_model or settings.judge_model
    key = settings.map_llm_api_key or settings.judge_api_key or "not-needed"
    if not base or not model:
        raise ValueError("No map model configured. Set map_llm_* or judge_* (GLM-5.2).")
    return instrument(AsyncOpenAI(api_key=key, base_url=base)), model


def _extract_json(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


async def map_report(client: AsyncOpenAI, model: str, q: str, hit: CommunityHit, *,
                     relevance_min: int) -> MapResult | None:
    prompt = _MAP_PROMPT.format(q=q, title=hit.title, summary=hit.summary,
                                full_report=hit.full_report)
    obj: dict | None = None
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}])
        obj = _extract_json(resp.choices[0].message.content or "")
        if obj is not None:
            break
    if obj is None:
        return None
    try:
        relevance = int(obj.get("relevance", 0) or 0)
    except (TypeError, ValueError):
        relevance = 0
    if relevance < relevance_min:
        return None
    valid = {u for u in hit.cited_fact_uuids}
    fact_ids = [f for f in (obj.get("fact_ids") or []) if f in valid]
    key_points = [str(p) for p in (obj.get("key_points") or [])]
    return MapResult(community_id=hit.community_id, title=hit.title, relevance=relevance,
                     key_points=key_points, fact_ids=fact_ids)
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_global_map.py -q`
Expected: PASS (4 tests). ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/global_search.py tests/unit/test_global_map.py
git commit -m "feat(global): per-report map (relevance + key points + validated fact ids)"
```

---

### Task 4: Reduce + resolve — `global_search`

**Files:**
- Modify: `src/answer_api/global_search.py`
- Test: `tests/integration/test_global_search.py` (extend)

**Interfaces:**
- Consumes: `shortlist_communities`, `map_report` (this module), `Provenance`, `_finalize_answer`.
- Produces: `async global_search(driver, embedder, map_client, map_model, synth_client, synth_model, *, q, level, k, group_id, relevance_min) -> dict` returning `{query, answer, citations:[{marker,fact_uuid,sources}], communities_used:[{community_id,title,relevance}]}`.

- [ ] **Step 1: Write the failing integration test** (testcontainer + fake LLM clients + fake embedder; seeds facts+episodes+article so provenance resolves)

```python
# add to tests/integration/test_global_search.py
async def test_global_search_end_to_end(extract_driver):
    from answer_api.global_search import global_search
    import json
    g = "backup-docs"
    # seed a community + a real fact whose provenance resolves to a URL
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")   # isolate: shared module-scoped container
        await s.run("CREATE (a:Article {id:'art1', source_url:'https://x/art1', title:'T'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1', group_id:$g})", g=g)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1'], "
                    "fact:'AWS Backup supports S3'}]->(y:Entity)", g=g)
        await s.run("CREATE (:Community {group_id:$g, level:1, community_id:'c1', title:'S3 backup', "
                    "summary:'s', rating:8.0, cited_fact_uuids:['f1'], full_report:'[]', embedding:[1.0,0.0]})", g=g)

    class _Emb:
        async def create_batch(self, texts): return [[1.0, 0.0] for _ in texts]

    class _MapClient:   # returns a valid map result citing f1
        def __init__(self): self.chat = self; self.completions = self
        async def create(self, **kw):
            c = json.dumps({"relevance": 9, "key_points": ["S3 supported"], "fact_ids": ["f1"]})
            return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": c})()})()]})

    class _ReduceClient:  # cites [1] and (badly) writes a URL + an invalid marker
        def __init__(self): self.chat = self; self.completions = self
        async def create(self, **kw):
            c = "AWS Backup supports S3 [1]. See https://evil/x [9]."
            return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": c})()})()]})

    res = await global_search(extract_driver, _Emb(), _MapClient(), "mm", _ReduceClient(), "rm",
                              q="how is S3 backed up", level=1, k=5, group_id=g, relevance_min=2)
    assert res["citations"][0]["fact_uuid"] == "f1"
    # Provenance.resolve_citations returns sources keyed {url, title, article_id}
    assert res["citations"][0]["sources"][0]["url"] == "https://x/art1"          # #2 chain
    assert "http" not in res["answer"]                     # URL stripped
    assert [c["marker"] for c in res["citations"]] == [1]  # invalid [9] dropped
    assert res["communities_used"][0]["community_id"] == "c1"


async def test_global_search_empty_shortlist_refuses(extract_driver):
    from answer_api.global_search import global_search, _REFUSAL
    g = "backup-docs-empty"

    class _Emb:
        async def create_batch(self, texts): return [[1.0, 0.0] for _ in texts]

    class _Boom:   # must NOT be called (no communities -> no LLM)
        def __init__(self): self.chat = self; self.completions = self
        async def create(self, **kw): raise AssertionError("no LLM on empty shortlist")

    res = await global_search(extract_driver, _Emb(), _Boom(), "mm", _Boom(), "rm",
                              q="x", level=1, k=5, group_id=g, relevance_min=2)
    assert res["answer"] == _REFUSAL and res["citations"] == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/integration/test_global_search.py -k end_to_end -q`
Expected: FAIL (`global_search` missing).

- [ ] **Step 3: Implement** — append to `src/answer_api/global_search.py`:

```python
_REDUCE_PROMPT = (
    "Answer the QUESTION by synthesizing across these community findings, organized "
    "by theme and vendor. Cite every claim with the [N] fact markers shown. Use ONLY "
    "these findings. Do NOT write any URL. If nothing is relevant, reply exactly: "
    "\"" + _REFUSAL + "\"\n\nQUESTION: {q}\n\nFINDINGS:\n{blocks}\n\nAnswer:"
)


async def global_search(driver, embedder, map_client: AsyncOpenAI, map_model: str,
                        synth_client: AsyncOpenAI, synth_model: str, *, q: str,
                        level: int, k: int, group_id: str, relevance_min: int) -> dict:
    hits = await shortlist_communities(driver, embedder, q, level=level, k=k, group_id=group_id)
    if not hits:
        return {"query": q, "answer": _REFUSAL, "citations": [], "communities_used": []}
    maps = await asyncio.gather(
        *[map_report(map_client, map_model, q, h, relevance_min=relevance_min) for h in hits],
        return_exceptions=True)
    results: list[MapResult] = []
    for m in maps:
        if isinstance(m, BaseException):   # gather(return_exceptions=True) yields BaseException; narrows else to MapResult
            logger.warning("global map failed: %s", m)
        elif m is not None:
            results.append(m)
    if not results:
        return {"query": q, "answer": _REFUSAL, "citations": [], "communities_used": []}
    # number the ordered-unique union of fact ids -> marker_map
    marker_map: dict[int, dict] = {}
    fact_to_marker: dict[str, int] = {}
    for m in results:
        for fid in m.fact_ids:
            if fid not in fact_to_marker:
                idx = len(fact_to_marker) + 1
                fact_to_marker[fid] = idx
                marker_map[idx] = {"fact_uuid": fid}
    blocks = []
    for m in results:
        markers = " ".join(f"[{fact_to_marker[f]}]" for f in m.fact_ids)
        pts = "\n".join(f"- {p}" for p in m.key_points)
        blocks.append(f"COMMUNITY \"{m.title}\" (relevance {m.relevance}):\n{pts}\n"
                      f"Supporting facts: {markers}")
    resp = await synth_client.chat.completions.create(
        model=synth_model, temperature=0, max_tokens=3000,
        messages=[{"role": "user", "content": _REDUCE_PROMPT.format(q=q, blocks="\n\n".join(blocks))}])
    answer, cited = _finalize_answer(resp.choices[0].message.content or "", marker_map)
    resolved = await Provenance(driver).resolve_citations(
        [marker_map[m]["fact_uuid"] for m in cited])
    citations = [{"marker": m, "fact_uuid": marker_map[m]["fact_uuid"],
                  "sources": resolved.get(marker_map[m]["fact_uuid"], [])} for m in cited]
    return {"query": q, "answer": answer, "citations": citations,
            "communities_used": [{"community_id": m.community_id, "title": m.title,
                                  "relevance": m.relevance} for m in results]}
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/integration/test_global_search.py -q`
Expected: PASS. ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/global_search.py tests/integration/test_global_search.py
git commit -m "feat(global): reduce synthesis + fact-id citation resolution"
```

---

### Task 5: App wiring — lifespan + `GET /search/global`

**Files:**
- Modify: `src/answer_api/app.py`
- Test: `tests/unit/test_answer_api_app.py` (extend)

**Interfaces:**
- Consumes: `global_search.global_search`, `global_search._map_client_and_model`, `graphiti_client.build_embedder`.
- Produces: `app.state.embedder`, `app.state.map_client`, `app.state.map_model`; `GET /search/global?q=&level=&k=`.

- [ ] **Step 1: Write the failing test** (extend `tests/unit/test_answer_api_app.py`; the `_stub_deps` autouse fixture already stubs build_graphiti/_build_driver/_synthesis_client_and_model — add stubs for the embedder, map client, and `global_search`)

```python
# add near the other fakes in tests/unit/test_answer_api_app.py
async def _fake_global_search(driver, embedder, map_client, map_model, synth_client,
                              synth_model, *, q, level, k, group_id, relevance_min):
    return {"query": q, "answer": "AWS and Azure both back up S3 [1].",
            "citations": [{"marker": 1, "fact_uuid": "f1",
                           "sources": [{"article_id": "art1", "source_url": "https://x/art1"}]}],
            "communities_used": [{"community_id": "c1", "title": "S3", "relevance": 9}]}
```

Add to the `_stub_deps` autouse fixture body:

```python
    import answer_api.global_search as global_mod
    monkeypatch.setattr(app_mod, "build_embedder", lambda s: object())
    monkeypatch.setattr(global_mod, "_map_client_and_model",
                        lambda s: (FakeSynthClient(), "map-model"))
    monkeypatch.setattr(global_mod, "global_search", _fake_global_search)
```

And the tests:

```python
async def test_global_returns_stubbed_answer():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/global", params={"q": "compare vendors on S3"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"query", "answer", "citations", "communities_used"}
    assert body["citations"][0]["fact_uuid"] == "f1"


async def test_global_requires_q():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/global")
    assert resp.status_code == 422
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_answer_api_app.py -k global -q`
Expected: FAIL (route missing / `build_embedder` not imported in app).

- [ ] **Step 3: Implement** — in `src/answer_api/app.py`:

1. Imports (top): add
```python
from answer_api import global_search as global_mod
from graph_extract.graphiti_client import build_embedder
```

2. In `_lifespan`, after `synth_client, synth_model = _synthesis_client_and_model(settings)` and its `except` guard, build the embedder + map client (guard-and-cleanup on failure):
```python
    try:
        embedder = build_embedder(settings)
        map_client, map_model = global_mod._map_client_and_model(settings)
    except Exception:
        await graphiti.close()
        await driver.close()
        await synth_client.close()
        raise
    app.state.embedder = embedder
    app.state.map_client = map_client
    app.state.map_model = map_model
```
Store them on `app.state` (alongside the existing lines). In the shutdown `finally` tuple, add `("map_client", map_client.close)` (the embedder has no async client to close here — it shares the TEI endpoint; do not close it).

3. Add the endpoint inside `create_app`:
```python
    @app.get("/search/global")
    async def search_global(
        q: str, level: int | None = Query(None, ge=0), k: int | None = Query(None, ge=1)
    ) -> dict[str, Any]:
        st = app.state
        return await global_mod.global_search(
            st.driver, st.embedder, st.map_client, st.map_model,
            st.synth_client, st.synth_model,
            q=q, level=st.settings.global_default_level if level is None else level,
            k=st.settings.global_shortlist_k if k is None else k,
            group_id=st.settings.group_id,
            relevance_min=st.settings.global_map_relevance_min)
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_answer_api_app.py -q`
Expected: PASS (all app tests, incl. the 2 new + `/health`,`/search/local`,`/answer`,`/timeline` regressions). ruff + `mypy src/answer_api/app.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/app.py tests/unit/test_answer_api_app.py
git commit -m "feat(answer-api): GET /search/global endpoint + lifespan embedder/map client"
```

---

### Task 6: Full-suite gate + `@live` smoke + demonstration

**Files:**
- Create: `tests/integration/test_global_search_live.py`, `docs/superpowers/global-search-report.md`

- [ ] **Step 1: Full non-live suite**

Run: `uv run --extra dev pytest -m "not live" -q`
Expected: PASS (all green). Fix any fallout before continuing.

- [ ] **Step 2: `@live` smoke** (real GLM + embedder + the populated `:Community` layer)

```python
# tests/integration/test_global_search_live.py
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_global_search_live(live_extract_driver):
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_embedder
    from answer_api.synthesize import _synthesis_client_and_model
    from answer_api.global_search import global_search, _map_client_and_model
    s = get_extract_settings()
    emb = build_embedder(s)
    mc, mm = _map_client_and_model(s)
    sc, sm = _synthesis_client_and_model(s)
    try:
        res = await global_search(
            live_extract_driver, emb, mc, mm, sc, sm,
            q="How do AWS Backup and Azure Backup handle backup retention?",
            level=s.global_default_level, k=s.global_shortlist_k,
            group_id=s.group_id, relevance_min=s.global_map_relevance_min)
        assert res["answer"] and res["communities_used"]
        assert "http" not in res["answer"]                       # no LLM-authored URL
        for c in res["citations"]:
            assert c["sources"], "cited fact must resolve to >=1 source"
    finally:
        await mc.close(); await sc.close()
```

Run: `uv run --extra dev pytest -m live tests/integration/test_global_search_live.py -q`
Expected: PASS.

- [ ] **Step 3: Controller demonstration**

Start the API (`uv run --extra dev uvicorn answer_api.app:main --factory --port 8099`) or call `global_search` directly; run a real cross-vendor query on `backup-docs`; write `docs/superpowers/global-search-report.md` showing the cited cross-vendor answer, the `communities_used`, and that `citations[].sources` resolve to real source URLs (and no URL was authored by the LLM).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_global_search_live.py docs/superpowers/global-search-report.md
git commit -m "test(global): live smoke + demonstration report"
```

---

## Notes for the implementer

- `global_search`/`map_report`/`shortlist_communities` are all injectable (module-level names) so tests substitute fake LLM/embedder clients; only the `@live` smoke + demonstration need real infra.
- Do not let any LLM text carry a citation URL (decision #2): map fact_ids are validated ⊆ the report's `cited_fact_uuids`; reduce markers validated via `marker_map`; `_finalize_answer` strips URLs.
- The reduce/map use generous `max_tokens` (GLM reasoning headroom); a too-small cap truncates to empty (slice-1 lesson).
- `build_embedder`'s embedder shares the TEI endpoint (no owned client to close in the app shutdown); only the `map_client` (a fresh `AsyncOpenAI`) is closed.
