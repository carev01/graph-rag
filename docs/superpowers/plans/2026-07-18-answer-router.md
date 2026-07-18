# `/answer` Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `/answer` into a router that classifies each query to a retrieval mode (local / global / drift / timeline), dispatches to the existing mode function, and returns a uniform `{mode, answer, citations, routing, …}` envelope.

**Architecture:** A new `answer_api/router.py` with `classify` (heuristic guardrails → cheap-tier LLM → default drift), a deterministic `_render_timeline`, a pure `_normalize`, and the `answer_router` orchestrator (dispatch + one local-empty→drift fallback). `/answer` is rewired to it; the former local-synthesis behavior becomes `/answer?mode=local`. The router authors no prose and no URL — answers/citations come from the mode functions or the deterministic timeline render.

**Tech Stack:** Python 3.12, FastAPI (`answer_api`), the cheap tier (`cheap_llm_*`, ling-2.6-flash) via OpenAI-compatible client, pytest (unit + app tests with fakes; `@live` for the real stack).

## Global Constraints

- **Design decision #2 — citations are graph traversal, never LLM:** the router only emits a mode label and reshapes; it never synthesizes prose or writes a URL. Every `answer`/`citations` pair comes from a mode function or the deterministic `_render_timeline`, all of which resolve URLs via `Provenance`. The cheap classifier's output is a mode label, never user-facing text.
- **Uniform envelope:** `{mode, query, answer, citations, routing}` always present; `communities_used` (global/drift), `follow_ups` (drift), `timeline` (timeline) only for their modes. `routing = {chosen, via, fallback_from}` (+ optional `degraded`). Citation shape is the existing `{marker, fact_uuid, sources:[{url,title,article_id}]}`.
- **Classification order:** `?mode=` override → temporal-regex→timeline → cross-vendor-regex→global → cheap LLM → default `drift`. No cheap key ⇒ heuristics-only, default drift. Never crash on classifier failure.
- **One fallback only:** `local` with `retrieved==0` → re-dispatch as `drift` (`routing.fallback_from="local"`). No loops.
- **`Mode = Literal["local","global","drift","timeline"]`** — an invalid `?mode=` value → 422 at the endpoint.
- Run with `uv`: `uv run --extra dev pytest …`, `uv run ruff check src tests` (CI gate — lints tests; **no semicolons in fakes (E702), imports at file top (E402)**), `uv run mypy src`. Hermetic settings: `ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")`.

---

## File Structure

- **Modify** `src/graph_extract/config.py` — `router_default_mode` field.
- **Create** `src/answer_api/router.py` — `Mode`, `classify`, `_cheap_classify_client`, `_render_timeline`, `_normalize`, `_dispatch`, `answer_router`, prompts/regexes.
- **Modify** `src/answer_api/app.py` — rewire `/answer` to the router; lifespan builds the cheap classifier client.
- **Tests:** `tests/unit/test_config.py` (extend), `tests/unit/test_router_classify.py`, `tests/unit/test_router_normalize.py`, `tests/unit/test_router_dispatch.py`, `tests/unit/test_answer_api_app.py` (modify the `/answer` tests), `tests/integration/test_answer_router_live.py`.

---

### Task 1: Config field

**Files:**
- Modify: `src/graph_extract/config.py`
- Test: `tests/unit/test_config.py` (extend)

**Interfaces:**
- Produces on `ExtractSettings`: `router_default_mode: str`.

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_config.py`)

```python
def test_router_default_mode():
    from graph_extract.config import ExtractSettings
    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    assert s.router_default_mode == "drift"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_config.py::test_router_default_mode -q`
Expected: FAIL (field does not exist).

- [ ] **Step 3: Implement** — in `src/graph_extract/config.py`, inside `class ExtractSettings`, after the DRIFT-search block, add:

```python
    # --- /answer router (design: answer-router) ---
    router_default_mode: str = "drift"   # mode when no heuristic fires and the cheap classifier is absent/uncertain
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_config.py -q`
Expected: PASS. `uv run ruff check src/graph_extract/config.py` + `uv run mypy src/graph_extract/config.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/graph_extract/config.py tests/unit/test_config.py
git commit -m "feat(config): router_default_mode"
```

---

### Task 2: `router.py` scaffold — `Mode`, `classify`, `_cheap_classify_client`

**Files:**
- Create: `src/answer_api/router.py`
- Test: `tests/unit/test_router_classify.py`

**Interfaces:**
- Produces:
  - `Mode = Literal["local","global","drift","timeline"]`
  - `_cheap_classify_client(settings: ExtractSettings) -> tuple[AsyncOpenAI | None, str]`
  - `async classify(q, *, cheap_client, cheap_model, mode_override=None, default_mode="drift") -> tuple[Mode, str]` — returns `(mode, via)`, `via ∈ {"override","heuristic","llm","default"}`.

- [ ] **Step 1: Write the failing test** (`tests/unit/test_router_classify.py`)

```python
import pytest
from answer_api.router import classify, _cheap_classify_client
from graph_extract.config import ExtractSettings

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


class _FakeCheap:
    def __init__(self, label):
        self._label = label
        self.chat = self
        self.completions = self
    async def create(self, **kw):
        return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": self._label})()})()]})


@pytest.mark.asyncio
async def test_override_wins():
    mode, via = await classify("anything", cheap_client=None, cheap_model="",
                               mode_override="global")
    assert (mode, via) == ("global", "override")


@pytest.mark.asyncio
async def test_temporal_heuristic():
    mode, via = await classify("How has Veeam immutability changed over time?",
                               cheap_client=None, cheap_model="")
    assert (mode, via) == ("timeline", "heuristic")


@pytest.mark.asyncio
async def test_cross_vendor_heuristic():
    mode, via = await classify("Compare all vendors on ransomware protection",
                               cheap_client=None, cheap_model="")
    assert (mode, via) == ("global", "heuristic")


@pytest.mark.asyncio
async def test_llm_path_for_uncaught_query():
    mode, via = await classify("Does Veeam support S3 Object Lock?",
                               cheap_client=_FakeCheap("local"), cheap_model="ling")
    assert (mode, via) == ("local", "llm")


@pytest.mark.asyncio
async def test_no_cheap_client_defaults_to_drift():
    mode, via = await classify("What should I consider for retention?",
                               cheap_client=None, cheap_model="")
    assert (mode, via) == ("drift", "default")


@pytest.mark.asyncio
async def test_unknown_llm_label_defaults():
    mode, via = await classify("some question", cheap_client=_FakeCheap("banana"),
                               cheap_model="ling")
    assert (mode, via) == ("drift", "default")


def test_cheap_client_none_without_key():
    s = ExtractSettings(_env_file=None, cheap_llm_api_key="", **_MIN)
    client, model = _cheap_classify_client(s)
    assert client is None and model == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_router_classify.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — create `src/answer_api/router.py`:

```python
"""The /answer router: classify a query to a retrieval mode (heuristics first, a
cheap-tier LLM otherwise, default drift), dispatch to the existing mode function,
and normalize every mode's output into one uniform envelope. The router authors
no prose and no URL (design decision #2) — it only picks a mode label and
reshapes; answers/citations come from the mode functions or the deterministic
timeline render."""
from __future__ import annotations

import logging
import re
from typing import Literal, cast

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.usage import instrument

logger = logging.getLogger(__name__)

Mode = Literal["local", "global", "drift", "timeline"]
_MODE_ORDER: tuple[str, ...] = ("local", "global", "drift", "timeline")
_MODES = frozenset(_MODE_ORDER)

# High-confidence heuristic guardrails (checked before the LLM). Case-insensitive.
_TEMPORAL_RE = re.compile(
    r"\b(chang(e|ed|es|ing)|history|used to|since \d|over time|evolv|"
    r"deprecat|no longer|when did|previously)\b", re.IGNORECASE)
_CROSS_VENDOR_RE = re.compile(
    r"\b(compare|comparison|across (all )?vendors|all vendors|which vendors|"
    r"every vendor)\b", re.IGNORECASE)

_CLASSIFY_PROMPT = (
    "Classify the QUESTION about backup products into exactly one retrieval mode. "
    "Reply with ONLY the lowercase label, nothing else.\n"
    "- local: a specific factual question about one product/feature\n"
    "- global: a broad thematic/cross-corpus question spanning many vendors\n"
    "- drift: a broad question that still expects concrete, sourced detail\n"
    "- timeline: how something changed over time\n\n"
    "QUESTION: {q}\nLabel:"
)


def _cheap_classify_client(settings: ExtractSettings) -> tuple[AsyncOpenAI | None, str]:
    """The cheap classifier client (ling), or (None, '') when no cheap key is set
    -> the router runs heuristics-only and defaults uncaught queries to drift."""
    if not settings.cheap_llm_api_key:
        return None, ""
    client = instrument(AsyncOpenAI(
        api_key=settings.cheap_llm_api_key, base_url=settings.cheap_llm_base_url))
    return client, settings.cheap_llm_model


async def classify(q: str, *, cheap_client, cheap_model,
                   mode_override: str | None = None,
                   default_mode: str = "drift") -> tuple[Mode, str]:
    chosen, via = default_mode, "default"
    if mode_override in _MODES:
        chosen, via = cast(str, mode_override), "override"
    elif _TEMPORAL_RE.search(q):
        chosen, via = "timeline", "heuristic"
    elif _CROSS_VENDOR_RE.search(q):
        chosen, via = "global", "heuristic"
    elif cheap_client is not None:
        try:
            resp = await cheap_client.chat.completions.create(
                model=cheap_model, temperature=0, max_tokens=8,
                messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(q=q)}])
            label = (resp.choices[0].message.content or "").strip().lower()
            match = next((m for m in _MODE_ORDER if m in label), None)
            if match is not None:
                chosen, via = match, "llm"
        except Exception:
            logger.warning("cheap classifier failed; defaulting", exc_info=True)
    return cast(Mode, chosen), via
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_router_classify.py -q`
Expected: PASS (7 tests). ruff + `mypy src/answer_api/router.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/router.py tests/unit/test_router_classify.py
git commit -m "feat(router): mode classifier (heuristics + cheap LLM, default drift)"
```

---

### Task 3: Pure normalization — `_render_timeline`, `_normalize`

**Files:**
- Modify: `src/answer_api/router.py`
- Test: `tests/unit/test_router_normalize.py`

**Interfaces:**
- Consumes: `Mode`.
- Produces:
  - `_render_timeline(timeline_result: dict) -> tuple[str, list[dict]]`
  - `_normalize(mode: Mode, via: str, fallback_from: str | None, raw: dict, q: str) -> dict`

- [ ] **Step 1: Write the failing test** (`tests/unit/test_router_normalize.py`)

```python
from answer_api.router import _render_timeline, _normalize


def test_render_timeline_builds_markdown_and_citations():
    raw = {"timeline": [
        {"fact": "Veeam added immutability", "fact_uuid": "f1", "valid_at": "2023-01-01",
         "invalid_at": None, "status": "current", "sources": [{"url": "https://x/1"}]},
        {"fact": "Old behavior", "fact_uuid": "f2", "valid_at": "2020-01-01",
         "invalid_at": "2023-01-01", "status": "superseded", "sources": [{"url": "https://x/2"}]},
    ]}
    answer, citations = _render_timeline(raw)
    assert "[1]" in answer and "[2]" in answer
    assert "Veeam added immutability" in answer
    assert [c["marker"] for c in citations] == [1, 2]
    assert citations[0]["fact_uuid"] == "f1"
    assert citations[1]["sources"][0]["url"] == "https://x/2"


def test_render_timeline_empty():
    answer, citations = _render_timeline({"timeline": []})
    assert citations == []
    assert "No recorded changes" in answer


def test_normalize_global_carries_communities():
    raw = {"query": "q", "answer": "A [1]",
           "citations": [{"marker": 1, "fact_uuid": "f1", "sources": []}],
           "communities_used": [{"community_id": "c1", "title": "T"}]}
    env = _normalize("global", "heuristic", None, raw, "q")
    assert env["mode"] == "global"
    assert env["routing"] == {"chosen": "global", "via": "heuristic", "fallback_from": None}
    assert env["communities_used"] == [{"community_id": "c1", "title": "T"}]
    assert "follow_ups" not in env and "timeline" not in env


def test_normalize_drift_degrade_surfaces_flag():
    raw = {"query": "q", "answer": "local A", "citations": [],
           "retrieved": 2, "cited": 0, "degraded": "no-primer-communities"}
    env = _normalize("drift", "llm", None, raw, "q")
    assert env["routing"]["degraded"] == "no-primer-communities"
    assert env["answer"] == "local A"
    assert "communities_used" not in env      # degrade shape has none


def test_normalize_timeline_renders_and_carries_array():
    raw = {"timeline": [{"fact": "F", "fact_uuid": "f1", "valid_at": "2023-01-01",
                         "invalid_at": None, "status": "current", "sources": []}]}
    env = _normalize("timeline", "heuristic", None, raw, "q")
    assert env["mode"] == "timeline"
    assert "[1]" in env["answer"]
    assert env["citations"][0]["fact_uuid"] == "f1"
    assert env["timeline"] == raw["timeline"]


def test_normalize_records_fallback_from():
    raw = {"query": "q", "answer": "A", "citations": []}
    env = _normalize("drift", "llm", "local", raw, "q")
    assert env["routing"]["fallback_from"] == "local"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_router_normalize.py -q`
Expected: FAIL (symbols missing).

- [ ] **Step 3: Implement** — append to `src/answer_api/router.py`:

```python
def _render_timeline(timeline_result: dict) -> tuple[str, list[dict]]:
    events = timeline_result.get("timeline", [])
    if not events:
        return "No recorded changes for that query.", []
    lines: list[str] = []
    citations: list[dict] = []
    for i, e in enumerate(events, 1):
        valid = e.get("valid_at")
        invalid = e.get("invalid_at")
        span = f"valid_at {valid}" if valid else "no recorded start"
        if invalid:
            span += f", invalid_at {invalid}"
        lines.append(f"- **{e['fact']}** — {span} ({e.get('status', '')}) [{i}]")
        citations.append({"marker": i, "fact_uuid": e["fact_uuid"],
                          "sources": e.get("sources", [])})
    return "\n".join(lines), citations


def _normalize(mode: Mode, via: str, fallback_from: str | None, raw: dict,
               q: str) -> dict:
    routing: dict = {"chosen": mode, "via": via, "fallback_from": fallback_from}
    if "degraded" in raw:
        routing["degraded"] = raw["degraded"]
    if mode == "timeline":
        answer, citations = _render_timeline(raw)
    else:
        answer = raw.get("answer", "")
        citations = raw.get("citations", [])
    env: dict = {"mode": mode, "query": q, "answer": answer,
                 "citations": citations, "routing": routing}
    if mode == "timeline":
        env["timeline"] = raw.get("timeline", [])
    if "communities_used" in raw:
        env["communities_used"] = raw["communities_used"]
    if "follow_ups" in raw:
        env["follow_ups"] = raw["follow_ups"]
    return env
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_router_normalize.py -q`
Expected: PASS (6 tests). ruff + mypy clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/router.py tests/unit/test_router_normalize.py
git commit -m "feat(router): deterministic timeline render + uniform-envelope normalizer"
```

---

### Task 4: Orchestrator — `_dispatch`, `answer_router`

**Files:**
- Modify: `src/answer_api/router.py`
- Test: `tests/unit/test_router_dispatch.py`

**Interfaces:**
- Consumes: `classify`, `_normalize`; the mode functions `synthesize.answer_local`, `global_search.global_search`, `drift.drift_search`, `timeline.timeline_local`.
- Produces:
  - `async _dispatch(mode, graphiti, driver, embedder, synth_client, synth_model, map_client, map_model, *, q, vendor, settings) -> dict`
  - `async answer_router(graphiti, driver, embedder, synth_client, synth_model, map_client, map_model, cheap_client, cheap_model, *, q, mode_override, vendor, settings) -> dict`

- [ ] **Step 1: Write the failing test** (`tests/unit/test_router_dispatch.py`)

```python
import pytest

import answer_api.synthesize as synth_mod
import answer_api.global_search as global_mod
import answer_api.drift as drift_mod
import answer_api.timeline as timeline_mod
from answer_api.router import answer_router
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


async def _fake_local(*a, **k):
    return {"query": k["q"], "answer": "local [1]",
            "citations": [{"marker": 1, "fact_uuid": "f1", "sources": [{"url": "u"}]}],
            "retrieved": 2, "cited": 1}


async def _fake_local_empty(*a, **k):
    return {"query": k["q"], "answer": "refusal", "citations": [], "retrieved": 0, "cited": 0}


async def _fake_global(*a, **k):
    return {"query": k["q"], "answer": "global [1]",
            "citations": [{"marker": 1, "fact_uuid": "g1", "sources": []}],
            "communities_used": [{"community_id": "c1", "title": "T"}]}


async def _fake_drift(*a, **k):
    return {"query": k["q"], "answer": "drift [1]",
            "citations": [{"marker": 1, "fact_uuid": "d1", "sources": []}],
            "communities_used": [{"community_id": "c1", "title": "T"}],
            "follow_ups": [{"query": "fu", "community_id": "c1", "iteration": 1}]}


async def _fake_timeline(*a, **k):
    return {"query": k["q"], "count": 1,
            "timeline": [{"fact": "F", "fact_uuid": "t1", "valid_at": "2023-01-01",
                          "invalid_at": None, "status": "current", "sources": []}]}


@pytest.fixture(autouse=True)
def _patch_modes(monkeypatch):
    monkeypatch.setattr(synth_mod, "answer_local", _fake_local)
    monkeypatch.setattr(global_mod, "global_search", _fake_global)
    monkeypatch.setattr(drift_mod, "drift_search", _fake_drift)
    monkeypatch.setattr(timeline_mod, "timeline_local", _fake_timeline)


async def _route(mode_override, **over):
    return await answer_router(None, None, None, None, "sm", None, "mm", None, "",
                               q=over.get("q", "q"), mode_override=mode_override,
                               vendor=None, settings=_S)


async def test_routes_global_via_override():
    env = await _route("global")
    assert env["mode"] == "global"
    assert env["routing"]["via"] == "override"
    assert env["communities_used"][0]["community_id"] == "c1"


async def test_routes_timeline_and_renders():
    env = await _route("timeline")
    assert env["mode"] == "timeline"
    assert "[1]" in env["answer"]
    assert env["citations"][0]["fact_uuid"] == "t1"
    assert env["timeline"][0]["fact"] == "F"


async def test_local_empty_escalates_to_drift(monkeypatch):
    monkeypatch.setattr(synth_mod, "answer_local", _fake_local_empty)
    env = await _route("local")
    assert env["mode"] == "drift"
    assert env["routing"]["fallback_from"] == "local"
    assert env["citations"][0]["fact_uuid"] == "d1"


async def test_local_with_results_stays_local():
    env = await _route("local")
    assert env["mode"] == "local"
    assert env["routing"]["fallback_from"] is None
    assert env["citations"][0]["fact_uuid"] == "f1"


async def test_drift_degrade_normalized(monkeypatch):
    async def _degrade(*a, **k):
        return {"query": k["q"], "answer": "local fallback", "citations": [],
                "retrieved": 1, "cited": 0, "degraded": "no-primer-communities"}
    monkeypatch.setattr(drift_mod, "drift_search", _degrade)
    env = await _route("drift")
    assert env["routing"]["degraded"] == "no-primer-communities"
    assert env["answer"] == "local fallback"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_router_dispatch.py -q`
Expected: FAIL (`answer_router` missing).

- [ ] **Step 3: Implement** — in `src/answer_api/router.py`, add the mode-module imports to the top import block (after the existing imports):

```python
from answer_api import synthesize as synth_mod
from answer_api import global_search as global_mod
from answer_api import drift as drift_mod
from answer_api import timeline as timeline_mod
```

Then append:

```python
async def _dispatch(mode, graphiti, driver, embedder, synth_client, synth_model,
                    map_client, map_model, *, q, vendor, settings) -> dict:
    g = settings.group_id
    if mode == "local":
        return await synth_mod.answer_local(
            graphiti, driver, synth_client, synth_model, q=q, vendor=vendor, group_id=g)
    if mode == "global":
        return await global_mod.global_search(
            driver, embedder, map_client, map_model, synth_client, synth_model,
            q=q, level=settings.global_default_level, k=settings.global_shortlist_k,
            group_id=g, relevance_min=settings.global_map_relevance_min)
    if mode == "drift":
        return await drift_mod.drift_search(
            graphiti, driver, embedder, synth_client, synth_model, q=q,
            level=settings.drift_primer_level, iterations=settings.drift_iterations,
            primer_k=settings.drift_primer_k, max_followups=settings.drift_max_followups,
            followup_k=settings.drift_followup_k, group_id=g)
    return await timeline_mod.timeline_local(
        graphiti, driver, q=q, vendor=vendor, group_id=g)


async def answer_router(graphiti, driver, embedder, synth_client, synth_model,
                        map_client, map_model, cheap_client, cheap_model, *,
                        q, mode_override, vendor, settings) -> dict:
    mode, via = await classify(q, cheap_client=cheap_client, cheap_model=cheap_model,
                               mode_override=mode_override,
                               default_mode=settings.router_default_mode)
    raw = await _dispatch(mode, graphiti, driver, embedder, synth_client, synth_model,
                          map_client, map_model, q=q, vendor=vendor, settings=settings)
    fallback_from: str | None = None
    if mode == "local" and raw.get("retrieved") == 0:
        fallback_from = "local"
        mode = "drift"
        raw = await _dispatch("drift", graphiti, driver, embedder, synth_client,
                              synth_model, map_client, map_model, q=q, vendor=vendor,
                              settings=settings)
    return _normalize(mode, via, fallback_from, raw, q)
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_router_dispatch.py -q`
Expected: PASS (5 tests). `uv run ruff check src/answer_api/router.py` + `uv run mypy src/answer_api/router.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/router.py tests/unit/test_router_dispatch.py
git commit -m "feat(router): dispatch + local-empty->drift fallback orchestrator"
```

---

### Task 5: App wiring — `/answer` → router + lifespan cheap client

**Files:**
- Modify: `src/answer_api/app.py`
- Test: `tests/unit/test_answer_api_app.py` (modify the `/answer` tests)

**Interfaces:**
- Consumes: `router.answer_router`, `router._cheap_classify_client`, `router.Mode`.
- Produces: `GET /answer?q=&mode=&vendor=` returning the uniform envelope; `app.state.cheap_client`/`cheap_model`.

- [ ] **Step 1: Modify the app tests** (`tests/unit/test_answer_api_app.py`)

The current `/answer` endpoint calls `answer_local` and has a `k` param. The new one calls the router and drops `k`. Make these edits:

1. Add near the other fakes:
```python
async def _fake_answer_router(graphiti, driver, embedder, synth_client, synth_model,
                              map_client, map_model, cheap_client, cheap_model, *,
                              q, mode_override, vendor, settings):
    return {"mode": mode_override or "drift", "query": q, "answer": "routed answer [1].",
            "citations": [{"marker": 1, "fact_uuid": "f1",
                           "sources": [{"url": "https://x/art1", "title": "T", "article_id": "art1"}]}],
            "routing": {"chosen": mode_override or "drift", "via": "override" if mode_override else "default",
                        "fallback_from": None}}
```
2. In the `_stub_deps` autouse fixture, add:
```python
    import answer_api.router as router_mod
    monkeypatch.setattr(router_mod, "answer_router", _fake_answer_router)
    monkeypatch.setattr(router_mod, "_cheap_classify_client", lambda s: (None, ""))
```
3. **Replace** the two existing `/answer` tests (`test_answer_returns_stubbed_synthesis`, `test_answer_requires_q`) with:
```python
async def test_answer_router_returns_envelope():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/answer", params={"q": "plan retention"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "drift"
    assert body["routing"]["chosen"] == "drift"
    assert body["citations"][0]["fact_uuid"] == "f1"


async def test_answer_mode_override_forces_mode():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/answer", params={"q": "x", "mode": "timeline"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "timeline"


async def test_answer_invalid_mode_is_422():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/answer", params={"q": "x", "mode": "bogus"})
    assert resp.status_code == 422


async def test_answer_requires_q():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/answer")
    assert resp.status_code == 422
```
4. In the two `@pytest.mark.parametrize` lists, **remove the `/answer` entries** (`("/answer", {"q": "x", "k": 0})` from the non-positive-limit test and `("/answer", {"q": "x", "k": 1})` from the limit-one test) — `/answer` no longer has a `k` param, so those cases no longer apply.

- [ ] **Step 2: Run to verify the new tests fail**

Run: `uv run --extra dev pytest tests/unit/test_answer_api_app.py -k "answer" -q`
Expected: FAIL (endpoint still calls `answer_local` / has `k`; `router_mod` not wired).

- [ ] **Step 3: Implement** — edit `src/answer_api/app.py`:

1. Add imports near the other module imports (top of file):
```python
from answer_api import router as router_mod
from answer_api.router import Mode
```
2. In `_lifespan`, after the `map_client` guard block and before the `app.state.*` assignments, build the cheap client:
```python
    cheap_client, cheap_model = router_mod._cheap_classify_client(settings)
```
   (`_cheap_classify_client` never raises — it returns `(None, "")` without a key — so no guard block is needed.) Then add the two state assignments alongside the others:
```python
    app.state.cheap_client = cheap_client
    app.state.cheap_model = cheap_model
```
3. In the shutdown `finally`, change the fixed tuple to a list and conditionally append the cheap client (it may be `None`):
```python
        closers = [
            ("graphiti", graphiti.close),
            ("driver", driver.close),
            ("synth_client", synth_client.close),
            ("map_client", map_client.close),
            ("embedder", lambda: embedder.client.close()),
        ]
        if cheap_client is not None:
            closers.append(("cheap_client", cheap_client.close))
        for name, closer in closers:
            try:
                await closer()
            except Exception:
                logger.exception("error closing %s during shutdown", name)
```
4. **Replace** the `/answer` endpoint with the router call:
```python
    @app.get("/answer")
    async def answer(
        q: str, mode: Mode | None = None, vendor: str | None = None
    ) -> dict[str, Any]:
        st = app.state
        return await router_mod.answer_router(
            st.graphiti, st.driver, st.embedder, st.synth_client, st.synth_model,
            st.map_client, st.map_model, st.cheap_client, st.cheap_model,
            q=q, mode_override=mode, vendor=vendor, settings=st.settings)
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_answer_api_app.py -q`
Expected: PASS (all app tests: the 4 new `/answer` tests + `/health`, `/search/local`, `/answer` gone-old, `/timeline`, `/search/global`, `/search/drift`, and the parametrized cases with `/answer` removed). `uv run ruff check src/answer_api/app.py tests/unit/test_answer_api_app.py` + `uv run mypy src/answer_api/app.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/app.py tests/unit/test_answer_api_app.py
git commit -m "feat(answer-api): /answer becomes the router; lifespan cheap classifier client"
```

---

### Task 6: Full-suite gate + `@live` smoke + demonstration

**Files:**
- Create: `tests/integration/test_answer_router_live.py`, `docs/superpowers/answer-router-report.md`

- [ ] **Step 1: Full non-live suite + CI lint gate**

Run: `uv run --extra dev pytest -m "not live" -q` → all pass.
Run: `uv run ruff check src tests` → clean. `uv run mypy src` → clean.
Fix any fallout before continuing.

- [ ] **Step 2: `@live` smoke** (`tests/integration/test_answer_router_live.py`) — real cheap classifier + the four real modes on `backup-docs`

```python
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_answer_router_live(live_extract_driver):
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
        # a temporal question should route to timeline via the heuristic
        env = await answer_router(graphiti, live_extract_driver, emb, sc, sm, mc, mm,
                                  cc, cmodel, q="How has AWS Backup vault lock changed over time?",
                                  mode_override=None, vendor=None, settings=s)
        assert env["mode"] == "timeline"
        assert env["routing"]["via"] == "heuristic"
        assert "http" not in env["answer"]                    # timeline render authored no URL
        for c in env["citations"]:
            assert c["sources"], "cited fact must resolve to >=1 source"
    finally:
        await graphiti.close()
        await sc.close()
        await mc.close()
        if cc is not None:
            await cc.close()
```

Run: `uv run --extra dev pytest -m live tests/integration/test_answer_router_live.py -q`
Expected: PASS.

- [ ] **Step 3: Controller demonstration**

Call `answer_router` (or the running API) on `backup-docs` with several intent-varied queries (a specific-fact one → local, a cross-vendor one → global, a broad "what should I consider" → drift, a "how has X changed" → timeline). Write `docs/superpowers/answer-router-report.md` showing each query, its `routing` block (chosen mode + via), the cited answer, and that citations resolve to real URLs with no LLM-authored URL.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_answer_router_live.py docs/superpowers/answer-router-report.md
git commit -m "test(router): live smoke + demonstration report"
```

---

## Notes for the implementer

- `answer_router`/`classify`/`_dispatch` reference the mode functions via module aliases (`synth_mod.answer_local`, etc.), so tests monkeypatch the source module attrs (`synthesize.answer_local`, …) and the router sees them. The endpoint calls `router_mod.answer_router` (module-attr seam) so app tests stub the whole router.
- Decision #2: the router authors no prose/URL. `_render_timeline` is deterministic; `sources` come from the mode's already-resolved citations.
- The cheap classifier makes at most one call, only for queries the heuristics miss; `cheap_client=None` (no key) ⇒ heuristics-only, default drift.
- Test fakes: one statement per line, no semicolons (E702); imports at file top (E402). LLM fakes expose `.chat.completions.create`.
- `/answer` drops its old `k` param (each mode uses its own settings knobs); remember to remove the `/answer` `k`-validation cases from the parametrized app tests (Task 5).
