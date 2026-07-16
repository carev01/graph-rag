# Answer Synthesis (`/answer`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `/answer` endpoint that writes a GLM-5.2-synthesized cited prose answer over retrieved facts, where the LLM only emits `[N]` markers and a deterministic resolver expands them to source URLs (design-decision #2).

**Architecture:** `answer_api.answer_local` = search_local → numbered facts → GLM synthesis → deterministic post-process (strip URLs, keep valid markers) → citations from the marker→fact→source map. FastAPI `/answer` wraps it. GLM-5.2 via the judge config through a swappable synthesis seam.

**Tech Stack:** Python 3.12, FastAPI, GLM-5.2 via Ollama (OpenAI-compatible, `AsyncOpenAI`), Neo4j, graphiti-core, pytest + testcontainers, ruff, mypy.

## Global Constraints

- **The LLM NEVER authors a URL** (design-decision #2). The synthesis prompt contains marker+fact text only (no URLs/titles); post-processing strips any URL the model emits and drops invented markers; citations come ONLY from the deterministic marker→fact→source map.
- **Refuse when unsupported:** the prompt makes GLM reply with a fixed refusal if the facts don't cover the question; zero retrieval short-circuits with no LLM call.
- **Deterministic post-processing is pure** (`_finalize_answer`) — unit-testable without any LLM.
- **Synthesis model = GLM-5.2 via `judge_*` config** through a synthesis-named seam (`_synthesis_client_and_model`), swappable later; token usage captured via `instrument()`.
- Reuse `search_local`, `resolve_citations`; no duplicate plumbing. Secrets only in `.env`.
- Tests pass (testcontainer); ruff/mypy clean.

---

## File Structure

- `src/answer_api/synthesize.py` — **new:** `_finalize_answer`, `_synthesis_client_and_model`, `answer_local`, the prompt.
- `src/answer_api/app.py` — **modify:** lifespan builds+closes the synth client; `GET /answer`.
- Tests: `tests/unit/test_finalize_answer.py`, `tests/integration/test_answer_local.py`, `tests/unit/test_answer_api_app.py` (extend for `/answer`).

---

### Task 1: Deterministic post-processing + synthesis client seam

**Files:**
- Create: `src/answer_api/synthesize.py` (the two pure/seam functions + the prompt; `answer_local` lands in Task 2)
- Test: `tests/unit/test_finalize_answer.py`

**Interfaces:**
- `_finalize_answer(raw: str, marker_map: dict[int, dict]) -> tuple[str, list[int]]` — URL-stripped answer + ordered-unique valid cited markers.
- `_synthesis_client_and_model(settings) -> tuple[AsyncOpenAI, str]`.

- [ ] **Step 1: Write failing unit tests** for `_finalize_answer` (no LLM):

```python
from answer_api.synthesize import _finalize_answer

MM = {1: {"fact_uuid": "f1"}, 2: {"fact_uuid": "f2"}}

def test_strips_url():
    ans, cited = _finalize_answer("See https://evil/x for details [1].", MM)
    assert "http" not in ans and cited == [1]

def test_keeps_valid_drops_invented_markers():
    ans, cited = _finalize_answer("Immutable [1], and cross-region [2], and made-up [9].", MM)
    assert cited == [1, 2]        # 9 not in marker_map -> dropped

def test_ordered_unique():
    _, cited = _finalize_answer("[2] then [1] then [2] again.", MM)
    assert cited == [2, 1]        # first-seen order, deduped

def test_no_markers():
    ans, cited = _finalize_answer("I don't have enough information.", MM)
    assert cited == []
```

- [ ] **Step 2: Run, confirm fail** (module missing). `uv run --extra dev pytest tests/unit/test_finalize_answer.py -v`.

- [ ] **Step 3: Implement `synthesize.py`'s pure pieces:**

```python
from __future__ import annotations

import re

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.usage import instrument

_URL_RE = re.compile(r"https?://\S+")
_MARKER_RE = re.compile(r"\[(\d+)\]")

_PROMPT = (
    "You are answering a question about backup products using ONLY the numbered "
    "facts below. Cite every claim inline with its [N] marker. Do NOT use outside "
    "knowledge. Do NOT write any URL or link. If the facts do not answer the "
    "question, reply exactly: "
    "\"I don't have enough information to answer that from the available sources.\"\n\n"
    "FACTS:\n{facts}\n\nQUESTION: {q}\n\nAnswer:"
)

_REFUSAL = "I don't have enough information to answer that from the available sources."


def _finalize_answer(raw: str, marker_map: dict[int, dict]) -> tuple[str, list[int]]:
    """Deterministic design-decision #2 enforcement: strip any URL the model
    emitted (it must never write one), then keep the ordered-unique [N] markers
    that map to a retrieved fact (drop invented ones)."""
    text = _URL_RE.sub("", raw).strip()
    cited: list[int] = []
    for m in _MARKER_RE.findall(text):
        n = int(m)
        if n in marker_map and n not in cited:
            cited.append(n)
    return text, cited


def _synthesis_client_and_model(settings: ExtractSettings) -> tuple[AsyncOpenAI, str]:
    """Synthesis LLM = GLM-5.2 via the judge_* config (shared endpoint for now;
    point at a dedicated synthesis model later without touching the judge)."""
    if not settings.judge_base_url:
        raise ValueError(
            "No synthesis model configured. Set JUDGE_BASE_URL / JUDGE_MODEL / "
            "JUDGE_API_KEY in .env (synthesis currently uses the GLM-5.2 judge endpoint)."
        )
    client = instrument(AsyncOpenAI(
        api_key=settings.judge_api_key or "not-needed", base_url=settings.judge_base_url))
    return client, settings.judge_model
```

- [ ] **Step 4: Run tests green** + gate on `synthesize.py`. (Note: `pyproject` already lists `src/answer_api`; no packaging change.)

- [ ] **Step 5: Commit.** `git add src/answer_api/synthesize.py tests/unit/test_finalize_answer.py && git commit -m "feat(answer-api): deterministic answer post-processing (strip URLs, valid markers) + synthesis client seam"`

---

### Task 2: `answer_local` synthesis core

**Files:**
- Modify: `src/answer_api/synthesize.py`
- Test: `tests/integration/test_answer_local.py`

**Interfaces:** `async def answer_local(graphiti, driver, synth_client, synth_model, *, q, k=15, vendor=None, group_id) -> dict` → `{"query","answer","citations":[{marker,fact,fact_uuid,sources}],"retrieved","cited"}`.

- [ ] **Step 1: Write failing tests** (real Neo4j `extract_driver` for citation data; a **stub search_local** OR seed the graph and use real search with a stub graphiti; simplest: stub `answer_api.search.search_local` to return canned results, and a **stub synth client** returning canned prose). Cover:
  - synthesis path: stub search returns 2 facts (markers [1],[2] with sources); stub GLM returns `"Immutable [1] and cross-region [2]."`; assert `answer` present, `citations` has 2 entries with the right sources, `cited==2`.
  - **URL guard end-to-end:** stub GLM returns `"See https://evil/x [1]."`; assert no `http` in `answer` and citations only from marker [1].
  - **zero-retrieval short-circuit:** stub search returns `results: []`; a synth client that RAISES if called; assert the fixed refusal answer, empty citations, and the client was never called.

```python
async def test_answer_local_synthesizes_with_citations(monkeypatch):
    import answer_api.search as search_mod
    async def _fake_search(*a, **k):
        return {"query": "q", "count": 2, "results": [
            {"fact": "immutability", "fact_uuid": "f1", "valid_at": None,
             "sources": [{"url": "u1", "title": "t1", "article_id": "a1"}]},
            {"fact": "cross-region copy", "fact_uuid": "f2", "valid_at": None,
             "sources": [{"url": "u2", "title": "t2", "article_id": "a2"}]}]}
    monkeypatch.setattr(search_mod, "search_local", _fake_search)
    class _Msg:  content = "Immutable [1] and cross-region [2]."
    class _Choice: message = _Msg()
    class _Resp: choices = [_Choice()]
    class _Chat:
        class completions:
            @staticmethod
            async def create(**kw): return _Resp()
    class _StubGLM: chat = _Chat()
    from answer_api.synthesize import answer_local
    out = await answer_local(object(), object(), _StubGLM(), "glm", q="q", group_id="g")
    assert out["cited"] == 2 and out["citations"][0]["sources"][0]["article_id"] == "a1"
    assert "http" not in out["answer"]
```
(Model the other two tests on this; the zero-retrieval one uses a synth stub whose `create` raises.)

- [ ] **Step 2: Run, confirm fail** (`answer_local` missing).

- [ ] **Step 3: Implement `answer_local`** in `synthesize.py`:

```python
from answer_api.search import search_local


async def answer_local(graphiti, driver, synth_client, synth_model, *,
                       q, k=15, vendor=None, group_id) -> dict:
    res = await search_local(graphiti, driver, q=q, k=k, vendor=vendor, group_id=group_id)
    results = res["results"]
    if not results:
        return {"query": q, "answer": _REFUSAL, "citations": [],
                "retrieved": 0, "cited": 0}
    marker_map = {i: r for i, r in enumerate(results, 1)}
    facts_block = "\n".join(f"[{i}] {r['fact']}" for i, r in marker_map.items())
    resp = await synth_client.chat.completions.create(
        model=synth_model, temperature=0, max_tokens=800,
        messages=[{"role": "user", "content": _PROMPT.format(facts=facts_block, q=q)}])
    raw = resp.choices[0].message.content or ""
    answer, cited = _finalize_answer(raw, marker_map)
    citations = [{"marker": m, "fact": marker_map[m]["fact"],
                  "fact_uuid": marker_map[m]["fact_uuid"],
                  "sources": marker_map[m]["sources"]} for m in cited]
    return {"query": q, "answer": answer, "citations": citations,
            "retrieved": len(results), "cited": len(cited)}
```

- [ ] **Step 4: Run tests green** + full non-live suite. Gate.

- [ ] **Step 5: Add a `@live` GLM smoke test** (marked `live`, deselected normally): build the real graph + `_synthesis_client_and_model(get_extract_settings())`, call `answer_local(...)` for `"What does AWS Backup Vault Lock require?"` → non-empty answer, ≥1 citation with a resolved source, no `http` in the answer; and an off-corpus question → the refusal. (Controller runs it in T4; author it here.)

- [ ] **Step 6: Commit.** `git add src/answer_api/synthesize.py tests/integration/test_answer_local.py && git commit -m "feat(answer-api): answer_local (GLM-5.2 synthesis + marker citations + refuse-when-unsupported)"`

---

### Task 3: `GET /answer` FastAPI endpoint

**Files:**
- Modify: `src/answer_api/app.py`
- Test: `tests/unit/test_answer_api_app.py`

- [ ] **Step 1: Write failing FastAPI tests** (extend the existing app test file; stub `answer_api.synthesize.answer_local` and the resource builders as the `/search/local` tests do). Assert `/answer?q=x` → 200 with `{query,answer,citations,retrieved,cited}`; missing `q` → 422; `/search/local` + `/health` still 200.

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement.** In `app.py`'s lifespan, after building graphiti+driver, build the synthesis client+model via `synthesize._synthesis_client_and_model(settings)` and store on `app.state.synth_client`/`app.state.synth_model`; in the shutdown loop, also `await app.state.synth_client.close()` (log+swallow, alongside graphiti/driver). Guard construction like the graphiti/driver partial-failure fix (close already-built resources if a later build raises). Add `GET /answer` reading `q: str` (required), `k: int = 15`, `vendor: str | None = None`, calling the module-level `synthesize.answer_local(app.state.graphiti, app.state.driver, app.state.synth_client, app.state.synth_model, q=q, k=k, vendor=vendor, group_id=app.state.settings.group_id)`. Keep `answer_local` injectable (call via `synthesize.answer_local`) so tests stub it.

- [ ] **Step 4: Run tests green** + confirm routes register (`/answer` present) + gate.

- [ ] **Step 5: Commit.** `git add src/answer_api/app.py tests/unit/test_answer_api_app.py && git commit -m "feat(answer-api): GET /answer endpoint (synthesis + citations)"`

---

### Task 4: Groundedness harness + report (controller-run)

**Files:**
- Modify: `src/answer_api/eval_golden.py` (add an `--answer` mode) or a small `answer_api/eval_answer.py`
- Report: `docs/superpowers/answer-synthesis-report.md`

Controller-run live (real GLM-5.2).

- [ ] **Step 1: Add answer-groundedness scoring.** Extend `eval_golden.py` (or a sibling `eval_answer.py`) with an answer mode: for each golden question, call `answer_local(real graph + GLM)`, and score **answer-groundedness** = the answer cites ≥1 fact whose `sources` include an expected article-id (reuse `precision_at_k` over `out["citations"]`'s sources), plus record `cited` count and whether the answer is the refusal. Reuse `golden.precision_at_k` by passing the citations' sources in the `results` shape it expects (or a small adapter).

- [ ] **Step 2: Run the answer harness (live).** `uv run --extra dev python -m answer_api.eval_answer` (or `eval_golden --answer`). Capture: answer-groundedness@k over the 15 questions; how many returned the refusal (should be the genuinely-unanswerable ones, not the adjacent-article ones); and **spot-read 3–4 answers** to confirm they are faithful to the cited facts and contain NO URLs (the design-decision #2 guarantee, observed end-to-end).

- [ ] **Step 3: Write `docs/superpowers/answer-synthesis-report.md`** — the groundedness number vs the `/search/local` citation-precision@10 (0.800) baseline, the refusal behavior, the spot-read (faithful + URL-free), and any honest caveats (GLM verbosity, over/under-citation). Commit.

- [ ] **Step 4: Commit.** `git add src/answer_api/eval_*.py docs/superpowers/answer-synthesis-report.md && git commit -m "docs: answer-synthesis groundedness report (GLM-5.2, URL-free citations)"`

---

## Self-Review Notes

- **Spec coverage:** post-processing + client seam → T1; `answer_local` core → T2; `/answer` endpoint → T3; groundedness harness → T4. All §-components covered.
- **Deps:** T1 → T2 (answer_local uses `_finalize_answer` + the seam); T2 → T3 (endpoint wraps answer_local); T2 → T4. Order T1, T2, T3, T4.
- **Design-decision #2:** enforced in `_finalize_answer` (URL strip + valid-marker-only), tested purely in T1 and end-to-end (stubbed) in T2, observed live in T4. The prompt never shows the LLM a URL.
- **Type consistency:** `_finalize_answer(raw, marker_map)->(str,list[int])`; `answer_local(...group_id=...)->{query,answer,citations,retrieved,cited}`; `_synthesis_client_and_model(settings)->(client,model)`.
- **No-placeholder check:** every code step carries real code; the two `...`-referenced tests (URL-guard, zero-retrieval) explicitly mirror the shown test with the stated stub change.
