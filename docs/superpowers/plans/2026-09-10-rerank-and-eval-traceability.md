# Reranked Community Selection + Eval Traceability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Community relevance is scored by a calibrated cross-encoder instead of an LLM's improvised 0-10 rating, and the eval stops being a black box.

**Architecture:** Part 1 makes the eval observable and honest about failures — it ships first so Part 2's own validation runs are visible. Part 2 puts Voyage `rerank-3` inside `shortlist_communities` (so DRIFT inherits it), cuts to top-N above a score floor, and reduces `map_report` to extraction only.

**Tech Stack:** Python 3.12, `uv`, pytest, `httpx`, Voyage AI Cohere-compatible `/v1/rerank`.

**Spec:** `docs/superpowers/specs/2026-09-10-rerank-and-eval-traceability-design.md`

## Global Constraints

- **`rerank()` returns `None` when it could not score.** `None` means "could not score", NEVER "nothing is relevant". Four separate defects in this codebase came from a failure coerced into a legitimate-looking value; do not add a fifth.
- **A degraded answer carries a reader-visible disclaimer**, added AFTER `_finalize_answer` (or marker-stripping and whitespace repair would mangle it), containing **no `[N]` markers and no URL** so it cannot read as cited content or violate design decision #2.
- The disclaimer is added only when a real answer is produced — **never appended to a refusal**.
- Only `rerank-unavailable` is wired to a disclaimer in this slice. `drift.py`'s `no-primer-communities` is deliberately NOT disclaimed — the reader gets a valid local answer, not a less accurate one.
- **Defaults are measured, not guessed.** `rerank_top_n` and `rerank_score_floor` come from Task 5's measured distributions. Picking a floor from one question would repeat the mistake that produced the summary rule this project just had to undo.
- The API key lives ONLY in the untracked `.env`. Never print it, never commit it, never echo it into a log.
- CI gate, clean at every commit: `uv run ruff check src tests` (lints tests too: E702, E402), `uv run mypy src`, `uv run --extra dev pytest -m "not live"`.
- **On long commands:** the Bash tool auto-backgrounds anything over 120 seconds — a harness limit, not yours. Do NOT wait for a completion notification; poll the command's output file until you see the final summary line.

## Verified current state (do not re-derive)

```python
# src/answer_api/eval_router.py:171 — the swallow that must become visible
        except Exception:
            logger.warning("eval question failed: %s", q["question"], exc_info=True)
            rec = {"question": q["question"], "intent": q["intent"], "chosen": "error",
                   "routing_hit": False, "grounding_hit": None, "faithfulness": None}

# src/answer_api/global_search.py — cosine only, no reranking
async def shortlist_communities(driver, embedder, q: str, *, level: int, k: int,
                                group_id: str, rating_boost: float = 0.1) -> list[CommunityHit]:

# called from exactly TWO places: global_search.py and drift.py:64
```

Live facts: `global_default_level = 1`, which holds **11** retrievable communities, and
`global_shortlist_k = 10` — the cosine shortlist admits 10 of 11 and is not selecting.

---

# Part 1 — Eval traceability

### Task 1: Make the eval observable and honest about failures

**Files:**
- Modify: `src/answer_api/eval_router.py` (`run_eval` ~line 156, `format_report` ~line 179)
- Modify: `src/answer_api/router_eval.py` (`aggregate`)
- Test: `tests/unit/test_eval_traceability.py`

**Interfaces:**
- Produces: per-question records gain `"failed": bool`; `aggregate` returns `questions_failed: int` and computes `routing_accuracy` over questions that ran.

**Why:** a 2h09m run printed nothing, so a hung run and a working run were
indistinguishable. Worse, an earlier run swallowed **14 `APIConnectionError`s** into
`routing_hit: False` records — a plausible-looking report whose routing accuracy was
depressed purely by network failures.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_eval_traceability.py`:

```python
"""A failed eval question must be visible and must not be scored as a routing miss."""
from answer_api.eval_router import format_report
from answer_api.router_eval import aggregate


def _rec(intent, chosen, hit, *, failed=False, faith=5):
    return {"question": f"q-{intent}-{chosen}", "intent": intent, "chosen": chosen,
            "routing_hit": hit, "grounding_hit": True, "faithfulness": faith,
            "failed": failed}


def test_counts_failed_questions():
    s = aggregate([_rec("local", "local", True),
                   _rec("global", "error", False, failed=True, faith=None)])
    assert s["questions_failed"] == 1


def test_routing_accuracy_excludes_failed_questions():
    """A network error is not a routing miss. Scoring it as one silently depresses
    the metric -- 14 connection errors did exactly that on 2026-09-10."""
    s = aggregate([_rec("local", "local", True),
                   _rec("global", "error", False, failed=True, faith=None)])
    assert s["routing_accuracy"] == 1.0


def test_routing_accuracy_is_zero_when_every_question_failed():
    s = aggregate([_rec("local", "error", False, failed=True, faith=None)])
    assert s["questions_failed"] == 1
    assert s["routing_accuracy"] == 0.0


def test_a_genuine_miss_still_counts_against_routing():
    s = aggregate([_rec("local", "timeline", False), _rec("global", "global", True)])
    assert s["questions_failed"] == 0
    assert s["routing_accuracy"] == 0.5


def test_report_shows_the_failed_count():
    s = aggregate([_rec("local", "local", True),
                   _rec("global", "error", False, failed=True, faith=None)])
    assert "failed: 1" in format_report(s)


def test_failed_row_renders_as_dash_not_false():
    s = aggregate([_rec("global", "error", False, failed=True, faith=None)])
    row = [ln for ln in format_report(s).splitlines() if "q-global-error" in ln][0]
    assert "| - |" in row


def test_records_without_a_failed_key_are_treated_as_ran():
    """Backward compatibility: older records have no `failed` key."""
    s = aggregate([{"question": "q", "intent": "local", "chosen": "local",
                    "routing_hit": True, "grounding_hit": True, "faithfulness": 5}])
    assert s["questions_failed"] == 0 and s["routing_accuracy"] == 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_eval_traceability.py -q`
Expected: FAIL — `KeyError: 'questions_failed'`.

- [ ] **Step 3: Mark failures in `run_eval`**

In `src/answer_api/eval_router.py`, add `"failed": False` to the success record and
`"failed": True` to the except record:

```python
            rec: dict = {"question": q["question"], "intent": q["intent"], "chosen": chosen,
                         "routing_hit": routing_hit(chosen, q["expected_modes"]),
                         "grounding_hit": ghit, "faithfulness": faith, "failed": False}
```

```python
        except Exception:
            logger.warning("eval question failed: %s", q["question"], exc_info=True)
            rec = {"question": q["question"], "intent": q["intent"], "chosen": "error",
                   "routing_hit": False, "grounding_hit": None, "faithfulness": None,
                   "failed": True}
```

- [ ] **Step 4: Count them and fix the routing denominator**

In `src/answer_api/router_eval.py`'s `aggregate`, mirror the existing `grounded` filter
pattern:

```python
    ran = [r for r in per_question if not r.get("failed")]
```

Compute `routing_accuracy` over `ran` instead of `per_question`, keep `n` as the total, and
add to the returned dict:

```python
        "questions_failed": n - len(ran),
```

`routing_by_intent` should also be built from `ran`, so a failed question does not drag its
intent's accuracy down. Leave `grounding_precision` and the faithfulness means as they are —
a failed question already contributes `None` to both and is filtered out by the existing
`is not None` checks.

- [ ] **Step 5: Surface it in the report**

In `format_report`, change the questions line:

```python
             f"Questions: {summary['n']} (failed: {summary['questions_failed']})\n",
```

and render a failed row's routing cell as `-`:

```python
    for r in summary["per_question"]:
        faith_cell = r["faithfulness"] if r["faithfulness"] is not None else "-"
        routing_cell = "-" if r.get("failed") else r["routing_hit"]
        lines.append(f"| {r['intent']} | {r['chosen']} | {routing_cell} | "
                     f"{r['grounding_hit']} | {faith_cell} | {r['question']} |")
```

- [ ] **Step 6: Add per-question progress**

In `run_eval`, time each question and print a line as it completes, so a long run is
distinguishable from a hung one:

```python
async def run_eval(clients, questions, settings) -> dict:
    per_question: list[dict] = []
    total = len(questions)
    for i, q in enumerate(questions, 1):
        started = time.monotonic()
        try:
            ...
        except Exception:
            ...
        elapsed = time.monotonic() - started
        # A 2h09m run printed nothing until it finished, so a hung run and a working
        # one looked identical. One line per question makes progress visible.
        print(f"[{i}/{total}] {rec['intent']:8} -> {rec['chosen']:8} "
              f"routing={'-' if rec.get('failed') else rec['routing_hit']} "
              f"grounding={rec['grounding_hit']} "
              f"faith={rec['faithfulness'] if rec['faithfulness'] is not None else '-'} "
              f"{elapsed:.0f}s", flush=True)
        per_question.append(rec)
    return aggregate(per_question)
```

Add `import time` at the top if absent. Use `print(..., flush=True)`, not `logger.info` —
this is progress output for a human watching a CLI, and it must not be swallowed by log
configuration.

- [ ] **Step 7: Run the tests**

Run: `uv run --extra dev pytest tests/unit/test_eval_traceability.py tests/unit/test_eval_router_harness.py tests/unit/test_router_eval.py tests/unit/test_faithfulness_unscored.py -q`
Expected: PASS. If an existing test asserts on `routing_accuracy` with a failed question in
its fixture, its expected value legitimately changes — update it and say so in your report.

- [ ] **Step 8: Full suite, lint, type-check, commit**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live" -q`
Expected: all pass (~11 min; poll the output file).

```bash
git add src tests
git commit -F - <<'EOF'
feat(eval): per-question progress, and failures that are visible

A 2h09m run printed nothing, so a hung run and a working one were
indistinguishable. Separately, an earlier run swallowed 14 APIConnectionErrors
into routing_hit: False records -- a plausible-looking report whose routing
accuracy was depressed purely by network failures.

Failures are now counted, rendered as "-", and excluded from the routing
denominator. A network error is not a routing miss.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

# Part 2 — Reranked community selection

### Task 2: The Voyage rerank client

**Files:**
- Create: `src/answer_api/rerank.py`
- Modify: `src/graph_extract/config.py`
- Test: `tests/unit/test_rerank_client.py`

**Interfaces:**
- Produces:
  - `async def rerank(query: str, documents: list[str], *, top_k: int, settings) -> list[tuple[int, float]] | None`
  - `ExtractSettings.rerank_base_url`, `rerank_model`, `rerank_api_key` (all `str = ""`), `rerank_candidates: int = 50`, `rerank_top_n: int = 4`, `rerank_score_floor: float = 0.45`
  - `rerank_configured(settings) -> bool`

The `rerank_top_n` / `rerank_score_floor` defaults above are **provisional** and are set
from measurement in Task 5.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_rerank_client.py`:

```python
"""The reranker returns None when it CANNOT score. None never means "nothing is
relevant" -- four defects in this codebase came from a failure coerced into a
legitimate-looking value."""
import httpx
import pytest

from answer_api.rerank import rerank, rerank_configured
from graph_extract.config import ExtractSettings


def _settings(**kw):
    base = dict(_env_file=None, rerank_base_url="https://rr.example/v1",
                rerank_model="rerank-3", rerank_api_key="k")
    base.update(kw)
    return ExtractSettings(**base)


def _transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_parses_index_and_score_pairs():
    def handler(request):
        return httpx.Response(200, json={"data": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.4}]})
    got = await rerank("q", ["a", "b", "c"], top_k=3, settings=_settings(),
                       transport=_transport(handler))
    assert got == [(2, 0.9), (0, 0.4)]


@pytest.mark.asyncio
async def test_none_on_non_200():
    def handler(request):
        return httpx.Response(500, text="boom")
    assert await rerank("q", ["a"], top_k=1, settings=_settings(),
                        transport=_transport(handler)) is None


@pytest.mark.asyncio
async def test_none_on_transport_error():
    def handler(request):
        raise httpx.ConnectError("no route")
    assert await rerank("q", ["a"], top_k=1, settings=_settings(),
                        transport=_transport(handler)) is None


@pytest.mark.asyncio
async def test_none_on_malformed_payload():
    def handler(request):
        return httpx.Response(200, json={"unexpected": True})
    assert await rerank("q", ["a"], top_k=1, settings=_settings(),
                        transport=_transport(handler)) is None


@pytest.mark.asyncio
async def test_out_of_range_indices_are_dropped_not_crashed_on():
    def handler(request):
        return httpx.Response(200, json={"data": [
            {"index": 99, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5}]})
    got = await rerank("q", ["a", "b"], top_k=2, settings=_settings(),
                       transport=_transport(handler))
    assert got == [(1, 0.5)]


@pytest.mark.asyncio
async def test_empty_documents_returns_empty_without_calling_the_api():
    called = []

    def handler(request):
        called.append(1)
        return httpx.Response(200, json={"data": []})
    got = await rerank("q", [], top_k=3, settings=_settings(),
                       transport=_transport(handler))
    assert got == [] and called == []


@pytest.mark.asyncio
async def test_sends_model_query_and_documents():
    seen = {}

    def handler(request):
        import json as _json
        seen.update(_json.loads(request.content))
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"data": []})
    await rerank("what about encryption?", ["d1", "d2"], top_k=2,
                 settings=_settings(), transport=_transport(handler))
    assert seen["model"] == "rerank-3"
    assert seen["query"] == "what about encryption?"
    assert seen["documents"] == ["d1", "d2"]


def test_rerank_configured_reports_missing_config():
    assert rerank_configured(_settings()) is True
    assert rerank_configured(ExtractSettings(_env_file=None)) is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_rerank_client.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'answer_api.rerank'`.

- [ ] **Step 3: Add the settings**

In `src/graph_extract/config.py`, near the other LLM tiers:

```python
    # Cross-encoder reranker (Voyage AI, Cohere-compatible /v1/rerank). Scores
    # community relevance for the global path, replacing the LLM's improvised
    # 0-10 rating. rerank_top_n / rerank_score_floor are set from measured score
    # distributions -- see the 2026-09-10 spec, not guessed.
    rerank_base_url: str = ""
    rerank_model: str = ""
    rerank_api_key: str = ""
    rerank_candidates: int = 50    # cosine pre-cut before the API call
    rerank_top_n: int = 4          # max communities reaching extraction
    rerank_score_floor: float = 0.45
```

- [ ] **Step 4: Write the client**

Create `src/answer_api/rerank.py`:

```python
"""Cross-encoder reranking for community selection (Voyage AI, Cohere-compatible).

The global map step used to ask an LLM for `"relevance": 0-10 (how useful)` with no
anchors and no definition of relevance. Measured 2026-09-10, it inverted: a community
about KMS key policies scored 3 on an encryption question while an Azure BCDR platform
overview with no encryption facts scored 6. A cross-encoder is trained and calibrated
for exactly this comparison; an LLM rating 0-10 is improvising a scale.
"""
from __future__ import annotations

import logging

import httpx

from graph_extract.config import ExtractSettings

logger = logging.getLogger(__name__)


def rerank_configured(settings: ExtractSettings) -> bool:
    return bool(settings.rerank_base_url and settings.rerank_model)


async def rerank(query: str, documents: list[str], *, top_k: int,
                 settings: ExtractSettings,
                 transport: httpx.AsyncBaseTransport | None = None,
                 ) -> list[tuple[int, float]] | None:
    """Score `documents` against `query`, best first.

    Returns None when relevance COULD NOT BE SCORED -- a non-200, a transport error,
    or a payload we cannot read. None is not "nothing is relevant": the caller must
    degrade visibly rather than treat an outage as an empty result.
    """
    if not documents:
        return []
    payload = {"model": settings.rerank_model, "query": query,
               "documents": documents, "top_k": min(top_k, len(documents))}
    url = settings.rerank_base_url.rstrip("/") + "/rerank"
    try:
        async with httpx.AsyncClient(timeout=30.0, transport=transport) as client:
            resp = await client.post(
                url, json=payload,
                headers={"Authorization": f"Bearer {settings.rerank_api_key}"})
    except httpx.HTTPError as exc:
        logger.warning("rerank transport error (%s); caller must degrade visibly", exc)
        return None
    if resp.status_code != 200:
        logger.warning("rerank returned HTTP %s; caller must degrade visibly",
                       resp.status_code)
        return None
    try:
        rows = resp.json()["data"]
        out = [(int(r["index"]), float(r["relevance_score"])) for r in rows]
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("rerank payload unreadable (%s); caller must degrade visibly", exc)
        return None
    return [(i, s) for i, s in out if 0 <= i < len(documents)]
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run --extra dev pytest tests/unit/test_rerank_client.py -q`
Expected: PASS (8 passed).

- [ ] **Step 6: Lint, type-check, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src tests
git commit -F - <<'EOF'
feat(answers): Voyage cross-encoder rerank client

Returns None when relevance could not be scored -- never an empty result, which
would read as "nothing is relevant" and silently produce a refusal.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

### Task 3: Rerank inside the shortlist; map extracts only

**Files:**
- Modify: `src/answer_api/global_search.py`
- Test: `tests/unit/test_shortlist_rerank.py`

**Interfaces:**
- Consumes: `rerank`, `rerank_configured` (Task 2).
- Produces:
  - `@dataclass class RerankStats: degraded: str | None = None`
  - `shortlist_communities(driver, embedder, q, *, level, k, group_id, rating_boost=0.1, settings=None, stats=None) -> list[CommunityHit]`
  - `CommunityHit.similarity` unchanged; `MapResult.relevance` becomes a **float** carrying the rerank score.

**Why reranking lives in `shortlist_communities`:** DRIFT calls it (`drift.py:64`) and must
inherit reranking. The degraded signal travels on an optional `stats` out-param, matching
the `ReportStats` precedent, so neither of the two call sites changes shape.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_shortlist_rerank.py`:

```python
"""Reranking replaces the LLM's relevance score and supplies the selection the
cosine shortlist was never doing (level 1 holds 11 communities and k was 10)."""
import pytest

from answer_api.global_search import RerankStats, _apply_rerank


def _hit(cid, title):
    from answer_api.global_search import CommunityHit
    return CommunityHit(community_id=cid, title=title, summary=f"{title} summary",
                        level=1, rating=5.0, cited_fact_uuids=["f1"],
                        full_report="[]", similarity=0.5)


HITS = [_hit("a", "Alpha"), _hit("b", "Bravo"), _hit("c", "Charlie")]


def test_keeps_top_n_above_the_floor_in_rerank_order():
    scored = [(2, 0.9), (0, 0.6), (1, 0.2)]
    st = RerankStats()
    out = _apply_rerank(HITS, scored, top_n=2, floor=0.5, stats=st)
    assert [h.community_id for h in out] == ["c", "a"]
    assert st.degraded is None


def test_floor_can_reject_everything():
    """An off-topic question must be able to yield zero survivors, so the caller
    takes the refusal path instead of synthesising junk."""
    out = _apply_rerank(HITS, [(0, 0.1), (1, 0.05)], top_n=3, floor=0.5,
                        stats=RerankStats())
    assert out == []


def test_top_n_caps_even_when_all_clear_the_floor():
    out = _apply_rerank(HITS, [(0, 0.9), (1, 0.8), (2, 0.7)], top_n=2, floor=0.5,
                        stats=RerankStats())
    assert len(out) == 2


def test_relevance_carries_the_rerank_score():
    out = _apply_rerank(HITS, [(1, 0.77)], top_n=3, floor=0.5, stats=RerankStats())
    assert out[0].relevance == pytest.approx(0.77)


def test_none_degrades_to_cosine_order_and_marks_it():
    """None means the reranker could not score -- NOT that nothing is relevant."""
    st = RerankStats()
    out = _apply_rerank(HITS, None, top_n=2, floor=0.5, stats=st)
    assert [h.community_id for h in out] == ["a", "b"]
    assert st.degraded == "rerank-unavailable"


def test_degraded_fallback_still_respects_top_n():
    st = RerankStats()
    out = _apply_rerank(HITS, None, top_n=1, floor=0.5, stats=st)
    assert len(out) == 1 and st.degraded == "rerank-unavailable"


def test_empty_scored_list_is_not_degradation():
    """[] means "scored, nothing cleared the bar". That is a real verdict."""
    st = RerankStats()
    out = _apply_rerank(HITS, [], top_n=2, floor=0.5, stats=st)
    assert out == [] and st.degraded is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_shortlist_rerank.py -q`
Expected: FAIL — `ImportError: cannot import name 'RerankStats'`.

- [ ] **Step 3: Add `RerankStats`, `_apply_rerank`, and give `CommunityHit` a relevance field**

In `src/answer_api/global_search.py`:

```python
@dataclass
class RerankStats:
    """Out-param so the two shortlist_communities call sites keep their shape."""
    degraded: str | None = None


def _apply_rerank(hits: list[CommunityHit],
                  scored: list[tuple[int, float]] | None,
                  *, top_n: int, floor: float,
                  stats: RerankStats) -> list[CommunityHit]:
    """Cut the candidate list using rerank scores.

    `scored is None` means the reranker COULD NOT SCORE: fall back to cosine order,
    capped at top_n, and mark the result degraded so the caller can tell the reader.
    An empty list is different -- it means the reranker scored and nothing cleared
    the floor, which is a real verdict and must reach the refusal path.
    """
    if scored is None:
        stats.degraded = "rerank-unavailable"
        return hits[:top_n]
    out: list[CommunityHit] = []
    for idx, score in scored[:top_n]:
        if score < floor:
            continue
        # `replace`, not mutation: a caller may hold the same candidate list, and
        # silently rewriting its objects is the kind of surprise that is very hard
        # to trace later. Same reason report.py's _with_findings uses it.
        out.append(replace(hits[idx], relevance=score))
    return out
```

Add `relevance: float = 0.0` to `CommunityHit` so the score travels with the hit. It is a
plain (non-frozen) `@dataclass`, so a default is all that is needed. Add `replace` to the
existing `from dataclasses import ...` line.

- [ ] **Step 4: Wire it into `shortlist_communities`**

```python
async def shortlist_communities(driver, embedder, q: str, *, level: int, k: int,
                                group_id: str, rating_boost: float = 0.1,
                                settings=None,
                                stats: RerankStats | None = None) -> list[CommunityHit]:
```

After the existing cosine ranking, take `settings.rerank_candidates` candidates instead of
`k`, then rerank and cut:

```python
    candidates = _rank_hits(query_vec, rows, k=(settings.rerank_candidates if settings
                                                else k), rating_boost=rating_boost)
    if settings is None or not rerank_configured(settings):
        return candidates[:k]
    st = stats if stats is not None else RerankStats()
    docs = [f"{h.title}: {h.summary}" for h in candidates]
    scored = await rerank(q, docs, top_k=settings.rerank_top_n, settings=settings)
    return _apply_rerank(candidates, scored, top_n=settings.rerank_top_n,
                         floor=settings.rerank_score_floor, stats=st)
```

**Thread `settings` to both call sites — neither has it today.** This is not optional
plumbing: skipping the DRIFT half is exactly how the previous slice shipped two Criticals,
by leaving the non-obvious path out. Verified signatures:

- `global_search(driver, embedder, map_client, map_model, synth_client, synth_model, *, q, level, k, group_id, relevance_min)` (`global_search.py:168`) — **add `settings`**, drop `relevance_min`.
- `drift_search(graphiti, driver, embedder, synth_client, synth_model, *, q, level, iterations, primer_k, max_followups, followup_k, group_id)` (`drift.py:173`) — **add `settings`** and pass it to `_primer`.
- `_primer(embedder, synth_client, synth_model, driver, *, q, level, k, max_followups, group_id)` (`drift.py:62`) — **add `settings`**, pass to `shortlist_communities`.

The four outer callers already HAVE settings, so this is threading only:

- `router.py:130` (`global_search`) and `router.py:135` (`drift_search`) — inside `answer_router`, which takes `settings` (`router.py:146`). Pass `settings=settings`; delete the `relevance_min=settings.global_map_relevance_min` argument.
- `app.py:147` (`global_search`) and `app.py:162` (`drift_search`) — use `st.settings`. Pass `settings=st.settings`; delete the `relevance_min=` argument.

`drift.py` needs no change beyond threading `settings` through those two functions —
its ranking, primer prompt and follow-up logic are untouched.

- [ ] **Step 5: Strip relevance from the map step**

Rewrite `_MAP_PROMPT` to drop the relevance field entirely:

```python
_MAP_PROMPT = (
    "Extract from this COMMUNITY report only what answers the QUESTION. Using ONLY "
    "the report, respond with JSON: {{\"key_points\": [short strings that answer the "
    "question], \"fact_ids\": [the fact uuids from the report that support those "
    "points]}}. Include ONLY points that bear on the question, and ONLY fact_ids that "
    "support the points you listed. fact_ids MUST be uuids that appear in the report. "
    "Do NOT write URLs.\n\n"
    "QUESTION: {q}\n\nCOMMUNITY \"{title}\": {summary}\nFINDINGS: {full_report}"
)
```

In `map_report`, delete the relevance parse and the `relevance < relevance_min` gate, drop
the `relevance_min` parameter, and populate `MapResult.relevance` from `hit.relevance` (the
rerank score). Change `MapResult.relevance` to `float`. Remove `global_map_relevance_min`
from `config.py` and from the `global_search(...)` call chain.

- [ ] **Step 6: Run the tests**

Run: `uv run --extra dev pytest tests/unit/test_shortlist_rerank.py tests/unit/test_global_map.py tests/unit/test_global_rank.py tests/unit/test_router_dispatch.py -q`
Expected: PASS. Tests referencing `relevance_min` or an integer `MapResult.relevance` must
be updated to the new contract — update them honestly, do not weaken assertions.

- [ ] **Step 7: Lint, type-check, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src tests
git commit -F - <<'EOF'
feat(global): rerank the community shortlist; map extracts only

Reranking lives in shortlist_communities so DRIFT inherits it. The degraded
signal travels on an optional RerankStats out-param, so neither call site
changes shape.

_MAP_PROMPT loses its relevance field: the rubric problem disappears rather than
needing a rubric written for it. MapResult.relevance now carries the calibrated
rerank score.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

### Task 4: Degradation the reader can see

**Files:**
- Modify: `src/answer_api/global_search.py`
- Test: `tests/unit/test_degraded_disclaimer.py`

**Interfaces:**
- Consumes: `RerankStats` (Task 3).
- Produces: `_DEGRADED_DISCLAIMERS: dict[str, str]`; `_with_disclaimer(answer: str, reason: str | None) -> str`; `global_search(...)` returns `degraded` in its envelope when set.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_degraded_disclaimer.py`:

```python
"""A degraded answer must tell the READER, not just the envelope."""
import re

from answer_api.global_search import _REFUSAL, _with_disclaimer


def test_adds_a_disclaimer_when_degraded():
    out = _with_disclaimer("Both vendors encrypt at rest [1].", "rerank-unavailable")
    assert "Both vendors encrypt at rest [1]." in out
    assert out != "Both vendors encrypt at rest [1]."
    assert "relevance ranking was unavailable" in out.lower()


def test_no_disclaimer_when_not_degraded():
    text = "Both vendors encrypt at rest [1]."
    assert _with_disclaimer(text, None) == text


def test_disclaimer_carries_no_citation_markers():
    """It must not read as cited content -- design decision #2."""
    out = _with_disclaimer("Answer [1].", "rerank-unavailable")
    added = out.replace("Answer [1].", "")
    assert re.search(r"\[\d+\]", added) is None
    assert "http" not in added.lower()


def test_a_refusal_is_never_disclaimed():
    """There is no answer to qualify."""
    assert _with_disclaimer(_REFUSAL, "rerank-unavailable") == _REFUSAL


def test_an_unknown_reason_adds_nothing():
    text = "Answer [1]."
    assert _with_disclaimer(text, "some-future-reason") == text


def test_empty_answer_is_not_disclaimed():
    assert _with_disclaimer("", "rerank-unavailable") == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_degraded_disclaimer.py -q`
Expected: FAIL — `ImportError: cannot import name '_with_disclaimer'`.

- [ ] **Step 3: Implement**

In `src/answer_api/global_search.py`:

```python
# Reader-visible degradation notices. The `degraded` envelope field is
# machine-readable only -- nothing shows it to the person reading the answer.
# Keyed by reason so other degraded modes can adopt it; only rerank-unavailable
# is wired up. drift.py's "no-primer-communities" is deliberately NOT here: the
# reader gets a valid local answer, not a less accurate one.
_DEGRADED_DISCLAIMERS = {
    "rerank-unavailable": (
        "Note: relevance ranking was unavailable for this answer, so the sources it "
        "draws on may be less relevant than usual. Please verify against the cited "
        "sources."),
}


def _with_disclaimer(answer: str, reason: str | None) -> str:
    """Prepend a reader-visible notice to a degraded answer.

    Called AFTER _finalize_answer: marker stripping and whitespace repair must not
    treat this text as answer prose. The notice carries no [N] markers and no URL,
    so it cannot be mistaken for cited content.
    """
    note = _DEGRADED_DISCLAIMERS.get(reason or "")
    if not note or not answer.strip() or answer.strip() == _REFUSAL:
        return answer
    return f"{note}\n\n{answer}"
```

- [ ] **Step 4: Wire it into `global_search`**

Create a `RerankStats()` before calling `shortlist_communities`, pass it, and apply the
disclaimer to the finalized answer:

```python
    stats = RerankStats()
    hits = await shortlist_communities(driver, embedder, q, level=level, k=k,
                                       group_id=group_id, settings=settings, stats=stats)
```

and at the return, after `_finalize_answer` has run:

```python
    answer = _with_disclaimer(answer, stats.degraded)
    ...
    return {"query": q, "answer": answer, "citations": citations,
            "degraded": stats.degraded,
            "communities_used": [...]}
```

Add `"degraded": stats.degraded` to the two early-return refusal envelopes as well, so the
field is always present, then confirm the refusal text itself is unchanged.

- [ ] **Step 5: Run the tests**

Run: `uv run --extra dev pytest tests/unit/test_degraded_disclaimer.py tests/unit/test_answer_api_app.py tests/unit/test_router_dispatch.py -q`
Expected: PASS.

- [ ] **Step 6: Full suite, lint, type-check, commit**

Run: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live" -q`
Expected: all pass (~11 min; poll the output file).

```bash
git add src tests
git commit -F - <<'EOF'
feat(global): a degraded answer tells the reader, not just the envelope

The `degraded` field is machine-readable only. A rerank outage now prepends a
notice with no [N] markers and no URL, added after _finalize_answer so marker
stripping cannot mangle it, and never appended to a refusal.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

### Task 5: Set the thresholds from measurement

**Files:**
- Create: `docs/superpowers/rerank-threshold-measurement.md`
- Modify: `src/graph_extract/config.py` (final `rerank_top_n` / `rerank_score_floor`)

**Interfaces:**
- Consumes: everything above.

**Why:** Task 2's defaults are provisional. Choosing a floor from one question would repeat
the mistake that produced the summary rule this project just had to undo.

- [ ] **Step 1: Measure the score distribution**

`src/answer_api/router_golden.json` is a list of 29 dicts with keys `question`, `intent`,
`expected_modes`, `expected_article_ids`; exactly 10 have `intent` in `("global", "drift")`.

Run this from the repo root (it writes nothing to the repo):

```
uv run --extra dev python -c "
import asyncio, json, pathlib
from neo4j import AsyncGraphDatabase
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_embedder
from answer_api.rerank import rerank
async def m():
    s = get_extract_settings()
    qs = [q for q in json.loads(pathlib.Path('src/answer_api/router_golden.json').read_text())
          if q['intent'] in ('global','drift')]
    d = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    async with d.session() as sess:
        r = await sess.run('MATCH (c:Community {group_id:\$g, level:\$l}) '
                           'WHERE c.embedding IS NOT NULL '
                           'RETURN c.title AS t, coalesce(c.summary,\'\') AS s',
                           g=s.group_id, l=s.global_default_level)
        rows = [dict(x) async for x in r]
    docs = [f\"{x['t']}: {x['s']}\" for x in rows]
    print('communities at level', s.global_default_level, '=', len(rows))
    for q in qs:
        scored = await rerank(q['question'], docs, top_k=len(docs), settings=s)
        print()
        print('Q:', q['question'])
        if scored is None:
            print('  RERANK FAILED'); continue
        for rank,(i,sc) in enumerate(scored,1):
            print(f'  {rank:2}. {sc:.4f}  {rows[i][\"t\"][:64]}')
    await d.close(); await embedder_close(s)
async def embedder_close(s):
    e = build_embedder(s)
    await e.client.close()
asyncio.run(m())
"
```

Record the full output; it is the raw material for Step 2.

- [ ] **Step 2: Record the measurement**

Create `docs/superpowers/rerank-threshold-measurement.md` with a table per question and,
across all ten:

- the score of the highest-ranked community per question (how strong is a good match);
- the gap between rank 1 and rank 5;
- how many communities per question would survive at candidate floors 0.40 / 0.45 / 0.50 / 0.55;
- how many questions would be left with **zero** survivors at each floor.

- [ ] **Step 3: Choose the defaults and justify them**

Pick `rerank_top_n` and `rerank_score_floor` from that table and write one paragraph saying
why, naming the trade-off: a higher floor removes junk but risks refusing an answerable
question. State explicitly how many of the ten questions would refuse at the chosen floor.
If that number is above zero, say whether that is correct behaviour for those specific
questions or a sign the floor is too high.

Update the defaults in `config.py` to the chosen values.

- [ ] **Step 4: Acceptance check on the observed inversion**

Run the two communities from the spec §2.2 against the question *"What should I consider
for backup encryption across cloud providers?"* and confirm "KMS Key Policy Management for
AWS Backup" ranks **above** "Resiliency in Azure: Unified BCDR Platform". Record both
scores in the measurement document.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/rerank-threshold-measurement.md src/graph_extract/config.py
git commit -F - <<'EOF'
docs: set rerank thresholds from measured score distributions

Measured across the ten golden global/DRIFT questions rather than picked from a
single example.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

### Task 6: Re-run the eval and report

**Files:**
- Modify: `docs/superpowers/router-eval-report.md`

- [ ] **Step 1: Confirm the graph is unchanged**

Run the same counts used previously and confirm 385 episodes / 2,173 facts / 41 communities
(37 retrievable). If they differ, the comparison is not like-for-like and must be labelled
as such.

- [ ] **Step 2: Run the eval**

Run: `uv run --extra dev python -m answer_api.eval_router`

Part 1's per-question progress is now active, so this run is observable. It may take over
an hour; poll the output file. Note the elapsed time and whether `questions_failed` is
non-zero — a non-zero count invalidates the comparison and must be reported, not averaged over.

- [ ] **Step 3: Write the interpretive header**

Prepend a `>` blockquote to `docs/superpowers/router-eval-report.md` covering honestly:

- global faithfulness against the **2.33** baseline and grounding against **0.86**;
- `questions_failed`, and whether any question failed;
- how many communities reached extraction per question versus 10 before, and the LLM-call
  saving that represents;
- whether any answer carried the degraded disclaimer;
- **the attribution caveat from spec §2.8**: reranking changes what is *selected*, not
  whether content is invented. If faithfulness moved, check which communities were selected
  before crediting the reranker. Do not let a movement here be attributed to reranking
  without that check;
- if answers got shorter or any question now refuses, report it as the correct outcome of
  narrower selection rather than as a regression.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/router-eval-report.md
git commit -F - <<'EOF'
docs(eval): re-run after reranked community selection

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0174DUVF7CPn91yApHMLiVcv
EOF
```

---

## Verification checklist

1. A rerank failure never reads as "nothing is relevant" (Tasks 2, 3).
2. A degraded answer carries a reader-visible disclaimer with no `[N]` markers, never on a refusal (Task 4).
3. `map_report` no longer scores relevance; `_MAP_PROMPT` has no relevance field (Task 3).
4. Thresholds set from measured distributions, and the measurement is recorded (Task 5).
5. The spec §2.2 examples come out in the corrected order (Task 5 Step 4).
6. The eval is re-run with progress and `questions_failed` visible, reported against 2.33 (Task 6).
7. Full non-live suite, ruff and mypy clean.
