# Router Golden-Set Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A labeled golden question set across all four retrieval modes + a harness that scores the `/answer` router's routing accuracy, per-mode citation grounding, and LLM-judge faithfulness, plus a comparative "DRIFT vs local/global on broad questions" pass — the Phase 4 exit criterion.

**Architecture:** Pure scoring (`router_eval.py`: routing hit, judge-score parse, aggregation — reuses `golden.precision_at_k`); a labeled set (`router_golden.json`); an `@live` harness (`eval_router.py`) that runs `answer_router` + a GLM faithfulness judge per question and writes a report. Leaves the existing local-only eval (`golden_questions.json`/`eval_golden.py`) untouched.

**Tech Stack:** Python 3.12, FastAPI `answer_api` router, Neo4j, GLM judge tier, pytest (pure units + a fixture-mocked harness test; the scored run is `@live`).

## Global Constraints

- The new set/harness are ADDITIVE — `golden_questions.json`, `golden.py`, `eval_golden.py`, `eval_answer.py` are unchanged. Reuse `golden.precision_at_k(citations, expected_article_ids)`.
- Three metrics per question: **routing hit** (`env["routing"]["chosen"] ∈ expected_modes`), **grounding** (`precision_at_k(env["citations"], expected_article_ids)`; `None`/skipped when `expected_article_ids == []`), **faithfulness** (GLM judge 0–5 over the cited FACT TEXTS, not URLs).
- **Comparative** only for `intent ∈ {global, drift}`: run `mode=local/global/drift` (via `?mode` override) and report `drift_wins` (drift faithfulness ≥ both others AND drift grounding ≥ both others where grounding is scored).
- Scores are informational (no hard CI threshold). The `@live` run writes `docs/superpowers/router-eval-report.md`. Design #2 holds (no LLM authors a URL).
- Run with `uv`: `uv run --extra dev pytest …`, `uv run ruff check src tests` (CI gate — lints tests; **no semicolons in fakes (E702), imports at file top (E402)**), `uv run mypy src`. Hermetic settings: `ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")`.

---

## File Structure

- **Create** `src/answer_api/router_eval.py` — pure scoring: `routing_hit`, `_parse_judge_score`, `aggregate`.
- **Create** `src/answer_api/router_golden.json` — the labeled 4-intent set.
- **Create** `src/answer_api/eval_router.py` — the `@live` harness (judge, fact-text resolver, `run_eval`, `format_report`, `main`).
- **Tests:** `tests/unit/test_router_eval.py`, `tests/unit/test_router_golden_schema.py`, `tests/unit/test_eval_router_harness.py`, `tests/integration/test_router_eval_live.py`.

---

### Task 1: Pure scoring — `router_eval.py`

**Files:**
- Create: `src/answer_api/router_eval.py`
- Test: `tests/unit/test_router_eval.py`

**Interfaces:**
- Produces: `routing_hit(chosen: str, expected_modes: list[str]) -> bool`; `_parse_judge_score(raw: str) -> int` (first int, clamped [0,5], junk→0); `aggregate(per_question: list[dict]) -> dict`.

- [ ] **Step 1: Write the failing test** (`tests/unit/test_router_eval.py`)

```python
from answer_api.router_eval import routing_hit, _parse_judge_score, aggregate


def test_routing_hit():
    assert routing_hit("global", ["global", "drift"]) is True
    assert routing_hit("local", ["global", "drift"]) is False


def test_parse_judge_score():
    assert _parse_judge_score("5") == 5
    assert _parse_judge_score("The score is 4.") == 4
    assert _parse_judge_score("```3```") == 3
    assert _parse_judge_score("banana") == 0
    assert _parse_judge_score("7") == 5        # clamp high
    assert _parse_judge_score("-2") == 0       # clamp low


def test_aggregate_core_metrics():
    pq = [
        {"question": "a", "intent": "local", "chosen": "local",
         "routing_hit": True, "grounding_hit": True, "faithfulness": 5},
        {"question": "b", "intent": "local", "chosen": "drift",
         "routing_hit": False, "grounding_hit": False, "faithfulness": 2},
        {"question": "c", "intent": "global", "chosen": "global",
         "routing_hit": True, "grounding_hit": None, "faithfulness": 4},
    ]
    s = aggregate(pq)
    assert s["n"] == 3
    assert abs(s["routing_accuracy"] - 2 / 3) < 1e-9
    assert s["routing_by_intent"]["local"] == 0.5 and s["routing_by_intent"]["global"] == 1.0
    assert s["grounding_precision"] == 0.5          # only a,b scored (c is None)
    assert abs(s["faithfulness_mean"] - 11 / 3) < 1e-9
    assert s["comparative"] is None                 # no comparative blocks


def test_aggregate_comparative_drift_wins():
    def broad(qid, comp):
        return {"question": qid, "intent": "drift", "chosen": "drift",
                "routing_hit": True, "grounding_hit": None, "faithfulness": 4,
                "comparative": comp}
    win = [broad("q", {
        "local": {"grounding_hit": None, "faithfulness": 2},
        "global": {"grounding_hit": None, "faithfulness": 3},
        "drift": {"grounding_hit": None, "faithfulness": 5}})]
    assert aggregate(win)["drift_wins"] is True
    lose = [broad("q", {
        "local": {"grounding_hit": None, "faithfulness": 2},
        "global": {"grounding_hit": None, "faithfulness": 5},
        "drift": {"grounding_hit": None, "faithfulness": 3}})]
    assert aggregate(lose)["drift_wins"] is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_router_eval.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — create `src/answer_api/router_eval.py`:

```python
"""Golden-set scoring for the /answer router: routing accuracy, citation grounding
(reuses golden.precision_at_k), and LLM-judge faithfulness. Pure aggregation only;
the @live harness that runs the router + judge lives in eval_router.py."""
from __future__ import annotations

import re


def routing_hit(chosen: str, expected_modes: list[str]) -> bool:
    return chosen in expected_modes


def _parse_judge_score(raw: str) -> int:
    m = re.search(r"-?\d+", raw or "")
    if m is None:
        return 0
    return max(0, min(5, int(m.group())))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(per_question: list[dict]) -> dict:
    n = len(per_question)
    by_intent: dict[str, list[bool]] = {}
    for r in per_question:
        by_intent.setdefault(r["intent"], []).append(bool(r["routing_hit"]))
    grounded = [r for r in per_question if r["grounding_hit"] is not None]
    ground_by_mode: dict[str, list[bool]] = {}
    for r in grounded:
        ground_by_mode.setdefault(r["chosen"], []).append(bool(r["grounding_hit"]))
    faith_by_mode: dict[str, list[int]] = {}
    for r in per_question:
        faith_by_mode.setdefault(r["chosen"], []).append(r["faithfulness"])

    comp_rows = [r for r in per_question if "comparative" in r]
    comparative: dict | None = None
    drift_wins: bool | None = None
    if comp_rows:
        comparative = {}
        for m in ("local", "global", "drift"):
            faiths = [r["comparative"][m]["faithfulness"] for r in comp_rows]
            gh = [r["comparative"][m]["grounding_hit"] for r in comp_rows
                  if r["comparative"][m]["grounding_hit"] is not None]
            comparative[m] = {
                "faithfulness": _mean(faiths),
                "grounding": (sum(1 for x in gh if x) / len(gh)) if gh else None,
            }
        d, lo, gl = comparative["drift"], comparative["local"], comparative["global"]
        grounding_ok = (d["grounding"] is None
                        or ((lo["grounding"] is None or d["grounding"] >= lo["grounding"])
                            and (gl["grounding"] is None or d["grounding"] >= gl["grounding"])))
        drift_wins = (d["faithfulness"] >= lo["faithfulness"]
                      and d["faithfulness"] >= gl["faithfulness"] and grounding_ok)

    return {
        "n": n,
        "routing_accuracy": (sum(1 for r in per_question if r["routing_hit"]) / n) if n else 0.0,
        "routing_by_intent": {k: sum(v) / len(v) for k, v in by_intent.items()},
        "grounding_precision": (sum(1 for r in grounded if r["grounding_hit"]) / len(grounded)
                                if grounded else None),
        "grounding_by_mode": {k: sum(v) / len(v) for k, v in ground_by_mode.items()},
        "faithfulness_mean": _mean([r["faithfulness"] for r in per_question]),
        "faithfulness_by_mode": {k: _mean(v) for k, v in faith_by_mode.items()},
        "comparative": comparative,
        "drift_wins": drift_wins,
        "per_question": per_question,
    }
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_router_eval.py -q`
Expected: PASS (4 tests). ruff + `mypy src/answer_api/router_eval.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/router_eval.py tests/unit/test_router_eval.py
git commit -m "feat(eval): pure router golden-set scoring (routing, judge parse, aggregate)"
```

---

### Task 2: The labeled set — `router_golden.json`

**Files:**
- Create: `src/answer_api/router_golden.json`
- Test: `tests/unit/test_router_golden_schema.py`

**Interfaces:**
- Produces: a JSON list; each entry has `question` (str), `intent` (one of local/global/drift/timeline), `expected_modes` (non-empty list ⊆ the four modes), `expected_article_ids` (list[str], possibly empty).

- [ ] **Step 1: Write the failing schema test** (`tests/unit/test_router_golden_schema.py`)

```python
import json
from pathlib import Path

_MODES = {"local", "global", "drift", "timeline"}
_PATH = Path(__file__).resolve().parents[2] / "src" / "answer_api" / "router_golden.json"


def test_router_golden_schema():
    data = json.loads(_PATH.read_text())
    assert isinstance(data, list) and len(data) >= 25
    intents = set()
    for e in data:
        assert set(e) >= {"question", "intent", "expected_modes", "expected_article_ids"}
        assert isinstance(e["question"], str) and e["question"].strip()
        assert e["intent"] in _MODES
        assert isinstance(e["expected_modes"], list) and e["expected_modes"]
        assert set(e["expected_modes"]) <= _MODES
        assert isinstance(e["expected_article_ids"], list)
        intents.add(e["intent"])
    assert intents == _MODES        # all four intents represented
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_router_golden_schema.py -q`
Expected: FAIL (file missing).

- [ ] **Step 3: Create `src/answer_api/router_golden.json`** with this exact content:

```json
[
  {"question": "What does AWS Backup Vault Lock enforce on recovery points?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["dad69d7c-189b-4d53-8d70-40697eae685a"]},
  {"question": "How does AWS Backup copy backups to another AWS Region?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["78e2b9e5-bf0c-4e76-8b83-fc54e76f52cd"]},
  {"question": "How are backups encrypted in AWS Backup?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["06059606-e70d-4d97-bcf3-c2bc223f1b47"]},
  {"question": "How do you restore an Amazon S3 backup?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["49d019c4-ceb8-4a0a-b908-cd35602cab20"]},
  {"question": "How does AWS Backup support continuous backups and point-in-time recovery?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["d4d25628-1f6e-42ea-a5c5-e66b2e1bf2f8"]},
  {"question": "How are Amazon Redshift clusters backed up?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["d1f26f66-1acb-421e-889d-d0d79e3af387"]},
  {"question": "How does AWS Backup perform cross-account backup?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["5d7d817e-7045-41c1-8b36-faa94a5b2c40"]},
  {"question": "How do you restore an Amazon EC2 instance from a backup?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["82ed2eb9-24a3-4322-bccb-958dd88c0f31"]},
  {"question": "What is soft delete in Azure Backup and how long does it retain deleted items?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["1cde879f-5152-4b0e-a9e8-9d4e762ee9ed", "b0eadd12-baf7-484e-8fb0-f0fd756aaba5"]},
  {"question": "How does Azure Backup encrypt backup data?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["f3199c91-3ccb-4538-b3e7-b4e45521d8ec"]},
  {"question": "How do you configure and run Cross Region Restore in Azure Backup?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["e2f3ee74-04f0-4d6f-9925-83dc0b43b193"]},
  {"question": "How do you restore a SQL Server database from an Azure Backup vault?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["8f869443-b48d-4a45-a9fb-7d0df772e632"]},
  {"question": "How do you back up an encrypted Azure VM?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["a0e14280-837e-4e23-a930-9119e7692e94"]},
  {"question": "How do you restore VMware VMs with Azure Backup Server?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["9967084e-7d84-4f8f-bca7-0ce534bee0a6"]},
  {"question": "How do you change the backup retention period in AWS Backup?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["b64e0602-ddf5-46dd-a7b5-789403ed3fd6"]},
  {"question": "Compare how AWS Backup and Azure Backup encrypt backup data.", "intent": "global", "expected_modes": ["global", "drift"], "expected_article_ids": ["06059606-e70d-4d97-bcf3-c2bc223f1b47", "f3199c91-3ccb-4538-b3e7-b4e45521d8ec"]},
  {"question": "Compare cross-region restore between AWS Backup and Azure Backup.", "intent": "global", "expected_modes": ["global", "drift"], "expected_article_ids": ["78e2b9e5-bf0c-4e76-8b83-fc54e76f52cd", "e2f3ee74-04f0-4d6f-9925-83dc0b43b193"]},
  {"question": "How do AWS Backup and Azure Backup each protect recovery points from deletion?", "intent": "global", "expected_modes": ["global", "drift"], "expected_article_ids": ["dad69d7c-189b-4d53-8d70-40697eae685a"]},
  {"question": "Which backup capabilities do AWS and Azure share for compliance retention?", "intent": "global", "expected_modes": ["global", "drift"], "expected_article_ids": []},
  {"question": "Compare AWS Backup and Azure Backup database restore workflows.", "intent": "global", "expected_modes": ["global", "drift"], "expected_article_ids": ["8f869443-b48d-4a45-a9fb-7d0df772e632"]},
  {"question": "What should I consider when planning long-term backup retention across cloud vendors?", "intent": "drift", "expected_modes": ["drift", "global"], "expected_article_ids": []},
  {"question": "What should I consider for backup encryption across cloud providers?", "intent": "drift", "expected_modes": ["drift", "global"], "expected_article_ids": ["06059606-e70d-4d97-bcf3-c2bc223f1b47", "f3199c91-3ccb-4538-b3e7-b4e45521d8ec"]},
  {"question": "What are the key considerations for cross-region disaster recovery of cloud backups?", "intent": "drift", "expected_modes": ["drift", "global"], "expected_article_ids": ["78e2b9e5-bf0c-4e76-8b83-fc54e76f52cd", "e2f3ee74-04f0-4d6f-9925-83dc0b43b193"]},
  {"question": "How should I approach immutability and ransomware protection for cloud backups?", "intent": "drift", "expected_modes": ["drift", "global"], "expected_article_ids": ["dad69d7c-189b-4d53-8d70-40697eae685a"]},
  {"question": "What should I think about when restoring databases from cloud backups?", "intent": "drift", "expected_modes": ["drift", "global"], "expected_article_ids": []},
  {"question": "How has Azure Backup soft-delete retention changed over time?", "intent": "timeline", "expected_modes": ["timeline"], "expected_article_ids": ["1cde879f-5152-4b0e-a9e8-9d4e762ee9ed", "b0eadd12-baf7-484e-8fb0-f0fd756aaba5"]},
  {"question": "How has AWS Backup vault lock behavior evolved?", "intent": "timeline", "expected_modes": ["timeline"], "expected_article_ids": ["dad69d7c-189b-4d53-8d70-40697eae685a"]},
  {"question": "How has the AWS Backup retention period configuration changed?", "intent": "timeline", "expected_modes": ["timeline"], "expected_article_ids": ["b64e0602-ddf5-46dd-a7b5-789403ed3fd6"]},
  {"question": "How has Azure Backup encryption support changed over time?", "intent": "timeline", "expected_modes": ["timeline"], "expected_article_ids": ["f3199c91-3ccb-4538-b3e7-b4e45521d8ec"]}
]
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_router_golden_schema.py -q`
Expected: PASS (29 entries, all four intents). `python -c "import json; json.load(open('src/answer_api/router_golden.json'))"` parses.

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/router_golden.json tests/unit/test_router_golden_schema.py
git commit -m "feat(eval): labeled router golden set (29 questions, 4 intents)"
```

---

### Task 3: The harness — `eval_router.py`

**Files:**
- Create: `src/answer_api/eval_router.py`
- Test: `tests/unit/test_eval_router_harness.py`

**Interfaces:**
- Consumes: `router_eval.aggregate`/`routing_hit`/`_parse_judge_score`, `golden.precision_at_k`, `router.answer_router` (via `router_mod`).
- Produces: `_faithfulness_judge(client, model, q, answer, cited_facts) -> int`; `_cited_fact_texts(driver, group_id, fact_uuids) -> list[str]`; `run_eval(clients, questions, settings) -> dict`; `format_report(summary) -> str`. `clients` is the 9-tuple `(graphiti, driver, embedder, synth_client, synth_model, map_client, map_model, cheap_client, cheap_model)`.

- [ ] **Step 1: Write the failing fixture test** (`tests/unit/test_eval_router_harness.py`)

```python
import pytest

import answer_api.eval_router as er
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


def _env(mode, article_id):
    cites = [{"marker": 1, "fact_uuid": "f1",
              "sources": [{"article_id": article_id, "url": "u", "title": "t"}]}] if article_id else []
    return {"mode": mode, "answer": "ans", "citations": cites,
            "routing": {"chosen": mode, "via": "llm", "fallback_from": None}}


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    async def _fake_router(*a, q, mode_override, vendor, settings):
        if mode_override is not None:
            return _env(mode_override, "A1")          # comparative forced mode
        return _env("local" if q == "loc" else "drift", "A1")
    async def _fake_judge(client, model, q, answer, facts):
        return 4
    async def _fake_facts(driver, g, uuids):
        return ["fact"]
    monkeypatch.setattr(er.router_mod, "answer_router", _fake_router)
    monkeypatch.setattr(er, "_faithfulness_judge", _fake_judge)
    monkeypatch.setattr(er, "_cited_fact_texts", _fake_facts)


async def test_run_eval_aggregates_and_comparative():
    questions = [
        {"question": "loc", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["A1"]},
        {"question": "broad", "intent": "drift", "expected_modes": ["drift", "global"], "expected_article_ids": []},
    ]
    summary = await er.run_eval((None,) * 9, questions, _S)
    assert summary["n"] == 2
    assert summary["routing_accuracy"] == 1.0              # loc->local, broad->drift both expected
    assert summary["grounding_precision"] == 1.0           # only 'loc' scored (broad is [])
    assert summary["comparative"] is not None              # the broad question ran the comparative pass
    report = er.format_report(summary)
    assert "Routing accuracy" in report and "drift_wins" in report.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_eval_router_harness.py -q`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement** — create `src/answer_api/eval_router.py`:

```python
"""@live harness: run router_golden.json through the /answer router, score routing
accuracy + citation grounding + LLM-judge faithfulness, run the comparative broad
pass, and write docs/superpowers/router-eval-report.md.

    uv run --extra dev python -m answer_api.eval_router
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from neo4j import AsyncGraphDatabase

from answer_api import router as router_mod
from answer_api.golden import precision_at_k
from answer_api.global_search import _map_client_and_model
from answer_api.router import _cheap_classify_client
from answer_api.router_eval import _parse_judge_score, aggregate, routing_hit
from answer_api.synthesize import _synthesis_client_and_model
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_embedder, build_graphiti

logger = logging.getLogger(__name__)
_QUESTIONS = Path(__file__).with_name("router_golden.json")

_JUDGE_PROMPT = (
    "You are scoring a backup-products ANSWER for faithfulness to its CITED FACTS. "
    "Score 0-5 how fully the answer's claims are supported by ONLY these facts "
    "(5 = every claim supported, 0 = unsupported/hallucinated). Reply with ONLY the "
    "integer.\n\nQUESTION: {q}\n\nANSWER:\n{answer}\n\nCITED FACTS:\n{facts}"
)


async def _faithfulness_judge(client, model, q, answer, cited_facts) -> int:
    facts_block = "\n".join(f"- {f}" for f in cited_facts) or "(none)"
    resp = await client.chat.completions.create(
        model=model, temperature=0, max_tokens=8,
        messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
            q=q, answer=answer, facts=facts_block)}])
    return _parse_judge_score(resp.choices[0].message.content or "")


async def _cited_fact_texts(driver, group_id, fact_uuids) -> list[str]:
    if not fact_uuids:
        return []
    async with driver.session() as s:
        r = await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $u "
            "RETURN f.fact AS fact", g=group_id, u=list(fact_uuids))
        return [rec["fact"] async for rec in r]


async def _score_one(clients, q, mode_override, settings):
    graphiti, driver, embedder, sc, sm, mc, mm, cc, cmodel = clients
    env = await router_mod.answer_router(
        graphiti, driver, embedder, sc, sm, mc, mm, cc, cmodel,
        q=q["question"], mode_override=mode_override, vendor=None, settings=settings)
    ghit = (precision_at_k(env["citations"], q["expected_article_ids"])
            if q["expected_article_ids"] else None)
    facts = await _cited_fact_texts(driver, settings.group_id,
                                    [c["fact_uuid"] for c in env["citations"]])
    faith = await _faithfulness_judge(sc, sm, q["question"], env["answer"], facts)
    return env, ghit, faith


async def run_eval(clients, questions, settings) -> dict:
    per_question: list[dict] = []
    for q in questions:
        try:
            env, ghit, faith = await _score_one(clients, q, None, settings)
            chosen = env["routing"]["chosen"]
            rec: dict = {"question": q["question"], "intent": q["intent"], "chosen": chosen,
                         "routing_hit": routing_hit(chosen, q["expected_modes"]),
                         "grounding_hit": ghit, "faithfulness": faith}
            if q["intent"] in ("global", "drift"):
                comp: dict = {}
                for m in ("local", "global", "drift"):
                    _e, _g, _f = await _score_one(clients, q, m, settings)
                    comp[m] = {"grounding_hit": _g, "faithfulness": _f}
                rec["comparative"] = comp
        except Exception:
            logger.warning("eval question failed: %s", q["question"], exc_info=True)
            rec = {"question": q["question"], "intent": q["intent"], "chosen": "error",
                   "routing_hit": False, "grounding_hit": None, "faithfulness": 0}
        per_question.append(rec)
    return aggregate(per_question)


def format_report(summary: dict) -> str:
    lines = ["# /answer Router Golden-Set Eval\n",
             f"Questions: {summary['n']}\n",
             f"**Routing accuracy: {summary['routing_accuracy']:.2f}**",
             f"by intent: {summary['routing_by_intent']}",
             f"Grounding precision: {summary['grounding_precision']}",
             f"by mode: {summary['grounding_by_mode']}",
             f"Faithfulness mean: {summary['faithfulness_mean']:.2f}",
             f"by mode: {summary['faithfulness_by_mode']}\n",
             f"Comparative (broad): {summary['comparative']}",
             f"drift_wins: {summary['drift_wins']}\n",
             "## Per question\n",
             "| intent | chosen | routing | grounding | faithfulness | question |",
             "|---|---|---|---|---|---|"]
    for r in summary["per_question"]:
        lines.append(f"| {r['intent']} | {r['chosen']} | {r['routing_hit']} | "
                     f"{r['grounding_hit']} | {r['faithfulness']} | {r['question']} |")
    return "\n".join(lines) + "\n"


async def main() -> None:
    settings = get_extract_settings()
    questions = json.loads(_QUESTIONS.read_text())
    graphiti = build_graphiti(settings)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    embedder = build_embedder(settings)
    sc, sm = _synthesis_client_and_model(settings)
    mc, mm = _map_client_and_model(settings)
    cc, cmodel = _cheap_classify_client(settings)
    clients = (graphiti, driver, embedder, sc, sm, mc, mm, cc, cmodel)
    try:
        summary = await run_eval(clients, questions, settings)
        report = format_report(summary)
        out = (Path(__file__).resolve().parents[2]
               / "docs" / "superpowers" / "router-eval-report.md")
        out.write_text(report)
        print(report)
    finally:
        await graphiti.close()
        await driver.close()
        await sc.close()
        await mc.close()
        if cc is not None:
            await cc.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run to verify passes**

Run: `uv run --extra dev pytest tests/unit/test_eval_router_harness.py -q`
Expected: PASS. `uv run ruff check src/answer_api/eval_router.py tests/unit/test_eval_router_harness.py` + `uv run mypy src/answer_api/eval_router.py` clean. (Note: `_map_client_and_model`/`_cheap_classify_client`/`build_embedder` imports are used only in `main`; that's fine — `main` uses them.)

- [ ] **Step 5: Commit**

```bash
git add src/answer_api/eval_router.py tests/unit/test_eval_router_harness.py
git commit -m "feat(eval): @live router golden-set harness (judge + comparative + report)"
```

---

### Task 4: `@live` smoke + full-suite gate + the scored run

**Files:**
- Create: `tests/integration/test_router_eval_live.py`

- [ ] **Step 1: Full non-live suite + CI lint gate**

Run: `uv run --extra dev pytest -m "not live" -q` → all pass.
Run: `uv run ruff check src tests` → clean. `uv run mypy src` → clean.
Fix any fallout before continuing.

- [ ] **Step 2: `@live` smoke** (`tests/integration/test_router_eval_live.py`) — runs the real harness over a SMALL slice (to bound cost) and sanity-checks it

```python
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_router_eval_smoke_live(live_extract_driver):
    import answer_api.eval_router as er
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_embedder, build_graphiti
    from answer_api.synthesize import _synthesis_client_and_model, _URL_RE
    from answer_api.global_search import _map_client_and_model
    from answer_api.router import _cheap_classify_client
    s = get_extract_settings()
    graphiti = build_graphiti(s)
    emb = build_embedder(s)
    sc, sm = _synthesis_client_and_model(s)
    mc, mm = _map_client_and_model(s)
    cc, cmodel = _cheap_classify_client(s)
    clients = (graphiti, live_extract_driver, emb, sc, sm, mc, mm, cc, cmodel)
    # one per intent (comparative fires for the global/drift ones) — bounds cost
    questions = [
        {"question": "What does AWS Backup Vault Lock enforce on recovery points?", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["dad69d7c-189b-4d53-8d70-40697eae685a"]},
        {"question": "How has Azure Backup soft-delete retention changed over time?", "intent": "timeline", "expected_modes": ["timeline"], "expected_article_ids": []},
    ]
    try:
        summary = await er.run_eval(clients, questions, s)
        assert summary["n"] == 2
        assert summary["routing_accuracy"] >= 0.5
        for r in summary["per_question"]:
            assert 0 <= r["faithfulness"] <= 5
        report = er.format_report(summary)
        assert "Routing accuracy" in report
    finally:
        await graphiti.close()
        await sc.close()
        await mc.close()
        if cc is not None:
            await cc.close()
```

Run: `uv run --extra dev pytest -m live tests/integration/test_router_eval_live.py -q`
Expected: PASS.

- [ ] **Step 3: The full scored run (controller)**

Run `uv run --extra dev python -m answer_api.eval_router` on `backup-docs` to produce `docs/superpowers/router-eval-report.md` (all 29 questions + the comparative broad pass). This is the Phase 4 exit measurement — the routing accuracy, per-mode grounding/faithfulness, and `drift_wins` verdict.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_router_eval_live.py docs/superpowers/router-eval-report.md
git commit -m "test(eval): live router-eval smoke + the scored golden-set report"
```

---

## Notes for the implementer

- Pure scoring (`router_eval.py`) is unit-tested with no infra; the harness loop (`run_eval`) is fixture-tested by monkeypatching `router_mod.answer_router`, `_faithfulness_judge`, and `_cited_fact_texts`; only the `@live` smoke + the scored run need real infra.
- Grounding reuses `golden.precision_at_k` directly on `env["citations"]` (citations carry `sources[].article_id` for every mode). `expected_article_ids == []` → grounding is `None` (skipped), never counted against precision.
- Faithfulness judges over the resolved cited FACT TEXTS (`_cited_fact_texts`), not URLs — that's what makes the score meaningful.
- The comparative pass runs only for `intent ∈ {global, drift}` and triples the calls on that subset (hence the `@live` smoke uses a 2-question slice to bound cost; the full scored run is a manual controller step).
- Test fakes: one statement per line, no semicolons (E702); imports at file top (E402).
- The existing `golden_questions.json`/`eval_golden.py`/`eval_answer.py` are NOT touched.
