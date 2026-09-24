# Coverage-Prompt A/B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether a coverage instruction (plus removing the cheap tier's fixed fact count) recovers today's distinct extracted content when chunks are packed to 1,200 tokens.

**Architecture:** `scripts/chunk_ab.py` gains a `--coverage` mode that reuses the finished `today`/`pack1200` arms and runs one new arm with overridden tier instructions, gated on proof (LLM capture) that graphiti actually sent the directive. A new `scripts/chunk_ab_judge.py` scores all three arms by distinct ideas with the eval-judge LLM.

**Tech Stack:** Python 3.12, graphiti-core 0.30.1 (unmodified), testcontainers, the project's `bounded_llm_client`.

**Spec:** `docs/superpowers/specs/2026-09-24-coverage-prompt-ab-design.md`.

## Global Constraints

- Production code (`src/`) is NOT changed. The variant instructions live in `scripts/chunk_ab.py`.
- `CHEAP_TIER_SALIENCE_UNCAPPED` = `CHEAP_TIER_SALIENCE` with exactly the sentence `" A dense table chunk should yield roughly 10-25 facts total, not dozens."` removed; built by string replacement with an assertion that the sentence was present.
- The coverage directive contains the marker `COVERAGE_MARKER = "COVERAGE (work through the whole message)"`.
- Engagement gate: the new arm's smoke refuses unless a captured `extract_nodes.*` or `extract_edges.*` prompt contains `COVERAGE_MARKER`, the usage tally moved, and every smoke row with `episodes > 0` has `prompt_tokens > 0`.
- The reused arms' article ids must equal the newly selected article list, in order; refuse otherwise.
- Judge = `answer_api.eval_router._eval_judge_client_and_model(settings)`; one call per article; arm labels shuffled per article with `random.Random(article_id)`; `response_format={"type": "json_object"}`, `temperature=0`, `max_tokens=32000`.
- A judgement that does not label every fact of every arm exactly once, in order, with an idea id present in the inventory, is retried once, then recorded unscored (`None`, never 0).
- Articles whose largest arm has more than 150 facts are excluded from judging and listed.
- Decision rule: adopt if paired median of `distinct(pack1200_coverage)/distinct(today)` ≥ 0.95 AND total unsupported(coverage) ≤ total unsupported(today) AND paired median `cost(coverage)/cost(today)` ≤ 0.75.
- Tests hermetic; `uv run ruff check src tests scripts/chunk_ab.py scripts/chunk_ab_judge.py`; `uv run mypy src`. Subagents never run either script and never launch long background jobs.

---

### Task 1: Coverage arm in `chunk_ab.py`

**Files:**
- Modify: `scripts/chunk_ab.py`
- Test: `tests/unit/test_chunk_ab.py`

**Interfaces:**
- Produces: `COVERAGE_MARKER: str`, `COVERAGE_DIRECTIVE: str`, `CHEAP_TIER_SALIENCE_UNCAPPED: str`, `ARM_INSTRUCTIONS: dict[str, tuple[str, str]]` (arm → (strong, cheap)), `directive_seen(capture_path: Path, marker: str) -> bool`, `reused_rows(prev: dict, arm: str, article_ids: list[str]) -> list[dict]`, and the `--coverage --reuse PREV.json` CLI mode writing `<out>/results_coverage.json` with keys `today`, `pack1200`, `pack1200_coverage` (each `{"summary", "rows"}`), `spent_usd_new_arm`, `method_notes`.

- [ ] **Step 1: Failing tests** — append to `tests/unit/test_chunk_ab.py`:

```python
import json as _json

from graph_extract.ontology import CHEAP_TIER_SALIENCE, EXTRACTION_INSTRUCTIONS


def test_uncapped_salience_drops_only_the_fixed_count():
    assert "10-25 facts" in CHEAP_TIER_SALIENCE
    assert "10-25 facts" not in ab.CHEAP_TIER_SALIENCE_UNCAPPED
    assert "NOT EXHAUSTIVE" in ab.CHEAP_TIER_SALIENCE_UNCAPPED  # anti-per-cell rules stay


def test_coverage_arm_instructions_carry_the_marker_on_both_tiers():
    strong, cheap = ab.ARM_INSTRUCTIONS["pack1200_coverage"]
    assert strong.startswith(EXTRACTION_INSTRUCTIONS) and ab.COVERAGE_MARKER in strong
    assert cheap.startswith(EXTRACTION_INSTRUCTIONS) and ab.COVERAGE_MARKER in cheap
    assert "10-25 facts" not in cheap
    assert ab.ARMS["pack1200_coverage"] == ab.ARMS["pack1200"]


def test_directive_seen_requires_an_extraction_prompt_with_the_marker(tmp_path):
    p = tmp_path / "cap.jsonl"
    rec = lambda name, text: _json.dumps(
        {"prompt_name": name, "messages": [{"role": "user", "content": text}]})
    p.write_text(rec("dedupe_nodes.nodes", ab.COVERAGE_MARKER) + "\n"
                 + rec("extract_nodes.extract_text", "no marker") + "\n")
    assert ab.directive_seen(p, ab.COVERAGE_MARKER) is False
    p.write_text(p.read_text() + rec("extract_edges.edge", "x " + ab.COVERAGE_MARKER) + "\n")
    assert ab.directive_seen(p, ab.COVERAGE_MARKER) is True


def test_directive_seen_is_false_for_a_missing_capture(tmp_path):
    assert ab.directive_seen(tmp_path / "absent.jsonl", ab.COVERAGE_MARKER) is False


def test_reused_rows_must_match_the_article_order():
    prev = {"today": {"rows": [{"article_id": "a"}, {"article_id": "b"}]}}
    assert [r["article_id"] for r in ab.reused_rows(prev, "today", ["a", "b"])] == ["a", "b"]
    import pytest
    with pytest.raises(SystemExit):
        ab.reused_rows(prev, "today", ["b", "a"])
```

(ruff E731 may flag the `rec = lambda` line — if so, use a nested `def rec(name, text)`.)

- [ ] **Step 2: Run to verify failure** — `uv run --extra dev pytest tests/unit/test_chunk_ab.py -q` → the new tests FAIL.

- [ ] **Step 3: Implement** — in `scripts/chunk_ab.py` add the import `from graph_extract.ontology import CHEAP_TIER_SALIENCE, EXTRACTION_INSTRUCTIONS` and, after `ARMS`:

```python
ARMS["pack1200_coverage"] = dict(ARMS["pack1200"])

_FIXED_COUNT = " A dense table chunk should yield roughly 10-25 facts total, not dozens."
assert _FIXED_COUNT in CHEAP_TIER_SALIENCE, "CHEAP_TIER_SALIENCE changed; update the variant"
CHEAP_TIER_SALIENCE_UNCAPPED = CHEAP_TIER_SALIENCE.replace(_FIXED_COUNT, "")

COVERAGE_MARKER = "COVERAGE (work through the whole message)"
COVERAGE_DIRECTIVE = (
    f"\n\n{COVERAGE_MARKER}: the CURRENT MESSAGE may be long and cover several "
    "sections. Go through it section by section, from the first line to the last, and "
    "extract EVERY product, feature, workload, platform, requirement, limitation and "
    "availability statement each section states -- do not stop after the first topics "
    "and do not condense the message into a few summary items. A longer message "
    "should yield proportionally more entities and facts. Never restate one fact in "
    "different words: each fact must add a claim or a concrete detail (a version, a "
    "limit, a region, a condition) that no other fact already states."
)
ARM_INSTRUCTIONS: dict[str, tuple[str, str]] = {
    "pack1200_coverage": (
        EXTRACTION_INSTRUCTIONS + COVERAGE_DIRECTIVE,
        EXTRACTION_INSTRUCTIONS + CHEAP_TIER_SALIENCE_UNCAPPED + COVERAGE_DIRECTIVE,
    ),
}


def directive_seen(capture_path: Path, marker: str) -> bool:
    """True iff graphiti actually SENT the directive in an extraction prompt --
    proof the override reached the model, not merely that an attribute was set."""
    if not capture_path.exists():
        return False
    for line in capture_path.read_text().splitlines():
        rec = json.loads(line)
        if not str(rec.get("prompt_name", "")).startswith(("extract_nodes.", "extract_edges.")):
            continue
        if any(marker in str(m.get("content", "")) for m in rec.get("messages") or []):
            return True
    return False


def reused_rows(prev: dict, arm: str, article_ids: list[str]) -> list[dict]:
    rows = prev[arm]["rows"]
    if [r["article_id"] for r in rows] != article_ids:
        raise SystemExit(f"reused arm {arm!r} does not match the selected articles "
                         "in order -- refusing to compare different samples")
    return rows
```

In `run_arm`, add a keyword parameter `capture_path: Path | None = None`; include `"llm_capture_path": str(capture_path)` in the `model_copy` update when it is given; and directly after `_build_ingest_driver(s)` returns, inside the `try`:

```python
            if name in ARM_INSTRUCTIONS:
                strong_instr, cheap_instr = ARM_INSTRUCTIONS[name]
                if ingest._cheap is None:
                    raise SystemExit("cheap tier not configured; the variant needs both tiers")
                ingest._strong.instructions = strong_instr
                ingest._cheap.instructions = cheap_instr
```

Add a `--coverage` flag and `--reuse PREV.json` option to `main`, and branch at the top of `main` after selecting `articles`:

```python
    if args.coverage:
        if not args.reuse:
            raise SystemExit("--coverage needs --reuse PREV_results.json")
        prev = json.loads(Path(args.reuse).read_text())
        ids = [a["id"] for a in articles]
        reused = {arm: reused_rows(prev, arm, ids) for arm in ("today", "pack1200")}
        cap = out / "capture-smoke.jsonl"
        tally_before = get_tally().prompt_tokens
        smoke_rows = await run_arm("pack1200_coverage", articles[:SMOKE_N], 0.0, out,
                                   capture_path=cap)
        (out / "smoke_coverage.json").write_text(json.dumps(smoke_rows, indent=1))
        if get_tally().prompt_tokens == tally_before:
            raise SystemExit("usage tally did not move -- refusing")
        if any(r["episodes"] > 0 and r["prompt_tokens"] == 0 for r in smoke_rows):
            raise SystemExit("an unmetered smoke row -- refusing")
        if not directive_seen(cap, COVERAGE_MARKER):
            raise SystemExit("coverage directive not found in any captured extraction "
                             "prompt -- the variable is not engaged; refusing")
        spent = sum(r["cost"] for r in smoke_rows)
        print(f"coverage smoke passed; spent=${spent:.3f}", flush=True)
        if args.smoke_only:
            return
        rows = await run_arm("pack1200_coverage", articles, spent, out)
        spent += sum(r["cost"] for r in rows)
        results = {arm: {"summary": summarise(r), "rows": r} for arm, r in reused.items()}
        results["pack1200_coverage"] = {"summary": summarise(rows), "rows": rows}
        results["spent_usd_new_arm"] = spent
        results["method_notes"] = prev.get("method_notes", []) + [
            "today and pack1200 reused from " + str(args.reuse)
            + "; pack1200_coverage run separately with overridden tier instructions"]
        (out / "results_coverage.json").write_text(json.dumps(results, indent=1, default=str))
        for arm in results:
            if isinstance(results[arm], dict) and "summary" in results[arm]:
                print(arm, results[arm]["summary"], flush=True)
        return
```

Update the module docstring's usage line to document `--coverage --reuse PREV.json`.

- [ ] **Step 4: Run to verify pass** — `uv run --extra dev pytest tests/unit/test_chunk_ab.py -q`; `uv run ruff check src tests scripts/chunk_ab.py`; `uv run mypy src`.

- [ ] **Step 5: Mutation-test** (script asserting `old in s`; restore; never commit): (a) make `directive_seen` return True unconditionally at the end; (b) drop the `startswith(("extract_nodes.", "extract_edges."))` filter; (c) remove `_FIXED_COUNT` from the replace (replace with ""→ no-op, i.e. use `CHEAP_TIER_SALIENCE` itself). Each killed.

- [ ] **Step 6: Commit** — `feat(scripts): coverage-prompt arm for the chunk A/B (paid; controller-run)`.

---

### Task 2: The distinct-idea judge

**Files:**
- Create: `scripts/chunk_ab_judge.py`
- Test: `tests/unit/test_chunk_ab_judge.py`

**Interfaces:**
- Consumes: `answer_api.eval_router._eval_judge_client_and_model(settings) -> (AsyncOpenAI, str)`; `graph_extract.content_fetch.fetch_article(client, id)` (`.content_markdown`); `docext.client.make_docext_client`.
- Produces: `labels_for(article_id: str, arms: list[str]) -> dict[str, str]` (arm → "A"/"B"/"C"), `build_prompt(markdown: str, facts_by_label: dict[str, list[str]]) -> str`, `validate(obj: Any, counts: dict[str, int]) -> str | None` (error or None), `score(labels: list[dict]) -> dict` (`facts`, `supported`, `distinct`, `redundant`, `unsupported`), `decide(per_article: list[dict]) -> dict`.

- [ ] **Step 1: Failing tests** — `tests/unit/test_chunk_ab_judge.py`:

```python
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "judge", Path(__file__).parents[2] / "scripts" / "chunk_ab_judge.py")
j = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(j)

ARMS = ["today", "pack1200", "pack1200_coverage"]


def test_labels_are_a_deterministic_shuffle():
    a = j.labels_for("art-1", ARMS)
    assert sorted(a.values()) == ["A", "B", "C"] and a == j.labels_for("art-1", ARMS)


def test_prompt_numbers_every_fact_under_its_label():
    p = j.build_prompt("SOURCE TEXT", {"A": ["f1", "f2"], "B": ["g1"]})
    assert "SOURCE TEXT" in p and "A1: f1" in p and "A2: f2" in p and "B1: g1" in p


def test_validate_accepts_a_complete_judgement():
    obj = {"ideas": [{"id": 1, "text": "x"}, {"id": 2, "text": "y"}],
           "labels": {"A": [{"idea": 1, "supported": True}, {"idea": 2, "supported": False}],
                      "B": [{"idea": 1, "supported": True}]}}
    assert j.validate(obj, {"A": 2, "B": 1}) is None


def test_validate_rejects_wrong_counts_unknown_ideas_and_bad_types():
    base = {"ideas": [{"id": 1, "text": "x"}],
            "labels": {"A": [{"idea": 1, "supported": True}], "B": []}}
    assert j.validate(base, {"A": 1, "B": 1}) is not None          # B missing a label
    bad_id = {"ideas": [{"id": 1, "text": "x"}],
              "labels": {"A": [{"idea": 9, "supported": True}]}}
    assert j.validate(bad_id, {"A": 1}) is not None                  # unknown idea
    bad_type = {"ideas": [{"id": 1, "text": "x"}],
                "labels": {"A": [{"idea": 1, "supported": "yes"}]}}
    assert j.validate(bad_type, {"A": 1}) is not None                # non-bool
    assert j.validate("not a dict", {"A": 1}) is not None


def test_score_counts_distinct_supported_ideas():
    labels = [{"idea": 1, "supported": True}, {"idea": 1, "supported": True},
              {"idea": 2, "supported": True}, {"idea": 3, "supported": False}]
    assert j.score(labels) == {"facts": 4, "supported": 3, "distinct": 2,
                               "redundant": 1, "unsupported": 1}


def test_decide_applies_the_spec_rule():
    rows = [{"today": {"distinct": 10, "unsupported": 1}, "pack1200_coverage":
             {"distinct": 10, "unsupported": 1}, "cost_ratio": 0.6}] * 3
    assert j.decide(rows)["adopt"] is True
    rows2 = [{"today": {"distinct": 10, "unsupported": 1}, "pack1200_coverage":
              {"distinct": 8, "unsupported": 1}, "cost_ratio": 0.6}] * 3
    assert j.decide(rows2)["adopt"] is False
```

- [ ] **Step 2: Run to verify failure** — `uv run --extra dev pytest tests/unit/test_chunk_ab_judge.py -q` → FAIL.

- [ ] **Step 3: Implement** — `scripts/chunk_ab_judge.py`:

```python
"""Score the chunk A/B arms by DISTINCT ideas, not raw fact counts. PAID (judge LLM).

    uv run python scripts/chunk_ab_judge.py --results results_coverage.json --out DIR

Per article: the source markdown plus every arm's facts under shuffled anonymous
labels go to the eval judge (a different model family from both extraction tiers),
which returns an inventory of distinct ideas and, per fact, its idea id and whether
the source supports it. Spec: docs/superpowers/specs/2026-09-24-coverage-prompt-ab-design.md
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
from pathlib import Path
from typing import Any

from answer_api.eval_router import _eval_judge_client_and_model
from docext.client import make_docext_client
from graph_extract.config import get_extract_settings
from graph_extract.content_fetch import fetch_article

ARMS = ["today", "pack1200", "pack1200_coverage"]
MAX_FACTS = 150
LETTERS = "ABC"

INSTRUCTIONS = """You are auditing knowledge-graph extraction from one documentation page.
SOURCE is the page. Below it are fact lists produced by independent extraction runs,
labelled by letter; each fact is numbered (A1, A2, ...).

1. Build an inventory of the DISTINCT claims across ALL lists. Two facts express the
   same idea if they make the same claim about the same things, even worded
   differently, or if one merely restates the other. A fact that adds a concrete detail
   (a version, a limit, a region, a condition, a component) is a DIFFERENT idea.
2. For EVERY fact of EVERY list, in order, give the id of the idea it expresses and
   whether SOURCE supports it (true/false).

Return only JSON:
{"ideas": [{"id": <int>, "text": <short statement>}],
 "labels": {"A": [{"idea": <int>, "supported": <bool>}, ...], "B": [...], ...}}
Each label list must have exactly one entry per fact of that list, in the same order.
"""


def labels_for(article_id: str, arms: list[str]) -> dict[str, str]:
    letters = list(LETTERS[: len(arms)])
    random.Random(article_id).shuffle(letters)
    return dict(zip(arms, letters))


def build_prompt(markdown: str, facts_by_label: dict[str, list[str]]) -> str:
    parts = [INSTRUCTIONS, "SOURCE:\n" + markdown]
    for label in sorted(facts_by_label):
        lines = [f"{label}{i}: {f}" for i, f in enumerate(facts_by_label[label], 1)]
        parts.append(f"LIST {label}:\n" + ("\n".join(lines) if lines else "(empty)"))
    return "\n\n".join(parts)


def validate(obj: Any, counts: dict[str, int]) -> str | None:
    if not isinstance(obj, dict):
        return "reply is not a JSON object"
    ideas = obj.get("ideas")
    labels = obj.get("labels")
    if not isinstance(ideas, list) or not isinstance(labels, dict):
        return "missing ideas/labels"
    ids = {i.get("id") for i in ideas if isinstance(i, dict)}
    for label, n in counts.items():
        got = labels.get(label)
        if not isinstance(got, list) or len(got) != n:
            return f"list {label}: expected {n} labels, got {len(got) if isinstance(got, list) else got!r}"
        for k, entry in enumerate(got):
            if not isinstance(entry, dict) or entry.get("idea") not in ids:
                return f"list {label} fact {k + 1}: unknown or missing idea id"
            if not isinstance(entry.get("supported"), bool):
                return f"list {label} fact {k + 1}: supported is not a bool"
    return None


def score(labels: list[dict]) -> dict:
    supported = [e for e in labels if e["supported"]]
    distinct = len({e["idea"] for e in supported})
    return {"facts": len(labels), "supported": len(supported), "distinct": distinct,
            "redundant": len(supported) - distinct,
            "unsupported": len(labels) - len(supported)}


def decide(per_article: list[dict]) -> dict:
    ratios = [r["pack1200_coverage"]["distinct"] / r["today"]["distinct"]
              for r in per_article if r["today"]["distinct"]]
    unsup_cov = sum(r["pack1200_coverage"]["unsupported"] for r in per_article)
    unsup_today = sum(r["today"]["unsupported"] for r in per_article)
    cost = statistics.median(r["cost_ratio"] for r in per_article)
    distinct_ratio = statistics.median(ratios) if ratios else 0.0
    adopt = distinct_ratio >= 0.95 and unsup_cov <= unsup_today and cost <= 0.75
    return {"adopt": adopt, "distinct_ratio_median": distinct_ratio,
            "unsupported_coverage": unsup_cov, "unsupported_today": unsup_today,
            "cost_ratio_median": cost, "articles": len(per_article)}


async def _judge_one(client, model, prompt: str, counts: dict[str, int]) -> dict | None:
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0, max_tokens=32000)
        text = resp.choices[0].message.content if resp.choices else None
        if not text:
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        if validate(obj, counts) is None:
            return obj
    return None


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads(Path(args.results).read_text())
    rows = {arm: {r["article_id"]: r for r in res[arm]["rows"]} for arm in ARMS}
    s = get_extract_settings()
    client, model = _eval_judge_client_and_model(s)
    dx = make_docext_client(base_url=s.docext_base_url, read_key=s.docext_read_key,
                            admin_key=s.docext_admin_key, verify_tls=s.docext_verify_tls)
    per_article: list[dict] = []
    excluded: list[str] = []
    unscored: list[str] = []
    try:
        for aid in rows["today"]:
            if max(rows[arm][aid]["facts"] for arm in ARMS) > MAX_FACTS:
                excluded.append(aid)
                continue
            md = (await fetch_article(dx, aid)).content_markdown
            lab = labels_for(aid, ARMS)
            facts = {lab[arm]: rows[arm][aid]["fact_texts"] for arm in ARMS}
            obj = await _judge_one(client, model, build_prompt(md, facts),
                                   {k: len(v) for k, v in facts.items()})
            if obj is None:
                unscored.append(aid)
                continue
            row = {"article_id": aid, **{arm: score(obj["labels"][lab[arm]]) for arm in ARMS},
                   "cost_ratio": (rows["pack1200_coverage"][aid]["cost"]
                                  / rows["today"][aid]["cost"]) if rows["today"][aid]["cost"] else 1.0}
            per_article.append(row)
            with (out / "judged.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(aid[:8], {arm: row[arm]["distinct"] for arm in ARMS}, flush=True)
    finally:
        await dx.aclose()
        await client.close()
    summary = {"decision": decide(per_article) if per_article else None,
               "totals": {arm: {k: sum(r[arm][k] for r in per_article)
                                for k in ("facts", "supported", "distinct", "redundant", "unsupported")}
                          for arm in ARMS},
               "excluded_over_max_facts": excluded, "unscored": unscored, "judge_model": model}
    (out / "judge_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run to verify pass** — `uv run --extra dev pytest tests/unit/test_chunk_ab_judge.py -q`; `uv run ruff check src tests scripts/chunk_ab_judge.py`; `uv run mypy src`.

- [ ] **Step 5: Mutation-test**: (a) `validate` skips the length check; (b) `score` counts distinct over ALL labels instead of supported; (c) `decide` drops the unsupported condition. Each killed.

- [ ] **Step 6: Commit** — `feat(scripts): distinct-idea judge for the chunk A/B (paid; controller-run)`.

---

### Step 7 (controller only)

Run the coverage smoke, then the full coverage arm, then the judge; write
`docs/superpowers/coverage-prompt-ab-2026-09-24.md` with the three arms' distinct/redundant/
unsupported totals, paired ratios, cost/time, the decision, and a hand check of 2 articles
calibrating the judge.
