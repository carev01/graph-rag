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
from graph_extract.usage import usable_content

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


def judgeable(rows: dict[str, dict[str, Any]], arms: list[str],
              max_facts: int) -> tuple[list[str], list[str], list[str]]:
    """Partition articles into judgeable, excluded (over fact limit), and missing from some arm.

    Preserves today's order. Articles are in `to_judge` if present in all arms and under max_facts.
    """
    to_judge: list[str] = []
    excluded: list[str] = []
    missing: list[str] = []
    for aid in rows[arms[0]]:
        if any(aid not in rows[arm] for arm in arms):
            missing.append(aid)
        elif max(rows[arm][aid]["facts"] for arm in arms) > max_facts:
            excluded.append(aid)
        else:
            to_judge.append(aid)
    return to_judge, excluded, missing


async def _judge_one(client, model, prompt: str, counts: dict[str, int]) -> dict | None:
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0, max_tokens=32000)
        text = usable_content(resp)
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
    to_judge, excluded, missing = judgeable(rows, ARMS, MAX_FACTS)
    unscored: list[str] = []
    try:
        for aid in to_judge:
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
               "excluded_over_max_facts": excluded, "unscored": unscored,
               "missing_from_some_arm": missing, "judge_model": model}
    (out / "judge_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
