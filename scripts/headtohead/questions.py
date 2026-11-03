"""Schema and validator for the head-to-head eval set (`eval/headtohead/questions.yaml`).

Run: uv run --extra dev python -m scripts.headtohead.questions --check \
    eval/headtohead/questions.yaml
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

TYPES = ("single_vendor", "multi_article", "cross_vendor", "unanswerable")
TARGET_COUNTS = {"single_vendor": 50, "multi_article": 15, "cross_vendor": 15, "unanswerable": 20}
ORIGINS = ("new", "golden_questions", "router_golden_tier1")
REQUIRED = (
    "id", "type", "question", "gold_facts", "gold_sources", "vendors",
    "written_from", "decides", "origin",
)
MAX_GOLD_FACTS = 8

DEFAULT_CORPUS = "eval/headtohead/corpus.json"
DEFAULT_INGEST_REPORT = "eval/headtohead/ingest_report.json"


def load_questions(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a YAML list of questions")
    return data


def _names(vendor: str, facts: list[str]) -> bool:
    pattern = re.compile(r"(?<!\w)" + re.escape(vendor) + r"(?!\w)", re.IGNORECASE)
    return any(pattern.search(f) for f in facts)


def _check_ids(label: str, ids: list[str], corpus: set[str], changed: set[str],
               what: str) -> list[str]:
    out = []
    for i in ids:
        if i in changed:
            out.append(f"{label}: {what} {i} changed since the graph was built")
        elif i not in corpus:
            out.append(f"{label}: {what} {i} is not in the corpus")
    return out


def _article_id(v: Any) -> str | None:
    """An article id as written in the YAML: a string, or an int (coerced to str, as
    written_from ids are). Anything else (bool, None, float, mapping) is not an id."""
    if isinstance(v, bool):
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    return None


def _validate_one(q: dict, corpus: set[str], changed: set[str],
                  corpus_vendors: set[str]) -> list[str]:
    label = str(q.get("id", "<no id>"))
    problems = [f"{label}: missing key {k}" for k in REQUIRED if k not in q]
    qtype = q.get("type")
    if "type" in q and qtype not in TYPES:
        problems.append(f"{label}: unknown type {qtype!r}")
    if "origin" in q and q["origin"] not in ORIGINS:
        problems.append(f"{label}: unknown origin {q['origin']!r}")

    facts = q.get("gold_facts")
    if isinstance(facts, list):
        if qtype == "unanswerable":
            if facts:
                problems.append(f"{label}: unanswerable gold_facts must be empty")
            sc = q.get("search_check")
            if not isinstance(sc, str) or not sc.strip():
                problems.append(f"{label}: unanswerable needs a non-empty search_check")
        elif not 1 <= len(facts) <= MAX_GOLD_FACTS:
            problems.append(
                f"{label}: gold_facts has {len(facts)} entries, need 1 to {MAX_GOLD_FACTS}"
            )
    elif "gold_facts" in q:
        problems.append(f"{label}: gold_facts must be a list")

    answerable = qtype != "unanswerable"
    sources = q.get("gold_sources")
    if isinstance(sources, list):
        if answerable and not sources:
            problems.append(f"{label}: an answerable question needs non-empty gold_sources")
        for s in sources:
            sid = _article_id(s.get("article_id")) if isinstance(s, dict) else None
            if sid is None:
                problems.append(f"{label}: gold source without article_id: {s!r}")
            else:
                problems += _check_ids(label, [sid], corpus, changed, "gold source")
    elif "gold_sources" in q:
        problems.append(f"{label}: gold_sources must be a list")
    wf = q.get("written_from")
    if isinstance(wf, list):
        if answerable and not wf:
            problems.append(f"{label}: an answerable question needs non-empty written_from")
        for i in wf:
            wid = _article_id(i)
            if wid is None:
                problems.append(f"{label}: written_from entry is not an article id: {i!r}")
            else:
                problems += _check_ids(label, [wid], corpus, changed, "written_from")
    elif "written_from" in q:
        problems.append(f"{label}: written_from must be a list")

    vendors = q.get("vendors")
    if isinstance(vendors, list):
        for v in vendors:
            if str(v) not in corpus_vendors:
                problems.append(f"{label}: vendor {v!r} is not a corpus vendor "
                                f"({', '.join(sorted(corpus_vendors))})")
    elif "vendors" in q:
        problems.append(f"{label}: vendors must be a list")

    if "decides" in q and "origin" in q and q["decides"] is not (q["origin"] == "new"):
        problems.append(f"{label}: decides must be true exactly when origin is new")

    if qtype == "cross_vendor":
        if not isinstance(vendors, list) or len(vendors) < 2:
            problems.append(f"{label}: cross_vendor needs at least 2 vendors")
        elif isinstance(facts, list):
            for v in vendors:
                if not _names(str(v), [str(f) for f in facts]):
                    problems.append(f"{label}: gold_facts do not name vendor {v}")
    return problems


def validate(questions: list[Any], corpus_ids: set[str], changed_ids: set[str], *,
             corpus_vendors: set[str]) -> list[str]:
    problems: list[str] = []
    items = [q for q in questions if isinstance(q, dict)]
    problems += [f"item {n}: not a mapping: {q!r}"[:200] for n, q in enumerate(questions, 1)
                 if not isinstance(q, dict)]
    seen: Counter[str] = Counter(str(q.get("id")) for q in items if "id" in q)
    problems += [f"{i}: duplicate id ({n} times)" for i, n in seen.items() if n > 1]
    for q in items:
        problems += _validate_one(q, corpus_ids, changed_ids, corpus_vendors)
    counts = Counter(q.get("type") for q in items if q.get("origin") == "new")
    for t, want in TARGET_COUNTS.items():
        if counts[t] != want:
            problems.append(f"new {t} questions: {counts[t]}, target {want}")
    return problems


def _load_corpus(corpus: Path) -> tuple[set[str], set[str]]:
    """(article ids, vendors) of the corpus."""
    rows = json.loads(corpus.read_text(encoding="utf-8"))
    return ({str(r["article_id"]) for r in rows},
            {str(r["vendor"]) for r in rows if r.get("vendor")})


def _load_changed(report: Path) -> set[str]:
    data: Any = json.loads(report.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("changed_since_graph"), list):
        raise ValueError(f"{report}: no changed_since_graph list")
    return {str(e).split(":", 1)[0].strip() for e in data["changed_since_graph"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", type=Path, required=True, metavar="PATH")
    ap.add_argument("--corpus", type=Path, default=Path(DEFAULT_CORPUS))
    ap.add_argument("--ingest-report", type=Path, default=Path(DEFAULT_INGEST_REPORT))
    args = ap.parse_args(argv)
    for p in (args.corpus, args.ingest_report):
        if not p.exists():
            print(f"missing required input: {p} (cannot check without it)", file=sys.stderr)
            return 2
    try:
        questions = load_questions(args.check)
        corpus_ids, corpus_vendors = _load_corpus(args.corpus)
        changed = _load_changed(args.ingest_report)
    except (ValueError, KeyError, TypeError, yaml.YAMLError) as e:  # bad input file
        print(f"cannot check: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    problems = validate(questions, corpus_ids, changed, corpus_vendors=corpus_vendors)
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
