"""Agreement of a served student's answers with the teacher's reference, per prompt type.

Reads one or more `eval_finetune.py --out` files (rows carrying `output` and `reference`)
and prints, per prompt type, how close the student is to the teacher:

- extract_text: entity-name P/R/F1 (case-insensitive), entity-type agreement on matches;
- extract_edges.edge: (source, target) pair P/R/F1, relation-type agreement on matched
  pairs, and the ratio of predicted to reference fact counts;
- dedupe_nodes.nodes: per-entity accuracy of duplicate_candidate_id, false merges (a
  duplicate where the teacher said none) and missed merges;
- dedupe_edges.resolve_edge: exact duplicate_facts set match, and agreement on whether
  there is any duplicate at all;
- extract_summaries_batch: entity-name P/R/F1 and mean word Jaccard of matched summaries.

An unusable output (no reply, bad JSON) counts as every reference item missed.

    uv run python -m scripts.eval_agreement new.json [old.json ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(txt: str | None) -> dict | None:
    if not txt:
        return None
    try:
        obj = json.loads(txt)
    except (TypeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _norm(name: Any) -> str:
    return " ".join(str(name or "").lower().split())


def _words(text: Any) -> set[str]:
    return set(str(text or "").lower().split())


def _prf(m: dict) -> None:
    tp, fp, fn = m["tp"], m["fp"], m["fn"]
    m["precision"] = tp / (tp + fp) if tp + fp else 0.0
    m["recall"] = tp / (tp + fn) if tp + fn else 0.0
    p, r = m["precision"], m["recall"]
    m["f1"] = 2 * p * r / (p + r) if p + r else 0.0


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def score(rows: list[dict]) -> dict[str, dict]:
    acc: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        name = row["prompt"]
        ref = _load(row.get("reference")) or {}
        out = _load(row.get("output"))
        m = acc[name]
        m["n"] += 1
        if out is None:
            m["unusable"] += 1
            out = {}
        if name == "extract_nodes.extract_text":
            r = {_norm(e.get("name")): e.get("entity_type_id") for e in ref.get("extracted_entities", [])}
            o = {_norm(e.get("name")): e.get("entity_type_id") for e in out.get("extracted_entities", [])}
            hit = r.keys() & o.keys()
            m["tp"] += len(hit)
            m["fp"] += len(o.keys() - r.keys())
            m["fn"] += len(r.keys() - o.keys())
            m["type_match"] += sum(1 for k in hit if r[k] == o[k])
        elif name == "extract_edges.edge":
            def pairs(d: dict) -> dict:
                return {(_norm(e.get("source_entity_name")), _norm(e.get("target_entity_name"))):
                        e.get("relation_type") for e in d.get("edges", [])}
            r, o = pairs(ref), pairs(out)
            hit = r.keys() & o.keys()
            m["tp"] += len(hit)
            m["fp"] += len(o.keys() - r.keys())
            m["fn"] += len(r.keys() - o.keys())
            m["type_match"] += sum(1 for k in hit if r[k] == o[k])
            m["ref_facts"] += len(ref.get("edges", []))
            m["out_facts"] += len(out.get("edges", []))
        elif name == "dedupe_nodes.nodes":
            r = {e.get("id"): e.get("duplicate_candidate_id") for e in ref.get("entity_resolutions", [])}
            o = {e.get("id"): e.get("duplicate_candidate_id") for e in out.get("entity_resolutions", [])}
            for k, want in r.items():
                got = o.get(k)
                m["entities"] += 1
                m["correct"] += got == want
                if want == -1 and got not in (-1, None):
                    m["false_merge"] += 1
                if want != -1 and got in (-1, None):
                    m["missed_merge"] += 1
        elif name == "dedupe_edges.resolve_edge":
            r = sorted(ref.get("duplicate_facts") or [])
            o = sorted(out.get("duplicate_facts") or [])
            m["exact_hits"] += r == o
            m["flag_hits"] += bool(r) == bool(o)
        elif name == "extract_nodes.extract_summaries_batch":
            r = {_norm(e.get("name")): e.get("summary") for e in ref.get("summaries", [])}
            o = {_norm(e.get("name")): e.get("summary") for e in out.get("summaries", [])}
            hit = r.keys() & o.keys()
            m["tp"] += len(hit)
            m["fp"] += len(o.keys() - r.keys())
            m["fn"] += len(r.keys() - o.keys())
            for k in hit:
                a, b = _words(r[k]), _words(o[k])
                m["jaccard_sum"] += _ratio(len(a & b), len(a | b))
    result: dict[str, dict] = {}
    for name, m in acc.items():
        m = dict(m)
        if "tp" in m or name in ("extract_nodes.extract_text", "extract_edges.edge",
                                 "extract_nodes.extract_summaries_batch"):
            for k in ("tp", "fp", "fn"):
                m.setdefault(k, 0.0)
            _prf(m)
        if name in ("extract_nodes.extract_text", "extract_edges.edge"):
            m["type_agree"] = _ratio(m.get("type_match", 0), m["tp"])
        if name == "extract_edges.edge":
            m["count_ratio"] = _ratio(m.get("out_facts", 0), m.get("ref_facts", 0))
        if name == "dedupe_nodes.nodes":
            m["accuracy"] = _ratio(m.get("correct", 0), m.get("entities", 0))
            m.setdefault("false_merge", 0)
            m.setdefault("missed_merge", 0)
        if name == "dedupe_edges.resolve_edge":
            m["exact"] = _ratio(m.get("exact_hits", 0), m["n"])
            m["dup_flag_agree"] = _ratio(m.get("flag_hits", 0), m["n"])
        if name == "extract_nodes.extract_summaries_batch":
            m["word_jaccard"] = _ratio(m.get("jaccard_sum", 0), m["tp"])
        m.setdefault("unusable", 0)
        result[name] = m
    return result


def _line(name: str, m: dict) -> str:
    base = f"n={int(m['n'])} unusable={int(m['unusable'])}"
    if name == "extract_nodes.extract_text":
        return f"{base} name P/R/F1={m['precision']:.2f}/{m['recall']:.2f}/{m['f1']:.2f} type_agree={m['type_agree']:.2f}"
    if name == "extract_edges.edge":
        return (f"{base} pair P/R/F1={m['precision']:.2f}/{m['recall']:.2f}/{m['f1']:.2f} "
                f"rel_agree={m['type_agree']:.2f} facts out/ref={m['count_ratio']:.2f}")
    if name == "dedupe_nodes.nodes":
        return (f"{base} accuracy={m['accuracy']:.2f} false_merge={int(m['false_merge'])} "
                f"missed_merge={int(m['missed_merge'])} (of {int(m.get('entities', 0))})")
    if name == "dedupe_edges.resolve_edge":
        return f"{base} exact={m['exact']:.2f} dup_flag_agree={m['dup_flag_agree']:.2f}"
    if name == "extract_nodes.extract_summaries_batch":
        return f"{base} name F1={m['f1']:.2f} word_jaccard={m['word_jaccard']:.2f}"
    return base


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for path in argv:
        print(f"== {path}")
        for name, m in sorted(score(json.loads(Path(path).read_text())).items()):
            print(f"  {name:40} {_line(name, m)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
