"""Fact-verifier dataset (spec docs/superpowers/specs/2026-10-27-fact-verifier-design.md
section 3). One chat record per ckpt100 `extract_edges.edge` reply:

  system     VERIFIER_INSTRUCTIONS (the labelling brief's definitions, shorter output)
  user       PREVIOUS MESSAGES + CURRENT MESSAGE + DOCUMENT CONTEXT + numbered FACTS,
             facts in reply order -- production builds the same text from the same call
  assistant  {"labels": [{"n", "verdict", "quote" | "category"}]}

Labels are Opus's final ones (`dpo_coverage.final_label`). Disputed facts and `stated`
facts whose quote is not in the chunk are left out of the FACTS list, so the model never
trains on an uncertain label. Split by article group: dpo4's val articles are the test
set; a seeded share of the rest is dev, written as val.jsonl.

    uv run --extra dev python -m scripts.distill.verifier_data \\
        --capture D3/capture-run/capture-v2.jsonl --capture D/capture-run/capture-v2.jsonl \\
        --labels D/store/label-opus --recheck D/store/recheck-opus --out D/verifier1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.distill.dpo_coverage import classify, current_message, final_label
from scripts.distill.dpo_labels import (LABEL_INSTRUCTIONS, load_answers, load_recheck,
                                        read_records)
from scripts.distill.dpo_pairs import fact_line, split_of, tag
from scripts.distill.teacher import AnswerStore

PROMPT_NAME = "verify_edges"
_OUTPUT = """Return only JSON: {"labels": [{"n": <fact number>, "verdict": "...", "category": "..."
(unsupported only), "quote": "..." (stated only)}]} with exactly one entry per fact, in order.
"""
VERIFIER_INSTRUCTIONS = LABEL_INSTRUCTIONS.split("Return only JSON:")[0] + _OUTPUT
SPLIT_FILES = {"train": "train.jsonl", "dev": "val.jsonl", "test": "test.jsonl"}


def verifier_user(rec: dict, facts: list[str]) -> str:
    u = rec["messages"][1]["content"]
    numbered = "\n".join(f"{n}. {f}" for n, f in enumerate(facts, 1))
    return "\n\n".join(["PREVIOUS MESSAGES:\n" + tag(u, "PREVIOUS_MESSAGES"),
                        "CURRENT MESSAGE:\n" + tag(u, "CURRENT_MESSAGE"), rec["context"],
                        "FACTS:\n" + numbered])


def build_record(rec: dict, labels: list[dict], recheck: dict[int, dict]) -> tuple[dict | None, Counter]:
    edges = rec["payload"]["edges"]
    if len(labels) != len(edges):
        return None, Counter({"label_count_mismatch": 1})
    current = current_message(rec)
    rl = classify(labels, recheck, current)
    why: Counter = Counter()
    keep: list[int] = []
    finals: list[str] = []
    entries: list[dict] = []
    for i in range(len(edges)):
        final = final_label(rl, i)
        if final == "disputed":
            why["disputed"] += 1
            continue
        if final == "stated" and not rl.verified[i]:
            why["stated_unverified"] += 1
            continue
        n = len(keep) + 1
        if final == "stated":
            entry = {"n": n, "verdict": "stated", "quote": labels[i]["quote"]}
        elif final == "confirmed":
            entry = {"n": n, "verdict": "unsupported",
                     "category": recheck.get(i, {}).get("category") or labels[i].get("category")}
        else:
            entry = {"n": n, "verdict": "previous_only"}
        keep.append(i)
        finals.append(final)
        entries.append(entry)
        why[final] += 1
    if not keep:
        why["no_facts_left"] += 1
        return None, why
    facts = [fact_line(edges[i]) for i in keep]
    record = {
        "prompt_name": PROMPT_NAME,
        "messages": [{"role": "system", "content": VERIFIER_INSTRUCTIONS},
                     {"role": "user", "content": verifier_user(rec, facts)},
                     {"role": "assistant",
                      "content": json.dumps({"labels": entries}, ensure_ascii=False)}],
        "meta": {"call_id": rec["id"], "group": rec["group"], "edge_index": keep,
                 "final": finals, "categories": [e.get("category") for e in entries],
                 "facts": facts, "current": current},
    }
    return record, why


def split_name(group: str, val_fraction: float, dev_fraction: float, seed: str) -> str:
    if split_of(group, val_fraction) == "val":
        return "test"
    h = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < dev_fraction else "train"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verifier_data")
    ap.add_argument("--capture", type=Path, action="append", required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--recheck", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--dev-fraction", type=float, default=0.05)
    ap.add_argument("--seed", default="20261027")
    a = ap.parse_args(argv)
    recs = read_records(a.capture)
    ids = [r["id"] for r in recs]
    labels = load_answers(AnswerStore(a.labels), "label", ids)
    rechecks = load_recheck(AnswerStore(a.recheck), ids)
    rows: dict[str, list[dict]] = {s: [] for s in SPLIT_FILES}
    facts: dict[str, Counter] = {s: Counter() for s in SPLIT_FILES}
    skipped: Counter = Counter()
    for rec in recs:
        lab = labels.get(rec["id"])
        if lab is None:
            skipped["unlabelled"] += 1
            continue
        record, why = build_record(rec, lab, rechecks.get(rec["id"], {}))
        split = split_name(rec["group"], a.val_fraction, a.dev_fraction, a.seed)
        facts[split].update(why)
        if record is None:
            skipped.update(k for k in why if k in ("label_count_mismatch", "no_facts_left"))
            continue
        rows[split].append(record)
    a.out.mkdir(parents=True, exist_ok=True)
    for split, name in SPLIT_FILES.items():
        with (a.out / name).open("w", encoding="utf-8") as fh:
            for r in rows[split]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"replies": len(recs), "records": {s: len(v) for s, v in rows.items()},
               "skipped": dict(skipped), "facts": {s: dict(c) for s, c in facts.items()},
               "groups": {s: len({r["meta"]["group"] for r in v}) for s, v in rows.items()}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
