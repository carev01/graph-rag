"""Fact-verifier scoring (spec docs/superpowers/specs/2026-10-27-fact-verifier-design.md
sections 5-6). Local, free: reads the dataset's test.jsonl and the notebook's predictions.

    uv run --extra dev python -m scripts.distill.verifier_eval --data D/verifier1 \\
        --predictions D/verifier1/test_predictions.jsonl --out D/verifier1/report.json

A reply that is not exactly the JSON the model was trained to write -- including an empty
reply, a code fence or trailing prose -- is a parse failure and keeps every fact
(production fails open). A missing reply counts the same way.

P1 drops a fact on "unsupported", or on "stated" with a quote that is not word for word in
the CURRENT MESSAGE. P2 drops on "unsupported" only. "previous_only" is kept by both.
Exit code 0 when a policy passes the bar, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from scripts.distill.dpo_coverage import norm_ws, quote_ok
from scripts.distill.dpo_pairs import VERDICTS

MIN_CATCH = 0.50
MAX_FALSE_DROP = 0.02
MAX_PARSE_FAIL = 0.01
POLICIES = ("P1", "P2")
_WORD = re.compile(r"[a-z0-9][a-z0-9.\-_/]*")
STOPWORDS = frozenset("""a an and are as at be by can for from has have in is it its may of on or
that the this to via was were when which with without you your not also into only use used uses
using supports support supported""".split())


def parse_reply(text: str, n: int) -> list[dict] | None:
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return None
    labels = obj.get("labels") if isinstance(obj, dict) else None
    if not isinstance(labels, list) or len(labels) != n:
        return None
    out: list[dict | None] = [None] * n
    for lab in labels:
        k = lab.get("n") if isinstance(lab, dict) else None
        if type(k) is not int or not 1 <= k <= n or out[k - 1] is not None:
            return None
        verdict = lab.get("verdict")
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            return None
        out[k - 1] = lab
    return [lab for lab in out if lab is not None]


def drops(parsed: list[dict] | None, current: str, policy: str, n: int) -> list[bool]:
    if parsed is None:
        return [False] * n
    out = []
    for lab in parsed:
        bad_quote = lab["verdict"] == "stated" and not quote_ok(lab.get("quote"), current)
        out.append(lab["verdict"] == "unsupported" or (policy == "P1" and bad_quote))
    return out


def read_predictions(path: Path) -> tuple[dict[str, str], int, int]:
    """(replies by call id, repeated ids, unusable lines). First reply wins. A line that
    does not parse or lacks call_id/reply -- a dead runtime's half-written tail -- is
    counted and skipped; its call id then counts as missing unless written again."""
    replies: dict[str, str] = {}
    dupes = bad = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if not (isinstance(row, dict) and isinstance(row.get("call_id"), str)
                and isinstance(row.get("reply"), str)):
            bad += 1
            continue
        if row["call_id"] in replies:
            dupes += 1
            continue
        replies[row["call_id"]] = row["reply"]
    return replies, dupes, bad


def _reply_counts(rec: dict, dropped: list[bool]) -> Counter:
    c: Counter = Counter()
    for final, cat, d in zip(rec["meta"]["final"], rec["meta"]["categories"], dropped):
        c[final] += 1
        c[f"{final}_dropped"] += d
        if final == "confirmed":
            c[f"cat:{cat}"] += 1
            c[f"cat:{cat}_dropped"] += d
    return c


def _share(c: Counter, key: str) -> float:
    return c[f"{key}_dropped"] / c[key] if c[key] else 0.0


def _ci95(xs: list[float]) -> list[float]:
    """2.5th and 97.5th percentile of a sorted list."""
    return [xs[int(0.025 * len(xs))], xs[min(int(0.975 * len(xs)), len(xs) - 1)]]


def score(records: list[dict], replies: dict[str, str], policy: str, resamples: int = 10_000,
          seed: int = 20261027) -> dict:
    per_reply: list[Counter] = []
    missing = failed = 0
    for rec in records:
        n = len(rec["meta"]["final"])
        text = replies.get(rec["meta"]["call_id"])
        parsed = parse_reply(text, n) if text is not None else None
        missing += text is None
        failed += parsed is None
        per_reply.append(_reply_counts(rec, drops(parsed, rec["meta"]["current"], policy, n)))
    tot: Counter = sum(per_reply, Counter())
    rng = random.Random(seed)
    catches, false_drops = [], []
    for _ in range(resamples):
        s: Counter = sum((per_reply[rng.randrange(len(per_reply))] for _ in per_reply), Counter())
        catches.append(_share(s, "confirmed"))
        false_drops.append(_share(s, "stated"))
    catches.sort()
    false_drops.sort()
    cats = sorted(k[4:] for k in tot if k.startswith("cat:") and not k.endswith("_dropped"))
    return {
        "policy": policy, "replies": len(records),
        "catch": _share(tot, "confirmed"), "catch_ci": _ci95(catches),
        "false_drop": _share(tot, "stated"), "false_drop_ci": _ci95(false_drops),
        "previous_only_dropped": _share(tot, "previous_only"),
        "parse_fail": failed / len(records) if records else 0.0,
        "catch_by_category": {c: _share(tot, f"cat:{c}") for c in cats},
        "counts": {"confirmed": tot["confirmed"], "stated": tot["stated"],
                   "previous_only": tot["previous_only"], "missing": missing,
                   "parse_failed": failed},
    }


def passes(m: dict) -> tuple[bool, list[str]]:
    why = []
    if m["catch"] < MIN_CATCH:
        why.append(f"catch {m['catch']:.3f} < {MIN_CATCH}")
    if m["false_drop"] > MAX_FALSE_DROP:
        why.append(f"false drops {m['false_drop']:.3f} > {MAX_FALSE_DROP}")
    if m["parse_fail"] > MAX_PARSE_FAIL:
        why.append(f"parse failures {m['parse_fail']:.3f} > {MAX_PARSE_FAIL}")
    return not why, why


def _words(text: str) -> set[str]:
    return {w.rstrip(".").removesuffix("s") for w in _WORD.findall(norm_ws(text).lower())}


def lexical_drops(record: dict, threshold: float) -> list[bool]:
    """Code-only baseline: drop a fact when under `threshold` of its distinctive words
    (outside its two entity names and the stopwords) occur in the CURRENT MESSAGE."""
    current = _words(record["meta"]["current"])
    out = []
    for fact in record["meta"]["facts"]:
        names, _, text = fact.partition(": ")
        words = _words(text) - _words(names.replace(" -> ", " ")) - STOPWORDS
        out.append(bool(words) and len(words & current) / len(words) < threshold)
    return out


def _lexical_totals(records: list[dict], threshold: float) -> Counter:
    return sum((_reply_counts(r, lexical_drops(r, threshold)) for r in records), Counter())


def tune_threshold(records: list[dict]) -> float:
    """The threshold with the best catch whose false drops stay within the bar (train)."""
    best, best_catch = 0.0, -1.0
    for t in (i / 20 for i in range(21)):
        tot = _lexical_totals(records, t)
        if _share(tot, "stated") <= MAX_FALSE_DROP and _share(tot, "confirmed") > best_catch:
            best, best_catch = t, _share(tot, "confirmed")
    return best


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="verifier_eval")
    ap.add_argument("--data", type=Path, required=True, help="dir with train.jsonl, test.jsonl")
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resamples", type=int, default=10_000)
    a = ap.parse_args(argv)
    test = _read_jsonl(a.data / "test.jsonl")
    replies, dupes, bad = read_predictions(a.predictions)
    policies = {p: score(test, replies, p, resamples=a.resamples) for p in POLICIES}
    verdicts = {p: passes(m) for p, m in policies.items()}
    passing = [p for p in POLICIES if verdicts[p][0]]
    threshold = tune_threshold(_read_jsonl(a.data / "train.jsonl"))
    base = _lexical_totals(test, threshold)
    report = {
        "passed": bool(passing), "policy": passing[0] if passing else None,
        "policies": {p: {**m, "passed": verdicts[p][0], "why": verdicts[p][1]}
                     for p, m in policies.items()},
        "duplicate_predictions": dupes, "unusable_prediction_lines": bad,
        "baseline": {"threshold": threshold, "catch": _share(base, "confirmed"),
                     "false_drop": _share(base, "stated")},
        "bar": {"MIN_CATCH": MIN_CATCH, "MAX_FALSE_DROP": MAX_FALSE_DROP,
                "MAX_PARSE_FAIL": MAX_PARSE_FAIL},
    }
    a.out.write_text(json.dumps(report, indent=2))
    for p, m in policies.items():
        print(f"{p}: catch {m['catch']:.3f} {m['catch_ci']}, false drops {m['false_drop']:.3f} "
              f"{m['false_drop_ci']}, parse failures {m['parse_fail']:.3f} -> "
              f"{'PASS' if verdicts[p][0] else 'FAIL ' + '; '.join(verdicts[p][1])}")
    print(f"baseline (threshold {threshold}): catch {report['baseline']['catch']:.3f}, "
          f"false drops {report['baseline']['false_drop']:.3f}")
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
