"""Score the head-to-head judgements with the pre-stated decision rule (spec §7).

Reads every `<run>/pass-*/tasks.jsonl` and `<run>/pass-*/judgements/*.jsonl`, maps the
anonymous labels back to graph/vendorkb, validates each judgement, combines the passes,
applies the decision rule and builds the reported tables. Writes `<run>/result.json`.

Exit status: 1 if any judgement of pass 1 or 2 is missing or invalid (listed by key);
3 if the passes are valid but some questions still need a third pass (the decision printed
is then PROVISIONAL); 0 when complete.

Run: uv run --extra dev python -m scripts.headtohead.score --run eval/headtohead/runs/<date>
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.headtohead.judge_context import SIDES, needs_judge, read_answers

LABELS = ("A", "B")
ALPHA = 0.05


def perfect(rec: dict, qtype: str) -> bool:
    """Spec §7 "Perfect answer": every gold fact present; no unsupported claim; every
    citation supporting; for unanswerable questions: abstained, with no invented answer.
    An errored answer (`{"errored": True}`, the run recorded no answer) is never perfect."""
    if rec.get("errored"):
        return False
    claims = rec.get("claims") or []
    if qtype == "unanswerable":
        return bool(rec["abstained"]) and not claims
    return (all(rec["gold"]) and all(c["supported"] for c in claims)
            and bool(rec["citations_ok"]) and not rec["abstained"])


def _check_record(rec: Any, n_gold: int) -> str | None:
    if not isinstance(rec, dict):
        return "record is not an object"
    gold = rec.get("gold")
    if not isinstance(gold, list) or not all(isinstance(g, bool) for g in gold):
        return "gold must be a list of booleans"
    if len(gold) != n_gold:
        return f"gold has {len(gold)} entries, the question has {n_gold} gold facts"
    claims = rec.get("claims")
    if not isinstance(claims, list) or not all(
            isinstance(c, dict) and isinstance(c.get("text"), str)
            and isinstance(c.get("supported"), bool) for c in claims):
        return "claims must be a list of {text: str, supported: bool}"
    for k in ("citations_ok", "abstained"):
        if not isinstance(rec.get(k), bool):
            return f"{k} must be a boolean"
    return None


def validate_judgement(j: dict, task: dict) -> str | None:
    """None if valid, else a problem naming the key. Both labels must be present with
    `len(gold) == len(task["gold_facts"])`; a label whose answer errored needs no record
    (it is scored not perfect without a judge)."""
    key = task["key"]
    if j.get("key") != key:
        return f"{key}: judgement key {j.get('key')!r} does not match"
    errored = task.get("errored") or {}
    for label in LABELS:
        if errored.get(label):
            continue
        if label not in j:
            return f"{key}: label {label} missing"
        err = _check_record(j[label], len(task["gold_facts"]))
        if err:
            return f"{key}: label {label}: {err}"
    return None


def _verdict(p: Any) -> bool | None:
    if isinstance(p, dict):
        p = p.get("perfect")
    return p if isinstance(p, bool) else None


def combine_passes(passes: Sequence[bool | dict | None]) -> bool | None:
    """The majority of the available verdicts (bool, or a dict with "perfect"); None with
    fewer than 2 available or no majority (two passes that disagree: a third is needed)."""
    vs = [v for v in (_verdict(p) for p in passes) if v is not None]
    if len(vs) < 2:
        return None
    yes = sum(vs)
    no = len(vs) - yes
    if yes == no:
        return None
    return yes > no


def needs_third_pass(p1: bool, p2: bool) -> bool:
    """Spec §7: passes 1 and 2 disagree on this answer's "perfect" verdict."""
    return p1 != p2


def sign_test_p(wins_a: int, wins_b: int) -> float:
    """Exact two-sided binomial (sign) test at p=0.5 over n = wins_a + wins_b questions,
    capped at 1.0; n = 0 gives 1.0."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def decide(per_question: list[dict]) -> dict:
    """Spec §7, "Decision rule" (verbatim):

    - Applies to the 100 new questions only.
    - Compare the two systems question by question. A question is a win for the side whose
      answer is perfect when the other's is not.
    - The side with more wins wins, provided a two-sided sign test on the questions with a
      winner gives p < 0.05. For the expected numbers that is a margin of roughly 10
      questions.
    - Otherwise it is a **tie**, and a tie goes to the new design on cost and simplicity
      (my call, accepted by the user).

    The new design is vendorkb. Only `decides=True` questions count; both perfect or both
    not perfect is no win (excluded from the sign test). A deciding question without a
    combined verdict on both sides is counted in `incomplete`, not scored.
    """
    wins = {"graph": 0, "vendorkb": 0}
    incomplete = 0
    n = 0
    for q in per_question:
        if not q.get("decides"):
            continue
        g, v = q.get("graph"), q.get("vendorkb")
        if g is None or v is None:
            incomplete += 1
            continue
        n += 1
        if g and not v:
            wins["graph"] += 1
        elif v and not g:
            wins["vendorkb"] += 1
    p = sign_test_p(wins["graph"], wins["vendorkb"])
    winner = "tie"
    if p < ALPHA and wins["graph"] != wins["vendorkb"]:
        winner = "graph" if wins["graph"] > wins["vendorkb"] else "vendorkb"
    return {"wins_graph": wins["graph"], "wins_vendorkb": wins["vendorkb"], "p": p,
            "winner": winner, "decision": "vendorkb" if winner == "tie" else winner,
            "deciding_scored": n, "incomplete": incomplete}


# --- reported tables -------------------------------------------------------------------


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def _side_metrics(qs: list[dict], judgements: dict[str, list[dict]], side: str) -> dict:
    gold_present = gold_total = claims_ok = claims_total = 0
    cit_ok = abst_ok = recs = 0
    for q in qs:
        for per_pass in judgements.get(q["qid"], []):
            rec = per_pass.get(side)
            if rec is None:
                continue
            recs += 1
            n_gold = q.get("n_gold", 0)
            gold_total += n_gold
            if rec.get("errored"):
                continue  # no gold, no claims, citations not ok, abstention wrong
            gold_present += sum(rec["gold"])
            claims_total += len(rec["claims"])
            claims_ok += sum(c["supported"] for c in rec["claims"])
            cit_ok += bool(rec["citations_ok"])
            abst_ok += bool(rec["abstained"]) == (q["type"] == "unanswerable")
    hits = [q["hit"][side] for q in qs if q.get("hit", {}).get(side) is not None]
    lat = [q["latency"][side] for q in qs if q.get("latency", {}).get(side) is not None]
    cost = [q["cost"][side] for q in qs if q.get("cost", {}).get(side) is not None]
    verdicts = [q[side] for q in qs if q.get(side) is not None]
    return {
        "n": len(qs), "perfect": sum(verdicts), "perfect_rate": _ratio(sum(verdicts),
                                                                        len(verdicts)),
        "fact_recall": _ratio(gold_present, gold_total),
        "claim_precision": _ratio(claims_ok, claims_total),
        "citation_precision": _ratio(cit_ok, recs),  # share of answers with citations_ok
        "abstention_correct": _ratio(abst_ok, recs),
        "hit_rate": _ratio(sum(hits), len(hits)),
        "latency_mean_s": statistics.fmean(lat) if lat else None,
        "latency_median_s": statistics.median(lat) if lat else None,
        "cost_mean": statistics.fmean(cost) if cost else None,
        "errored": sum(1 for q in qs if q.get("errored", {}).get(side)),
    }


def report_tables(per_question: list[dict], judgements: dict[str, list[dict]]) -> dict:
    """Per type and overall, per side; separately for `decides=True` and the reported
    (existing) questions. `judgements` maps qid -> one {side: record} per judged pass
    (errored sides as {"errored": True}); every pass's records are pooled. Fact recall and
    claim precision are pooled counts; citation precision is the share of judged answers
    with citations_ok (the record format carries one verdict per answer); the retrieval
    hit rate excludes unanswerable questions."""
    out: dict[str, Any] = {}
    for name, sel in (("deciding", True), ("reported", False)):
        qs = [q for q in per_question if bool(q.get("decides")) is sel]
        table: dict[str, Any] = {}
        groups = {"all": qs} | {t: [q for q in qs if q["type"] == t]
                                for t in sorted({q["type"] for q in qs})}
        for g, gqs in groups.items():
            table[g] = {s: _side_metrics(gqs, judgements, s) for s in SIDES}
        out[name] = table
    return out


# --- I/O ------------------------------------------------------------------------------


def hit(row: dict, gold_ids: set[str]) -> bool:
    """A gold article among the cited sources' article ids (graph) or doc ids (vendorkb)."""
    ids: set[str] = set()
    for c in row.get("citations") or []:
        if row["side"] == "graph":
            ids |= {str(s.get("article_id")).lower() for s in c.get("sources") or []}
        else:
            ids.add(str(c.get("doc_id")).lower())
    return bool(ids & gold_ids)


def _read_jsonl(path: Path, problems: list[str], where: str) -> list[dict]:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            problems.append(f"{where}: {path.name} line {i} is not JSON")
            continue
        if isinstance(obj, dict):
            out.append(obj)
        else:
            problems.append(f"{where}: {path.name} line {i} is not an object")
    return out


def _pass_dirs(run_dir: Path) -> dict[int, Path]:
    out = {}
    for d in run_dir.glob("pass-*"):
        try:
            out[int(d.name.split("-", 1)[1])] = d
        except ValueError:
            continue
    return dict(sorted(out.items()))


def score_run(run_dir: Path, questions: list[dict]) -> tuple[dict, list[str]]:
    """Returns (result, problems). Problems are missing/invalid judgements of passes 1-2
    (by key) and malformed files; a missing pass-3 judgement for a question that needs one
    is not a problem but leaves the result incomplete."""
    problems: list[str] = []
    warnings: list[str] = []
    passes = _pass_dirs(run_dir)
    for n in (1, 2):
        if n not in passes:
            problems.append(f"pass {n}: directory {run_dir / f'pass-{n}'} missing")
    qmap = {str(q["id"]): q for q in questions}
    # verdicts[qid][side] = {pass_n: bool | None}; records[qid] = [{side: rec}, ...]
    verdicts: dict[str, dict[str, dict[int, bool | None]]] = {
        qid: {s: {} for s in SIDES} for qid in qmap}
    records: dict[str, list[dict]] = {qid: [] for qid in qmap}
    truncated = {s: 0 for s in SIDES}
    unresolvable = {s: 0 for s in SIDES}
    pass3_keys: set[str] = set()
    for n, pdir in passes.items():
        if n > 3:
            warnings.append(f"pass {n}: ignored (at most 3 passes)")
            continue
        tpath = pdir / "tasks.jsonl"
        if not tpath.exists():
            problems.append(f"pass {n}: tasks.jsonl missing")
            continue
        tasks = {t["key"]: t for t in _read_jsonl(tpath, problems, f"pass {n}")}
        if n == 3:
            pass3_keys = set(tasks)
        judged: dict[str, dict] = {}
        jdir = pdir / "judgements"
        for jf in sorted(jdir.glob("*.jsonl")) if jdir.exists() else []:
            for line in _read_jsonl(jf, problems, f"pass {n}"):
                if line.get("key") in tasks:
                    judged[line["key"]] = line  # a re-asked judgement (later file/line) wins
                else:
                    warnings.append(
                        f"pass {n}: judgement for unknown key {line.get('key')!r}")
        if n in (1, 2):
            missing_q = sorted(set(qmap) - set(tasks))
            problems += [f"pass {n}: {k}: no task" for k in missing_q]
        for key, task in tasks.items():
            if key not in qmap:
                warnings.append(f"pass {n}: task {key} is not in the questions")
                continue
            if n == 1:
                for s in SIDES:
                    truncated[s] += bool(task["truncated"][task["labels"][s]])
                    unresolvable[s] += task.get("unresolvable", {}).get(task["labels"][s], 0)
            j = judged.get(key)
            err = None
            if needs_judge(task):
                if j is None:
                    err = f"pass {n}: {key}: judgement missing"
                else:
                    v = validate_judgement(j, task)
                    err = f"pass {n}: {v}" if v else None
            if err:
                (problems if n in (1, 2) else warnings).append(err)
                for s in SIDES:
                    verdicts[key][s][n] = None
                continue
            per_side = {}
            for s in SIDES:
                label = task["labels"][s]
                if task["errored"][label]:
                    rec = {"errored": True}
                else:
                    assert j is not None  # validated above
                    rec = j[label]
                per_side[s] = rec
                verdicts[key][s][n] = perfect(rec, task["type"])
            records[key].append(per_side)

    latest = read_answers(run_dir / "answers.jsonl")
    per_question = []
    third: list[str] = []
    for qid, q in qmap.items():
        combined = {}
        for s in SIDES:
            vs = verdicts[qid][s]
            combined[s] = combine_passes([vs.get(1), vs.get(2), vs.get(3)])
        for s in SIDES:
            p1, p2 = verdicts[qid][s].get(1), verdicts[qid][s].get(2)
            if p1 is not None and p2 is not None and needs_third_pass(p1, p2):
                third.append(qid)
                break
        gold_ids = {str(g["article_id"]).lower() for g in q.get("gold_sources") or []}
        entry: dict[str, Any] = {
            "qid": qid, "type": q["type"], "decides": bool(q.get("decides")),
            "n_gold": len(q.get("gold_facts") or []), **combined,
            "passes": {s: verdicts[qid][s] for s in SIDES},
            "hit": {}, "latency": {}, "cost": {}, "errored": {}}
        for s in SIDES:
            row = latest.get((qid, s))
            if row is None:
                problems.append(f"answers: {qid}/{s} missing")
                continue
            entry["errored"][s] = row.get("answer") is None
            entry["latency"][s] = row.get("latency_s")
            entry["cost"][s] = (row.get("usage") or {}).get("cost")
            entry["hit"][s] = (None if q["type"] == "unanswerable"
                               else hit(row, gold_ids) if row.get("answer") is not None
                               else False)
        per_question.append(entry)
    pending = sorted(qid for qid in third
                     if any(verdicts[qid][s].get(3) is None for s in SIDES))
    complete = not problems and all(
        e["graph"] is not None and e["vendorkb"] is not None for e in per_question)
    result = {
        "complete": complete,
        "decision": decide(per_question),
        "needs_third_pass": sorted(third),
        "third_pass_pending": pending,
        "pass3_tasks": sorted(pass3_keys),
        "truncated_pass1": truncated,
        "unresolvable_citations_pass1": unresolvable,
        "tables": report_tables(per_question, records),
        "per_question": per_question,
        "problems": problems,
        "warnings": warnings,
    }
    return result, problems


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    return f"{v:.3f}" if isinstance(v, float) else str(v)


COLUMNS = ("n", "perfect", "fact_recall", "claim_precision", "citation_precision",
           "abstention_correct", "hit_rate", "latency_mean_s", "cost_mean", "errored")


def print_tables(tables: dict) -> None:
    for name, table in tables.items():
        print(f"\n== {name} ==")
        print("group/side".ljust(26) + "".join(c[:12].rjust(13) for c in COLUMNS))
        for g, sides in table.items():
            for s, m in sides.items():
                print(f"{g}/{s}".ljust(26) + "".join(_fmt(m[c]).rjust(13) for c in COLUMNS))


def main(argv: list[str] | None = None) -> int:
    from scripts.headtohead.questions import load_questions

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--questions", default="eval/headtohead/questions.yaml")
    a = ap.parse_args(argv)
    run_dir = Path(a.run)
    result, problems = score_run(run_dir, load_questions(Path(a.questions)))
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    for p in problems:
        print(f"PROBLEM {p}")
    for w in result["warnings"]:
        print(f"warning {w}")
    if result["needs_third_pass"]:
        print(f"needs a third pass: {', '.join(result['needs_third_pass'])}")
    if result["third_pass_pending"]:
        print(f"third pass still pending: {', '.join(result['third_pass_pending'])}")
    d = result["decision"]
    tag = "" if result["complete"] else "PROVISIONAL "
    print(f"\n{tag}decision: {d['decision']} (winner {d['winner']}; wins graph "
          f"{d['wins_graph']} vs vendorkb {d['wins_vendorkb']}, p={d['p']:.4f}, "
          f"scored {d['deciding_scored']}, incomplete {d['incomplete']})")
    print(f"truncated (pass 1): {result['truncated_pass1']}; unresolvable citations "
          f"(pass 1): {result['unresolvable_citations_pass1']}")
    print_tables(result["tables"])
    if problems:
        sys.exit(1)
    if not result["complete"]:
        sys.exit(3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
