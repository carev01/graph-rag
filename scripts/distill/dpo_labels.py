"""dpo4 labelling by Claude Code subagents (spec 2026-10-24 section 3). No API spend.

    uv run --extra dev python -m scripts.distill.dpo_labels tasks --kind label \\
        --capture CAP.jsonl [--capture ...] --store D/store/label-sonnet [--ids ids.json]
    uv run --extra dev python -m scripts.distill.dpo_labels batches --store S --size 10 \\
        --out D/batches/<store>-round-N
    #  a subagent answers each batch-NNN.jsonl into batch-NNN.answers.jsonl
    #  (scripts/distill/DPO_LABEL_BRIEF.md)
    uv run --extra dev python -m scripts.distill.dpo_labels intake --store S \\
        --answers D/batches/<store>-round-N
    uv run --extra dev python -m scripts.distill.dpo_labels calibration-sample ...
    uv run --extra dev python -m scripts.distill.dpo_labels calibrate ...

Three task kinds, one AnswerStore each (and per labelling model):
- `label`: every edge of a reply -> stated / previous_only / unsupported (+ category), a
  quote from CURRENT MESSAGE for `stated`. Sonnet if the calibration passes, else Opus.
- `recheck`: only the edges labelled unsupported, first label hidden -> supported /
  unsupported. Opus, a fresh subagent.
- `missed`: the original extraction prompt plus ckpt100's reply -> the edges it missed,
  each with a quote. Opus.

A task is keyed sha256("<kind>:<call_id>"). An answer that fails validation is re-asked
once (its batch line then carries `rejection` and `previous_answer`), then dropped and
logged -- never guessed. Quotes are not checked here: `dpo_coverage` does that, and an
unverified `stated` edge simply cannot be B's dropped edge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.distill.dpo_coverage import classify, current_message
from scripts.distill.dpo_pairs import (CATEGORIES, VERDICTS, edge_records, fact_line,
                                       serialise, tag)
from scripts.distill.teacher import AnswerStore

KINDS = ("label", "recheck", "missed")
EDGE_KEYS = ("source_entity_name", "target_entity_name", "relation_type", "fact")

LABEL_INSTRUCTIONS = """You are auditing knowledge-graph facts extracted from ONE chunk of a vendor
documentation page. CURRENT MESSAGE is the chunk. PREVIOUS MESSAGES are earlier chunks of
the same page, given only as context. DOCUMENT CONTEXT names the documented product.

For EVERY numbered fact below, in order, decide:
- "stated": the CURRENT MESSAGE states it, or implies it unambiguously (resolving a
  reference with PREVIOUS MESSAGES or DOCUMENT CONTEXT is fine). Give "quote": the
  shortest span of the CURRENT MESSAGE that states it, copied character for character
  (a few words at least -- never just an entity name).
- "previous_only": the claim is supported only by the PREVIOUS MESSAGES, not the CURRENT MESSAGE.
- "unsupported": neither supports it. Be strict: plausible general knowledge the text does
  not state is unsupported.
For an unsupported fact give a category:
- "over_inference": a claim the text does not make (generalisation, assumed capability, cause).
- "subject_substitution": a claim the text makes about something else, attributed to the wrong
  product, component or entity.
- "taxonomy_as_fact": a list/menu/heading membership turned into a relationship the text does not assert.
- "wrong_detail": right subject and gist, but a wrong number, version, condition or limit.
- "other".

Return only JSON: {"labels": [{"n": <fact number>, "verdict": "...", "category": "..." or null,
"quote": "..." or null, "why": "<= 15 words"}]} with exactly one entry per fact, in order.
"""

RECHECK_INSTRUCTIONS = """Each numbered fact below was extracted from ONE chunk of a vendor documentation
page (CURRENT MESSAGE). PREVIOUS MESSAGES are earlier chunks of the same page; DOCUMENT
CONTEXT names the documented product.

For EVERY fact decide:
- "supported": the CURRENT MESSAGE or the PREVIOUS MESSAGES state it, or imply it
  unambiguously.
- "unsupported": they do not. Be strict: plausible general knowledge the text does not
  state is unsupported. Give a category: "over_inference", "subject_substitution",
  "taxonomy_as_fact", "wrong_detail" or "other".

Return only JSON: {"labels": [{"n": <fact number>, "verdict": "supported" or "unsupported",
"category": "..." or null, "why": "<= 15 words"}]} with exactly one entry per fact, in order.
"""

MISSED_INSTRUCTIONS = """Below are an extraction prompt (SYSTEM and USER) and the EXISTING REPLY a
smaller model gave to it. List the edges the prompt asks for that the CURRENT MESSAGE
states but the EXISTING REPLY misses.

- Follow every rule in the prompt (naming, the documented product, relation direction,
  the fields and their formats).
- Only edges between two entities in the prompt's ENTITIES list, names copied exactly.
- Skip any pair of entities the EXISTING REPLY already connects, in either direction.
- Each edge has the prompt's edge fields plus "quote": the span of the CURRENT MESSAGE
  that states it, copied character for character (a few words at least).
- Invent nothing. When unsure, leave it out. Nothing missing is a fine answer.

Return only JSON: {"edges": [...]} -- [] when nothing is missing.
"""


def task_key(kind: str, call_id: str) -> str:
    return hashlib.sha256(f"{kind}:{call_id}".encode()).hexdigest()


def _check(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


def _context_prompt(instructions: str, rec: dict, facts: str) -> str:
    u = rec["messages"][1]["content"]
    return "\n\n".join([instructions, "PREVIOUS MESSAGES:\n" + tag(u, "PREVIOUS_MESSAGES"),
                        "CURRENT MESSAGE:\n" + tag(u, "CURRENT_MESSAGE"), rec["context"],
                        "FACTS:\n" + facts])


def label_request(rec: dict) -> dict:
    edges = rec["payload"]["edges"]
    order = list(range(len(edges)))
    random.Random(f"{rec['id']}:label").shuffle(order)
    facts = "\n".join(f"{n}. {fact_line(edges[i])}" for n, i in enumerate(order, 1))
    return {"kind": "label", "call_id": rec["id"], "order": order,
            "prompt": _context_prompt(LABEL_INSTRUCTIONS, rec, facts)}


def recheck_request(rec: dict, idxs: list[int]) -> dict:
    shown = list(idxs)
    random.Random(f"{rec['id']}:recheck").shuffle(shown)
    edges = rec["payload"]["edges"]
    facts = "\n".join(f"{n}. {fact_line(edges[i])}" for n, i in enumerate(shown, 1))
    return {"kind": "recheck", "call_id": rec["id"], "edges": shown,
            "prompt": _context_prompt(RECHECK_INSTRUCTIONS, rec, facts)}


def missed_request(rec: dict) -> dict:
    system, user = rec["messages"]
    prompt = "\n\n".join([MISSED_INSTRUCTIONS, "=== SYSTEM ===\n" + system["content"],
                          "=== USER ===\n" + user["content"],
                          "=== EXISTING REPLY ===\n" + serialise(rec["payload"])])
    return {"kind": "missed", "call_id": rec["id"], "prompt": prompt}


def _numbered(answer: Any, slots: list[int], verdicts: set[str],
              unsupported_needs_category: bool) -> tuple[dict[int, dict] | None, str]:
    """`slots[n-1]` = the edge index fact n shows. Returns {edge index: label}."""
    labels = answer.get("labels") if isinstance(answer, dict) else None
    if not isinstance(labels, list) or len(labels) != len(slots):
        return None, f"expected exactly {len(slots)} labels, one per fact"
    out: dict[int, dict] = {}
    for lab in labels:
        n = lab.get("n") if isinstance(lab, dict) else None
        if not isinstance(n, int) or not 1 <= n <= len(slots) or slots[n - 1] in out:
            return None, f"bad or repeated fact number {n!r}"
        v = lab.get("verdict")
        if v not in verdicts:
            return None, f"fact {n}: verdict {v!r} is not one of {sorted(verdicts)}"
        if unsupported_needs_category and v == "unsupported" and lab.get("category") not in CATEGORIES:
            return None, f"fact {n}: an unsupported fact needs a category from {sorted(CATEGORIES)}"
        out[slots[n - 1]] = lab
    return out, ""


def normalise(kind: str, request: dict, answer: Any) -> tuple[Any | None, str]:
    """The answer in its stored shape, or (None, why it was rejected)."""
    if kind == "label":
        got, why = _numbered(answer, request["order"], VERDICTS, True)
        if got is None:
            return None, why
        for n, i in enumerate(request["order"], 1):
            if got[i]["verdict"] == "stated" and not isinstance(got[i].get("quote"), str):
                return None, f"fact {n}: a stated fact needs a quote"
        return [{"verdict": got[i]["verdict"],
                 "category": got[i].get("category") if got[i]["verdict"] == "unsupported" else None,
                 "quote": got[i].get("quote") if got[i]["verdict"] == "stated" else None,
                 "why": got[i].get("why")} for i in range(len(request["order"]))], ""
    if kind == "recheck":
        got, why = _numbered(answer, request["edges"], {"supported", "unsupported"}, True)
        if got is None:
            return None, why
        return {str(i): {"verdict": lab["verdict"],
                         "category": lab.get("category") if lab["verdict"] == "unsupported" else None,
                         "why": lab.get("why")} for i, lab in sorted(got.items())}, ""
    if kind == "missed":
        edges = answer.get("edges") if isinstance(answer, dict) else None
        if not isinstance(edges, list):
            return None, 'expected {"edges": [...]}'
        for n, e in enumerate(edges, 1):
            if not isinstance(e, dict) or not all(isinstance(e.get(k), str) for k in EDGE_KEYS):
                return None, f"edge {n}: needs string fields {list(EDGE_KEYS)}"
            if not isinstance(e.get("quote"), str):
                return None, f"edge {n}: needs a quote from the CURRENT MESSAGE"
        return edges, ""
    raise ValueError(f"unknown task kind {kind!r}")


def write_batches(store: AnswerStore, out_dir: Path, *, size: int) -> list[Path]:
    """`out_dir/batch-NNN.jsonl`: every pending task, one kind per file, `size` per file,
    each line with its 1-based `ordinal` and `check` (12 hex chars of the prompt's sha256)."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.glob("batch-*.jsonl")):
        raise FileExistsError(f"{out_dir} already holds batches -- use a new round directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    by_kind: dict[str, list[tuple[str, dict]]] = {}
    for key, req in sorted(store.pending()):
        by_kind.setdefault(req["kind"], []).append((key, req))
    paths: list[Path] = []
    for kind in sorted(by_kind):
        rows = by_kind[kind]
        for start in range(0, len(rows), size):
            path = out_dir / f"batch-{len(paths) + 1:03d}.jsonl"
            with path.open("w", encoding="utf-8") as fh:
                for ordinal, (key, req) in enumerate(rows[start:start + size], 1):
                    line = {"key": key, "ordinal": ordinal, "check": _check(req["prompt"]),
                            "kind": kind, "prompt": req["prompt"]}
                    for extra in ("rejection", "previous_answer"):
                        if extra in req:
                            line[extra] = req[extra]
                    fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            paths.append(path)
    return paths


def _reject(store: AnswerStore, key: str, request: dict, answer: Any, reason: str,
            counts: Counter) -> None:
    if "rejection" in request:
        store.drop(key, reason, prompt_name=f"dpo_{request['kind']}")
        counts["dropped"] += 1
        return
    store.unqueue(key)
    store.add_pending(key, {**request, "rejection": reason, "previous_answer": answer})
    counts["reasked"] += 1


def intake(store: AnswerStore, answers_dir: Path) -> Counter:
    """Read every `batch-NNN.answers.jsonl` under `answers_dir`: an answer is stored only
    under a key of its own batch file, with that line's ordinal and check echoed."""
    counts: Counter = Counter()
    pending = dict(store.pending())
    for ans_path in sorted(Path(answers_dir).glob("batch-*.answers.jsonl")):
        batch_path = ans_path.with_name(ans_path.name.replace(".answers.jsonl", ".jsonl"))
        lines = {}
        for raw in batch_path.read_text(encoding="utf-8").splitlines():
            ln = json.loads(raw)
            lines[ln["key"]] = ln
        for raw in ans_path.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                counts["unparseable_line"] += 1
                continue
            key = row.get("key") if isinstance(row, dict) else None
            if not isinstance(key, str):
                counts["unparseable_line"] += 1
                continue
            if key not in lines:
                counts["unknown_key"] += 1
                continue
            if store.get(key) is not None or key not in pending:
                counts["not_pending"] += 1
                continue
            request = pending.pop(key)
            if (row.get("ordinal"), row.get("check")) != (lines[key]["ordinal"], lines[key]["check"]):
                _reject(store, key, request, row.get("answer"),
                        "answer filed under the wrong prompt (ordinal/check do not match)", counts)
                continue
            answer = row.get("answer")
            if isinstance(answer, str):
                try:
                    answer = json.loads(answer)
                except json.JSONDecodeError:
                    pass
            stored, why = normalise(request["kind"], request, answer)
            if stored is None:
                _reject(store, key, request, answer, why, counts)
                continue
            store.put(key, json.dumps(stored, ensure_ascii=False),
                      {"kind": request["kind"], "call_id": request["call_id"]})
            counts["stored"] += 1
    return counts


def read_records(paths: list[Path]) -> list[dict]:
    """The judgeable student edge replies of every capture, one per call_id."""
    lines = [ln for p in paths for ln in Path(p).read_text().splitlines() if ln.strip()]
    recs, _ = edge_records(lines)
    seen: dict[str, dict] = {}
    for r in recs:
        seen.setdefault(r["id"], r)
    return list(seen.values())


def load_answers(store: AnswerStore, kind: str, ids: list[str]) -> dict[str, Any]:
    out = {}
    for cid in ids:
        raw = store.get(task_key(kind, cid))
        if raw is not None:
            out[cid] = json.loads(raw)
    return out


def load_recheck(store: AnswerStore, ids: list[str]) -> dict[str, dict[int, dict]]:
    return {cid: {int(i): lab for i, lab in ans.items()}
            for cid, ans in load_answers(store, "recheck", ids).items()}


def deepseek_verdicts(pairs_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """dpo3's two DeepSeek passes (`dpo_pairs` judged-1/2.jsonl), per edge (pass 1, pass 2)."""
    passes = []
    for n in (1, 2):
        rows = {}
        for raw in (Path(pairs_dir) / f"judged-{n}.jsonl").read_text().splitlines():
            r = json.loads(raw)
            rows[r["id"]] = [lab["verdict"] for lab in r["labels"]]
        passes.append(rows)
    return {cid: list(zip(v1, passes[1][cid])) for cid, v1 in passes[0].items()
            if cid in passes[1]}


def _seeded_ids(ids: list[str], seed: str) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256(f"{seed}:{i}".encode()).hexdigest())


def calibration_ids(recs: list[dict], ds: dict[str, list[tuple[str, str]]], n: int,
                    seed: str) -> list[str]:
    """n/2 replies with a DeepSeek both-pass unsupported edge, the rest without; a short
    half is filled from the other."""
    ids = [r["id"] for r in recs if r["id"] in ds]
    flagged = [i for i in ids if ("unsupported", "unsupported") in ds[i]]
    flagged_set = set(flagged)
    other = [i for i in ids if i not in flagged_set]
    flagged, other = _seeded_ids(flagged, f"cal-f:{seed}"), _seeded_ids(other, f"cal-o:{seed}")
    take_f = min(len(flagged), n // 2)
    take_o = min(len(other), n - take_f)
    take_f = min(len(flagged), n - take_o)
    return flagged[:take_f] + other[:take_o]


def calibrate(ids: list[str], sonnet: dict[str, list[dict]],
              opus: dict[str, list[dict]]) -> dict:
    """Recall of Opus `unsupported` edges by Sonnet; the labeller is sonnet at >= 0.90."""
    matrix: Counter = Counter()
    for cid in ids:
        if cid not in sonnet or cid not in opus:
            continue
        for o, s in zip(opus[cid], sonnet[cid]):
            matrix[f"{o['verdict']}|{s['verdict']}"] += 1
    opus_unsup = sum(v for k, v in matrix.items() if k.startswith("unsupported|"))
    if not opus_unsup:
        raise ValueError("Opus labelled no edge unsupported in the calibration sample -- "
                         "the 0.90 recall bar cannot decide")
    recall = matrix["unsupported|unsupported"] / opus_unsup
    passed = recall >= 0.90 - 1e-9
    return {"replies": sum(cid in sonnet and cid in opus for cid in ids),
            "opus_unsupported": opus_unsup, "recall": recall, "passed": passed,
            "labeller": "sonnet" if passed else "opus", "matrix": dict(sorted(matrix.items()))}


def recheck_targets(recs: list[dict], labels: dict[str, list[dict]]) -> dict[str, list[int]]:
    """call_id -> the edges its first label calls unsupported (replies with any)."""
    out: dict[str, list[int]] = {}
    for r in recs:
        idx = [i for i, lab in enumerate(labels.get(r["id"], []))
               if lab["verdict"] == "unsupported"]
        if idx:
            out[r["id"]] = idx
    return out


def missed_candidates(recs: list[dict], labels: dict[str, list[dict]],
                      rechecks: dict[str, dict[int, dict]], n: int, seed: str) -> list[str]:
    clean = [r["id"] for r in recs if r["id"] in labels
             and classify(labels[r["id"]], rechecks.get(r["id"], {}), current_message(r)).clean]
    return _seeded_ids(clean, f"missed:{seed}")[:n]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dpo_labels")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks", help="queue tasks of one kind into a store")
    t.add_argument("--kind", choices=KINDS, required=True)
    t.add_argument("--capture", type=Path, action="append", required=True)
    t.add_argument("--store", type=Path, required=True)
    t.add_argument("--ids", type=Path, help="JSON list of call_ids: queue only these")
    t.add_argument("--labels", type=Path, help="label store (recheck, missed)")
    t.add_argument("--recheck", type=Path, help="recheck store (missed)")
    t.add_argument("--n", type=int, default=600, help="missed: how many clean replies")
    t.add_argument("--seed", default="20261024")
    b = sub.add_parser("batches")
    b.add_argument("--store", type=Path, required=True)
    b.add_argument("--size", type=int, default=10)
    b.add_argument("--out", type=Path, required=True)
    i = sub.add_parser("intake")
    i.add_argument("--store", type=Path, required=True)
    i.add_argument("--answers", type=Path, required=True)
    c = sub.add_parser("calibration-sample")
    c.add_argument("--capture", type=Path, action="append", required=True)
    c.add_argument("--deepseek", type=Path, required=True, help="dpo3 pairs dir (judged-1/2)")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--n", type=int, default=100)
    c.add_argument("--seed", default="20261024")
    k = sub.add_parser("calibrate")
    k.add_argument("--ids", type=Path, required=True)
    k.add_argument("--sonnet", type=Path, required=True)
    k.add_argument("--opus", type=Path, required=True)
    k.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("status")
    s.add_argument("--store", type=Path, action="append", required=True)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = _parser().parse_args(argv)
    if a.cmd == "tasks":
        recs = read_records(a.capture)
        if a.ids:
            wanted = set(json.loads(a.ids.read_text()))
            recs = [r for r in recs if r["id"] in wanted]
        ids = [r["id"] for r in recs]
        store = AnswerStore(a.store)
        if a.kind == "label":
            todo = {r["id"]: label_request(r) for r in recs}
        else:
            if not a.labels:
                raise SystemExit("--labels is required for recheck and missed")
            labels = load_answers(AnswerStore(a.labels), "label", ids)
            if a.kind == "recheck":
                by_id = {r["id"]: r for r in recs}
                todo = {cid: recheck_request(by_id[cid], idx)
                        for cid, idx in recheck_targets(recs, labels).items()}
            else:
                if not a.recheck:
                    raise SystemExit("--recheck is required for missed")
                rechecks = load_recheck(AnswerStore(a.recheck), ids)
                unchecked = [cid for cid, idx in recheck_targets(recs, labels).items()
                             if cid not in rechecks]
                if unchecked:
                    raise SystemExit(f"{len(unchecked)} replies still await their re-check -- "
                                     "finish recheck before choosing missed tasks")
                by_id = {r["id"]: r for r in recs}
                todo = {cid: missed_request(by_id[cid])
                        for cid in missed_candidates(recs, labels, rechecks, a.n, a.seed)}
        queued = 0
        for cid, req in todo.items():
            key = task_key(a.kind, cid)
            if store.get(key) is None and store.dropped_reason(key) is None:
                store.add_pending(key, req)
                queued += 1
        print(f"{a.kind}: {len(recs)} replies, {len(todo)} tasks, {queued} queued in {a.store}")
    elif a.cmd == "batches":
        paths = write_batches(AnswerStore(a.store), a.out, size=a.size)
        print(f"{len(paths)} batch files in {a.out}")
    elif a.cmd == "intake":
        print(json.dumps(dict(intake(AnswerStore(a.store), a.answers))))
    elif a.cmd == "calibration-sample":
        ids = calibration_ids(read_records(a.capture), deepseek_verdicts(a.deepseek), a.n, a.seed)
        a.out.write_text(json.dumps(ids, indent=1))
        print(f"{len(ids)} calibration replies -> {a.out}")
    elif a.cmd == "calibrate":
        ids = json.loads(a.ids.read_text())
        res = calibrate(ids, load_answers(AnswerStore(a.sonnet), "label", ids),
                        load_answers(AnswerStore(a.opus), "label", ids))
        a.out.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
    elif a.cmd == "status":
        for root in a.store:
            st = AnswerStore(root)
            print(f"{root}: answered {len(st.answered_keys())}, pending {len(st.pending())}, "
                  f"dropped {len(st.dropped_keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
