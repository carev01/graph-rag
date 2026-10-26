"""dpo4 pair construction (spec docs/superpowers/specs/2026-10-24-dpo-coverage-pairs-design.md
section 2). Three pair types from ckpt100's own `extract_edges.edge` replies, each differing
from the reply in one kind of change only:

  A  rejected = the reply,                         chosen = minus its confirmed-unsupported edges
  B  rejected = a clean reply minus one verified   chosen = the clean reply
               `stated` edge
  C  rejected = a clean reply,                     chosen = plus Opus's verified missed edges,
                                                            appended after its last edge

dpo3's pairs were all type A, so "write fewer edges" scored on every pair; `balance` keeps
the mean edge-count difference between chosen and rejected within +/-BAND per split so
length carries no signal.

Labels come from `dpo_labels` (Claude subagents). A "verified" edge carries a quote found
word for word in the prompt's CURRENT MESSAGE -- a code check, not a model's word. Pure:
the `build` CLI (task 3) does the I/O.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from scripts.distill.dpo_pairs import PROMPT, fact_line, serialise, split_of, tag

MIN_QUOTE_CHARS = 12   # a shorter "quote" (a bare entity name) proves nothing
BAND = 0.1             # |mean(len(chosen edges) - len(rejected edges))| per split
_WS = re.compile(r"\s+")


def norm_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def quote_ok(quote: Any, current: str) -> bool:
    """True when `quote` appears word for word in `current`, whitespace runs (including
    non-breaking spaces) counted as one space."""
    if not isinstance(quote, str):
        return False
    q = norm_ws(quote)
    return len(q) >= MIN_QUOTE_CHARS and q in norm_ws(current)


def current_message(rec: dict) -> str:
    return tag(rec["messages"][1]["content"], "CURRENT_MESSAGE")


def entity_names(rec: dict) -> set[str]:
    """Names in the prompt's ENTITIES block; empty when the block is missing or unparseable."""
    try:
        rows = json.loads(tag(rec["messages"][1]["content"], "ENTITIES"))
    except json.JSONDecodeError:
        return set()
    if not isinstance(rows, list):
        return set()
    return {r["name"] for r in rows if isinstance(r, dict) and isinstance(r.get("name"), str)}


@dataclass(frozen=True)
class ReplyLabels:
    verdicts: tuple[str, ...]    # first label per edge: stated / previous_only / unsupported
    verified: tuple[bool, ...]   # stated AND its quote is in CURRENT MESSAGE
    confirmed: tuple[int, ...]   # unsupported, and the Opus re-check agrees
    disputed: tuple[int, ...]    # unsupported, re-check says supported or never answered

    @property
    def clean(self) -> bool:
        return not self.confirmed and not self.disputed

    @property
    def has_unverified(self) -> bool:
        """A `stated` label whose quote is not in the chunk: the reply can never be a C
        pair (spec section 3)."""
        return any(v == "stated" and not ok for v, ok in zip(self.verdicts, self.verified))

    @property
    def c_eligible(self) -> bool:
        return self.clean and not self.has_unverified


def classify(labels: list[dict], recheck: dict[int, dict], current: str) -> ReplyLabels:
    verdicts = tuple(lab["verdict"] for lab in labels)
    verified = tuple(v == "stated" and quote_ok(lab.get("quote"), current)
                     for v, lab in zip(verdicts, labels))
    unsupported = [i for i, v in enumerate(verdicts) if v == "unsupported"]
    confirmed = tuple(i for i in unsupported
                      if recheck.get(i, {}).get("verdict") == "unsupported")
    disputed = tuple(i for i in unsupported if i not in confirmed)
    return ReplyLabels(verdicts, verified, confirmed, disputed)


def _pair(rec: dict, chosen: dict, rejected: dict, kind: str, labeller: str,
          changed: list[dict]) -> dict:
    return {"prompt_name": PROMPT, "response_format": rec["response_format"],
            "messages": rec["messages"], "chosen": serialise(chosen),
            "rejected": serialise(rejected),
            "meta": {"call_id": rec["id"], "group": rec["group"], "type": kind,
                     "labeller": labeller,
                     "delta": len(chosen["edges"]) - len(rejected["edges"]),
                     "changed": changed}}


def build_a(rec: dict, rl: ReplyLabels, labels: list[dict], labeller: str) -> dict | None:
    """chosen = the reply minus its confirmed-unsupported edges; disputed edges stay on both
    sides. None when nothing is confirmed or nothing would be left."""
    edges = rec["payload"]["edges"]
    bad = set(rl.confirmed)
    if not bad or len(bad) == len(edges):
        return None
    chosen = {**rec["payload"], "edges": [e for i, e in enumerate(edges) if i not in bad]}
    changed = [{"fact": fact_line(edges[i]), "category": labels[i].get("category"),
                "why": labels[i].get("why")} for i in sorted(bad)]
    pair = _pair(rec, chosen, rec["payload"], "A", labeller, changed)
    pair["meta"]["disputed"] = [fact_line(edges[i]) for i in rl.disputed]
    return pair


def build_b(rec: dict, rl: ReplyLabels, labels: list[dict], labeller: str) -> dict | None:
    """rejected = a clean reply minus one verified `stated` edge (seeded on call_id)."""
    if not rl.clean:
        return None
    candidates = [i for i, ok in enumerate(rl.verified) if ok]
    if len(candidates) < 2:
        return None
    drop = random.Random(f"B:{rec['id']}").choice(candidates)
    edges = rec["payload"]["edges"]
    rejected = {**rec["payload"], "edges": [e for i, e in enumerate(edges) if i != drop]}
    changed = [{"fact": fact_line(edges[drop]), "quote": labels[drop].get("quote")}]
    return _pair(rec, rec["payload"], rejected, "B", labeller, changed)


def _pair_key(e: dict) -> frozenset:
    return frozenset((e.get("source_entity_name"), e.get("target_entity_name")))


_DEFAULT_EDGE_KEYS = ("source_entity_name", "target_entity_name", "relation_type", "fact",
                      "valid_at", "invalid_at", "episode_indices")


def _edge_keys(rec: dict) -> list[str]:
    """The Edge properties in the call's schema order (the order ckpt100 writes them);
    the reply's own first edge, then the default seven, when the schema has none."""
    schema = ((rec.get("response_format") or {}).get("json_schema") or {}).get("schema") or {}
    props = ((schema.get("$defs") or {}).get("Edge") or {}).get("properties")
    if isinstance(props, dict) and props:
        return list(props)
    edges = rec["payload"]["edges"]
    return list(edges[0]) if edges else list(_DEFAULT_EDGE_KEYS)


def _most_common(values: list[Any]) -> Any:
    """The most frequent value (JSON-compared, first seen wins a tie); None for no values."""
    if not values:
        return None
    count = Counter(json.dumps(v, sort_keys=True) for v in values)
    top = max(count.values())
    for v in values:
        if count[json.dumps(v, sort_keys=True)] == top:
            return v


def rebuild_edge(rec: dict, e: dict) -> dict:
    """Opus's edge in ckpt100's shape: every schema property in schema order, `quote` and
    unknown keys dropped. `valid_at` and `episode_indices` copy the reply's own convention
    (ckpt100 writes the prompt's REFERENCE_TIME and [0]); an Opus `valid_at` whose date
    differs from the REFERENCE_TIME's is a date stated in the text and is kept."""
    reply = rec["payload"]["edges"]
    ref_date = tag(rec["messages"][1]["content"], "REFERENCE_TIME")[:10]
    reply_valid = [x["valid_at"] for x in reply if isinstance(x.get("valid_at"), str)]
    opus_valid = e.get("valid_at") if isinstance(e.get("valid_at"), str) else None
    if opus_valid is not None and ref_date and opus_valid[:10] != ref_date:
        valid_at = opus_valid
    elif reply_valid:
        valid_at = _most_common(reply_valid)
    else:
        valid_at = opus_valid
    episodes = _most_common([x["episode_indices"] for x in reply
                             if isinstance(x.get("episode_indices"), list)])
    values = {**e, "valid_at": valid_at,
              "invalid_at": e["invalid_at"] if isinstance(e.get("invalid_at"), str) else None,
              "episode_indices": episodes if episodes is not None else [0]}
    return {k: values.get(k) for k in _edge_keys(rec)}


def missed_edges(rec: dict, proposed: list[Any]) -> tuple[list[tuple[dict, str]], Counter]:
    """The Opus edges that may be appended, as (rebuilt edge, quote), and why the others
    were not: both entities listed, not a self-loop, no existing (or already added) edge
    between the same two entities in either direction, a verified quote, and a valid edge
    under the call's response schema (checked on the rebuilt edge)."""
    names = entity_names(rec)
    current = current_message(rec)
    schema = ((rec.get("response_format") or {}).get("json_schema") or {}).get("schema")
    have = {_pair_key(e) for e in rec["payload"]["edges"]}
    keep: list[tuple[dict, str]] = []
    why: Counter = Counter()
    for e in proposed:
        if not isinstance(e, dict):
            why["not_object"] += 1
            continue
        src, tgt = e.get("source_entity_name"), e.get("target_entity_name")
        edge = rebuild_edge(rec, e)
        if not isinstance(src, str) or not isinstance(tgt, str) \
                or src not in names or tgt not in names:
            why["entity_not_listed"] += 1
        elif src == tgt:
            why["self_loop"] += 1
        elif _pair_key(e) in have:
            why["pair_already_present"] += 1
        elif not quote_ok(e.get("quote"), current):
            why["quote_not_found"] += 1
        elif schema and not _valid(schema, {**rec["payload"], "edges": [edge]}):
            why["schema"] += 1
        else:
            have.add(_pair_key(e))
            keep.append((edge, e["quote"]))
    return keep, why


def _valid(schema: dict, payload: dict) -> bool:
    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError:
        return False
    return True


def build_c(rec: dict, rl: ReplyLabels, proposed: list[Any],
            labeller: str) -> tuple[dict | None, Counter]:
    """chosen = a clean reply plus its verified missed edges, appended after its last edge."""
    if not rl.clean:
        return None, Counter({"reply_not_clean": 1})
    if rl.has_unverified:
        return None, Counter({"reply_has_unverified": 1})
    added, why = missed_edges(rec, proposed)
    if not added:
        return None, why
    chosen = {**rec["payload"], "edges": [*rec["payload"]["edges"], *(e for e, _ in added)]}
    changed = [{"fact": fact_line(e), "quote": q} for e, q in added]
    return _pair(rec, chosen, rec["payload"], "C", labeller, changed), why


class BalanceError(ValueError):
    pass


def mean_delta(pairs: list[dict]) -> float:
    return sum(p["meta"]["delta"] for p in pairs) / len(pairs) if pairs else 0.0


def _seeded(rows: list[dict], seed: str) -> list[dict]:
    return sorted(rows, key=lambda p: hashlib.sha256(
        f"{seed}:{p['meta']['call_id']}".encode()).hexdigest())


def balance(a: list[dict], b_pool: list[dict], c: list[dict], *, seed: str,
            band: float = BAND, target: float = 0.0) -> list[dict]:
    """Every A pair; C pairs dropped in seeded order while the mean is above +band; then B
    candidates added in seeded order while it is below `target` (0: filling only to -band
    left chosen 0.1 edge shorter on average, a lean towards dpo3's "write less"). Raises
    when the band is still out of reach."""
    pairs = [*a, *c]
    c_order = _seeded(c, f"C:{seed}")
    while pairs and mean_delta(pairs) > band and c_order:
        gone = c_order.pop()["meta"]["call_id"]
        pairs = [p for p in pairs if p["meta"]["call_id"] != gone]
    b_order = _seeded(b_pool, f"B:{seed}")
    while mean_delta(pairs) < target and b_order:
        pairs.append(b_order.pop())
    if pairs and abs(mean_delta(pairs)) > band:
        raise BalanceError(f"mean edge delta {mean_delta(pairs):+.3f} outside +/-{band} with "
                           f"{len(pairs)} pairs: not enough B candidates or C pairs")
    return pairs


def val_eval_subset(val: list[dict], n: int, seed: str) -> list[dict]:
    """A fixed subset of `n` val pairs for the trainer's evals, each type in proportion
    (largest remainder), seeded. All of `val` when it has `n` or fewer."""
    if len(val) <= n:
        return list(val)
    by_type: dict[str, list[dict]] = {}
    for p in val:
        by_type.setdefault(p["meta"]["type"], []).append(p)
    quota = {t: n * len(rows) / len(val) for t, rows in by_type.items()}
    take = {t: int(q) for t, q in quota.items()}
    for t in sorted(quota, key=lambda t: (-(quota[t] - take[t]), t))[: n - sum(take.values())]:
        take[t] += 1
    out: list[dict] = []
    for t in sorted(by_type):
        out += _seeded(by_type[t], f"E:{seed}")[: take[t]]
    return out


def assemble(recs: list[dict], labels: dict[str, list[dict]],
             rechecks: dict[str, dict[int, dict]], missed: dict[str, list],
             labeller: str, val_fraction: float) -> tuple[dict, Counter, Counter]:
    """Each labelled reply gives at most one pair: A when an edge is confirmed unsupported;
    else, for a clean reply, C when its missed edges survive, otherwise a B candidate."""
    by_split = {s: {"A": [], "B": [], "C": []} for s in ("train", "val")}
    stats: Counter = Counter()
    c_why: Counter = Counter()
    for rec in recs:
        lab = labels.get(rec["id"])
        if lab is None:
            stats["unlabelled"] += 1
            continue
        if len(lab) != len(rec["payload"]["edges"]):
            stats["label_count_mismatch"] += 1
            continue
        rl = classify(lab, rechecks.get(rec["id"], {}), current_message(rec))
        bucket = by_split[split_of(rec["group"], val_fraction)]
        if rl.confirmed:
            pair = build_a(rec, rl, lab, labeller)
            if pair is None:
                stats["a_all_unsupported"] += 1
            else:
                bucket["A"].append(pair)
            continue
        if not rl.clean:
            stats["disputed_only"] += 1
            continue
        stats["clean"] += 1
        if rec["id"] in missed:
            pair, why = build_c(rec, rl, missed[rec["id"]], labeller)
            c_why.update(why)
            if pair is not None:
                bucket["C"].append(pair)
                continue
        pair = build_b(rec, rl, lab, labeller)
        if pair is None:
            stats["b_too_few_verified"] += 1
        else:
            bucket["B"].append(pair)
    return by_split, stats, c_why


def final_label(rl: ReplyLabels, i: int) -> str:
    if i in rl.confirmed:
        return "confirmed"
    if i in rl.disputed:
        return "disputed"
    return rl.verdicts[i]


def labels_report(recs: list[dict], labels: dict[str, list[dict]],
                  rechecks: dict[str, dict[int, dict]], c_why: Counter,
                  ds: dict[str, list[tuple[str, str]]] | None) -> dict:
    verdicts: Counter = Counter()
    finals: Counter = Counter()
    categories: Counter = Counter()
    stated = unverified = 0
    vs_ds: Counter = Counter()
    for rec in recs:
        lab = labels.get(rec["id"])
        if lab is None or len(lab) != len(rec["payload"]["edges"]):
            continue
        rl = classify(lab, rechecks.get(rec["id"], {}), current_message(rec))
        for i, x in enumerate(lab):
            verdicts[x["verdict"]] += 1
            finals[final_label(rl, i)] += 1
            if i in rl.confirmed:
                categories[x.get("category")] += 1
            if x["verdict"] == "stated":
                stated += 1
                unverified += not rl.verified[i]
        if ds is not None and rec["id"] in ds and len(ds[rec["id"]]) == len(lab):
            for i, (v1, v2) in enumerate(ds[rec["id"]]):
                both = "both_unsupported" if v1 == v2 == "unsupported" else (
                    "one_unsupported" if "unsupported" in (v1, v2) else "neither")
                vs_ds[f"{both}|{final_label(rl, i)}"] += 1
    report: dict = {"first_label": dict(verdicts), "final": dict(finals),
                    "confirmed_categories": dict(categories),
                    "stated_unverified_share": unverified / stated if stated else 0.0,
                    "missed_edges_rejected": dict(c_why)}
    if ds is not None:
        both = sum(v for k, v in vs_ds.items() if k.startswith("both_unsupported|"))
        report["deepseek"] = {"matrix": dict(sorted(vs_ds.items())),
                              "both_unsupported_edges": both,
                              "of_which_confirmed": vs_ds["both_unsupported|confirmed"]}
    return report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    from scripts.distill.dpo_labels import (deepseek_verdicts, load_answers, load_recheck,
                                            read_records)
    from scripts.distill.teacher import AnswerStore

    ap = argparse.ArgumentParser(prog="dpo_coverage")
    ap.add_argument("--capture", type=Path, action="append", required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--labeller", choices=("sonnet", "opus"), required=True)
    ap.add_argument("--recheck", type=Path, required=True)
    ap.add_argument("--missed", type=Path, required=True)
    ap.add_argument("--deepseek", type=Path, help="dpo3 pairs dir, for the comparison")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--eval-size", type=int, default=120)
    ap.add_argument("--seed", default="20261024")
    a = ap.parse_args(argv)
    recs = read_records(a.capture)
    ids = [r["id"] for r in recs]
    labels = load_answers(AnswerStore(a.labels), "label", ids)
    rechecks = load_recheck(AnswerStore(a.recheck), ids)
    missed = load_answers(AnswerStore(a.missed), "missed", ids)
    by_split, stats, c_why = assemble(recs, labels, rechecks, missed, a.labeller,
                                      a.val_fraction)
    out: dict[str, list[dict]] = {}
    for split, kinds in by_split.items():     # check both splits before writing anything
        if not kinds["A"]:
            raise SystemExit(f"no A pairs in {split} -- nothing to measure or train on")
        out[split] = balance(kinds["A"], kinds["B"], kinds["C"], seed=f"{a.seed}:{split}")
    a.out.mkdir(parents=True, exist_ok=True)
    for split, rows in out.items():
        _write_jsonl(a.out / f"{split}.jsonl", rows)
    val_eval = val_eval_subset(out["val"], a.eval_size, seed=a.seed)
    _write_jsonl(a.out / "val_eval.jsonl", val_eval)
    summary = {
        "replies": len(recs), "labelled": len(labels), "labeller": a.labeller, **stats,
        "candidates": {s: {k: len(v) for k, v in kinds.items()} for s, kinds in by_split.items()},
        "pairs": {s: dict(Counter(p["meta"]["type"] for p in rows)) for s, rows in out.items()},
        "mean_delta": {s: round(mean_delta(rows), 4) for s, rows in out.items()},
        "val_eval": dict(Counter(p["meta"]["type"] for p in val_eval)),
        "groups": {s: len({p["meta"]["group"] for p in rows}) for s, rows in out.items()},
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    ds = deepseek_verdicts(a.deepseek) if a.deepseek else None
    (a.out / "labels_report.json").write_text(
        json.dumps(labels_report(recs, labels, rechecks, c_why, ds), indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
