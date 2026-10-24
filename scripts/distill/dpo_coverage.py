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

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import jsonschema

from scripts.distill.dpo_pairs import PROMPT, fact_line, serialise, tag

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


def missed_edges(rec: dict, proposed: list[Any]) -> tuple[list[tuple[dict, str]], Counter]:
    """The Opus edges that may be appended, as (edge without its quote, quote), and why
    the others were not: both entities listed, not a self-loop, no existing (or already
    added) edge between the same two entities in either direction, a verified quote, and
    a valid edge under the call's response schema."""
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
        edge = {k: v for k, v in e.items() if k != "quote"}
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
            band: float = BAND) -> list[dict]:
    """Every A pair; C pairs dropped in seeded order while the mean is above +band; then B
    candidates added in seeded order while it is below -band. Raises when the band is
    still out of reach."""
    pairs = [*a, *c]
    c_order = _seeded(c, f"C:{seed}")
    while pairs and mean_delta(pairs) > band and c_order:
        gone = c_order.pop()["meta"]["call_id"]
        pairs = [p for p in pairs if p["meta"]["call_id"] != gone]
    b_order = _seeded(b_pool, f"B:{seed}")
    while mean_delta(pairs) < -band and b_order:
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
