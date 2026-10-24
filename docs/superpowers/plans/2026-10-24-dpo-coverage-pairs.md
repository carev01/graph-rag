# dpo4: coverage-balanced DPO pairs -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build dpo4's training data (A/B/C pairs labelled by Claude subagents, length-balanced, ~150 val A pairs), the training-notebook changes (likelihood-based checkpoint selection, lr 2e-5, NLL 2.0), and a gate that runs on the Vast GPU against the merged bf16 model. Then run the round.

**Architecture:** Three new modules under `scripts/distill/`:
- `dpo_coverage.py`: pure pair construction plus the `build` CLI;
- `dpo_labels.py`: subagent task files, intake into `AnswerStore`, calibration;
- `dpo_gate.py`: runs on the Vast instance; pure scoring functions plus GPU commands with lazy imports.

Also:
- `dpo_select.py` holds the selection rule. The notebook builder embeds its source verbatim, so the unit tests and Colab run the same code.
- The notebook builder moves from the session scratchpad into the repo (`scripts/distill/build_dpo_nb.py`).
- Code tasks 1-6 come first. Operational tasks O1-O8 follow spec section 8 and stop at the points the spec names.

**Tech Stack:** Python 3.12, `uv`, pytest, `jsonschema` (already installed), TRL/Unsloth on Colab, vLLM + `peft` + `transformers` on the Vast instance (imported lazily, never in the dev env).

**Spec:** `docs/superpowers/specs/2026-10-24-dpo-coverage-pairs-design.md`. Executors read it before starting.

## Global Constraints

- Distillation data stays outside the public repo: `/home/openclaw/Documents/Shared/projects/graph-rag-distill/2026-10-24-dpo4/` (`D` below).
- API keys and tokens: never printed, never put on a command line, never written to disk by Claude. Tokens travel through the environment or stdin only. Never read or print `.env`.
- Every unit test is hermetic: no network, GPU, torch, vllm, peft, safetensors or huggingface_hub in the dev env. Import those lazily, inside the GPU-side functions only.
- Pass bar, unchanged from dpo3: removed share <= **0.75**, kept share >= **0.90**, valid JSON not fewer, ckpt100 removed written >= **10**.
- Calibration bar: Sonnet labels the full set only if recall of Opus `unsupported` edges >= **0.90**, on **100** existing replies (**50** with a DeepSeek both-pass `unsupported` edge, **50** without).
- Length balance: per split, the mean `len(chosen edges) - len(rejected edges)` within **±0.1**.
- Split: `split_of(group, 0.2)` (`--val-fraction 0.2`). `val_eval.jsonl` = **120** pairs, by type in proportion.
- Training: `LEARNING_RATE = 2e-5`, `NLL_ALPHA = 2.0`, `load_best_model_at_end = False`, `SAVE_TOTAL_LIMIT = None`. Selection: the last checkpoint with `eval_logps/chosen >= step-0 value - 1.0`. Stop at the first eval below that line.
- Adapter: private HF repo `carev01/qwen35-4b-graphrag-dpo-adapters`, revision `dpo4-ckpt<step>`. Export revision: `opus-distill-2026-10-dpo4`. Dataset: `carev01/ft-opus-distill@dpo4` (private).
- `missed` tasks: about **600** clean replies, seeded. The quote check is a word-for-word match against CURRENT MESSAGE with whitespace runs normalised to one space, and a minimum of **12** characters after normalising.
- Vast: **stop** (never destroy) between capture and gate and the adoption trial. Destroy only when the round ends (O8). Confirm with the Vast API.
- No dpo5: a failed gate ends DPO and ckpt100 stays.
- Lint: `uv run ruff check src tests` (E702: no semicolons in test fakes; E402: imports at the top). Also run ruff on the new `scripts/distill/*.py`.
- Commits end with the session's attribution lines. Pushing to `main` needs the user's approval, and gitleaks runs before every push.

## Review Focus

1. **A prompt whose `<ENTITIES>` block is missing or unparseable:** C must reject every proposed edge as `entity_not_listed`, without crashing. Test in Task 1.
2. **The same `call_id` in two capture files** (old and new capture passed together, or one passed twice): exactly one pair per reply. Test in Task 3.
3. **A subagent answer filed under a key that isn't in its batch, or with the wrong `ordinal`/`check`:** it is never stored as that key's answer. It is counted, and the right key stays pending. Test in Task 2.
4. **A val pair whose `chosen` and `rejected` tokenise identically** (no split point): the gate skips it and counts it, and does not crash or score it. Test in Task 4.
5. **A resumed Colab run, whose `log_history` has unsorted or duplicate eval entries:** the selection rule and the live stop must still pick from the step-0 baseline. Test in Task 5.

---

### Task 1: Pair construction (`dpo_coverage.py`, pure functions)

**Files:**
- Create: `scripts/distill/dpo_coverage.py`
- Test: `tests/unit/test_distill_dpo_coverage.py`

**Interfaces:**
- Consumes: `scripts.distill.dpo_pairs` exports `PROMPT`, `fact_line(e) -> str`, `serialise(payload) -> str`, `tag(text, name) -> str`. A reply record `rec` is `dpo_pairs.edge_records` output: `{"id", "messages": [system, user], "payload": {"edges": [...]}, "response_format", "context", "group"}`.
- Produces:
  - `MIN_QUOTE_CHARS = 12`, `BAND = 0.1`;
  - `norm_ws(text) -> str`, `quote_ok(quote, current) -> bool`;
  - `current_message(rec) -> str`, `entity_names(rec) -> set[str]`;
  - `ReplyLabels` (frozen dataclass: `verdicts`, `verified`, `confirmed`, `disputed`; property `clean`);
  - `classify(labels, recheck, current) -> ReplyLabels`;
  - `build_a(rec, rl, labels, labeller) -> dict | None`, `build_b(rec, rl, labels, labeller) -> dict | None`;
  - `missed_edges(rec, proposed) -> tuple[list[tuple[dict, str]], Counter]`, `build_c(rec, rl, proposed, labeller) -> tuple[dict | None, Counter]`;
  - `BalanceError`, `mean_delta(pairs) -> float`, `balance(a, b_pool, c, *, seed, band=BAND) -> list[dict]`;
  - `val_eval_subset(val, n, seed) -> list[dict]`.

  Label shapes:
  - `labels[i]` (edge i's first label): `{"verdict", "category", "quote", "why"}`;
  - `recheck`: `{edge_index: {"verdict": "supported"|"unsupported", "category", "why"}}`;
  - every pair's `meta`: `{"call_id", "group", "type", "labeller", "delta", "changed"}`.

- [ ] **Step 1: Write the failing tests**

```python
"""dpo_coverage: the quote check, label classification, pair types A/B/C, the length
balance and the val-eval subset (spec 2026-10-24-dpo-coverage-pairs-design.md section 2)."""
from __future__ import annotations

import json

import pytest

from scripts.distill import dpo_coverage as dc

CUR = "Veeam Backup supports Amazon S3 as a backup repository. Restores go to the original VM."
Q = "supports Amazon S3 as a backup repository"
NAMES = ("S0", "T0", "S1", "T1", "S2", "T2", "S3", "T3", "X", "Y")

SCHEMA = {
    "type": "object", "required": ["edges"],
    "properties": {"edges": {"type": "array", "items": {"$ref": "#/$defs/Edge"}}},
    "$defs": {"Edge": {
        "type": "object",
        "required": ["source_entity_name", "target_entity_name", "relation_type", "fact",
                     "episode_indices"],
        "properties": {
            "source_entity_name": {"type": "string"}, "target_entity_name": {"type": "string"},
            "relation_type": {"type": "string"}, "fact": {"type": "string"},
            "valid_at": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "invalid_at": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "episode_indices": {"type": "array", "items": {"type": "integer"}}}}}}


def _user(cur: str = CUR, entities: str | None = None) -> str:
    ents = json.dumps([{"name": n, "entity_types": ["Entity"]} for n in NAMES]) \
        if entities is None else entities
    return ("<PREVIOUS_MESSAGES>\n[]\n</PREVIOUS_MESSAGES>\n<CURRENT_MESSAGE>\n" + cur
            + "\n</CURRENT_MESSAGE>\n<ENTITIES>\n" + ents + "\n</ENTITIES>\n")


def _edge(i: int | str, j: int | str | None = None) -> dict:
    src, tgt = (f"S{i}", f"T{i}") if j is None else (str(i), str(j))
    return {"source_entity_name": src, "target_entity_name": tgt, "relation_type": "R",
            "fact": f"fact {src}-{tgt}", "valid_at": None, "invalid_at": None,
            "episode_indices": [0]}


def _rec(n: int = 3, call_id: str = "c1", entities: str | None = None) -> dict:
    return {"id": call_id, "group": "g|Article", "context": "DOCUMENT CONTEXT: x",
            "messages": [{"role": "system", "content": "sys"},
                         {"role": "user", "content": _user(entities=entities)}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "ExtractedEdges", "schema": SCHEMA}},
            "payload": {"edges": [_edge(i) for i in range(n)]}}


def _lab(verdict: str, quote: str | None = Q, category: str | None = "over_inference") -> dict:
    return {"verdict": verdict,
            "category": category if verdict == "unsupported" else None,
            "quote": quote if verdict == "stated" else None, "why": "w"}


def _p(call_id: str, kind: str, delta: int) -> dict:
    return {"meta": {"call_id": call_id, "type": kind, "delta": delta}}


def test_quote_ok_is_word_for_word_with_whitespace_normalised() -> None:
    assert dc.quote_ok(Q, CUR)
    assert dc.quote_ok("supports  Amazon\nS3 as a backup repository", CUR)
    assert not dc.quote_ok("supports Azure Blob as a backup repository", CUR)
    assert not dc.quote_ok("Amazon S3", CUR)           # under MIN_QUOTE_CHARS
    assert not dc.quote_ok(None, CUR)
    assert not dc.quote_ok(42, CUR)


def test_entity_names_reads_the_entities_block_and_survives_a_bad_one() -> None:
    assert dc.entity_names(_rec()) == set(NAMES)
    assert dc.entity_names(_rec(entities="not json")) == set()
    no_block = {**_rec(), "messages": [{"role": "system", "content": "sys"},
                                       {"role": "user", "content": "<CURRENT_MESSAGE>\nx\n"
                                                                   "</CURRENT_MESSAGE>"}]}
    assert dc.entity_names(no_block) == set()


def test_classify_confirmed_disputed_verified() -> None:
    labels = [_lab("stated"), _lab("stated", quote="nowhere in the chunk at all"),
              _lab("unsupported"), _lab("unsupported"), _lab("unsupported"),
              _lab("previous_only")]
    rl = dc.classify(labels, {2: {"verdict": "unsupported"}, 3: {"verdict": "supported"}}, CUR)
    assert rl.verified == (True, False, False, False, False, False)
    assert rl.confirmed == (2,)
    assert rl.disputed == (3, 4)          # 4: no re-check answer -> disputed, never confirmed
    assert not rl.clean
    clean = dc.classify([_lab("stated"), _lab("previous_only")], {}, CUR)
    assert clean.clean and clean.confirmed == () and clean.disputed == ()


def test_build_a_removes_only_confirmed_and_keeps_disputed_on_both_sides() -> None:
    rec = _rec(4)
    labels = [_lab("stated"), _lab("stated"), _lab("unsupported"), _lab("unsupported")]
    rl = dc.classify(labels, {2: {"verdict": "unsupported"}, 3: {"verdict": "supported"}}, CUR)
    pair = dc.build_a(rec, rl, labels, "sonnet")
    chosen, rejected = json.loads(pair["chosen"])["edges"], json.loads(pair["rejected"])["edges"]
    assert rejected == rec["payload"]["edges"]
    assert chosen == [rec["payload"]["edges"][i] for i in (0, 1, 3)]
    assert pair["meta"]["type"] == "A" and pair["meta"]["delta"] == -1
    assert pair["meta"]["labeller"] == "sonnet"
    assert pair["meta"]["disputed"] == ["S3 -> T3: fact S3-T3"]
    assert [c["fact"] for c in pair["meta"]["changed"]] == ["S2 -> T2: fact S2-T2"]


def test_build_a_none_when_nothing_or_everything_is_confirmed() -> None:
    rec = _rec(2)
    labels = [_lab("unsupported"), _lab("unsupported")]
    every = dc.classify(labels, {0: {"verdict": "unsupported"}, 1: {"verdict": "unsupported"}}, CUR)
    assert dc.build_a(rec, every, labels, "opus") is None
    none = dc.classify([_lab("stated"), _lab("stated")], {}, CUR)
    assert dc.build_a(rec, none, labels, "opus") is None


def test_build_b_drops_one_verified_edge_seeded() -> None:
    rec = _rec(3)
    labels = [_lab("stated"), _lab("stated"), _lab("stated", quote="not in the text, no")]
    rl = dc.classify(labels, {}, CUR)
    pair = dc.build_b(rec, rl, labels, "sonnet")
    chosen, rejected = json.loads(pair["chosen"])["edges"], json.loads(pair["rejected"])["edges"]
    assert chosen == rec["payload"]["edges"]
    dropped = [e for e in chosen if e not in rejected]
    assert len(dropped) == 1 and dropped[0] in rec["payload"]["edges"][:2]   # never the unverified
    assert pair["meta"]["type"] == "B" and pair["meta"]["delta"] == 1
    assert dc.build_b(rec, rl, labels, "sonnet") == pair                     # deterministic


def test_build_b_needs_a_clean_reply_with_two_verified_edges() -> None:
    rec = _rec(3)
    one = [_lab("stated"), _lab("previous_only"), _lab("stated", quote="absent from this chunk")]
    assert dc.build_b(rec, dc.classify(one, {}, CUR), one, "sonnet") is None
    dirty = [_lab("stated"), _lab("stated"), _lab("unsupported")]
    assert dc.build_b(rec, dc.classify(dirty, {}, CUR), dirty, "sonnet") is None


def test_missed_edges_filters_and_build_c_appends() -> None:
    rec = _rec(3)
    rl = dc.classify([_lab("stated")] * 3, {}, CUR)
    good = {**_edge("X", "Y"), "quote": Q}
    proposed = [
        good,
        {**_edge("T0", "S0"), "quote": Q},          # reverse of an existing pair
        {**_edge("Nope", "X"), "quote": Q},         # not in ENTITIES
        {**_edge("Y", "X"), "quote": Q},            # duplicate of `good`, reversed
        {**_edge("S1", "Y"), "quote": "no such words in the chunk"},
        {k: v for k, v in {**_edge("S0", "Y"), "quote": Q}.items() if k != "fact"},
        {**_edge("X", "X"), "quote": Q},
        "not an edge",
    ]
    pair, why = dc.build_c(rec, rl, proposed, "sonnet")
    chosen = json.loads(pair["chosen"])["edges"]
    assert chosen[:3] == rec["payload"]["edges"]
    assert chosen[3:] == [_edge("X", "Y")]                       # appended, quote stripped
    assert json.loads(pair["rejected"])["edges"] == rec["payload"]["edges"]
    assert pair["meta"]["type"] == "C" and pair["meta"]["delta"] == 1
    assert pair["meta"]["changed"] == [{"fact": "X -> Y: fact X-Y", "quote": Q}]
    assert why == {"pair_already_present": 2, "entity_not_listed": 1, "quote_not_found": 1,
                   "schema": 1, "self_loop": 1, "not_object": 1}


def test_build_c_none_when_nothing_survives_or_reply_not_clean() -> None:
    rec = _rec(2)
    clean = dc.classify([_lab("stated")] * 2, {}, CUR)
    pair, why = dc.build_c(rec, clean, [{**_edge("S0", "T0"), "quote": Q}], "opus")
    assert pair is None and why == {"pair_already_present": 1}
    dirty = dc.classify([_lab("stated"), _lab("unsupported")], {}, CUR)
    pair, why = dc.build_c(rec, dirty, [{**_edge("X", "Y"), "quote": Q}], "opus")
    assert pair is None and why == {"reply_not_clean": 1}


def test_build_c_with_a_bad_entities_block_rejects_everything() -> None:
    rec = _rec(2, entities="[{broken")
    clean = dc.classify([_lab("stated")] * 2, {}, CUR)
    pair, why = dc.build_c(rec, clean, [{**_edge("X", "Y"), "quote": Q}], "opus")
    assert pair is None and why == {"entity_not_listed": 1}


def test_balance_fills_b_until_the_mean_is_in_band() -> None:
    a = [_p(f"a{i}", "A", -2) for i in range(3)]
    c = [_p("c0", "C", 1)]
    b_pool = [_p(f"b{i}", "B", 1) for i in range(10)]
    out = dc.balance(a, b_pool, c, seed="s")
    assert abs(dc.mean_delta(out)) <= dc.BAND
    assert sum(p["meta"]["type"] == "B" for p in out) == 5
    assert all(p in out for p in a + c)
    assert out == dc.balance(a, b_pool, c, seed="s")


def test_balance_trims_c_then_fills_b_when_c_overshoots() -> None:
    a = [_p("a0", "A", -1)]
    c = [_p(f"c{i}", "C", 3) for i in range(3)]
    out = dc.balance(a, [_p("b0", "B", 1), _p("b1", "B", 1)], c, seed="s")
    assert abs(dc.mean_delta(out)) <= dc.BAND
    assert not any(p["meta"]["type"] == "C" for p in out)


def test_balance_raises_when_the_band_cannot_be_reached() -> None:
    with pytest.raises(dc.BalanceError):
        dc.balance([_p("a0", "A", -2), _p("a1", "A", -2)], [], [], seed="s")


def test_balance_of_nothing_is_nothing() -> None:
    assert dc.balance([], [], [], seed="s") == []


def test_val_eval_subset_is_proportional_and_seeded() -> None:
    val = ([_p(f"a{i}", "A", -1) for i in range(100)] + [_p(f"b{i}", "B", 1) for i in range(60)]
           + [_p(f"c{i}", "C", 1) for i in range(40)])
    sub = dc.val_eval_subset(val, 120, seed="s")
    kinds = [p["meta"]["type"] for p in sub]
    assert (kinds.count("A"), kinds.count("B"), kinds.count("C")) == (60, 36, 24)
    assert sub == dc.val_eval_subset(val, 120, seed="s")
    assert dc.val_eval_subset(val[:50], 120, seed="s") == val[:50]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_distill_dpo_coverage.py -q`
Expected: FAIL with `ImportError: cannot import name 'dpo_coverage'`.

- [ ] **Step 3: Write `scripts/distill/dpo_coverage.py`**

```python
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
        if src not in names or tgt not in names:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_distill_dpo_coverage.py -q && uv run ruff check scripts/distill/dpo_coverage.py tests/unit/test_distill_dpo_coverage.py`
Expected: all tests pass; ruff clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/distill/dpo_coverage.py tests/unit/test_distill_dpo_coverage.py
git commit -m "feat(distill): dpo4 pair types A/B/C, quote check, length balance"
```

---

### Task 2: Subagent labelling tasks, batches, intake and calibration (`dpo_labels.py`)

**Files:**
- Create: `scripts/distill/dpo_labels.py`
- Create: `scripts/distill/DPO_LABEL_BRIEF.md`
- Test: `tests/unit/test_distill_dpo_labels.py`

**Interfaces:**
- Consumes:
  - `scripts.distill.teacher.AnswerStore(root)` with `get(key)`, `put(key, answer_json, meta)`, `add_pending(key, request)`, `pending() -> list[(key, request)]`, `unqueue(key)`, `drop(key, reason, prompt_name)`, `dropped_reason(key)`;
  - `dpo_pairs.edge_records`, `fact_line`, `serialise`, `tag`, `VERDICTS`, `CATEGORIES`;
  - `dpo_coverage.classify`, `current_message`.
- Produces:
  - `KINDS = ("label", "recheck", "missed")`, `task_key(kind, call_id) -> str`;
  - `label_request(rec) -> dict`, `recheck_request(rec, idxs) -> dict`, `missed_request(rec) -> dict`;
  - `normalise(kind, request, answer) -> tuple[Any | None, str]`;
  - `write_batches(store, out_dir, size) -> list[Path]`, `intake(store, answers_dir) -> Counter`;
  - `load_answers(store, kind, ids) -> dict[str, Any]`, `load_recheck(store, ids) -> dict[str, dict[int, dict]]`;
  - `read_records(paths) -> list[dict]` (deduplicated by `id`);
  - `deepseek_verdicts(pairs_dir) -> dict[str, list[tuple[str, str]]]`;
  - `calibration_ids(recs, ds, n, seed) -> list[str]`;
  - `calibrate(ids, sonnet, opus) -> dict`;
  - `recheck_targets(recs, labels) -> dict[str, list[int]]`;
  - `missed_candidates(recs, labels, rechecks, n, seed) -> list[str]`;
  - CLI `python -m scripts.distill.dpo_labels {tasks,batches,intake,calibration-sample,calibrate,status}`.

  Stored answer shapes:
  - `label`: a list of per-edge dicts `{"verdict", "category", "quote", "why"}` in edge order;
  - `recheck`: `{"<edge index>": {"verdict", "category", "why"}}`;
  - `missed`: a list of edge dicts, each carrying `quote`.

- [ ] **Step 1: Write the failing tests**

```python
"""dpo_labels: subagent task files, intake into an AnswerStore (one re-ask, then drop),
calibration and task selection (spec 2026-10-24 section 3)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.distill import dpo_labels as dl
from scripts.distill.teacher import AnswerStore

CUR = "Veeam Backup supports Amazon S3 as a backup repository. Restores go to the original VM."
Q = "supports Amazon S3 as a backup repository"


def _edge(i: int) -> dict:
    return {"source_entity_name": f"S{i}", "target_entity_name": f"T{i}",
            "relation_type": "R", "fact": f"fact {i}", "valid_at": None,
            "invalid_at": None, "episode_indices": [0]}


def _rec(call_id: str = "c1", n: int = 3) -> dict:
    user = ("<PREVIOUS_MESSAGES>\n[]\n</PREVIOUS_MESSAGES>\n<CURRENT_MESSAGE>\n" + CUR
            + "\n</CURRENT_MESSAGE>\n<ENTITIES>\n[]\n</ENTITIES>\n")
    return {"id": call_id, "group": "g|A", "context": "DOCUMENT CONTEXT: x",
            "messages": [{"role": "system", "content": "SYSTEM RULES"},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_schema"},
            "payload": {"edges": [_edge(i) for i in range(n)]}}


def _labels_answer(order: list[int], verdict_of) -> dict:
    return {"labels": [{"n": n, "verdict": verdict_of(i),
                        "category": "over_inference" if verdict_of(i) == "unsupported" else None,
                        "quote": Q if verdict_of(i) == "stated" else None, "why": "w"}
                       for n, i in enumerate(order, 1)]}


def test_task_keys_differ_per_kind() -> None:
    keys = {dl.task_key(k, "c1") for k in dl.KINDS}
    assert len(keys) == 3 and dl.task_key("label", "c1") == dl.task_key("label", "c1")


def test_label_request_shows_shuffled_facts_and_the_chunk() -> None:
    req = dl.label_request(_rec(n=5))
    assert sorted(req["order"]) == [0, 1, 2, 3, 4] and req["kind"] == "label"
    shown = [ln.split(": ", 1)[1] for ln in req["prompt"].split("FACTS:\n")[1].splitlines()]
    assert shown == [f"fact {i}" for i in req["order"]]
    assert CUR in req["prompt"] and "quote" in req["prompt"]
    assert dl.label_request(_rec(n=5)) == req


def test_normalise_label_reindexes_by_edge_and_rejects_bad_answers() -> None:
    req = dl.label_request(_rec(n=3))
    ans = _labels_answer(req["order"], lambda i: "unsupported" if i == 1 else "stated")
    out, why = dl.normalise("label", req, ans)
    assert why == "" and [x["verdict"] for x in out] == ["stated", "unsupported", "stated"]
    assert out[0]["quote"] == Q and out[1]["category"] == "over_inference"
    short = {"labels": ans["labels"][:2]}
    assert dl.normalise("label", req, short)[0] is None
    rep = {"labels": [ans["labels"][0]] * 3}
    assert "repeated" in dl.normalise("label", req, rep)[1]
    nocat = json.loads(json.dumps(ans))
    for lab in nocat["labels"]:
        lab["category"] = None
    assert dl.normalise("label", req, nocat)[0] is None
    noquote = json.loads(json.dumps(ans))
    for lab in noquote["labels"]:
        lab["quote"] = None
    assert dl.normalise("label", req, noquote)[0] is None


def test_normalise_recheck_maps_subset_numbers_to_edges() -> None:
    req = dl.recheck_request(_rec(n=5), [1, 4])
    ans = {"labels": [{"n": n, "verdict": "unsupported" if i == 4 else "supported",
                       "category": "wrong_detail" if i == 4 else None, "why": "w"}
                      for n, i in enumerate(req["edges"], 1)]}
    out, why = dl.normalise("recheck", req, ans)
    assert why == "" and out == {"1": {"verdict": "supported", "category": None, "why": "w"},
                                 "4": {"verdict": "unsupported", "category": "wrong_detail",
                                       "why": "w"}}
    assert "SYSTEM RULES" not in req["prompt"] and "fact 1" in req["prompt"]


def test_normalise_missed_accepts_empty_and_rejects_quoteless() -> None:
    req = dl.missed_request(_rec())
    assert "SYSTEM RULES" in req["prompt"] and '"fact 0"' in req["prompt"]
    assert dl.normalise("missed", req, {"edges": []}) == ([], "")
    good = {**_edge(9), "quote": Q}
    assert dl.normalise("missed", req, {"edges": [good]}) == ([good], "")
    assert dl.normalise("missed", req, {"edges": [_edge(9)]})[0] is None
    assert dl.normalise("missed", req, {"edges": ["x"]})[0] is None
    assert dl.normalise("missed", req, {"nope": []})[0] is None


def _answers_file(batch: Path, rows: list[dict]) -> None:
    batch.with_suffix(".answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_batches_intake_reask_once_then_drop(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "store")
    reqs = {r["id"]: dl.label_request(r) for r in (_rec("c1"), _rec("c2"), _rec("c3"))}
    for cid, req in reqs.items():
        store.add_pending(dl.task_key("label", cid), req)
    batches = dl.write_batches(store, tmp_path / "round-1", size=2)
    assert [p.name for p in batches] == ["batch-001.jsonl", "batch-002.jsonl"]
    lines = [json.loads(x) for x in batches[0].read_text().splitlines()]
    assert [ln["ordinal"] for ln in lines] == [1, 2] and all(len(ln["check"]) == 12 for ln in lines)
    by_key = {ln["key"]: ln for ln in lines}
    k1, k2 = list(by_key)
    cid = {dl.task_key("label", c): c for c in reqs}
    good = _labels_answer(reqs[cid[k1]]["order"], lambda i: "stated")
    _answers_file(batches[0], [
        {"key": k1, "ordinal": 1, "check": by_key[k1]["check"], "answer": good},
        {"key": k2, "ordinal": 2, "check": by_key[k2]["check"], "answer": {"labels": []}},
        {"key": "f" * 64, "ordinal": 9, "check": "x", "answer": good},
    ])
    counts = dl.intake(store, tmp_path / "round-1")
    assert counts == {"stored": 1, "reasked": 1, "unknown_key": 1}
    assert [x["verdict"] for x in json.loads(store.get(k1))] == ["stated"] * 3
    pending = dict(store.pending())
    assert "rejection" in pending[k2] and pending[k2]["previous_answer"] == {"labels": []}
    k3 = json.loads(batches[1].read_text())["key"]
    assert k3 in pending                                   # never answered: still pending
    batches2 = dl.write_batches(store, tmp_path / "round-2", size=10)
    lines2 = {json.loads(x)["key"]: json.loads(x) for x in batches2[0].read_text().splitlines()}
    assert "rejection" in lines2[k2]
    _answers_file(batches2[0], [
        {"key": k2, "ordinal": lines2[k2]["ordinal"], "check": lines2[k2]["check"],
         "answer": {"labels": []}}])
    assert dl.intake(store, tmp_path / "round-2") == {"dropped": 1}
    assert store.dropped_reason(k2) and k2 not in dict(store.pending())


def test_intake_rejects_an_answer_under_the_wrong_ordinal(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "store")
    req = dl.label_request(_rec("c1"))
    key = dl.task_key("label", "c1")
    store.add_pending(key, req)
    batch = dl.write_batches(store, tmp_path / "r", size=10)[0]
    line = json.loads(batch.read_text())
    _answers_file(batch, [{"key": key, "ordinal": 2, "check": line["check"],
                           "answer": _labels_answer(req["order"], lambda i: "stated")}])
    assert dl.intake(store, tmp_path / "r") == {"reasked": 1}
    assert store.get(key) is None
    assert "wrong prompt" in dict(store.pending())[key]["rejection"]


def test_write_batches_refuses_a_used_directory(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "store")
    store.add_pending(dl.task_key("label", "c1"), dl.label_request(_rec("c1")))
    dl.write_batches(store, tmp_path / "r", size=10)
    with pytest.raises(FileExistsError):
        dl.write_batches(store, tmp_path / "r", size=10)


def test_deepseek_verdicts_pairs_both_passes(tmp_path: Path) -> None:
    for n, verdicts in ((1, ["unsupported", "stated"]), (2, ["unsupported", "unsupported"])):
        (tmp_path / f"judged-{n}.jsonl").write_text(json.dumps(
            {"id": "c1", "labels": [{"verdict": v} for v in verdicts]}) + "\n")
    assert dl.deepseek_verdicts(tmp_path) == {"c1": [("unsupported", "unsupported"),
                                                    ("stated", "unsupported")]}


def test_calibration_ids_half_flagged_half_not_seeded() -> None:
    recs = [_rec(f"c{i}") for i in range(300)]
    ds = {f"c{i}": [("unsupported", "unsupported")] if i % 5 == 0 else [("stated", "stated")]
          for i in range(300)}
    ids = dl.calibration_ids(recs, ds, 100, seed="s")
    flagged = [i for i in ids if ds[i][0] == ("unsupported", "unsupported")]
    assert len(ids) == 100 and len(flagged) == 50 and len(set(ids)) == 100
    assert ids == dl.calibration_ids(recs, ds, 100, seed="s")
    few = {k: [("stated", "stated")] for k in ds} | {"c0": [("unsupported", "unsupported")]}
    ids2 = dl.calibration_ids(recs, few, 100, seed="s")
    assert "c0" in ids2 and len(ids2) == 100


def _lab(v: str) -> dict:
    return {"verdict": v, "category": "other" if v == "unsupported" else None,
            "quote": Q if v == "stated" else None, "why": "w"}


def test_calibrate_recall_bar_is_inclusive_at_090() -> None:
    opus = {f"c{i}": [_lab("unsupported")] for i in range(10)}
    nine = {f"c{i}": [_lab("unsupported" if i < 9 else "stated")] for i in range(10)}
    res = dl.calibrate(list(opus), nine, opus)
    assert res["recall"] == pytest.approx(0.9) and res["passed"] and res["labeller"] == "sonnet"
    assert res["matrix"]["unsupported|unsupported"] == 9
    eight = {f"c{i}": [_lab("unsupported" if i < 8 else "stated")] for i in range(10)}
    assert dl.calibrate(list(opus), eight, opus)["labeller"] == "opus"
    with pytest.raises(ValueError):
        dl.calibrate(["c0"], {"c0": [_lab("stated")]}, {"c0": [_lab("stated")]})


def test_recheck_targets_and_missed_candidates() -> None:
    recs = [_rec(f"c{i}") for i in range(6)]
    labels = {f"c{i}": [_lab("stated"), _lab("stated"),
                        _lab("unsupported" if i < 2 else "stated")] for i in range(6)}
    assert dl.recheck_targets(recs, labels) == {"c0": [2], "c1": [2]}
    rechecks = {"c0": {2: {"verdict": "unsupported"}}, "c1": {2: {"verdict": "supported"}}}
    picked = dl.missed_candidates(recs, labels, rechecks, n=3, seed="s")
    assert len(picked) == 3 and not {"c0", "c1"} & set(picked)
    assert picked == dl.missed_candidates(recs, labels, rechecks, n=3, seed="s")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_distill_dpo_labels.py -q`
Expected: FAIL with `ImportError: cannot import name 'dpo_labels'`.

- [ ] **Step 3: Write `scripts/distill/dpo_labels.py`**

```python
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
            key = row.get("key")
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
    other = [i for i in ids if i not in set(flagged)]
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
```

CLI (append to the module):

```python
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
```

- [ ] **Step 4: Write `scripts/distill/DPO_LABEL_BRIEF.md`**

```markdown
# DPO labelling brief (dpo4)

You label data for a small extraction model. Each line of your batch file is one task.
Every task's `prompt` is self-contained: it holds its own instructions, the documentation
chunk and the facts or reply to judge. Use ONLY the prompt's own text; no outside
knowledge.

- Answer every line with exactly one JSON object, as the prompt's last paragraph
  specifies -- no prose, no markdown fences.
- Quotes are checked by code, character for character (runs of whitespace count as one
  space) against the chunk's CURRENT MESSAGE. Copy them; never paraphrase. A quote that
  is not in the CURRENT MESSAGE costs the fact.
- Be strict about support. A fact the text does not state is unsupported, however
  plausible.
- Echo each line's `key`, `ordinal` and `check` unchanged next to your answer.
- If a line carries `rejection`, fix exactly what it names (your earlier answer is in
  `previous_answer`). This is the last chance: a second rejection drops the task.

Write your answers to `<batch>.answers.jsonl`, one line per task:

    {"key": "<sha256>", "ordinal": <n>, "check": "<12 hex>", "answer": {...}}

then report only the counts (answered, skipped).
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_distill_dpo_labels.py -q && uv run ruff check scripts/distill/dpo_labels.py tests/unit/test_distill_dpo_labels.py`
Expected: all tests pass; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add scripts/distill/dpo_labels.py scripts/distill/DPO_LABEL_BRIEF.md tests/unit/test_distill_dpo_labels.py
git commit -m "feat(distill): dpo4 subagent labelling -- tasks, batches, intake, calibration"
```

---

### Task 3: The `build` CLI (`dpo_coverage.assemble` + `main`)

**Files:**
- Modify: `scripts/distill/dpo_coverage.py` (append)
- Modify: `scripts/distill/package_dataset.py:75-77` (ship `val_eval.jsonl`)
- Test: `tests/unit/test_distill_dpo_coverage_build.py`

**Interfaces:**
- Consumes:
  - Task 1's builders and `balance`/`val_eval_subset`;
  - Task 2's `read_records`, `load_answers`, `load_recheck`, `deepseek_verdicts`, `task_key`;
  - `dpo_pairs.split_of(group, val_fraction) -> "train"|"val"`.
- Produces:
  - `assemble(recs, labels, rechecks, missed, labeller, val_fraction) -> tuple[dict, Counter, Counter]`. The first value is `{"train": {"A": [...], "B": [...], "C": [...]}, "val": {...}}`; the B lists are candidates.
  - `final_label(rl, i) -> str`;
  - `labels_report(recs, labels, rechecks, c_why, ds) -> dict`;
  - CLI `python -m scripts.distill.dpo_coverage --capture ... --labels S --labeller sonnet|opus --recheck S --missed S [--deepseek DIR] --out DIR [--val-fraction 0.2] [--eval-size 120] [--seed 20261024]`, writing `train.jsonl`, `val.jsonl`, `val_eval.jsonl`, `summary.json`, `labels_report.json`.

- [ ] **Step 1: Write the failing tests**

```python
"""dpo_coverage build: replies to A/B/C by their labels, one pair per reply, per-split
balance, val_eval, and the two reports (spec 2026-10-24 sections 2 and 7)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.distill import dpo_coverage as dc
from scripts.distill import dpo_labels as dl
from scripts.distill import dpo_pairs as dp
from scripts.distill.teacher import AnswerStore

CTX = "DOCUMENT CONTEXT: This chunk comes from the Veeam Backup & Replication documentation."
CUR = "Veeam Backup supports Amazon S3 as a backup repository. Restores go to the original VM."
Q = "supports Amazon S3 as a backup repository"
NAMES = ["S0", "T0", "S1", "T1", "S2", "T2", "X", "Y"]


def _edge(src: str, tgt: str) -> dict:
    return {"source_entity_name": src, "target_entity_name": tgt, "relation_type": "R",
            "fact": f"fact {src}-{tgt}", "valid_at": None, "invalid_at": None,
            "episode_indices": [0]}


def _line(call_id: str, title: str) -> str:
    user = (f"<PREVIOUS_MESSAGES>\n[]\n</PREVIOUS_MESSAGES>\n<CURRENT_MESSAGE>\n[{title} › Steps]\n\n"
            f"{CUR}\n</CURRENT_MESSAGE>\n<ENTITIES>\n"
            + json.dumps([{"name": n, "entity_types": ["Entity"]} for n in NAMES])
            + "\n</ENTITIES>\n")
    return json.dumps({
        "call_id": call_id, "tier": "cheap", "prompt_name": dp.PROMPT,
        "model": dp.STUDENT_MODEL, "finish_reason": "stop",
        "messages": [{"role": "system", "content": f"sys\n{CTX}"}, {"role": "user", "content": user}],
        "response_format": {"type": "json_schema"},
        "completion": json.dumps({"edges": [_edge(f"S{i}", f"T{i}") for i in range(3)]})})


def _lab(v: str) -> dict:
    return {"verdict": v, "category": "over_inference" if v == "unsupported" else None,
            "quote": Q if v == "stated" else None, "why": "w"}


def _titles(split: str, k: int) -> list[str]:
    out, i = [], 0
    while len(out) < k:
        title = f"Article {i}"
        if dp.split_of(f"{CTX}|{title}", 0.5) == split:
            out.append(title)
        i += 1
    return out


def test_assemble_routes_each_reply_once() -> None:
    lines = [_line(cid, f"T{cid}") for cid in ("a", "c", "b", "u", "d")] + [_line("a", "Ta")]
    recs = dl.read_records_from_lines(lines)
    assert sorted(r["id"] for r in recs) == ["a", "b", "c", "d", "u"]     # "a" once
    labels = {"a": [_lab("stated"), _lab("stated"), _lab("unsupported")],
              "c": [_lab("stated")] * 3, "b": [_lab("stated")] * 3,
              "d": [_lab("stated"), _lab("stated"), _lab("unsupported")]}
    rechecks = {"a": {2: {"verdict": "unsupported"}}, "d": {2: {"verdict": "supported"}}}
    missed = {"c": [{**_edge("X", "Y"), "quote": Q}], "b": [{**_edge("S0", "T0"), "quote": Q}]}
    by_split, stats, c_why = dc.assemble(recs, labels, rechecks, missed, "sonnet", 0.0)
    train = by_split["train"]
    assert [p["meta"]["call_id"] for p in train["A"]] == ["a"]
    assert [p["meta"]["call_id"] for p in train["C"]] == ["c"]
    assert [p["meta"]["call_id"] for p in train["B"]] == ["b"]            # C failed -> B
    assert stats["unlabelled"] == 1 and stats["disputed_only"] == 1
    assert c_why == {"pair_already_present": 1}
    assert by_split["val"] == {"A": [], "B": [], "C": []}


def _store(root: Path, kind: str, answers: dict) -> None:
    st = AnswerStore(root)
    for cid, ans in answers.items():
        st.put(dl.task_key(kind, cid), json.dumps(ans), {"kind": kind, "call_id": cid})


def test_main_writes_balanced_splits_and_reports(tmp_path: Path) -> None:
    rows, labels, rechecks, ds_rows = [], {}, {}, []
    for split in ("train", "val"):
        for j, title in enumerate(_titles(split, 3)):
            cid = f"{split}{j}"
            rows.append(_line(cid, title))
            if j == 0:      # A: edge 2 confirmed unsupported
                labels[cid] = [_lab("stated"), _lab("stated"), _lab("unsupported")]
                rechecks[cid] = {"2": {"verdict": "unsupported", "category": "other", "why": "w"}}
            else:           # B candidates
                labels[cid] = [_lab("stated")] * 3
            ds_rows.append({"id": cid, "labels": [{"verdict": "unsupported"}] * 3})
    cap = tmp_path / "capture.jsonl"
    cap.write_text("\n".join(rows) + "\n")
    _store(tmp_path / "label", "label", labels)
    _store(tmp_path / "recheck", "recheck", rechecks)
    AnswerStore(tmp_path / "missed")
    ds = tmp_path / "ds"
    ds.mkdir()
    for n in (1, 2):
        (ds / f"judged-{n}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ds_rows))
    out = tmp_path / "pairs"
    rc = dc.main(["--capture", str(cap), "--labels", str(tmp_path / "label"), "--labeller",
                  "sonnet", "--recheck", str(tmp_path / "recheck"), "--missed",
                  str(tmp_path / "missed"), "--deepseek", str(ds), "--out", str(out),
                  "--val-fraction", "0.5", "--eval-size", "2"])
    assert rc == 0
    for split in ("train", "val"):
        pairs = [json.loads(x) for x in (out / f"{split}.jsonl").read_text().splitlines()]
        assert abs(dc.mean_delta(pairs)) <= dc.BAND
        assert sum(p["meta"]["type"] == "A" for p in pairs) == 1
    assert len((out / "val_eval.jsonl").read_text().splitlines()) == 2
    summary = json.loads((out / "summary.json").read_text())
    assert summary["pairs"]["train"]["A"] == 1 and summary["mean_delta"]["val"] == 0.0
    report = json.loads((out / "labels_report.json").read_text())
    assert report["deepseek"]["both_unsupported_edges"] == 18
    assert report["deepseek"]["of_which_confirmed"] == 2
```

`read_records_from_lines` is a small helper that Task 3 adds to `dpo_labels`. `read_records(paths)` is then rewritten to call it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_distill_dpo_coverage_build.py -q`
Expected: FAIL with `AttributeError: module 'scripts.distill.dpo_labels' has no attribute 'read_records_from_lines'`.

- [ ] **Step 3: Add `read_records_from_lines` to `dpo_labels.py`**

Replace `read_records` with:

```python
def read_records_from_lines(lines: list[str]) -> list[dict]:
    """The judgeable student edge replies, one per call_id (first occurrence wins)."""
    recs, _ = edge_records([ln for ln in lines if ln.strip()])
    seen: dict[str, dict] = {}
    for r in recs:
        seen.setdefault(r["id"], r)
    return list(seen.values())


def read_records(paths: list[Path]) -> list[dict]:
    return read_records_from_lines([ln for p in paths for ln in Path(p).read_text().splitlines()])
```

- [ ] **Step 4: Append `assemble`, the reports and the CLI to `dpo_coverage.py`**

At the top, add `import argparse` and `from pathlib import Path` to the imports. `split_of` comes from `dpo_pairs`: extend the existing import to `from scripts.distill.dpo_pairs import PROMPT, fact_line, serialise, split_of, tag`. Then append:

```python
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
    a.out.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[dict]] = {}
    for split, kinds in by_split.items():
        if not kinds["A"]:
            raise SystemExit(f"no A pairs in {split} -- nothing to measure or train on")
        out[split] = balance(kinds["A"], kinds["B"], kinds["C"], seed=f"{a.seed}:{split}")
        _write_jsonl(a.out / f"{split}.jsonl", out[split])
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
```

The stores hold recheck keys as strings and `load_recheck` converts them to `int`, which is what `classify` expects.

- [ ] **Step 5: Ship `val_eval.jsonl` in the dataset package**

In `scripts/distill/package_dataset.py`, change lines 72-77 to:

```python
SAMPLE_FILE = "sample.json"
VAL_EVAL_FILE = "val_eval.jsonl"   # dpo4: the trainer's fixed eval subset
MANIFEST_FILE = "MANIFEST.json"
REQUIRED_FILES = (TRAIN_FILE, VAL_FILE)
OPTIONAL_FILES = (SAMPLE_FILE, VAL_EVAL_FILE)

ZIP_MEMBERS = (TRAIN_FILE, VAL_FILE, VAL_EVAL_FILE, SAMPLE_FILE, MANIFEST_FILE)
```

Keep whatever line 76 holds between `OPTIONAL_FILES` and `ZIP_MEMBERS` unchanged. Append to `tests/unit/test_distill_package.py`:

```python
def test_package_ships_val_eval_when_present(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for name in ("train.jsonl", "val.jsonl", "val_eval.jsonl"):
        (in_dir / name).write_text(json.dumps({"prompt_name": "p", "messages": []}) + "\n")
    out = tmp_path / "out.zip"
    package_dataset.package(in_dir, out)
    with zipfile.ZipFile(out) as zf:
        assert "val_eval.jsonl" in zf.namelist()
```

Before writing this test, check the existing file's imports and its module alias (`package_dataset`, `json`, `zipfile`) and match them.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_distill_dpo_coverage_build.py tests/unit/test_distill_dpo_coverage.py tests/unit/test_distill_dpo_labels.py tests/unit/test_distill_package.py -q && uv run ruff check scripts/distill tests/unit/test_distill_dpo_coverage_build.py`
Expected: all pass; ruff clean.

- [ ] **Step 7: Commit**

```bash
git add scripts/distill/dpo_coverage.py scripts/distill/dpo_labels.py scripts/distill/package_dataset.py tests/unit/test_distill_dpo_coverage_build.py tests/unit/test_distill_package.py
git commit -m "feat(distill): dpo4 build CLI -- one pair per reply, per-split balance, val_eval, label reports"
```

---

### Task 4: The Vast gate (`dpo_gate.py` + `dpo_gate_remote.sh`)

**Files:**
- Create: `scripts/distill/dpo_gate.py`
- Create: `scripts/distill/dpo_gate_remote.sh`
- Test: `tests/unit/test_distill_dpo_gate.py`

**Interfaces:**
- Consumes: val pairs from Task 3 (`meta.type`, `meta.call_id`, `chosen`, `rejected`, `messages`); `chat_format.serving_prompt(tokenizer, messages)`.
- Produces:
  - constants `MAX_REMOVED_SHARE = 0.75`, `MIN_KEPT_SHARE = 0.90`, `MIN_BASELINE_REMOVED = 10`;
  - `triples(text) -> tuple[set, bool]`, `split_point(a, b) -> int | None`;
  - `prepare(record, tok) -> dict | None`, `score_pair(record, prefix_text, gen_text) -> dict`, `totals(rows, kind) -> dict`;
  - `decide(base, dpo) -> tuple[bool, list[str]]`;
  - `sign_test(base, dpo) -> dict`, `bootstrap(base_rows, dpo_rows, n, seed) -> dict`;
  - `differing(names, equal) -> list[str]`;
  - CLI subcommands `merge`, `generate`, `score`, `push`. `score` exits 0 on a pass, 1 on a fail, and 2 when the smoke test finds identical outputs.

- [ ] **Step 1: Write the failing tests**

```python
"""dpo_gate (spec 2026-10-24 section 6): triple scoring from the split point for A/B/C
pairs, the unchanged pass bar, the statistics, the adapter-engaged check, and `score`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.distill import dpo_gate as g


def _e(i: int) -> dict:
    return {"source_entity_name": f"S{i}", "target_entity_name": f"T{i}",
            "relation_type": "R", "fact": f"fact {i}", "valid_at": None,
            "invalid_at": None, "episode_indices": [0]}


def _ser(edges: list[dict]) -> str:
    return json.dumps({"edges": edges}, ensure_ascii=False)


def _rec(kind: str, chosen: list[int], rejected: list[int], cid: str = "c") -> dict:
    return {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            "chosen": _ser([_e(i) for i in chosen]), "rejected": _ser([_e(i) for i in rejected]),
            "meta": {"call_id": cid, "type": kind}}


class CharTok:
    """One token per character -- enough to test split points and decoding."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        return "<p>" + messages[-1]["content"] + "</p>"


def test_triples_json_and_regex_fallback() -> None:
    assert g.triples(_ser([_e(1)])) == ({("S1", "T1", "R")}, True)
    text = _ser([_e(1), _e(2)])
    cut = text[: text.index('"target_entity_name": "T2"')]   # second triple incomplete
    assert g.triples(cut) == ({("S1", "T1", "R")}, False)


def test_split_point() -> None:
    assert g.split_point([1, 2, 3], [1, 2, 4]) == 2
    assert g.split_point([1, 2], [1, 2]) is None


def test_prepare_finds_the_split_and_sizes_the_budget() -> None:
    r = _rec("A", [0, 2], [0, 1, 2])
    p = g.prepare(r, CharTok())
    prompt = "<p>u</p>"
    full_c, full_r = prompt + r["chosen"], prompt + r["rejected"]
    split = next(i for i, (a, b) in enumerate(zip(full_c, full_r)) if a != b)
    assert p["prefix_ids"] == [ord(c) for c in full_c[:split]]
    assert p["prefix_text"] == full_c[len(prompt):split]
    assert p["max_tokens"] == min(len(full_r) - split + 256, 4096)
    same = _rec("A", [0], [0])
    assert g.prepare(same, CharTok()) is None


def test_score_pair_a_b_c() -> None:
    a = _rec("A", [0, 2], [0, 1, 2])
    prefix = '{"edges": [' + json.dumps(_e(0)) + ", "
    row = g.score_pair(a, prefix, json.dumps(_e(1)) + ", " + json.dumps(_e(2)) + "]}")
    assert (row["removed"], row["removed_written"], row["kept"], row["kept_written"]) == (1, 1, 1, 1)
    assert row["json"] is True
    b = _rec("B", [0, 1, 2], [0, 2])
    row = g.score_pair(b, prefix, json.dumps(_e(1)) + ", " + json.dumps(_e(2)) + "]}")
    assert (row["added"], row["added_written"]) == (1, 1)
    c = _rec("C", [0, 1], [0])
    row = g.score_pair(c, '{"edges": [' + json.dumps(_e(0)), "]}")
    assert (row["added"], row["added_written"], row["json"]) == (1, 0, True)


def test_decide_applies_the_unchanged_bar() -> None:
    base = {"removed_written": 20, "kept_written": 100, "json": 50}
    assert g.decide(base, {"removed_written": 15, "kept_written": 90, "json": 50}) == (True, [])
    assert not g.decide(base, {"removed_written": 16, "kept_written": 90, "json": 50})[0]
    assert not g.decide(base, {"removed_written": 15, "kept_written": 89, "json": 50})[0]
    assert not g.decide(base, {"removed_written": 15, "kept_written": 90, "json": 49})[0]
    ok, why = g.decide({**base, "removed_written": 9}, {"removed_written": 0, "kept_written": 100,
                                                        "json": 50})
    assert not ok and "baseline" in why[0]


def test_sign_test_matches_the_dpo3_numbers() -> None:
    base = [1] * 8 + [0] * 29 + [0] * 2
    dpo = [0] * 8 + [0] * 29 + [1] * 2
    res = g.sign_test(base, dpo)
    assert (res["down"], res["same"], res["up"]) == (8, 29, 2)
    assert res["p"] == pytest.approx(0.109375)


def test_bootstrap_is_seeded_and_brackets_the_point() -> None:
    base = [{"removed_written": i % 3, "kept_written": 3} for i in range(40)]
    dpo = [{"removed_written": (i % 3) // 2, "kept_written": 3} for i in range(40)]
    res = g.bootstrap(base, dpo, n=2000, seed=1)
    assert res == g.bootstrap(base, dpo, n=2000, seed=1)
    lo, hi = res["removed_ratio_ci"]
    assert lo <= res["removed_ratio"] <= hi
    assert res["kept_ratio"] == pytest.approx(1.0)


def test_differing_only_looks_at_lora_targets() -> None:
    names = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.input_layernorm.weight",
             "model.layers.0.mlp.down_proj.weight"]
    changed = {"model.layers.0.mlp.down_proj.weight"}
    assert g.differing(names, lambda n: n not in changed) == ["model.layers.0.mlp.down_proj.weight"]


def _gen(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_score_command_writes_gate_json_and_exit_code(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(g, "MIN_BASELINE_REMOVED", 1)
    val = [_rec("A", [0, 2], [0, 1, 2], "a1"), _rec("C", [0, 1], [0], "c1")]
    (tmp_path / "val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
    prefix_a = '{"edges": [' + json.dumps(_e(0)) + ", "
    prefix_c = '{"edges": [' + json.dumps(_e(0))
    base = [{"call_id": "a1", "prefix_text": prefix_a, "gen_text": json.dumps(_e(1)) + ", "
             + json.dumps(_e(2)) + "]}", "gen_ids": [1]},
            {"call_id": "c1", "prefix_text": prefix_c, "gen_text": "]}", "gen_ids": [1]}]
    dpo = [{"call_id": "a1", "prefix_text": prefix_a, "gen_text": json.dumps(_e(2)) + "]}",
            "gen_ids": [2]},
           {"call_id": "c1", "prefix_text": prefix_c, "gen_text": ", " + json.dumps(_e(1)) + "]}",
            "gen_ids": [2]}]
    _gen(tmp_path / "base.jsonl", base)
    _gen(tmp_path / "dpo.jsonl", dpo)
    rc = g.main(["score", "--val", str(tmp_path / "val.jsonl"), "--base",
                 str(tmp_path / "base.jsonl"), "--dpo", str(tmp_path / "dpo.jsonl"),
                 "--out", str(tmp_path / "gate.json")])
    gate = json.loads((tmp_path / "gate.json").read_text())
    assert rc == 0 and gate["passed"] is True
    assert gate["totals"]["A"]["dpo"]["removed_written"] == 0
    assert gate["totals"]["C"]["dpo"]["added_written"] == 1
    _gen(tmp_path / "same.jsonl", base)
    rc = g.main(["score", "--smoke", "--val", str(tmp_path / "val.jsonl"), "--base",
                 str(tmp_path / "base.jsonl"), "--dpo", str(tmp_path / "same.jsonl"),
                 "--out", str(tmp_path / "smoke.json")])
    assert rc == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_distill_dpo_gate.py -q`
Expected: FAIL with `ImportError: cannot import name 'dpo_gate'`.

- [ ] **Step 3: Write `scripts/distill/dpo_gate.py`**

```python
"""dpo4 gate (spec docs/superpowers/specs/2026-10-24-dpo-coverage-pairs-design.md section 6).
Runs ON the Vast GPU instance (scripts/distill/dpo_gate_remote.sh drives it); the scoring
and statistics are pure and unit-tested, the GPU commands import torch / transformers /
peft / vllm lazily.

    python dpo_gate.py merge --base B --adapter A --out M      # bf16 merge, adapter engaged?
    python dpo_gate.py generate --model M --tokenizer B --val V --out G [--limit N] [--full]
    python dpo_gate.py score --base G0 --dpo G1 --val V --out gate.json [--smoke]
    python dpo_gate.py push --merged M --repo R --revision REV --smoke-json G   # after a pass

Each model continues every val pair greedily from the token prefix chosen and rejected
share, and edges are compared as exact (source, target, relation) triples. The pass bar,
unchanged from dpo3, is on the A pairs: removed written <= 0.75 x ckpt100's, kept written
>= 0.90 x, valid JSON not fewer, ckpt100's removed written >= 10. B and C are reported.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from scripts.distill.chat_format import serving_prompt
except ImportError:   # on the instance: chat_format.py is copied next to this file
    from chat_format import serving_prompt   # type: ignore[no-redef]

MAX_REMOVED_SHARE = 0.75
MIN_KEPT_SHARE = 0.90
MIN_BASELINE_REMOVED = 10
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_TRIPLE = re.compile(r'"source_entity_name": "(.*?)", "target_entity_name": "(.*?)", '
                     r'"relation_type": "(.*?)"')
COUNTS = ("removed", "removed_written", "added", "added_written", "kept", "kept_written", "json")


def triples(text: str) -> tuple[set, bool]:
    try:
        return {(e["source_entity_name"], e["target_entity_name"], e["relation_type"])
                for e in json.loads(text)["edges"]}, True
    except (ValueError, KeyError, TypeError):
        return set(_TRIPLE.findall(text)), False


def split_point(a: list[int], b: list[int]) -> int | None:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def prepare(record: dict, tok: Any) -> dict | None:
    """The shared token prefix up to where chosen and rejected first differ, its reply
    part as text, and the generation budget; None when they tokenise identically."""
    prompt = serving_prompt(tok, record["messages"])
    n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
    ch = tok(prompt + record["chosen"], add_special_tokens=False)["input_ids"]
    rj = tok(prompt + record["rejected"], add_special_tokens=False)["input_ids"]
    split = split_point(ch, rj)
    if split is None:
        return None
    return {"call_id": record["meta"]["call_id"], "type": record["meta"]["type"],
            "prefix_ids": ch[:split],
            "prefix_text": tok.decode(ch[n_prompt:split], skip_special_tokens=True),
            "max_tokens": min(len(rj) - split + 256, 4096)}


def score_pair(record: dict, prefix_text: str, gen_text: str) -> dict:
    gen, ok = triples(prefix_text + gen_text)
    done = set(_TRIPLE.findall(prefix_text))   # edges complete before the split
    ch, rj = triples(record["chosen"])[0], triples(record["rejected"])[0]
    removed, added, kept = rj - ch, ch - rj, ch - done
    return {"call_id": record["meta"]["call_id"], "type": record["meta"]["type"],
            "removed": len(removed), "removed_written": len(removed & gen),
            "added": len(added), "added_written": len(added & gen),
            "kept": len(kept), "kept_written": len(kept & gen), "json": ok}


def totals(rows: list[dict], kind: str) -> dict:
    sel = [r for r in rows if r["type"] == kind]
    return {"pairs": len(sel), **{k: sum(int(r[k]) for r in sel) for k in COUNTS}}


def decide(base: dict, dpo: dict) -> tuple[bool, list[str]]:
    why = []
    if base["removed_written"] < MIN_BASELINE_REMOVED:
        why.append(f"baseline: ckpt100 writes only {base['removed_written']} removed edges "
                   f"(< {MIN_BASELINE_REMOVED}) -- the val set cannot show a reduction")
    if dpo["removed_written"] > MAX_REMOVED_SHARE * base["removed_written"]:
        why.append(f"removed: {dpo['removed_written']} > {MAX_REMOVED_SHARE} x "
                   f"{base['removed_written']}")
    if dpo["kept_written"] < MIN_KEPT_SHARE * base["kept_written"]:
        why.append(f"kept: {dpo['kept_written']} < {MIN_KEPT_SHARE} x {base['kept_written']}")
    if dpo["json"] < base["json"]:
        why.append(f"json: {dpo['json']} < {base['json']}")
    return not why, why


def sign_test(base: list[int], dpo: list[int]) -> dict:
    down = sum(d < b for b, d in zip(base, dpo))
    up = sum(d > b for b, d in zip(base, dpo))
    n = down + up
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(down, up) + 1)) / 2 ** n) if n else 1.0
    return {"down": down, "same": len(base) - n, "up": up, "p": p}


def _ratio(rows_b: list[dict], rows_d: list[dict], key: str) -> float | None:
    den = sum(r[key] for r in rows_b)
    return sum(r[key] for r in rows_d) / den if den else None


def bootstrap(base_rows: list[dict], dpo_rows: list[dict], n: int = 10_000,
              seed: int = 20261024) -> dict:
    """Pairs resampled together: CIs for removed ratio, kept ratio and their quotient."""
    rng = random.Random(seed)
    idx = range(len(base_rows))
    rem, kep, sel = [], [], []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        rb, rd = [base_rows[i] for i in s], [dpo_rows[i] for i in s]
        r, k = _ratio(rb, rd, "removed_written"), _ratio(rb, rd, "kept_written")
        if r is None or k is None or not k:
            continue
        rem.append(r)
        kep.append(k)
        sel.append(r / k)

    def ci(xs: list[float]) -> list[float]:
        xs = sorted(xs)
        return [xs[int(0.025 * len(xs))], xs[min(len(xs) - 1, int(0.975 * len(xs)))]]

    r0, k0 = _ratio(base_rows, dpo_rows, "removed_written"), _ratio(base_rows, dpo_rows, "kept_written")
    return {"removed_ratio": r0, "removed_ratio_ci": ci(rem), "kept_ratio": k0,
            "kept_ratio_ci": ci(kep),
            "selectivity": r0 / k0 if r0 is not None and k0 else None,
            "selectivity_ci": ci(sel), "resamples": len(rem)}


def differing(names: Iterable[str], equal: Callable[[str], bool]) -> list[str]:
    return [n for n in names if any(f".{t}." in n for t in TARGET_MODULES) and not equal(n)]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in Path(path).read_text(encoding="utf-8").splitlines()
            if ln.strip()]


# --- GPU-side commands (lazy imports; not unit-tested) -------------------------------------

def merge(base: Path, adapter: Path, out: Path) -> None:
    import torch
    import transformers
    from peft import PeftModel
    from safetensors import safe_open

    arch = json.loads((base / "config.json").read_text())["architectures"][0]
    model = getattr(transformers, arch).from_pretrained(str(base), torch_dtype=torch.bfloat16)
    peft = PeftModel.from_pretrained(model, str(adapter))
    live = sum(int(p.detach().abs().sum().item() > 0)
               for n, p in peft.named_parameters() if "lora_B" in n)
    assert live > 0, "every lora_B is zero after loading: the adapter's keys did not match"
    print(f"adapter loaded: {live} non-zero lora_B tensors", flush=True)
    peft.merge_and_unload().save_pretrained(str(out), safe_serialization=True)
    for f in base.iterdir():   # tokenizer, chat template, processor configs
        if f.is_file() and not f.name.endswith(".safetensors") and not (out / f.name).exists():
            shutil.copy2(f, out / f.name)

    def weight_map(d: Path) -> dict[str, str]:
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            return json.loads(idx.read_text())["weight_map"]
        return {k: "model.safetensors" for k in safe_open(str(d / "model.safetensors"), "pt").keys()}

    wb, wm = weight_map(base), weight_map(out)
    common = sorted(set(wb) & set(wm))
    targets = [n for n in common if any(f".{t}." in n for t in TARGET_MODULES)]
    assert targets, "no LoRA target tensor has the same name in ckpt100 and the merged model"

    def equal(name: str) -> bool:
        with safe_open(str(base / wb[name]), "pt") as fb, safe_open(str(out / wm[name]), "pt") as fm:
            return torch.equal(fb.get_tensor(name), fm.get_tensor(name))

    diff = differing(targets, equal)
    assert diff, "the merged model's LoRA target weights equal ckpt100's: the adapter is not engaged"
    print(f"merged: {len(diff)} of {len(targets)} target tensors differ from ckpt100", flush=True)


def generate(model: Path, tokenizer: Path, val: Path, out: Path, limit: int, full: bool,
             max_model_len: int) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = AutoTokenizer.from_pretrained(str(tokenizer))
    records = _read_jsonl(val)
    prepared, skipped = [], 0
    if full:   # whole replies, the shortest prompts first (the export's JSON smoke test)
        for r in sorted(records, key=lambda r: len(r["messages"][1]["content"]))[: limit or 3]:
            ids = tok(serving_prompt(tok, r["messages"]), add_special_tokens=False)["input_ids"]
            prepared.append({"call_id": r["meta"]["call_id"], "type": r["meta"]["type"],
                             "prefix_ids": ids, "prefix_text": "", "max_tokens": 4096})
    else:
        for r in records[: limit or None]:
            p = prepare(r, tok)
            if p is None:
                skipped += 1
            else:
                prepared.append(p)
    llm = LLM(model=str(model), tokenizer=str(tokenizer), dtype="bfloat16",
              max_model_len=max_model_len, seed=0, limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.generate([TokensPrompt(prompt_token_ids=p["prefix_ids"]) for p in prepared],
                        [SamplingParams(temperature=0.0, max_tokens=p["max_tokens"])
                         for p in prepared])
    with Path(out).open("w", encoding="utf-8") as fh:
        for p, o in zip(prepared, outs):
            fh.write(json.dumps({"call_id": p["call_id"], "type": p["type"],
                                 "prefix_text": p["prefix_text"], "gen_text": o.outputs[0].text,
                                 "gen_ids": list(o.outputs[0].token_ids)},
                                ensure_ascii=False) + "\n")
    print(f"generated {len(prepared)} ({skipped} pairs with no split point skipped) -> {out}")


def push(merged: Path, repo: str, revision: str, smoke_json: Path) -> None:
    import os

    from huggingface_hub import HfApi

    smoke = _read_jsonl(smoke_json)
    assert len(smoke) == 3, f"expected 3 smoke replies, got {len(smoke)}"
    for row in smoke:
        assert isinstance(json.loads(row["gen_text"])["edges"], list), row["call_id"]
    api = HfApi(token=os.environ["HF_TOKEN"])
    refs = {b.name: b.target_commit for b in api.list_repo_refs(repo, repo_type="model").branches}
    assert refs.get(revision, refs["main"]) == refs["main"], (
        f"{repo}@{revision} already has its own commits -- choose a new revision")
    api.create_branch(repo_id=repo, repo_type="model", branch=revision, exist_ok=True)
    index = json.loads((merged / "model.safetensors.index.json").read_text())
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(merged), revision=revision,
                      delete_patterns=["*.safetensors", "*.gguf"],
                      commit_message="dpo4: DPO on opus-distill-2026-10-ckpt100, merged bf16")
    pushed = {f for f in api.list_repo_files(repo, repo_type="model", revision=revision)
              if f.endswith((".safetensors", ".gguf"))}
    assert pushed == set(index["weight_map"].values()), f"Hub shards != merged index: {pushed}"
    print(f"DONE: {repo}@{revision}")


def score(val: Path, base_gen: Path, dpo_gen: Path, out: Path, smoke: bool) -> int:
    records = {r["meta"]["call_id"]: r for r in _read_jsonl(val)}
    gb = {r["call_id"]: r for r in _read_jsonl(base_gen)}
    gd = {r["call_id"]: r for r in _read_jsonl(dpo_gen)}
    ids = [cid for cid in records if cid in gb and cid in gd]
    if smoke:
        same = all(gb[c]["gen_ids"] == gd[c]["gen_ids"] for c in ids)
        Path(out).write_text(json.dumps({"pairs": len(ids), "identical": same}))
        print(f"smoke: {len(ids)} pairs, token-identical: {same}")
        return 2 if same or not ids else 0
    rows = {lbl: [score_pair(records[c], g[c]["prefix_text"], g[c]["gen_text"]) for c in ids]
            for lbl, g in (("ckpt100", gb), ("dpo", gd))}
    tot = {k: {lbl: totals(rows[lbl], k) for lbl in rows} for k in ("A", "B", "C")}
    passed, why = decide(tot["A"]["ckpt100"], tot["A"]["dpo"])
    a_b = [r for r in rows["ckpt100"] if r["type"] == "A"]
    a_d = [r for r in rows["dpo"] if r["type"] == "A"]
    gate = {"passed": passed, "why": why, "pairs_scored": len(ids),
            "pairs_missing": len(records) - len(ids), "totals": tot,
            "stats": {**bootstrap(a_b, a_d),
                      "sign_removed": sign_test([r["removed_written"] for r in a_b],
                                                [r["removed_written"] for r in a_d]),
                      "sign_kept": sign_test([r["kept_written"] for r in a_b],
                                             [r["kept_written"] for r in a_d])},
            "bar": {"MAX_REMOVED_SHARE": MAX_REMOVED_SHARE, "MIN_KEPT_SHARE": MIN_KEPT_SHARE,
                    "MIN_BASELINE_REMOVED": MIN_BASELINE_REMOVED},
            "per_pair": rows}
    Path(out).write_text(json.dumps(gate, indent=1))
    print(json.dumps({k: v for k, v in gate.items() if k != "per_pair"}, indent=1))
    print("GATE PASSED" if passed else "GATE FAILED: " + "; ".join(why))
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dpo_gate")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--base", type=Path, required=True)
    m.add_argument("--adapter", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    gn = sub.add_parser("generate")
    gn.add_argument("--model", type=Path, required=True)
    gn.add_argument("--tokenizer", type=Path, required=True)
    gn.add_argument("--val", type=Path, required=True)
    gn.add_argument("--out", type=Path, required=True)
    gn.add_argument("--limit", type=int, default=0)
    gn.add_argument("--full", action="store_true")
    gn.add_argument("--max-model-len", type=int, default=32768)
    s = sub.add_parser("score")
    s.add_argument("--val", type=Path, required=True)
    s.add_argument("--base", type=Path, required=True)
    s.add_argument("--dpo", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--smoke", action="store_true")
    p = sub.add_parser("push")
    p.add_argument("--merged", type=Path, required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--smoke-json", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "merge":
        merge(a.base, a.adapter, a.out)
    elif a.cmd == "generate":
        generate(a.model, a.tokenizer, a.val, a.out, a.limit, a.full, a.max_model_len)
    elif a.cmd == "score":
        return score(a.val, a.base, a.dpo, a.out, a.smoke)
    elif a.cmd == "push":
        push(a.merged, a.repo, a.revision, a.smoke_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Write `scripts/distill/dpo_gate_remote.sh`**

```bash
#!/bin/bash
# dpo4 gate, ON the Vast instance (spec 2026-10-24 section 6). The local driver copies
# dpo_gate.py, chat_format.py and val.jsonl to $W, then runs:
#   ssh ... 'bash /workspace/dpo4/dpo_gate_remote.sh gate <adapter-revision>' < hf-token
#   ssh ... 'bash /workspace/dpo4/dpo_gate_remote.sh push'                    < hf-token
# The HF token is read from stdin (one line) into the environment: never on a command
# line, never written to disk. ckpt100 is already on the instance's disk from the capture.
set -euo pipefail
CMD="${1:?gate <adapter-revision> | push}"
W=/workspace/dpo4
BASE=/workspace/hf/model-opus-distill-2026-10-ckpt100
ADAPTER_REPO=carev01/qwen35-4b-graphrag-dpo-adapters
EXPORT_REPO=carev01/qwen35-4b-graphrag
EXPORT_REV=opus-distill-2026-10-dpo4
read -r HF_TOKEN
export HF_TOKEN
source /venv/main/bin/activate
cd "$W"
[ -s "$BASE/config.json" ] || { echo "ckpt100 is not at $BASE"; exit 1; }

if [ "$CMD" = gate ]; then
  REV="${2:?adapter revision, e.g. dpo4-ckpt40}"
  supervisorctl stop vllm >/dev/null 2>&1 || true
  pkill -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 5
  echo "GPU memory in use after stopping the server: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  python -c "import peft" 2>/dev/null || pip install -q peft
  rm -rf adapter merged
  hf download "$ADAPTER_REPO" --revision "$REV" --local-dir adapter \
      --include adapter_model.safetensors adapter_config.json
  unset HF_TOKEN
  python dpo_gate.py merge --base "$BASE" --adapter adapter --out merged
  python dpo_gate.py generate --model "$BASE" --tokenizer "$BASE" --val val.jsonl --out smoke-base.jsonl --limit 2
  python dpo_gate.py generate --model merged --tokenizer "$BASE" --val val.jsonl --out smoke-dpo.jsonl --limit 2
  python dpo_gate.py score --smoke --val val.jsonl --base smoke-base.jsonl --dpo smoke-dpo.jsonl --out smoke.json \
    || { echo "smoke: both models' continuations are token-identical -- adapter not engaged; no gate"; exit 2; }
  python dpo_gate.py generate --model "$BASE" --tokenizer "$BASE" --val val.jsonl --out gen-base.jsonl
  python dpo_gate.py generate --model merged --tokenizer "$BASE" --val val.jsonl --out gen-dpo.jsonl
  python dpo_gate.py generate --model merged --tokenizer "$BASE" --val val.jsonl --out smoke-json.jsonl --full --limit 3
  set +e
  python dpo_gate.py score --val val.jsonl --base gen-base.jsonl --dpo gen-dpo.jsonl --out gate.json
  rc=$?
  set -e
  echo "gate exit code: $rc (0 pass, 1 fail)"
  exit $rc
elif [ "$CMD" = push ]; then
  python -c "import json; assert json.load(open('gate.json'))['passed'], 'gate did not pass'"
  python dpo_gate.py push --merged merged --repo "$EXPORT_REPO" --revision "$EXPORT_REV" --smoke-json smoke-json.jsonl
else
  echo "unknown command $CMD"; exit 1
fi
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_distill_dpo_gate.py -q && uv run ruff check scripts/distill/dpo_gate.py tests/unit/test_distill_dpo_gate.py && bash -n scripts/distill/dpo_gate_remote.sh`
Expected: all pass; ruff clean; `bash -n` silent.

- [ ] **Step 6: Commit**

```bash
chmod +x scripts/distill/dpo_gate_remote.sh
git add scripts/distill/dpo_gate.py scripts/distill/dpo_gate_remote.sh tests/unit/test_distill_dpo_gate.py
git commit -m "feat(distill): dpo4 gate on Vast -- bf16 merge, vLLM greedy continuation, unchanged bar"
```

---

### Task 5: Selection rule and the training notebook

**Files:**
- Create: `scripts/distill/dpo_select.py`
- Create: `scripts/distill/build_dpo_nb.py`, copied from `/tmp/claude-1000/-home-openclaw-Documents-Shared-projects-graph-rag/964252d0-58b1-4131-93c7-53185320cfde/scratchpad/build_dpo_nb.py` and then edited
- Modify: `notebooks/distill_dpo.ipynb` (regenerated, never hand-edited)
- Test: `tests/unit/test_distill_dpo_select.py`, `tests/unit/test_distill_build_dpo_nb.py`

**Interfaces:**
- Consumes: the dataset files from Task 3 (`train.jsonl`, `val.jsonl`, `val_eval.jsonl`, with `meta.type` and `meta.delta`).
- Produces:
  - `dpo_select.LOGPS_KEY = "eval_logps/chosen"`, `eval_logps(log_history) -> list[tuple[int, float]]`, `below_line(start, value, tolerance) -> bool`, `select_step(log_history, tolerance=1.0) -> int | None`;
  - the notebook's selection cell writes `ADAPTER_REVISION = f"dpo4-ckpt{SELECTED_STEP}"` to `carev01/qwen35-4b-graphrag-dpo-adapters`;
  - `build_dpo_nb.py [OUT]` writes the notebook (default `notebooks/distill_dpo.ipynb`).

- [ ] **Step 1: Write the failing selection tests**

```python
"""dpo_select: the dpo4 checkpoint rule -- the last eval within 1 nat of step 0's
logps/chosen, stopping at the first breach (spec 2026-10-24 section 5)."""
from __future__ import annotations

import pytest

from scripts.distill import dpo_select as s


def _h(step: int, logps: float) -> dict:
    return {"step": step, "eval_logps/chosen": logps, "eval_loss": 0.5}


def test_select_step_takes_the_last_eval_before_the_first_breach() -> None:
    hist = [_h(0, -45.0), {"step": 5, "loss": 0.6}, _h(10, -44.4), _h(20, -45.9), _h(30, -46.2),
            _h(40, -45.5)]
    assert s.select_step(hist) == 20         # 30 breaches (-46.2 < -46.0); 40 never counts


def test_select_step_none_when_the_first_eval_breaches() -> None:
    assert s.select_step([_h(0, -45.0), _h(10, -47.0)]) is None


def test_select_step_survives_unsorted_and_duplicate_entries_of_a_resumed_run() -> None:
    hist = [_h(20, -45.5), _h(0, -45.0), _h(10, -44.0), _h(10, -44.0), _h(30, -50.0)]
    assert s.select_step(hist) == 20


def test_select_step_needs_a_step_zero_eval() -> None:
    with pytest.raises(ValueError):
        s.select_step([_h(10, -45.0)])


def test_below_line_is_strict() -> None:
    assert not s.below_line(-45.0, -46.0, 1.0)
    assert s.below_line(-45.0, -46.01, 1.0)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_distill_dpo_select.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write `scripts/distill/dpo_select.py`**

The notebook embeds this source verbatim. Use no imports and no `from __future__`.

```python
"""dpo4 checkpoint selection (spec 2026-10-24-dpo-coverage-pairs-design.md section 5).
dpo3's val loss kept falling while the clean replies' log-probability collapsed, so val
loss picked the worst checkpoint. The rule: the last checkpoint whose eval logps/chosen is
within `tolerance` nats of its step-0 value, stopping at the first eval below that line.
build_dpo_nb.py embeds this module verbatim in the notebook: the unit tests and Colab run
the same code."""

LOGPS_KEY = "eval_logps/chosen"


def eval_logps(log_history):
    """(step, eval logps/chosen) per eval, sorted, one per step (a resumed run repeats some)."""
    by_step = {h["step"]: h[LOGPS_KEY] for h in log_history if LOGPS_KEY in h}
    return sorted(by_step.items())


def below_line(start, value, tolerance):
    return value < start - tolerance


def select_step(log_history, tolerance=1.0):
    """The selected step, or None when no eval after step 0 stays within the line."""
    series = eval_logps(log_history)
    if not series or series[0][0] != 0:
        raise ValueError("no step-0 eval (eval_on_start) in the log history")
    start = series[0][1]
    chosen = None
    for step, value in series[1:]:
        if below_line(start, value, tolerance):
            break
        chosen = step
    return chosen
```

- [ ] **Step 4: Run the selection tests to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_distill_dpo_select.py -q`
Expected: PASS.

- [ ] **Step 5: Move the builder into the repo, unchanged, and prove it reproduces the committed notebook**

```bash
cp /tmp/claude-1000/-home-openclaw-Documents-Shared-projects-graph-rag/964252d0-58b1-4131-93c7-53185320cfde/scratchpad/build_dpo_nb.py scripts/distill/build_dpo_nb.py
uv run python scripts/distill/build_dpo_nb.py && git diff --exit-code notebooks/distill_dpo.ipynb
git add scripts/distill/build_dpo_nb.py
git commit -m "chore(distill): move the DPO notebook builder into the repo (verbatim; regenerates the committed notebook)"
```

Expected: `git diff --exit-code` exits 0. If it doesn't, stop. The committed notebook and the builder have drifted, and that must be resolved before editing.

- [ ] **Step 6: Write the failing builder test**

```python
"""build_dpo_nb.py: the dpo4 notebook carries the approved settings, embeds dpo_select
verbatim, and no longer gates or exports from Colab (spec 2026-10-24 section 5)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _cells(tmp_path: Path) -> list[str]:
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, str(REPO / "scripts/distill/build_dpo_nb.py"), str(out)],
                   check=True, cwd=REPO)
    return ["".join(c["source"]) for c in json.loads(out.read_text())["cells"]]


def test_dpo4_settings_and_no_colab_gate(tmp_path: Path) -> None:
    cells = _cells(tmp_path)
    text = "\n".join(cells)
    for needle in ('HF_DATASET_REVISION = "dpo4"', "LEARNING_RATE = 2e-5", "NLL_ALPHA = 2.0",
                   "SAVE_TOTAL_LIMIT = None", "load_best_model_at_end=False",
                   "LOGPS_TOLERANCE = 1.0",
                   'HF_ADAPTER_REPO = "carev01/qwen35-4b-graphrag-dpo-adapters"',
                   '"val_eval.jsonl"', "class LikelihoodStop", "eval_dataset=build(val_eval_records",
                   'ADAPTER_REVISION = f"dpo4-ckpt{SELECTED_STEP}"'):
        assert needle in text, needle
    for gone in ("def rest_from_split", "## 10a. Export", "save_pretrained_merged",
                 'metric_for_best_model="eval_loss"', "trainer.evaluate()"):
        assert gone not in text, gone


def test_embedded_selection_is_the_module_and_runs(tmp_path: Path) -> None:
    src = (REPO / "scripts/distill/dpo_select.py").read_text()
    cell = next(c for c in _cells(tmp_path) if "def select_step" in c)
    assert src.strip() in cell
    ns: dict = {}
    exec(src, ns)
    hist = [{"step": 0, "eval_logps/chosen": -45.0}, {"step": 10, "eval_logps/chosen": -45.5}]
    assert ns["select_step"](hist, 1.0) == 10
```

Run: `uv run --extra dev pytest tests/unit/test_distill_build_dpo_nb.py -q`
Expected: FAIL. The builder takes no output argument yet and has none of the dpo4 settings.

- [ ] **Step 7: Edit `scripts/distill/build_dpo_nb.py`**

Make each change below as an exact text replacement in the builder. Every "old" snippet is quoted from the builder as it stands after Step 5.

**7a. Output path argument.** Replace `out = REPO / "notebooks/distill_dpo.ipynb"` with:

```python
out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "notebooks/distill_dpo.ipynb"
```

Below the `REPO = ...` line, add:

```python
SELECT_SRC = (REPO / "scripts/distill/dpo_select.py").read_text()
```

**7b. Header markdown.** Replace the first `md("""\n# DPO round -- ...` cell, up to and including its line `A dead runtime resumes from the last checkpoint on Drive.\n"""),`, with:

```python
    md("""
# DPO round dpo4 -- qwen3.5-4b extraction model (Colab)

Starts from the SFT checkpoint `opus-distill-2026-10-ckpt100` and runs one round of
Direct Preference Optimisation on coverage-balanced pairs. It then uploads ONE selected
LoRA adapter to the private repo `carev01/qwen35-4b-graphrag-dpo-adapters`. The gate and
the export run on the Vast GPU (`scripts/distill/dpo_gate.py`), never from Colab.

**Why dpo4 (2026-10-24).** dpo3's pairs were all deletions (`chosen` = `rejected` minus the
unsupported edges), so "write fewer edges" scored on every pair. Its checkpoint-10 cut the
removed edges to 0.71 of ckpt100's but the kept ones to 0.85, failing the 0.90 bar.

**The dataset** (`scripts/distill/dpo_coverage.py`, labels by Claude subagents) has three
pair types, each from ckpt100's own replies:
- **A** drops the confirmed-unsupported edges;
- **B** keeps a supported edge that `rejected` drops;
- **C** appends the supported edges ckpt100 missed.

The mean edge-count difference between chosen and rejected is within +/-0.1, so length
carries no signal.

**Checkpoint selection by likelihood, not val loss** (dpo3: val loss kept falling while
`logps/chosen` collapsed). The rule is the last checkpoint whose eval `logps/chosen` is
within `LOGPS_TOLERANCE` (1 nat) of step 0's. `LikelihoodStop` ends training at the first
eval below that line, since nothing after it could be selected.

**Guards kept from dpo3:** causal_conv1d kernel off for training, torch.compile off, the
gradient guard, `eval_on_start`, and the check that the trainer trains the serving prompt.

**Use:** set the HF token secret (section 2), check section 3, then **Runtime -> Run all**.
A dead runtime resumes from the last checkpoint on Drive. Hand back the adapter revision
that section 10 prints.
"""),
```

**7c. Config cell.** Make these replacements inside the section-3 code cell:

| old | new |
|---|---|
| `HF_DATASET_REVISION = "dpo1"` | `HF_DATASET_REVISION = "dpo4"` |
| `DRIVE_DATA_DIR = "/content/drive/MyDrive/ft-opus-dpo1"   # only when DATA_SOURCE == "drive"` | `DRIVE_DATA_DIR = "/content/drive/MyDrive/ft-opus-dpo4"   # only when DATA_SOURCE == "drive"` |
| `RUN_NAME = "opus-distill-dpo3"     # dpo1: 5e-6, too weak; dpo2: 5e-5 on a broken kernel` | `RUN_NAME = "opus-distill-dpo4"     # dpo3: 5e-5, deletion-only pairs taught brevity` |
| `NLL_ALPHA = 1.0            # NLL on chosen (rpo_alpha / "sft" loss): protects coverage; 0 = plain DPO` | `NLL_ALPHA = 2.0            # NLL on chosen ("sft" loss, a per-token mean); 1.0 did not stop dpo3's displacement` |
| `LEARNING_RATE = 5e-5       # dpo1's 5e-6 moved the split-point log-odds by only +0.34` | `LEARNING_RATE = 2e-5       # dpo3 5e-5 displaced chosen by step 20; dpo1 5e-6 did nothing` |
| `SAVE_TOTAL_LIMIT = 4` | `SAVE_TOTAL_LIMIT = None     # adapters are small; selection needs every checkpoint` |

Then replace the whole block from `# --- export ---...` through `HF_MODEL_PRIVATE = False   # the repo is public by design; a branch cannot be more private` with:

```python
LOGPS_TOLERANCE = 1.0      # selection: eval logps/chosen may fall at most this far below step 0

# --- hand-off: ONE adapter to a PRIVATE repo; the gate and export run on Vast ---------------
HF_ADAPTER_REPO = "carev01/qwen35-4b-graphrag-dpo-adapters"
```

**7d. Data cell.** Replace `DATA_DIR = Path("/content/data/ft-opus-dpo1")` with `DATA_DIR = Path("/content/data/ft-opus-dpo4")`. Replace `for fname in ("train.jsonl", "val.jsonl", "MANIFEST.json", "sample.json"):` with `for fname in ("train.jsonl", "val.jsonl", "val_eval.jsonl", "MANIFEST.json", "sample.json"):`. Replace the assert line with:

```python
assert all((DATA_DIR / f).exists() for f in ("train.jsonl", "val.jsonl", "val_eval.jsonl")), DATA_DIR
```

**7e. Section 5 (markdown and code).** Replace the section-5 markdown body with:

```python
    md("""
## 5. Load the pairs and check their shape

Every record: `messages` = `[system, user]` exactly as vLLM received them; `chosen` and
`rejected` = the reply JSON. An A pair's chosen is a strict subset of rejected; a B or C
pair's rejected is a strict subset of chosen. `meta.delta` must match, and the train set's
mean delta must be within +/-0.1. `val_eval.jsonl` is the trainer's fixed eval subset.
"""),
```

Then replace the section-5 code, from `train_records = read_jsonl(DATA_DIR / "train.jsonl")` through `assert len(train_records) >= 50, "too few pairs for a DPO round -- capture more first"`, with:

```python
train_records = read_jsonl(DATA_DIR / "train.jsonl")
val_records = read_jsonl(DATA_DIR / "val.jsonl")
val_eval_records = read_jsonl(DATA_DIR / "val_eval.jsonl")
for _r in train_records + val_records:
    assert [m["role"] for m in _r["messages"]] == ["system", "user"], _r["meta"]
    _c, _x = json.loads(_r["chosen"])["edges"], json.loads(_r["rejected"])["edges"]
    _small, _big = (_c, _x) if _r["meta"]["type"] == "A" else (_x, _c)
    assert 0 < len(_small) < len(_big), _r["meta"]
    assert all(e in _big for e in _small), _r["meta"]
    assert _r["meta"]["delta"] == len(_c) - len(_x), _r["meta"]
_val_ids = {r["meta"]["call_id"] for r in val_records}
assert all(r["meta"]["call_id"] in _val_ids for r in val_eval_records), "val_eval is not a subset of val"
for _name, _rows in (("train", train_records), ("val", val_records), ("val_eval", val_eval_records)):
    _types = Counter(r["meta"]["type"] for r in _rows)
    _mean = sum(r["meta"]["delta"] for r in _rows) / len(_rows)
    print(f"{_name}: {len(_rows)} pairs {dict(sorted(_types.items()))}, mean edge delta {_mean:+.3f}")
assert abs(sum(r["meta"]["delta"] for r in train_records) / len(train_records)) <= 0.1, (
    "train set is not length-balanced (spec: |mean delta| <= 0.1)")
assert len(train_records) >= 50, "too few pairs for a DPO round -- capture more first"
```

**7f. `build()` length check.** Replace

```python
        n = len(TEXT_TOK(p["prompt"] + p["rejected"], add_special_tokens=False)["input_ids"])
```

with

```python
        n = max(len(TEXT_TOK(p["prompt"] + p[k], add_special_tokens=False)["input_ids"])
                for k in ("chosen", "rejected"))   # a C pair's chosen is the longer one
```

**7g. Trainer config.** Replace

```python
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
```

with

```python
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=False,   # dpo4: selection by logps/chosen (section 10), not val loss
```

**7h. LikelihoodStop.** The section-8 code cell is one `code("""...""")` literal. Split it just before `def make_trainer(mark: str) -> DPOTrainer:` so the cell becomes `code("""<everything up to and including the GradGuard class>\n\n""" + SELECT_SRC + """\n\n<the class below>\n\n\n<def make_trainer and the rest of the cell>""")`. That puts `LOGPS_KEY`, `eval_logps`, `below_line` and `select_step` in the notebook namespace, from the module source itself. The class, as it appears inside that `"""` literal:

```python
class LikelihoodStop(TrainerCallback):
    # The dpo4 selection rule, live: stop at the first eval whose logps/chosen falls more
    # than LOGPS_TOLERANCE below step 0's -- nothing after it could be selected. Stateless
    # (reads state.log_history), so a resumed run keeps its step-0 baseline.
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if LOGPS_KEY not in (metrics or {}):
            raise RuntimeError(f"eval metrics have no {LOGPS_KEY}: the selection rule cannot run")
        series = eval_logps(state.log_history)
        if not series or series[0][0] != 0:
            raise RuntimeError("no step-0 eval in the log history -- is eval_on_start off?")
        start, (step, value) = series[0][1], series[-1]
        if step and below_line(start, value, LOGPS_TOLERANCE):
            print(f"  step {step}: eval logps/chosen {value:.2f} < {start:.2f} - "
                  f"{LOGPS_TOLERANCE} -- stopping (selection rule)", flush=True)
            control.should_training_stop = True
        return control
```

Trainer.evaluate logs the metrics before it calls `on_evaluate`, so `series[-1]` is the current eval.

**7i. Trainer construction.** Replace

```python
                      eval_dataset=build(val_records, mark), callbacks=[GradGuard()],
```

with

```python
                      eval_dataset=build(val_eval_records, mark),
                      callbacks=[GradGuard(), LikelihoodStop()],
```

**7j. Section 9 (train).** Replace the section-9 markdown with:

```python
    md("""
## 9. Train (resumable)

Watch `eval_logps/chosen` at every eval: section 10 keeps the last checkpoint within
`LOGPS_TOLERANCE` of step 0's value, and training stops by itself at the first eval below
that line ("stopping (selection rule)"). `eval_rewards/accuracies` well above 0.5 means
the model separates the pairs.

The gradient guard prints "grad guard: inspecting gradients..." once, then the gradient
norm of each of the first `GRAD_NORM_FROM_STEP` steps; it stops on a NaN/inf gradient, or
a norm above `GRAD_NORM_LIMIT` later.
"""),
```

In the section-9 code, delete these lines:

```python
    eval_metrics = trainer.evaluate()   # the best checkpoint (load_best_model_at_end)
    print(_json.dumps(eval_metrics, indent=2))
```

```python
    (DRIVE_RUN_DIR / "eval_metrics.json").write_text(_json.dumps(eval_metrics, indent=2))
```

```python
    print(f"best checkpoint: {trainer.state.best_model_checkpoint}")
    _acc = eval_metrics.get("eval_rewards/accuracies")
    if _acc is not None and _acc < 0.6:
        print(f"WARNING: held-out reward accuracy {_acc:.2f} -- DPO barely separated the pairs")
```

**7l. Section 6 markdown.** Replace `saved adapter records it as its base, and the export (section 10) reloads from there.` with `saved adapter records it as its base.` (there is no Colab export any more).

**7k. Sections 10, 10a, 11.** Delete the `## 10. Gate` markdown, its code cell, the `## 10a. Export` markdown and its code cell. Replace the `## 11. Next` markdown with these three cells:

```python
    md("""
## 10. Select the checkpoint and hand it off

Applies the rule to `log_history.json` and prints the trajectory. If no checkpoint after
step 0 qualifies, the round fails here, with no gate. Otherwise the selected checkpoint's
adapter goes to the PRIVATE repo `HF_ADAPTER_REPO`, revision `dpo4-ckpt<step>`.
"""),
    code("""
import json as _json

from huggingface_hub import HfApi

_hist = _json.loads((DRIVE_RUN_DIR / "log_history.json").read_text())
for _s, _v in eval_logps(_hist):
    print(f"eval @ {_s:>4}: logps/chosen {_v:.2f}")
SELECTED_STEP = select_step(_hist, LOGPS_TOLERANCE)
assert SELECTED_STEP is not None, (
    f"no checkpoint after step 0 kept logps/chosen within {LOGPS_TOLERANCE} nat of step 0's "
    "-- the round fails here, no gate (spec section 5)")
_ckpt = CKPT_DIR / f"checkpoint-{SELECTED_STEP}"
assert (_ckpt / "adapter_model.safetensors").exists(), f"{_ckpt} has no adapter"
ADAPTER_REVISION = f"dpo4-ckpt{SELECTED_STEP}"
api = HfApi(token=HF_TOKEN)
api.create_repo(HF_ADAPTER_REPO, repo_type="model", private=True, exist_ok=True)
assert api.model_info(HF_ADAPTER_REPO).private, f"{HF_ADAPTER_REPO} is not private -- refusing"
api.create_branch(HF_ADAPTER_REPO, repo_type="model", branch=ADAPTER_REVISION, exist_ok=True)
for _f in ("adapter_model.safetensors", "adapter_config.json"):
    api.upload_file(path_or_fileobj=str(_ckpt / _f), path_in_repo=_f, repo_id=HF_ADAPTER_REPO,
                    repo_type="model", revision=ADAPTER_REVISION)
api.upload_file(path_or_fileobj=str(DRIVE_RUN_DIR / "log_history.json"),
                path_in_repo="log_history.json", repo_id=HF_ADAPTER_REPO, repo_type="model",
                revision=ADAPTER_REVISION)
print(f"DONE: {HF_ADAPTER_REPO}@{ADAPTER_REVISION} -- hand this name back")
"""),
    md("""
## 11. Next -- outside this notebook

Hand back the adapter revision. The gate runs on the Vast GPU against the bf16 merge
(`scripts/distill/dpo_gate_remote.sh gate <revision>`), with the pass bar unchanged from
dpo3. Only after a pass, and with authorisation, the same merged weights are pushed to
`carev01/qwen35-4b-graphrag@opus-distill-2026-10-dpo4` and the adoption trial runs.
"""),
```

- [ ] **Step 8: Regenerate and run the builder tests**

Run: `uv run python scripts/distill/build_dpo_nb.py && uv run --extra dev pytest tests/unit/test_distill_build_dpo_nb.py tests/unit/test_distill_dpo_select.py -q && uv run ruff check scripts/distill/dpo_select.py scripts/distill/build_dpo_nb.py tests/unit/test_distill_build_dpo_nb.py tests/unit/test_distill_dpo_select.py`
Expected: the builder prints `all code cells compile`; tests pass; ruff clean. If ruff flags the builder's long embedded strings (E501), add `# ruff: noqa: E501` at the top of the builder: it is a generator of notebook text.

- [ ] **Step 9: Commit**

```bash
git add scripts/distill/dpo_select.py scripts/distill/build_dpo_nb.py notebooks/distill_dpo.ipynb tests/unit/test_distill_dpo_select.py tests/unit/test_distill_build_dpo_nb.py
git commit -m "feat(dpo notebook): dpo4 -- likelihood selection + stop, lr 2e-5, NLL 2.0, val_eval, adapter hand-off"
```

`notebooks/distill_dpo_eval.ipynb` (the dpo3 Colab gate) is left as committed. Its scratchpad builder reads cells this task removed, so it must not be regenerated.

---

### Task 6: Runbook section 9

**Files:**
- Modify: `docs/superpowers/distill-runbook.md` (append after section 8)

- [ ] **Step 1: Append section 9**

````markdown
## 9. dpo4: coverage-balanced pairs, Claude labels, gate on Vast (2026-10-24)

Spec: `docs/superpowers/specs/2026-10-24-dpo-coverage-pairs-design.md`. Data in
`graph-rag-distill/2026-10-24-dpo4/` (`D`). dpo3's capture and DeepSeek labels are in
`graph-rag-distill/2026-10-21-dpo/` (`D3`).

```bash
# 1. Calibration: Sonnet vs Opus on 100 existing replies (bar: recall >= 0.90)
uv run --extra dev python -m scripts.distill.dpo_labels calibration-sample \
    --capture D3/capture-run/capture-v2.jsonl --deepseek D3/pairs --out D/calibration-ids.json
for m in sonnet opus; do
  uv run --extra dev python -m scripts.distill.dpo_labels tasks --kind label \
      --capture D3/capture-run/capture-v2.jsonl --ids D/calibration-ids.json --store D/store/label-$m
done
#    batches -> subagents (model per store, DPO_LABEL_BRIEF.md) -> intake, until nothing is pending
uv run --extra dev python -m scripts.distill.dpo_labels calibrate --ids D/calibration-ids.json \
    --sonnet D/store/label-sonnet --opus D/store/label-opus --out D/calibration.json
# 2. Fresh articles (none from any earlier sample) and the Vast capture (instance STOPPED after)
uv run --extra dev python -m scripts.distill.dpo_sample --out D/dpo_sample.json --articles 1000 \
    --seed 20261024 --exclude DISTILL/sample.json --exclude TRIAL/d6_sample.json \
    --exclude D3/dpo_sample.json
# 3. Labelling with the calibrated labeller L, the Opus re-check, the Opus missed edges
C="--capture D3/capture-run/capture-v2.jsonl --capture D/capture-run/capture-v2.jsonl"
uv run --extra dev python -m scripts.distill.dpo_labels tasks --kind label $C --store D/store/label-L
uv run --extra dev python -m scripts.distill.dpo_labels tasks --kind recheck $C \
    --labels D/store/label-L --store D/store/recheck-opus
uv run --extra dev python -m scripts.distill.dpo_labels tasks --kind missed $C \
    --labels D/store/label-L --recheck D/store/recheck-opus --store D/store/missed-opus --n 600
# 4. Pairs and reports
uv run --extra dev python -m scripts.distill.dpo_coverage $C --labels D/store/label-L \
    --labeller L --recheck D/store/recheck-opus --missed D/store/missed-opus \
    --deepseek D3/pairs --out D/pairs
# 5. Package (private) -> notebooks/distill_dpo.ipynb on Colab -> adapter revision dpo4-ckpt<step>
# 6. Gate on the restarted instance: scripts/distill/dpo_gate_remote.sh gate dpo4-ckpt<step>
```

The gate is the dpo3 bar, unchanged, on the A val pairs, against the bf16 merge served by
vLLM. B and C are reported. A failed gate ends DPO: ckpt100 stays.
````

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/distill-runbook.md
git commit -m "docs(runbook): section 9 -- dpo4 commands"
```

---

## Operational tasks (spec section 8, steps 2-9)

These run after Tasks 1-6 are merged locally. They spend plan usage and the authorised Vast money. They follow CLAUDE.md's cost rule: smoke on about 2 items with a hard assertion before any full run.

### O0: HF token route (ask the user once, before O3)

Three steps need a token: the dataset push (O4), the adapter download on Vast (O6), and the export push (O7). No token is stored on this machine. Ask the user to choose one:
- **(a)** they create `~/.config/graph-rag/hf_token` (mode 600, outside the repo, like the Vast keys), and the scripts pipe it over stdin or env;
- **(b)** they run those three commands themselves with `!`.

Claude never writes the token anywhere.

### O1: Calibration (plan usage)

- [ ] Run the runbook section 9 step 1 commands up to `tasks` for both stores.
- [ ] **Smoke:** `batches --store D/store/label-sonnet --size 2 --out D/batches/label-sonnet-smoke`. Dispatch ONE Sonnet subagent (Agent tool, `model: "sonnet"`) on `batch-001.jsonl` with `scripts/distill/DPO_LABEL_BRIEF.md`, then run `intake`. Assert `{"stored": 2}`. Do the same for Opus (`model: "opus"`). If anything other than 2 stored comes back, fix the brief or the code before any batch run.
- [ ] Full rounds per store: `batches --size 10` into a new `round-N` dir. Dispatch one subagent per batch file, at most 4 at a time. `intake`. Repeat until `status` shows 0 pending.
- [ ] `calibrate`. Record the labeller `L` and the recall in the ledger. Report the result to the user (report only; no stop).

### O2: Capture about 1,000 fresh articles on Vast (authorised, about $3)

- [ ] Run the runbook section 9 step 2 `dpo_sample` command (free, read-only). Assert 1,000 articles, none in any exclude file. The script enforces the exclusion; confirm the count.
- [ ] Copy `/tmp/claude-1000/-home-openclaw-Documents-Shared-projects-graph-rag/964252d0-58b1-4131-93c7-53185320cfde/scratchpad/vast_adopt.sh` to `.../scratchpad/vast_dpo4_capture.sh` and edit:
  - set `D=/home/openclaw/Documents/Shared/projects/graph-rag-distill/2026-10-24-dpo4`;
  - delete sections A and B (the adoption trial);
  - before section C, add a smoke run: `--articles 2 --out "$D/capture-smoke"`. Then assert that `capture-smoke/capture-v2.jsonl` holds `extract_edges.edge` lines from model `qwen35-graphrag` with finish reason `stop` (copy the section-B Python check, threshold `-gt 0`), and that none of the article ids is in the dpo3 `dpo_sample.json`;
  - section C: `--articles 1000`, `--out "$D/capture-run"`;
  - `FINAL=stop` on every path, success included. Write the instance id to `$D/vast-instance-id`.
- [ ] Run it in the background (`run_in_background`). Expect about 3.3 h. When it finishes, confirm through the Vast API that the instance is `stopped`, not destroyed.

### O3: Full labelling (plan usage)

- [ ] `tasks --kind label` over both captures into `D/store/label-L`. Calibration answers already under that store's keys are reused if `L` matches a calibration store: copy that store directory to `label-L` first.
- [ ] Batch rounds until 0 pending (labeller model `L`).
- [ ] `tasks --kind recheck` -> `D/store/recheck-opus`; Opus rounds until 0 pending.
- [ ] `tasks --kind missed --n 600` -> `D/store/missed-opus`; Opus rounds until 0 pending.

### O4: Build pairs, then STOP for the user's review

- [ ] Run the runbook section 9 step 4 command. Check that `summary.json` has `mean_delta` within ±0.1 per split (the builder asserts this), and roughly 150 val A pairs.
- [ ] Run the notebook harness against the real pairs: `cd /tmp/claude-1000/-home-openclaw-Documents-Shared-projects-graph-rag/964252d0-58b1-4131-93c7-53185320cfde/scratchpad/nbcheck && HF_HUB_OFFLINE=1 ../trlenv/bin/python run_cells.py processor /home/openclaw/Documents/Shared/projects/graph-rag/notebooks/distill_dpo.ipynb D/pairs w4-processor`. It must pass the section-5 shape check and build the trainer.
- [ ] Report to the user:
  - pairs per type per split;
  - mean deltas;
  - `labels_report.json`: calibration, Claude vs DeepSeek (how many of the 554 confirmed), stated-unverified share, missed-edge rejection reasons.

  **Stop until the user approves.**

### O5: Package, push, train (after approval)

- [ ] `cp D/dpo_sample.json D/pairs/sample.json`, then `package_dataset --in D/pairs --out D/dist/dpo4.zip --hf-repo carev01/ft-opus-distill --hf-revision dpo4`, using the token route from O0. Check that `val_eval.jsonl` is in the zip.
- [ ] Push the code commits only with the user's approval, after gitleaks (`~/.local/bin/gitleaks git --log-opts=<range> --no-banner .`). Colab pulls the notebook from GitHub.
- [ ] The user runs `notebooks/distill_dpo.ipynb` and hands back `dpo4-ckpt<step>`, or reports that no checkpoint qualified, in which case the round ends: go to O8.

### O6: Gate on Vast (about $1), then STOP

- [ ] Restart the stopped instance (`PUT .../instances/<id>/ {"state":"running"}`, then wait for ssh, as in `vast_adopt.sh`'s status loop).
- [ ] `scp scripts/distill/{dpo_gate.py,chat_format.py,dpo_gate_remote.sh} D/pairs/val.jsonl root@HOST:/workspace/dpo4/`.
- [ ] Run `ssh ... 'bash /workspace/dpo4/dpo_gate_remote.sh gate dpo4-ckpt<step>'`, with the token on stdin via the O0 route. It merges, asserts the adapter is engaged, smoke-tests 2 pairs, generates and scores.
- [ ] `scp root@HOST:/workspace/dpo4/gate.json D/gate.json`. **Stop the instance.** Report pass/fail with the totals and the bootstrap CIs. **Stop for the user.**

### O7: Pass only: export and adoption trial (needs authorisation)

- [ ] On authorisation: restart, run `dpo_gate_remote.sh push` (token on stdin), and confirm `DONE`.
- [ ] Price the adoption trial (runbook section 7, same 60 articles), get authorisation, and run it on this instance, provisioned with `MODEL_REVISION=opus-distill-2026-10-dpo4`. Then go to O8.

### O8: End of round

- [ ] Destroy the Vast instance, then confirm through the Vast API that it is no longer listed.
- [ ] Update memory `distill-colab-log-notes.md` with the outcome. On a fail, record that DPO has ended and ckpt100 stays.
````

