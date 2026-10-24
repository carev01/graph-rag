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
    assert dc.quote_ok("supports  Amazon\nS3 as a backup repository", CUR)
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


def test_build_c_counts_a_non_string_entity_name_instead_of_crashing() -> None:
    rec = _rec(2)
    clean = dc.classify([_lab("stated")] * 2, {}, CUR)
    listed = {**_edge("X", "Y"), "source_entity_name": ["X"], "quote": Q}
    keyed = {**_edge("X", "Y"), "target_entity_name": {"name": "Y"}, "quote": Q}
    pair, why = dc.build_c(rec, clean, [listed, keyed], "opus")
    assert pair is None and why == {"entity_not_listed": 2}
