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


def test_bootstrap_without_a_usable_denominator_does_not_crash() -> None:
    zero = [{"removed_written": 0, "kept_written": 3} for _ in range(5)]
    res = g.bootstrap(zero, zero, n=50, seed=1)
    assert res["removed_ratio"] is None and res["removed_ratio_ci"] == [None, None]
    assert g.bootstrap([], [], n=50, seed=1)["resamples"] == 0


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
