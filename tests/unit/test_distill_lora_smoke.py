"""lora_smoke: the checks a LoRA-serving rental must pass before any trial spend (spec
2026-10-29-verifier-trial-design.md section 3) -- the extractor adapter reproduces merged
ckpt100's captured replies, and the served verifier agrees with the notebook's predictions."""
from __future__ import annotations

import json

from scripts.distill import lora_smoke as ls


def _cap(cid: str, edges: list[tuple[str, str]], prompt: str = "extract_edges.edge",
         tier: str = "cheap", finish: str = "stop") -> str:
    payload = {"edges": [{"source_entity_name": s, "target_entity_name": t,
                          "relation_type": "R", "fact": f"{s}-{t}"} for s, t in edges]}
    return json.dumps({"call_id": cid, "prompt_name": prompt, "tier": tier, "model": "qwen35-graphrag",
                       "finish_reason": finish, "messages": [{"role": "user", "content": cid}],
                       "response_format": {"type": "json_object"},
                       "completion": json.dumps(payload)})


def test_replay_requests_pick_a_seeded_sample_of_finished_edge_replies() -> None:
    lines = [_cap(f"c{i}", [("A", "B")]) for i in range(30)]
    lines += [_cap("x1", [("A", "B")], prompt="dedupe_edges.resolve_edge"),
              _cap("x2", [("A", "B")], tier="strong"), _cap("x3", [("A", "B")], finish="length")]
    got = ls.replay_requests(lines, 10, seed=7)
    assert len(got) == 10 and len({r["call_id"] for r in got}) == 10
    assert all(r["call_id"].startswith("c") for r in got)
    assert got == ls.replay_requests(lines, 10, seed=7)
    assert set(got[0]) == {"call_id", "messages", "response_format", "completion"}


def test_extractor_equivalence_counts_identical_and_reports_overlap() -> None:
    a = json.loads(_cap("a", [("A", "B"), ("C", "D")]))["completion"]
    b = json.loads(_cap("b", [("A", "B"), ("E", "F")]))["completion"]
    rep = ls.extractor_equivalence([(a, a), (a, b), (a, "not json")])
    assert rep["identical"] == 1 and rep["n"] == 3 and rep["unparseable"] == 1
    assert rep["differing"] == [{"index": 1, "edge_jaccard": 1 / 3}, {"index": 2, "edge_jaccard": None}]
    assert not ls.extractor_passes(rep)
    assert ls.extractor_passes(ls.extractor_equivalence([(a, a)] * 45 + [(a, b)] * 5))
    assert not ls.extractor_passes(ls.extractor_equivalence([(a, a)] * 44 + [(a, b)] * 6))


def _labels(*verdicts: str) -> str:
    return json.dumps({"labels": [{"n": n, "verdict": v} for n, v in enumerate(verdicts, 1)]})


def test_verifier_agreement_is_per_fact_and_fails_on_unparseable() -> None:
    nb = {"r1": _labels("stated", "unsupported"), "r2": _labels("stated")}
    served = {"r1": _labels("stated", "stated"), "r2": _labels("stated")}
    rep = ls.verifier_agreement({"r1": 2, "r2": 1}, nb, served)
    assert rep == {"facts": 3, "agree": 2, "share": 2 / 3, "unparseable": 0}
    assert not ls.verifier_passes(rep)
    rep = ls.verifier_agreement({"r1": 2}, nb, {"r1": "garbage"})
    assert rep["unparseable"] == 1 and rep["agree"] == 0
    assert ls.verifier_passes({"facts": 100, "agree": 95, "share": 0.95, "unparseable": 0})


def test_no_empty_reply_coercion() -> None:
    import inspect
    assert 'or ""' not in inspect.getsource(ls)


def test_noise_floor_rule_is_the_pre_stated_one() -> None:
    # A = adapter vs merged-now, M = merged-now vs captured; equivalent iff A >= M - 5, all parse.
    assert ls.adapter_equivalent(identical_adapter=22, identical_merged=27, unparseable=0)
    assert not ls.adapter_equivalent(identical_adapter=21, identical_merged=27, unparseable=0)
    assert not ls.adapter_equivalent(identical_adapter=50, identical_merged=27, unparseable=1)
