"""`upsample_train` -- repetition toward TARGET_MIX under each basis.

`basis="example"` exists for a trainer whose loss is a per-example mean (batch size
1, micro-batch loss averaged over its own tokens): there every example carries the
same gradient weight whatever its length, so the mix has to be hit in EXAMPLE share,
not token share.
"""
from __future__ import annotations

from collections import Counter

import pytest

from scripts.build_extraction_dataset import TARGET_MIX, upsample_train


def _rec(prompt: str, completion_chars: int = 40, i: int = 0) -> dict:
    return {"prompt_name": prompt,
            "messages": [{"role": "user", "content": "u" * 400},
                         {"role": "assistant", "content": "a" * completion_chars}],
            "meta": {"article_id": f"a{i}"}}


def _train(counts: dict[str, int], completion_chars: dict[str, int] | None = None) -> list[dict]:
    chars = completion_chars or {}
    return [_rec(p, chars.get(p, 40), i) for p, n in counts.items() for i in range(n)]


# The v1 Opus train split's unique records per prompt type.
V1_UNIQUE = {"extract_edges.edge": 243, "extract_nodes.extract_text": 243,
             "extract_nodes.extract_summaries_batch": 119,
             "dedupe_edges.resolve_edge": 706, "dedupe_nodes.nodes": 129}


def test_example_basis_repeats_by_example_share():
    out = upsample_train(_train(V1_UNIQUE), max_repeat=12, basis="example")
    n = Counter(r["prompt_name"] for r in out)
    # resolve_edge is the most over-represented by count -> binding, one copy.
    assert n == {"extract_edges.edge": 243 * 4, "extract_nodes.extract_text": 243 * 2,
                 "extract_nodes.extract_summaries_batch": 119 * 4,
                 "dedupe_edges.resolve_edge": 706, "dedupe_nodes.nodes": 129 * 2}
    total = sum(n.values())
    for p, want in TARGET_MIX.items():
        assert abs(n[p] / total - want) < 0.04, (p, n[p] / total)


def test_example_basis_ignores_record_length():
    long_edges = {"extract_edges.edge": 5000}
    short = upsample_train(_train(V1_UNIQUE), max_repeat=12, basis="example")
    long = upsample_train(_train(V1_UNIQUE, long_edges), max_repeat=12, basis="example")
    assert Counter(r["prompt_name"] for r in short) == Counter(r["prompt_name"] for r in long)


def test_completion_basis_still_weighs_tokens():
    # Same counts, but resolve_edge completions are tiny: under token share it is far
    # BELOW target and gets repeated, the opposite of the example-share answer.
    chars = {"extract_edges.edge": 2000, "dedupe_edges.resolve_edge": 20}
    out = upsample_train(_train(V1_UNIQUE, chars), max_repeat=12, basis="completion")
    n = Counter(r["prompt_name"] for r in out)
    assert n["dedupe_edges.resolve_edge"] > 706
    assert n["extract_edges.edge"] == 243


def test_copies_are_tagged_with_repeat_index_and_basis():
    out = upsample_train(_train(V1_UNIQUE), max_repeat=12, basis="example")
    edges = [r for r in out if r["prompt_name"] == "extract_edges.edge"
             and r["meta"]["article_id"] == "a0"]
    assert sorted(r["meta"]["repeat_index"] for r in edges) == [0, 1, 2, 3]
    assert {r["meta"]["upsample_basis"] for r in out} == {"example"}


def test_unknown_basis_is_rejected():
    with pytest.raises(ValueError, match="basis"):
        upsample_train(_train(V1_UNIQUE), max_repeat=12, basis="tokens")
