"""`dpo_sample`: excluded, navigation, overlong and dense articles stay out; sources are
drawn without replacement."""
from __future__ import annotations

import json
import random

from scripts.distill import dpo_sample as ds

KW = {"max_chars": 100, "ratio": 0.25, "pipes": 200}


def test_excluded_ids_reads_distillation_and_trial_samples(tmp_path) -> None:
    (tmp_path / "d.json").write_text(json.dumps(
        {"articles": [{"article_id": "a", "split": "train"},
                      {"article_id": "b", "split": "heldout"}]}))
    (tmp_path / "t.json").write_text(json.dumps([{"id": "c", "vendor": "V"}]))
    assert ds.excluded_ids([tmp_path / "d.json", tmp_path / "t.json"]) == {"a", "b", "c"}


def test_skip_reasons() -> None:
    ok = "Some prose about backups.\n"
    assert ds.skip_reason("x", "Restore a VM", ok, excluded={"x"}, **KW) == "excluded"
    assert ds.skip_reason("y", "Restore a VM", "  \n", excluded=set(), **KW) == \
        "navigation_or_empty"
    assert ds.skip_reason("y", "Restore a VM", "w" * 101, excluded=set(), **KW) == "too_long"
    table = "| a | b |\n| - | - |\n| 1 | 2 |\ntext\n"
    assert ds.skip_reason("y", "Matrix", table, excluded=set(), **KW) == "dense"
    assert ds.skip_reason("y", "Restore a VM", ok, excluded=set(), **KW) == ""


def test_pick_sources_is_distinct_and_skips_empty_sources() -> None:
    rows = [{"id": i, "article_count": c} for i, c in enumerate([5, 0, 3, 9])]
    got = ds.pick_sources(rows, 10, random.Random(1))
    assert sorted(x["id"] for x in got) == [0, 2, 3]
