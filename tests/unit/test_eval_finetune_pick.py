"""`eval_finetune.pick_records`: a seeded per-prompt subsample, or the full held-out
set with `--per-prompt 0`; `--base-url` defaults to the in-house endpoint."""
from __future__ import annotations

from scripts import eval_finetune as ef


def _by_prompt() -> dict[str, list[dict]]:
    return {"a": [{"i": i} for i in range(30)], "b": [{"i": i} for i in range(5)]}


def test_per_prompt_zero_picks_every_record() -> None:
    assert len(ef.pick_records(_by_prompt(), 0)) == 35


def test_per_prompt_n_is_a_deterministic_subsample() -> None:
    first = ef.pick_records(_by_prompt(), 25)
    assert len(first) == 30                                 # 25 of a + all 5 of b
    assert first == ef.pick_records(_by_prompt(), 25)


def test_pick_does_not_mutate_its_input() -> None:
    data = _by_prompt()
    ef.pick_records(data, 3)
    assert data == _by_prompt()
