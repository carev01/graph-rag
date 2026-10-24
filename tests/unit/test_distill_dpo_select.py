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
