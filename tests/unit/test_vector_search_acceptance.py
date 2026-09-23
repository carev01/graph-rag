import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "vsa", Path(__file__).parents[2] / "scripts" / "vector_search_acceptance.py")
vsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vsa)


def test_overlap_at_counts_shared_uuids_in_the_top_k():
    assert vsa.overlap_at(["a", "b", "c"], ["c", "b", "x"], 3) == 2 / 3
    assert vsa.overlap_at(["a", "b"], ["a", "b"], 2) == 1.0


def test_overlap_of_two_empty_lists_is_perfect_not_zero():
    """Both paths agreeing on 'nothing above the threshold' is agreement."""
    assert vsa.overlap_at([], [], 10) == 1.0


def test_overlap_when_only_one_side_is_empty_is_zero():
    assert vsa.overlap_at(["a"], [], 10) == 0.0
