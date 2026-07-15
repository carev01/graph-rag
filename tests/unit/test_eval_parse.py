import pytest

from graph_extract.eval import _parse_type


@pytest.mark.parametrize("text,expected", [
    ("<think>maybe Platform or Tool</think>\nTool", "Tool"),
    ("The answer is: **Workload**.", "Workload"),
    ("Concept", "Concept"),
    ("i cannot decide", None),
    ("", None),
])
def test_parse_type(text, expected):
    assert _parse_type(text) == expected
