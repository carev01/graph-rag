import json
from pathlib import Path

_MODES = {"local", "global", "drift", "timeline"}
_PATH = Path(__file__).resolve().parents[2] / "src" / "answer_api" / "router_golden.json"


def test_router_golden_schema():
    data = json.loads(_PATH.read_text())
    assert isinstance(data, list) and len(data) >= 25
    intents = set()
    for e in data:
        assert set(e) >= {"question", "intent", "expected_modes", "expected_article_ids"}
        assert isinstance(e["question"], str) and e["question"].strip()
        assert e["intent"] in _MODES
        assert isinstance(e["expected_modes"], list) and e["expected_modes"]
        assert set(e["expected_modes"]) <= _MODES
        assert isinstance(e["expected_article_ids"], list)
        intents.add(e["intent"])
    assert intents == _MODES        # all four intents represented


def test_tier1_golden_schema():
    """The Tier 1 vendor set (spec 2026-09-25 vendor/product end to end, section 6):
    6 scoped, 4 two-vendor comparisons, 2 explicitly cross-vendor."""
    path = _PATH.with_name("router_golden_tier1.json")
    data = json.loads(path.read_text())
    kinds = {"scoped": 0, "comparison": 0, "cross_vendor": 0}
    for e in data:
        assert set(e) >= {"question", "intent", "expected_modes", "expected_article_ids", "kind"}
        assert e["intent"] in _MODES and set(e["expected_modes"]) <= _MODES
        assert e["intent"] in e["expected_modes"]
        assert e["expected_article_ids"], "every Tier 1 question is grounded"
        kinds[e["kind"]] += 1
    assert kinds == {"scoped": 6, "comparison": 4, "cross_vendor": 2}
