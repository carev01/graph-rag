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
