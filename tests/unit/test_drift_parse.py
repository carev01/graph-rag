import json
import pytest
from answer_api.drift import _parse_followups, _refine_followups


def test_parse_orders_by_relevance_and_budgets():
    obj = {"follow_ups": [
        {"query": "a", "community_id": "c1", "relevance": 3},
        {"query": "b", "community_id": "c2", "relevance": 9},
        {"query": "c", "community_id": None, "relevance": 7},
    ]}
    fus = _parse_followups(obj, {"c1", "c2"}, max_followups=2, iteration=1)
    assert [f.query for f in fus] == ["b", "c"]          # top-2 by relevance
    assert all(f.iteration == 1 for f in fus)


def test_parse_drops_invalid_community_and_blank_query():
    obj = {"follow_ups": [
        {"query": "a", "community_id": "unknown", "relevance": 5},
        {"query": "", "community_id": "c1", "relevance": 9},
        {"query": "keep", "community_id": "c1", "relevance": 1},
    ]}
    fus = _parse_followups(obj, {"c1"}, max_followups=5, iteration=2)
    assert [f.query for f in fus] == ["a", "keep"]       # blank-query dropped
    a = next(f for f in fus if f.query == "a")
    assert a.community_id is None                         # unknown id -> None
    assert a.iteration == 2


def test_parse_empty_or_garbage():
    assert _parse_followups({}, set(), 4, 1) == []
    assert _parse_followups({"follow_ups": ["notadict", 3]}, set(), 4, 1) == []


class _FakeLLM:
    def __init__(self, contents):
        self._c = list(contents)
        self.chat = self
        self.completions = self
    async def create(self, **kw):
        c = self._c.pop(0)
        return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": c})()})()]})


@pytest.mark.asyncio
async def test_refine_followups_parses_and_tags_iteration_2():
    payload = json.dumps({"follow_ups": [{"query": "deeper", "community_id": "c1", "relevance": 8}]})
    fus = await _refine_followups(_FakeLLM([payload]), "m", q="q",
                                  facts=[{"fact": "f", "fact_uuid": "u1"}],
                                  max_followups=4, hit_ids={"c1"})
    assert [f.query for f in fus] == ["deeper"]
    assert fus[0].iteration == 2


@pytest.mark.asyncio
async def test_refine_followups_bad_json_returns_empty():
    fus = await _refine_followups(_FakeLLM(["nope", "still nope"]), "m", q="q",
                                  facts=[{"fact": "f", "fact_uuid": "u1"}],
                                  max_followups=4, hit_ids=set())
    assert fus == []
