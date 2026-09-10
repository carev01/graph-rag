import json
import pytest
from answer_api.global_search import CommunityHit, map_report, _map_client_and_model
from graph_extract.config import ExtractSettings

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


class _FakeClient:
    def __init__(self, contents):
        self._c = list(contents)
        self.chat = self
        self.completions = self
    async def create(self, **kw):
        c = self._c.pop(0)
        return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": c})()})()]})


def _hit(cited=("f1", "f2"), relevance=0.0):
    return CommunityHit("c1", "T", "sum", 1, 7.0, list(cited), "[]", 0.9, relevance)


@pytest.mark.asyncio
async def test_map_keeps_only_real_fact_ids_and_carries_hit_relevance():
    """The map step no longer scores relevance itself (that rubric problem is
    gone) -- MapResult.relevance is just the rerank score that already lives
    on the hit, passed through unchanged."""
    payload = json.dumps({"key_points": ["kp1", "kp2"], "fact_ids": ["f1", "f9-halluc"]})
    m = await map_report(_FakeClient([payload]), "mm", "q", _hit(relevance=0.77))
    assert m is not None and m.fact_ids == ["f1"] and m.relevance == pytest.approx(0.77)
    assert m.key_points == ["kp1", "kp2"] and m.community_id == "c1"


@pytest.mark.asyncio
async def test_map_bad_json_retries_then_none():
    assert await map_report(_FakeClient(["nope", "still nope"]), "mm", "q", _hit()) is None


def test_map_tier_defaults_to_judge():
    s = ExtractSettings(_env_file=None, judge_base_url="http://glm", judge_model="glm-5.2:cloud",
                        judge_api_key="jk", **_MIN)
    _, model = _map_client_and_model(s)
    assert model == "glm-5.2:cloud"
