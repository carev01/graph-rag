import json
import pytest
from theme_builder.context import ContextResult
from theme_builder.report import generate_report, _report_client_and_model
from graph_extract.config import ExtractSettings

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


class _FakeClient:
    def __init__(self, contents):
        self._contents = list(contents)
        self.chat = self
        self.completions = self
    async def create(self, **kw):
        c = self._contents.pop(0)
        class _R:
            choices = [type("m", (), {"message": type("mm", (), {"content": c})()})()]
        return _R()


def _ctx():
    return ContextResult(text="FACTS:\n[u1] a\n[u2] b", fact_uuids={"u1", "u2"})


@pytest.mark.asyncio
async def test_valid_report_keeps_only_real_fact_ids():
    payload = json.dumps({"title": "T", "summary": "S",
        "full_report": [{"finding": "f1", "fact_ids": ["u1", "u9-hallucinated"]},
                        {"finding": "f2", "fact_ids": ["u2"]}],
        "rating": 7, "rating_explanation": "why", "tags": ["AWS"]})
    rep = await generate_report(_FakeClient([payload]), "m", _ctx())
    assert rep is not None
    assert set(rep.cited_fact_uuids) == {"u1", "u2"}      # u9 dropped
    assert rep.rating == 7.0 and rep.tags == ["AWS"]


@pytest.mark.asyncio
async def test_url_stripped_from_text():
    payload = json.dumps({"title": "T https://evil/x", "summary": "S http://e/y",
        "full_report": [{"finding": "see http://z", "fact_ids": ["u1"]}],
        "rating": 5, "rating_explanation": "", "tags": []})
    rep = await generate_report(_FakeClient([payload]), "m", _ctx())
    assert "http" not in rep.title and "http" not in rep.summary and "http" not in rep.full_report


@pytest.mark.asyncio
async def test_bad_json_retries_then_none():
    rep = await generate_report(_FakeClient(["not json", "still not json"]), "m", _ctx())
    assert rep is None                                    # one retry, then give up


@pytest.mark.asyncio
async def test_fenced_json_is_parsed():
    payload = "```json\n" + json.dumps({"title": "T", "summary": "S",
        "full_report": [{"finding": "f", "fact_ids": ["u1"]}], "rating": 3,
        "rating_explanation": "", "tags": []}) + "\n```"
    rep = await generate_report(_FakeClient([payload]), "m", _ctx())
    assert rep is not None and rep.cited_fact_uuids == ["u1"]


def test_report_tier_defaults_to_judge():
    s = ExtractSettings(_env_file=None, judge_base_url="http://glm", judge_model="glm-5.2:cloud",
                        judge_api_key="jk", **_MIN)
    client, model = _report_client_and_model(s)
    assert model == "glm-5.2:cloud"


@pytest.mark.asyncio
async def test_url_in_tag_is_stripped():
    payload = json.dumps({"title": "T", "summary": "S",
        "full_report": [{"finding": "f", "fact_ids": ["u1"]}], "rating": 5,
        "rating_explanation": "", "tags": ["see http://evil.com/leak", "AWS"]})
    rep = await generate_report(_FakeClient([payload]), "m", _ctx())
    assert all("http" not in t for t in rep.tags)


@pytest.mark.asyncio
async def test_unknown_finding_key_does_not_leak_url():
    payload = json.dumps({"title": "T", "summary": "S",
        "full_report": [{"finding": "ok", "fact_ids": ["u1"], "evidence_url": "http://leak/x"}],
        "rating": 5, "rating_explanation": "", "tags": []})
    rep = await generate_report(_FakeClient([payload]), "m", _ctx())
    assert "http" not in rep.full_report and rep.cited_fact_uuids == ["u1"]


@pytest.mark.asyncio
async def test_non_dict_findings_do_not_crash():
    payload = json.dumps({"title": "T", "summary": "S",
        "full_report": ["just a string finding"], "rating": 5,
        "rating_explanation": "", "tags": []})
    rep = await generate_report(_FakeClient([payload]), "m", _ctx())
    assert rep is not None and rep.cited_fact_uuids == []


@pytest.mark.asyncio
async def test_chatty_suffix_with_braces_parses():
    payload = ('{"title":"T","summary":"S","full_report":[{"finding":"f","fact_ids":["u1"]}],'
               '"rating":3,"rating_explanation":"","tags":[]}  note: see {1} above.')
    rep = await generate_report(_FakeClient([payload]), "m", _ctx())
    assert rep is not None and rep.cited_fact_uuids == ["u1"]
