import pytest

import answer_api.eval_router as er
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


def _env(mode, article_id):
    cites = [{"marker": 1, "fact_uuid": "f1",
              "sources": [{"article_id": article_id, "url": "u", "title": "t"}]}] if article_id else []
    return {"mode": mode, "answer": "ans", "citations": cites,
            "routing": {"chosen": mode, "via": "llm", "fallback_from": None}}


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    async def _fake_router(*a, q, mode_override, vendor, settings):
        if mode_override is not None:
            return _env(mode_override, "A1")          # comparative forced mode
        return _env("local" if q == "loc" else "drift", "A1")
    async def _fake_judge(client, model, q, answer, facts):
        return 4
    async def _fake_facts(driver, g, uuids):
        return ["fact"]
    monkeypatch.setattr(er.router_mod, "answer_router", _fake_router)
    monkeypatch.setattr(er, "_faithfulness_judge", _fake_judge)
    monkeypatch.setattr(er, "_cited_fact_texts", _fake_facts)


async def test_run_eval_aggregates_and_comparative():
    questions = [
        {"question": "loc", "intent": "local", "expected_modes": ["local"], "expected_article_ids": ["A1"]},
        {"question": "broad", "intent": "drift", "expected_modes": ["drift", "global"], "expected_article_ids": []},
    ]
    summary = await er.run_eval((None,) * 9, questions, _S)
    assert summary["n"] == 2
    assert summary["routing_accuracy"] == 1.0              # loc->local, broad->drift both expected
    assert summary["grounding_precision"] == 1.0           # only 'loc' scored (broad is [])
    assert summary["comparative"] is not None              # the broad question ran the comparative pass
    report = er.format_report(summary)
    assert "Routing accuracy" in report and "drift_wins" in report.lower()
