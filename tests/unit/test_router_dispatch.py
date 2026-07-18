import pytest

import answer_api.synthesize as synth_mod
import answer_api.global_search as global_mod
import answer_api.drift as drift_mod
import answer_api.timeline as timeline_mod
from answer_api.router import answer_router
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


async def _fake_local(*a, **k):
    return {"query": k["q"], "answer": "local [1]",
            "citations": [{"marker": 1, "fact_uuid": "f1", "sources": [{"url": "u"}]}],
            "retrieved": 2, "cited": 1}


async def _fake_local_empty(*a, **k):
    return {"query": k["q"], "answer": "refusal", "citations": [], "retrieved": 0, "cited": 0}


async def _fake_global(*a, **k):
    return {"query": k["q"], "answer": "global [1]",
            "citations": [{"marker": 1, "fact_uuid": "g1", "sources": []}],
            "communities_used": [{"community_id": "c1", "title": "T"}]}


async def _fake_drift(*a, **k):
    return {"query": k["q"], "answer": "drift [1]",
            "citations": [{"marker": 1, "fact_uuid": "d1", "sources": []}],
            "communities_used": [{"community_id": "c1", "title": "T"}],
            "follow_ups": [{"query": "fu", "community_id": "c1", "iteration": 1}]}


async def _fake_timeline(*a, **k):
    return {"query": k["q"], "count": 1,
            "timeline": [{"fact": "F", "fact_uuid": "t1", "valid_at": "2023-01-01",
                          "invalid_at": None, "status": "current", "sources": []}]}


@pytest.fixture(autouse=True)
def _patch_modes(monkeypatch):
    monkeypatch.setattr(synth_mod, "answer_local", _fake_local)
    monkeypatch.setattr(global_mod, "global_search", _fake_global)
    monkeypatch.setattr(drift_mod, "drift_search", _fake_drift)
    monkeypatch.setattr(timeline_mod, "timeline_local", _fake_timeline)


async def _route(mode_override, **over):
    return await answer_router(None, None, None, None, "sm", None, "mm", None, "",
                               q=over.get("q", "q"), mode_override=mode_override,
                               vendor=None, settings=_S)


async def test_routes_global_via_override():
    env = await _route("global")
    assert env["mode"] == "global"
    assert env["routing"]["via"] == "override"
    assert env["communities_used"][0]["community_id"] == "c1"


async def test_routes_timeline_and_renders():
    env = await _route("timeline")
    assert env["mode"] == "timeline"
    assert "[1]" in env["answer"]
    assert env["citations"][0]["fact_uuid"] == "t1"
    assert env["timeline"][0]["fact"] == "F"


async def test_local_empty_escalates_to_drift(monkeypatch):
    monkeypatch.setattr(synth_mod, "answer_local", _fake_local_empty)
    env = await _route("local")
    assert env["mode"] == "drift"
    assert env["routing"]["fallback_from"] == "local"
    assert env["citations"][0]["fact_uuid"] == "d1"


async def test_local_with_results_stays_local():
    env = await _route("local")
    assert env["mode"] == "local"
    assert env["routing"]["fallback_from"] is None
    assert env["citations"][0]["fact_uuid"] == "f1"


async def test_drift_degrade_normalized(monkeypatch):
    async def _degrade(*a, **k):
        return {"query": k["q"], "answer": "local fallback", "citations": [],
                "retrieved": 1, "cited": 0, "degraded": "no-primer-communities"}
    monkeypatch.setattr(drift_mod, "drift_search", _degrade)
    env = await _route("drift")
    assert env["routing"]["degraded"] == "no-primer-communities"
    assert env["answer"] == "local fallback"
