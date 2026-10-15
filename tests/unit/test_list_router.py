"""The router sends single list questions to list mode (spec 2026-10-15 §2, §5)."""
import pytest

import answer_api.router as router_mod
from answer_api.scope import Scope
from answer_api.shape import Facet, QuestionShape

Q = "Where can I restore a Windows agent backup?"
_ENV_RAW = {"query": Q, "answer": "- Azure [1]", "list_mode": True,
            "expansion_queries": ["q1"], "citations": [{"marker": 1, "fact_uuid": "az",
            "sources": [{"url": "u", "vendor": "Veeam", "product": "VBR"}]}],
            "retrieved": 1, "cited": 1, "applies_to": []}


@pytest.fixture
def wired(monkeypatch):
    seen = {"list": 0, "modes": []}

    async def _answer_list(*a, **kw):
        seen["list"] += 1
        return dict(seen.get("raw") or _ENV_RAW)

    async def _run_mode(mode, *a, **kw):
        seen["modes"].append(mode)
        return mode, None, {"answer": "normal", "citations": [{"marker": 1, "fact_uuid": "x",
                                                               "sources": []}]}

    async def _classify(q, **kw):
        return ("local", "override") if kw.get("mode_override") else ("drift", "llm")

    async def _fresh(*a, **kw):
        return {}

    monkeypatch.setattr(router_mod.list_mod, "answer_list", _answer_list)
    monkeypatch.setattr(router_mod, "_run_mode", _run_mode)
    monkeypatch.setattr(router_mod, "classify", _classify)
    monkeypatch.setattr(router_mod.freshness_mod, "freshness", _fresh)
    return seen


def _settings():
    from graph_extract.config import ExtractSettings
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


async def _route(q=Q, override=None, shape=None):
    return await router_mod.answer_router(
        None, None, None, None, "s", None, "m", None, "c", q=q, mode_override=override,
        scope=Scope(), settings=_settings(),
        shape=shape or QuestionShape("single", (Facet("", Scope()),)))


async def test_a_single_list_question_takes_list_mode(wired):
    env = await _route()
    assert wired["list"] == 1 and wired["modes"] == []
    assert env["routing"]["path"] == "list" and env["list_mode"] is True
    assert env["mode"] == "local" and env["expansion_queries"] == ["q1"]


async def test_an_explicit_mode_skips_list_mode(wired):
    await _route(override="global")
    assert wired["list"] == 0 and wired["modes"] == ["local"]


async def test_a_non_list_question_takes_the_normal_route(wired):
    await _route(q="How does SureBackup verify backups?")
    assert wired["list"] == 0 and wired["modes"] == ["drift"]


async def test_an_ungrounded_list_answer_falls_back_to_the_normal_route(wired):
    wired["raw"] = {**_ENV_RAW, "citations": [], "answer": "I don't have enough information"}
    env = await _route()
    assert wired["list"] == 1 and wired["modes"] == ["drift"]
    assert env["routing"].get("path") != "list"


async def test_list_degradation_reaches_the_envelope(wired):
    wired["raw"] = {**_ENV_RAW, "degraded": ["retrieval failed for: q1"]}
    env = await _route()
    assert env["routing"]["degraded"] == ["retrieval failed for: q1"]


async def test_comparisons_never_take_list_mode(wired, monkeypatch):
    async def _fq(client, model, q, facets):
        return facets
    monkeypatch.setattr(router_mod, "facet_queries", _fq)
    shape = QuestionShape("comparison", (Facet("A", Scope(vendors=("A",))),
                                         Facet("B", Scope(vendors=("B",)))))
    await _route(q="Which features differ between A and B?", shape=shape)
    assert wired["list"] == 0
