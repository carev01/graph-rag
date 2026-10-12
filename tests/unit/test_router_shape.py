"""The router passes the question shape to every mode (spec §3)."""
import pytest

import answer_api.router as router
from answer_api.scope import Scope
from answer_api.shape import Facet, QuestionShape
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
_V = Facet("Veeam", Scope(vendors=("Veeam",), source="detected"))
_A = Facet("AWS Backup", Scope(products=("AWS Backup",), source="detected"))
_CITE = [{"marker": 1, "fact_uuid": "f", "sources": []}]


@pytest.fixture
def seen(monkeypatch):
    calls = []

    async def _classify(q, **kw):
        return "global", "heuristic"

    async def _global(*a, **kw):
        calls.append(("global", kw.get("facets")))
        return {"answer": "A [1]", "citations": _CITE, "communities_used": [{"x": 1}],
                "applies_to": [], "degraded_facets": []}

    async def _enum(*a, **kw):
        calls.append(("enumerate", kw["facets"]))
        return {"answer": "E", "citations": _CITE, "applies_to": [],
                "checked_vendors": ["AWS"], "documented_vendors": ["AWS"],
                "degraded_facets": []}

    async def _fresh(*a, **kw):
        return {}

    monkeypatch.setattr(router, "classify", _classify)
    monkeypatch.setattr(router.global_mod, "global_search", _global)
    monkeypatch.setattr(router.enum_mod, "enumerate_vendors", _enum)
    monkeypatch.setattr(router.freshness_mod, "freshness", _fresh)
    return calls


async def _answer(shape, scope=Scope()):
    return await router.answer_router(None, None, None, None, "s", None, "m", None, "c",
                                      q="q", mode_override=None, scope=scope,
                                      settings=_S, shape=shape)


async def test_a_comparison_passes_its_facets_to_the_mode(seen):
    shape = QuestionShape("comparison", (_V, _A))
    env = await _answer(shape, Scope(vendors=("Veeam",), products=("AWS Backup",),
                                     source="detected"))
    assert seen == [("global", (_V, _A))]
    assert env["shape"] == "comparison"
    assert [f["label"] for f in env["facets"]] == ["Veeam", "AWS Backup"]


async def test_an_enumeration_runs_the_enumeration_path(seen):
    shape = QuestionShape("enumeration", (Facet("AWS", Scope(vendors=("AWS",))),))
    env = await _answer(shape)
    assert seen[0][0] == "enumerate"
    assert env["routing"]["path"] == "enumeration"
    assert env["mode"] == "global"
    assert env["checked_vendors"] == ["AWS"] and env["documented_vendors"] == ["AWS"]


async def test_no_shape_is_single_and_passes_no_facets(seen):
    env = await _answer(None)
    assert seen == [("global", None)]
    assert env["shape"] == "single"


async def test_a_comparison_whose_detected_scope_grounds_nothing_reruns_single(
        monkeypatch):
    calls = []

    async def _classify(q, **kw):
        return "global", "heuristic"

    async def _global(*a, scope, facets=None, **kw):
        calls.append(("global", scope.vendors, facets))
        if scope.vendors:
            return {"answer": "", "citations": [], "communities_used": []}
        return {"answer": "A [1]", "citations": _CITE, "communities_used": [{"x": 1}]}

    async def _local(*a, scope, facets=None, **kw):
        calls.append(("local", scope.vendors, facets))
        return {"answer": "", "citations": [], "retrieved": 3,
                "degraded_facets": ["AWS Backup"]}

    async def _fresh(*a, **kw):
        return {}

    monkeypatch.setattr(router, "classify", _classify)
    monkeypatch.setattr(router.global_mod, "global_search", _global)
    monkeypatch.setattr(router.synth_mod, "answer_local", _local)
    monkeypatch.setattr(router.freshness_mod, "freshness", _fresh)
    env = await _answer(QuestionShape("comparison", (_V, _A)),
                        Scope(vendors=("Veeam",), products=("AWS Backup",),
                              source="detected"))
    assert calls == [("global", ("Veeam",), (_V, _A)), ("local", ("Veeam",), (_V, _A)),
                     ("global", (), None)]
    assert env["scope"]["source"] == "detected-relaxed"
    assert env["shape"] == "single" and env["facets"] == []
    assert "degraded_facets" not in env
