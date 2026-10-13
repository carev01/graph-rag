"""Per-facet questions: one cheap call rewrites a comparison/enumeration question
into one standalone question per side, used for that side's retrieval."""
import logging
from types import SimpleNamespace

import pytest

import answer_api.router as router
from answer_api.scope import Scope
from answer_api.shape import Facet, QuestionShape, facet_queries
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
_V = Facet("Veeam", Scope(vendors=("Veeam",), source="detected"))
_A = Facet("AWS Backup", Scope(products=("AWS Backup",), source="detected"))
_Q = "Compare Veeam backup copy jobs with AWS Backup cross-Region copy"
_CITE = [{"marker": 1, "fact_uuid": "f", "sources": []}]


class _Client:
    def __init__(self, reply=None, exc=None):
        self.reply = reply
        self.exc = exc
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.calls.append(kw)
        if self.exc is not None:
            raise self.exc
        choices = [] if self.reply is None else [SimpleNamespace(
            message=SimpleNamespace(content=self.reply), finish_reason="stop")]
        return SimpleNamespace(choices=choices)


async def test_parses_the_json_and_sets_each_facets_query():
    c = _Client('```json\n{"Veeam": "How do Veeam backup copy jobs work?", '
                '"AWS Backup": "How does AWS Backup cross-Region copy work?"}\n```')
    out = await facet_queries(c, "map", _Q, (_V, _A))
    assert [f.query for f in out] == ["How do Veeam backup copy jobs work?",
                                      "How does AWS Backup cross-Region copy work?"]
    assert [(f.label, f.scope) for f in out] == [(_V.label, _V.scope), (_A.label, _A.scope)]
    assert len(c.calls) == 1
    call = c.calls[0]
    assert call["model"] == "map" and call["max_tokens"] >= 1500
    assert "reasoning" not in call and "extra_body" not in call
    prompt = call["messages"][0]["content"]
    assert _Q in prompt and "- Veeam" in prompt and "- AWS Backup" in prompt


async def test_a_missing_or_empty_label_falls_back_to_the_question(caplog):
    c = _Client('{"Veeam": "How do Veeam backup copy jobs work?", "Other": "x"}')
    with caplog.at_level(logging.WARNING):
        out = await facet_queries(c, "map", _Q, (_V, _A))
    assert [f.query for f in out] == ["How do Veeam backup copy jobs work?", _Q]
    assert "AWS Backup" in caplog.text
    c = _Client('{"Veeam": "  ", "AWS Backup": "How does AWS Backup copy?"}')
    out = await facet_queries(c, "map", _Q, (_V, _A))
    assert [f.query for f in out] == [_Q, "How does AWS Backup copy?"]


@pytest.mark.parametrize("client", [_Client(None), _Client(""), _Client("no json here"),
                                    _Client(exc=RuntimeError("down"))])
async def test_an_unusable_reply_falls_back_for_every_facet(client, caplog):
    with caplog.at_level(logging.WARNING):
        out = await facet_queries(client, "map", _Q, (_V, _A))
    assert [f.query for f in out] == [_Q, _Q]
    assert "facet question rewrite" in caplog.text


# --- router -------------------------------------------------------------------

@pytest.fixture
def run(monkeypatch):
    seen = SimpleNamespace(rewrites=[], modes=[])

    async def _rewrite(client, model, q, facets):
        seen.rewrites.append((client, model, q, tuple(facets)))
        return tuple(Facet(f.label, f.scope, query=f"about {f.label}") for f in facets)

    async def _classify(q, *, mode_override=None, **kw):
        if mode_override:
            return mode_override, "override"
        return "global", "heuristic"

    async def _global(*a, **kw):
        seen.modes.append(("global", kw.get("facets")))
        return {"answer": "A [1]", "citations": _CITE, "communities_used": [{"x": 1}],
                "applies_to": [], "degraded_facets": []}

    async def _local(*a, **kw):
        seen.modes.append(("local", kw.get("facets")))
        return {"answer": "L [1]", "citations": _CITE, "retrieved": 1}

    async def _enum(*a, **kw):
        seen.modes.append(("enumerate", kw["facets"]))
        return {"answer": "E", "citations": _CITE, "applies_to": [],
                "checked_vendors": [], "documented_vendors": [], "degraded_facets": []}

    async def _fresh(*a, **kw):
        return {}

    monkeypatch.setattr(router, "facet_queries", _rewrite)
    monkeypatch.setattr(router, "classify", _classify)
    monkeypatch.setattr(router.global_mod, "global_search", _global)
    monkeypatch.setattr(router.synth_mod, "answer_local", _local)
    monkeypatch.setattr(router.enum_mod, "enumerate_vendors", _enum)
    monkeypatch.setattr(router.freshness_mod, "freshness", _fresh)

    async def _go(shape, mode_override=None, scope=Scope()):
        env = await router.answer_router(None, None, None, None, "s", "MAPC", "mapm",
                                         "CHEAPC", "cheapm", q=_Q,
                                         mode_override=mode_override, scope=scope,
                                         settings=_S, shape=shape)
        return seen, env
    return _go


async def test_router_rewrites_once_for_a_comparison_with_the_map_tier(run):
    seen, env = await run(QuestionShape("comparison", (_V, _A)))
    assert seen.rewrites == [("MAPC", "mapm", _Q, (_V, _A))]
    (mode, facets), = seen.modes
    assert mode == "global" and [f.query for f in facets] == ["about Veeam",
                                                             "about AWS Backup"]
    assert [f["query"] for f in env["facets"]] == ["about Veeam", "about AWS Backup"]


async def test_router_rewrites_once_for_an_enumeration(run):
    facets = (Facet("AWS", Scope(vendors=("AWS",))), Facet("Veeam", Scope(vendors=("Veeam",))))
    seen, _ = await run(QuestionShape("enumeration", facets))
    assert len(seen.rewrites) == 1
    assert seen.modes[0][0] == "enumerate"
    assert [f.query for f in seen.modes[0][1]] == ["about AWS", "about Veeam"]


async def test_router_never_rewrites_a_single_question(run):
    seen, _ = await run(QuestionShape("single", (Facet("Veeam", _V.scope),)))
    assert seen.rewrites == []
    seen, _ = await run(None)
    assert seen.rewrites == []


async def test_router_never_rewrites_an_overridden_enumeration(run):
    facets = (Facet("AWS", Scope(vendors=("AWS",))), Facet("Veeam", Scope(vendors=("Veeam",))))
    seen, _ = await run(QuestionShape("enumeration", facets), mode_override="local")
    assert seen.rewrites == []
    assert seen.modes == [("local", None)]


async def test_a_single_envelope_carries_no_facet_query(run):
    _, env = await run(QuestionShape("single", (Facet("Veeam", _V.scope),)))
    assert env["facets"] == [{"label": "Veeam", "scope": _V.scope.as_dict()}]
