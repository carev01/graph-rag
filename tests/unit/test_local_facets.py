"""Local answers retrieve per facet and group the facts (spec §3.1)."""
from types import SimpleNamespace

import pytest

import answer_api.synthesize as synth
from answer_api.scope import Scope
from answer_api.shape import Facet

pytestmark = pytest.mark.asyncio

_V = Facet("Veeam", Scope(vendors=("Veeam",)))
_A = Facet("AWS Backup", Scope(products=("AWS Backup",)))


def _fact(u, vendor, product):
    return {"fact": f"fact {u}", "fact_uuid": u, "valid_at": None, "invalid_at": None,
            "sources": [{"vendor": vendor, "product": product, "article_id": u}]}


class _Synth:
    def __init__(self, reply="Veeam does X [1]. AWS does Y [2]."):
        self.prompts = []
        self.reply = reply
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.reply), finish_reason="stop")])


@pytest.fixture
def searches(monkeypatch):
    calls = []

    async def _search(graphiti, driver, *, q, k, scope=None, group_id,
                      fetch_cap=450, **kw):
        calls.append((scope, k))
        if scope == _V.scope:
            rows = [_fact("v1", "Veeam", "Veeam ONE"), _fact("shared", "Veeam", "Veeam ONE")]
        elif scope == _A.scope:
            rows = [_fact("a1", "AWS", "AWS Backup"), _fact("shared", "Veeam", "Veeam ONE")]
        else:
            rows = [_fact("u1", "Veeam", "Veeam ONE")]
        return {"query": q, "count": len(rows), "results": rows}

    monkeypatch.setattr(synth.search_mod, "search_local", _search)
    return calls


async def test_each_facet_searches_with_its_own_scope_and_share_of_k(searches):
    await synth.answer_local(None, None, _Synth(), "m", q="q", k=15, group_id="g",
                             facets=[_V, _A], min_per_facet=6)
    assert searches == [(_V.scope, 8), (_A.scope, 8)]


async def test_facts_are_grouped_under_facet_headings_and_deduplicated(searches):
    s = _Synth()
    res = await synth.answer_local(None, None, s, "m", q="q", k=15, group_id="g",
                                   facets=[_V, _A])
    p = s.prompts[0]
    assert p.index("=== Veeam ===") < p.index("[1]") < p.index("=== AWS Backup ===")
    assert p.count("fact shared") == 1
    assert "Address EVERY side" in p and "Veeam; AWS Backup" in p
    assert res["retrieved"] == 3 and res["degraded_facets"] == []


async def test_a_facet_with_no_facts_is_shown_empty_not_dropped(searches, monkeypatch):
    async def _search(graphiti, driver, *, q, k, scope=None, group_id, fetch_cap=450, **kw):
        rows = [_fact("v1", "Veeam", "Veeam ONE")] if scope == _V.scope else []
        return {"query": q, "count": len(rows), "results": rows}
    monkeypatch.setattr(synth.search_mod, "search_local", _search)
    s = _Synth("Veeam does X [1].")
    await synth.answer_local(None, None, s, "m", q="q", group_id="g", facets=[_V, _A])
    assert "=== AWS Backup ===\n(no facts retrieved)" in s.prompts[0]


async def test_a_failed_facet_is_reported_never_read_as_absence(monkeypatch):
    async def _search(graphiti, driver, *, q, k, scope=None, group_id, fetch_cap=450, **kw):
        if scope == _A.scope:
            raise RuntimeError("neo4j down")
        return {"query": q, "count": 1, "results": [_fact("v1", "Veeam", "Veeam ONE")]}
    monkeypatch.setattr(synth.search_mod, "search_local", _search)
    s = _Synth("Veeam does X [1].")
    res = await synth.answer_local(None, None, s, "m", q="q", group_id="g",
                                   facets=[_V, _A])
    assert res["degraded_facets"] == ["AWS Backup"]
    assert res["answer"].endswith("Could not check: AWS Backup.")
    assert "=== AWS Backup ===" not in s.prompts[0]


async def test_single_facet_path_is_unchanged(searches):
    s = _Synth("Answer [1].")
    res = await synth.answer_local(None, None, s, "m", q="q", group_id="g",
                                   facets=[_V])
    assert searches == [(None, 15)]
    assert "===" not in s.prompts[0]
    assert "degraded_facets" not in res
