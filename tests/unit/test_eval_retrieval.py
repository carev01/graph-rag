"""The free retrieval gate's pass/fail logic."""
import pytest

import answer_api.eval_retrieval as er
from answer_api.eval_retrieval import check_question
from answer_api.scope import Scope
from answer_api.shape import Facet, QuestionShape
from graph_extract.config import ExtractSettings

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
_V = Facet("Veeam", Scope(vendors=("Veeam",), source="detected"))
_A = Facet("AWS", Scope(vendors=("AWS",), source="detected"))


def _ev(label, articles=(), vendors=(), communities=1, facts=1):
    return {"label": label, "articles": set(articles), "vendors": set(vendors),
            "communities": communities, "facts": facts}


def test_passes_when_every_facet_has_evidence_and_an_expected_article_is_found():
    q = {"question": "q", "expected_article_ids": ["a1"]}
    r = check_question(q, [_ev("Veeam", ["a1"], ["Veeam"]), _ev("AWS", ["b"], ["AWS"])])
    assert r["pass"] and r["expected_hit"] is True and r["facets_with_evidence"] == 2


def test_fails_when_a_facet_has_no_evidence():
    q = {"question": "q", "expected_article_ids": ["a1"]}
    r = check_question(q, [_ev("Veeam", ["a1"]), _ev("AWS", communities=0, facts=0)])
    assert not r["pass"]


def test_fails_when_no_expected_article_is_retrieved():
    q = {"question": "q", "expected_article_ids": ["zz"]}
    assert not check_question(q, [_ev("Veeam", ["a1"])])["pass"]


def test_enumeration_needs_every_expected_vendor():
    q = {"question": "q", "expected_article_ids": [], "expected_vendors": ["AWS", "Veeam"]}
    ok = check_question(q, [_ev("AWS", vendors=["AWS"]), _ev("Veeam", vendors=["Veeam"])])
    assert ok["pass"] and ok["expected_hit"] is None
    bad = check_question(q, [_ev("AWS", vendors=["AWS"]), _ev("Veeam", facts=0,
                                                               communities=0)])
    assert not bad["pass"]


def test_an_enumeration_passes_when_a_non_expected_vendor_has_no_evidence():
    """A vendor that genuinely does not document the feature has evidence 0;
    that must not fail the gate. Its count is reported as information."""
    q = {"question": "q", "_shape": "enumeration", "expected_article_ids": [],
         "expected_vendors": ["AWS"]}
    r = check_question(q, [_ev("AWS", vendors=["AWS"]),
                           _ev("Cohesity", communities=0, facts=0),
                           _ev("Veeam", communities=1, facts=2)])
    assert r["pass"]
    assert r["other_vendors"] == {"Cohesity": 0, "Veeam": 3}


def test_an_enumeration_fails_when_an_expected_vendor_has_no_evidence():
    q = {"question": "q", "_shape": "enumeration", "expected_article_ids": [],
         "expected_vendors": ["AWS", "Veeam"]}
    r = check_question(q, [_ev("AWS", vendors=["AWS"]), _ev("Veeam", communities=0, facts=0),
                           _ev("Cohesity")])
    assert not r["pass"]


def test_an_enumeration_still_checks_expected_articles():
    q = {"question": "q", "_shape": "enumeration", "expected_article_ids": ["zz"],
         "expected_vendors": ["AWS"]}
    assert not check_question(q, [_ev("AWS", ["a1"], ["AWS"])])["pass"]
    q["expected_article_ids"] = ["a1"]
    assert check_question(q, [_ev("AWS", ["a1"], ["AWS"]),
                              _ev("Veeam", communities=0, facts=0)])["pass"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["comparison", "enumeration"])
async def test_comparison_and_enumeration_facets_get_their_own_questions(monkeypatch, kind):
    rewrites = []

    async def _rewrite(client, model, q, facets):
        rewrites.append((client, model, q))
        return tuple(Facet(f.label, f.scope, query=f"about {f.label}") for f in facets)

    monkeypatch.setattr(er, "facet_queries", _rewrite)
    shape = await er._with_facet_queries("C", "m", "q", QuestionShape(kind, (_V, _A)))
    assert rewrites == [("C", "m", "q")]
    assert [f.query for f in shape.facets] == ["about Veeam", "about AWS"]


@pytest.mark.asyncio
async def test_a_single_question_is_not_rewritten(monkeypatch):
    async def _rewrite(*a):
        raise AssertionError("no rewrite for single")

    monkeypatch.setattr(er, "facet_queries", _rewrite)
    single = QuestionShape("single", (Facet("", Scope()),))
    assert await er._with_facet_queries("C", "m", "q", single) is single


@pytest.mark.asyncio
async def test_evidence_retrieves_with_each_facets_question(monkeypatch):
    asked = []

    async def _search(graphiti, driver, *, q, k, scope=None, **kw):
        asked.append(("search", q))
        return {"results": [], "count": 0}

    async def _shortlist(driver, embedder, q, **kw):
        asked.append(("shortlist", q))
        return []

    monkeypatch.setattr(er, "search_local", _search)
    monkeypatch.setattr(er, "shortlist_communities", _shortlist)
    shape = QuestionShape("comparison", (Facet("Veeam", _V.scope, query="veeam q"), _A))
    await er._evidence(None, None, None, _S, "orig", shape)
    assert asked == [("search", "veeam q"), ("shortlist", "veeam q"),
                     ("search", "orig"), ("shortlist", "orig")]
