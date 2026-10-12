"""The free retrieval gate's pass/fail logic."""
from answer_api.eval_retrieval import check_question


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
