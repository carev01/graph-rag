"""Facet coverage and enumeration completeness are scored from citations."""
from answer_api.eval_router import enumeration_hit, facet_coverage_hit


def _cite(vendor, product):
    return {"sources": [{"vendor": vendor, "product": product}]}


_FACETS = [{"label": "Veeam", "scope": {"vendors": ["Veeam"], "products": [],
                                        "sources": [], "source": "detected"}},
           {"label": "AWS Backup", "scope": {"vendors": [], "products": ["AWS Backup"],
                                             "sources": [], "source": "detected"}}]


def test_comparison_covered_only_when_every_facet_is_cited():
    env = {"shape": "comparison", "facets": _FACETS,
           "citations": [_cite("Veeam", "Veeam ONE"), _cite("AWS", "AWS Backup")]}
    assert facet_coverage_hit(env) is True
    env["citations"] = [_cite("Veeam", "Veeam ONE")]
    assert facet_coverage_hit(env) is False


def test_facet_coverage_is_not_scored_for_other_shapes():
    assert facet_coverage_hit({"shape": "single", "facets": [], "citations": []}) is None


def test_enumeration_complete_needs_the_exact_set_and_a_citation_each():
    env = {"shape": "enumeration", "documented_vendors": ["AWS", "Veeam"],
           "citations": [_cite("AWS", "AWS Backup"), _cite("Veeam", "Veeam ONE")]}
    assert enumeration_hit(env, ["Veeam", "AWS"]) is True
    assert enumeration_hit(env, ["Veeam", "AWS", "Cohesity"]) is False
    env["citations"] = [_cite("AWS", "AWS Backup")]
    assert enumeration_hit(env, ["Veeam", "AWS"]) is False


def test_enumeration_unscored_without_ground_truth():
    assert enumeration_hit({"shape": "enumeration", "documented_vendors": []}, None) is None
