"""Facet coverage and enumeration completeness are scored from citations."""
from answer_api.eval_router import enumeration_hit, facet_coverage_hit
from answer_api.scope import Scope
from answer_api.shape import Facet, QuestionShape


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


_COMPARISON = QuestionShape("comparison", (
    Facet("Veeam", Scope(vendors=("Veeam",), source="detected")),
    Facet("AWS Backup", Scope(products=("AWS Backup",), source="detected"))))


def test_a_relaxed_comparison_is_scored_from_the_shape_it_was_asked_as():
    """Relaxation answers single and unscoped; scored from env it would be None
    and silently dropped from the metric. Scored from the requested shape, an
    answer citing one side only is a miss."""
    env = {"shape": "single", "facets": [], "citations": [_cite("Veeam", "Veeam ONE")]}
    assert facet_coverage_hit(env) is None
    assert facet_coverage_hit(env, _COMPARISON) is False
    env["citations"].append(_cite("AWS", "AWS Backup"))
    assert facet_coverage_hit(env, _COMPARISON) is True


def test_a_requested_non_comparison_shape_is_not_scored():
    env = {"shape": "comparison", "facets": _FACETS, "citations": []}
    assert facet_coverage_hit(env, QuestionShape("single", (Facet("", Scope()),))) is None


def test_enumeration_complete_needs_the_exact_set_and_a_citation_each():
    env = {"shape": "enumeration", "documented_vendors": ["AWS", "Veeam"],
           "citations": [_cite("AWS", "AWS Backup"), _cite("Veeam", "Veeam ONE")]}
    assert enumeration_hit(env, ["Veeam", "AWS"]) is True
    assert enumeration_hit(env, ["Veeam", "AWS", "Cohesity"]) is False
    env["citations"] = [_cite("AWS", "AWS Backup")]
    assert enumeration_hit(env, ["Veeam", "AWS"]) is False


def test_enumeration_unscored_without_ground_truth():
    assert enumeration_hit({"shape": "enumeration", "documented_vendors": []}, None) is None
