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


@pytest.fixture(autouse=True)
def _no_rewrite(monkeypatch):
    """The per-facet question rewrite is covered in test_facet_queries; here it
    returns the facets unchanged so shape assertions compare like with like."""
    async def _identity(client, model, q, facets):
        return tuple(facets)

    monkeypatch.setattr(router, "facet_queries", _identity)


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
                "degraded_facets": [], "vendor_scores": {"AWS": 0.824},
                "relevance_unchecked": ["Cohesity"]}

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
    assert env["shape"] == "comparison" and env["shape_requested"] == "comparison"
    assert [f["label"] for f in env["facets"]] == ["Veeam", "AWS Backup"]


async def test_an_enumeration_runs_the_enumeration_path(seen):
    shape = QuestionShape("enumeration", (Facet("AWS", Scope(vendors=("AWS",))),))
    env = await _answer(shape)
    assert seen[0][0] == "enumerate"
    assert env["routing"]["path"] == "enumeration"
    assert env["mode"] == "global"
    assert env["shape_requested"] == "enumeration"
    assert env["checked_vendors"] == ["AWS"] and env["documented_vendors"] == ["AWS"]
    assert env["vendor_scores"] == {"AWS": 0.824}
    assert env["relevance_unchecked"] == ["Cohesity"]


async def test_no_shape_is_single_and_passes_no_facets(seen):
    env = await _answer(None)
    assert seen == [("global", None)]
    assert env["shape"] == "single" and env["shape_requested"] == "single"


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
    assert env["shape_requested"] == "comparison"
    # The scoped run could not check AWS Backup: relaxation must not hide that.
    assert env["degraded_facets"] == ["AWS Backup"]
    assert env["answer"] == "A [1]\n\nCould not check: AWS Backup."


async def test_a_temporal_enumeration_keeps_its_answer(seen, monkeypatch):
    """Temporal wording classifies as timeline, but an enumeration's raw result
    has no timeline: rendering it as one would replace the answer with
    'No recorded changes'."""
    async def _timeline_label(q, **kw):
        return "timeline", "heuristic"

    monkeypatch.setattr(router, "classify", _timeline_label)
    shape = QuestionShape("enumeration", (Facet("AWS", Scope(vendors=("AWS",))),))
    env = await _answer(shape)
    assert env["answer"] == "E" and env["citations"] == _CITE
    assert env["mode"] == "global" and env["routing"]["path"] == "enumeration"
    assert "timeline" not in env


async def _override(q, *, mode_override=None, **kw):
    return mode_override, "override"


async def test_an_override_bypasses_the_enumeration_path(seen, monkeypatch):
    """An explicit mode is the caller's instruction (the eval harness forces each
    mode in turn): an enumeration-shaped question runs that mode, unfaceted."""
    async def _local(*a, facets=None, **kw):
        seen.append(("local", facets))
        return {"answer": "L [1]", "citations": _CITE, "retrieved": 3}

    monkeypatch.setattr(router, "classify", _override)
    monkeypatch.setattr(router.synth_mod, "answer_local", _local)
    shape = QuestionShape("enumeration", (Facet("AWS", Scope(vendors=("AWS",))),))
    env = await router.answer_router(None, None, None, None, "s", None, "m", None, "c",
                                     q="q", mode_override="local", scope=Scope(),
                                     settings=_S, shape=shape)
    assert seen == [("local", None)]
    assert env["mode"] == "local" and "path" not in env["routing"]
    assert env["shape"] == "single" and env["facets"] == []
    assert env["shape_requested"] == "enumeration"
    assert "checked_vendors" not in env


async def test_an_override_keeps_a_comparisons_facets(seen, monkeypatch):
    async def _drift(*a, facets=None, **kw):
        seen.append(("drift", facets))
        return {"answer": "D [1]", "citations": _CITE, "applies_to": []}

    monkeypatch.setattr(router, "classify", _override)
    monkeypatch.setattr(router.drift_mod, "drift_search", _drift)
    shape = QuestionShape("comparison", (_V, _A))
    env = await router.answer_router(None, None, None, None, "s", None, "m", None, "c",
                                     q="q", mode_override="drift",
                                     scope=Scope(vendors=("Veeam",),
                                                 products=("AWS Backup",),
                                                 source="detected"),
                                     settings=_S, shape=shape)
    assert seen == [("drift", (_V, _A))]
    assert env["shape"] == "comparison"


async def test_relaxation_does_not_duplicate_a_note_the_answer_already_has(monkeypatch):
    async def _classify(q, **kw):
        return "local", "heuristic"

    async def _local(*a, scope, facets=None, **kw):
        if scope.vendors:
            return {"answer": "", "citations": [], "retrieved": 3,
                    "degraded_facets": ["AWS Backup"]}
        return {"answer": "L [1]\n\nCould not check: AWS Backup.", "citations": _CITE,
                "retrieved": 1}

    async def _fresh(*a, **kw):
        return {}

    monkeypatch.setattr(router, "classify", _classify)
    monkeypatch.setattr(router.synth_mod, "answer_local", _local)
    monkeypatch.setattr(router.freshness_mod, "freshness", _fresh)
    env = await _answer(QuestionShape("comparison", (_V, _A)),
                        Scope(vendors=("Veeam",), products=("AWS Backup",),
                              source="detected"))
    assert env["answer"].count("Could not check: AWS Backup.") == 1
    assert env["degraded_facets"] == ["AWS Backup"]


async def test_a_relaxation_with_no_degraded_facet_adds_none(monkeypatch):
    async def _classify(q, **kw):
        return "local", "heuristic"

    async def _local(*a, scope, facets=None, **kw):
        if scope.vendors:
            return {"answer": "", "citations": [], "retrieved": 3, "degraded_facets": []}
        return {"answer": "L [1]", "citations": _CITE, "retrieved": 1}

    async def _fresh(*a, **kw):
        return {}

    monkeypatch.setattr(router, "classify", _classify)
    monkeypatch.setattr(router.synth_mod, "answer_local", _local)
    monkeypatch.setattr(router.freshness_mod, "freshness", _fresh)
    env = await _answer(QuestionShape("comparison", (_V, _A)),
                        Scope(vendors=("Veeam",), products=("AWS Backup",),
                              source="detected"))
    assert env["answer"] == "L [1]"
    assert "degraded_facets" not in env


async def test_a_timeline_routed_comparison_reports_single_with_no_facets(monkeypatch):
    """Timeline ignores facets, so the envelope must not claim a balanced
    comparison; shape_requested still records what was asked."""
    async def _classify(q, **kw):
        return "timeline", "heuristic"

    async def _timeline(*a, **kw):
        return {"query": "q", "count": 1, "timeline": [
            {"fact": "F", "fact_uuid": "f", "valid_at": None, "invalid_at": None,
             "status": "current", "sources": []}]}

    async def _fresh(*a, **kw):
        return {}

    monkeypatch.setattr(router, "classify", _classify)
    monkeypatch.setattr(router.timeline_mod, "timeline_local", _timeline)
    monkeypatch.setattr(router.freshness_mod, "freshness", _fresh)
    env = await _answer(QuestionShape("comparison", (_V, _A)),
                        Scope(vendors=("Veeam",), products=("AWS Backup",),
                              source="detected"))
    assert env["mode"] == "timeline"
    assert env["shape"] == "single" and env["facets"] == []
    assert env["shape_requested"] == "comparison"


async def test_a_comparison_routed_to_timeline_skips_the_facet_rewrite(monkeypatch):
    """Timeline ignores facets, so the per-facet rewrite call (a cheap-tier LLM
    call) is pure waste when that is the mode that will run."""
    rewrite_calls = []

    async def _rewrite(client, model, q, facets):
        rewrite_calls.append(facets)
        return tuple(facets)

    async def _classify(q, **kw):
        return "timeline", "heuristic"

    async def _timeline(*a, **kw):
        return {"query": "q", "count": 0, "timeline": []}

    async def _fresh(*a, **kw):
        return {}

    monkeypatch.setattr(router, "facet_queries", _rewrite)
    monkeypatch.setattr(router, "classify", _classify)
    monkeypatch.setattr(router.timeline_mod, "timeline_local", _timeline)
    monkeypatch.setattr(router.freshness_mod, "freshness", _fresh)
    env = await _answer(QuestionShape("comparison", (_V, _A)),
                        Scope(vendors=("Veeam",), products=("AWS Backup",),
                              source="detected"))
    assert rewrite_calls == []
    assert env["mode"] == "timeline"


@pytest.fixture
def fallback(monkeypatch):
    """Global returns `state["global"]`; local records it ran."""
    state = {"global": None, "calls": []}

    async def _classify(q, **kw):
        return "global", "heuristic"

    async def _global(*a, **kw):
        state["calls"].append(("global", kw.get("facets")))
        return state["global"]

    async def _local(*a, **kw):
        state["calls"].append(("local", kw.get("facets")))
        return {"answer": "L [1]", "citations": _CITE, "retrieved": 1}

    async def _fresh(*a, **kw):
        return {}

    monkeypatch.setattr(router, "classify", _classify)
    monkeypatch.setattr(router.global_mod, "global_search", _global)
    monkeypatch.setattr(router.synth_mod, "answer_local", _local)
    monkeypatch.setattr(router.freshness_mod, "freshness", _fresh)
    return state


async def test_a_faceted_global_answer_grounded_by_local_supplements_is_kept(fallback):
    # Every citation came from the facets' local supplements: no community was
    # used, but the answer is grounded -- rerunning as local would discard it.
    fallback["global"] = {"answer": "G [1]", "citations": _CITE, "communities_used": [],
                          "applies_to": [], "degraded_facets": []}
    env = await _answer(QuestionShape("comparison", (_V, _A)))
    assert [m for m, _ in fallback["calls"]] == ["global"]
    assert env["mode"] == "global" and env["answer"].startswith("G")


async def test_a_faceted_global_answer_with_no_citation_reruns_as_local(fallback):
    fallback["global"] = {"answer": "refused", "citations": [],
                          "communities_used": [{"x": 1}], "applies_to": [],
                          "degraded_facets": []}
    env = await _answer(QuestionShape("comparison", (_V, _A)))
    assert fallback["calls"] == [("global", (_V, _A)), ("local", (_V, _A))]
    assert env["mode"] == "local"


async def test_a_single_global_answer_with_no_community_still_reruns_as_local(fallback):
    fallback["global"] = {"answer": "G [1]", "citations": _CITE, "communities_used": [],
                          "applies_to": []}
    env = await _answer(None)
    assert [m for m, _ in fallback["calls"]] == ["global", "local"]
    assert env["mode"] == "local"
