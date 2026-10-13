"""DRIFT primes per facet and keeps each follow-up in its facet (spec §3.3)."""
import asyncio
from types import SimpleNamespace

import pytest

import answer_api.drift as drift
import answer_api.global_search as gs
from answer_api.scope import Scope
from answer_api.shape import Facet
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_V = Facet("Veeam", Scope(vendors=("Veeam",)))
_A = Facet("AWS Backup", Scope(products=("AWS Backup",)))
_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


def _hit(cid):
    return gs.CommunityHit(community_id=cid, title=cid, summary="s", level=1, rating=5,
                           cited_fact_uuids=[], full_report="[]", similarity=0.9)


class _Synth:
    """Primer reply: only a Veeam-community follow-up. Final synthesis: fixed text
    (or an unusable empty reply when `final` is None)."""

    def __init__(self, final="Veeam X [1]. AWS Y [2]."):
        self.prompts = []
        self.final = final
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        p = kw["messages"][0]["content"]
        self.prompts.append(p)
        if "follow-up questions" in p:
            text = ('{"preliminary_answer": "d", "follow_ups": [{"query": "veeam detail",'
                    ' "community_id": "cv", "relevance": 9}]}')
        else:
            text = self.final
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason="stop")])


class _Prov:
    def __init__(self, driver):
        pass

    async def resolve_citations(self, uuids):
        return {}


def _fact(query, scope):
    vendor, product = (("Veeam", "Veeam ONE") if scope == _V.scope
                       else ("AWS", "AWS Backup"))
    return {"fact": query, "fact_uuid": query, "valid_at": None, "invalid_at": None,
            "sources": [{"vendor": vendor, "product": product, "article_id": "x"}]}


@pytest.fixture
def stubs(monkeypatch):
    ran = []

    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        return [_hit("cv")] if scope == _V.scope else [_hit("ca")]

    async def _followup(graphiti, driver, fu, *, k, group_id, scope=None,
                        fetch_cap=450, scan_max_episodes=6000, settings=None):
        ran.append((fu.query, scope))
        return [_fact(fu.query, scope)]

    monkeypatch.setattr(drift, "shortlist_communities", _shortlist)
    monkeypatch.setattr(drift, "_run_followup", _followup)
    monkeypatch.setattr(drift, "Provenance", _Prov)
    return ran


async def _run(synth, facets=(_V, _A)):
    return await drift.drift_search(None, None, None, synth, "m", q="compare", level=1,
                                    iterations=1, primer_k=5, max_followups=4,
                                    followup_k=8, group_id="g", settings=_S,
                                    facets=list(facets))


async def test_every_facet_gets_a_follow_up_in_its_own_scope(stubs):
    await _run(_Synth())
    assert ("veeam detail", _V.scope) in stubs
    assert ("compare", _A.scope) in stubs        # AWS had none -> scoped original question


async def test_final_synthesis_groups_facts_by_facet(stubs):
    s = _Synth()
    res = await _run(s)
    final = s.prompts[-1]
    assert final.index("=== Veeam ===") < final.index("=== AWS Backup ===")
    assert "Address EVERY side" in final
    assert res["degraded_facets"] == []


async def test_facet_that_retrieved_nothing_renders_no_facts(monkeypatch, stubs):
    async def _followup(graphiti, driver, fu, *, k, group_id, scope=None,
                        fetch_cap=450, scan_max_episodes=6000, settings=None):
        return [_fact(fu.query, scope)] if scope == _V.scope else []

    monkeypatch.setattr(drift, "_run_followup", _followup)
    s = _Synth()
    res = await _run(s)
    final = s.prompts[-1]
    assert "=== AWS Backup ===\n(no facts retrieved)" in final
    assert res["degraded_facets"] == []
    assert "Could not check" not in res["answer"]


async def test_primer_failure_degrades_facet(monkeypatch, stubs):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        if scope == _A.scope:
            raise RuntimeError("boom")
        return [_hit("cv")]

    monkeypatch.setattr(drift, "shortlist_communities", _shortlist)
    s = _Synth()
    res = await _run(s)
    assert res["degraded_facets"] == ["AWS Backup"]
    assert all(scope != _A.scope for _, scope in stubs)
    assert "=== AWS Backup ===" not in s.prompts[-1]
    assert res["answer"].endswith("Could not check: AWS Backup.")


async def test_followups_all_failing_for_a_facet_degrade_it(monkeypatch, stubs):
    async def _followup(graphiti, driver, fu, *, k, group_id, scope=None,
                        fetch_cap=450, scan_max_episodes=6000, settings=None):
        if scope == _A.scope:
            raise RuntimeError("search down")
        return [_fact(fu.query, scope)]

    monkeypatch.setattr(drift, "_run_followup", _followup)
    s = _Synth()
    res = await _run(s)
    assert res["degraded_facets"] == ["AWS Backup"]
    final = s.prompts[-1]
    assert "=== AWS Backup ===" not in final
    assert "(no facts retrieved)" not in final
    assert res["answer"].endswith("Could not check: AWS Backup.")


async def test_unusable_synthesis_with_degraded_facet_keeps_note(monkeypatch, stubs):
    async def _followup(graphiti, driver, fu, *, k, group_id, scope=None,
                        fetch_cap=450, scan_max_episodes=6000, settings=None):
        if scope == _A.scope:
            raise RuntimeError("search down")
        return [_fact(fu.query, scope)]

    monkeypatch.setattr(drift, "_run_followup", _followup)
    res = await _run(_Synth(final=None))
    assert res["answer"].startswith(drift._REFUSAL)
    assert res["answer"].endswith("Could not check: AWS Backup.")
    assert res["degraded_facets"] == ["AWS Backup"]


async def test_no_facts_refusal_with_degraded_facet_keeps_note(monkeypatch, stubs):
    async def _followup(graphiti, driver, fu, *, k, group_id, scope=None,
                        fetch_cap=450, scan_max_episodes=6000, settings=None):
        if scope == _A.scope:
            raise RuntimeError("search down")
        return []

    monkeypatch.setattr(drift, "_run_followup", _followup)
    res = await _run(_Synth())
    assert res["answer"] == drift._REFUSAL + " Could not check: AWS Backup."
    assert res["degraded_facets"] == ["AWS Backup"]


async def test_single_facet_path_unchanged(stubs):
    s = _Synth(final="Veeam X [1].")
    res = await _run(s, facets=[_V])
    assert "degraded_facets" not in res
    assert "=== " not in s.prompts[-1]
    assert stubs == [("veeam detail", None)]


async def test_a_cancelled_primer_shortlist_propagates_not_degrades(monkeypatch, stubs):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        if scope == _A.scope:
            raise asyncio.CancelledError()
        return [_hit("cv")]

    monkeypatch.setattr(drift, "shortlist_communities", _shortlist)
    with pytest.raises(asyncio.CancelledError):
        await _run(_Synth())


async def test_primer_and_coverage_follow_ups_use_each_facets_question(monkeypatch, stubs):
    shortlisted = []

    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        shortlisted.append((q, scope))
        return [_hit("cv")] if scope == _V.scope else [_hit("ca")]

    monkeypatch.setattr(drift, "shortlist_communities", _shortlist)
    s = _Synth()
    await _run(s, facets=[Facet("Veeam", _V.scope, query="veeam q"),
                          Facet("AWS Backup", _A.scope, query="aws q")])
    assert shortlisted == [("veeam q", _V.scope), ("aws q", _A.scope)]
    # AWS had no follow-up from the primer: its coverage follow-up asks its own question.
    assert ("aws q", _A.scope) in stubs
    assert ("compare", _A.scope) not in stubs
    # The primer and the final synthesis still answer the original question.
    assert "QUESTION: compare" in s.prompts[0]
    assert "QUESTION: compare" in s.prompts[-1]
    assert "QUESTION: aws q" not in s.prompts[-1]
