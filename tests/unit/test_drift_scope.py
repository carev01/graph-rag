"""Task 5: DRIFT threads Scope through the primer, every follow-up, and the
no-primer fallback; synthesis renders vendor/product-labelled fact lines and
carries ATTRIBUTION_RULES; the envelope gains `applies_to` (spec §4.3 item 6,
§3). Fakes mirror tests/unit/test_drift_parse.py and
tests/integration/test_drift.py."""
from __future__ import annotations

import json

import pytest

from answer_api import drift
from answer_api.attribution import ATTRIBUTION_RULES
from answer_api.drift import FollowUp
from answer_api.global_search import CommunityHit
from answer_api.scope import Scope
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")

_SCOPE = Scope(("Veeam",), (), "detected")


class _FakeLLM:
    def __init__(self, contents):
        self._c = list(contents)
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        c = self._c.pop(0)
        return type("R", (), {"choices": [type("m", (), {
            "message": type("mm", (), {"content": c})(), "finish_reason": "stop"})()]})


class _CapturingLLM:
    def __init__(self, contents):
        self._c = list(contents)
        self.chat = self
        self.completions = self
        self.prompts: list[str] = []

    async def create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        c = self._c.pop(0)
        return type("R", (), {"choices": [type("m", (), {
            "message": type("mm", (), {"content": c})(), "finish_reason": "stop"})()]})


class _FakeProvenance:
    def __init__(self, sources_by_uuid):
        self._by_uuid = sources_by_uuid

    async def resolve_citations(self, uuids):
        return {u: {"sources": self._by_uuid.get(u, [])} for u in uuids}


def _hit(cid="c1"):
    return CommunityHit(cid, f"Title {cid}", "sum", 1, 8.0, ["f1"], "[]", 0.9)


# ---------------------------------------------------------------------------
# _primer threads scope to shortlist_communities
# ---------------------------------------------------------------------------

async def test_primer_passes_scope_to_shortlist_communities(monkeypatch):
    captured = {}

    async def fake_shortlist(driver, embedder, q, *, level, k, group_id,
                             rating_boost=0.1, settings=None, stats=None, scope=None):
        captured["scope"] = scope
        return [_hit()]

    monkeypatch.setattr(drift, "shortlist_communities", fake_shortlist)
    payload = json.dumps({"preliminary_answer": "d",
                          "follow_ups": [{"query": "q2", "community_id": "c1", "relevance": 9}]})
    out = await drift._primer(object(), _FakeLLM([payload]), "m", object(), q="q",
                              level=1, k=5, max_followups=4, group_id="g", settings=_S,
                              scope=_SCOPE)
    assert out is not None
    assert captured["scope"] is _SCOPE


async def test_primer_defaults_scope_to_none(monkeypatch):
    captured = {}

    async def fake_shortlist(driver, embedder, q, *, level, k, group_id,
                             rating_boost=0.1, settings=None, stats=None, scope=None):
        captured["scope"] = scope
        return []

    monkeypatch.setattr(drift, "shortlist_communities", fake_shortlist)
    out = await drift._primer(object(), _FakeLLM([]), "m", object(), q="q", level=1,
                              k=5, max_followups=4, group_id="g", settings=_S)
    assert out is None
    assert captured["scope"] is None


# ---------------------------------------------------------------------------
# _run_followup threads scope to search_local
# ---------------------------------------------------------------------------

async def test_run_followup_passes_scope_to_search_local(monkeypatch):
    captured = {}

    async def fake_search_local(graphiti, driver, *, q, k, scope=None,
                                include_invalid=False, group_id, center_node_uuid=None):
        captured["scope"] = scope
        return {"results": []}

    monkeypatch.setattr(drift, "search_local", fake_search_local)
    rows = await drift._run_followup(object(), object(), FollowUp("q", None, 1), k=8,
                                     group_id="g", scope=_SCOPE)
    assert rows == []
    assert captured["scope"] is _SCOPE


async def test_run_followup_defaults_scope_to_none(monkeypatch):
    captured = {}

    async def fake_search_local(graphiti, driver, *, q, k, scope=None,
                                include_invalid=False, group_id, center_node_uuid=None):
        captured["scope"] = scope
        return {"results": []}

    monkeypatch.setattr(drift, "search_local", fake_search_local)
    await drift._run_followup(object(), object(), FollowUp("q", None, 1), k=8, group_id="g")
    assert captured["scope"] is None


# ---------------------------------------------------------------------------
# drift_search's no-primer fallback threads scope to answer_local
# ---------------------------------------------------------------------------

async def test_drift_search_no_primer_fallback_passes_scope_to_answer_local(monkeypatch):
    captured = {}

    async def fake_primer(*a, **k):
        return None

    async def fake_answer_local(graphiti, driver, synth_client, synth_model, *,
                                q, group_id, scope=None, k=15):
        captured["scope"] = scope
        return {"query": q, "answer": "ans", "citations": [], "retrieved": 0,
                "cited": 0, "applies_to": []}

    monkeypatch.setattr(drift, "_primer", fake_primer)
    monkeypatch.setattr(drift, "answer_local", fake_answer_local)
    res = await drift.drift_search(object(), object(), object(), object(), "m", q="q",
                                   level=1, iterations=1, primer_k=5, max_followups=4,
                                   followup_k=8, group_id="g", settings=_S, scope=_SCOPE)
    assert captured["scope"] is _SCOPE
    assert res["degraded"] == "no-primer-communities"
    assert res["applies_to"] == []


# ---------------------------------------------------------------------------
# _synthesize: labelled fact lines + ATTRIBUTION_RULES in the prompt
# ---------------------------------------------------------------------------

async def test_synth_prompt_carries_attribution_rules():
    assert ATTRIBUTION_RULES in drift._SYNTH_PROMPT


async def test_synthesize_renders_labelled_fact_lines(monkeypatch):
    monkeypatch.setattr(drift, "Provenance", lambda d: _FakeProvenance(
        {"f1": [{"vendor": "Veeam", "product": "VBR", "article_id": "a1",
                "url": "https://x/a1"}]}))
    facts = [{"fact": "Hardened repositories keep backups immutable", "fact_uuid": "f1",
             "sources": [{"vendor": "Veeam", "product": "VBR", "article_id": "a1",
                         "url": "https://x/a1"}]}]
    llm = _CapturingLLM(["Immutable [1]."])
    answer, citations = await drift._synthesize(llm, "m", object(), q="q",
                                                preliminary_answer="draft", facts=facts)
    assert "[1] (Veeam · VBR) Hardened repositories keep backups immutable" in llm.prompts[0]
    assert citations[0]["sources"][0]["vendor"] == "Veeam"


async def test_synthesize_facts_without_sources_render_unlabelled(monkeypatch):
    monkeypatch.setattr(drift, "Provenance", lambda d: _FakeProvenance({}))
    facts = [{"fact": "Some fact", "fact_uuid": "f1"}]   # no "sources" key at all
    llm = _CapturingLLM(["Some fact [1]."])
    await drift._synthesize(llm, "m", object(), q="q", preliminary_answer="", facts=facts)
    assert "[1] Some fact" in llm.prompts[0]


# ---------------------------------------------------------------------------
# drift_search envelope: applies_to on success and [] on refusal
# ---------------------------------------------------------------------------

async def test_drift_search_zero_facts_refusal_has_empty_applies_to(monkeypatch):
    async def fake_primer(*a, **k):
        return "d", [FollowUp("q", "c1", 1)], [_hit()]

    async def fake_run_followup(*a, **k):
        return []

    monkeypatch.setattr(drift, "_primer", fake_primer)
    monkeypatch.setattr(drift, "_run_followup", fake_run_followup)
    res = await drift.drift_search(object(), object(), object(), object(), "m", q="q",
                                   level=1, iterations=1, primer_k=5, max_followups=4,
                                   followup_k=8, group_id="g", settings=_S)
    assert res["answer"] == drift._REFUSAL
    assert res["applies_to"] == []


async def test_drift_search_success_returns_applies_to_from_citations(monkeypatch):
    async def fake_primer(*a, **k):
        return "d", [FollowUp("q", "c1", 1)], [_hit()]

    fact = {"fact": "AWS Backup supports S3", "fact_uuid": "f1",
           "sources": [{"vendor": "Veeam", "product": "VBR", "article_id": "a1",
                       "url": "https://x/a1"}]}

    async def fake_run_followup(*a, **k):
        return [fact]

    monkeypatch.setattr(drift, "_primer", fake_primer)
    monkeypatch.setattr(drift, "_run_followup", fake_run_followup)
    monkeypatch.setattr(drift, "Provenance", lambda d: _FakeProvenance(
        {"f1": fact["sources"]}))
    llm = _FakeLLM(["AWS Backup supports S3 [1]."])
    res = await drift.drift_search(object(), object(), object(), llm, "m", q="q",
                                   level=1, iterations=1, primer_k=5, max_followups=4,
                                   followup_k=8, group_id="g", settings=_S)
    assert res["applies_to"] == [{"vendor": "Veeam", "products": ["VBR"], "facts": 1}]
