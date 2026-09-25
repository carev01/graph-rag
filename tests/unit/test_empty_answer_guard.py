"""An empty/None-content chat completion must never surface as a successful
blank answer with citations == [] hidden inside HTTP 200, and the faithfulness
judge must never score a blank answer 5 by vacuous truth.

Two hazards, both observed live: an HTTP 200 with an empty `choices` list
(already fixed once in theme_builder/report.py, never generalised here), and
`content=None` because the GLM-5.2 reasoning-model synthesis tier spent its
whole max_tokens budget on reasoning tokens with finish_reason='length'.
"""
from __future__ import annotations

from answer_api import drift as drift_mod
from answer_api import eval_router as er
from answer_api import global_search as global_search_mod
from answer_api import synthesize
from graph_extract.config import ExtractSettings

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


class _FakeChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = type("M", (), {"content": content})()
        self.finish_reason = finish_reason


class _FakeResp:
    def __init__(self, choices):
        self.choices = choices


def _usable(text: str) -> _FakeResp:
    return _FakeResp([_FakeChoice(text)])


def _no_choices() -> _FakeResp:
    return _FakeResp([])


def _none_content(finish_reason: str = "length") -> _FakeResp:
    return _FakeResp([_FakeChoice(None, finish_reason)])


class _FakeClient:
    """Queue of canned chat-completion responses; records each call's kwargs
    so a test can assert the retry widened max_tokens rather than repeating it."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        self.calls.append(kw)
        return self._responses.pop(0)


class _CountingClient:
    """Fails the test loudly if `create` is ever invoked."""

    def __init__(self):
        self.calls = 0
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        self.calls += 1
        raise AssertionError("the judge must not call the model for a blank answer")


# ---------------------------------------------------------------------------
# _usable_content
# ---------------------------------------------------------------------------

def test_usable_content_empty_choices_is_none():
    assert synthesize._usable_content(_no_choices()) is None


def test_usable_content_none_message_is_none():
    assert synthesize._usable_content(_FakeResp([_FakeChoice(None)])) is None


def test_usable_content_whitespace_only_is_none():
    assert synthesize._usable_content(_FakeResp([_FakeChoice("   \n\t  ")])) is None


def test_usable_content_normal_reply_is_returned():
    assert synthesize._usable_content(_usable("a real answer [1].")) == "a real answer [1]."


# ---------------------------------------------------------------------------
# _complete_or_none
# ---------------------------------------------------------------------------

async def test_complete_or_none_retries_once_with_larger_budget():
    client = _FakeClient([_none_content(), _usable("second reply")])
    result = await synthesize._complete_or_none(client, "model", "prompt", max_tokens=1000)
    assert result == "second reply"
    assert len(client.calls) == 2
    assert client.calls[0]["max_tokens"] == 1000
    assert client.calls[1]["max_tokens"] == 3000        # widened, not repeated


async def test_complete_or_none_returns_none_after_both_attempts_fail():
    client = _FakeClient([_no_choices(), _none_content()])
    result = await synthesize._complete_or_none(client, "model", "prompt", max_tokens=500)
    assert result is None
    assert len(client.calls) == 2                        # does not raise, does not loop forever


# ---------------------------------------------------------------------------
# local (answer_local)
# ---------------------------------------------------------------------------

async def _fake_search_local_with_hit(graphiti, driver, *, q, k=15, scope=None, group_id):
    return {"query": q, "count": 1, "results": [
        {"fact": "AWS Backup uses Vault Lock", "fact_uuid": "f1", "valid_at": None,
         "invalid_at": None, "sources": [{"article_id": "a1", "source_url": "https://x/a1"}]}]}


async def test_answer_local_refuses_on_unusable_synthesis(monkeypatch):
    monkeypatch.setattr(synthesize.search_mod, "search_local", _fake_search_local_with_hit)
    client = _FakeClient([_none_content(), _none_content()])
    result = await synthesize.answer_local(None, None, client, "model", q="q", group_id="g")
    assert result["answer"] == synthesize._REFUSAL
    assert result["citations"] == []


async def test_answer_local_empty_choices_does_not_raise(monkeypatch):
    monkeypatch.setattr(synthesize.search_mod, "search_local", _fake_search_local_with_hit)
    client = _FakeClient([_no_choices(), _no_choices()])
    result = await synthesize.answer_local(None, None, client, "model", q="q", group_id="g")
    assert result["answer"] == synthesize._REFUSAL
    assert result["citations"] == []


# ---------------------------------------------------------------------------
# global reduce
# ---------------------------------------------------------------------------

def _hit():
    return global_search_mod.CommunityHit("c1", "T", "sum", 1, 7.0, ["f1"], "[]", 0.9, 0.8)


async def _fake_shortlist(driver, embedder, q, *, level, k, group_id, rating_boost=0.1,
                          settings=None, stats=None, scope=None):
    return [_hit()]


async def _fake_map_report(client, model, q, hit):
    return global_search_mod.MapResult(community_id="c1", title="T", relevance=hit.relevance,
                                       key_points=["kp"], fact_ids=["f1"])


async def _fake_fact_texts(driver, group_id, fact_uuids):
    # The reduce step reads fact text before calling the model (BACKLOG 0b);
    # these tests pass no driver, so serve the text directly.
    return {u: f"fact {u}" for u in fact_uuids}


class _FakeProvenance:
    """global_search now resolves provenance once, unconditionally, right
    after the map step (for labels -- spec §3.1); these tests pass no driver,
    so fake it out with no sources (labels are not what they exercise)."""

    def __init__(self, driver):
        pass

    async def resolve_citations(self, fact_uuids):
        return {u: {"valid_at": None, "invalid_at": None, "sources": []} for u in fact_uuids}


async def test_global_search_refuses_on_unusable_synthesis(monkeypatch):
    """The communities WERE shortlisted and mapped before the reduce LLM
    failed -- `communities_used` must report that (the same list the success
    path builds from `results`), not collapse "coverage existed, the LLM
    failed" into the same `[]` the "no thematic coverage existed" path uses."""
    monkeypatch.setattr(global_search_mod, "shortlist_communities", _fake_shortlist)
    monkeypatch.setattr(global_search_mod, "map_report", _fake_map_report)
    monkeypatch.setattr(global_search_mod, "_fact_texts", _fake_fact_texts)
    monkeypatch.setattr(global_search_mod, "Provenance", _FakeProvenance)
    synth_client = _FakeClient([_none_content(), _none_content()])
    result = await global_search_mod.global_search(
        None, None, object(), "map-model", synth_client, "synth-model",
        q="q", level=1, k=3, group_id="g", settings=_S)
    assert result["answer"] == global_search_mod._REFUSAL
    assert result["citations"] == []
    assert result["communities_used"] == [
        {"community_id": "c1", "title": "T", "relevance": 0.8}]


async def test_global_search_empty_choices_does_not_raise(monkeypatch):
    monkeypatch.setattr(global_search_mod, "shortlist_communities", _fake_shortlist)
    monkeypatch.setattr(global_search_mod, "map_report", _fake_map_report)
    monkeypatch.setattr(global_search_mod, "_fact_texts", _fake_fact_texts)
    monkeypatch.setattr(global_search_mod, "Provenance", _FakeProvenance)
    synth_client = _FakeClient([_no_choices(), _no_choices()])
    result = await global_search_mod.global_search(
        None, None, object(), "map-model", synth_client, "synth-model",
        q="q", level=1, k=3, group_id="g", settings=_S)
    assert result["answer"] == global_search_mod._REFUSAL
    assert result["citations"] == []


# ---------------------------------------------------------------------------
# DRIFT synthesis
# ---------------------------------------------------------------------------

async def test_drift_synthesize_refuses_on_unusable_synthesis():
    facts = [{"fact": "AWS Backup uses Vault Lock", "fact_uuid": "f1"}]
    client = _FakeClient([_none_content(), _none_content()])
    answer, citations = await drift_mod._synthesize(
        client, "model", None, q="q", preliminary_answer="prelim", facts=facts)
    assert answer == drift_mod._REFUSAL
    assert citations == []


async def test_drift_synthesize_empty_choices_does_not_raise():
    facts = [{"fact": "AWS Backup uses Vault Lock", "fact_uuid": "f1"}]
    client = _FakeClient([_no_choices(), _no_choices()])
    answer, citations = await drift_mod._synthesize(
        client, "model", None, q="q", preliminary_answer="prelim", facts=facts)
    assert answer == drift_mod._REFUSAL
    assert citations == []


async def test_drift_search_end_to_end_refuses_on_unusable_synthesis(monkeypatch):
    """Confirms the fallback lands in the full drift_search envelope with the
    other fields (follow_ups, communities_used) intact, mirroring the shape of
    its existing no-deduped-facts refusal path."""
    hit = _hit()

    async def _fake_primer(embedder, synth_client, synth_model, driver, *, q, level, k,
                           max_followups, group_id, settings, stats=None):
        return "prelim", [drift_mod.FollowUp(query="q2", community_id="c1", iteration=1)], [hit]

    async def _fake_run_followup(graphiti, driver, fu, *, k, group_id):
        return [{"fact": "AWS Backup uses Vault Lock", "fact_uuid": "f1"}]

    monkeypatch.setattr(drift_mod, "_primer", _fake_primer)
    monkeypatch.setattr(drift_mod, "_run_followup", _fake_run_followup)
    synth_client = _FakeClient([_none_content(), _none_content()])
    result = await drift_mod.drift_search(
        None, None, None, synth_client, "model", q="q", level=1, iterations=1,
        primer_k=3, max_followups=3, followup_k=5, group_id="g", settings=_S)
    assert result["answer"] == drift_mod._REFUSAL
    assert result["citations"] == []
    assert result["follow_ups"] == [{"query": "q2", "community_id": "c1", "iteration": 1}]
    assert result["communities_used"] == [{"community_id": "c1", "title": "T"}]


# ---------------------------------------------------------------------------
# faithfulness judge must not reward a blank answer
# ---------------------------------------------------------------------------

async def test_faithfulness_judge_skips_blank_answer_without_calling_client():
    client = _CountingClient()
    assert await er._faithfulness_judge(client, "model", "q", "", ["fact"]) is None
    assert await er._faithfulness_judge(client, "model", "q", "   ", ["fact"]) is None
    assert await er._faithfulness_judge(client, "model", "q", None, ["fact"]) is None
    assert client.calls == 0


async def test_faithfulness_judge_skips_refusal_without_calling_client():
    """A refusal string is non-blank, so it used to reach the judge with
    'CITED FACTS: (none)' and score 5 by vacuous truth ('no claims, so every
    claim is supported') -- exactly the inflation this eval exists to catch.
    Each mode's refusal string is module-local and worded differently; all
    three must be recognised."""
    client = _CountingClient()
    for refusal in (synthesize._REFUSAL, global_search_mod._REFUSAL, drift_mod._REFUSAL):
        assert await er._faithfulness_judge(client, "model", "q", refusal, ["fact"]) is None
    assert client.calls == 0
