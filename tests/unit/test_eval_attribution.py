"""Attribution check (spec §6): does the answer attribute a claim to a
vendor/product that is not among the (Vendor · Product) labels of the facts it
cites? `judge_attribution` mirrors `_faithfulness_judge`'s bounded max_tokens
retry pattern on the SAME independent eval-judge client, and must never coerce
an unusable reply to 0 (memory: "LLM empty reply coerced to a value")."""
from __future__ import annotations

import types

import pytest

import answer_api.eval_router as er

pytestmark = pytest.mark.asyncio


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content, finish_reason = self._responses.pop(0)
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
        return types.SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self, responses):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(responses))


# ---------------------------------------------------------------------------
# judge_attribution
# ---------------------------------------------------------------------------

async def test_genuine_zero_is_preserved():
    client = _FakeClient([("0", "stop")])
    result = await er.judge_attribution(client, "m", "q", "answer", "[1] (V · P) fact")
    assert result == 0
    assert len(client.chat.completions.calls) == 1


async def test_genuine_count_is_returned():
    client = _FakeClient([("2", "stop")])
    result = await er.judge_attribution(client, "m", "q", "answer", "[1] (V · P) fact")
    assert result == 2


async def test_unusable_both_attempts_yields_none_not_zero():
    client = _FakeClient([(None, "length"), (None, "length")])
    result = await er.judge_attribution(client, "m", "q", "answer", "[1] (V · P) fact")
    assert result is None
    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 8000
    assert calls[1]["max_tokens"] == 16000


async def test_retry_recovers_on_second_attempt():
    client = _FakeClient([(None, "length"), ("1", "stop")])
    result = await er.judge_attribution(client, "m", "q", "answer", "[1] (V · P) fact")
    assert result == 1
    assert len(client.chat.completions.calls) == 2


async def test_prompt_carries_the_labelled_facts_verbatim():
    client = _FakeClient([("0", "stop")])
    await er.judge_attribution(client, "m", "which vendor?", "the answer text",
                                "[1] (Veeam · B&R) does X")
    prompt = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "which vendor?" in prompt
    assert "the answer text" in prompt
    assert "[1] (Veeam · B&R) does X" in prompt
    assert "Reply with ONLY the integer" in prompt


# ---------------------------------------------------------------------------
# _score_one wiring: labelled facts built with attribution.fact_line, in
# citation order, from `_cited_fact_texts`.
# ---------------------------------------------------------------------------

async def test_score_one_builds_labelled_facts_from_citation_sources(monkeypatch):
    from answer_api.scope import ScopeResolver
    from graph_extract.config import ExtractSettings

    settings = ExtractSettings(_env_file=None, docext_base_url="http://x",
                                docext_read_key="k", neo4j_uri="bolt://x",
                                neo4j_user="u", neo4j_password="p")
    resolver = ScopeResolver([], [], {})

    citations = [
        {"marker": 1, "fact_uuid": "f1",
         "sources": [{"vendor": "Veeam", "product": "B&R", "url": "u1", "title": "t1"}]},
        {"marker": 2, "fact_uuid": "f2",
         "sources": [{"vendor": "Rubrik", "product": "RSC", "url": "u2", "title": "t2"}]},
    ]
    env = {"mode": "local", "answer": "ans [1] [2].", "citations": citations,
           "routing": {"chosen": "local", "via": "llm", "fallback_from": None}}

    async def _fake_router(*a, q, mode_override, scope, settings):
        return env

    async def _fake_facts(driver, g, uuids):
        return ["Veeam fact", "Rubrik fact"]

    captured: dict = {}

    async def _fake_judge(client, model, q, answer, facts):
        return 3

    async def _fake_attribution(client, model, q, answer, labelled_facts):
        captured["labelled_facts"] = labelled_facts
        return 1

    monkeypatch.setattr(er.router_mod, "answer_router", _fake_router)
    monkeypatch.setattr(er, "_cited_fact_texts", _fake_facts)
    monkeypatch.setattr(er, "_faithfulness_judge", _fake_judge)
    monkeypatch.setattr(er, "judge_attribution", _fake_attribution)

    q = {"question": "who does X?", "expected_article_ids": []}
    result = await er._score_one((None,) * 11, q, None, settings, resolver)
    assert len(result) == 5
    _env, _ghit, _faith, _facts, misattributed = result
    assert misattributed == 1
    assert captured["labelled_facts"] == (
        "[1] (Veeam · B&R) Veeam fact\n[2] (Rubrik · RSC) Rubrik fact")
