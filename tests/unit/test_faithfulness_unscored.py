"""The faithfulness judge must never silently report 0 when it fails to measure.

`_faithfulness_judge` calls a reasoning model that can burn its whole token budget
on reasoning and return no content at all (`finish_reason == 'length'`,
`content is None`). Before this fix that flowed into `_parse_judge_score(None)`
which returns 0 -- "I could not measure this" recorded as "this is completely
unfaithful", corrupting the headline mean. This module locks in the fix:
`_faithfulness_judge` returns `int | None`, retries once on an unusable reply,
and `aggregate` excludes `None` from every mean it computes.
"""
from __future__ import annotations

import types

import pytest

import answer_api.eval_router as er
from answer_api.router_eval import aggregate


class _FakeCompletions:
    def __init__(self, responses):
        # responses: list of (content, finish_reason) tuples, one per call.
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


class _FakeCompletionsNoChoices:
    """Every reply is an HTTP 200 with an empty `choices` list -- seen in
    production (theme_builder/report.py, commit a0603ec). `_attempt` must not
    do `resp.choices[0]` unguarded against this."""

    def __init__(self, n):
        self._n = n
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(choices=[])


class _FakeClientNoChoices:
    def __init__(self, n=2):
        self.chat = types.SimpleNamespace(completions=_FakeCompletionsNoChoices(n))


# ---------------------------------------------------------------------------
# _faithfulness_judge
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_choices_list_yields_none_not_indexerror():
    """A choices-less HTTP 200 must be treated like any other unusable reply
    (return None) rather than raising IndexError from `resp.choices[0]`."""
    client = _FakeClientNoChoices()
    result = await er._faithfulness_judge(client, "m", "q", "answer", ["fact"])
    assert result is None
    assert len(client.chat.completions.calls) == 2


@pytest.mark.asyncio
async def test_unusable_both_attempts_yields_none_not_zero():
    client = _FakeClient([(None, "length"), (None, "length")])
    result = await er._faithfulness_judge(client, "m", "q", "answer", ["fact"])
    assert result is None
    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 8000
    assert calls[1]["max_tokens"] == 16000


@pytest.mark.asyncio
async def test_genuine_zero_is_preserved():
    client = _FakeClient([("0", "stop")])
    result = await er._faithfulness_judge(client, "m", "q", "answer", ["fact"])
    assert result == 0
    assert len(client.chat.completions.calls) == 1


@pytest.mark.asyncio
async def test_genuine_four_is_returned():
    client = _FakeClient([("4", "stop")])
    result = await er._faithfulness_judge(client, "m", "q", "answer", ["fact"])
    assert result == 4


@pytest.mark.asyncio
async def test_retry_recovers_on_second_attempt():
    client = _FakeClient([(None, "length"), ("3", "stop")])
    result = await er._faithfulness_judge(client, "m", "q", "answer", ["fact"])
    assert result == 3
    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 8000
    assert calls[1]["max_tokens"] == 16000


@pytest.mark.asyncio
async def test_unmeasurable_logs_a_warning_naming_the_question(caplog):
    client = _FakeClient([(None, "length"), (None, "length")])
    with caplog.at_level("WARNING"):
        result = await er._faithfulness_judge(client, "m", "which vendor?", "answer", ["fact"])
    assert result is None
    assert any("which vendor?" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def test_aggregate_excludes_none_faithfulness_from_mean_and_counts_unscored():
    pq = [
        {"question": "a", "intent": "local", "chosen": "local",
         "routing_hit": True, "grounding_hit": True, "faithfulness": 4},
        {"question": "b", "intent": "local", "chosen": "local",
         "routing_hit": True, "grounding_hit": True, "faithfulness": None},
        {"question": "c", "intent": "global", "chosen": "global",
         "routing_hit": True, "grounding_hit": None, "faithfulness": 2},
    ]
    s = aggregate(pq)
    assert s["faithfulness_unscored"] == 1
    assert s["faithfulness_mean"] == pytest.approx(3.0)          # mean of [4, 2], not [4, 0, 2]
    assert s["faithfulness_by_mode"]["local"] == pytest.approx(4.0)  # only the scored 'local' row
    assert s["faithfulness_by_mode"]["global"] == pytest.approx(2.0)


def test_aggregate_mode_with_only_none_reports_none_not_zero():
    pq = [
        {"question": "a", "intent": "drift", "chosen": "drift",
         "routing_hit": True, "grounding_hit": None, "faithfulness": None},
        {"question": "b", "intent": "local", "chosen": "local",
         "routing_hit": True, "grounding_hit": None, "faithfulness": 5},
    ]
    s = aggregate(pq)
    assert s["faithfulness_by_mode"]["drift"] is None
    assert s["faithfulness_by_mode"]["local"] == pytest.approx(5.0)
    assert s["faithfulness_unscored"] == 1


def test_aggregate_all_unscored_reports_none_mean():
    pq = [
        {"question": "a", "intent": "drift", "chosen": "drift",
         "routing_hit": True, "grounding_hit": None, "faithfulness": None},
    ]
    s = aggregate(pq)
    assert s["faithfulness_mean"] is None
    assert s["faithfulness_unscored"] == 1


def test_drift_wins_is_none_when_a_comparative_faithfulness_is_unmeasurable():
    pq = [
        {"question": "q", "intent": "drift", "chosen": "drift",
         "routing_hit": True, "grounding_hit": None, "faithfulness": 3,
         "comparative": {
             "local": {"grounding_hit": None, "faithfulness": 3},
             "global": {"grounding_hit": None, "faithfulness": 2},
             "drift": {"grounding_hit": None, "faithfulness": None},
         }},
    ]
    s = aggregate(pq)
    assert s["drift_wins"] is None
