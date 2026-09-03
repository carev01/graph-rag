"""The repetition-penalty wrapper (graph_extract.graphiti_client._inject_penalties).

Context: graphiti sends only model/messages/temperature/max_tokens/response_format
and we pin temperature=0.0, so a weaker model that starts an ascending-integer run
in an unbounded `list[int]` field cannot escape it -- the strict JSON grammar leaves
only digits, ',' and ']' as legal continuations. A frequency penalty is the
decoding-level fix; these tests pin its wiring.
"""
from __future__ import annotations

import pytest

from graph_extract.config import ExtractSettings
from graph_extract.graphiti_client import _inject_penalties, _llm_client


class _FakeCompletions:
    def __init__(self) -> None:
        self.seen: dict = {}

    async def create(self, **kwargs):
        self.seen = kwargs
        return "resp"


class _FakeClient:
    def __init__(self) -> None:
        self.chat = type("C", (), {})()
        self.chat.completions = _FakeCompletions()


def _settings(**kw) -> ExtractSettings:
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    base.update(kw)
    return ExtractSettings(**base)


@pytest.mark.asyncio
async def test_injects_both_penalties():
    client = _FakeClient()
    _inject_penalties(client, frequency_penalty=0.4, presence_penalty=0.1)
    await client.chat.completions.create(model="m", messages=[])
    assert client.chat.completions.seen["frequency_penalty"] == 0.4
    assert client.chat.completions.seen["presence_penalty"] == 0.1


@pytest.mark.asyncio
async def test_caller_supplied_values_win():
    """setdefault, not overwrite: an explicit per-call value must survive."""
    client = _FakeClient()
    _inject_penalties(client, frequency_penalty=0.4, presence_penalty=0.1)
    await client.chat.completions.create(model="m", messages=[], frequency_penalty=0.9)
    assert client.chat.completions.seen["frequency_penalty"] == 0.9


@pytest.mark.asyncio
async def test_preserves_other_kwargs():
    client = _FakeClient()
    _inject_penalties(client, frequency_penalty=0.2, presence_penalty=0.0)
    await client.chat.completions.create(
        model="m", messages=[], max_tokens=16384, response_format={"type": "json_object"})
    seen = client.chat.completions.seen
    assert seen["max_tokens"] == 16384
    assert seen["response_format"] == {"type": "json_object"}


def test_zero_penalties_leave_the_client_unwrapped():
    """Default 0.0/0.0 must not touch the proven gpt-5-mini path -- some reasoning
    endpoints reject these params outright."""
    unwrapped = _llm_client(_settings(
        llm_base_url="http://local/v1", llm_model="m")).client.chat.completions.create
    wrapped = _llm_client(_settings(
        llm_base_url="http://local/v1", llm_model="m",
        llm_frequency_penalty=0.3)).client.chat.completions.create
    # The unwrapped one is the SDK's own bound method; the wrapped one is our closure.
    assert unwrapped.__qualname__ != wrapped.__qualname__
    assert wrapped.__closure__ is not None
