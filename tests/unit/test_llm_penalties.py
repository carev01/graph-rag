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
from graph_extract.graphiti_client import _inject_penalties


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


class _FakeCap:
    """Captures the response_format the wrapper forwards."""

    def __init__(self) -> None:
        self.seen: dict = {}

    async def create(self, **kwargs):
        self.seen = kwargs
        return "resp"


def _client_with(fmt_holder):
    c = _FakeClient()
    c.chat.completions = fmt_holder
    return c


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


@pytest.mark.asyncio
async def test_zero_penalties_send_no_penalty_params():
    """Default 0.0/0.0 must not send these at all -- some reasoning endpoints reject
    them outright, and the proven gpt-5-mini path must stay byte-identical.

    Asserted behaviourally rather than by inspecting wrappers: the index-array bound
    defaults ON, so the client is always wrapped and "is it wrapped" proves nothing.
    """
    from graph_extract.graphiti_client import _bound_index_arrays

    holder = _FakeCap()
    client = _client_with(holder)
    _bound_index_arrays(client, 25)          # the wrapper that IS applied by default
    await client.chat.completions.create(model="m", messages=[])
    assert "frequency_penalty" not in holder.seen
    assert "presence_penalty" not in holder.seen


def test_penalty_wrapper_is_only_installed_when_configured():
    """_llm_client must not add the penalty wrapper at the 0.0 default."""
    import inspect

    from graph_extract import graphiti_client as gc

    src = inspect.getsource(gc._llm_client)
    assert "if s.llm_frequency_penalty or s.llm_presence_penalty:" in src
    assert "_inject_penalties" in src
    # sanity: a settings object at defaults reports no penalties
    assert _settings().llm_frequency_penalty == 0.0
    assert _settings().llm_presence_penalty == 0.0


@pytest.mark.asyncio
async def test_bounds_integer_arrays_but_not_object_arrays():
    """Index lists get capped; the `edges` list of objects must NOT be, or we would
    silently cap how many facts a single call can extract."""
    from graph_extract.graphiti_client import _bound_index_arrays
    holder = _FakeCap()
    client = _client_with(holder)
    _bound_index_arrays(client, 25)
    schema = {"type": "json_schema", "json_schema": {"schema": {"properties": {
        "edges": {"type": "array", "items": {"type": "object", "properties": {
            "episode_indices": {"type": "array", "items": {"type": "integer"}}}}}}}}}
    await client.chat.completions.create(model="m", messages=[], response_format=schema)
    out = holder.seen["response_format"]["json_schema"]["schema"]["properties"]
    assert out["edges"]["items"]["properties"]["episode_indices"]["maxItems"] == 25
    assert "maxItems" not in out["edges"]


@pytest.mark.asyncio
async def test_does_not_mutate_the_callers_schema():
    """graphiti reuses its schema dict across calls; mutating it would leak."""
    from graph_extract.graphiti_client import _bound_index_arrays
    holder = _FakeCap()
    client = _client_with(holder)
    _bound_index_arrays(client, 25)
    original = {"type": "json_schema", "json_schema": {"schema": {
        "properties": {"idx": {"type": "array", "items": {"type": "integer"}}}}}}
    await client.chat.completions.create(model="m", messages=[], response_format=original)
    assert "maxItems" not in original["json_schema"]["schema"]["properties"]["idx"]


@pytest.mark.asyncio
async def test_existing_maxitems_is_respected():
    from graph_extract.graphiti_client import _bound_index_arrays
    holder = _FakeCap()
    client = _client_with(holder)
    _bound_index_arrays(client, 25)
    schema = {"json_schema": {"schema": {"properties": {
        "idx": {"type": "array", "items": {"type": "integer"}, "maxItems": 3}}}}}
    await client.chat.completions.create(model="m", messages=[], response_format=schema)
    got = holder.seen["response_format"]["json_schema"]["schema"]["properties"]["idx"]
    assert got["maxItems"] == 3


@pytest.mark.asyncio
async def test_generate_report_retries_a_choices_less_response():
    """A flaky provider can return HTTP 200 with an error payload and no choices.
    Indexing choices[0] then raised TypeError and the caller dropped the community
    outright -- 3 of 29 were lost that way on the first real theme-build."""
    from theme_builder.report import generate_report

    class _Resp:
        def __init__(self, choices):
            self.choices = choices

    class _Msg:
        content = '{"title":"t","summary":"s","rating":5,"rating_explanation":"e",' \
                  '"full_report":[]}'

    class _Choice:
        message = _Msg()

    class _Ctx:
        text = "ctx"
        fact_uuids: set = set()

    calls = {"n": 0}

    class _Client:
        def __init__(self):
            self.chat = type("C", (), {})()
            self.chat.completions = self

        async def create(self, **kw):
            calls["n"] += 1
            # first attempt: no choices at all; second: a valid reply
            return _Resp(None) if calls["n"] == 1 else _Resp([_Choice()])

    rep = await generate_report(_Client(), "m", _Ctx(), 16000)
    assert calls["n"] == 2, "the choices-less reply must consume a retry, not raise"
    assert rep is not None, "the retry's valid reply must produce a report"
