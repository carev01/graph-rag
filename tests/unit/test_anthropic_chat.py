"""Claude behind the chat-completions surface (graph_extract.anthropic_chat)."""
import asyncio
from types import SimpleNamespace

import pytest

import graph_extract.usage as usage_mod
from graph_extract.anthropic_chat import AnthropicChat, is_anthropic, split_system
from graph_extract.config import ExtractSettings
from graph_extract.usage import bounded_llm_client, usable_content
from theme_builder.report import _report_client_and_model, _verify_client_and_model

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


def _msg(text_blocks, stop="end_turn", **usage):
    content = [SimpleNamespace(type="thinking", thinking="")]
    content += [SimpleNamespace(type="text", text=t) for t in text_blocks]
    u = dict(input_tokens=100, output_tokens=20, cache_read_input_tokens=0,
             cache_creation_input_tokens=0)
    u.update(usage)
    return SimpleNamespace(id="msg_1", model="claude-haiku-5-5", content=content,
                           stop_reason=stop, stop_details=None, usage=SimpleNamespace(**u))


class _FakeStream:
    def __init__(self, reply):
        self.reply = reply

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_final_message(self):
        return self.reply


class _FakeMessages:
    def __init__(self, reply):
        self.reply, self.seen = reply, None

    def stream(self, **kwargs):
        self.seen = kwargs
        return _FakeStream(self.reply)


def _adapter(reply, effort=""):
    fake = SimpleNamespace(messages=_FakeMessages(reply))
    return AnthropicChat(api_key="k", reasoning_effort=effort, client=fake), fake.messages


def _call(client, **kw):
    kw = {"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "hi"}],
          "max_tokens": 4000, "temperature": 0, **kw}
    return asyncio.run(client.chat.completions.create(**kw))


def test_text_usage_and_finish_reason_map_to_the_chat_shape():
    client, _ = _adapter(_msg(['{"ok": ', 'true}'], cache_read_input_tokens=30,
                              cache_creation_input_tokens=5))
    resp = _call(client)
    assert usable_content(resp) == '{"ok": true}'
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens == 135          # uncached + cache read + cache write
    assert resp.usage.completion_tokens == 20
    assert resp.usage.prompt_tokens_details.cached_tokens == 30


def test_an_empty_reply_is_unusable_not_an_empty_string():
    client, _ = _adapter(_msg(["  "], stop="max_tokens"))
    resp = _call(client)
    assert resp.choices[0].message.content is None
    assert usable_content(resp) is None
    assert resp.choices[0].finish_reason == "length"


def test_system_messages_become_the_system_parameter_and_temperature_is_dropped():
    client, seen = _adapter(_msg(["x"]))
    _call(client, messages=[{"role": "system", "content": "be terse"},
                            {"role": "user", "content": "q"}])
    assert seen.seen["system"] == [{"type": "text", "text": "be terse",
                                    "cache_control": {"type": "ephemeral"}}]
    assert seen.seen["messages"] == [{"role": "user", "content": "q"}]
    assert "temperature" not in seen.seen
    assert split_system([{"role": "user", "content": "q"}]) == (None, [
        {"role": "user", "content": "q"}])


@pytest.mark.parametrize("effort, sent", [("low", {"effort": "low"}),
                                          ("HIGH", {"effort": "high"}),
                                          ("", None), ("minimal", None)])
def test_effort_is_sent_only_when_claude_accepts_it(effort, sent):
    client, seen = _adapter(_msg(["x"]), effort=effort)
    _call(client)
    assert seen.seen.get("output_config") == sent


def test_is_anthropic():
    assert is_anthropic("https://api.anthropic.com")
    assert not is_anthropic("https://openrouter.ai/api/v1")
    assert not is_anthropic("")


def test_bounded_client_routes_anthropic_and_tallies_usage(monkeypatch):
    built = {}

    class _Spy(AnthropicChat):
        def __init__(self, **kw):
            built.update(kw)
            fake = SimpleNamespace(messages=_FakeMessages(_msg(["ok"])))
            super().__init__(api_key=kw["api_key"], reasoning_effort=kw["reasoning_effort"],
                             client=fake)

    monkeypatch.setattr(usage_mod, "AnthropicChat", _Spy)
    tallied = []
    monkeypatch.setattr(usage_mod, "record",
                        lambda kind, **kw: tallied.append((kind, kw)))
    client = bounded_llm_client("https://api.anthropic.com", "key", reasoning_effort="low",
                                timeout=60.0, max_retries=2)
    resp = _call(client)
    assert usable_content(resp) == "ok"
    assert built == {"api_key": "key", "reasoning_effort": "low", "timeout": 60.0,
                     "max_retries": 2}
    assert tallied == [("llm", {"prompt": 100, "completion": 20, "cached": 0})]


def test_report_tier_on_claude_with_the_verifier_elsewhere():
    s = ExtractSettings(_env_file=None, **_MIN,
                        report_llm_base_url="https://api.anthropic.com",
                        report_llm_model="claude-haiku-5-5", report_llm_api_key="ak",
                        verify_llm_base_url="https://openrouter.ai/api/v1",
                        verify_llm_model="~deepseek/deepseek-v4-flash-latest",
                        verify_llm_api_key="ok")
    rclient, rmodel = _report_client_and_model(s)
    vclient, vmodel = _verify_client_and_model(s)
    assert isinstance(rclient, AnthropicChat) and rmodel == "claude-haiku-5-5"
    assert not isinstance(vclient, AnthropicChat)
    assert vmodel == "~deepseek/deepseek-v4-flash-latest"


def test_close_closes_the_sdk_client():
    """theme-build closes every tier client on exit; a missing close() crashed it."""
    closed = []

    async def _close():
        closed.append(True)

    fake = SimpleNamespace(messages=_FakeMessages(_msg(["x"])), close=_close)
    asyncio.run(AnthropicChat(api_key="k", client=fake).close())
    assert closed == [True]


def test_a_reply_cut_off_at_max_tokens_is_logged(caplog):
    client, _ = _adapter(_msg(['{"title": "trunc'], stop="max_tokens", output_tokens=16000))
    with caplog.at_level("WARNING"):
        resp = _call(client, max_tokens=16000)
    assert resp.choices[0].finish_reason == "length"
    assert any("hit max_tokens=16000" in r.getMessage() for r in caplog.records)


def test_only_the_system_block_carries_a_cache_breakpoint():
    # BACKLOG 58: the static prefix is cached; per-call user content never is.
    client, seen = _adapter(_msg(["x"]))
    _call(client, messages=[{"role": "system", "content": "rules"},
                            {"role": "user", "content": "chunk"}])
    assert seen.seen["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert seen.seen["messages"] == [{"role": "user", "content": "chunk"}]
    assert "cache_control" not in seen.seen


def test_no_system_message_sends_no_system_and_no_breakpoint():
    client, seen = _adapter(_msg(["x"]))
    _call(client)
    assert "system" not in seen.seen and "cache_control" not in seen.seen


def test_cache_system_false_sends_the_plain_system_string():
    fake = SimpleNamespace(messages=_FakeMessages(_msg(["x"])))
    client = AnthropicChat(api_key="k", client=fake, cache_system=False)
    _call(client, messages=[{"role": "system", "content": "rules"},
                            {"role": "user", "content": "q"}])
    assert fake.messages.seen["system"] == "rules"


def test_cache_reads_are_reported_as_cached_prompt_tokens():
    client, _ = _adapter(_msg(["x"], input_tokens=15, cache_read_input_tokens=782))
    resp = _call(client, messages=[{"role": "system", "content": "rules"},
                                   {"role": "user", "content": "q"}])
    assert resp.usage.prompt_tokens == 797
    assert resp.usage.prompt_tokens_details.cached_tokens == 782
