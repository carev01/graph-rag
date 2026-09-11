"""The report verifier must never be the report model. A model grading its own
output inflates the result in a way the result cannot reveal — the same guard
answer_api.eval_router applies to the faithfulness judge."""
import asyncio

import pytest

import theme_builder.report as report_mod
from graph_extract.config import ExtractSettings
from theme_builder.report import _verify_client_and_model


_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


def _settings(**kw):
    base = dict(_env_file=None, judge_base_url="https://judge.example/v1",
                judge_model="z-ai/glm-5.3-flash", judge_api_key="k", **_MIN)
    base.update(kw)
    return ExtractSettings(**base)


def test_uses_verify_llm_when_set():
    s = _settings(verify_llm_base_url="https://verify.example/v1",
                  verify_llm_model="~deepseek/deepseek-v4-flash-latest",
                  verify_llm_api_key="vk")
    _, model = _verify_client_and_model(s)
    assert model == "~deepseek/deepseek-v4-flash-latest"


def test_falls_back_to_eval_judge():
    s = _settings(eval_judge_base_url="https://verify.example/v1",
                  eval_judge_model="~deepseek/deepseek-v4-flash-latest",
                  eval_judge_api_key="vk")
    _, model = _verify_client_and_model(s)
    assert model == "~deepseek/deepseek-v4-flash-latest"


def test_raises_when_it_resolves_to_the_report_model():
    """The whole point of the guard: self-grading is silently worthless."""
    s = _settings(verify_llm_base_url="https://judge.example/v1",
                  verify_llm_model="z-ai/glm-5.3-flash")
    with pytest.raises(ValueError, match="report model"):
        _verify_client_and_model(s)


def test_the_same_model_behind_a_different_base_url_still_raises():
    """The guard is about the MODEL, not the endpoint. Comparing base URL AND model
    let the identical model reached through a second provider/gateway self-grade --
    the failure modes it must not share travel with the weights, not the host."""
    s = _settings(verify_llm_base_url="https://some-other-gateway.example/v1",
                  verify_llm_model="z-ai/glm-5.3-flash")
    with pytest.raises(ValueError, match="report model"):
        _verify_client_and_model(s)


def test_raises_when_nothing_is_configured():
    with pytest.raises(ValueError, match="verif"):
        _verify_client_and_model(_settings())


# --- reasoning-budget bound (measured: max_tokens=4000/12000 both returned
# finish_reason='length', completion_tokens exhausted, content=None on an
# 11-finding report -- the verifier's whole budget went to reasoning tokens and
# it never wrote a verdict). _prefer_fast_provider fixes this for the report
# client already; these tests pin the same fix on the VERIFIER client. ---

class _FakeCompletions:
    def __init__(self) -> None:
        self.seen: dict = {}

    async def create(self, **kwargs):
        self.seen = kwargs
        return type("R", (), {"choices": [], "usage": None})()


class _FakeAsyncOpenAI:
    """Stands in for openai.AsyncOpenAI so _verify_client_and_model's real
    instrument()/_prefer_fast_provider() wrapping runs against a client that
    never makes a network call."""

    def __init__(self, *a, **kw) -> None:
        self.chat = type("C", (), {})()
        self.chat.completions = _FakeCompletions()


def test_reasoning_bound_applied_for_openrouter(monkeypatch):
    monkeypatch.setattr(report_mod, "AsyncOpenAI", _FakeAsyncOpenAI)
    s = _settings(verify_llm_base_url="https://openrouter.ai/api/v1",
                  verify_llm_model="~deepseek/deepseek-v4-flash-latest",
                  verify_llm_api_key="vk", verify_reasoning_effort="low")
    client, model = _verify_client_and_model(s)
    asyncio.run(client.chat.completions.create(model=model, messages=[], max_tokens=4000))
    seen = client.chat.completions.seen
    assert seen["extra_body"]["reasoning"] == {"effort": "low"}


def test_reasoning_bound_not_applied_for_non_openrouter(monkeypatch):
    """Matching _report_client_and_model's guard: no OpenRouter base, no
    provider/reasoning routing added at all."""
    monkeypatch.setattr(report_mod, "AsyncOpenAI", _FakeAsyncOpenAI)
    s = _settings(verify_llm_base_url="https://verify.example/v1",
                  verify_llm_model="~deepseek/deepseek-v4-flash-latest",
                  verify_llm_api_key="vk", verify_reasoning_effort="low")
    client, model = _verify_client_and_model(s)
    asyncio.run(client.chat.completions.create(model=model, messages=[], max_tokens=4000))
    seen = client.chat.completions.seen
    assert "extra_body" not in seen


def test_configured_effort_value_is_used_not_hardcoded(monkeypatch):
    """The setting is actually read: a non-default effort must show up verbatim."""
    monkeypatch.setattr(report_mod, "AsyncOpenAI", _FakeAsyncOpenAI)
    s = _settings(verify_llm_base_url="https://openrouter.ai/api/v1",
                  verify_llm_model="~deepseek/deepseek-v4-flash-latest",
                  verify_llm_api_key="vk", verify_reasoning_effort="medium")
    client, model = _verify_client_and_model(s)
    asyncio.run(client.chat.completions.create(model=model, messages=[], max_tokens=4000))
    seen = client.chat.completions.seen
    assert seen["extra_body"]["reasoning"] == {"effort": "medium"}
