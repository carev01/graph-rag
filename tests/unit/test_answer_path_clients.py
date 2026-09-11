"""Every answer-path LLM client is built through one shared helper that gives it
a bounded timeout, throughput routing and a per-tier reasoning bound.

Before this, all four (synthesis/reduce, map, eval judge, router classifier)
were bare `AsyncOpenAI(api_key, base_url)`: no timeout, so one bad OpenRouter
route could hang an /answer request indefinitely (measured: 478s and 528s
eval questions, a 2h09m eval against ~20 min), and no reasoning bound, so a
reasoning model could spend its whole max_tokens on reasoning and return no
content (8 empty-content retries in ~40 reduce calls). The fix already existed
as theme_builder.report._prefer_fast_provider -- this is its generalisation
into graph_extract.usage, the layer both services import.

The reasoning bound is MODEL-bound, not tier-bound. Measured 2026-09-11 on
upstage/solar-pro4 (the map + classifier model): it does not reason by default
(reasoning_tokens=0), and sending `reasoning: {effort: low}` SWITCHES THINKING
ON -- classifier at max_tokens=8 and 64: content=None every call; map at
max_tokens=2000: one of two reports finish=length with all 2000 tokens on
reasoning. So each tier carries its own effort setting and the classifier
sends none."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import graph_extract.usage as usage_mod
import theme_builder.report as report_mod
from answer_api.eval_router import _eval_judge_client_and_model
from answer_api.global_search import _map_client_and_model
from answer_api.router import _cheap_classify_client
from answer_api.synthesize import _synthesis_client_and_model
from graph_extract.config import ExtractSettings

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")
_OR = "https://openrouter.ai/api/v1"


def _settings(**kw) -> ExtractSettings:
    base = dict(_env_file=None, judge_base_url=_OR, judge_model="z-ai/glm-5.3-flash",
                judge_api_key="jk", eval_judge_base_url=_OR,
                eval_judge_model="~deepseek/deepseek-v4-flash-latest",
                cheap_llm_api_key="ck", **_MIN)
    base.update(kw)
    return ExtractSettings(**base)


class _FakeCompletions:
    def __init__(self) -> None:
        self.seen: dict = {}

    async def create(self, **kwargs):
        self.seen = kwargs
        return SimpleNamespace(choices=[], usage=None)


class _FakeAsyncOpenAI:
    """Stands in for openai.AsyncOpenAI inside graph_extract.usage so the real
    instrument()/prefer_fast_provider() wrapping runs without a network call,
    and records the constructor kwargs (timeout, max_retries) the builder used."""

    def __init__(self, **kw) -> None:
        self.init_kwargs = kw
        self.chat = SimpleNamespace(completions=_FakeCompletions())


@pytest.fixture
def fake_openai(monkeypatch):
    monkeypatch.setattr(usage_mod, "AsyncOpenAI", _FakeAsyncOpenAI)


def _call(client, model):
    asyncio.run(client.chat.completions.create(model=model, messages=[], max_tokens=100))
    return client.chat.completions.seen


_BUILDERS = [
    ("synthesis", _synthesis_client_and_model),
    ("map", _map_client_and_model),
    ("eval_judge", _eval_judge_client_and_model),
    ("classifier", _cheap_classify_client),
]


@pytest.mark.parametrize("name,builder", _BUILDERS, ids=[b[0] for b in _BUILDERS])
def test_every_answer_path_client_has_a_bounded_timeout(fake_openai, name, builder):
    """A bare client waits forever on a hung route. Each tier must be built
    with a finite timeout and a bounded retry count."""
    client, _ = builder(_settings())
    kw = client.init_kwargs
    assert isinstance(kw.get("timeout"), (int, float)) and 0 < kw["timeout"] < 600, name
    assert isinstance(kw.get("max_retries"), int) and 0 <= kw["max_retries"] <= 5, name


def test_classifier_timeout_is_tighter_than_synthesis(fake_openai):
    """The classifier emits a 2-token label; a route that takes 36s for that
    (measured) is a bad route to cut, not to wait on. Synthesis writes ~1-3k
    tokens and legitimately needs minutes."""
    synth, _ = _synthesis_client_and_model(_settings())
    cheap, _ = _cheap_classify_client(_settings())
    assert cheap.init_kwargs["timeout"] <= 30 < synth.init_kwargs["timeout"]


@pytest.mark.parametrize("builder,effort_field", [
    (_synthesis_client_and_model, "synthesis_reasoning_effort"),
    (_map_client_and_model, "map_reasoning_effort"),
    (_eval_judge_client_and_model, "eval_judge_reasoning_effort"),
], ids=["synthesis", "map", "eval_judge"])
def test_reasoning_bound_and_throughput_routing_on_openrouter(fake_openai, builder,
                                                              effort_field):
    client, model = builder(_settings(**{effort_field: "low"}))
    seen = _call(client, model)
    assert seen["extra_body"]["reasoning"] == {"effort": "low"}
    assert seen["extra_body"]["provider"]["sort"] == "throughput"


def test_configured_effort_value_is_used_not_hardcoded(fake_openai):
    client, model = _synthesis_client_and_model(_settings(synthesis_reasoning_effort="medium"))
    assert _call(client, model)["extra_body"]["reasoning"] == {"effort": "medium"}


def test_synthesis_and_eval_judge_default_to_low_never_off():
    """`low` is the floor, not a tuning default: the verifier measured
    `enabled: false` as a 13-token rubber stamp. The default must keep
    reasoning on AND bounded for the two tiers whose models reason by default."""
    s = _settings()
    assert s.synthesis_reasoning_effort == "low"
    assert s.eval_judge_reasoning_effort == "low"


def test_map_default_sends_no_reasoning_parameter_but_still_routes(fake_openai):
    """Measured: `effort: low` turns thinking ON for solar-pro4 and one of two
    map calls returned nothing. The map default therefore sends no reasoning
    parameter at all -- while still getting the timeout and throughput sort."""
    client, model = _map_client_and_model(_settings())
    seen = _call(client, model)
    assert "reasoning" not in seen["extra_body"]
    assert seen["extra_body"]["provider"]["sort"] == "throughput"


def test_classifier_sends_no_reasoning_parameter_but_still_routes(fake_openai):
    """Same measurement at max_tokens=8: with `effort: low` every reply was
    content=None. The classifier is hard-wired to send none."""
    client, model = _cheap_classify_client(_settings())
    seen = _call(client, model)
    assert "reasoning" not in seen["extra_body"]
    assert seen["extra_body"]["provider"]["sort"] == "throughput"


def test_no_routing_added_for_a_non_openrouter_base(fake_openai):
    """`provider`/`reasoning` are OpenRouter extensions; another gateway must
    receive a plain request -- matching the report tier's guard."""
    client, model = _synthesis_client_and_model(_settings(judge_base_url="https://glm.example/v1"))
    assert "extra_body" not in _call(client, model)


def test_shared_helper_is_reexported_for_theme_builder():
    """theme_builder keeps its `_prefer_fast_provider` name so the report and
    verifier tiers are untouched -- but it must be the SAME object as the shared
    one, not a fourth copy."""
    assert report_mod._prefer_fast_provider is usage_mod.prefer_fast_provider
