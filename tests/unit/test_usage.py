from types import SimpleNamespace

import pytest

from graph_extract import usage
from graph_extract.eval import cost_report
from graph_extract.usage import UsageTally


def test_usage_tally_accumulates():
    t = UsageTally()
    t.add("entity", prompt=100, completion=40)
    t.add("entity", prompt=50, completion=10)
    t.add("edge", prompt=30, completion=5)
    assert t.prompt_tokens == 180 and t.completion_tokens == 55
    assert t.calls == 3
    assert t.by_call["entity"]["calls"] == 2


def _resp(prompt, completion, cached=None):
    u = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    if cached is not None:
        u.prompt_tokens_details = SimpleNamespace(cached_tokens=cached)
    return SimpleNamespace(usage=u)


def test_tally_captures_cached_tokens():
    usage.reset_tally()
    usage._tally_usage(_resp(1000, 200, cached=768))
    t = usage.get_tally()
    assert t.prompt_tokens == 1000 and t.cached_tokens == 768


def test_tally_cached_defaults_zero_when_absent():
    usage.reset_tally()
    usage._tally_usage(_resp(500, 100))  # no prompt_tokens_details
    assert usage.get_tally().cached_tokens == 0


def test_tally_responses_api_cached():
    usage.reset_tally()
    u = SimpleNamespace(input_tokens=800, output_tokens=100,
                        input_tokens_details=SimpleNamespace(cached_tokens=640))
    usage._tally_usage(SimpleNamespace(usage=u))
    assert usage.get_tally().cached_tokens == 640


@pytest.mark.asyncio
async def test_cost_report_includes_cached_tokens_and_hit_rate():
    usage.reset_tally()
    usage._tally_usage(_resp(1000, 200, cached=768))
    report = await cost_report()
    assert report["cached_tokens"] == 768
    assert report["cache_hit_rate"] == pytest.approx(0.768)
