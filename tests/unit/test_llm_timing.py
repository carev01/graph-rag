"""BACKLOG 31. Two questions this must answer on the next paid run:
where an episode's ~45 s goes (only dedup was ever counted -- `_extract_edge_timestamps`
issues a second call per new fact that nothing measured), and whether the provider
actually runs our 20-wide concurrent dedup calls in parallel. The second decides
strategy: if latency is flat across in-flight buckets the calls really are parallel and
reducing call COUNT saves nothing; if it climbs, count is back on the critical path.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from graph_extract.dedup_guard import (
    DEDUP_PROMPT_NAME, DedupIndexStats, install_dedup_guard,
)
from graph_extract.llm_timing import PromptTimings, _bucket


def test_buckets_straddle_the_semaphore_limit():
    assert _bucket(1) == "1"
    assert _bucket(3) == "2-4"
    assert _bucket(7) == "5-9"          # mean 6.9 facts/episode lands mid-range
    assert _bucket(19) == "10-19"
    assert _bucket(20) == "20+"


def test_report_is_safe_with_no_calls():
    assert "no calls" in PromptTimings().report()


def test_totals_and_percentiles_are_per_prompt_name():
    t = PromptTimings()
    for ms in (100.0, 200.0, 300.0):
        at = t.start()
        t.finish("a.prompt", ms, at)
    at = t.start()
    t.finish("b.prompt", 50.0, at)
    assert t.by_prompt["a.prompt"].calls == 3
    assert t.by_prompt["b.prompt"].calls == 1
    rep = t.report()
    assert "a.prompt" in rep and "b.prompt" in rep
    # the heavier prompt is listed first -- the report is for finding the cost
    assert rep.index("a.prompt") < rep.index("b.prompt")


def test_inflight_is_recorded_at_dispatch_not_at_completion():
    """The whole experiment rests on this. A call dispatched alongside nine others
    must be bucketed by that, even though by the time it finishes the others may
    have drained."""
    t = PromptTimings()
    ats = [t.start() for _ in range(10)]
    assert ats[-1] == 10 and t.max_inflight == 10
    # they complete one at a time, in-flight falling as they go
    for at in ats:
        t.finish("p", 123.0, at)
    assert t.inflight == 0
    buckets = t.by_prompt["p"].by_inflight
    assert sorted(buckets) == ["1", "10-19", "2-4", "5-9"]
    assert len(buckets["10-19"]) == 1          # only the tenth dispatch saw 10
    assert len(buckets["5-9"]) == 5            # dispatches 5..9


def test_merge_combines_two_tiers():
    a, b = PromptTimings(), PromptTimings()
    at = a.start()
    a.finish("p", 10.0, at)
    for _ in range(3):
        at = b.start()
        b.finish("p", 20.0, at)
    b.max_inflight = 7
    a.merge(b)
    assert a.by_prompt["p"].calls == 4
    assert a.max_inflight == 7


@pytest.mark.asyncio
async def test_the_guard_times_non_dedup_prompts_too():
    """`_extract_edge_timestamps` is a second per-fact LLM call that the dedup
    counters never saw. It must show up here."""
    async def _reply(messages, *a, **kw):
        return {"duplicate_facts": [], "contradicted_facts": []}

    timings = PromptTimings()
    holder = SimpleNamespace(llm_client=SimpleNamespace(generate_response=_reply))
    install_dedup_guard(holder, fallback=None, unscoped=DedupIndexStats(),
                        timings=timings)
    await holder.llm_client.generate_response([], prompt_name="extract_edges.extract_edge_dates")
    await holder.llm_client.generate_response([], prompt_name="dedupe_nodes.node_list")
    assert set(timings.by_prompt) == {"extract_edges.extract_edge_dates",
                                      "dedupe_nodes.node_list"}
    assert all(st.calls == 1 for st in timings.by_prompt.values())


@pytest.mark.asyncio
async def test_concurrent_calls_are_bucketed_by_real_concurrency():
    """Drives the guard with genuinely overlapping calls, the shape the real
    dispatch has."""
    started = asyncio.Event()

    async def _slow(messages, *a, **kw):
        started.set()
        await asyncio.sleep(0.01)
        return {}

    timings = PromptTimings()
    holder = SimpleNamespace(llm_client=SimpleNamespace(generate_response=_slow))
    install_dedup_guard(holder, fallback=None, unscoped=DedupIndexStats(),
                        timings=timings)
    await asyncio.gather(*[
        holder.llm_client.generate_response([], prompt_name="p") for _ in range(6)])
    assert timings.max_inflight == 6
    assert timings.inflight == 0
    assert timings.by_prompt["p"].calls == 6
    assert "5-9" in timings.by_prompt["p"].by_inflight


@pytest.mark.asyncio
async def test_a_failing_call_still_decrements_inflight():
    """Otherwise one error poisons every later bucket."""
    async def _boom(messages, *a, **kw):
        raise RuntimeError("provider down")

    timings = PromptTimings()
    holder = SimpleNamespace(llm_client=SimpleNamespace(generate_response=_boom))
    install_dedup_guard(holder, fallback=None, unscoped=DedupIndexStats(),
                        timings=timings)
    with pytest.raises(RuntimeError):
        await holder.llm_client.generate_response([], prompt_name="p")
    assert timings.inflight == 0
    assert timings.by_prompt["p"].calls == 1


def test_the_report_names_what_the_reader_must_decide():
    t = PromptTimings()
    at = t.start()
    t.finish(DEDUP_PROMPT_NAME, 1800.0, at)
    rep = t.report()
    assert "in-flight" in rep
    assert "flat" in rep and "queues" in rep, "the report must say how to read it"


@pytest.mark.asyncio
async def test_a_slow_call_is_bucketed_by_dispatch_concurrency_not_completion():
    """The discriminating case, found by mutation testing: dispatch a slow call
    LAST, when five others are already in flight, and let it finish alone. At
    dispatch it saw 6 in flight ("5-9"); at completion it sees 1. Recording at
    completion would file it under "1" and flatten exactly the signal the
    provider-serialisation experiment reads."""
    async def _reply(messages, *a, **kw):
        await asyncio.sleep(0.05 if kw.get("prompt_name") == "slow" else 0.005)
        return {}

    timings = PromptTimings()
    holder = SimpleNamespace(llm_client=SimpleNamespace(generate_response=_reply))
    install_dedup_guard(holder, fallback=None, unscoped=DedupIndexStats(),
                        timings=timings)

    async def _fast(i):
        await holder.llm_client.generate_response([], prompt_name="fast")

    fast = [asyncio.create_task(_fast(i)) for i in range(5)]
    await asyncio.sleep(0)                       # let all five dispatch
    slow = asyncio.create_task(
        holder.llm_client.generate_response([], prompt_name="slow"))
    await asyncio.gather(*fast, slow)

    assert timings.max_inflight == 6
    slow_buckets = timings.by_prompt["slow"].by_inflight
    assert list(slow_buckets) == ["5-9"], (
        f"slow call bucketed {list(slow_buckets)}; it was dispatched with 6 in "
        "flight and finished with 1, so anything else means completion-time")
