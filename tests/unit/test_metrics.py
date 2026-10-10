"""Ingestion metrics: worker counters and the queue exporter (graph_sync.metrics)."""
from __future__ import annotations

import asyncio

import httpx
import pytest
from prometheus_client import REGISTRY

from graph_sync.metrics import QUEUE_REGISTRY

from graph_sync import metrics
from graph_sync.semantic_worker import ProviderUnreachable, run_worker, run_worker_once
from tests.unit.test_worker_credit_pause import _KW, _Ingest, _job, _Store

pytestmark = pytest.mark.asyncio

T = metrics.TIER
_REQ = httpx.Request("POST", "http://graph-rag-vast-tunnel:8000/v1/chat/completions")


def _v(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


class _Delta:
    """Counter values before/after a block: the registry is process-global."""

    def __init__(self, *series: tuple[str, dict]):
        self.series = series

    def __enter__(self):
        self.before = [_v(n, **lbl) for n, lbl in self.series]
        return self

    def __exit__(self, *exc):
        self.after = [_v(n, **lbl) for n, lbl in self.series]

    @property
    def deltas(self):
        return [a - b for a, b in zip(self.after, self.before)]


def _jobs(outcome, route="-"):
    return ("graphrag_worker_jobs_total", {"tier": T, "outcome": outcome, "route": route})


async def test_done_and_failed_jobs_are_counted_with_their_time():
    store = _Store([[_job(1, "a1"), _job(2, "a2")]])
    hist = ("graphrag_worker_job_seconds_count", {"tier": T, "outcome": "done"})
    with _Delta(_jobs("done"), _jobs("failed"), hist) as d:
        await run_worker_once(store, _Ingest({"a2": ValueError("bad JSON")}), **_KW)
    assert d.deltas == [1, 1, 1]


async def test_a_halted_batch_counts_its_deferrals_and_the_pause():
    def conn_error():
        try:
            try:
                raise httpx.ConnectError("refused", request=_REQ)
            except httpx.ConnectError as inner:
                raise RuntimeError("graphiti") from inner
        except RuntimeError as e:
            return e

    store = _Store([[_job(1, "a1"), _job(2, "a2")]])
    pauses = ("graphrag_worker_pauses_total", {"tier": T, "reason": "unreachable"})
    with _Delta(_jobs("deferred_halt"), _jobs("failed"), pauses) as d:
        await run_worker(store, _Ingest({"a1": conn_error()}), poll_seconds=0.01,
                         stop_event=asyncio.Event(), max_batches=1,
                         **dict(_KW, unreachable_pause_seconds=0.01))
    assert d.deltas == [2, 0, 1]


async def test_run_worker_once_still_raises_for_the_pause():
    store = _Store([[_job(1, "a1")]])
    err = httpx.ConnectError("refused", request=_REQ)
    with pytest.raises(ProviderUnreachable):
        await run_worker_once(store, _Ingest({"a1": err}), **_KW)


async def test_tokens_are_counted_by_kind():
    series = [("graphrag_worker_llm_tokens_total", {"tier": T, "kind": k})
              for k in ("prompt", "cached", "completion")]
    with _Delta(*series) as d:
        metrics.job_tokens(1000, 200, 50)
        metrics.job_tokens(0, 0, 0)
    assert d.deltas == [1000, 200, 50]


class _Pool:
    def __init__(self, counts, done_1h, retrying=3, today=12345):
        self.counts, self.done_1h, self.retrying, self.today = counts, done_1h, retrying, today

    async def fetch(self, sql):
        return self.counts if "GROUP BY 1, 2" in sql else self.done_1h

    async def fetchval(self, sql):
        return self.today if "token_ledger" in sql else self.retrying


_NAMES = {"s1": ("Veeam", "VBR", "User Guide"), "s2": ("Veeam", "VBR", "Agent Guide")}


def _q(vendor, product, source, status):
    return QUEUE_REGISTRY.get_sample_value("graphrag_queue_jobs", dict(
        vendor=vendor, product=product, source=source, status=status))


async def test_refresh_names_sources_and_maps_the_unknown():
    pool = _Pool(
        counts=[{"source_id": "s1", "status": "done", "n": 10},
                {"source_id": "s1", "status": "pending", "n": 4},
                {"source_id": "s2", "status": "done", "n": 1},
                {"source_id": "zz", "status": "pending", "n": 2},
                {"source_id": "", "status": "pending", "n": 7}],
        done_1h=[{"source_id": "s1", "n": 5}, {"source_id": "s2", "n": 2}])

    await metrics.refresh_queue(pool, _NAMES)

    assert _q("Veeam", "VBR", "User Guide", "done") == 10
    assert _q("Veeam", "VBR", "User Guide", "pending") == 4
    assert _q("(unknown)", "(unknown)", "zz", "pending") == 2
    assert _q("(no source)", "(no source)", "(no source)", "pending") == 7
    assert QUEUE_REGISTRY.get_sample_value("graphrag_queue_done_last_hour", {"vendor": "Veeam"}) == 7
    assert QUEUE_REGISTRY.get_sample_value("graphrag_queue_retrying_jobs") == 3
    assert QUEUE_REGISTRY.get_sample_value("graphrag_tokens_today") == 12345
    assert QUEUE_REGISTRY.get_sample_value("graphrag_exporter_last_refresh_timestamp_seconds") > 0


async def test_refresh_drops_label_sets_that_disappeared():
    await metrics.refresh_queue(_Pool([{"source_id": "s2", "status": "pending", "n": 1}], []),
                                _NAMES)
    await metrics.refresh_queue(_Pool([{"source_id": "s2", "status": "done", "n": 1}], []),
                                _NAMES)
    assert _q("Veeam", "VBR", "Agent Guide", "pending") is None
    assert _q("Veeam", "VBR", "Agent Guide", "done") == 1


async def test_openrouter_credits_reads_the_balance_and_never_raises():
    ok = httpx.MockTransport(lambda r: httpx.Response(
        200, json={"data": {"total_credits": 330, "total_usage": 314.51}}))
    assert await metrics.openrouter_credits("k", transport=ok) == pytest.approx(15.49)
    bad = httpx.MockTransport(lambda r: httpx.Response(401))
    assert await metrics.openrouter_credits("k", transport=bad) is None


async def test_exporter_serves_the_last_good_values_when_a_refresh_fails():
    stop = asyncio.Event()
    calls = {"n": 0}

    async def names():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("neo4j down")
        return _NAMES

    passes = {"n": 0}

    async def credits():
        passes["n"] += 1
        if passes["n"] == 2:      # the second pass, whose names load failed, is over
            stop.set()
        return 21.5

    await metrics.run_exporter(
        pool=_Pool([{"source_id": "s1", "status": "done", "n": 9}], []), port=0,
        interval=0.01, stop=stop, names_loader=names, credits_loader=credits,
        gpu_hourly_cost=0.383, names_every=0.0, credits_every=0.0)

    assert calls["n"] == 2, "the failing second names load must have run"
    assert _q("Veeam", "VBR", "User Guide", "done") == 9, \
        "the queue keeps refreshing with the last good names"
    assert QUEUE_REGISTRY.get_sample_value("graphrag_openrouter_credits_remaining_usd") == 21.5
    assert QUEUE_REGISTRY.get_sample_value("graphrag_gpu_hourly_cost_usd") == 0.383


async def test_workers_never_serve_the_exporter_gauges():
    """On one shared registry every worker served the queue gauges as 0, and each
    dashboard panel got 8 bogus zeros beside the exporter's real value."""
    await metrics.refresh_queue(_Pool([{"source_id": "s1", "status": "done", "n": 9}], []),
                                _NAMES)
    worker_names = {m.name for m in REGISTRY.collect()}
    queue_names = {m.name for m in QUEUE_REGISTRY.collect()}
    assert queue_names, "the exporter registry must hold the queue gauges"
    assert not worker_names & queue_names
    assert not any(n.startswith(("graphrag_queue", "graphrag_openrouter", "graphrag_gpu",
                                 "graphrag_tokens_today", "graphrag_exporter"))
                   for n in worker_names)


class _RoutedIngest(_Ingest):
    """Returns results the router has routed, like the real driver."""

    async def ingest_article(self, article_id):
        res = await super().ingest_article(article_id)
        res.tier, res.routed = "cheap", True
        return res


async def test_a_done_job_is_counted_under_its_routing_tier(caplog):
    store = _Store([[_job(1, "a1")]])
    with _Delta(_jobs("done", "cheap")) as d, caplog.at_level("INFO"):
        await run_worker_once(store, _RoutedIngest({}), **_KW)
    assert d.deltas == [1]
    assert any("done: article=a1 tier=cheap" in r.getMessage() for r in caplog.records)


async def test_a_failed_job_logs_its_tier_and_the_call_that_failed(caplog):
    req = httpx.Request("POST", "http://graph-rag-vast-tunnel:8000/v1/chat/completions",
                        json={"model": "qwen35-graphrag", "messages": []})
    try:
        try:
            raise httpx.ReadTimeout("timed out", request=req)
        except httpx.ReadTimeout as inner:
            raise ValueError("graphiti gave up") from inner
    except ValueError as e:
        e.add_note("graph-rag tier=cheap")
        err = e
    store = _Store([[_job(1, "a1")]])
    with _Delta(_jobs("failed", "cheap")) as d, caplog.at_level("ERROR"):
        await run_worker_once(store, _Ingest({"a1": err}), **_KW)
    assert d.deltas == [1]
    msg = next(r.getMessage() for r in caplog.records if "failed:" in r.getMessage())
    assert "tier=cheap" in msg and "call=qwen35-graphrag@graph-rag-vast-tunnel" in msg
