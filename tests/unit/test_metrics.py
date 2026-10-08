"""Ingestion metrics: worker counters and the queue exporter (graph_sync.metrics)."""
from __future__ import annotations

import asyncio

import httpx
import pytest
from prometheus_client import REGISTRY

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


def _jobs(outcome):
    return ("graphrag_worker_jobs_total", {"tier": T, "outcome": outcome})


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
    return REGISTRY.get_sample_value("graphrag_queue_jobs", dict(
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
    assert REGISTRY.get_sample_value("graphrag_queue_done_last_hour", {"vendor": "Veeam"}) == 7
    assert REGISTRY.get_sample_value("graphrag_queue_retrying_jobs") == 3
    assert REGISTRY.get_sample_value("graphrag_tokens_today") == 12345
    assert REGISTRY.get_sample_value("graphrag_exporter_last_refresh_timestamp_seconds") > 0


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

    assert calls["n"] == 2, "the failing second refresh must have run"
    assert _q("Veeam", "VBR", "User Guide", "done") == 9
    assert REGISTRY.get_sample_value("graphrag_openrouter_credits_remaining_usd") == 21.5
    assert REGISTRY.get_sample_value("graphrag_gpu_hourly_cost_usd") == 0.383
