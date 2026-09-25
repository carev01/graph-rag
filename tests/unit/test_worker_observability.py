"""The production worker discarded everything the ingest driver measures.

`run_worker_once` called `ingest.ingest_article(...)` and threw the result away,
so the dedup counters (BACKLOG 33's same-pair number), the reference-time basis,
and the per-prompt timings (BACKLOG 31) were invisible on the ONE path that runs
at scale. The CLI printed them; the worker did not -- so measurements built to
ride on a real ingest would have produced nothing where it mattered.
"""
from __future__ import annotations

import logging

import pytest

from graph_extract.dedup_guard import DedupIndexStats
from graph_extract.ingest_driver import CRAWL_FALLBACK, IngestArticleResult
from graph_extract.llm_timing import PromptTimings
from graph_sync.semantic_worker import run_worker_once

pytestmark = pytest.mark.asyncio


class _Store:
    def __init__(self, jobs):
        self._jobs = jobs
        self.completed: list[str] = []

    async def reap_stale_jobs(self, *a, **k): return None
    async def today_token_total(self): return 0
    async def claim_semantic_jobs(self, batch, include_bootstrap, source_ids=None): return self._jobs
    async def complete_semantic_job(self, jid, claimed_at): self.completed.append(jid)
    async def fail_semantic_job(self, *a, **k): raise AssertionError("no job should fail")
    async def record_tokens(self, n): return None


def _job(i, article):
    return {"id": f"j{i}", "op": "upsert", "article_id": article,
            "claimed_at": "t", "attempts": 0}


class _Ingest:
    def __init__(self, results):
        self._results = results
        self.timings = PromptTimings()

    async def ingest_article(self, article_id):
        return self._results[article_id]


def _result(article, *, basis, same_pair=0):
    d = DedupIndexStats()
    d.calls = 1
    d.contradicted_same_pair = same_pair
    return IngestArticleResult(article_id=article, reference_basis=basis, dedup=d)


_KW = dict(batch=10, budget=10**9, max_attempts=3, backoff_base=1.0,
           backoff_cap=60.0, lease=300.0)


async def test_the_batch_reports_dedup_counters_and_the_basis_mix(caplog):
    store = _Store([_job(1, "a1"), _job(2, "a2")])
    ingest = _Ingest({"a1": _result("a1", basis="exact", same_pair=3),
                      "a2": _result("a2", basis="lower_bound")})
    with caplog.at_level(logging.INFO, logger="graph_sync.semantic_worker"):
        assert await run_worker_once(store, ingest, **_KW) == 2
    text = caplog.text
    assert "contradicted_same_pair=3" in text, "BACKLOG 33's counter must surface"
    assert "exact" in text and "lower_bound" in text


async def test_crawl_fallback_articles_are_called_out_as_untrustworthy(caplog):
    store = _Store([_job(1, "a1")])
    ingest = _Ingest({"a1": _result("a1", basis=CRAWL_FALLBACK)})
    with caplog.at_level(logging.INFO, logger="graph_sync.semantic_worker"):
        await run_worker_once(store, ingest, **_KW)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "ordering by crawl time must not pass silently"
    assert "crawl time" in caplog.text


async def test_per_prompt_timings_reach_the_worker_log(caplog):
    store = _Store([_job(1, "a1")])
    ingest = _Ingest({"a1": _result("a1", basis="exact")})
    at = ingest.timings.start()
    ingest.timings.finish("dedupe_edges.resolve_edge", 1800.0, at)
    with caplog.at_level(logging.INFO, logger="graph_sync.semantic_worker"):
        await run_worker_once(store, ingest, **_KW)
    assert "llm timing by prompt" in caplog.text


async def test_an_empty_batch_logs_nothing(caplog):
    with caplog.at_level(logging.INFO, logger="graph_sync.semantic_worker"):
        assert await run_worker_once(_Store([]), _Ingest({}), **_KW) == 0
    assert caplog.records == [], "an idle worker must not emit a batch summary"


async def test_a_driver_returning_none_does_not_fail_the_job(caplog):
    """Observability must never fail the work it observes. A double -- or a future
    refactor -- returning None must not raise inside the job's try block, which
    would fail the job and poison the queue for the sake of a counter."""
    class _NullIngest:
        timings = PromptTimings()

        async def ingest_article(self, article_id):
            return None

    store = _Store([_job(1, "a1")])
    assert await run_worker_once(store, _NullIngest(), **_KW) == 1
    assert store.completed == ["j1"], "the job must complete, not fail"


async def test_the_batch_reports_prompt_and_cached_tokens(caplog):
    """The prompt cache layout is only worth what the provider actually serves from
    cache: the batch line must say how many prompt tokens were cached, summed over
    the batch's jobs, so production can prove it engaged (CLAUDE.md cost rule)."""
    from graph_extract.usage import record

    class _Spending(_Ingest):
        async def ingest_article(self, article_id):
            record("llm", prompt=100, completion=10, cached=60)
            return self._results[article_id]

    store = _Store([_job(1, "a1"), _job(2, "a2")])
    ingest = _Spending({"a1": _result("a1", basis="exact"), "a2": _result("a2", basis="exact")})
    with caplog.at_level(logging.INFO, logger="graph_sync.semantic_worker"):
        await run_worker_once(store, ingest, **_KW)
    assert "llm tokens: prompt=200 cached=120 (60%) completion=20" in caplog.text
