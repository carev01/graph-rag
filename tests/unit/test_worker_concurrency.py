"""The worker fans out over ARTICLE GROUPS, never raw jobs.

`claim_semantic_jobs` has no DISTINCT on article_id (state_store.py:145-147), so
one batch can legitimately hold an upsert and a later remove for the same article.
Fanning out over raw jobs could apply them out of order -- tombstoning episodes
that the upsert then recreates, or the reverse.

The fake ingest logs a START and an END event per upsert. Start-only logging
cannot tell overlap from sequence: a slow article's start always precedes a fast
article's start whether or not they overlap, so an assertion on start order alone
passes under every implementation. The interleaving of starts with ends is what
actually distinguishes the three cases (ordered, overlapping, sequential).
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from graph_extract.ingest_driver import IngestArticleResult
from graph_extract.llm_timing import PromptTimings
from graph_sync import semantic_worker
from graph_sync.semantic_worker import run_worker, run_worker_once

pytestmark = pytest.mark.asyncio


class _Store:
    def __init__(self, jobs):
        self._jobs = jobs
        self.completed: list[str] = []
        self.failed: list[str] = []

    async def reap_stale_jobs(self, *a, **k): return None
    async def today_token_total(self): return 0
    async def claim_semantic_jobs(self, batch, include_bootstrap): return self._jobs
    async def complete_semantic_job(self, jid, claimed_at): self.completed.append(jid)
    async def fail_semantic_job(self, jid, *a, **k): self.failed.append(jid)
    async def record_tokens(self, n): return None


def _job(jid, article, op="upsert"):
    return {"id": jid, "op": op, "article_id": article,
            "claimed_at": "t", "attempts": 0}


class _Ingest:
    def __init__(self, log, delays=None):
        self.log = log
        self.delays = delays or {}
        self.timings = PromptTimings()

    async def ingest_article(self, article_id):
        self.log.append(("upsert-start", article_id))
        await asyncio.sleep(self.delays.get(article_id, 0))
        self.log.append(("upsert-end", article_id))
        return IngestArticleResult(article_id=article_id)

    async def tombstone_article_episodes(self, article_id):
        self.log.append(("remove", article_id))
        return 1


_KW = dict(batch=10, budget=10**9, max_attempts=3, backoff_base=1.0,
           backoff_cap=60.0, lease=300.0)


async def test_two_jobs_for_one_article_stay_in_order():
    # The upsert is slow so that a raw-job fan-out would run the remove while
    # the upsert is still in flight -- exactly the tombstone-before-episodes
    # hazard. Grouping must hold the remove until the upsert has ENDED.
    log: list[tuple[str, str]] = []
    store = _Store([_job("j1", "a1", "upsert"), _job("j2", "a1", "remove")])
    await run_worker_once(store, _Ingest(log, delays={"a1": 0.02}),
                          concurrency=4, **_KW)
    assert log == [("upsert-start", "a1"), ("upsert-end", "a1"), ("remove", "a1")]
    assert store.completed == ["j1", "j2"]


async def test_different_articles_overlap():
    log: list[tuple[str, str]] = []
    store = _Store([_job("j1", "slow"), _job("j2", "fast")])
    ingest = _Ingest(log, delays={"slow": 0.05})
    await run_worker_once(store, ingest, concurrency=4, **_KW)
    assert log.index(("upsert-start", "fast")) < log.index(("upsert-end", "slow")), (
        f"the fast article must start before the slow one finishes; got {log}")
    assert sorted(store.completed) == ["j1", "j2"]


async def test_default_concurrency_is_one():
    log: list[tuple[str, str]] = []
    store = _Store([_job("j1", "slow"), _job("j2", "fast")])
    ingest = _Ingest(log, delays={"slow": 0.03})
    await run_worker_once(store, ingest, **_KW)
    assert log == [("upsert-start", "slow"), ("upsert-end", "slow"),
                   ("upsert-start", "fast"), ("upsert-end", "fast")], (
        "without an explicit concurrency the worker must behave like today's loop")
    assert store.completed == ["j1", "j2"]


async def test_one_failing_group_does_not_stop_another():
    log: list[tuple[str, str]] = []

    class _Boom(_Ingest):
        async def ingest_article(self, article_id):
            if article_id == "bad":
                raise ValueError("boom")
            return await super().ingest_article(article_id)

    store = _Store([_job("j1", "bad"), _job("j2", "good")])
    await run_worker_once(store, _Boom(log), concurrency=4, **_KW)
    assert store.failed == ["j1"]
    assert store.completed == ["j2"]


# --- What escapes a job must not vanish into a discarded result slot ---------
#
# `_run_job` catches Exception, so the only things that can reach a
# run_concurrently result slot are (a) a BaseException such as CancelledError,
# and (b) an Exception raised by the failure path itself -- the store's
# fail_semantic_job or record_tokens, i.e. Postgres being down. Today's loop
# propagates both out of run_worker_once. A fan-out that ignores its results
# would swallow them: no log, no failed job, no trace.


async def test_a_store_failure_in_the_failure_path_escapes(caplog):
    log: list[tuple[str, str]] = []

    class _PgDown(_Store):
        async def fail_semantic_job(self, jid, *a, **k):
            raise RuntimeError("pg down")

    class _Boom(_Ingest):
        async def ingest_article(self, article_id):
            if article_id == "bad":
                raise ValueError("boom")
            return await super().ingest_article(article_id)

    store = _PgDown([_job("j1", "bad"), _job("j2", "good")])
    with caplog.at_level(logging.ERROR, logger="graph_sync.semantic_worker"):
        with pytest.raises(RuntimeError, match="pg down"):
            await run_worker_once(store, _Boom(log), concurrency=4, **_KW)
    assert store.completed == ["j2"], "the sibling group still runs to completion"
    assert any("bad" in r.getMessage() for r in caplog.records
               if r.levelno >= logging.ERROR), (
        "the escaped error must be logged naming its article group")


async def test_a_cancelled_job_does_not_vanish(caplog):
    log: list[tuple[str, str]] = []

    class _Cancels(_Ingest):
        async def ingest_article(self, article_id):
            if article_id == "cancelled":
                raise asyncio.CancelledError()
            return await super().ingest_article(article_id)

    store = _Store([_job("j1", "cancelled"), _job("j2", "good")])
    with caplog.at_level(logging.ERROR, logger="graph_sync.semantic_worker"):
        with pytest.raises(asyncio.CancelledError):
            await run_worker_once(store, _Cancels(log), concurrency=4, **_KW)
    assert store.completed == ["j2"]
    assert store.failed == [], "CancelledError is not a job failure to be retried"
    assert any("cancelled" in r.getMessage() for r in caplog.records
               if r.levelno >= logging.ERROR)


# --- The knob must reach run_worker_once through run_worker -----------------
#
# `run_worker` is the production loop; `run_worker_once` is what fans out. A
# `concurrency` that `run_worker` accepts but does not forward leaves the
# fan-out unreachable from the cli while every unit test above still passes.
# So this asserts the value `run_worker_once` RECEIVES, not the signature.


async def test_run_worker_forwards_concurrency_to_run_worker_once(monkeypatch):
    received: dict = {}

    async def _fake_once(store, ingest, **kw):
        received.update(kw)
        return 0

    monkeypatch.setattr(semantic_worker, "run_worker_once", _fake_once)
    await run_worker(
        _Store([]), None, poll_seconds=0.01, stop_event=asyncio.Event(),
        max_batches=1, concurrency=7, **_KW)
    assert received.get("concurrency") == 7, (
        f"run_worker_once must observe the concurrency run_worker was given; got {received}")
