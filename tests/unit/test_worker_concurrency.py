"""The worker fans out over ARTICLE GROUPS, never raw jobs.

Today the queue cannot produce two jobs for one article in a batch: the
`ux_semantic_jobs_pending` index is UNIQUE on (article_id) WHERE status='pending'
(state_store.py:28-29), `enqueue_semantic_job` collapses an upsert-then-remove into
one row (:129-136), and `claim_semantic_jobs` selects only status='pending'
(:138-149). The grouping is defence in depth that does not depend on that index
staying: without it, fanning out over raw jobs could apply an upsert and a later
remove out of order -- tombstoning episodes that the upsert then recreates, or the
reverse.

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

from graph_extract import usage
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
    """Pins the grouping contract. The pending-unique index means the queue
    cannot currently produce this input; the contract must hold if it ever does."""
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


# --- What escapes a job must not vanish, and must stop the batch -------------
#
# `_run_job` catches Exception, so the only things that can reach a
# run_concurrently result slot are (a) a BaseException such as CancelledError,
# and (b) an Exception raised by the failure path itself -- the store's
# fail_semantic_job or record_tokens, i.e. Postgres being down. A fan-out that
# ignored its results would swallow them: no log, no failed job, no trace.
#
# An escape is broken infrastructure, not a bad article. A bad article is
# per-item and its siblings should carry on (the test above); an escape means
# every further article would be ingested at full LLM cost with nowhere to
# record completion, so the groups not yet started must be skipped. Groups
# already in flight finish. Today's sequential loop aborted on the spot, which
# at concurrency=1 is the same outcome.


class _PgDown(_Store):
    """The failure path itself is broken: fail_semantic_job raises."""

    async def fail_semantic_job(self, jid, *a, **k):
        raise RuntimeError("pg down")


class _Boom(_Ingest):
    """`bad` fails after an await so a concurrent sibling is genuinely in flight
    when the failure path escapes; everything else behaves like `_Ingest`."""

    async def ingest_article(self, article_id):
        if article_id == "bad":
            self.log.append(("upsert-start", article_id))
            await asyncio.sleep(0.01)
            raise ValueError("boom")
        return await super().ingest_article(article_id)


def _attempted(log) -> list[str]:
    return [aid for ev, aid in log if ev == "upsert-start"]


async def test_an_escape_skips_the_groups_not_yet_started():
    """concurrency=1: nothing else is in flight when `bad` escapes, so neither
    of the two later articles may ever be handed to ingest_article."""
    log: list[tuple[str, str]] = []
    store = _PgDown([_job("j1", "bad"), _job("j2", "b"), _job("j3", "c")])
    with pytest.raises(RuntimeError, match="pg down"):
        await run_worker_once(store, _Boom(log), concurrency=1, **_KW)
    assert _attempted(log) == ["bad"], (
        f"after the failure path escaped, no further article may be ingested; got {log}")
    assert store.completed == []


async def test_an_escape_lets_in_flight_groups_finish_and_skips_the_rest(caplog):
    """concurrency=2: `slow` was dispatched alongside `bad` and is mid-flight
    when `bad` escapes, so it finishes and completes; `late` was still waiting
    on the semaphore and is never attempted. A sequential implementation would
    complete nothing, so `completed` is what tells the two apart."""
    log: list[tuple[str, str]] = []
    store = _PgDown([_job("j1", "bad"), _job("j2", "slow"), _job("j3", "late")])
    with caplog.at_level(logging.ERROR, logger="graph_sync.semantic_worker"):
        with pytest.raises(RuntimeError, match="pg down"):
            await run_worker_once(store, _Boom(log, delays={"slow": 0.03}),
                                  concurrency=2, **_KW)
    assert store.completed == ["j2"], (
        f"the group already in flight must finish, the unstarted one must not; got {log}")
    assert _attempted(log) == ["bad", "slow"]
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
    with caplog.at_level(logging.WARNING, logger="graph_sync.semantic_worker"):
        with pytest.raises(asyncio.CancelledError):
            await run_worker_once(store, _Cancels(log), concurrency=4, **_KW)
    assert store.completed == [], "the not-yet-started sibling is skipped, like any escape"
    assert store.failed == [], "CancelledError is not a job failure to be retried"
    named = [r for r in caplog.records if "cancelled" in r.getMessage()]
    assert named, "the cancelled group must be logged naming its article"
    # Only a worker-raised CancelledError reaches the slot loop (an outer
    # cancellation cancels the gather, which re-raises without returning
    # slots). A cancellation is a stop, not a fault: WARNING, no traceback.
    assert [r.levelno for r in named] == [logging.WARNING], (
        f"CancelledError must log at WARNING, not ERROR; got {[r.levelname for r in named]}")
    assert all(r.exc_info is None for r in named), "no traceback for a cancellation"


async def test_a_batch_with_an_escaped_job_still_logs_its_summary(caplog):
    """The groups that ran have finished, so the batch summary (dedup/basis mix
    and the timing report) exists -- raising before logging it would throw it
    away."""
    log: list[tuple[str, str]] = []
    store = _PgDown([_job("j1", "bad"), _job("j2", "good")])
    ingest = _Boom(log)
    at = ingest.timings.start()
    ingest.timings.finish("dedupe_edges.resolve_edge", 1800.0, at)
    with caplog.at_level(logging.INFO, logger="graph_sync.semantic_worker"):
        with pytest.raises(RuntimeError, match="pg down"):
            await run_worker_once(store, ingest, concurrency=4, **_KW)
    assert "semantic batch: jobs=2" in caplog.text, "the batch summary must be emitted"
    assert "llm timing by prompt" in caplog.text, "and so must the timing report"


# --- Each job records ITS OWN tokens, not the batch's -----------------------
#
# `record_tokens` accumulates into today's ledger and `today_token_total() <
# budget` decides whether the bootstrap lane runs at all. A before/after delta
# on the module-global tally includes every concurrent sibling's spend, so with
# N in flight the ledger is inflated and the budget gate fails closed. Harmless
# at concurrency=1; wrong the moment the knob is raised.


class _Ledger(_Store):
    def __init__(self, jobs):
        super().__init__(jobs)
        self.recorded: list[int] = []

    async def record_tokens(self, n): self.recorded.append(n)


class _Spends(_Ingest):
    """Spends through the production path (usage.record) with an await between
    two calls, so two in-flight articles interleave their spend."""

    async def ingest_article(self, article_id):
        usage.record("llm", prompt=100, completion=0)
        await asyncio.sleep(0.01)
        usage.record("llm", prompt=100, completion=0)
        return IngestArticleResult(article_id=article_id)


async def test_concurrent_jobs_record_their_own_tokens_not_the_batch_sum():
    store = _Ledger([_job("j1", "a1"), _job("j2", "a2")])
    await run_worker_once(store, _Spends([]), concurrency=4, **_KW)
    assert sorted(store.recorded) == [200, 200], (
        f"each job spent 200; a global before/after delta would record a sibling's "
        f"tokens too; got {store.recorded}")


async def test_a_failing_job_still_records_its_tokens():
    class _BurnThenBoom(_Ingest):
        async def ingest_article(self, article_id):
            usage.record("llm", prompt=100, completion=0)
            raise ValueError("write timeout after extraction")

    store = _Ledger([_job("j1", "a1")])
    await run_worker_once(store, _BurnThenBoom([]), concurrency=4, **_KW)
    assert store.failed == ["j1"]
    assert store.recorded == [100], (
        f"tokens burned before the failure must reach the ledger; got {store.recorded}")


async def test_the_global_tally_still_accumulates_across_jobs():
    # eval.py and probe.py read the global tally for a whole run; the per-job
    # scope is a second write, not a move.
    usage.reset_tally()
    store = _Ledger([_job("j1", "a1"), _job("j2", "a2")])
    await run_worker_once(store, _Spends([]), concurrency=4, **_KW)
    assert usage.get_tally().prompt_tokens == 400
    assert usage.get_tally().calls == 4


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
