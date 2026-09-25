"""The warm-up barrier serialises cold articles across worker PROCESSES.

`run_concurrently`'s barrier only serialises within one process. That is enough
for a single worker, and it is also -- by accident, not design -- what protects
cross-source hub entities: while any source is warming up, nothing else runs
anywhere. Scale to N workers and that protection disappears exactly where it
costs most (measured: entities in the warm-up window of 2+ sources carry 20.7%
of all duplicate risk weight, and they are the graph's biggest hubs).

`cold_lock` restores it. These tests pin WHICH articles take it, that it is held
for the whole article rather than just acquired, and that failing to take it
stops the batch instead of quietly proceeding unprotected.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import pytest

from graph_extract.ingest_driver import IngestArticleResult
from graph_extract.llm_timing import PromptTimings
from graph_sync.semantic_worker import run_worker_once

pytestmark = pytest.mark.asyncio


class _Store:
    def __init__(self, jobs):
        self._jobs = jobs
        self.completed: list[str] = []
        self.failed: list[str] = []
        self.deferred: list[tuple] = []

    async def reap_stale_jobs(self, *a, **k): return None
    async def today_token_total(self): return 0
    async def claim_semantic_jobs(self, batch, include_bootstrap, source_ids=None): return self._jobs
    async def complete_semantic_job(self, jid, claimed_at): self.completed.append(jid)
    async def fail_semantic_job(self, jid, *a, **k): self.failed.append(jid)
    async def record_tokens(self, n): return None

    async def defer_semantic_job(self, jid, claimed_at, delay):
        self.deferred.append((jid, delay))
        return True


class _Ingest:
    def __init__(self, log):
        self.log = log
        self.timings = PromptTimings()

    async def ingest_article(self, article_id):
        self.log.append(f"start-{article_id}")
        await asyncio.sleep(0.01)
        self.log.append(f"end-{article_id}")
        return IngestArticleResult(article_id=article_id)

    async def tombstone_article_episodes(self, article_id):
        self.log.append(f"remove-{article_id}")
        return 1


def _job(jid, article, op="upsert"):
    return {"id": jid, "op": op, "article_id": article, "claimed_at": "t", "attempts": 0}


_KW = dict(batch=10, budget=10**9, max_attempts=3, backoff_base=1.0,
           backoff_cap=60.0, lease=300.0)


def _lock_factory(log, *, fail: bool = False, busy: bool = False):
    """A stand-in for StateStore.warmup_lock that records acquire/release."""
    calls: list[str] = []

    def factory():
        @asynccontextmanager
        async def _cm():
            if fail:
                raise ConnectionError("postgres down")
            if busy:
                log.append("BUSY")
                yield False
                return
            log.append("LOCK")
            try:
                yield True
            finally:
                log.append("UNLOCK")
        return _cm()

    factory.calls = calls  # type: ignore[attr-defined]
    return factory


async def _run(jobs, *, cold_articles: set[str], log, cold_lock, concurrency=4,
               store=None, defer_seconds=None):
    async def is_cold(article_id: str) -> bool:
        return article_id in cold_articles

    return await run_worker_once(
        store if store is not None else _Store(jobs), _Ingest(log), **_KW,
        concurrency=concurrency, is_cold=is_cold, cold_lock=cold_lock,
        defer_seconds=defer_seconds)


async def test_only_cold_articles_take_the_lock():
    """A warm article must not pay a global serialisation it does not need --
    that would make the lock, not the barrier, the bottleneck for the 98% of
    articles that are warm."""
    log: list[str] = []
    await _run([_job(1, "a1"), _job(2, "a2"), _job(3, "a3")],
               cold_articles={"a2"}, log=log, cold_lock=_lock_factory(log))
    assert log.count("LOCK") == 1, f"exactly one article was cold; got {log}"
    lock_at = log.index("LOCK")
    assert log[lock_at + 1] == "start-a2", (
        f"the lock must wrap the COLD article, not whichever ran next; got {log}")


async def test_taking_the_lock_is_logged_per_cold_article(caplog):
    """The rehearsal could only prove the lock engaged by reconstructing episode
    timestamps from the graph. CLAUDE.md's cost rule asks for direct evidence that
    a mechanism fired, so each acquisition leaves one INFO line naming the article
    and how long it waited -- and a warm article leaves none."""
    log: list[str] = []
    with caplog.at_level(logging.INFO, logger="graph_sync.semantic_worker"):
        await _run([_job(1, "a1"), _job(2, "a2"), _job(3, "a3")],
                   cold_articles={"a2"}, log=log, cold_lock=_lock_factory(log))
    held = [r.getMessage() for r in caplog.records
            if r.getMessage().startswith("warm-up lock held")]
    assert len(held) == 1, held
    assert "a2" in held[0] and "waited" in held[0], held


async def test_the_lock_is_held_for_the_whole_article_not_just_acquired():
    """Acquiring and releasing around the dispatch would serialise nothing: the
    point is that no other process runs a cold article while this one extracts."""
    log: list[str] = []
    await _run([_job(1, "a1")], cold_articles={"a1"}, log=log,
               cold_lock=_lock_factory(log))
    assert log == ["LOCK", "start-a1", "end-a1", "UNLOCK"], log


async def test_without_a_lock_the_worker_is_unchanged():
    """cold_lock=None must be the byte-for-byte pre-lock path, so the default
    single-worker deployment is unaffected."""
    log: list[str] = []
    n = await _run([_job(1, "a1"), _job(2, "a2")], cold_articles={"a1"},
                   log=log, cold_lock=None)
    assert n == 2
    assert "LOCK" not in log and "UNLOCK" not in log
    assert sorted(x for x in log if x.startswith("start-")) == ["start-a1", "start-a2"]


async def test_a_predicate_failure_answers_cold_and_still_takes_the_lock():
    """The predicate failing (Neo4j unreachable) is answered COLD -- conservative.
    That conservatism is worth nothing if the article then runs unprotected while
    another process runs its own cold article: the fallback must take the lock
    like any other cold article."""
    log: list[str] = []

    async def is_cold(article_id: str) -> bool:
        raise ConnectionError("neo4j unreachable")

    await run_worker_once(
        _Store([_job(1, "a1")]), _Ingest(log), **_KW, concurrency=4,
        is_cold=is_cold, cold_lock=_lock_factory(log))
    assert log == ["LOCK", "start-a1", "end-a1", "UNLOCK"], (
        f"the cold fallback must be protected too; got {log}")


async def test_failing_to_take_the_lock_stops_the_batch():
    """Postgres being down is broken infrastructure, not a bad article. Carrying
    on would ingest every remaining article at full LLM cost with nowhere to
    record completion -- and, worse here, would do it with the protection the
    lock exists to provide silently absent."""
    log: list[str] = []
    with pytest.raises(ConnectionError, match="postgres down"):
        await _run([_job(1, "a1"), _job(2, "a2")], cold_articles={"a1", "a2"},
                   log=log, cold_lock=_lock_factory(log, fail=True),
                   concurrency=1)
    assert not any(x.startswith("start-") for x in log), (
        f"no article may be ingested once the lock is unavailable; got {log}")


async def test_a_busy_lock_defers_the_cold_article_and_the_warm_one_still_runs(caplog):
    """Defer mode: a cold article whose global warm-up lock is held elsewhere is
    handed back (no attempt spent) instead of idling the worker; a warm article
    in the same batch runs. Only the warm one counts as processed, so a batch
    that did nothing but defer returns 0 and run_worker sleeps its poll interval
    before claiming again -- no busy loop over an all-cold queue."""
    log: list[str] = []
    store = _Store([_job(1, "warm"), _job(2, "cold")])
    with caplog.at_level(logging.INFO, logger="graph_sync.semantic_worker"):
        n = await _run(store._jobs, cold_articles={"cold"}, log=log,
                       cold_lock=_lock_factory(log, busy=True), store=store,
                       defer_seconds=30.0)
    assert store.deferred == [(2, 30.0)]
    assert "start-cold" not in log and "start-warm" in log
    assert store.completed == [1] and store.failed == []
    assert n == 1
    assert any("deferred" in r.getMessage() and "cold" in r.getMessage()
               for r in caplog.records), "the first deferral is visible at INFO"


async def test_a_deferral_only_batch_reports_zero_processed():
    log: list[str] = []
    store = _Store([_job(1, "cold")])
    n = await _run(store._jobs, cold_articles={"cold"}, log=log,
                   cold_lock=_lock_factory(log, busy=True), store=store, defer_seconds=30.0)
    assert n == 0 and store.deferred == [(1, 30.0)]


async def test_wait_mode_still_runs_a_cold_article_without_the_lock():
    """defer_seconds=None is the old contract: after warmup_lock gives up (it
    yields False on timeout) the article runs anyway."""
    log: list[str] = []
    store = _Store([_job(1, "cold")])
    n = await _run(store._jobs, cold_articles={"cold"}, log=log,
                   cold_lock=_lock_factory(log, busy=True), store=store, defer_seconds=None)
    assert "start-cold" in log and store.deferred == [] and n == 1
