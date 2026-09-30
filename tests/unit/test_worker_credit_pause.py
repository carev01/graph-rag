"""An exhausted LLM account is an account-wide stop, not a bad article.

2026-09-26: OpenRouter ran out of credits and every job failed with
`402 - This request would exceed your available credits`. The worker treated each as
a poison job -- spent one of its attempts and claimed the next, failing it the same
way -- so 155 jobs burned attempts in 15 minutes and the rest would have gone dead.
Now a 402 defers the job (no attempt spent), defers the rest of the batch unrun,
and the worker stops claiming for `credit_pause_seconds`, saying so once at ERROR.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from graph_extract.ingest_driver import IngestArticleResult
from graph_extract.llm_timing import PromptTimings
from graph_sync.semantic_worker import CreditsExhausted, run_worker, run_worker_once

pytestmark = pytest.mark.asyncio


class _PaymentRequired(Exception):
    """Stands in for openai.APIStatusError(status_code=402)."""

    status_code = 402

    def __init__(self):
        super().__init__("Error code: 402 - {'error': {'message': 'This request would "
                         "exceed your available credits given your current in-flight "
                         "requests'}}")


class _Store:
    def __init__(self, batches):
        self._batches = list(batches)
        self.claims = 0
        self.completed: list = []
        self.failed: list = []
        self.deferred: list[tuple] = []

    async def reap_stale_jobs(self, *a, **k):
        return None

    async def today_token_total(self):
        return 0

    async def claim_semantic_jobs(self, batch, include_bootstrap, source_ids=None):
        self.claims += 1
        return self._batches.pop(0) if self._batches else []

    async def complete_semantic_job(self, jid, claimed_at):
        self.completed.append(jid)

    async def fail_semantic_job(self, jid, *a, **k):
        self.failed.append(jid)

    async def defer_semantic_job(self, jid, claimed_at, delay):
        self.deferred.append((jid, delay))
        return True

    async def record_tokens(self, n):
        return None


class _Ingest:
    def __init__(self, fail: dict[str, BaseException]):
        self._fail = fail
        self.ran: list[str] = []
        self.timings = PromptTimings()

    async def ingest_article(self, article_id):
        self.ran.append(article_id)
        if article_id in self._fail:
            raise self._fail[article_id]
        return IngestArticleResult(article_id=article_id)


def _job(i, article):
    return {"id": i, "op": "upsert", "article_id": article, "claimed_at": "t", "attempts": 0}


_KW = dict(batch=10, budget=10**9, max_attempts=5, backoff_base=1.0, backoff_cap=60.0,
           lease=300.0, credit_pause_seconds=300.0)


async def test_a_402_defers_the_job_and_the_rest_of_the_batch_without_spending_attempts():
    store = _Store([[_job(1, "a1"), _job(2, "a2")]])
    ingest = _Ingest({"a1": _PaymentRequired()})

    with pytest.raises(CreditsExhausted):
        await run_worker_once(store, ingest, **_KW)

    assert store.failed == [], "an empty account is not the article's fault"
    assert store.deferred == [(1, 300.0), (2, 300.0)]
    assert ingest.ran == ["a1"], "the rest of the batch must not be sent to a dead account"


async def test_a_402_wrapped_in_another_exception_is_still_recognised():
    try:
        try:
            raise _PaymentRequired()
        except _PaymentRequired as inner:
            raise RuntimeError("graphiti extraction failed") from inner
    except RuntimeError as wrapped:
        err = wrapped
    store = _Store([[_job(1, "a1")]])

    with pytest.raises(CreditsExhausted):
        await run_worker_once(store, _Ingest({"a1": err}), **_KW)
    assert store.failed == [] and store.deferred == [(1, 300.0)]


async def test_any_other_error_still_fails_the_job_as_before():
    store = _Store([[_job(1, "a1"), _job(2, "a2")]])

    n = await run_worker_once(store, _Ingest({"a1": ValueError("bad JSON")}), **_KW)

    assert store.failed == [1] and store.completed == [2] and store.deferred == []
    assert n == 2


async def test_run_worker_pauses_claims_then_resumes(caplog):
    store = _Store([[_job(1, "a1")], [_job(2, "a2")]])
    ingest = _Ingest({"a1": _PaymentRequired()})
    kw = dict(_KW, credit_pause_seconds=0.05)

    with caplog.at_level(logging.ERROR, logger="graph_sync.semantic_worker"):
        started = asyncio.get_running_loop().time()
        await run_worker(store, ingest, poll_seconds=0.01, stop_event=asyncio.Event(),
                         max_batches=2, **kw)
        waited = asyncio.get_running_loop().time() - started

    assert store.claims == 2 and store.completed == [2]
    assert waited >= 0.05, "the worker must stop claiming for the pause"
    assert any("out of credits" in r.getMessage() for r in caplog.records)
