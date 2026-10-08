"""An unreachable LLM provider is an outage, not a bad article.

2026-10-08: vast.ai's shared SSH proxy went away for ~10 minutes and took the GPU tier's
tunnel with it. Every GPU worker failed each article it claimed with
`openai.APIConnectionError` -- spending one of the job's attempts -- and claimed the
next: 24 jobs in a few minutes, and a long outage would march the queue to `dead`.
Now a connection failure defers the job (no attempt spent), defers the rest of the batch
unrun, and the worker stops claiming for `unreachable_pause_seconds`, saying so once at
ERROR. A TIMEOUT still fails the job: it can be the article (a huge prompt), and
`APITimeoutError` subclasses `APIConnectionError`, so it must be told apart explicitly.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
import openai
import pytest

from graph_sync.semantic_worker import ProviderUnreachable, run_worker, run_worker_once
from tests.unit.test_worker_credit_pause import _KW, _Ingest, _job, _PaymentRequired, _Store

pytestmark = pytest.mark.asyncio

_REQ = httpx.Request("POST", "http://graph-rag-vast-tunnel:8000/v1/chat/completions")
_KWU = dict(_KW, unreachable_pause_seconds=60.0)


def _conn_error() -> openai.APIConnectionError:
    try:
        try:
            raise httpx.ConnectError("All connection attempts failed", request=_REQ)
        except httpx.ConnectError as inner:
            raise openai.APIConnectionError(request=_REQ) from inner
    except openai.APIConnectionError as e:
        return e


async def test_a_connection_error_defers_the_job_and_the_rest_of_the_batch():
    store = _Store([[_job(1, "a1"), _job(2, "a2")]])
    ingest = _Ingest({"a1": _conn_error()})

    with pytest.raises(ProviderUnreachable):
        await run_worker_once(store, ingest, **_KWU)

    assert store.failed == [], "an unreachable provider is not the article's fault"
    assert store.deferred == [(1, 60.0), (2, 60.0)]
    assert ingest.ran == ["a1"], "the rest of the batch must not be sent to a dead endpoint"


async def test_a_connect_error_wrapped_by_graphiti_is_still_recognised():
    try:
        try:
            raise httpx.ConnectError("connection refused", request=_REQ)
        except httpx.ConnectError as inner:
            raise RuntimeError("graphiti extraction failed") from inner
    except RuntimeError as wrapped:
        err = wrapped
    store = _Store([[_job(1, "a1")]])

    with pytest.raises(ProviderUnreachable):
        await run_worker_once(store, _Ingest({"a1": err}), **_KWU)
    assert store.failed == [] and store.deferred == [(1, 60.0)]


async def test_a_dropped_connection_mid_request_counts_as_unreachable():
    err = httpx.RemoteProtocolError("Server disconnected without sending a response.",
                                    request=_REQ)
    store = _Store([[_job(1, "a1")]])

    with pytest.raises(ProviderUnreachable):
        await run_worker_once(store, _Ingest({"a1": err}), **_KWU)
    assert store.deferred == [(1, 60.0)]


async def test_a_timeout_still_fails_the_job():
    store = _Store([[_job(1, "a1"), _job(2, "a2")]])

    n = await run_worker_once(store, _Ingest({"a1": openai.APITimeoutError(request=_REQ)}),
                              **_KWU)

    assert store.failed == [1] and store.completed == [2] and store.deferred == []
    assert n == 2


async def test_out_of_credits_keeps_its_own_pause():
    store = _Store([[_job(1, "a1"), _job(2, "a2")]])

    from graph_sync.semantic_worker import CreditsExhausted
    with pytest.raises(CreditsExhausted):
        await run_worker_once(store, _Ingest({"a1": _PaymentRequired()}), **_KWU)
    assert store.deferred == [(1, 300.0), (2, 300.0)]


async def test_run_worker_pauses_claims_then_resumes(caplog):
    store = _Store([[_job(1, "a1")], [_job(2, "a2")]])
    ingest = _Ingest({"a1": _conn_error()})
    kw = dict(_KWU, unreachable_pause_seconds=0.05)

    with caplog.at_level(logging.ERROR, logger="graph_sync.semantic_worker"):
        started = asyncio.get_running_loop().time()
        await run_worker(store, ingest, poll_seconds=0.01, stop_event=asyncio.Event(),
                         max_batches=2, **kw)
        waited = asyncio.get_running_loop().time() - started

    assert store.claims == 2 and store.completed == [2] and store.failed == []
    assert waited >= 0.05, "the worker must stop claiming for the pause"
    assert any("unreachable" in r.getMessage() for r in caplog.records)
