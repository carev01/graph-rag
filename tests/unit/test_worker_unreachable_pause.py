"""An unreachable upstream endpoint is an outage, not a bad article.

2026-10-08: vast.ai's shared SSH proxy went away for ~10 minutes and took the GPU tier's
tunnel with it. Every GPU worker failed each article it claimed with
`openai.APIConnectionError` -- spending one of the job's attempts -- and claimed the
next: 24 jobs in a few minutes, and a long outage would march the queue to `dead`.

Now a failure to CONNECT defers the job (no attempt spent), defers the rest of the batch
unrun, and the worker stops claiming for `unreachable_pause_seconds`. A request DROPPED
mid-flight is ambiguous -- a proxy can cut off one very long request -- so it is an
outage only when a probe of the same endpoint fails too; otherwise the job fails as
before and can still reach `dead`. A read timeout always fails the job.
"""
from __future__ import annotations

import asyncio
import logging

import httpx
import openai
import pytest

from graph_sync.semantic_worker import (
    CreditsExhausted,
    ProviderUnreachable,
    endpoint_answers,
    run_worker,
    run_worker_once,
)
from tests.unit.test_worker_credit_pause import _KW, _Ingest, _job, _PaymentRequired, _Store

pytestmark = pytest.mark.asyncio

_REQ = httpx.Request("POST", "http://graph-rag-vast-tunnel:8000/v1/chat/completions")
PAUSE = 42.0   # distinctive: a dropped kwarg must not hide behind the default


class _Probe:
    def __init__(self, answers: bool):
        self.answers = answers
        self.urls: list[httpx.URL] = []

    async def __call__(self, url: httpx.URL) -> bool:
        self.urls.append(url)
        return self.answers


def _kw(probe: _Probe | None = None, **over):
    return dict(_KW, unreachable_pause_seconds=PAUSE, probe=probe or _Probe(False), **over)


def _wrapped(inner: BaseException, outer: type[BaseException] | None = None) -> BaseException:
    """`inner` raised inside graphiti/openai, re-raised as `outer` FROM it."""
    try:
        try:
            raise inner
        except BaseException as e:
            if outer is None:
                raise
            if issubclass(outer, openai.APIConnectionError):
                raise outer(request=_REQ) from e
            raise outer("graphiti extraction failed") from e
    except BaseException as e:
        return e


def _connect_error():
    return _wrapped(httpx.ConnectError("All connection attempts failed", request=_REQ),
                    openai.APIConnectionError)


def _dropped():
    return _wrapped(httpx.RemoteProtocolError("Server disconnected without sending a "
                                              "response.", request=_REQ),
                    openai.APIConnectionError)


async def test_a_failed_connect_defers_the_job_and_the_rest_of_the_batch():
    store = _Store([[_job(1, "a1"), _job(2, "a2")]])
    ingest = _Ingest({"a1": _connect_error()})
    probe = _Probe(True)

    with pytest.raises(ProviderUnreachable):
        await run_worker_once(store, ingest, **_kw(probe))

    assert store.failed == [], "an unreachable endpoint is not the article's fault"
    assert store.deferred == [(1, PAUSE), (2, PAUSE)]
    assert ingest.ran == ["a1"], "the rest of the batch must not be sent to a dead endpoint"
    assert probe.urls == [], "a failed connect needs no probe"


async def test_a_connect_error_wrapped_by_graphiti_is_still_recognised():
    err = _wrapped(httpx.ConnectError("connection refused", request=_REQ), RuntimeError)
    store = _Store([[_job(1, "a1")]])

    with pytest.raises(ProviderUnreachable):
        await run_worker_once(store, _Ingest({"a1": err}), **_kw())
    assert store.failed == [] and store.deferred == [(1, PAUSE)]


async def test_a_connect_error_reached_only_through_context_is_recognised():
    try:
        try:
            raise ConnectionRefusedError(111, "Connection refused")
        except ConnectionRefusedError:
            raise RuntimeError("while handling it, something else broke")  # noqa: B904
    except RuntimeError as e:
        err = e
    assert err.__cause__ is None and err.__context__ is not None
    store = _Store([[_job(1, "a1")]])

    with pytest.raises(ProviderUnreachable):
        await run_worker_once(store, _Ingest({"a1": err}), **_kw())


async def test_a_connect_timeout_is_an_outage_not_a_slow_article():
    err = _wrapped(httpx.ConnectTimeout("timed out connecting", request=_REQ),
                   openai.APITimeoutError)
    store = _Store([[_job(1, "a1")]])

    with pytest.raises(ProviderUnreachable):
        await run_worker_once(store, _Ingest({"a1": err}), **_kw())
    assert store.deferred == [(1, PAUSE)]


async def test_a_dropped_request_with_the_endpoint_down_is_an_outage():
    store = _Store([[_job(1, "a1")]])
    probe = _Probe(False)

    with pytest.raises(ProviderUnreachable):
        await run_worker_once(store, _Ingest({"a1": _dropped()}), **_kw(probe))
    assert store.deferred == [(1, PAUSE)]
    assert probe.urls == [_REQ.url], "the probe must hit the endpoint that failed"


async def test_a_dropped_request_with_the_endpoint_up_fails_the_job():
    """A proxy cutting off one very long request: the article's own doing. Deferring it
    would loop forever without ever spending an attempt."""
    store = _Store([[_job(1, "a1"), _job(2, "a2")]])

    n = await run_worker_once(store, _Ingest({"a1": _dropped()}), **_kw(_Probe(True)))

    assert store.failed == [1] and store.completed == [2] and store.deferred == []
    assert n == 2


async def test_a_read_timeout_still_fails_the_job():
    err = _wrapped(httpx.ReadTimeout("timed out reading", request=_REQ),
                   openai.APITimeoutError)
    store = _Store([[_job(1, "a1"), _job(2, "a2")]])
    probe = _Probe(False)

    n = await run_worker_once(store, _Ingest({"a1": err}), **_kw(probe))

    assert store.failed == [1] and store.completed == [2] and store.deferred == []
    assert n == 2 and probe.urls == []


class _InFlightIngest(_Ingest):
    """Yields before failing, so both articles of a concurrent batch are in flight."""

    async def ingest_article(self, article_id):
        await asyncio.sleep(0)
        return await super().ingest_article(article_id)


async def test_out_of_credits_wins_over_unreachable_in_one_batch():
    store = _Store([[_job(1, "a1"), _job(2, "a2")]])
    ingest = _InFlightIngest({"a1": _connect_error(), "a2": _PaymentRequired()})

    with pytest.raises(CreditsExhausted):
        await run_worker_once(store, ingest, **_kw(concurrency=2))

    assert ingest.ran == ["a1", "a2"], "both must have been in flight"
    assert store.failed == []
    assert sorted(j for j, _ in store.deferred) == [1, 2]
    assert {d for _, d in store.deferred} <= {PAUSE, 300.0}


async def test_run_worker_pauses_claims_then_resumes(caplog):
    store = _Store([[_job(1, "a1")], [_job(2, "a2")]])
    ingest = _Ingest({"a1": _connect_error()})
    kw = dict(_KW, unreachable_pause_seconds=0.05)

    with caplog.at_level(logging.ERROR, logger="graph_sync.semantic_worker"):
        started = asyncio.get_running_loop().time()
        await run_worker(store, ingest, poll_seconds=0.01, stop_event=asyncio.Event(),
                         max_batches=2, **kw)
        waited = asyncio.get_running_loop().time() - started

    assert store.claims == 2 and store.completed == [2] and store.failed == []
    assert waited >= 0.05, "the worker must stop claiming for the pause"
    assert any("unreachable" in r.getMessage() for r in caplog.records)


def _transport(respond):
    return httpx.MockTransport(respond)


def _status(code):
    return _transport(lambda req: httpx.Response(code))


def _raising(exc_type):
    def respond(req):
        raise exc_type("boom", request=req)
    return _transport(respond)


@pytest.mark.parametrize("code", [200, 401, 404, 500])
async def test_probe_any_response_means_up(code):
    assert await endpoint_answers(_REQ.url, transport=_status(code)) is True


@pytest.mark.parametrize("code", [502, 503, 504])
async def test_probe_a_gateway_error_means_down(code):
    assert await endpoint_answers(_REQ.url, transport=_status(code)) is False


@pytest.mark.parametrize("exc", [httpx.ConnectError, httpx.ReadError,
                                 httpx.RemoteProtocolError, httpx.ConnectTimeout])
async def test_probe_a_transport_failure_means_down(exc):
    assert await endpoint_answers(_REQ.url, transport=_raising(exc)) is False


async def test_probe_something_unexpected_means_up_so_the_job_fails_normally():
    def respond(req):
        raise ValueError("not a transport error")
    assert await endpoint_answers(_REQ.url, transport=_transport(respond)) is True


async def test_probe_hits_the_origin_without_verifying_tls(monkeypatch):
    seen: dict = {}
    real = httpx.AsyncClient

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", spy)
    urls: list[str] = []

    def respond(req):
        urls.append(str(req.url))
        return httpx.Response(200)

    url = httpx.URL("https://docextractor.k3s.home.lan/api/articles/x?y=1")
    assert await endpoint_answers(url, transport=_transport(respond)) is True
    assert seen["verify"] is False, "the private CA must not make a live endpoint look down"
    assert urls == ["https://docextractor.k3s.home.lan/"]
