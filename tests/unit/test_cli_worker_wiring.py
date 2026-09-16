"""The `worker` cli command passes `ExtractSettings.ingest_article_concurrency`
and a warm-up predicate built from `ExtractSettings.ingest_warmup_articles` to
`run_worker`. Every constructor is stubbed; nothing connects.

Both knobs live on `ExtractSettings` (the object `_build_worker_deps` gives the
driver), not on graph_sync's `Settings`, so the cli reads them from
`get_extract_settings()`. Distinctive values are used: `concurrency` defaults
to 1 and the warm-up threshold to 8, so a test built on the default would pass
with the argument missing or the value hard-coded.
"""
from __future__ import annotations

import pytest

from graph_extract.config import ExtractSettings
from graph_extract.warmup import WarmupGate
from graph_sync import cli
from graph_sync.config import Settings


def _sync_settings() -> Settings:
    return Settings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                    neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                    postgres_dsn="postgresql://x")


def _extract_settings(**kw) -> ExtractSettings:
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                cheap_llm_api_key="or-key")
    base.update(kw)
    return ExtractSettings(**base)


class _Closable:
    async def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _Ingest(_Closable):
    _cheap = None


def test_worker_command_passes_ingest_article_concurrency_to_run_worker(monkeypatch):
    received: dict = {}

    async def _deps(settings):
        return _Closable(), _Ingest(), _Closable(), _Closable(), _Closable()

    async def _fake_run_worker(store, ingest, **kw):
        received.update(kw)

    monkeypatch.setattr(cli, "get_settings", _sync_settings)
    monkeypatch.setattr(cli, "get_extract_settings",
                        lambda: _extract_settings(ingest_article_concurrency=7))
    monkeypatch.setattr(cli, "_build_worker_deps", _deps)
    monkeypatch.setattr(cli, "run_worker", _fake_run_worker)

    cli.worker(batch=3, poll_seconds=0.01, max_batches=1)

    assert received.get("concurrency") == 7, (
        f"run_worker must receive ExtractSettings.ingest_article_concurrency; got {received}")
    assert received.get("max_batches") == 1


@pytest.mark.parametrize("threshold", [5, 0])
def test_worker_command_passes_a_warmup_predicate_iff_the_threshold_is_positive(
        monkeypatch, threshold):
    """`ingest_warmup_articles > 0` -> `is_cold` is `WarmupGate.is_cold_article`,
    bound to a gate built on THIS threshold, THIS group and the neo4j driver
    `_build_worker_deps` returned. `0` (off) -> `is_cold=None`, so the fan-out
    takes the byte-for-byte pre-warm-up path rather than a predicate that
    always answers False."""
    received: dict = {}
    neo4j_driver = _Closable()

    async def _deps(settings):
        return _Closable(), _Ingest(), _Closable(), _Closable(), neo4j_driver

    async def _fake_run_worker(store, ingest, **kw):
        received.update(kw)

    monkeypatch.setattr(cli, "get_settings", _sync_settings)
    monkeypatch.setattr(cli, "get_extract_settings",
                        lambda: _extract_settings(ingest_warmup_articles=threshold,
                                                  group_id="g-wiring"))
    monkeypatch.setattr(cli, "_build_worker_deps", _deps)
    monkeypatch.setattr(cli, "run_worker", _fake_run_worker)

    cli.worker(batch=3, poll_seconds=0.01, max_batches=1)

    assert "is_cold" in received, f"run_worker must receive is_cold; got {received}"
    is_cold = received["is_cold"]
    if threshold == 0:
        assert is_cold is None, "threshold 0 is OFF: no predicate, not an always-False one"
        return
    assert is_cold is not None
    assert is_cold.__func__ is WarmupGate.is_cold_article, "the worker path is per ARTICLE"
    gate = is_cold.__self__
    assert gate._threshold == threshold, "the knob must reach the gate, not a hard-coded 8"
    assert gate._group_id == "g-wiring"
    assert gate._driver is neo4j_driver, "the gate counts on the neo4j driver, not the store"


class _Store(_Closable):
    """Records how the cli asks for the warm-up lock."""

    def __init__(self):
        self.warmup_lock_calls: list[float] = []

    def warmup_lock(self, *, timeout: float):
        self.warmup_lock_calls.append(timeout)
        return object()


@pytest.mark.parametrize(
    "warmup,enabled,expect_lock",
    [(5, True, True),      # on: the multi-worker bootstrap case
     (5, False, False),    # explicitly disabled
     (0, True, False)],    # no warm-up -> no cold articles to serialise
)
def test_worker_command_wires_the_global_warmup_lock(monkeypatch, warmup, enabled,
                                                     expect_lock):
    """The knob must reach `run_worker`, not merely exist. A lock that is
    configured, constructed and never passed is the inert-knob defect this
    project has now shipped twice; asserting on `Settings` alone would not catch
    it, so this calls the factory the cli built and checks it reaches the store
    with the configured timeout (1234.0, never the 1800.0 default)."""
    received: dict = {}
    store = _Store()

    async def _deps(settings):
        return store, _Ingest(), _Closable(), _Closable(), _Closable()

    async def _fake_run_worker(store_, ingest, **kw):
        received.update(kw)

    monkeypatch.setattr(cli, "get_settings",
                        lambda: _sync_settings().model_copy(update={
                            "semantic_global_warmup_lock": enabled,
                            "semantic_warmup_lock_timeout_seconds": 1234.0}))
    monkeypatch.setattr(cli, "get_extract_settings",
                        lambda: _extract_settings(ingest_warmup_articles=warmup))
    monkeypatch.setattr(cli, "_build_worker_deps", _deps)
    monkeypatch.setattr(cli, "run_worker", _fake_run_worker)

    cli.worker(batch=3, poll_seconds=0.01, max_batches=1)

    assert "cold_lock" in received, f"run_worker must receive cold_lock; got {received}"
    cold_lock = received["cold_lock"]
    if not expect_lock:
        assert cold_lock is None
        assert store.warmup_lock_calls == []
        return
    assert cold_lock is not None
    cold_lock()
    assert store.warmup_lock_calls == [1234.0], (
        f"the factory must call the STORE's warmup_lock with the configured "
        f"timeout; got {store.warmup_lock_calls}")
