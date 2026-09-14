"""The `worker` cli command passes `ExtractSettings.ingest_article_concurrency`
to `run_worker`. Every constructor is stubbed; nothing connects.

The knob lives on `ExtractSettings` (the object `_build_worker_deps` gives the
driver), not on graph_sync's `Settings`, so the cli reads it from
`get_extract_settings()`. A distinctive value is used: the parameter defaults
to 1, so a test built on 1 would pass with the argument missing altogether.
"""
from __future__ import annotations

from graph_extract.config import ExtractSettings
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
