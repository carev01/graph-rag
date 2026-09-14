"""`_build_ingest_driver` (the `ingest` cli command's constructor) hands the
driver a `WarmupGate` built from `ExtractSettings.ingest_warmup_articles`, and
no gate at all when the knob is 0. Every constructor is stubbed; nothing connects.

Companion to test_cli_worker_wiring: that file pins the WORKER path's predicate,
this one pins the INGEST path's gate. Without it the knob could be read by the
worker and silently ignored by `ingest_source`. A distinctive threshold is used:
the default is 8, so a test built on 8 would pass with the value hard-coded.
"""
from __future__ import annotations

import pytest

from graph_extract import cli
from graph_extract.config import ExtractSettings
from graph_extract.warmup import WarmupGate


def _settings(**kw) -> ExtractSettings:
    # No cheap_llm_api_key (default ""), so the cheap tier is never built.
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    base.update(kw)
    return ExtractSettings(**base)


class _Closable:
    async def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _Guard:
    raw = None


@pytest.mark.parametrize("threshold", [5, 0])
async def test_build_ingest_driver_builds_a_warmup_gate_iff_the_threshold_is_positive(
        monkeypatch, threshold):
    neo4j_driver = _Closable()

    class _GraphDatabase:
        @staticmethod
        def driver(uri, auth):
            return neo4j_driver

    async def _init_indices(graphiti):
        return None

    monkeypatch.setattr(cli, "build_graphiti", lambda s: _Closable())
    monkeypatch.setattr(cli, "make_docext_client", lambda **kw: _Closable())
    monkeypatch.setattr(cli, "AsyncGraphDatabase", _GraphDatabase)
    monkeypatch.setattr(cli, "init_indices", _init_indices)
    monkeypatch.setattr(cli, "Provenance", lambda driver: object())
    monkeypatch.setattr(cli, "install_dedup_guard", lambda *a, **kw: _Guard())

    settings = _settings(ingest_warmup_articles=threshold, group_id="g-ingest")
    ingest, _graphiti, _docext, driver = await cli._build_ingest_driver(settings)

    assert driver is neo4j_driver
    gate = ingest.warmup_gate
    if threshold == 0:
        assert gate is None, "threshold 0 is OFF: no gate, not an always-False one"
        return
    assert isinstance(gate, WarmupGate)
    assert gate._threshold == threshold, "the knob must reach the gate, not a hard-coded 8"
    assert gate._group_id == "g-ingest"
    assert gate._driver is neo4j_driver, "the gate counts on the same neo4j driver"
