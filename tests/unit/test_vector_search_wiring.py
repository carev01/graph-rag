"""The routing and the index lifecycle are wired where the spec says."""
from __future__ import annotations

from typing import Any

import pytest
from typer.testing import CliRunner

from graph_extract import cli
from graph_extract import graphiti_client as gc
from graph_extract.config import ExtractSettings


def _settings(**over: Any) -> ExtractSettings:
    base: dict[str, Any] = dict(_env_file=None, neo4j_uri="bolt://x", neo4j_user="u",
                                neo4j_password="p", docext_base_url="http://x",
                                docext_read_key="k")
    base.update(over)
    return ExtractSettings(**base)


def test_build_graphiti_installs_routing_per_the_setting(monkeypatch):
    seen: list[tuple[bool, int]] = []
    monkeypatch.setattr(gc, "install_vector_search",
                        lambda *, enabled, fetch_k: seen.append((enabled, fetch_k)) or enabled)
    s = _settings(vector_search_enabled=False, vector_search_fetch_k=77,
                  embed_base_url="http://e", llm_base_url="http://l")
    g = gc.build_graphiti(s)
    assert seen == [(False, 77)]
    del g


@pytest.mark.asyncio
async def test_init_indices_ensures_vector_indexes_only_when_given_a_dim(monkeypatch):
    calls: list[tuple[int, dict]] = []

    async def _ensure(driver, embed_dim, **kwargs):
        calls.append((embed_dim, kwargs))

    class _G:
        driver = object()

        async def build_indices_and_constraints(self):
            return None

    monkeypatch.setattr(gc, "ensure_vector_indexes", _ensure)
    await gc.init_indices(_G())
    assert calls == []
    await gc.init_indices(_G(), embed_dim=768, vector_index_wait_seconds=12.0)
    assert calls == [(768, {"wait_seconds": 12.0})]


@pytest.mark.asyncio
async def test_ingest_passes_the_startup_wait_to_init_indices(monkeypatch):
    seen: list[dict] = []

    async def _init(graphiti, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("stop after init_indices")

    class _Closable:
        async def close(self):
            return None

        async def aclose(self):
            return None

    monkeypatch.setattr(cli, "build_graphiti", lambda s: _Closable())
    monkeypatch.setattr(cli, "make_docext_client", lambda **kw: _Closable())
    monkeypatch.setattr(cli.AsyncGraphDatabase, "driver", lambda *a, **kw: _Closable())
    monkeypatch.setattr(cli, "init_indices", _init)
    s = _settings(vector_search_enabled=True, embed_dim=8,
                  vector_index_startup_wait_seconds=9.0)
    with pytest.raises(RuntimeError, match="stop after init_indices"):
        await cli._build_ingest_driver(s)
    assert seen == [{"embed_dim": 8, "vector_index_wait_seconds": 9.0}]


def test_vector_index_rebuild_waits_long_and_reports_state(monkeypatch):
    import json

    monkeypatch.setattr(cli, "get_extract_settings", lambda: _settings(embed_dim=8))
    events: list[tuple] = []

    class _Driver:
        async def close(self):
            events.append(("close",))

    monkeypatch.setattr(cli.AsyncGraphDatabase, "driver", lambda *a, **kw: _Driver())

    async def _drop(driver):
        events.append(("drop",))

    async def _ensure(driver, embed_dim, **kwargs):
        events.append(("ensure", embed_dim, kwargs))

    async def _status(driver):
        return {"relates_to_fact_embedding_vec": {
            "state": "POPULATING", "populationPercent": 55.0, "config": {}}}

    monkeypatch.setattr(cli.vector_search, "drop_vector_indexes", _drop)
    monkeypatch.setattr(cli.vector_search, "ensure_vector_indexes", _ensure)
    monkeypatch.setattr(cli.vector_search, "index_status", _status)
    result = CliRunner().invoke(cli.app, ["vector-index", "--rebuild", "--yes"])
    assert result.exit_code == 0, result.output
    assert events[:2] == [("drop",), ("ensure", 8, {"wait_seconds": 3600.0})]
    out = json.loads(result.output[result.output.index("{"):])
    assert out["relates_to_fact_embedding_vec"]["state"] == "POPULATING"
    assert out["relates_to_fact_embedding_vec"]["populationPercent"] == 55.0


def test_vector_index_rebuild_refuses_without_yes(monkeypatch):
    monkeypatch.setattr(cli, "get_extract_settings", lambda: _settings())

    def _forbidden_driver(*a, **kw):
        raise AssertionError(
            "the --rebuild/--yes guard must return before any Neo4j driver is built")

    monkeypatch.setattr(cli.AsyncGraphDatabase, "driver", _forbidden_driver)
    result = CliRunner().invoke(cli.app, ["vector-index", "--rebuild"])
    # exit_code == 2 (typer.Exit(code=2)) specifically, not merely nonzero: an
    # unguarded run also exits nonzero, just later and by crashing while
    # actually trying to reach Neo4j (see the AssertionError above).
    assert result.exit_code == 2
    assert "--yes" in result.output
