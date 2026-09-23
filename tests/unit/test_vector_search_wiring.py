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
    calls: list[int] = []

    async def _ensure(driver, embed_dim):
        calls.append(embed_dim)

    class _G:
        driver = object()

        async def build_indices_and_constraints(self):
            return None

    monkeypatch.setattr(gc, "ensure_vector_indexes", _ensure)
    await gc.init_indices(_G())
    assert calls == []
    await gc.init_indices(_G(), embed_dim=768)
    assert calls == [768]


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
