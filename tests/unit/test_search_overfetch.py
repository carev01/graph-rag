"""Scoped local search re-fetches until k edges survive the scope filter."""
from types import SimpleNamespace

import pytest

import answer_api.search as search_mod
from answer_api.scope import Scope

pytestmark = pytest.mark.asyncio


def _edge(i, in_scope):
    return SimpleNamespace(uuid=f"e{i}", fact=f"f{i}", invalid_at=None, valid_at=None,
                           episodes=[f"in{i}" if in_scope else f"out{i}"])


class _Graph:
    """Every 5th edge is in scope."""

    def __init__(self):
        self.limits = []

    async def _search(self, q, config, group_ids, center_node_uuid=None):
        self.limits.append(config.limit)
        return SimpleNamespace(edges=[_edge(i, i % 5 == 0) for i in range(config.limit)])


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    async def _allowed(driver, scope, candidates=None):
        return {c for c in candidates if c.startswith("in")}

    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            return {}

    monkeypatch.setattr(search_mod, "scope_episode_uuids", _allowed)
    monkeypatch.setattr(search_mod, "Provenance", _Prov)


async def test_scoped_search_refetches_until_k_survive():
    g = _Graph()
    res = await search_mod.search_local(g, None, q="q", k=15, group_id="g",
                                        scope=Scope(vendors=("Veeam",)))
    assert g.limits == [45, 150]          # 9 survive at 45, 30 at 150 -> stop
    assert res["count"] == 15


async def test_scoped_search_stops_at_the_cap():
    g = _Graph()
    res = await search_mod.search_local(g, None, q="q", k=15, group_id="g",
                                        scope=Scope(vendors=("Veeam",)), fetch_cap=100)
    assert g.limits == [45, 100]
    assert res["count"] == 15             # 20 survive at 100, trimmed to k


async def test_unscoped_search_fetches_once():
    g = _Graph()
    await search_mod.search_local(g, None, q="q", k=15, group_id="g")
    assert g.limits == [45]
