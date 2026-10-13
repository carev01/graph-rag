"""Scoped local search falls back to ranking the scope's own facts.

A small vendor next to a huge one (Cohesity: 92 articles, Veeam: 20k) has no edges in
the corpus-wide top 450, so the post-filter returns nothing; the scan ranks the scope's
own RELATES_TO edges by cosine instead (2026-10-12 amendment, task 11)."""
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import answer_api.search as search_mod
from answer_api.scope import Scope
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_SCOPE = Scope(vendors=("Cohesity",))


def _edge(uuid, in_scope=True):
    return SimpleNamespace(uuid=uuid, fact=f"fact {uuid}", invalid_at=None, valid_at=None,
                           episodes=["in" if in_scope else "out"])


class _Graph:
    """Returns the same edges whatever the fetch limit."""

    def __init__(self, edges):
        self.edges = edges
        self.limits = []

    async def _search(self, q, config, group_ids, center_node_uuid=None):
        self.limits.append(config.limit)
        return SimpleNamespace(edges=list(self.edges))


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    async def _allowed(driver, scope, candidates=None):
        return {c for c in candidates if c == "in"}

    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            return {}

    monkeypatch.setattr(search_mod, "scope_episode_uuids", _allowed)
    monkeypatch.setattr(search_mod, "Provenance", _Prov)


@pytest.fixture
def scan_calls(monkeypatch):
    calls = []

    async def _scan(graphiti, driver, *, q, scope, k, group_id, exclude,
                    include_invalid=False, max_episodes=6000, settings=None):
        calls.append({"k": k, "exclude": set(exclude), "max_episodes": max_episodes,
                      "include_invalid": include_invalid, "settings": settings})
        return [_edge("p1"), _edge("s1"), _edge("s2"), _edge("s3")]

    monkeypatch.setattr(search_mod, "_scoped_scan", _scan)
    return calls


async def test_scan_not_called_when_post_filter_fills_k(scan_calls):
    g = _Graph([_edge(f"p{i}") for i in range(5)])
    res = await search_mod.search_local(g, None, q="q", k=5, group_id="g", scope=_SCOPE)
    assert scan_calls == []
    assert [r["fact_uuid"] for r in res["results"]] == ["p0", "p1", "p2", "p3", "p4"]


async def test_scan_appends_after_post_filter_results_deduplicated(scan_calls):
    g = _Graph([_edge("p0"), _edge("p1"), _edge("x", in_scope=False)])
    res = await search_mod.search_local(g, None, q="q", k=4, group_id="g", scope=_SCOPE,
                                        scan_max_episodes=123)
    assert len(scan_calls) == 1
    assert scan_calls[0]["exclude"] == {"p0", "p1"}
    assert scan_calls[0]["max_episodes"] == 123
    assert scan_calls[0]["settings"] is None
    # p1 came back from the scan too: kept once, in its post-filter position.
    assert [r["fact_uuid"] for r in res["results"]] == ["p0", "p1", "s1", "s2"]
    assert res["count"] == 4


async def test_scan_not_called_for_unscoped_search(scan_calls):
    g = _Graph([_edge("p0")])
    res = await search_mod.search_local(g, None, q="q", k=10, group_id="g")
    assert scan_calls == []
    assert res["count"] == 1


async def test_scan_failure_returns_post_filter_results(monkeypatch, caplog):
    async def _boom(*a, **kw):
        raise RuntimeError("neo4j down")

    monkeypatch.setattr(search_mod, "_scoped_scan", _boom)
    g = _Graph([_edge("p0")])
    with caplog.at_level(logging.WARNING, logger="answer_api.search"):
        res = await search_mod.search_local(g, None, q="q", k=5, group_id="g",
                                            scope=_SCOPE)
    assert [r["fact_uuid"] for r in res["results"]] == ["p0"]
    assert "scoped scan failed" in caplog.text


# --- _scoped_scan itself -----------------------------------------------------------


class _Embedder:
    def __init__(self):
        self.calls = []

    async def create(self, input_data):
        self.calls.append(input_data)
        return [0.1, 0.2]


class _Driver:
    def __init__(self, records):
        self.records = records
        self.calls = []

    async def execute_query(self, query, **params):
        self.calls.append((query, params))
        return self.records, None, None


def _row(n, uuid=None, **kw):
    base = {"n": n, "uuid": uuid, "fact": None, "episodes": None,
            "valid_at": None, "invalid_at": None}
    base.update(kw)
    return base


async def test_scoped_scan_embeds_and_ranks_the_scope_edges():
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    emb = _Embedder()
    drv = _Driver([_row(356, "s1", fact="DataLock is WORM", episodes=["e1"], valid_at=when),
                   _row(356, "s2", fact="FortKnox vault", episodes=["e2"])])
    graphiti = SimpleNamespace(embedder=emb)
    out = await search_mod._scoped_scan(
        graphiti, drv, q="immutable\nbackups", scope=Scope(vendors=("Cohesity",)), k=7,
        group_id="g", exclude={"p0"}, max_episodes=500)
    # Same call shape graphiti's search uses (newlines flattened).
    assert emb.calls == [["immutable backups"]]
    _, params = drv.calls[0]
    assert params["vendors"] == ["cohesity"]
    assert params["vec"] == [0.1, 0.2]
    assert params["k"] == 7
    assert params["exclude"] == ["p0"]
    assert params["max_episodes"] == 500
    assert params["episode_cap"] == 501
    assert params["include_invalid"] is False
    assert params["group_id"] == "g"
    assert [e.uuid for e in out] == ["s1", "s2"]
    assert out[0].fact == "DataLock is WORM"
    assert out[0].episodes == ["e1"]
    assert out[0].valid_at == when
    assert out[1].invalid_at is None


async def test_scoped_scan_skips_a_scope_over_the_episode_limit(caplog):
    emb = _Embedder()
    drv = _Driver([_row(501)])          # n = episode_cap: the scope has > 500 episodes
    with caplog.at_level(logging.INFO, logger="answer_api.search"):
        out = await search_mod._scoped_scan(
            SimpleNamespace(embedder=emb), drv, q="q", scope=Scope(vendors=("Veeam",)),
            k=5, group_id="g", exclude=set(), max_episodes=500)
    assert out == []
    assert "skipped" in caplog.text


async def test_over_limit_scope_returns_post_filter_results(monkeypatch):
    emb = _Embedder()
    drv = _Driver([_row(20001)])
    g = _Graph([_edge("p0")])
    g.embedder = emb
    res = await search_mod.search_local(g, drv, q="q", k=5, group_id="g",
                                        scope=Scope(vendors=("Veeam",)))
    assert [r["fact_uuid"] for r in res["results"]] == ["p0"]
    assert len(drv.calls) == 1


async def test_scoped_scan_converts_neo4j_datetimes():
    from neo4j.time import DateTime
    emb = _Embedder()
    drv = _Driver([_row(3, "s1", fact="f", episodes=["e"],
                        invalid_at=DateTime(2030, 1, 1, tzinfo=timezone.utc))])
    out = await search_mod._scoped_scan(
        SimpleNamespace(embedder=emb), drv, q="q", scope=_SCOPE, k=5, group_id="g",
        exclude=set(), include_invalid=True)
    assert isinstance(out[0].invalid_at, datetime)
    assert drv.calls[0][1]["include_invalid"] is True


# --- cross-encoder rerank of the scan's candidates --------------------------------

def _settings(**kw):
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                local_scoped_scan_candidates=50)
    base.update(kw)
    return ExtractSettings(**base)


_RERANK = dict(rerank_base_url="http://rr", rerank_model="rerank-x")


def _cands(n):
    return [_row(100, f"c{i}", fact=f"fact c{i}", episodes=["e"]) for i in range(n)]


async def test_scan_reranks_the_cosine_candidates_when_configured(monkeypatch):
    seen = {}

    async def _rerank(query, documents, *, top_k, settings):
        seen.update(query=query, documents=documents, top_k=top_k)
        return [(4, 0.9), (0, 0.8), (2, 0.1)]

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    drv = _Driver(_cands(5))
    out = await search_mod._scoped_scan(
        SimpleNamespace(embedder=_Embedder()), drv, q="immutable", scope=_SCOPE, k=3,
        group_id="g", exclude=set(), settings=_settings(**_RERANK))
    assert drv.calls[0][1]["k"] == 50            # fetches local_scoped_scan_candidates
    assert seen == {"query": "immutable", "documents": [f"fact c{i}" for i in range(5)],
                    "top_k": 3}
    assert [e.uuid for e in out] == ["c4", "c0", "c2"]


async def test_scan_unconfigured_reranker_keeps_cosine_order(monkeypatch, caplog):
    async def _rerank(*a, **kw):
        raise AssertionError("rerank must not be called")

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    drv = _Driver(_cands(5))
    with caplog.at_level(logging.WARNING, logger="answer_api.search"):
        out = await search_mod._scoped_scan(
            SimpleNamespace(embedder=_Embedder()), drv, q="q", scope=_SCOPE, k=3,
            group_id="g", exclude=set(), settings=_settings())
    assert drv.calls[0][1]["k"] == 3
    assert [e.uuid for e in out] == ["c0", "c1", "c2"]
    assert "no reranker" in caplog.text


@pytest.mark.parametrize("outcome", ["none", "raise"])
async def test_scan_rerank_failure_keeps_cosine_order(monkeypatch, caplog, outcome):
    async def _rerank(*a, **kw):
        if outcome == "raise":
            raise RuntimeError("voyage down")
        return None

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    drv = _Driver(_cands(5))
    with caplog.at_level(logging.WARNING, logger="answer_api.search"):
        out = await search_mod._scoped_scan(
            SimpleNamespace(embedder=_Embedder()), drv, q="q", scope=_SCOPE, k=3,
            group_id="g", exclude=set(), settings=_settings(**_RERANK))
    assert [e.uuid for e in out] == ["c0", "c1", "c2"]
    assert "cosine order" in caplog.text


async def test_search_local_passes_settings_to_the_scan(scan_calls):
    s = _settings(**_RERANK)
    g = _Graph([_edge("p0")])
    await search_mod.search_local(g, None, q="q", k=4, group_id="g", scope=_SCOPE,
                                  settings=s)
    assert scan_calls[0]["settings"] is s


async def test_scan_query_excludes_edges_with_no_embedding():
    # ORDER BY ... DESC puts nulls first in Cypher, so an edge with no
    # fact_embedding would otherwise rank above every real candidate.
    assert "f.fact_embedding IS NOT NULL" in search_mod._SCAN


async def test_scan_rerank_pads_an_empty_reply_from_cosine_order(monkeypatch):
    async def _rerank(*a, **kw):
        return []

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    drv = _Driver(_cands(5))
    out = await search_mod._scoped_scan(
        SimpleNamespace(embedder=_Embedder()), drv, q="q", scope=_SCOPE, k=3,
        group_id="g", exclude=set(), settings=_settings(**_RERANK))
    assert [e.uuid for e in out] == ["c0", "c1", "c2"]


async def test_scan_rerank_pads_a_short_reply_from_remaining_cosine_order(monkeypatch):
    async def _rerank(query, documents, *, top_k, settings):
        return [(3, 0.9)]

    monkeypatch.setattr(search_mod, "rerank", _rerank)
    drv = _Driver(_cands(5))
    out = await search_mod._scoped_scan(
        SimpleNamespace(embedder=_Embedder()), drv, q="q", scope=_SCOPE, k=3,
        group_id="g", exclude=set(), settings=_settings(**_RERANK))
    # c3 (the one usable rerank hit) first, then the next unchosen cosine
    # candidates (c0, c1) padded in to reach k=3.
    assert [e.uuid for e in out] == ["c3", "c0", "c1"]
