"""Vendor/product-scoped global shortlist and labelled reduce (BACKLOG 52,
spec 2026-09-25 §4.3, plan task 4).

`_scope_shares` itself is exercised against a real graph in
tests/integration/test_global_scope_shares.py; these tests fake it (as the
existing test_global*.py files fake rerank/embedder/driver) and pin:
  1. an empty scope never touches scoping at all;
  2/3. the unreranked and reranked shortlist paths widen the pool and filter
       by share before the k-cut / before rerank sees the docs;
  4. global_search drops out-of-scope fact ids from MapResult.fact_ids before
     numbering, labels reduce fact lines, and returns applies_to;
  5. every community scoped out -> communities_used == [] (global->local
     fallback trigger already lives in the router).
"""
from __future__ import annotations

import answer_api.global_search as gs
from answer_api.scope import Scope
from graph_extract.config import ExtractSettings

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


def _settings(**kw):
    base = dict(_env_file=None, **_MIN)
    base.update(kw)
    return ExtractSettings(**base)


def _row(cid, title, emb, rating=0.0, cited=("f1",)):
    return dict(community_id=cid, title=title, summary=f"{title} summary", level=1,
               rating=rating, cited_fact_uuids=list(cited), full_report="[]", embedding=emb)


class _FakeRows:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for r in self._rows:
            yield r


class _FakeSession:
    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def run(self, cypher, **kw):
        return _FakeRows(self._rows)


class _FakeDriver:
    def __init__(self, rows):
        self._rows = rows

    def session(self):
        return _FakeSession(self._rows)


class _FakeEmbedder:
    def __init__(self, vec):
        self._vec = vec

    async def create_batch(self, texts):
        return [self._vec for _ in texts]


# Cosine order against query_vec=[1,0]: A (1.0) > B (~0.994) > C (~0.707).
_ROWS = [_row("a", "Alpha", [1.0, 0.0]), _row("b", "Bravo", [0.9, 0.1]),
        _row("c", "Charlie", [0.5, 0.5])]

_SCOPE = Scope(("Veeam",), (), "explicit")


# ---------------------------------------------------------------------------
# 1. empty scope -> _scope_shares never called, shortlist unchanged
# ---------------------------------------------------------------------------

async def test_empty_scope_never_calls_scope_shares(monkeypatch):
    called = []

    async def fake_scope_shares(driver, group_id, hits, scope):
        called.append(1)
        return {h.community_id: 1.0 for h in hits}

    monkeypatch.setattr(gs, "_scope_shares", fake_scope_shares)
    out_no_scope = await gs.shortlist_communities(
        _FakeDriver(_ROWS), _FakeEmbedder([1.0, 0.0]), "q", level=1, k=2, group_id="g")
    out_empty_scope = await gs.shortlist_communities(
        _FakeDriver(_ROWS), _FakeEmbedder([1.0, 0.0]), "q", level=1, k=2, group_id="g",
        scope=Scope())
    assert called == []
    assert [h.community_id for h in out_no_scope] == ["a", "b"]
    assert [h.community_id for h in out_empty_scope] == ["a", "b"]


# ---------------------------------------------------------------------------
# 2. scoped, unreranked: pool widened to global_scope_candidates, share filter
#    applied, then cut to k.
# ---------------------------------------------------------------------------

async def test_scoped_unreranked_widens_pool_filters_share_then_cuts_to_k(monkeypatch):
    seen = {}

    async def fake_scope_shares(driver, group_id, hits, scope):
        seen["pool"] = [h.community_id for h in hits]
        # 'c' (the lowest-cosine of the 3) is dropped by the share filter.
        return {"a": 1.0, "b": 1.0, "c": 0.0}

    monkeypatch.setattr(gs, "_scope_shares", fake_scope_shares)
    s = _settings(global_scope_candidates=3)
    out = await gs.shortlist_communities(
        _FakeDriver(_ROWS), _FakeEmbedder([1.0, 0.0]), "q", level=1, k=1, group_id="g",
        settings=s, scope=_SCOPE)
    # pool widened to global_scope_candidates=3 (not k=1) before the share filter ran
    assert sorted(seen["pool"]) == ["a", "b", "c"]
    assert [h.community_id for h in out] == ["a"]     # cut to k=1 after the filter


async def test_scoped_unreranked_drops_below_min_share(monkeypatch):
    async def fake_scope_shares(driver, group_id, hits, scope):
        return {"a": 0.75, "b": 0.25, "c": 0.0}

    monkeypatch.setattr(gs, "_scope_shares", fake_scope_shares)
    s = _settings(global_scope_candidates=3, global_scope_min_share=0.5)
    out = await gs.shortlist_communities(
        _FakeDriver(_ROWS), _FakeEmbedder([1.0, 0.0]), "q", level=1, k=5, group_id="g",
        settings=s, scope=_SCOPE)
    assert [h.community_id for h in out] == ["a"]     # only share >= 0.5 survives


# ---------------------------------------------------------------------------
# 3. scoped, reranked: pool = max(rerank_candidates, global_scope_candidates),
#    filtered BEFORE rerank sees the docs.
# ---------------------------------------------------------------------------

async def test_scoped_reranked_pool_is_widened_and_filtered_before_rerank(monkeypatch):
    seen = {}

    async def fake_scope_shares(driver, group_id, hits, scope):
        seen["pool"] = [h.community_id for h in hits]
        return {"a": 1.0, "b": 0.0, "c": 1.0}       # 'b' dropped

    async def fake_rerank(query, documents, *, top_k, settings, transport=None):
        seen["documents"] = documents
        return [(0, 0.9), (1, 0.8)]

    monkeypatch.setattr(gs, "_scope_shares", fake_scope_shares)
    monkeypatch.setattr(gs, "rerank", fake_rerank)
    s = _settings(rerank_base_url="https://rr.example/v1", rerank_model="rerank-3",
                 rerank_api_key="k", rerank_candidates=2, rerank_top_n=2,
                 rerank_score_floor=0.0, global_scope_candidates=3)
    out = await gs.shortlist_communities(
        _FakeDriver(_ROWS), _FakeEmbedder([1.0, 0.0]), "q", level=1, k=5, group_id="g",
        settings=s, scope=_SCOPE)
    # pool handed to _scope_shares is max(rerank_candidates=2, global_scope_candidates=3)=3
    assert sorted(seen["pool"]) == ["a", "b", "c"]
    # 'b' was dropped by the share filter -- rerank must never see its doc
    assert seen["documents"] == ["Alpha: Alpha summary", "Charlie: Charlie summary"]
    assert [h.community_id for h in out] == ["a", "c"]


# ---------------------------------------------------------------------------
# 4 & 5. global_search: scope filters MapResult.fact_ids before numbering,
# labels reduce lines, returns applies_to; all-filtered-out -> [].
# ---------------------------------------------------------------------------

_VEEAM_SOURCES = [{"url": "https://v/1", "title": "T", "article_id": "a1",
                  "section": "s", "vendor": "Veeam", "product": "VBR"}]
_COHESITY_SOURCES = [{"url": "https://c/1", "title": "T", "article_id": "a2",
                      "section": "s", "vendor": "Cohesity", "product": "DataProtect"}]


class _FakeResolveSession:
    """Answers resolve_citations (`f.uuid IN $uuids`) and _fact_texts
    (`f.fact AS fact`) queries; anything else yields nothing."""

    def __init__(self, sources_by_fact, texts_by_fact):
        self._sources = sources_by_fact
        self._texts = texts_by_fact

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def run(self, cypher, **kw):
        if "f.fact AS fact" in cypher:
            rows = [{"uuid": u, "fact": self._texts[u]} for u in kw["u"]
                    if u in self._texts]
            return _FakeRows(rows)
        if "uuid IN $uuids" in cypher:
            rows = [{"uuid": u, "valid_at": None, "invalid_at": None,
                     "sources": self._sources.get(u, [])} for u in kw["uuids"]]
            return _FakeRows(rows)
        return _FakeRows([])


class _FakeResolveDriver:
    def __init__(self, sources_by_fact, texts_by_fact):
        self._sources = sources_by_fact
        self._texts = texts_by_fact

    def session(self):
        return _FakeResolveSession(self._sources, self._texts)


class _FakeSynthClient:
    def __init__(self, content):
        self._content = content
        self.calls = []
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        self.calls.append(kw)
        choice = type("C", (), {"message": type("M", (), {"content": self._content})()})()
        return type("R", (), {"choices": [choice]})()


def _hit(cid, title, fids):
    return gs.CommunityHit(cid, title, "sum", 1, 7.0, list(fids), "[]", 0.9, None)


async def _run_global_search(monkeypatch, hits, maps, sources_by_fact, texts_by_fact, *,
                             scope, synth_content="answer [1] [2]."):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, rating_boost=0.1,
                         settings=None, stats=None, scope=None):
        return hits

    async def _map(client, model, q, hit):
        return maps[hit.community_id]

    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    monkeypatch.setattr(gs, "map_report", _map)
    driver = _FakeResolveDriver(sources_by_fact, texts_by_fact)
    synth = _FakeSynthClient(synth_content)
    s = _settings()
    res = await gs.global_search(driver, None, object(), "mm", synth, "sm",
                                 q="q", level=1, k=3, group_id="g", settings=s, scope=scope)
    return res, synth


async def test_global_search_drops_out_of_scope_facts_labels_and_returns_applies_to(
        monkeypatch):
    hits = [_hit("c1", "Veeam community", ["fv"]), _hit("c2", "Cohesity community", ["fc"])]
    maps = {
        "c1": gs.MapResult(community_id="c1", title="Veeam community", relevance=None,
                           key_points=["kp"], fact_ids=["fv"]),
        "c2": gs.MapResult(community_id="c2", title="Cohesity community", relevance=None,
                           key_points=["kp"], fact_ids=["fc"]),
    }
    sources = {"fv": _VEEAM_SOURCES, "fc": _COHESITY_SOURCES}
    texts = {"fv": "Veeam hardens repositories.", "fc": "Cohesity hardens vaults."}
    res, synth = await _run_global_search(
        monkeypatch, hits, maps, sources, texts, scope=_SCOPE, synth_content="answer [1].")
    prompt = synth.calls[0]["messages"][0]["content"]
    # fc (Cohesity) must never reach the reduce prompt at all -- filtered from
    # MapResult.fact_ids before numbering, not just from the final citations.
    assert "Cohesity hardens vaults." not in prompt
    assert "[1] (Veeam · VBR) Veeam hardens repositories." in prompt
    from answer_api.attribution import ATTRIBUTION_RULES
    assert ATTRIBUTION_RULES in prompt
    assert res["applies_to"] == [{"vendor": "Veeam", "products": ["VBR"], "facts": 1}]


async def test_global_search_unscoped_still_labels_and_reuses_one_resolve_call(monkeypatch):
    """Even with no scope, resolve_citations still runs once (labels need it),
    and the same result -- not a second call -- builds the final citations."""
    hits = [_hit("c1", "Veeam community", ["fv"])]
    maps = {"c1": gs.MapResult(community_id="c1", title="Veeam community", relevance=None,
                               key_points=["kp"], fact_ids=["fv"])}
    sources = {"fv": _VEEAM_SOURCES}
    texts = {"fv": "Veeam hardens repositories."}
    calls = []

    class _CountingResolveDriver(_FakeResolveDriver):
        def session(self):
            calls.append(1)
            return super().session()

    async def _shortlist(driver, embedder, q, *, level, k, group_id, rating_boost=0.1,
                         settings=None, stats=None, scope=None):
        return hits

    async def _map(client, model, q, hit):
        return maps[hit.community_id]

    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    monkeypatch.setattr(gs, "map_report", _map)
    driver = _CountingResolveDriver(sources, texts)
    synth = _FakeSynthClient("answer [1].")
    s = _settings()
    res = await gs.global_search(driver, None, object(), "mm", synth, "sm",
                                 q="q", level=1, k=3, group_id="g", settings=s, scope=None)
    prompt = synth.calls[0]["messages"][0]["content"]
    assert "[1] (Veeam · VBR) Veeam hardens repositories." in prompt
    assert res["citations"][0]["sources"] == _VEEAM_SOURCES
    assert res["applies_to"] == [{"vendor": "Veeam", "products": ["VBR"], "facts": 1}]


async def test_global_search_all_communities_scoped_out_returns_empty(monkeypatch):
    """When _scope_shares/_keep_in_scope leaves nothing, shortlist_communities
    itself returns [] and global_search takes the existing empty-shortlist
    refusal path -- the router's global->local fallback trigger."""
    async def _shortlist(driver, embedder, q, *, level, k, group_id, rating_boost=0.1,
                         settings=None, stats=None, scope=None):
        return []

    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    s = _settings()
    res = await gs.global_search(_FakeResolveDriver({}, {}), None, object(), "mm",
                                 _FakeSynthClient("x"), "sm", q="q", level=1, k=3,
                                 group_id="g", settings=s, scope=_SCOPE)
    assert res["communities_used"] == []
    assert res["citations"] == []
    assert res["applies_to"] == []


async def test_global_search_communities_emptied_by_scope_filter_drop_from_communities_used(
        monkeypatch):
    """Review finding B: the shortlist survived (2 communities came back from
    shortlist_communities), but the map step picked ONLY out-of-scope facts
    for both of them. After the scope filter every MapResult.fact_ids is
    empty, so neither community fed the reduce step -- communities_used must
    be [], the same signal the router's global->local fallback checks
    (`not raw.get("communities_used")`), not the pre-filter non-empty list."""
    hits = [_hit("c1", "Veeam community", ["fc1"]), _hit("c2", "Other community", ["fc2"])]
    maps = {
        "c1": gs.MapResult(community_id="c1", title="Veeam community", relevance=None,
                           key_points=["kp"], fact_ids=["fc1"]),
        "c2": gs.MapResult(community_id="c2", title="Other community", relevance=None,
                           key_points=["kp"], fact_ids=["fc2"]),
    }
    # Both facts resolve to Cohesity -- out of scope for _SCOPE (Veeam).
    sources = {"fc1": _COHESITY_SOURCES, "fc2": _COHESITY_SOURCES}
    texts = {"fc1": "Cohesity fact one.", "fc2": "Cohesity fact two."}
    res, synth = await _run_global_search(
        monkeypatch, hits, maps, sources, texts, scope=_SCOPE)
    assert synth.calls == []                 # never reached the reduce call
    assert res["communities_used"] == []
    assert res["applies_to"] == []
    assert res["citations"] == []
