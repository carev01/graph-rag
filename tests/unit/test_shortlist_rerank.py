"""Reranking replaces the LLM's relevance score and supplies the selection the
cosine shortlist was never doing (level 1 holds 11 communities and k was 10)."""
import pytest

import answer_api.global_search as global_search_mod
from answer_api.global_search import RerankStats, _apply_rerank, shortlist_communities
from graph_extract.config import ExtractSettings


def _hit(cid, title):
    from answer_api.global_search import CommunityHit
    return CommunityHit(community_id=cid, title=title, summary=f"{title} summary",
                        level=1, rating=5.0, cited_fact_uuids=["f1"],
                        full_report="[]", similarity=0.5)


HITS = [_hit("a", "Alpha"), _hit("b", "Bravo"), _hit("c", "Charlie")]


def test_keeps_top_n_above_the_floor_in_rerank_order():
    scored = [(2, 0.9), (0, 0.6), (1, 0.2)]
    st = RerankStats()
    out = _apply_rerank(HITS, scored, top_n=2, floor=0.5, stats=st)
    assert [h.community_id for h in out] == ["c", "a"]
    assert st.degraded is None


def test_floor_can_reject_everything():
    """An off-topic question must be able to yield zero survivors, so the caller
    takes the refusal path instead of synthesising junk."""
    out = _apply_rerank(HITS, [(0, 0.1), (1, 0.05)], top_n=3, floor=0.5,
                        stats=RerankStats())
    assert out == []


def test_top_n_caps_even_when_all_clear_the_floor():
    out = _apply_rerank(HITS, [(0, 0.9), (1, 0.8), (2, 0.7)], top_n=2, floor=0.5,
                        stats=RerankStats())
    assert len(out) == 2


def test_relevance_carries_the_rerank_score():
    out = _apply_rerank(HITS, [(1, 0.77)], top_n=3, floor=0.5, stats=RerankStats())
    assert out[0].relevance == pytest.approx(0.77)


def test_none_degrades_to_cosine_order_and_marks_it():
    """None means the reranker could not score -- NOT that nothing is relevant."""
    st = RerankStats()
    out = _apply_rerank(HITS, None, top_n=2, floor=0.5, stats=st)
    assert [h.community_id for h in out] == ["a", "b"]
    assert st.degraded == "rerank-unavailable"


def test_degraded_fallback_still_respects_top_n():
    st = RerankStats()
    out = _apply_rerank(HITS, None, top_n=1, floor=0.5, stats=st)
    assert len(out) == 1 and st.degraded == "rerank-unavailable"


def test_empty_scored_list_is_not_degradation():
    """[] means "scored, nothing cleared the bar". That is a real verdict."""
    st = RerankStats()
    out = _apply_rerank(HITS, [], top_n=2, floor=0.5, stats=st)
    assert out == [] and st.degraded is None


# ---------------------------------------------------------------------------
# shortlist_communities -- the caller of _apply_rerank / rerank(), previously
# untested. All community rows carry a citable fact so _rank_hits keeps them.
# ---------------------------------------------------------------------------

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


def _settings(**kw):
    base = dict(_env_file=None, **_MIN)
    base.update(kw)
    return ExtractSettings(**base)


def _row(cid, title, emb, rating=0.0):
    return dict(community_id=cid, title=title, summary=f"{title} summary", level=1,
               rating=rating, cited_fact_uuids=["f1"], full_report="[]", embedding=emb)


# Cosine order against query_vec=[1,0]: A (1.0) > B (~0.994) > C (~0.707).
_ROWS = [_row("a", "Alpha", [1.0, 0.0]), _row("b", "Bravo", [0.9, 0.1]),
        _row("c", "Charlie", [0.5, 0.5])]


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


async def test_shortlist_calls_rerank_with_candidate_docs_and_top_n(monkeypatch):
    seen = {}

    async def fake_rerank(query, documents, *, top_k, settings):
        seen["query"] = query
        seen["documents"] = documents
        seen["top_k"] = top_k
        return [(0, 0.9), (2, 0.8)]

    monkeypatch.setattr(global_search_mod, "rerank", fake_rerank)
    s = _settings(rerank_base_url="https://rr.example/v1", rerank_model="rerank-3",
                 rerank_api_key="k", rerank_candidates=10, rerank_top_n=2,
                 rerank_score_floor=0.5)
    out = await shortlist_communities(_FakeDriver(_ROWS), _FakeEmbedder([1.0, 0.0]),
                                      "q", level=1, k=5, group_id="g", settings=s)
    assert seen["query"] == "q"
    assert seen["documents"] == ["Alpha: Alpha summary", "Bravo: Bravo summary",
                                 "Charlie: Charlie summary"]
    assert seen["top_k"] == 2
    # rerank_top_n=2 and both scores clear rerank_score_floor=0.5
    assert [h.community_id for h in out] == ["a", "c"]
    assert out[0].relevance == pytest.approx(0.9)
    assert out[1].relevance == pytest.approx(0.8)


async def test_shortlist_rerank_score_floor_drops_low_scorers(monkeypatch):
    async def fake_rerank(query, documents, *, top_k, settings):
        return [(0, 0.9), (1, 0.2), (2, 0.1)]

    monkeypatch.setattr(global_search_mod, "rerank", fake_rerank)
    s = _settings(rerank_base_url="https://rr.example/v1", rerank_model="rerank-3",
                 rerank_api_key="k", rerank_candidates=10, rerank_top_n=3,
                 rerank_score_floor=0.5)
    out = await shortlist_communities(_FakeDriver(_ROWS), _FakeEmbedder([1.0, 0.0]),
                                      "q", level=1, k=5, group_id="g", settings=s)
    assert [h.community_id for h in out] == ["a"]        # only 0.9 clears the 0.5 floor


async def test_shortlist_rerank_none_falls_back_to_cosine_and_marks_degraded(monkeypatch):
    async def fake_rerank(query, documents, *, top_k, settings):
        return None

    monkeypatch.setattr(global_search_mod, "rerank", fake_rerank)
    s = _settings(rerank_base_url="https://rr.example/v1", rerank_model="rerank-3",
                 rerank_api_key="k", rerank_candidates=10, rerank_top_n=2,
                 rerank_score_floor=0.5)
    st = RerankStats()
    out = await shortlist_communities(_FakeDriver(_ROWS), _FakeEmbedder([1.0, 0.0]),
                                      "q", level=1, k=5, group_id="g", settings=s,
                                      stats=st)
    assert [h.community_id for h in out] == ["a", "b"]   # cosine order, capped at top_n
    assert st.degraded == "rerank-unavailable"


async def test_shortlist_rerank_not_configured_never_calls_rerank(monkeypatch):
    called = []

    async def fake_rerank(query, documents, *, top_k, settings):
        called.append(1)
        return None

    monkeypatch.setattr(global_search_mod, "rerank", fake_rerank)
    s = _settings()                                       # rerank_base_url="" default
    out = await shortlist_communities(_FakeDriver(_ROWS), _FakeEmbedder([1.0, 0.0]),
                                      "q", level=1, k=2, group_id="g", settings=s)
    assert called == []
    assert [h.community_id for h in out] == ["a", "b"]    # plain cosine top-k
    assert all(h.relevance is None for h in out)
