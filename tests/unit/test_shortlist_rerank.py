"""Reranking replaces the LLM's relevance score and supplies the selection the
cosine shortlist was never doing (level 1 holds 11 communities and k was 10)."""
import pytest

from answer_api.global_search import RerankStats, _apply_rerank


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
