from answer_api.global_search import _cosine, _rank_hits


def test_cosine_basic():
    assert _cosine([1, 0], [1, 0]) == 1.0
    assert abs(_cosine([1, 0], [0, 1])) < 1e-9
    assert _cosine([0, 0], [1, 1]) == 0.0            # zero vector -> 0, no div error


def _row(cid, emb, rating=5.0, cited=("f1",)):
    return dict(community_id=cid, title=cid, summary="s", level=1, rating=rating,
                cited_fact_uuids=list(cited), full_report="[]", embedding=emb)


def test_rank_by_cosine_then_rating_boost_and_k():
    q = [1.0, 0.0]
    rows = [_row("a", [1.0, 0.0]), _row("b", [0.9, 0.1]), _row("c", [0.0, 1.0])]
    hits = _rank_hits(q, rows, k=2, rating_boost=0.1)
    assert [h.community_id for h in hits] == ["a", "b"]     # closest two, c dropped by k
    assert hits[0].similarity > hits[1].similarity


def test_rank_skips_cite_less_communities():
    q = [1.0, 0.0]
    rows = [_row("a", [1.0, 0.0], cited=[]), _row("b", [0.5, 0.5])]
    hits = _rank_hits(q, rows, k=5, rating_boost=0.1)
    assert [h.community_id for h in hits] == ["b"]          # 'a' has no cited facts
