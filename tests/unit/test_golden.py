from answer_api.golden import first_hit_rank, precision_at_k

_R = [
    {"fact_uuid": "f1", "sources": [{"article_id": "art1", "url": "u1", "title": "t1"}]},
    {"fact_uuid": "f2", "sources": [{"article_id": "art2", "url": "u2", "title": "t2"}]},
    {"fact_uuid": "f3", "sources": []},  # dangling — no citation
]


def test_precision_hit():
    assert precision_at_k(_R, ["art2"]) is True
    assert precision_at_k(_R, ["artX", "art1"]) is True


def test_precision_miss():
    assert precision_at_k(_R, ["artX"]) is False
    assert precision_at_k([], ["art1"]) is False


def test_first_hit_rank():
    assert first_hit_rank(_R, ["art2"]) == 2
    assert first_hit_rank(_R, ["art1"]) == 1
    assert first_hit_rank(_R, ["artX"]) is None


def test_dangling_source_not_counted():
    # a result with empty sources never counts as a hit
    assert precision_at_k([_R[2]], ["art3"]) is False
