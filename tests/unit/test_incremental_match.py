from theme_builder.detect import Community
from theme_builder.incremental import PersistedCommunity, match_communities, classify


def _fresh(cid, level, members):
    return Community(community_id=cid, level=level, member_uuids=list(members), parent_id=None)


def _pers(cid, level, members, *, embedding=(0.0,), verified=True, stale=False):
    return PersistedCommunity(community_id=cid, level=level, members=set(members),
                              title="t", summary="s", full_report="[]", rating=1.0,
                              rating_explanation="", tags=[], cited_fact_uuids=[],
                              embedding=list(embedding) if embedding else embedding,
                              generated_at="old", verified=verified, stale=stale)


def test_exact_and_wobble_match():
    fresh = [_fresh("h1", 1, ["a", "b", "c", "d"])]     # 3/4 overlap with p1 -> J=0.6
    persisted = [_pers("stable1", 1, ["a", "b", "c", "e"])]
    m = match_communities(fresh, persisted, tau=0.5)
    assert m[0] is not None and m[0].community_id == "stable1"


def test_below_threshold_is_new():
    fresh = [_fresh("h1", 1, ["a", "b", "x", "y"])]     # 2/6 overlap -> J=0.33 < 0.5
    persisted = [_pers("stable1", 1, ["a", "b", "z", "w"])]
    assert match_communities(fresh, persisted, tau=0.5)[0] is None


def test_level_isolation():
    fresh = [_fresh("h1", 1, ["a", "b"])]
    persisted = [_pers("stable1", 0, ["a", "b"])]        # same members, different level
    assert match_communities(fresh, persisted, tau=0.5)[0] is None


def test_greedy_one_to_one():
    # two fresh both overlap one persisted; only the best-Jaccard fresh claims it
    fresh = [_fresh("h1", 1, ["a", "b", "c"]),           # J with p1 = 3/3 = 1.0
             _fresh("h2", 1, ["a", "b", "z"])]           # J with p1 = 2/4 = 0.5
    persisted = [_pers("stable1", 1, ["a", "b", "c"])]
    m = match_communities(fresh, persisted, tau=0.5)
    assert m[0] is not None and m[0].community_id == "stable1"
    assert m[1] is None                                  # persisted already claimed


def test_classify_dirty_and_clean():
    fresh = [_fresh("h1", 1, ["a", "b"]), _fresh("h2", 1, ["c", "d"]), _fresh("hnew", 1, ["x"])]
    p = _pers("s1", 1, ["a", "b"])
    q = _pers("s2", 1, ["c", "d"])
    matches = {0: p, 1: q, 2: None}
    dirty, clean = classify(fresh, matches, touched={"a"})   # community 0 has touched 'a'
    assert dirty == [0, 2]     # matched+touched, and the new one
    assert clean == [1]        # matched, untouched


def test_classify_treats_a_stale_carried_over_community_as_dirty():
    """BACKLOG 5c: when regeneration fails outright, the community's previously
    verified report is carried over rather than deleted -- but it describes the
    OLD member set, and the run stamps a fresh corpus_cursor, so next run's
    `touched` would no longer include the edits that made it dirty. Without the
    `stale` flag the carried-over report would look clean forever and never be
    regenerated. It keeps its embedding (it stays retrievable); only its
    dirtiness is forced."""
    fresh = [_fresh("h1", 1, ["a", "b"])]
    matches = {0: _pers("s1", 1, ["a", "b"], stale=True)}
    dirty, clean = classify(fresh, matches, touched=set())   # nothing touched
    assert dirty == [0] and clean == []


def test_classify_treats_a_staged_community_as_dirty():
    """The spec's named hazard. A report STAGED because verification could not
    complete is persisted with verified=false and NO embedding, and load_persisted
    reads it back like any other row. Untouched members would otherwise make it look
    'clean', reusing a pending/empty report as if it were a real one -- and it would
    then never be retried."""
    fresh = [_fresh("h1", 1, ["a", "b"]),          # matched, no embedding
             _fresh("h2", 1, ["c", "d"]),          # matched, verified=False
             _fresh("h3", 1, ["e", "f"])]          # matched, verified with embedding
    matches = {0: _pers("s1", 1, ["a", "b"], embedding=None),
               1: _pers("s2", 1, ["c", "d"], verified=False),
               2: _pers("s3", 1, ["e", "f"])}
    dirty, clean = classify(fresh, matches, touched=set())   # nothing touched at all
    assert dirty == [0, 1], "a staged community must always be regenerated"
    assert clean == [2]


def test_classify_cold_start_all_dirty():
    fresh = [_fresh("h1", 1, ["a"]), _fresh("h2", 1, ["b"])]
    dirty, clean = classify(fresh, {0: None, 1: None}, touched=None)
    assert dirty == [0, 1] and clean == []
