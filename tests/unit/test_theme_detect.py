from theme_builder.detect import _community_id, _build_communities_from_rows


def test_community_id_deterministic_order_independent():
    a = _community_id(0, ["u2", "u1", "u3"])
    b = _community_id(0, ["u3", "u2", "u1"])
    assert a == b and len(a) == 16
    assert _community_id(1, ["u1", "u2", "u3"]) != a       # level participates


def test_builds_levels_and_drops_dust():
    # 4 entities; level0 labels: {A:0,B:0,C:1,D:1}; level1: all -> 9
    rows = [
        {"uuid": "A", "levels": [0, 9]},
        {"uuid": "B", "levels": [0, 9]},
        {"uuid": "C", "levels": [1, 9]},
        {"uuid": "D", "levels": [1, 9]},
    ]
    comms = _build_communities_from_rows(rows, min_community_size=2, max_levels=3)
    by_level = {}
    for c in comms:
        by_level.setdefault(c.level, []).append(c)
    assert len(by_level[0]) == 2 and len(by_level[1]) == 1   # two leaves, one parent
    leaf = next(c for c in by_level[0] if "A" in c.member_uuids)
    parent = by_level[1][0]
    assert leaf.parent_id == parent.community_id            # nested parent linked
    assert parent.parent_id is None                          # top level


def test_dust_below_min_size_dropped():
    rows = [{"uuid": "A", "levels": [0]}, {"uuid": "B", "levels": [1]}]  # singletons
    comms = _build_communities_from_rows(rows, min_community_size=2, max_levels=3)
    assert comms == []                                       # both dust


def test_max_levels_caps_hierarchy():
    rows = [{"uuid": "A", "levels": [0, 5, 8]}, {"uuid": "B", "levels": [0, 5, 8]}]
    comms = _build_communities_from_rows(rows, min_community_size=1, max_levels=2)
    assert max(c.level for c in comms) == 1                  # levels 0,1 only
