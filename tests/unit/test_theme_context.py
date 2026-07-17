from theme_builder.context import EntityRow, FactRow, assemble_context


def _ent(u, deg): return EntityRow(uuid=u, name=u, type="Concept", summary="s", degree=deg)
def _fact(u, valid, invalid=None): return FactRow(uuid=u, fact="F"+u, valid_at=valid, invalid_at=invalid, name="Provides")


def test_members_ranked_by_degree_and_capped():
    members = [_ent("aaa", 1), _ent("bbb", 9), _ent("ccc", 5)]
    res = assemble_context(members, [], top_entities=2, token_budget=100000)
    # highest-degree first, capped at 2 -> bbb then ccc ; aaa excluded
    assert res.text.index("- bbb ") < res.text.index("- ccc ")
    assert "- aaa " not in res.text            # lowest degree dropped by the cap
    assert res.text.count("(Concept)") == 2    # only 2 members rendered


def test_facts_current_first_then_recency_and_labelled():
    facts = [_fact("f1", "2020", invalid="2021"), _fact("f2", "2019"), _fact("f3", "2022")]
    res = assemble_context([], facts, top_entities=30, token_budget=100000)
    # current (invalid_at is None) first, recency desc: f3(2022) then f2(2019) then superseded f1
    order = [res.text.index("[f3]"), res.text.index("[f2]"), res.text.index("[f1]")]
    assert order == sorted(order)
    assert res.fact_uuids == {"f1", "f2", "f3"}
    assert "[f1] Ff1" in res.text


def test_token_budget_truncates_facts():
    facts = [_fact(f"f{i}", "2020") for i in range(200)]
    res = assemble_context([], facts, top_entities=30, token_budget=200)  # ~800 chars
    assert len(res.fact_uuids) < 200          # truncated
    assert len(res.text) <= 200 * 4 + 200     # roughly within budget (chars ~= 4*tokens)


def test_single_oversized_fact_is_clipped_not_blown():
    big = FactRow(uuid="big", fact="x" * 10000, valid_at="2020", invalid_at=None, name="Provides")
    res = assemble_context([], [big], top_entities=30, token_budget=10)  # char_budget=40
    assert "big" in res.fact_uuids                      # still included (>=1 guarantee)
    assert len(res.text) <= 40 + 60                     # clipped, not a 10k overshoot
    assert res.text.rstrip().endswith(("x", "]")) or "[big]" in res.text  # label preserved
