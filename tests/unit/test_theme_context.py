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
    # verify fact_texts stores the FULL original fact, not the display text
    assert set(res.fact_texts) == res.fact_uuids       # lockstep on clipped path too
    assert res.fact_texts["big"] == "x" * 10000        # stored full text, not clipped display


def test_fact_texts_maps_uuid_to_text_for_included_facts():
    """The verifier needs each fact's TEXT, not just its uuid."""
    from theme_builder.context import EntityRow, FactRow, assemble_context
    facts = [
        FactRow(uuid="u1", fact="AWS Backup provides continuous backups for Aurora.",
                valid_at=None, invalid_at=None, name="PROVIDES"),
        FactRow(uuid="u2", fact="AWS Backup integrates with AWS KMS.",
                valid_at=None, invalid_at=None, name="INTEGRATES_WITH"),
    ]
    members = [EntityRow(uuid="e1", name="AWS Backup", type="Product", summary="s", degree=2)]
    ctx = assemble_context(members, facts, top_entities=5, token_budget=1000)
    assert ctx.fact_texts == {
        "u1": "AWS Backup provides continuous backups for Aurora.",
        "u2": "AWS Backup integrates with AWS KMS.",
    }
    assert set(ctx.fact_texts) == ctx.fact_uuids


def test_fact_texts_excludes_facts_dropped_by_the_budget():
    """A fact cut for budget is not citable, so it must not appear in fact_texts."""
    from theme_builder.context import EntityRow, FactRow, assemble_context
    facts = [FactRow(uuid=f"u{i}", fact="x" * 200, valid_at=None, invalid_at=None,
                     name="N") for i in range(10)]
    members = [EntityRow(uuid="e1", name="E", type="Product", summary="s", degree=1)]
    ctx = assemble_context(members, facts, top_entities=1, token_budget=100)
    assert set(ctx.fact_texts) == ctx.fact_uuids
    assert len(ctx.fact_texts) < 10


def _edge(u, src, dst, valid="2020"):
    return FactRow(uuid=u, fact="F" + u, valid_at=valid, invalid_at=None, name="R",
                   source_uuid=src, target_uuid=dst)


def test_a_hub_community_budget_goes_to_the_top_entities_facts():
    """2026-10-11: a 7,300-fact hub community kept its 42 most RECENT facts and none of
    the 82 about `AWS Backup Vault Lock`, one of its top-30 entities -- the report
    never mentioned Vault Lock. Facts touching top entities must come first."""
    members = [_ent("hub", 50), _ent("minor", 1)]
    noise = [_edge(f"n{i}", "minor", "x", valid="2026") for i in range(50)]   # newer
    about_hub = [_edge(f"h{i}", "hub", "y", valid="2020") for i in range(3)]
    res = assemble_context(members, noise + about_hub, top_entities=1, token_budget=60)
    assert {"h0", "h1", "h2"} <= res.fact_uuids
    assert len(res.fact_uuids) < len(noise)          # the budget did bind
    hub_end = max(res.text.index(f"[h{i}]") for i in range(3))
    assert all(res.text.index(f"[{u}]") > hub_end
               for u in res.fact_uuids if u.startswith("n"))


def test_top_entities_take_turns_so_one_hub_cannot_fill_the_budget():
    members = [_ent("a", 90), _ent("b", 80), _ent("c", 70)]
    facts = ([_edge(f"a{i}", "a", "z") for i in range(40)]
             + [_edge("b0", "b", "z"), _edge("c0", "z", "c")])
    res = assemble_context(members, facts, top_entities=3, token_budget=40)
    assert {"b0", "c0"} <= res.fact_uuids
    # each top entity's best fact comes before any entity's second
    assert res.text.index("[c0]") < res.text.index("[a1]")


def test_a_fact_between_two_top_entities_appears_once():
    members = [_ent("a", 9), _ent("b", 8)]
    res = assemble_context(members, [_edge("ab", "a", "b")], top_entities=2,
                           token_budget=1000)
    assert res.text.count("[ab]") == 1


def test_leftover_budget_is_filled_by_the_other_facts_in_recency_order():
    members = [_ent("a", 9)]
    facts = [_edge("a0", "a", "z"), _edge("o1", "p", "q", valid="2019"),
             _edge("o2", "p", "q", valid="2024")]
    res = assemble_context(members, facts, top_entities=1, token_budget=100000)
    order = [res.text.index(f"[{u}]") for u in ("a0", "o2", "o1")]
    assert order == sorted(order)


def test_context_budget_grows_with_facts_between_base_and_cap():
    from theme_builder.context import context_budget
    assert context_budget(10, base=12000, cap=48000) == 12000
    assert context_budget(300, base=12000, cap=48000) == 24000
    assert context_budget(7300, base=12000, cap=48000) == 48000
