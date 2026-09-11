from datetime import datetime

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

G = "backup-docs"


class _FakeEmbedder:
    class _Client:
        async def close(self):
            pass
    def __init__(self):
        self.client = _FakeEmbedder._Client()
    async def create_batch(self, texts):
        return [[0.5] for _ in texts]


class _FakeClient:
    async def close(self):
        pass


def _settings():
    from graph_extract.config import ExtractSettings
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


async def _seed_persisted_and_entities(driver, *, e_a_created):
    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        # persisted A {e1,e2}, B {e3,e4}, corpus_cursor T0
        for cid, members in (("sA", ["e1", "e2"]), ("sB", ["e3", "e4"])):
            await s.run("CREATE (c:Community {group_id:$g, community_id:$cid, level:1, title:'old', "
                        "summary:'old', full_report:'[]', rating:1.0, rating_explanation:'', tags:[], "
                        "cited_fact_uuids:[], embedding:[0.0], generated_at: datetime('2026-03-01'), "
                        "corpus_cursor:'2026-03-01T00:00:00Z', member_count:2})", g=G, cid=cid)
            for mu in members:
                await s.run("MATCH (c:Community {community_id:$cid, group_id:$g}) "
                            "MERGE (e:Entity {uuid:$mu, group_id:$g}) MERGE (e)-[:IN_COMMUNITY]->(c)",
                            cid=cid, g=G, mu=mu)
        # entity created_at: e1 varies (touch A or not), others old; plus an Episodic for _corpus_cursor
        await s.run("MATCH (e:Entity {group_id:$g}) SET e.created_at = datetime('2026-01-01')", g=G)
        await s.run("MATCH (e:Entity {uuid:'e1', group_id:$g}) SET e.created_at = datetime($t)", g=G, t=e_a_created)
        await s.run("CREATE (:Episodic {group_id:$g, uuid:'ep1', created_at: datetime('2026-04-01')})", g=G)


class _FrozenDatetime(datetime):
    """Pins cli._run_theme_build_incremental's `now` to 2026-04-15 so the
    generated_at assertions below are independent of the real wall clock."""
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 4, 15, tzinfo=tz)


def _patch(monkeypatch):
    import theme_builder.cli as cli
    from theme_builder.detect import Community
    from theme_builder.report import CommunityReport
    calls = {"n": 0}

    async def _fake_detect(driver, group_id, *, min_community_size, max_levels):
        return [Community("hA", 1, ["e1", "e2"], None), Community("hB", 1, ["e3", "e4"], None)]

    async def _fake_generate(client, model, ctx, max_tokens, *, verifier=None, stats=None):
        calls["n"] += 1
        return CommunityReport(title="NEW", summary="NEW", full_report="[]", rating=9.0,
                               rating_explanation="", tags=[], cited_fact_uuids=[])

    monkeypatch.setattr(cli, "detect_communities", _fake_detect)
    monkeypatch.setattr(cli, "generate_report", _fake_generate)
    monkeypatch.setattr(cli, "build_embedder", lambda s: _FakeEmbedder())
    monkeypatch.setattr(cli, "_report_client_and_model", lambda s: (_FakeClient(), "m"))
    monkeypatch.setattr(cli, "_verify_client_and_model", lambda s: (_FakeClient(), "vm"))
    # context assembly is irrelevant (generate_report is faked); stub it so the
    # real _fetch_members/assemble_context path can't crash on unnamed seed entities.
    monkeypatch.setattr(cli, "assemble_context", lambda *a, **k: None)
    monkeypatch.setattr(cli, "datetime", _FrozenDatetime)
    return calls


async def test_incremental_regenerates_only_touched(extract_driver, monkeypatch):
    import theme_builder.cli as cli
    calls = _patch(monkeypatch)
    await _seed_persisted_and_entities(extract_driver, e_a_created="2026-04-15")   # e1 touched (> T0)
    res = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)
    assert calls["n"] == 1                    # only community A regenerated
    assert res["reports_regenerated"] == 1 and res["reports_reused"] == 1
    async with extract_driver.session() as s:
        r = await s.run("MATCH (c:Community {group_id:$g, community_id:'sA'}) RETURN c.title AS t, toString(c.generated_at) AS ga", g=G)
        a = await r.single()
        assert a["t"] == "NEW" and a["ga"].startswith("2026-04")     # A regenerated, ga advanced
        r = await s.run("MATCH (c:Community {group_id:$g, community_id:'sB'}) RETURN c.title AS t, toString(c.generated_at) AS ga", g=G)
        b = await r.single()
        assert b["t"] == "old" and b["ga"].startswith("2026-03")     # B reused, ga unchanged (stable id kept)


async def test_incremental_no_change_regenerates_nothing(extract_driver, monkeypatch):
    import theme_builder.cli as cli
    calls = _patch(monkeypatch)
    await _seed_persisted_and_entities(extract_driver, e_a_created="2026-01-01")   # nothing after T0
    res = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)
    assert calls["n"] == 0                    # THE headline: zero LLM report calls
    assert res["reports_reused"] == 2 and res["reports_regenerated"] == 0


async def test_corpus_cursor_covers_entity_and_fact_created_at(extract_driver):
    # Regression: _corpus_cursor must be the max created_at across Episodic,
    # Entity, AND RELATES_TO. Graphiti writes entities/facts slightly after their
    # episode, so an Episodic-only max would leave them "after the cursor" and mark
    # every community dirty on an unchanged graph.
    import theme_builder.cli as cli
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (:Episodic {group_id:$g, uuid:'ep1', created_at: datetime('2026-01-01T00:00:00Z')})", g=G)
        await s.run("CREATE (:Entity {group_id:$g, uuid:'e1', created_at: datetime('2026-01-02T00:00:00Z')})", g=G)
        await s.run("CREATE (a:Entity {group_id:$g, uuid:'e2'})-[:RELATES_TO {group_id:$g, uuid:'f1', "
                    "created_at: datetime('2026-01-03T00:00:00Z')}]->(b:Entity {group_id:$g, uuid:'e3'})", g=G)
    cur = await cli._corpus_cursor(extract_driver, G)
    assert cur.startswith("2026-01-03")     # the fact's created_at, not the episode's


async def test_corpus_cursor_includes_sweep_invalid_at(extract_driver):
    import theme_builder.cli as cli
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (:Episodic {group_id:$g, uuid:'ep', created_at: datetime('2026-01-01')})", g=G)
        await s.run("CREATE (a:Entity {group_id:$g, uuid:'e1', created_at: datetime('2026-01-02')})"
                    "-[:RELATES_TO {group_id:$g, uuid:'f1', created_at: datetime('2026-01-03'), "
                    "expired_by_sweep: true, invalid_at: datetime('2026-05-01')}]->"
                    "(b:Entity {group_id:$g, uuid:'e2', created_at: datetime('2026-01-02')})", g=G)
    cur = await cli._corpus_cursor(extract_driver, G)
    assert cur.startswith("2026-05-01")     # sweep invalid_at dominates the created_ats


async def test_incremental_stages_an_unverifiable_report_instead_of_destroying_it(
        extract_driver, monkeypatch):
    """CRITICAL 1: incremental is the DEFAULT path. A verifier blip on a dirty
    community must STAGE the regenerated report, not drop the community out of
    `entries` -- write_communities_incremental's DETACH DELETE would then take the
    previously verified report with it. Measured blip rate: 5-8 of 41 per run."""
    import theme_builder.cli as cli
    from theme_builder.detect import Community
    from theme_builder.report import CommunityReport

    async def _fake_detect(driver, group_id, *, min_community_size, max_levels):
        return [Community("hA", 1, ["e1", "e2"], None), Community("hB", 1, ["e3", "e4"], None)]

    async def _fake_generate(client, model, ctx, max_tokens, *, verifier=None, stats=None):
        # generation succeeded; the verifier endpoint then blipped
        assert stats is not None
        stats.unverified = True
        stats.staged_report = CommunityReport(
            title="Staged Title", summary="Staged Summary",
            full_report='[{"finding": "F1", "fact_ids": ["f1"]}]', rating=5.0,
            rating_explanation="re", tags=["aws"], cited_fact_uuids=["f1"])
        return None

    monkeypatch.setattr(cli, "detect_communities", _fake_detect)
    monkeypatch.setattr(cli, "generate_report", _fake_generate)
    monkeypatch.setattr(cli, "build_embedder", lambda s: _FakeEmbedder())
    monkeypatch.setattr(cli, "_report_client_and_model", lambda s: (_FakeClient(), "m"))
    monkeypatch.setattr(cli, "_verify_client_and_model", lambda s: (_FakeClient(), "vm"))
    monkeypatch.setattr(cli, "assemble_context", lambda *a, **k: None)
    monkeypatch.setattr(cli, "datetime", _FrozenDatetime)

    await _seed_persisted_and_entities(extract_driver, e_a_created="2026-04-15")  # A dirty
    res = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)

    assert res["reports_unverified"] == 1
    assert res["reports_staged"] == 1, "the staged report must be surfaced"
    assert res["reports_regenerated"] == 0 and res["reports_reused"] == 1

    async with extract_driver.session() as s:
        rows = {r["cid"]: dict(r) async for r in await s.run(
            "MATCH (c:Community {group_id:$g}) RETURN c.community_id AS cid, "
            "c.title AS title, c.summary AS summary, c.full_report AS full_report, "
            "c.pending_summary AS ps, c.pending_full_report AS pfr, "
            "c.embedding AS embedding, c.verified AS verified, "
            "c.cited_fact_uuids AS cited", g=G)}

    assert set(rows) == {"sA", "sB"}, "the staged community must survive the rebuild"
    a = rows["sA"]
    assert a["ps"] == "Staged Summary"
    assert a["pfr"] == '[{"finding": "F1", "fact_ids": ["f1"]}]'
    assert a["title"] == "Staged Title"
    assert a["verified"] is False
    assert a["embedding"] is None, "a staged report must carry NO embedding"
    assert a["summary"] is None and a["full_report"] is None
    assert a["cited"] == ["f1"]

    b = rows["sB"]
    assert b["verified"] is True, "write_communities_incremental must write `verified`"
    assert b["embedding"] == [0.0]

    from answer_api.global_search import shortlist_communities

    class _QueryEmb:
        async def create_batch(self, texts):
            return [[0.0] for _ in texts]

    hits = await shortlist_communities(extract_driver, _QueryEmb(), "anything",
                                       level=1, k=10, group_id=G)
    assert all(h.community_id != "sA" for h in hits)


async def test_swept_community_regenerates_once(extract_driver, monkeypatch):
    import theme_builder.cli as cli
    calls = _patch(monkeypatch)
    # persisted A {e1,e2}, B {e3,e4}, corpus_cursor 2026-03-01; nothing created after it
    await _seed_persisted_and_entities(extract_driver, e_a_created="2026-01-01")
    # a fact among A's members is silently swept AFTER the community's corpus_cursor
    async with extract_driver.session() as s:
        await s.run("MATCH (a:Entity {uuid:'e1', group_id:$g}), (b:Entity {uuid:'e2', group_id:$g}) "
                    "CREATE (a)-[:RELATES_TO {group_id:$g, uuid:'fx', created_at: datetime('2026-01-01'), "
                    "expired_by_sweep: true, invalid_at: datetime('2026-06-01')}]->(b)", g=G)
    r1 = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)
    assert calls["n"] == 1                       # only community A regenerated (its fact was swept)
    assert r1["reports_regenerated"] == 1 and r1["reports_reused"] == 1
    # second run: the watermark now covers the sweep invalid_at -> A clean -> zero regen
    r2 = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)
    assert calls["n"] == 1                       # unchanged: no further regeneration
    assert r2["reports_regenerated"] == 0 and r2["reports_reused"] == 2


def _patch_failing_generation(monkeypatch, communities):
    """Report generation fails outright: no report AND nothing staged. This is the
    `except Exception` route and the `_generate_once` -> None route (measured 3/29
    empty-`choices` replies on a real theme-build), neither of which staging
    covers."""
    import theme_builder.cli as cli

    async def _fake_detect(driver, group_id, *, min_community_size, max_levels):
        return communities

    async def _fake_generate(client, model, ctx, max_tokens, *, verifier=None, stats=None):
        return None

    monkeypatch.setattr(cli, "detect_communities", _fake_detect)
    monkeypatch.setattr(cli, "generate_report", _fake_generate)
    monkeypatch.setattr(cli, "build_embedder", lambda s: _FakeEmbedder())
    monkeypatch.setattr(cli, "_report_client_and_model", lambda s: (_FakeClient(), "m"))
    monkeypatch.setattr(cli, "_verify_client_and_model", lambda s: (_FakeClient(), "vm"))
    monkeypatch.setattr(cli, "assemble_context", lambda *a, **k: None)
    monkeypatch.setattr(cli, "datetime", _FrozenDatetime)


async def test_a_failed_regeneration_preserves_the_previously_verified_report(
        extract_driver, monkeypatch):
    """BACKLOG 5c. Community A is dirty, its regeneration returns None, and
    nothing is staged. Before this fix A was omitted from `entries` and
    write_communities_incremental's DETACH DELETE removed the node -- taking a
    report that had been generated, verified and paid for. Nothing verified
    should leave retrieval because a NEW attempt failed."""
    import theme_builder.cli as cli
    from theme_builder.detect import Community
    _patch_failing_generation(monkeypatch, [Community("hA", 1, ["e1", "e2"], None),
                                            Community("hB", 1, ["e3", "e4"], None)])
    await _seed_persisted_and_entities(extract_driver, e_a_created="2026-04-15")  # A dirty

    res = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)

    assert res["reports_preserved"] == 1
    assert res["reports_skipped"] == 0, "nothing was lost, so nothing may be counted lost"
    assert res["lost_by_level"] == {}

    async with extract_driver.session() as s:
        r = await s.run(
            "MATCH (c:Community {group_id:$g, community_id:'sA'}) RETURN c.summary AS summary, "
            "c.embedding AS embedding, c.verified AS verified, c.stale AS stale, "
            "toString(c.generated_at) AS ga", g=G)
        a = await r.single()
    assert a is not None, "the community must survive the rebuild"
    assert a["summary"] == "old", "its verified report text is intact"
    assert a["embedding"] == [0.0], "it stays RETRIEVABLE -- a live report, not a staged one"
    assert a["verified"] is True, "it was verified; the new attempt failing does not unverify it"
    assert a["stale"] is True, "but it is known-stale, so the next run must retry it"
    assert a["ga"].startswith("2026-03"), "generated_at is the old report's, not this run's"


async def test_a_preserved_report_is_regenerated_on_the_next_run(extract_driver, monkeypatch):
    """The other half of 5c: carrying the report over must not make it look fresh.
    The run stamps a new corpus_cursor, so next run's `touched` no longer contains
    the edits that made it dirty -- only the `stale` flag forces the retry."""
    import theme_builder.cli as cli
    from theme_builder.detect import Community
    comms = [Community("hA", 1, ["e1", "e2"], None), Community("hB", 1, ["e3", "e4"], None)]
    _patch_failing_generation(monkeypatch, comms)
    await _seed_persisted_and_entities(extract_driver, e_a_created="2026-04-15")
    await cli._run_theme_build_incremental(_settings(), driver=extract_driver)

    calls = _patch(monkeypatch)          # generation works again; same hA/hB communities
    res = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)
    assert calls["n"] == 1, "the preserved community must be retried, not treated as clean"
    assert res["reports_regenerated"] == 1

    async with extract_driver.session() as s:
        r = await s.run("MATCH (c:Community {group_id:$g, community_id:'sA'}) "
                        "RETURN c.summary AS summary, c.stale AS stale", g=G)
        a = await r.single()
    assert a["summary"] == "NEW"
    assert not a["stale"], "a freshly regenerated report is not stale"


async def test_a_failed_regeneration_with_nothing_persisted_is_counted_as_lost(
        extract_driver, monkeypatch):
    """BACKLOG 5d. A brand-new community whose first report fails has nothing to
    preserve -- that IS a loss, and the incremental path could not report it,
    because `entries` only carries survivors. Incremental is the DEFAULT path, so
    the blind spot sat exactly where routine runs happen."""
    import theme_builder.cli as cli
    from theme_builder.detect import Community
    _patch_failing_generation(monkeypatch, [Community("hNEW", 2, ["e9"], None)])
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (:Entity {group_id:$g, uuid:'e9', created_at: datetime('2026-01-01')})", g=G)
        await s.run("CREATE (:Episodic {group_id:$g, uuid:'ep1', created_at: datetime('2026-04-01')})", g=G)

    res = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)

    assert res["reports_skipped"] == 1
    assert res["reports_preserved"] == 0
    assert res["lost_by_level"] == {2: 1}, "the level of the lost community must be visible"


async def test_a_failed_regeneration_keeps_a_staged_report_staged(extract_driver, monkeypatch):
    """A community staged by an earlier run is ALWAYS dirty, so it is retried; if
    that retry fails outright it must be carried over in its STAGED shape. Writing
    it as a normal entry would publish an unverified report (load_persisted reads
    `summary` as '' for a staged row, so it would also be published empty)."""
    import theme_builder.cli as cli
    from theme_builder.detect import Community
    _patch_failing_generation(monkeypatch, [Community("hA", 1, ["e1", "e2"], None)])
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (c:Community {group_id:$g, community_id:'sA', level:1, title:'T', "
                    "pending_summary:'PS', pending_full_report:'[{\"finding\":\"F\"}]', "
                    "rating:5.0, rating_explanation:'re', tags:['aws'], cited_fact_uuids:['f1'], "
                    "verified:false, generated_at: datetime('2026-03-01'), member_count:2})", g=G)
        for mu in ("e1", "e2"):
            await s.run("MATCH (c:Community {community_id:'sA', group_id:$g}) "
                        "MERGE (e:Entity {uuid:$mu, group_id:$g, created_at: datetime('2026-01-01')}) "
                        "MERGE (e)-[:IN_COMMUNITY]->(c)", g=G, mu=mu)
        await s.run("CREATE (:Episodic {group_id:$g, uuid:'ep1', created_at: datetime('2026-04-01')})", g=G)

    res = await cli._run_theme_build_incremental(_settings(), driver=extract_driver)

    assert res["reports_staged"] == 1
    assert res["reports_skipped"] == 0 and res["lost_by_level"] == {}

    async with extract_driver.session() as s:
        r = await s.run("MATCH (c:Community {group_id:$g, community_id:'sA'}) "
                        "RETURN c.pending_summary AS ps, c.pending_full_report AS pfr, "
                        "c.summary AS summary, c.embedding AS embedding, "
                        "c.verified AS verified", g=G)
        a = await r.single()
    assert a is not None, "the staged community must survive the rebuild"
    assert a["ps"] == "PS" and a["pfr"] == '[{"finding":"F"}]', "its pending text is intact"
    assert a["summary"] is None and a["embedding"] is None
    assert a["verified"] is False
