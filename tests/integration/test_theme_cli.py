import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_run_theme_build_end_to_end(extract_driver, monkeypatch):
    import theme_builder.cli as tc
    from theme_builder.detect import Community
    from theme_builder.report import CommunityReport
    from graph_extract.config import get_extract_settings
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {uuid:'e1', group_id:$g, name:'AWS Backup', summary:'svc'})", g=g)
        await s.run("CREATE (:Entity {uuid:'e2', group_id:$g, name:'Amazon S3', summary:'store'})", g=g)
        await s.run("MATCH (a:Entity {uuid:'e1'}),(b:Entity {uuid:'e2'}) "
                    "CREATE (a)-[:RELATES_TO {group_id:$g, uuid:'f1', fact:'AWS Backup supports Amazon S3', name:'Supports'}]->(b)", g=g)

    async def _fake_detect(driver, group_id, **k):
        return [Community("c1", 0, ["e1", "e2"], None)]
    monkeypatch.setattr(tc, "detect_communities", _fake_detect)

    captured = {}
    async def _fake_report(client, model, context, max_tokens, *, verifier=None, stats=None):
        captured["ctx"] = context
        captured["verifier"] = verifier
        return CommunityReport("T", "S", "[]", 5.0, "", ["AWS"], list(context.fact_uuids))
    monkeypatch.setattr(tc, "generate_report", _fake_report)

    class _Closeable:               # _run_theme_build closes the report client
        async def close(self): pass
    monkeypatch.setattr(tc, "_report_client_and_model", lambda s: (_Closeable(), "m"))
    monkeypatch.setattr(tc, "_verify_client_and_model", lambda s: (_Closeable(), "vm"))

    class _FakeEmb:                     # mirror OpenAIEmbedder: has .client to close
        class _C:
            async def close(self): pass
        client = _C()
        async def create_batch(self, texts): return [[0.0] for _ in texts]
    monkeypatch.setattr(tc, "build_embedder", lambda s: _FakeEmb())

    s = get_extract_settings.__wrapped__()
    s = s.model_copy(update=dict(group_id=g))
    res = await tc._run_theme_build(s, driver=extract_driver)
    assert res["reports_written"] == 1
    assert "f1" in captured["ctx"].fact_uuids            # the real fact reached the context
    assert captured["verifier"] is not None, "theme-build must verify its reports"
    assert res["findings_dropped"] == 0
    assert res["reports_reverified"] == 0
    assert res["reports_unverified"] == 0
    async with extract_driver.session() as s2:
        c = (await (await s2.run(
            "MATCH (:Entity {uuid:'e1'})-[:IN_COMMUNITY]->(c:Community {community_id:'c1'}) "
            "RETURN c.cited_fact_uuids AS cited")).single())
        assert c["cited"] == ["f1"]


async def test_one_community_error_does_not_abort_build(extract_driver, monkeypatch):
    import theme_builder.cli as tc
    from theme_builder.detect import Community
    from theme_builder.report import CommunityReport
    from graph_extract.config import get_extract_settings
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n {group_id:$g}) DETACH DELETE n", g=g)   # clean slate
        await s.run("CREATE (:Entity {uuid:'g1', group_id:$g, name:'Good', summary:'s'})", g=g)
        await s.run("CREATE (:Entity {uuid:'b1', group_id:$g, name:'Bad', summary:'s'})", g=g)

    async def _fake_detect(driver, group_id, **k):
        return [Community("good", 0, ["g1"], None), Community("bad", 0, ["b1"], None)]
    monkeypatch.setattr(tc, "detect_communities", _fake_detect)

    async def _fake_report(client, model, context, max_tokens, *, verifier=None, stats=None):
        # the "bad" community's context has no entities named Good -> raise for it
        if "Good" not in context.text:
            raise RuntimeError("simulated LLM 500")
        return CommunityReport("T", "S", "[]", 5.0, "", [], [])
    monkeypatch.setattr(tc, "generate_report", _fake_report)

    class _Closeable:
        async def close(self): pass
    monkeypatch.setattr(tc, "_report_client_and_model", lambda s: (_Closeable(), "m"))
    monkeypatch.setattr(tc, "_verify_client_and_model", lambda s: (_Closeable(), "vm"))

    class _FakeEmb:
        class _C:
            async def close(self): pass
        client = _C()
        async def create_batch(self, texts): return [[0.0] for _ in texts]
    monkeypatch.setattr(tc, "build_embedder", lambda s: _FakeEmb())

    s = get_extract_settings.__wrapped__().model_copy(update=dict(group_id=g))
    res = await tc._run_theme_build(s, driver=extract_driver)
    assert res["reports_written"] == 1 and res["reports_skipped"] == 1
    async with extract_driver.session() as s2:
        ids = [r["id"] async for r in await s2.run(
            "MATCH (c:Community {group_id:$g}) RETURN c.community_id AS id", g=g)]
    assert ids == ["good"]   # the good community was written despite the other erroring


async def test_unverifiable_report_is_staged_unreachable_by_answering(extract_driver, monkeypatch):
    """A community whose verifier could not complete is kept in the graph (not
    destroyed by the DETACH DELETE rebuild) but written with no embedding and
    verified=false, so it is unreachable from shortlist_communities -- and from
    DRIFT, which sources its ids from that same shortlist."""
    import theme_builder.cli as tc
    from theme_builder.detect import Community
    from graph_extract.config import get_extract_settings
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n {group_id:$g}) DETACH DELETE n", g=g)   # clean slate
        await s.run("CREATE (:Entity {uuid:'u1', group_id:$g, name:'Unverified', summary:'s'})", g=g)

    async def _fake_detect(driver, group_id, **k):
        return [Community("stagec", 0, ["u1"], None)]
    monkeypatch.setattr(tc, "detect_communities", _fake_detect)

    async def _fake_report(client, model, context, max_tokens, *, verifier=None, stats=None):
        # simulate: report generation succeeded, then the verifier endpoint failed
        assert stats is not None
        stats.unverified = True
        from theme_builder.report import CommunityReport
        stats.staged_report = CommunityReport(
            "Staged Title", "Staged Summary", "[]", 5.0, "", ["aws"], [])
        return None
    monkeypatch.setattr(tc, "generate_report", _fake_report)

    class _Closeable:
        async def close(self): pass
    monkeypatch.setattr(tc, "_report_client_and_model", lambda s: (_Closeable(), "m"))
    monkeypatch.setattr(tc, "_verify_client_and_model", lambda s: (_Closeable(), "vm"))

    class _FakeEmb:
        class _C:
            async def close(self): pass
        client = _C()
        async def create_batch(self, texts): return [[0.0] for _ in texts]
    monkeypatch.setattr(tc, "build_embedder", lambda s: _FakeEmb())

    s = get_extract_settings.__wrapped__().model_copy(update=dict(group_id=g))
    res = await tc._run_theme_build(s, driver=extract_driver)
    assert res["reports_written"] == 0
    assert res["reports_skipped"] == 1
    assert res["reports_staged"] == 1

    async with extract_driver.session() as s2:
        rec = await (await s2.run(
            "MATCH (c:Community {group_id:$g, community_id:'stagec'}) "
            "RETURN c.pending_full_report AS pfr, c.verified AS verified, "
            "c.embedding AS embedding", g=g)).single()
    assert rec["pfr"] == "[]"
    assert rec["verified"] is False
    assert rec["embedding"] is None

    from answer_api.global_search import shortlist_communities

    class _QueryEmb:
        async def create_batch(self, texts): return [[0.0] for _ in texts]

    hits = await shortlist_communities(extract_driver, _QueryEmb(), "anything",
                                       level=0, k=10, group_id=g)
    assert all(h.community_id != "stagec" for h in hits)


async def test_full_rebuild_preserves_a_report_whose_regeneration_fails(
        extract_driver, monkeypatch):
    """BACKLOG 5c, `--full` half. write_communities DETACH DELETEs the whole
    :Community layer and writes back only what it is handed, and _run_theme_build
    never loaded the persisted layer -- so one failed generation deleted a report
    that had been generated, verified and paid for. `--full` means "regenerate
    everything", not "destroy what we have if the regeneration fails"."""
    import theme_builder.cli as tc
    from theme_builder.detect import Community
    from graph_extract.config import get_extract_settings
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        for uuid in ("e1", "e2"):
            await s.run("CREATE (:Entity {uuid:$u, group_id:$g, name:$u, summary:'x', "
                        "created_at: datetime('2026-01-01')})", u=uuid, g=g)
        await s.run("CREATE (:Episodic {group_id:$g, uuid:'ep1', created_at: datetime('2026-01-01')})", g=g)
        # a previously verified, retrievable report for the community we will fail
        await s.run("CREATE (c:Community {group_id:$g, community_id:'c1', level:0, "
                    "title:'OLD T', summary:'OLD S', full_report:'[{\"finding\":\"old\"}]', "
                    "rating:7.0, rating_explanation:'re', tags:['aws'], "
                    "cited_fact_uuids:['f1'], embedding:[0.9], verified:true, "
                    "generated_at: datetime('2026-03-01'), member_count:2})", g=g)
        for uuid in ("e1", "e2"):
            await s.run("MATCH (c:Community {community_id:'c1', group_id:$g}), "
                        "(e:Entity {uuid:$u, group_id:$g}) MERGE (e)-[:IN_COMMUNITY]->(c)",
                        u=uuid, g=g)

    async def _fake_detect(driver, group_id, **k):
        return [Community("c1", 0, ["e1", "e2"], None)]

    async def _failing_report(client, model, context, max_tokens, *, verifier=None, stats=None):
        return None                      # generation failed; nothing staged

    class _Closeable:
        async def close(self): pass

    class _FakeEmb:
        class _C:
            async def close(self): pass
        client = _C()
        async def create_batch(self, texts): return [[0.0] for _ in texts]

    monkeypatch.setattr(tc, "detect_communities", _fake_detect)
    monkeypatch.setattr(tc, "generate_report", _failing_report)
    monkeypatch.setattr(tc, "_report_client_and_model", lambda s: (_Closeable(), "m"))
    monkeypatch.setattr(tc, "_verify_client_and_model", lambda s: (_Closeable(), "vm"))
    monkeypatch.setattr(tc, "build_embedder", lambda s: _FakeEmb())

    st = get_extract_settings.__wrapped__().model_copy(update=dict(group_id=g))
    res = await tc._run_theme_build(st, driver=extract_driver)

    assert res["reports_preserved"] == 1
    assert res["lost_by_level"] == {}, "nothing was lost, so nothing may be counted lost"

    async with extract_driver.session() as s:
        r = await s.run("MATCH (c:Community {group_id:$g, community_id:'c1'}) "
                        "RETURN c.summary AS summary, c.embedding AS embedding, "
                        "c.verified AS verified, c.stale AS stale, "
                        "toString(c.generated_at) AS ga, "
                        "size([(e)-[:IN_COMMUNITY]->(c) | e]) AS members", g=g)
        row = await r.single()
    assert row is not None, "the community must survive the DETACH DELETE rebuild"
    assert row["summary"] == "OLD S", "its verified report text is intact"
    assert row["embedding"] == [0.9], "it stays RETRIEVABLE, with its own embedding"
    assert row["verified"] is True
    assert row["stale"] is True, "but flagged so the next incremental run retries it"
    assert row["ga"].startswith("2026-03"), "the original report's age, not this run's"
    assert row["members"] == 2, "its IN_COMMUNITY edges were rewritten"


async def test_full_rebuild_reports_a_community_with_nothing_persisted_as_lost(
        extract_driver, monkeypatch):
    """The other side: a brand-new community whose first report fails has nothing
    to preserve. That IS a loss and must be counted, not quietly preserved."""
    import theme_builder.cli as tc
    from theme_builder.detect import Community
    from graph_extract.config import get_extract_settings
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (:Entity {uuid:'e9', group_id:$g, name:'n', summary:'x'})", g=g)
        await s.run("CREATE (:Episodic {group_id:$g, uuid:'ep1', created_at: datetime('2026-01-01')})", g=g)

    async def _fake_detect(driver, group_id, **k):
        return [Community("cNEW", 1, ["e9"], None)]

    async def _failing_report(client, model, context, max_tokens, *, verifier=None, stats=None):
        return None

    class _Closeable:
        async def close(self): pass

    class _FakeEmb:
        class _C:
            async def close(self): pass
        client = _C()
        async def create_batch(self, texts): return [[0.0] for _ in texts]

    monkeypatch.setattr(tc, "detect_communities", _fake_detect)
    monkeypatch.setattr(tc, "generate_report", _failing_report)
    monkeypatch.setattr(tc, "_report_client_and_model", lambda s: (_Closeable(), "m"))
    monkeypatch.setattr(tc, "_verify_client_and_model", lambda s: (_Closeable(), "vm"))
    monkeypatch.setattr(tc, "build_embedder", lambda s: _FakeEmb())

    st = get_extract_settings.__wrapped__().model_copy(update=dict(group_id=g))
    res = await tc._run_theme_build(st, driver=extract_driver)

    assert res["reports_preserved"] == 0
    assert res["lost_by_level"] == {1: 1}
