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
    async def _fake_report(client, model, context):
        captured["ctx"] = context
        return CommunityReport("T", "S", "[]", 5.0, "", ["AWS"], list(context.fact_uuids))
    monkeypatch.setattr(tc, "generate_report", _fake_report)

    class _Closeable:               # _run_theme_build closes the report client
        async def close(self): pass
    monkeypatch.setattr(tc, "_report_client_and_model", lambda s: (_Closeable(), "m"))

    class _FakeEmb:
        async def create_batch(self, texts): return [[0.0] for _ in texts]
    monkeypatch.setattr(tc, "build_embedder", lambda s: _FakeEmb())

    s = get_extract_settings.__wrapped__()
    s = s.model_copy(update=dict(group_id=g))
    res = await tc._run_theme_build(s, driver=extract_driver)
    assert res["reports_written"] == 1
    assert "f1" in captured["ctx"].fact_uuids            # the real fact reached the context
    async with extract_driver.session() as s2:
        c = (await (await s2.run(
            "MATCH (:Entity {uuid:'e1'})-[:IN_COMMUNITY]->(c:Community {community_id:'c1'}) "
            "RETURN c.cited_fact_uuids AS cited")).single())
        assert c["cited"] == ["f1"]
