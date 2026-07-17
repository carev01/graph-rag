import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


class _E:
    def __init__(self, uuid, fact, valid_at, invalid_at, episodes):
        self.uuid, self.fact, self.valid_at, self.invalid_at, self.episodes = \
            uuid, fact, valid_at, invalid_at, episodes


async def test_timeline_orders_and_statuses(extract_driver, monkeypatch):
    import answer_api.timeline as tl
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (a:Article {id:'a1', source_url:'u', title:'t'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1', group_id:$g})", g=g)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f_exp', episodes:['ep1'], expired_by_sweep:true}]->(y:Entity)", g=g)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f_cur', episodes:['ep1']}]->(y:Entity)", g=g)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f_sup', episodes:['ep1']}]->(y:Entity)", g=g)
    edges = [_E("f_cur", "current fact", "2021", None, ["ep1"]),
             _E("f_sup", "superseded fact", "2019", "2020", ["ep1"]),
             _E("f_exp", "expired fact", "2018", "2022", ["ep1"])]
    async def _fake_retrieve(*a, **k): return list(edges)
    monkeypatch.setattr(tl, "_retrieve_edges", _fake_retrieve)
    out = await tl.timeline_local(object(), extract_driver, q="x", limit=10, group_id=g)
    order = [r["fact_uuid"] for r in out["timeline"]]
    assert order == ["f_exp", "f_sup", "f_cur"]      # ascending valid_at 2018,2019,2021
    st = {r["fact_uuid"]: r["status"] for r in out["timeline"]}
    assert st == {"f_exp": "expired", "f_sup": "superseded", "f_cur": "current"}
    assert out["timeline"][0]["sources"][0]["article_id"] == "a1"


@pytest.mark.live
async def test_timeline_local_live_smoke(live_extract_driver):
    """Live smoke against the real compose Neo4j/Graphiti (already-bootstrapped
    AWS Backup graph). Opt-in only: excluded from the default `-m "not live"` lane.
    """
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_graphiti
    from answer_api.timeline import timeline_local

    settings = get_extract_settings()
    graphiti = build_graphiti(settings)
    try:
        out = await timeline_local(
            graphiti, live_extract_driver, q="AWS Backup CloudTrail events",
            limit=30, group_id=settings.group_id)
        timeline = out["timeline"]
        assert timeline
        valid_ats = [r["valid_at"] for r in timeline]
        # ascending order: None sorts last, so compare adjacent non-None pairs
        for a, b in zip(valid_ats, valid_ats[1:]):
            if a is not None and b is not None:
                assert str(a) <= str(b)
        statuses = {r["status"] for r in timeline}
        assert statuses & {"superseded", "expired"}
        assert any(r["sources"] for r in timeline)
    finally:
        await graphiti.close()
