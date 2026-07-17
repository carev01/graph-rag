from datetime import datetime, timedelta, timezone

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
    dt = lambda y: datetime(y, 1, 1, tzinfo=timezone.utc)  # noqa: E731
    edges = [_E("f_cur", "current fact", dt(2021), None, ["ep1"]),
             _E("f_sup", "superseded fact", dt(2019), dt(2020), ["ep1"]),
             _E("f_exp", "expired fact", dt(2018), dt(2022), ["ep1"])]
    async def _fake_retrieve(*a, **k): return list(edges)
    monkeypatch.setattr(tl, "_retrieve_edges", _fake_retrieve)
    out = await tl.timeline_local(object(), extract_driver, q="x", limit=10, group_id=g)
    order = [r["fact_uuid"] for r in out["timeline"]]
    assert order == ["f_exp", "f_sup", "f_cur"]      # ascending valid_at 2018,2019,2021
    st = {r["fact_uuid"]: r["status"] for r in out["timeline"]}
    assert st == {"f_exp": "expired", "f_sup": "superseded", "f_cur": "current"}
    assert out["timeline"][0]["sources"][0]["article_id"] == "a1"


async def test_timeline_orders_by_utc_instant_across_offsets(extract_driver, monkeypatch):
    """valid_at is a datetime; ordering must be by the true UTC instant, not by
    the stringified local wall-clock. Edge A is at 08:00+05:00 (03:00 UTC) and
    edge B at 05:00+00:00 (05:00 UTC): A is the earlier instant and must sort
    first, even though 'str(A)' ("...08:00...") sorts AFTER 'str(B)' ("...05:00...").
    """
    import answer_api.timeline as tl
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (a:Article {id:'a2', source_url:'u', title:'t'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep2', group_id:$g})", g=g)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f_tz_a', episodes:['ep2']}]->(y:Entity)", g=g)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f_tz_b', episodes:['ep2']}]->(y:Entity)", g=g)
    a = datetime(2020, 1, 1, 8, 0, 0, tzinfo=timezone(timedelta(hours=5)))   # 03:00 UTC
    b = datetime(2020, 1, 1, 5, 0, 0, tzinfo=timezone.utc)                    # 05:00 UTC
    # Deliberately supply B first so a stable sort can't accidentally pass.
    edges = [_E("f_tz_b", "later instant", b, None, ["ep2"]),
             _E("f_tz_a", "earlier instant", a, None, ["ep2"])]
    async def _fake_retrieve(*ar, **kw): return list(edges)
    monkeypatch.setattr(tl, "_retrieve_edges", _fake_retrieve)
    out = await tl.timeline_local(object(), extract_driver, q="x", limit=10, group_id=g)
    order = [r["fact_uuid"] for r in out["timeline"]]
    assert order == ["f_tz_a", "f_tz_b"]   # earlier UTC instant first


async def test_timeline_naive_and_none_valid_at_sort(extract_driver, monkeypatch):
    """A naive (tz-unaware) valid_at must not raise when compared with aware ones
    (treated as UTC), and a None valid_at still sorts last."""
    import answer_api.timeline as tl
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (a:Article {id:'a3', source_url:'u', title:'t'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep3', group_id:$g})", g=g)
        for u in ("f_naive", "f_aware", "f_none"):
            await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:$u, episodes:['ep3']}]->(y:Entity)",
                        g=g, u=u)
    naive = datetime(2021, 6, 1, 0, 0, 0)                       # tz-naive -> UTC
    aware = datetime(2022, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
    edges = [_E("f_none", "no time", None, None, ["ep3"]),
             _E("f_aware", "aware 2022", aware, None, ["ep3"]),
             _E("f_naive", "naive 2021", naive, None, ["ep3"])]
    async def _fake_retrieve(*ar, **kw): return list(edges)
    monkeypatch.setattr(tl, "_retrieve_edges", _fake_retrieve)
    out = await tl.timeline_local(object(), extract_driver, q="x", limit=10, group_id=g)
    order = [r["fact_uuid"] for r in out["timeline"]]
    assert order == ["f_naive", "f_aware", "f_none"]   # 2021, 2022, then None last


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
