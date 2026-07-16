import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_tombstone_navigation_articles(extract_driver):
    from graph_extract.graph_cleanup import tombstone_navigation_articles
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (a:Article {id:'nav1', title:'Archived release notes'}) "
                    "CREATE (a)-[:HAS_EPISODE]->(:Episodic {uuid:'ne1', group_id:$g}) "
                    "CREATE (a)-[:HAS_EPISODE]->(:Episodic {uuid:'ne2', group_id:$g})", g=g)
        await s.run("CREATE (b:Article {id:'real1', title:'Vault Lock'}) "
                    "CREATE (b)-[:HAS_EPISODE]->(:Episodic {uuid:'re1', group_id:$g})", g=g)
    res = await tombstone_navigation_articles(extract_driver, g)
    assert res["tombstoned_articles"] == 1 and res["tombstoned_episodes"] == 2
    assert res["titles"] == ["Archived release notes"]
    async with extract_driver.session() as s:
        removed = {r["u"]: r["rm"] async for r in await s.run(
            "MATCH (:Article)-[:HAS_EPISODE]->(e:Episodic) RETURN e.uuid AS u, e.removed AS rm")}
    assert removed["ne1"] is True and removed["ne2"] is True and removed.get("re1") is not True


async def test_nav_tombstone_then_sweep_expires_nav_only_facts(extract_driver):
    """Guards the maintenance composition crux: nav-tombstone BEFORE sweep, run
    in sequence, expires a fact supported only by a nav article's episodes while
    keeping a fact co-supported by a real article's episode."""
    from graph_extract.graph_cleanup import tombstone_navigation_articles
    from graph_extract.staleness_sweep import sweep_stale_facts
    g = "backup-docs"
    async with extract_driver.session() as s:
        # nav article + its episode; a real article + its episode
        await s.run("CREATE (a:Article {id:'navX', title:'Blogs, videos, tutorials, and other resources'}) "
                    "CREATE (a)-[:HAS_EPISODE]->(:Episodic {uuid:'nx1', group_id:$g})", g=g)
        await s.run("CREATE (b:Article {id:'realX', title:'Encryption in Azure Backup'}) "
                    "CREATE (b)-[:HAS_EPISODE]->(:Episodic {uuid:'rx1', group_id:$g})", g=g)
        # f_junk: supported ONLY by the nav episode -> should expire
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f_junk', episodes:['nx1']}]->(y:Entity)", g=g)
        # f_co: supported by BOTH nav and real episodes -> must survive
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f_co', episodes:['nx1','rx1']}]->(y:Entity)", g=g)
    # the maintenance order: nav-tombstone THEN sweep
    await tombstone_navigation_articles(extract_driver, g)
    await sweep_stale_facts(extract_driver, g)
    async with extract_driver.session() as s:
        inv = {r["u"]: r["i"] async for r in await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN ['f_junk','f_co'] "
            "RETURN f.uuid AS u, f.invalid_at AS i", g=g)}
    assert inv["f_junk"] is not None   # nav-only fact expired by the sweep
    assert inv["f_co"] is None         # co-supported by a real (live) episode -> kept
