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
