import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_prune_removes_noise_keeps_real(extract_driver):
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {name:'arn:aws:ec2:us-east-1::snapshot/snap-1', group_id:'g'})")
        await s.run("CREATE (:Entity {name:'CreatorRequestId', group_id:'g'})")
        await s.run("CREATE (:Entity {name:'immutability', group_id:'g'})")
        await s.run("CREATE (:Entity {name:'Amazon S3', group_id:'g'})")
        # a fact edge on a noise node -> must go with it
        await s.run("MATCH (a:Entity {name:'arn:aws:ec2:us-east-1::snapshot/snap-1'}), "
                    "(b:Entity {name:'immutability'}) "
                    "CREATE (a)-[:RELATES_TO {group_id:'g'}]->(b)")
    from graph_extract.graph_cleanup import prune_noise_entities
    res = await prune_noise_entities(extract_driver, "g")
    assert res["pruned"] == 2
    async with extract_driver.session() as s:
        r = await s.run("MATCH (e:Entity {group_id:'g'}) RETURN count(e) AS n")
        assert (await r.single())["n"] == 2                 # only the 2 real kept
        r = await s.run("MATCH ()-[x:RELATES_TO {group_id:'g'}]->() RETURN count(x) AS n")
        assert (await r.single())["n"] == 0                 # noise fact gone
