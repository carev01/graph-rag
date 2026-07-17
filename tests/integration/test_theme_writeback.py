import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


class _FakeEmbedder:
    async def create_batch(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


async def test_writeback_creates_community_subgraph(extract_driver):
    from theme_builder.detect import Community
    from theme_builder.report import CommunityReport
    from theme_builder.writeback import write_communities
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {uuid:'e1', group_id:$g, name:'AWS Backup'})", g=g)
        await s.run("CREATE (:Entity {uuid:'e2', group_id:$g, name:'Amazon S3'})", g=g)
    comms = [
        Community("child1", 0, ["e1", "e2"], parent_id="par1"),
        Community("par1", 1, ["e1", "e2"], parent_id=None),
    ]
    reps = {
        "child1": CommunityReport("Leaf", "leaf sum", "[]", 6.0, "why", ["AWS"], ["f1"]),
        "par1": CommunityReport("Top", "top sum", "[]", 8.0, "why2", ["AWS"], ["f1", "f2"]),
    }
    res = await write_communities(extract_driver, _FakeEmbedder(), g, comms, reps, corpus_cursor="cur-1")
    assert res["reports_written"] == 2
    async with extract_driver.session() as s:
        n = (await (await s.run("MATCH (c:Community {group_id:$g}) RETURN count(c) AS c", g=g)).single())["c"]
        assert n == 2
        pf = (await (await s.run(
            "MATCH (:Community {community_id:'par1'})-[:PARENT_OF]->(:Community {community_id:'child1'}) "
            "RETURN count(*) AS c")).single())["c"]
        assert pf == 1
        im = (await (await s.run(
            "MATCH (:Entity {uuid:'e1'})-[:IN_COMMUNITY]->(:Community {community_id:'child1'}) "
            "RETURN count(*) AS c")).single())["c"]
        assert im == 1
        emb = (await (await s.run(
            "MATCH (c:Community {community_id:'par1'}) RETURN c.embedding AS e, c.corpus_cursor AS cur")).single())
        assert emb["e"] == [0.1, 0.2, 0.3] and emb["cur"] == "cur-1"


async def test_writeback_is_full_rebuild(extract_driver):
    from theme_builder.detect import Community
    from theme_builder.report import CommunityReport
    from theme_builder.writeback import write_communities
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {uuid:'x1', group_id:$g, name:'X'})", g=g)
        await s.run("CREATE (:Community {community_id:'stale', group_id:$g, level:0})", g=g)
    comms = [Community("fresh", 0, ["x1"], None)]
    reps = {"fresh": CommunityReport("F", "s", "[]", 1.0, "", [], [])}
    await write_communities(extract_driver, _FakeEmbedder(), g, comms, reps, corpus_cursor=None)
    async with extract_driver.session() as s:
        ids = [r["id"] async for r in await s.run(
            "MATCH (c:Community {group_id:$g}) RETURN c.community_id AS id", g=g)]
    assert ids == ["fresh"]                    # stale deleted, only fresh remains
