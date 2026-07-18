import pytest
from datetime import datetime, timezone

pytestmark = pytest.mark.asyncio(loop_scope="module")

G = "backup-docs"


def _entry(cid, members, ga, parent=None, level=1, emb=None):
    return {"community_id": cid, "level": level, "member_uuids": list(members),
            "parent_id": parent, "title": f"T-{cid}", "summary": "S", "full_report": "[]",
            "rating": 5.0, "rating_explanation": "", "tags": [], "cited_fact_uuids": [],
            "embedding": emb or [0.1], "generated_at": ga}


async def test_incremental_writeback_reuse_dirty_and_dissolve(extract_driver):
    from theme_builder.writeback import write_communities_incremental
    old = datetime(2026, 3, 1, tzinfo=timezone.utc)
    new = datetime(2026, 4, 1, tzinfo=timezone.utc)
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        # a stale community that will be DISSOLVED (absent from entries)
        await s.run("CREATE (:Community {group_id:$g, community_id:'gone', level:1})", g=G)
        await s.run("CREATE (:Entity {group_id:$g, uuid:'m1'}), (:Entity {group_id:$g, uuid:'m2'})", g=G)
    entries = [
        _entry("clean1", ["m1"], old, emb=[9.0]),          # reused: old ga + old emb
        _entry("dirty1", ["m2"], new, emb=[7.0], parent="clean1"),
    ]
    res = await write_communities_incremental(extract_driver, G, entries, corpus_cursor="2026-04-01T00:00:00Z")
    assert res["reports_written"] == 2
    async with extract_driver.session() as s:
        r = await s.run("MATCH (c:Community {group_id:$g}) RETURN c.community_id AS cid, "
                        "toString(c.generated_at) AS ga, c.embedding AS emb, "
                        "toString(c.corpus_cursor) AS cur ORDER BY cid", g=G)
        rows = {x["cid"]: x async for x in r}
        assert set(rows) == {"clean1", "dirty1"}           # 'gone' dissolved
        assert rows["clean1"]["ga"].startswith("2026-03-01")   # reused old generated_at
        assert rows["clean1"]["emb"] == [9.0]
        assert rows["dirty1"]["ga"].startswith("2026-04-01")   # fresh generated_at
        assert rows["clean1"]["cur"].startswith("2026-04-01")  # run watermark on all
        # IN_COMMUNITY + PARENT_OF rebuilt
        r = await s.run("MATCH (:Entity {uuid:'m1'})-[:IN_COMMUNITY]->(c) RETURN c.community_id AS cid", )
        assert (await r.single())["cid"] == "clean1"
        r = await s.run("MATCH (:Community {community_id:'clean1'})-[:PARENT_OF]->(c) RETURN c.community_id AS cid")
        assert (await r.single())["cid"] == "dirty1"
