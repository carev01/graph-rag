import pytest
pytestmark = pytest.mark.asyncio(loop_scope="module")


def _ingest_driver(extract_driver):
    from graph_extract.ingest_driver import IngestDriver
    # These tests only exercise Cypher-only helpers (`tombstone_article_episodes`,
    # `_supersede_trailing_episodes`) which touch only `self._driver` -- no
    # Graphiti/DocExtractor/Provenance calls happen, so stub those deps.
    return IngestDriver(None, None, None, None, None, extract_driver)


async def test_tombstone_marks_removed_not_deleted(extract_driver):
    driver = _ingest_driver(extract_driver)
    async with extract_driver.session() as s:
        await s.run("""
        CREATE (a:Article {id:'a1'})
        CREATE (a)-[:HAS_EPISODE {chunk_index:0}]->(:Episodic {uuid:'e_t1'})
        CREATE (a)-[:HAS_EPISODE {chunk_index:1}]->(:Episodic {uuid:'e_t2'})
        """)
    n = await driver.tombstone_article_episodes("a1")
    assert n == 2
    async with extract_driver.session() as s:
        recs = [rec async for rec in await s.run(
            "MATCH (:Article {id:'a1'})-[:HAS_EPISODE]->(e:Episodic) "
            "RETURN e.uuid AS uuid, e.removed AS removed")]
    assert len(recs) == 2
    assert all(rec["removed"] is True for rec in recs)


async def test_shrink_supersedes_trailing(extract_driver):
    driver = _ingest_driver(extract_driver)
    async with extract_driver.session() as s:
        await s.run("""
        CREATE (a:Article {id:'a2'})
        CREATE (a)-[:HAS_EPISODE {chunk_index:0}]->(:Episodic {uuid:'e_s0'})
        CREATE (a)-[:HAS_EPISODE {chunk_index:1}]->(:Episodic {uuid:'e_s1'})
        CREATE (a)-[:HAS_EPISODE {chunk_index:2}]->(:Episodic {uuid:'e_s2'})
        """)
    await driver._supersede_trailing_episodes("a2", 1)
    async with extract_driver.session() as s:
        rec0 = await (await s.run(
            "MATCH (e:Episodic {uuid:'e_s0'}) RETURN e.superseded AS s")).single()
        rec1 = await (await s.run(
            "MATCH (e:Episodic {uuid:'e_s1'}) RETURN e.superseded AS s")).single()
        rec2 = await (await s.run(
            "MATCH (e:Episodic {uuid:'e_s2'}) RETURN e.superseded AS s")).single()
    assert rec0["s"] is not True
    assert rec1["s"] is True
    assert rec2["s"] is True
