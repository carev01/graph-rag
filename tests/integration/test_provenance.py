import pytest
pytestmark = pytest.mark.asyncio(loop_scope="module")

async def test_link_and_idempotency(extract_provenance, extract_driver):
    async with extract_driver.session() as s:
        await s.run("MERGE (a:Article {id:'a1', source_id:'s1', source_url:'https://u', title:'T'})")
        await s.run("CREATE (e:Episodic {uuid:'ep-1'})")
    p = extract_provenance
    assert await p.already_ingested("a1", 0, "hash1234") is False
    await p.link("a1", "ep-1", chunk_index=0, heading_path="C",
                 token_count=100, content_hash="hash1234")
    assert await p.already_ingested("a1", 0, "hash1234") is True
    # second link is a no-op (no duplicate edge)
    await p.link("a1", "ep-1", chunk_index=0, heading_path="C",
                 token_count=100, content_hash="hash1234")
    async with extract_driver.session() as s:
        r = await s.run("MATCH (:Article {id:'a1'})-[r:HAS_EPISODE]->() RETURN count(r) AS n")
        assert (await r.single())["n"] == 1
