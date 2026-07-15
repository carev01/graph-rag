import pytest
pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_link_rekey_supersedes_and_dedupes(extract_driver):
    from graph_extract.provenance import Provenance
    p = Provenance(extract_driver)
    async with extract_driver.session() as s:
        await s.run("CREATE (:Article {id:'a1'})")
        await s.run("CREATE (:Episodic {uuid:'e_old'})")
        await s.run("CREATE (:Episodic {uuid:'e_new'})")
    await p.link("a1", "e_old", chunk_index=0, heading_path="", token_count=1, content_hash="h1")
    await p.link("a1", "e_new", chunk_index=0, heading_path="", token_count=1, content_hash="h2")
    async with extract_driver.session() as s:
        n = (await (await s.run(
            "MATCH (:Article {id:'a1'})-[r:HAS_EPISODE {chunk_index:0}]->() RETURN count(r) AS c")).single())["c"]
        old_sup = (await (await s.run(
            "MATCH (e:Episodic {uuid:'e_old'}) RETURN e.superseded AS s")).single())["s"]
    assert n == 1 and old_sup is True
    # idempotent: re-linking the same new episode is a no-op
    await p.link("a1", "e_new", chunk_index=0, heading_path="", token_count=1, content_hash="h2")
    async with extract_driver.session() as s:
        n2 = (await (await s.run(
            "MATCH (:Article {id:'a1'})-[r:HAS_EPISODE {chunk_index:0}]->() RETURN count(r) AS c")).single())["c"]
    assert n2 == 1
