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


async def test_link_bad_uuid_is_noop(extract_driver):
    from graph_extract.provenance import Provenance
    p = Provenance(extract_driver)
    async with extract_driver.session() as s:
        await s.run("CREATE (:Article {id:'a2'})")
        await s.run("CREATE (:Episodic {uuid:'e_old2'})")
    await p.link("a2", "e_old2", chunk_index=0, heading_path="", token_count=1, content_hash="h1")
    # attempt to re-key to a non-existent episode uuid
    await p.link("a2", "e_does_not_exist", chunk_index=0, heading_path="", token_count=1, content_hash="h2")
    async with extract_driver.session() as s:
        rec = await (await s.run(
            "MATCH (:Article {id:'a2'})-[r:HAS_EPISODE {chunk_index:0}]->(e:Episodic) "
            "RETURN count(r) AS c, collect(e.uuid) AS uuids")).single()
        old_sup = (await (await s.run(
            "MATCH (e:Episodic {uuid:'e_old2'}) RETURN e.superseded AS s")).single())["s"]
    assert rec["c"] == 1
    assert rec["uuids"] == ["e_old2"]
    assert old_sup is not True


async def test_link_re_activate_clears_superseded(extract_driver):
    from graph_extract.provenance import Provenance
    p = Provenance(extract_driver)
    async with extract_driver.session() as s:
        await s.run("CREATE (:Article {id:'a3'})")
        await s.run("CREATE (:Episodic {uuid:'e_old3'})")
        await s.run("CREATE (:Episodic {uuid:'e_new3'})")
    await p.link("a3", "e_old3", chunk_index=0, heading_path="", token_count=1, content_hash="h1")
    await p.link("a3", "e_new3", chunk_index=0, heading_path="", token_count=1, content_hash="h2")
    await p.link("a3", "e_old3", chunk_index=0, heading_path="", token_count=1, content_hash="h3")
    async with extract_driver.session() as s:
        rec = await (await s.run(
            "MATCH (:Article {id:'a3'})-[r:HAS_EPISODE {chunk_index:0}]->(e:Episodic) "
            "RETURN count(r) AS c, collect(e.uuid) AS uuids")).single()
        old_sup = (await (await s.run(
            "MATCH (e:Episodic {uuid:'e_old3'}) RETURN e.superseded AS s")).single())["s"]
    assert rec["c"] == 1
    assert rec["uuids"] == ["e_old3"]
    assert old_sup is False


async def test_link_supersede_does_not_delete_episode_node(extract_driver):
    from graph_extract.provenance import Provenance
    p = Provenance(extract_driver)
    async with extract_driver.session() as s:
        await s.run("CREATE (:Article {id:'a4'})")
        await s.run("CREATE (:Episodic {uuid:'e_old4'})")
        await s.run("CREATE (:Episodic {uuid:'e_new4'})")
    await p.link("a4", "e_old4", chunk_index=0, heading_path="", token_count=1, content_hash="h1")
    await p.link("a4", "e_new4", chunk_index=0, heading_path="", token_count=1, content_hash="h2")
    async with extract_driver.session() as s:
        node = await (await s.run(
            "MATCH (e:Episodic {uuid:'e_old4'}) RETURN e")).single()
    assert node is not None
