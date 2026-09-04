import pytest
pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_link_rekey_keeps_old_edge_and_flags_it(extract_driver):
    """The old edge SURVIVES so the superseded fact stays citable (design decision
    #2 for history); the flag is what makes it dead for liveness."""
    from graph_extract.provenance import Provenance
    p = Provenance(extract_driver)
    async with extract_driver.session() as s:
        await s.run("CREATE (:Article {id:'a1'})")
        await s.run("CREATE (:Episodic {uuid:'e_old'})")
        await s.run("CREATE (:Episodic {uuid:'e_new'})")
    await p.link("a1", "e_old", chunk_index=0, heading_path="", token_count=1, content_hash="h1")
    await p.link("a1", "e_new", chunk_index=0, heading_path="", token_count=1, content_hash="h2")
    async with extract_driver.session() as s:
        rows = [rec async for rec in await s.run(
            "MATCH (:Article {id:'a1'})-[r:HAS_EPISODE {chunk_index:0}]->(e:Episodic) "
            "RETURN e.uuid AS uuid, coalesce(r.superseded, false) AS sup ORDER BY uuid")]
    # BOTH edges present -- the old one is retained, not deleted
    assert [r["uuid"] for r in rows] == ["e_new", "e_old"]
    assert {r["uuid"]: r["sup"] for r in rows} == {"e_old": True, "e_new": False}
    # exactly one LIVE edge at this chunk_index -- the new invariant
    assert sum(1 for r in rows if not r["sup"]) == 1


async def test_link_rekey_is_idempotent(extract_driver):
    """Re-linking the same episode must not add another edge."""
    from graph_extract.provenance import Provenance
    p = Provenance(extract_driver)
    async with extract_driver.session() as s:
        await s.run("CREATE (:Article {id:'a1b'})")
        await s.run("CREATE (:Episodic {uuid:'e_b'})")
    for _ in range(3):
        await p.link("a1b", "e_b", chunk_index=0, heading_path="", token_count=1, content_hash="h")
    async with extract_driver.session() as s:
        n = (await (await s.run(
            "MATCH (:Article {id:'a1b'})-[r:HAS_EPISODE {chunk_index:0}]->() "
            "RETURN count(r) AS c")).single())["c"]
    assert n == 1


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


async def test_link_re_activation_flips_which_edge_is_live(extract_driver):
    """Shrink-then-restore: relinking the original episode must make ITS edge live
    again and supersede the other. Both edges remain, so both stay citable."""
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
        rows = [rec async for rec in await s.run(
            "MATCH (:Article {id:'a3'})-[r:HAS_EPISODE {chunk_index:0}]->(e:Episodic) "
            "RETURN e.uuid AS uuid, coalesce(r.superseded, false) AS sup")]
    live = [r["uuid"] for r in rows if not r["sup"]]
    assert live == ["e_old3"]
    assert len(rows) == 2


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
