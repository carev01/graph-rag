import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def _fact(s, g, uuid, eps):
    # a RELATES_TO fact between two throwaway endpoint nodes, carrying an
    # episodes list -- the shape sweep_stale_facts scans.
    await s.run(
        "CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:$u, episodes:$eps}]->(y:Entity)",
        g=g, u=uuid, eps=eps)


async def test_sweep_expires_fact_with_no_live_episodes(extract_driver):
    from graph_extract.staleness_sweep import sweep_stale_facts
    g = "swp1"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Episodic {uuid:'e_dead'})")  # exists, NO HAS_EPISODE edge -> dead
        await _fact(s, g, "f1", ["e_dead"])
    res = await sweep_stale_facts(extract_driver, g)
    async with extract_driver.session() as s:
        row = await (await s.run(
            "MATCH ()-[f:RELATES_TO {uuid:'f1'}]->() RETURN f.invalid_at AS inv, "
            "f.expired_by_sweep AS ex")).single()
    assert res["expired"] == 1 and row["inv"] is not None and row["ex"] is True


async def test_sweep_keeps_fact_with_live_but_superseded_episode(extract_driver):
    # THE #6 CASE: episode has a HAS_EPISODE edge (alive) but the Episodic
    # NODE's superseded=true -> fact must NOT be expired. Liveness is keyed on
    # the edge's own `superseded` flag, never the node's, so the node-level
    # flag here is never consulted.
    from graph_extract.staleness_sweep import sweep_stale_facts
    g = "swp2"
    async with extract_driver.session() as s:
        await s.run(
            "CREATE (:Article {removed:false})-[:HAS_EPISODE]->"
            "(:Episodic {uuid:'e_live', superseded:true})")
        await _fact(s, g, "f2", ["e_live"])
    await sweep_stale_facts(extract_driver, g)
    async with extract_driver.session() as s:
        inv = (await (await s.run(
            "MATCH ()-[f:RELATES_TO {uuid:'f2'}]->() RETURN f.invalid_at AS i")).single())["i"]
    assert inv is None  # kept alive by the HAS_EPISODE edge; node-level superseded ignored


async def test_sweep_expires_removed_episode_fact(extract_driver):
    # episode is linked via HAS_EPISODE but flagged removed=true -> dead
    from graph_extract.staleness_sweep import sweep_stale_facts
    g = "swp3"
    async with extract_driver.session() as s:
        await s.run(
            "CREATE (:Article {removed:false})-[:HAS_EPISODE]->"
            "(:Episodic {uuid:'e_removed', removed:true})")
        await _fact(s, g, "f3", ["e_removed"])
    res = await sweep_stale_facts(extract_driver, g)
    async with extract_driver.session() as s:
        row = await (await s.run(
            "MATCH ()-[f:RELATES_TO {uuid:'f3'}]->() RETURN f.invalid_at AS inv, "
            "f.expired_by_sweep AS ex")).single()
    assert res["expired"] == 1 and row["inv"] is not None and row["ex"] is True


async def test_sweep_ignores_already_invalid(extract_driver):
    # fact with invalid_at already set (e.g. Graphiti-invalidated, or a prior
    # sweep) must not be touched again -- not a scan candidate.
    from graph_extract.staleness_sweep import sweep_stale_facts
    g = "swp4"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Episodic {uuid:'e_dead4'})")  # dead, no HAS_EPISODE
        await s.run(
            "CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f4', episodes:['e_dead4'], "
            "invalid_at:datetime(), expired_by_sweep:false}]->(y:Entity)",
            g=g)
    res = await sweep_stale_facts(extract_driver, g)
    async with extract_driver.session() as s:
        row = await (await s.run(
            "MATCH ()-[f:RELATES_TO {uuid:'f4'}]->() RETURN f.expired_by_sweep AS ex")).single()
    assert res["scanned"] == 0
    assert row["ex"] is False  # untouched, not re-swept


async def test_sweep_does_not_overwrite_existing_invalid_at(extract_driver):
    # fact IS expirable by the dead-episode rule (all episodes dead) BUT its
    # invalid_at is ALREADY set -- simulating Graphiti having invalidated it
    # concurrently (e.g. in the window between the sweep's scan and its
    # SET). The sweep must never clobber Graphiti's own invalid_at/flag.
    from graph_extract.staleness_sweep import sweep_stale_facts
    g = "swp6"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Episodic {uuid:'e_dead6'})")  # dead, no HAS_EPISODE
        await s.run(
            "CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f6', episodes:['e_dead6'], "
            "invalid_at:datetime('2020-01-01T00:00:00Z')}]->(y:Entity)",
            g=g)
    await sweep_stale_facts(extract_driver, g)
    async with extract_driver.session() as s:
        row = await (await s.run(
            "MATCH ()-[f:RELATES_TO {uuid:'f6'}]->() RETURN f.invalid_at AS inv, "
            "f.expired_by_sweep AS ex")).single()
    assert row["inv"].to_native().isoformat() == "2020-01-01T00:00:00+00:00"  # unchanged
    assert row["ex"] is None  # never touched by the sweep


async def test_sweep_keeps_fact_with_any_live_episode(extract_driver):
    # one dead episode + one live episode -> fact kept (any alive episode
    # is enough to keep the fact valid).
    from graph_extract.staleness_sweep import sweep_stale_facts
    g = "swp5"
    async with extract_driver.session() as s:
        await s.run(
            "CREATE (:Article {removed:false})-[:HAS_EPISODE]->(:Episodic {uuid:'e_live5'})")
        await s.run("CREATE (:Episodic {uuid:'e_dead5'})")  # no HAS_EPISODE -> dead
        await _fact(s, g, "f5", ["e_live5", "e_dead5"])
    await sweep_stale_facts(extract_driver, g)
    async with extract_driver.session() as s:
        inv = (await (await s.run(
            "MATCH ()-[f:RELATES_TO {uuid:'f5'}]->() RETURN f.invalid_at AS i")).single())["i"]
    assert inv is None


async def test_sweep_expires_facts_from_a_superseded_edge(extract_driver):
    """Defect B: a shrunk article's dropped chunk keeps its edge, so the old sweep
    (which keyed liveness on linkage alone) saw it as alive and never expired it."""
    from graph_extract.staleness_sweep import sweep_stale_facts
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("""
        CREATE (a:Article {id:'sup1'})
        CREATE (live:Episodic {uuid:'ep_live', group_id:'g'})
        CREATE (dead:Episodic {uuid:'ep_dead', group_id:'g'})
        CREATE (a)-[:HAS_EPISODE {chunk_index:0}]->(live)
        CREATE (a)-[:HAS_EPISODE {chunk_index:1, superseded:true}]->(dead)
        CREATE (x:Entity {uuid:'x', group_id:'g'})
        CREATE (y:Entity {uuid:'y', group_id:'g'})
        CREATE (x)-[:RELATES_TO {uuid:'f_live', group_id:'g', episodes:['ep_live']}]->(y)
        CREATE (x)-[:RELATES_TO {uuid:'f_dead', group_id:'g', episodes:['ep_dead']}]->(y)
        """)
    out = await sweep_stale_facts(extract_driver, "g")
    assert out["expired"] == 1
    assert out["expired_sample"] == ["f_dead"]
    async with extract_driver.session() as s:
        rows = [rec async for rec in await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:'g'}]->() "
            "RETURN f.uuid AS uuid, f.invalid_at IS NOT NULL AS expired ORDER BY uuid")]
    assert {r["uuid"]: r["expired"] for r in rows} == {"f_dead": True, "f_live": False}
