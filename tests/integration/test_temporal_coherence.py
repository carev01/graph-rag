"""Every temporal edge case from the spec's table, in one graph, asserted once.

The bug this slice fixed was not any single wrong behaviour -- each path was
internally consistent -- but that the paths disagreed. So the thing worth pinning
is agreement.
"""
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

_GRAPH = """
MATCH (n) DETACH DELETE n
WITH count(*) AS _
CREATE (art:Article {id:'art1', source_url:'https://example.invalid/art1'})
CREATE (gone:Article {id:'art2', removed:true, source_url:'https://example.invalid/art2'})
// 1. plain live episode
CREATE (e1:Episodic {uuid:'e1', group_id:'g'})
CREATE (art)-[:HAS_EPISODE {chunk_index:0}]->(e1)
// 2. re-keyed: old edge retained + flagged, new edge live
CREATE (e2old:Episodic {uuid:'e2old', group_id:'g'})
CREATE (e2new:Episodic {uuid:'e2new', group_id:'g'})
CREATE (art)-[:HAS_EPISODE {chunk_index:1, superseded:true}]->(e2old)
CREATE (art)-[:HAS_EPISODE {chunk_index:1}]->(e2new)
// 3. dropped trailing chunk (shrink)
CREATE (e3:Episodic {uuid:'e3', group_id:'g'})
CREATE (art)-[:HAS_EPISODE {chunk_index:2, superseded:true}]->(e3)
// 4. episode with NO superseded property at all -> must be alive (no backfill)
CREATE (e4:Episodic {uuid:'e4', group_id:'g'})
CREATE (art)-[:HAS_EPISODE {chunk_index:3}]->(e4)
// 5. episode whose only article is tombstoned
CREATE (e5:Episodic {uuid:'e5', group_id:'g'})
CREATE (gone)-[:HAS_EPISODE {chunk_index:0}]->(e5)
// 6. episode itself removed
CREATE (e6:Episodic {uuid:'e6', group_id:'g', removed:true})
CREATE (art)-[:HAS_EPISODE {chunk_index:4}]->(e6)
// 7. episode referenced by TWO articles, superseded by only one
CREATE (e7:Episodic {uuid:'e7', group_id:'g', superseded:true})
CREATE (art)-[:HAS_EPISODE {chunk_index:5, superseded:true}]->(e7)
CREATE (other:Article {id:'art3', source_url:'https://example.invalid/art3'})
CREATE (other)-[:HAS_EPISODE {chunk_index:0}]->(e7)
CREATE (x:Entity {uuid:'x', group_id:'g'})
CREATE (y:Entity {uuid:'y', group_id:'g'})
CREATE (x)-[:RELATES_TO {uuid:'f1', group_id:'g', episodes:['e1']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f2old', group_id:'g', episodes:['e2old']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f2new', group_id:'g', episodes:['e2new']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f3', group_id:'g', episodes:['e3']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f4', group_id:'g', episodes:['e4']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f5', group_id:'g', episodes:['e5']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f6', group_id:'g', episodes:['e6']}]->(y)
CREATE (x)-[:RELATES_TO {uuid:'f7', group_id:'g', episodes:['e7']}]->(y)
"""


async def test_sweep_expires_exactly_the_dead_facts(extract_driver):
    from graph_extract.staleness_sweep import sweep_stale_facts
    async with extract_driver.session() as s:
        await s.run(_GRAPH)
    out = await sweep_stale_facts(extract_driver, "g")
    async with extract_driver.session() as s:
        rows = [rec async for rec in await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:'g'}]->() "
            "RETURN f.uuid AS uuid, f.invalid_at IS NOT NULL AS dead ORDER BY uuid")]
    got = {r["uuid"]: r["dead"] for r in rows}
    assert got == {
        "f1": False,      # plain live
        "f2old": True,    # superseded by re-key
        "f2new": False,   # the replacement
        "f3": True,       # dropped trailing chunk
        "f4": False,      # no superseded property -> alive, no backfill needed
        "f5": True,       # article tombstoned
        "f6": True,       # episode removed
        "f7": False,      # still live via the OTHER article
    }
    assert out["expired"] == 4


async def test_superseded_facts_still_resolve_citations(extract_driver):
    """Design decision #2 must hold for history: a fact that was ever true stays
    attributable to the document version that asserted it."""
    from graph_extract.provenance import Provenance
    async with extract_driver.session() as s:
        await s.run(_GRAPH)
    resolved = await Provenance(extract_driver).resolve_citations(["f2old", "f3"])
    for uuid in ("f2old", "f3"):
        sources = resolved.get(uuid, {}).get("sources") or []
        assert sources, f"{uuid} lost its citation"
        assert sources[0]["url"] == "https://example.invalid/art1"


async def test_liveness_is_per_edge_not_per_episode(extract_driver):
    """e7 is superseded by art1 but still live via art3 -- a node-level flag would
    have killed it for both."""
    from graph_extract.staleness_sweep import sweep_stale_facts
    async with extract_driver.session() as s:
        await s.run(_GRAPH)
    await sweep_stale_facts(extract_driver, "g")
    async with extract_driver.session() as s:
        dead = (await (await s.run(
            "MATCH ()-[f:RELATES_TO {uuid:'f7'}]->() "
            "RETURN f.invalid_at IS NOT NULL AS dead")).single())["dead"]
    assert dead is False


async def test_sweep_dates_the_expiry_by_removed_at_not_its_own_clock(extract_driver):
    """The sweep used to stamp `invalid_at = datetime()` -- its own run clock, a
    schedule artefact of exactly the kind that made crawl-ordered `valid_at`
    meaningless. Upstream's soft-delete timestamp (100% populated on tombstones)
    dates it properly."""
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run(
            "CREATE (a:Article {id:'art1', removed:true, "
            "                   removed_at:'2026-03-04T05:06:07Z'})"
            "-[:HAS_EPISODE]->(e:Episodic {uuid:'ep1', group_id:$g})", g=g)
        await s.run(
            "CREATE (x:Entity {uuid:'e1', group_id:$g})"
            "-[:RELATES_TO {group_id:$g, uuid:'f1', fact:'a fact', "
            "               episodes:['ep1']}]->(y:Entity {uuid:'e2', group_id:$g})", g=g)

    from graph_extract.staleness_sweep import sweep_stale_facts
    res = await sweep_stale_facts(extract_driver, g)
    assert res["expired"] == 1

    async with extract_driver.session() as s:
        r = await s.run("MATCH ()-[f:RELATES_TO {uuid:'f1'}]->() "
                        "RETURN toString(f.invalid_at) AS at, f.expired_by_sweep AS sw")
        row = await r.single()
    assert row["sw"] is True
    assert row["at"].startswith("2026-03-04T05:06:07"), (
        f"expected the article's removed_at, got {row['at']} -- the sweep is still "
        "stamping its own run time")


async def test_sweep_falls_back_to_now_when_nothing_carries_a_death_date(extract_driver):
    """A superseded-but-not-removed episode has no death timestamp anywhere, so
    the fallback must remain -- and must not crash on the missing value."""
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run(
            "CREATE (a:Article {id:'art1'})"
            "-[:HAS_EPISODE {superseded:true}]->(e:Episodic {uuid:'ep1', group_id:$g})", g=g)
        await s.run(
            "CREATE (x:Entity {uuid:'e1', group_id:$g})"
            "-[:RELATES_TO {group_id:$g, uuid:'f1', fact:'a fact', "
            "               episodes:['ep1']}]->(y:Entity {uuid:'e2', group_id:$g})", g=g)

    from graph_extract.staleness_sweep import sweep_stale_facts
    res = await sweep_stale_facts(extract_driver, g)
    assert res["expired"] == 1

    async with extract_driver.session() as s:
        r = await s.run("MATCH ()-[f:RELATES_TO {uuid:'f1'}]->() "
                        "RETURN f.invalid_at IS NOT NULL AS dated")
        assert (await r.single())["dated"] is True


async def test_a_malformed_removed_at_does_not_fail_the_sweep(extract_driver):
    """`datetime()` throws on bad input and Cypher has no try, so one bad value
    would otherwise take down the whole run."""
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run(
            "CREATE (a:Article {id:'art1', removed:true, removed_at:'not-a-date'})"
            "-[:HAS_EPISODE]->(e:Episodic {uuid:'ep1', group_id:$g})", g=g)
        await s.run(
            "CREATE (x:Entity {uuid:'e1', group_id:$g})"
            "-[:RELATES_TO {group_id:$g, uuid:'f1', fact:'a fact', "
            "               episodes:['ep1']}]->(y:Entity {uuid:'e2', group_id:$g})", g=g)

    from graph_extract.staleness_sweep import sweep_stale_facts
    res = await sweep_stale_facts(extract_driver, g)
    assert res["expired"] == 1
