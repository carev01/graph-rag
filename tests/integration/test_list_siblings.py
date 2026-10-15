"""The relation-expansion sibling queries (spec 2026-10-15 §10.1) against a real
Neo4j: deterministic order, and the scope filter applied before the cap. Fixture:
one subject entity with three current `SUPPORTS` edges -- two from Veeam episodes,
one from a Cohesity episode that sorts first -- plus one invalidated edge."""
from __future__ import annotations

import pytest

import answer_api.list_mode as lm

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def _seed(driver):
    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run(
            "CREATE (:Vendor {id:'v1', name:'Veeam'})-[:HAS_PRODUCT]->"
            "(:Product {id:'p1', name:'VBR'})-[:HAS_SOURCE]->(:Source {id:'s1', name:'VBR UG'})"
            "-[:HAS_ARTICLE]->(:Article {id:'a1'})-[:HAS_EPISODE]->(:Episodic {uuid:'ev1'})")
        await s.run(
            "CREATE (:Vendor {id:'v2', name:'Cohesity'})-[:HAS_PRODUCT]->"
            "(:Product {id:'p2', name:'DataProtect'})-[:HAS_SOURCE]->"
            "(:Source {id:'s2', name:'DP Guide'})"
            "-[:HAS_ARTICLE]->(:Article {id:'a2'})-[:HAS_EPISODE]->(:Episodic {uuid:'ec1'})")
        await s.run(
            "CREATE (a:Entity {uuid:'subj'}) "
            "CREATE (a)-[:RELATES_TO {uuid:'f0', name:'SUPPORTS', fact:'c', episodes:['ec1']}]"
            "->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'f2', name:'SUPPORTS', fact:'v2', episodes:['ev1']}]"
            "->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'f1', name:'SUPPORTS', fact:'v1', "
            "episodes:['ec1', 'ev1']}]->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'f3', name:'SUPPORTS', fact:'old', episodes:['ev1'], "
            "invalid_at: datetime('2020-01-01T00:00:00Z')}]->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'f4', name:'OTHER', fact:'o', episodes:['ev1']}]"
            "->(:Entity)")


async def test_unscoped_siblings_are_ordered_and_capped(extract_driver):
    await _seed(extract_driver)
    rows, _, _ = await extract_driver.execute_query(
        lm.SIBLINGS_QUERY, src="subj", rel="SUPPORTS", cap=2)
    assert [r["uuid"] for r in rows] == ["f0", "f1"]


async def test_scoped_siblings_filter_before_the_cap(extract_driver):
    await _seed(extract_driver)
    rows, _, _ = await extract_driver.execute_query(
        lm.SCOPED_SIBLINGS_QUERY, src="subj", rel="SUPPORTS", cap=2,
        vendors=["veeam"], products=[], sources=[])
    assert [r["uuid"] for r in rows] == ["f1", "f2"]          # Cohesity-only f0 excluded
    rows, _, _ = await extract_driver.execute_query(
        lm.SCOPED_SIBLINGS_QUERY, src="subj", rel="SUPPORTS", cap=5,
        vendors=[], products=[], sources=["dp guide"])
    assert [r["uuid"] for r in rows] == ["f0", "f1"]
