"""The relation-expansion sibling fetch (spec 2026-10-15 §10.1, Task 8) against a real
Neo4j: an unordered bounded fetch, then the scope filter (one bounded episode lookup)
and the uuid sort + cap in Python -- deterministic for a static graph, the scope
filter applied before the cap. Fixture: one subject entity with three current
`SUPPORTS` edges -- two from Veeam episodes, one from a Cohesity episode that sorts
first -- plus one invalidated edge."""
from __future__ import annotations

import pytest

import answer_api.list_mode as lm
from answer_api.scope import Scope

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


async def test_the_siblings_query_is_bounded_and_skips_invalid_and_other_relations(
        extract_driver):
    await _seed(extract_driver)
    rows, _, _ = await extract_driver.execute_query(
        lm.SIBLINGS_QUERY, src="subj", rel="SUPPORTS", fetch=10)
    assert sorted(r["uuid"] for r in rows) == ["f0", "f1", "f2"]
    rows, _, _ = await extract_driver.execute_query(
        lm.SIBLINGS_QUERY, src="subj", rel="SUPPORTS", fetch=2)
    assert len(rows) == 2


async def test_unscoped_siblings_are_ordered_and_capped(extract_driver):
    await _seed(extract_driver)
    for _ in range(3):                                       # same answer every run
        rows = await lm._group_siblings(extract_driver, "subj", "SUPPORTS",
                                        scope=Scope(), cap=2)
        assert [r["uuid"] for r in rows] == ["f0", "f1"]


async def test_scoped_siblings_filter_before_the_cap(extract_driver):
    await _seed(extract_driver)
    rows = await lm._group_siblings(extract_driver, "subj", "SUPPORTS",
                                    scope=Scope(vendors=("Veeam",)), cap=2)
    assert [r["uuid"] for r in rows] == ["f1", "f2"]          # Cohesity-only f0 excluded
    rows = await lm._group_siblings(extract_driver, "subj", "SUPPORTS",
                                    scope=Scope(sources=("DP Guide",)), cap=5)
    assert [r["uuid"] for r in rows] == ["f0", "f1"]
    rows = await lm._group_siblings(extract_driver, "subj", "SUPPORTS",
                                    scope=Scope(products=("vbr",)), cap=1)
    assert [r["uuid"] for r in rows] == ["f1"]
