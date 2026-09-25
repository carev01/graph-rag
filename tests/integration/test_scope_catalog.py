"""ScopeResolver.load must read only the STRUCTURAL Vendor/Product catalog --
graphiti-extracted :Entity nodes carry the same :Vendor/:Product labels but have
no `id` and no HAS_PRODUCT edge to a structural node, and must never be detected
or scoped."""
import pytest
from neo4j import AsyncGraphDatabase

from answer_api.scope import Scope, ScopeResolver, scope_episode_uuids

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_load_reads_only_structural_catalog_and_scopes_episodes(extract_neo4j):
    uri, user, password = extract_neo4j
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        await driver.execute_query("MATCH (n) DETACH DELETE n")
        await driver.execute_query("""
            CREATE (:Vendor {id:'v1', name:'Veeam'})
                   -[:HAS_PRODUCT]->(:Product {id:'p1', name:'Veeam ONE'})
                   -[:HAS_SOURCE]->(:Source {id:'s1'})
                   -[:HAS_ARTICLE]->(:Article {id:'a1'})
                   -[:HAS_EPISODE]->(:Episodic {uuid:'e1'})
            CREATE (:Entity:Vendor {uuid:'x', name:'Fake Vendor'})
            CREATE (:Entity:Product {uuid:'y', name:'Cohesity Copilot'})
        """)

        resolver = await ScopeResolver.load(driver)

        detected = resolver.detect("How do I configure Veeam ONE?")
        assert detected.products == ("Veeam ONE",)
        assert resolver.detect("Fake Vendor").is_empty()
        assert resolver.detect("Cohesity Copilot").is_empty()

        assert await scope_episode_uuids(driver, Scope(("Veeam",), (), "x")) == {"e1"}
        assert await scope_episode_uuids(driver, Scope((), ("Veeam ONE",), "x")) == {"e1"}
    finally:
        await driver.execute_query("MATCH (n) DETACH DELETE n")
        await driver.close()
