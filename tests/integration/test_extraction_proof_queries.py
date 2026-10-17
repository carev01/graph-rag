"""scripts/extraction_proof.py's Cypher against a real Neo4j (the integration image):
the affected-article selection, the per-article foreign-unnamed flag, the structural
chain copy and the write-target sentinel. No LLM, no production."""
import importlib.util
import sys
from pathlib import Path

import pytest
import pytest_asyncio

pytestmark = pytest.mark.asyncio(loop_scope="module")

_spec = importlib.util.spec_from_file_location(
    "extraction_proof", Path(__file__).parents[2] / "scripts" / "extraction_proof.py")
proof = importlib.util.module_from_spec(_spec)
sys.modules["extraction_proof"] = proof
_spec.loader.exec_module(proof)

# Three structural chains; one episode each. Securiti's chunk never names AWS Backup
# (forced subject), Veeam Data Cloud's never names Cohesity DataProtect (other
# subject), AWS Backup's own chunk names its subject. a0 (superseded edge) and a00
# (removed episode) repeat a1's failure and must be neither sampled nor counted.
_SEED = """
CREATE (v1:Vendor {id:'v1', name:'Veeam'}), (v2:Vendor {id:'v2', name:'AWS'}),
       (v3:Vendor {id:'v3', name:'Cohesity'}),
       (p1:Product {id:'p1', name:'Securiti.ai'}), (p2:Product {id:'p2', name:'AWS Backup'}),
       (p3:Product {id:'p3', name:'Cohesity DataProtect'}),
       (p4:Product {id:'p4', name:'Veeam Data Cloud'}),
       (s1:Source {id:'s1', name:'Securiti docs'}), (s2:Source {id:'s2', name:'AWS docs'}),
       (s4:Source {id:'s4', name:'VDC docs'}),
       (a1:Article {id:'a1', title:'Access verification', content_hash:'h1'}),
       (a2:Article {id:'a2', title:'Controls', content_hash:'h2'}),
       (a3:Article {id:'a3', title:'VDC page', content_hash:'h3'}),
       (a4:Article {id:'a0', title:'Old', content_hash:'h0'}),
       (a5:Article {id:'a00', title:'Removed', content_hash:'h5'}),
       (c1:Chapter {id:'c1', title:'Connectors'}),
       (v1)-[:HAS_PRODUCT]->(p1)-[:HAS_SOURCE]->(s1)-[:HAS_ARTICLE]->(a1),
       (v2)-[:HAS_PRODUCT]->(p2)-[:HAS_SOURCE]->(s2)-[:HAS_ARTICLE]->(a2),
       (v3)-[:HAS_PRODUCT]->(p3),
       (v1)-[:HAS_PRODUCT]->(p4)-[:HAS_SOURCE]->(s4)-[:HAS_ARTICLE]->(a3),
       (s1)-[:HAS_ARTICLE]->(a4), (s1)-[:HAS_ARTICLE]->(a5),
       (a1)-[:IN_CHAPTER]->(c1),
       (e1:Episodic {uuid:'e1', content:'Connectors: Azure Blob Storage, Oracle'}),
       (e2:Episodic {uuid:'e2', content:'AWS Backup: recovery points are encrypted'}),
       (e3:Episodic {uuid:'e3', content:'Restore to Proxmox VE is supported'}),
       (e0:Episodic {uuid:'e0', content:'Connectors'}),
       (e5:Episodic {uuid:'e5', content:'Connectors', removed:true}),
       (a1)-[:HAS_EPISODE {chunk_index:0}]->(e1),
       (a2)-[:HAS_EPISODE {chunk_index:0}]->(e2),
       (a3)-[:HAS_EPISODE {chunk_index:0}]->(e3),
       (a4)-[:HAS_EPISODE {chunk_index:0, superseded:true}]->(e0),
       (a5)-[:HAS_EPISODE {chunk_index:0}]->(e5),
       (aws:Entity:Product {uuid:'n1', name:'AWS Backup'}),
       (blob:Entity {uuid:'n2', name:'Azure Blob Storage'}),
       (rp:Entity {uuid:'n3', name:'recovery point'}),
       (coh:Entity:Product {uuid:'n4', name:'Cohesity DataProtect'}),
       (px:Entity {uuid:'n5', name:'Proxmox VE'}),
       (old:Entity {uuid:'n6', name:'Oracle'}),
       (aws)-[:RELATES_TO {uuid:'f1', name:'Supports', fact:'AWS Backup supports Azure Blob',
                           episodes:['e1']}]->(blob),
       (aws)-[:RELATES_TO {uuid:'f2', name:'Supports', fact:'AWS Backup has recovery points',
                           episodes:['e2']}]->(rp),
       (coh)-[:RELATES_TO {uuid:'f3', name:'RestoresTo', fact:'Restore to Proxmox VE',
                           episodes:['e3']}]->(px),
       (aws)-[:RELATES_TO {uuid:'f4', name:'Supports', fact:'AWS Backup supports Oracle',
                           episodes:['e0', 'e5']}]->(old),
       (e1)-[:MENTIONS]->(aws), (e1)-[:MENTIONS]->(blob),
       (e2)-[:MENTIONS]->(aws), (e2)-[:MENTIONS]->(rp),
       (e3)-[:MENTIONS]->(coh), (e3)-[:MENTIONS]->(px),
       (e0)-[:MENTIONS]->(aws), (e0)-[:MENTIONS]->(old),
       (e5)-[:MENTIONS]->(aws), (e5)-[:MENTIONS]->(old)
"""


@pytest_asyncio.fixture(loop_scope="module")
async def seeded(extract_driver):
    await extract_driver.execute_query("MATCH (n) DETACH DELETE n")
    await extract_driver.execute_query(_SEED)
    return extract_driver


async def test_product_names_are_structural_and_lower_case(seeded):
    names = await proof.read_product_names(seeded)
    assert names == ["aws backup", "cohesity dataprotect", "securiti.ai", "veeam data cloud"]


async def test_affected_selection_splits_example_and_other_subjects(seeded):
    names = await proof.read_product_names(seeded)
    picked = await proof.select_affected(seeded, names, 4, exclude=[])
    # a2 names its own product; a0's episode is superseded, a00's removed
    assert picked == [("a1", "affected-example"), ("a3", "affected-other")]
    assert await proof.select_affected(seeded, names, 2, exclude=["a1"]) == [
        ("a3", "affected-other")]


async def test_article_facts_flag_the_foreign_unnamed_subject(seeded):
    names = await proof.read_product_names(seeded)
    a1 = proof.summarise_facts(await proof.read_article_facts(seeded, "a1", names))
    assert a1.total == 1 and a1.foreign == [("AWS Backup", "AWS Backup supports Azure Blob")]
    a2 = proof.summarise_facts(await proof.read_article_facts(seeded, "a2", names))
    assert a2.total == 1 and a2.foreign == []
    a3 = proof.summarise_facts(await proof.read_article_facts(seeded, "a3", names))
    assert a3.restores == [("Cohesity DataProtect", "Proxmox VE")]
    assert len(a3.foreign) == 1
    for dead in ("a0", "a00"):
        assert await proof.read_article_facts(seeded, dead, names) == []


async def test_chain_copy_round_trips(seeded, extract_neo4j):
    uri = extract_neo4j[0]
    (chain,) = await proof.read_chains(seeded, "a1")
    assert chain["article"]["content_hash"] == "h1" and chain["chapter"]["id"] == "c1"
    await seeded.execute_query("MATCH (n) DETACH DELETE n")
    graph = proof.ThrowawayGraph(seeded, uri, container_uri=uri,
                                 production_uri="bolt://production.invalid:7687")
    await graph.copy_chain(chain)
    rec = (await seeded.execute_query(
        "MATCH (:Vendor {name:'Veeam'})-[:HAS_PRODUCT]->(:Product {name:'Securiti.ai'})"
        "-[:HAS_SOURCE]->(:Source)-[:HAS_ARTICLE]->(a:Article {id:'a1'})"
        "-[:IN_CHAPTER]->(c:Chapter) RETURN a.content_hash AS h, c.title AS t")).records
    assert [(r["h"], r["t"]) for r in rec] == [("h1", "Connectors")]
    await graph.write_sentinel("tok")
    await proof.verify_write_targets({"container": seeded}, "tok")
    with pytest.raises(proof.WriteTargetRefused):
        await proof.verify_write_targets({"container": seeded}, "other")


async def test_the_production_ingest_stack_writes_only_to_the_container(extract_neo4j):
    """The script's construction path on the container: _build_ingest_driver (no LLM
    call is made building it), then the sentinel seen through every write driver."""
    from graph_extract.cli import _build_ingest_driver
    from graph_extract.config import ExtractSettings

    uri, user, password = extract_neo4j
    s = ExtractSettings(_env_file=None, neo4j_uri=uri, neo4j_user=user,
                        neo4j_password=password, docext_base_url="https://docext.invalid",
                        docext_read_key="k", extraction_routing=True,
                        cheap_llm_api_key="k", embed_dim=768)
    ingest, graphiti, docext, ingest_neo4j = await _build_ingest_driver(s)
    try:
        drivers = proof.ingest_write_drivers(ingest, graphiti, ingest_neo4j)
        assert set(drivers) == {"ingest/provenance", "strong graphiti", "cheap graphiti"}
        graph = proof.ThrowawayGraph(ingest_neo4j, uri, container_uri=uri,
                                     production_uri="bolt://production.invalid:7687")
        await graph.write_sentinel("stack-token")
        await proof.verify_write_targets(drivers, "stack-token")
    finally:
        await ingest_neo4j.close()
        await docext.aclose()
        await graphiti.close()
        await ingest._cheap.graphiti.close()


async def test_read_routed_queries_cannot_write(extract_driver):
    """The production side's read-only claim holds on a standalone server: a write in
    a READ-routed query is rejected, not silently applied."""
    from neo4j.exceptions import ClientError

    with pytest.raises(ClientError):
        await extract_driver.execute_query("CREATE (:ShouldNotExist)", routing_=proof.READ)
    rec = (await extract_driver.execute_query(
        "MATCH (n:ShouldNotExist) RETURN count(n) AS c")).records
    assert rec[0]["c"] == 0
