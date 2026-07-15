import pytest
pytestmark = pytest.mark.asyncio(loop_scope="module")

async def _seed_cross_vendor(driver):
    # immutability mentioned by an AWS article AND an Azure article -> cross-vendor.
    async with driver.session() as s:
        await s.run("""
        CREATE (vA:Vendor {id:'vA', name:'AWS'})-[:HAS_PRODUCT]->(pA:Product {id:'pA'})
              -[:HAS_SOURCE]->(sA:Source {id:'sA'})-[:HAS_ARTICLE]->(aA:Article {id:'aA'})
              -[:HAS_EPISODE]->(eA:Episodic {uuid:'epA'})
        CREATE (vM:Vendor {id:'vM', name:'Microsoft'})-[:HAS_PRODUCT]->(pM:Product {id:'pM'})
              -[:HAS_SOURCE]->(sM:Source {id:'sM'})-[:HAS_ARTICLE]->(aM:Article {id:'aM'})
              -[:HAS_EPISODE]->(eM:Episodic {uuid:'epM'})
        CREATE (imm:Entity:Capability {name:'immutability', group_id:'g'})
        CREATE (eA)-[:MENTIONS]->(imm)
        CREATE (eM)-[:MENTIONS]->(imm)
        CREATE (s3:Entity:Workload {name:'Amazon S3', group_id:'g'})
        CREATE (blob:Entity:Workload {name:'Azure Blob Storage', group_id:'g'})
        // vendor-branded entity with cross-vendor support -> a suspect false merge
        CREATE (vl:Entity:Product {name:'AWS Backup Vault Lock', group_id:'g'})
        CREATE (eA)-[:MENTIONS]->(vl)
        CREATE (eM)-[:MENTIONS]->(vl)
        """)

async def test_dedup_report_v2(extract_driver):
    await _seed_cross_vendor(extract_driver)
    from graph_extract.eval import dedup_report_v2
    from graph_extract import quality_labels
    rep = await dedup_report_v2(extract_driver, "g", quality_labels)
    assert rep["should_merge"]["immutability"]["node_count"] == 1
    assert rep["should_merge"]["immutability"]["cross_vendor"] is True
    pair = next(p for p in rep["should_distinct"] if p["pair"] == ["Amazon S3", "Azure Blob Storage"])
    assert pair["collapsed"] is False
    assert any("AWS Backup Vault Lock" in nm for nm in rep["suspect_false_merge"]["names"])

async def test_noise_report(extract_driver):
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {name:'arn:aws:x', group_id:'g2'})")
        await s.run("CREATE (:Entity {name:'immutability', group_id:'g2'})")
    from graph_extract.eval import noise_report
    rep = await noise_report(extract_driver, "g2")
    assert rep["entities"]["total"] == 2 and rep["entities"]["noise"] == 1
