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

async def test_silent_merge_suspects(extract_driver):
    # Silent merge: "Amazon S3" is a SHOULD_DISTINCT member (paired against
    # "Azure Blob Storage") but here it's mentioned by BOTH vendors' episodes
    # -- cross-vendor support on a single labelled-distinct name is exactly
    # the "attached to the other vendor's node without a separate node ever
    # being created" failure mode this field surfaces. "AWS Backup" is also a
    # labelled member but is single-vendor here, so it must NOT show up.
    g = "silent"
    async with extract_driver.session() as s:
        await s.run("""
        CREATE (vA:Vendor {id:'vA2', name:'AWS'})-[:HAS_PRODUCT]->(pA:Product {id:'pA2'})
              -[:HAS_SOURCE]->(sA:Source {id:'sA2'})-[:HAS_ARTICLE]->(aA:Article {id:'aA2'})
              -[:HAS_EPISODE]->(eA:Episodic {uuid:'epA2'})
        CREATE (vM:Vendor {id:'vM2', name:'Microsoft'})-[:HAS_PRODUCT]->(pM:Product {id:'pM2'})
              -[:HAS_SOURCE]->(sM:Source {id:'sM2'})-[:HAS_ARTICLE]->(aM:Article {id:'aM2'})
              -[:HAS_EPISODE]->(eM:Episodic {uuid:'epM2'})
        CREATE (s3:Entity:Workload {name:'Amazon S3', group_id:$g})
        CREATE (eA)-[:MENTIONS]->(s3)
        CREATE (eM)-[:MENTIONS]->(s3)
        CREATE (backup:Entity:Product {name:'AWS Backup', group_id:$g})
        CREATE (eA)-[:MENTIONS]->(backup)
        """, g=g)
    from graph_extract.eval import dedup_report_v2
    from graph_extract import quality_labels
    rep = await dedup_report_v2(extract_driver, g, quality_labels)
    assert "Amazon S3" in rep["silent_merge_suspects"]["names"]
    assert rep["silent_merge_suspects"]["count"] >= 1
    assert "AWS Backup" not in rep["silent_merge_suspects"]["names"]


async def test_dedup_v2_distinct_tristate(extract_driver):
    from graph_extract.eval import dedup_report_v2
    from types import SimpleNamespace
    g = "tri"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {group_id:$g, name:'AWS Backup'})", g=g)
        await s.run("CREATE (:Entity {group_id:$g, name:'Azure Backup'})", g=g)   # distinct pair
        await s.run("CREATE (:Entity {group_id:$g, name:'Amazon S3'})", g=g)       # S3 present, Blob absent
        await s.run("CREATE (:Entity {group_id:$g, name:'Shared Vault'})", g=g)    # one node for a merged pair
    labels = SimpleNamespace(
        SHOULD_MERGE=[],
        SHOULD_DISTINCT=[
            ["AWS Backup", "Azure Backup"],
            ["Amazon S3", "Azure Blob Storage"],
            ["Shared Vault", ["Shared Vault", "Also Shared Vault"]],  # B aliases onto the same node
        ],
        VENDOR_TOKENS=["aws", "amazon", "azure"],
    )
    rep = await dedup_report_v2(extract_driver, g, labels)
    states = [e["state"] for e in rep["should_distinct"]]
    assert states == ["distinct", "absent", "merged"]  # order matches SHOULD_DISTINCT
    for e in rep["should_distinct"]:
        assert e["collapsed"] == (e["state"] == "merged")
