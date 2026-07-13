import pytest
pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_dedup_report_counts_nodes_and_distinctness(extract_driver):
    async with extract_driver.session() as s:
        # one canonical "immutability" node, two distinct S3/Blob nodes
        await s.run("CREATE (:Entity:Concept {name:'immutability', group_id:'g'})")
        await s.run("CREATE (:Entity:Workload {name:'Amazon S3', group_id:'g'})")
        await s.run("CREATE (:Entity:Workload {name:'Azure Blob Storage', group_id:'g'})")
    from graph_extract.eval import dedup_report
    rep = await dedup_report(extract_driver, "g",
        canon_merge=["immutability"],
        distinct_pairs=[("Amazon S3", "Azure Blob Storage")])
    assert rep["merge"]["immutability"]["node_count"] == 1
    assert rep["distinct"][0]["collapsed"] is False
    assert rep["totals"]["entities"] == 3
    assert rep["totals"]["by_label"]["Workload"] == 2
    assert rep["totals"]["by_label"]["Concept"] == 1


async def test_dedup_report_flags_collapsed_pair(extract_driver):
    async with extract_driver.session() as s:
        # a "collapsed" pair: only one node exists for what should be two
        await s.run("CREATE (:Entity {name:'Collapsed A', group_id:'g2'})")
    from graph_extract.eval import dedup_report
    rep = await dedup_report(extract_driver, "g2",
        canon_merge=[],
        distinct_pairs=[("Collapsed A", "Collapsed B")])
    assert rep["distinct"][0]["node_count"] == 1
    assert rep["distinct"][0]["collapsed"] is True


async def test_provenance_report_counts_resolved_and_dangling(extract_driver):
    async with extract_driver.session() as s:
        await s.run(
            "CREATE (:Article {id:'pa1', source_id:'s1', source_url:'https://u', title:'T'})"
            "-[:HAS_EPISODE {chunk_index:0}]->(:Episodic {uuid:'pep-1', content:'S3 supports immutability.'})"
        )
        await s.run(
            "CREATE (e1:Entity {name:'Amazon S3', group_id:'g'}), "
            "(e2:Entity {name:'immutability', group_id:'g'}) "
            "CREATE (e1)-[:RELATES_TO {uuid:'pf-resolved', fact:'S3 supports immutability', "
            "episodes:['pep-1'], group_id:'g'}]->(e2)"
        )
        await s.run(
            "CREATE (e3:Entity {name:'Azure Blob', group_id:'g'}), "
            "(e4:Entity {name:'immutability', group_id:'g'}) "
            "CREATE (e3)-[:RELATES_TO {uuid:'pf-dangling', fact:'Azure Blob supports immutability', "
            "episodes:['nonexistent-episode'], group_id:'g'}]->(e4)"
        )
    from graph_extract.eval import provenance_report
    rep = await provenance_report(extract_driver, sample=2)
    assert rep["sampled"] == 2
    assert rep["resolved"] == 1
    assert rep["dangling"] == 1
