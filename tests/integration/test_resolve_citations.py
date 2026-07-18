import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

GROUP = "backup-docs"


async def test_resolve_citations_batch(extract_driver):
    from graph_extract.provenance import Provenance
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (a:Article {id:'art1', source_url:'https://x/1', title:'T1'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1'})")
        await s.run("CREATE (:Episodic {uuid:'ep_dangling'})")   # no Article
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1']}]->(y:Entity)", g=GROUP)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f2', episodes:['ep_dangling']}]->(y:Entity)", g=GROUP)
    out = await Provenance(extract_driver).resolve_citations(["f1", "f2", "f_absent"])
    assert out["f1"]["sources"] == [
        {"url": "https://x/1", "title": "T1", "article_id": "art1",
         "section": None, "vendor": None, "product": None}]
    assert out["f1"]["valid_at"] is None and out["f1"]["invalid_at"] is None
    assert out["f2"] == {"valid_at": None, "invalid_at": None, "sources": []}
    assert "f_absent" not in out          # absent fact stays absent (caller contract)


async def test_resolve_citations_enriched(extract_driver):
    from graph_extract.provenance import Provenance
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run(
            "CREATE (v:Vendor {name:'Veeam'})-[:HAS_PRODUCT]->(p:Product {name:'B&R'})"
            "-[:HAS_SOURCE]->(src:Source)-[:HAS_ARTICLE]->"
            "(a:Article {id:'art1', source_url:'https://x/1', title:'Vault Lock'})"
            "-[:HAS_EPISODE {heading_path:'Backup > Vault Lock'}]->(:Episodic {uuid:'ep1'})")
        await s.run(
            "CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1'], "
            "valid_at: datetime('2023-01-01'), invalid_at: null}]->(y:Entity)", g=GROUP)
    out = await Provenance(extract_driver).resolve_citations(["f1"])
    assert out["f1"]["valid_at"].startswith("2023-01-01")
    assert out["f1"]["invalid_at"] is None
    src = out["f1"]["sources"][0]
    assert src["url"] == "https://x/1"
    assert src["vendor"] == "Veeam"
    assert src["product"] == "B&R"
    assert src["section"] == "Backup > Vault Lock"
