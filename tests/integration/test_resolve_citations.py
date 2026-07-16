import pytest
pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_resolve_citations_batch(extract_driver):
    from graph_extract.provenance import Provenance
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("CREATE (a:Article {id:'art1', source_url:'https://x/1', title:'T1'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1'})")
        await s.run("CREATE (:Episodic {uuid:'ep_dangling'})")   # no Article
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1']}]->(y:Entity)", g=g)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f2', episodes:['ep_dangling']}]->(y:Entity)", g=g)
    prov = Provenance(extract_driver)
    out = await prov.resolve_citations(["f1", "f2"])
    assert out["f1"] == [{"url": "https://x/1", "title": "T1", "article_id": "art1"}]
    assert out["f2"] == []       # dangling -> empty, surfaced not dropped
