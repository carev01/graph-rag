import pytest
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_NODE_DISTANCE, EDGE_HYBRID_SEARCH_RRF)

pytestmark = pytest.mark.asyncio(loop_scope="module")

GROUP_ID = "backup-docs"


class _Edge:
    def __init__(self, uuid, fact, episodes):
        self.uuid = uuid
        self.fact = fact
        self.episodes = episodes
        self.valid_at = None
        self.invalid_at = None


class _Results:
    def __init__(self, edges):
        self.edges = edges


class _CapGraphiti:
    def __init__(self, edges):
        self._edges = edges
        self.last_kw = None
        self.last_config = None

    async def _search(self, query, config, group_ids=None, **kw):
        self.last_kw = kw
        self.last_config = config
        return _Results(self._edges)


async def _seed(driver):
    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (a:Article {id:'art1', source_url:'https://x/art1', title:'T'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1'})")
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1']}]->(y:Entity)",
                    g=GROUP_ID)


async def test_center_node_switches_recipe_and_passes_through(extract_driver):
    from answer_api.search import search_local
    await _seed(extract_driver)
    g = _CapGraphiti([_Edge("f1", "AWS Backup fact", ["ep1"])])
    out = await search_local(g, extract_driver, q="q", k=10, group_id=GROUP_ID,
                             center_node_uuid="center-entity")
    assert g.last_kw.get("center_node_uuid") == "center-entity"
    assert g.last_config.edge_config.reranker == EDGE_HYBRID_SEARCH_NODE_DISTANCE.edge_config.reranker
    assert out["results"][0]["sources"][0]["url"] == "https://x/art1"   # still resolves


async def test_no_center_node_keeps_rrf(extract_driver):
    from answer_api.search import search_local
    await _seed(extract_driver)
    g = _CapGraphiti([_Edge("f1", "AWS Backup fact", ["ep1"])])
    await search_local(g, extract_driver, q="q", k=10, group_id=GROUP_ID)
    assert g.last_kw.get("center_node_uuid") is None
    assert g.last_config.edge_config.reranker == EDGE_HYBRID_SEARCH_RRF.edge_config.reranker
