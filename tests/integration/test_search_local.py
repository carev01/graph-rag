from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

GROUP_ID = "backup-docs"
# graphiti's EntityEdge.invalid_at is datetime | None, never a string.
_PAST = datetime(2020, 1, 1, tzinfo=timezone.utc)


class _Edge:
    def __init__(self, uuid, fact, episodes, valid_at=None, invalid_at=None):
        self.uuid, self.fact, self.episodes = uuid, fact, episodes
        self.valid_at, self.invalid_at = valid_at, invalid_at


class _Results:
    def __init__(self, edges):
        self.edges = edges


class _StubGraphiti:
    def __init__(self, edges):
        self._edges = edges

    async def _search(self, query, config, group_ids=None, **kw):
        return _Results(self._edges)


async def _seed_article_episode_fact(driver, *, article_id="art1", episode_uuid="ep1"):
    async with driver.session() as s:
        await s.run(
            "CREATE (a:Article {id:$a, source_url:$u, title:$t})-[:HAS_EPISODE]->(:Episodic {uuid:$e})",
            a=article_id, u=f"https://x/{article_id}", t="T1", e=episode_uuid)
        await s.run(
            "CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:[$e]}]->(y:Entity)",
            g=GROUP_ID, e=episode_uuid)
        await s.run(
            "CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f_invalid', episodes:[$e]}]->(y:Entity)",
            g=GROUP_ID, e=episode_uuid)


async def test_search_local_returns_cited_facts(extract_driver):
    from answer_api.search import search_local

    await _seed_article_episode_fact(extract_driver, article_id="art1", episode_uuid="ep1")
    g = _StubGraphiti([
        _Edge("f1", "AWS Backup Vault Lock requires compliance mode", ["ep1"]),
        _Edge("f_invalid", "stale fact", ["ep1"], invalid_at=_PAST),
    ])
    out = await search_local(g, extract_driver, q="vault lock", k=10, group_id=GROUP_ID)
    facts = {r["fact_uuid"] for r in out["results"]}
    assert "f1" in facts and "f_invalid" not in facts  # invalid excluded by default
    result = next(r for r in out["results"] if r["fact_uuid"] == "f1")
    assert result["sources"][0]["article_id"] == "art1"


async def test_search_local_include_invalid(extract_driver):
    from answer_api.search import search_local

    await _seed_article_episode_fact(extract_driver, article_id="art2", episode_uuid="ep2")
    g = _StubGraphiti([
        _Edge("f1", "AWS Backup Vault Lock requires compliance mode", ["ep1"]),
        _Edge("f_invalid", "stale fact", ["ep1"], invalid_at=_PAST),
    ])
    out = await search_local(g, extract_driver, q="vault lock", k=10, group_id=GROUP_ID,
                             include_invalid=True)
    facts = {r["fact_uuid"] for r in out["results"]}
    assert "f_invalid" in facts


async def test_search_local_vendor_scope(extract_driver):
    from answer_api.search import search_local

    async with extract_driver.session() as s:
        await s.run(
            "CREATE (:Vendor {name:'AWS'})-[:HAS_PRODUCT]->(:Product)-[:HAS_SOURCE]->(:Source)"
            "-[:HAS_ARTICLE]->(:Article {id:'art_vendor'})-[:HAS_EPISODE]->(:Episodic {uuid:'ep_vendor'})")
        await s.run(
            "CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f_vendor', episodes:['ep_vendor']}]->(y:Entity)",
            g=GROUP_ID)
    g = _StubGraphiti([_Edge("f_vendor", "vendor scoped fact", ["ep_vendor"])])

    out_aws = await search_local(g, extract_driver, q="q", k=10, group_id=GROUP_ID, vendor="AWS")
    facts_aws = {r["fact_uuid"] for r in out_aws["results"]}
    assert "f_vendor" in facts_aws

    out_ms = await search_local(g, extract_driver, q="q", k=10, group_id=GROUP_ID, vendor="Microsoft")
    facts_ms = {r["fact_uuid"] for r in out_ms["results"]}
    assert "f_vendor" not in facts_ms


@pytest.mark.live
async def test_search_local_live_smoke(live_extract_driver):
    """Hits the real compose Neo4j (already-bootstrapped AWS Backup graph)
    with a real Graphiti (embeddings + hybrid search, no LLM at query time).
    Opt-in only: excluded from the default `-m "not live"` lane.
    """
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_graphiti
    from answer_api.search import search_local

    settings = get_extract_settings()
    graphiti = build_graphiti(settings)
    try:
        out = await search_local(
            graphiti, live_extract_driver,
            q="What does AWS Backup Vault Lock require?", k=5,
            group_id=settings.group_id)
        assert out["count"] >= 1
        assert any(r["sources"] for r in out["results"])
    finally:
        await graphiti.close()
