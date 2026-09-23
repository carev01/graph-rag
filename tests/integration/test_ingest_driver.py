"""Live end-to-end test for IngestDriver: the real DocExtractor cluster and LLM
stack, writing into a THROWAWAY Neo4j testcontainer.

It used to target whatever Neo4j `.env` names and reset its article by
DETACH-DELETEing that article's episodes -- on today's `.env` that is the live
`backup-docs` graph, and AWS_ART is one of its ingested pilot articles (BACKLOG
16). Now the graph side is a testcontainer seeded with the one `:Article` and
`:Chapter` the driver reads, so the test cannot touch any real graph. The LLM,
embedder and DocExtractor calls are still real, which is the point of `@live`.

Opt-in only (excluded from the default lane by `-m 'not live'`), and it spends
money on the extraction LLM:

    uv run --extra dev pytest tests/integration/test_ingest_driver.py -m live -v
"""
import pytest
import pytest_asyncio
from neo4j import AsyncGraphDatabase

from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import ExtractionTier, build_graphiti
from graph_extract.ingest_driver import IngestDriver
from graph_extract.ontology import EXTRACTION_INSTRUCTIONS
from graph_extract.provenance import Provenance

pytestmark = [pytest.mark.live, pytest.mark.asyncio(loop_scope="module")]

AWS_ART = "011dc3fa-62ea-4832-8588-e7b815e8a380"  # Vault access policies (small)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def container_ingest(extract_neo4j, live_docext_client):
    uri, user, password = extract_neo4j
    s = get_extract_settings().model_copy(
        update={"neo4j_uri": uri, "neo4j_user": user, "neo4j_password": password})
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    graphiti = build_graphiti(s)
    await graphiti.build_indices_and_constraints()
    async with driver.session() as sess:
        await sess.run(
            "MERGE (a:Article {id:$a}) SET a.source_url = 'https://example.invalid/aws' "
            "MERGE (c:Chapter {id:'test-chapter'}) SET c.title = 'Vault access' "
            "MERGE (a)-[:IN_CHAPTER]->(c)", a=AWS_ART)
    strong = ExtractionTier("strong", graphiti, EXTRACTION_INSTRUCTIONS, s.max_chunk_tokens)
    ing = IngestDriver(s, strong, None, live_docext_client, Provenance(driver), driver)
    yield ing, driver
    await graphiti.close()
    await driver.close()


async def test_ingest_one_article_links_provenance(container_ingest):
    ing, driver = container_ingest
    r = await ing.ingest_article(AWS_ART)
    assert r.episodes_added >= 1
    async with driver.session() as s:
        q = await s.run(
            "MATCH (:Article {id:$a})-[:HAS_EPISODE]->(e) RETURN count(e) AS n",
            a=AWS_ART,
        )
        assert (await q.single())["n"] == r.episodes_added
    # re-run is idempotent
    r2 = await ing.ingest_article(AWS_ART)
    assert r2.episodes_added == 0 and r2.episodes_skipped >= 1
