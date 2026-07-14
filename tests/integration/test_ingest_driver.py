"""Live end-to-end test for IngestDriver against the docker-compose Neo4j
(5.26) and the real DocExtractor cluster / LLM stack.

Prerequisite: `graph_sync` bootstrap of the AWS Backup source
(21632f3b-5a4c-4c93-9f00-6701d0e9f677) must already have populated
`:Article {id: AWS_ART}` (and its `:IN_CHAPTER` link) in the same Neo4j
instance this test targets -- see tests/e2e/test_live.py.

Opt-in only (excluded from the default `pytest` lane by `-m 'not live'`).
Run explicitly with:

    uv run --extra dev pytest tests/integration/test_ingest_driver.py -m live -v

This performs a REAL extraction through the local LLM (gpt-oss-20b) and can
take several minutes.
"""
import pytest

pytestmark = [pytest.mark.live, pytest.mark.asyncio]

AWS_ART = "011dc3fa-62ea-4832-8588-e7b815e8a380"  # Vault access policies (small)


async def test_ingest_one_article_links_provenance(live_ingest_driver, live_extract_driver):
    # Reset this article's episodes first so `episodes_added >= 1` holds on
    # every run (the compose Neo4j is persistent; without this a re-run would
    # be fully hash-gated and add 0).
    async with live_extract_driver.session() as s:
        await s.run(
            "MATCH (:Article {id:$a})-[:HAS_EPISODE]->(e:Episodic) DETACH DELETE e",
            a=AWS_ART)
    r = await live_ingest_driver.ingest_article(AWS_ART)
    assert r.episodes_added >= 1
    async with live_extract_driver.session() as s:
        q = await s.run(
            "MATCH (:Article {id:$a})-[:HAS_EPISODE]->(e) RETURN count(e) AS n",
            a=AWS_ART,
        )
        assert (await q.single())["n"] == r.episodes_added
    # re-run is idempotent
    r2 = await live_ingest_driver.ingest_article(AWS_ART)
    assert r2.episodes_added == 0 and r2.episodes_skipped >= 1
