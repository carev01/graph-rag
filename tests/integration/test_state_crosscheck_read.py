"""`state_crosscheck._read`'s stranded-article query against real stores
(BACKLOG 50): only an Article with content, not removed, with no episodes AND no
job row of any status counts -- TOC placeholders (no content_hash) do not."""
import pytest
from neo4j import AsyncGraphDatabase

import graph_sync.state_crosscheck as cc
from graph_sync.config import Settings

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_read_reports_only_truly_stranded_articles(extract_neo4j, state_store, monkeypatch):
    uri, user, password = extract_neo4j
    monkeypatch.setattr(cc, "get_settings", lambda: Settings(
        docext_base_url="https://x", docext_read_key="k", neo4j_uri=uri,
        neo4j_user=user, neo4j_password=password, postgres_dsn=state_store._dsn))
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        await driver.execute_query("MATCH (n) DETACH DELETE n")
        await driver.execute_query("""
            CREATE (:Article {id:'stranded', source_id:'s1', content_hash:'h'}),
                   (:Article {id:'has-done-job', source_id:'s1', content_hash:'h'}),
                   (:Article {id:'has-dead-job', source_id:'s1', content_hash:'h'}),
                   (:Article {id:'placeholder', source_id:'s1'}),
                   (:Article {id:'removed', source_id:'s1', content_hash:'h', removed:true}),
                   (:Article {id:'other-src', source_id:'s2', content_hash:'h'})
            CREATE (:Article {id:'extracted', source_id:'s1', content_hash:'h'})
                   -[:HAS_EPISODE]->(:Episodic {uuid:'e1'})""")
        pool = await state_store._get_pool()
        await pool.execute("DELETE FROM semantic_jobs WHERE article_id = ANY($1::text[])",
                           ["stranded", "has-done-job", "has-dead-job", "placeholder",
                            "removed", "other-src", "extracted"])
        for aid, status in (("has-done-job", "done"), ("has-dead-job", "dead")):
            await pool.execute("INSERT INTO semantic_jobs (article_id, op, status, lane) "
                               "VALUES ($1, 'upsert', $2, 'bootstrap')", aid, status)

        _, _, stranded = await cc._read(sample=5)

        assert stranded == {"s1": ["stranded"], "s2": ["other-src"]}
    finally:
        await driver.execute_query("MATCH (n) DETACH DELETE n")
        await driver.close()
