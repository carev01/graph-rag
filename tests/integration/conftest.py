import pytest_asyncio
from testcontainers.neo4j import Neo4jContainer

from graph_sync.neo4j_repo import Neo4jRepo


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def neo4j_repo():
    with Neo4jContainer("neo4j:5.22") as neo:
        repo = Neo4jRepo(neo.get_connection_url(), "neo4j", neo.password)
        await repo.init_schema()
        yield repo
        await repo.close()
