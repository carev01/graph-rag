import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer
from testcontainers.postgres import PostgresContainer

from graph_extract.provenance import Provenance
from graph_sync.neo4j_repo import Neo4jRepo
from graph_sync.state_store import StateStore


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def neo4j_repo():
    with Neo4jContainer("neo4j:5.22") as neo:
        repo = Neo4jRepo(neo.get_connection_url(), "neo4j", neo.password)
        await repo.init_schema()
        yield repo
        await repo.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def state_store():
    with PostgresContainer("postgres:16") as pg:
        dsn = pg.get_connection_url().replace("+psycopg2", "")
        store = StateStore(dsn)
        await store.init_schema()
        yield store
        await store.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def extract_driver():
    with Neo4jContainer("neo4j:5.22") as neo:
        driver = AsyncGraphDatabase.driver(
            neo.get_connection_url(), auth=("neo4j", neo.password))
        yield driver
        await driver.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def extract_provenance(extract_driver):
    return Provenance(extract_driver)
