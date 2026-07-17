import httpx
import pytest_asyncio
from neo4j import AsyncGraphDatabase
from testcontainers.neo4j import Neo4jContainer
from testcontainers.postgres import PostgresContainer

from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_graphiti, ExtractionTier
from graph_extract.ingest_driver import IngestDriver
from graph_extract.ontology import EXTRACTION_INSTRUCTIONS
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


# --- Live fixtures: target the docker-compose Neo4j (5.26), NOT a
# testcontainer. graphiti-core 0.29.2 issues dynamic-label Cypher that
# Neo4j 5.22 (the testcontainer image above) rejects, so the live e2e test
# needs the real, already-bootstrapped compose instance (bolt://localhost:7687
# per .env) where the AWS Backup source's articles already exist.

@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def live_extract_driver():
    s = get_extract_settings()
    driver = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    yield driver
    await driver.close()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def live_docext_client():
    s = get_extract_settings()
    client = httpx.AsyncClient(
        base_url=s.docext_base_url,
        headers={"X-API-Key": s.docext_read_key},
        verify=s.docext_verify_tls,
        timeout=300.0,
    )
    yield client
    await client.aclose()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def live_ingest_driver(live_extract_driver, live_docext_client):
    s = get_extract_settings()
    graphiti = build_graphiti(s)
    provenance = Provenance(live_extract_driver)
    strong = ExtractionTier("strong", graphiti, EXTRACTION_INSTRUCTIONS, s.max_chunk_tokens)
    driver = IngestDriver(s, strong, None, live_docext_client, provenance, live_extract_driver)
    yield driver
    await graphiti.close()
