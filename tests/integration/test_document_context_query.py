"""`IngestDriver._document_context` against a real Neo4j (testcontainer).

The unit tests (tests/unit/test_ingest_document_context.py) fake the driver and
hand back a record, so they never run the Cypher itself: the two `CALL () {...}`
subqueries, the lowest-source-id pick and the chain count. This runs the real
query over a seeded structural chain: one chain, no chain, and two chains
(lowest source id wins, with a WARNING naming the count).
"""
from __future__ import annotations

import logging

import pytest
import pytest_asyncio

from graph_extract.config import ExtractSettings
from graph_extract.graphiti_client import ExtractionTier
from graph_extract.ingest_driver import IngestDriver

pytestmark = pytest.mark.asyncio(loop_scope="module")

_LOGGER = "graph_extract.ingest_driver"


def _ingest(driver) -> IngestDriver:
    settings = ExtractSettings(_env_file=None, docext_base_url="http://x",
                               docext_read_key="k", neo4j_uri="bolt://x",
                               neo4j_user="u", neo4j_password="p")
    return IngestDriver(settings=settings,
                        strong_tier=ExtractionTier("strong", object(), "I", 1000),
                        cheap_tier=None, docext=object(), provenance=object(),
                        driver=driver)


async def _chain(driver, *, vendor: str, product: str, source_id: str, source: str,
                 article: str) -> None:
    async with driver.session() as s:
        await s.run(
            "MERGE (v:Vendor {id:$vendor}) SET v.name = $vendor "
            "MERGE (p:Product {id:$product}) SET p.name = $product "
            "MERGE (so:Source {id:$sid}) SET so.name = $source "
            "MERGE (a:Article {id:$article}) "
            "MERGE (v)-[:HAS_PRODUCT]->(p) "
            "MERGE (p)-[:HAS_SOURCE]->(so) "
            "MERGE (so)-[:HAS_ARTICLE]->(a)",
            vendor=vendor, product=product, sid=source_id, source=source,
            article=article)


@pytest_asyncio.fixture(loop_scope="module")
async def clean_driver(extract_driver):
    # The container is shared module-wide: start every test from an empty graph.
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
    return extract_driver


async def test_one_chain_is_returned_without_a_warning(clean_driver, caplog):
    await _chain(clean_driver, vendor="Veeam", product="Veeam Backup & Replication",
                 source_id="src-vbr", source="VBR User Guide", article="a1")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        got = await _ingest(clean_driver)._document_context("a1")
    assert got == ("Veeam", "Veeam Backup & Replication", "VBR User Guide")
    assert [r for r in caplog.records if r.name == _LOGGER] == []


async def test_no_chain_gives_an_empty_context_and_a_warning(clean_driver, caplog):
    async with clean_driver.session() as s:
        # the article exists, but nothing structural points at it
        await s.run("MERGE (:Article {id:'orphan'})")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        got = await _ingest(clean_driver)._document_context("orphan")
    assert got == ("", "", "")
    msgs = [r.getMessage() for r in caplog.records
            if r.name == _LOGGER and r.levelno == logging.WARNING]
    assert len(msgs) == 1 and "orphan" in msgs[0] and "no structural chain" in msgs[0]


async def test_two_chains_pick_the_lowest_source_id_and_warn(clean_driver, caplog):
    # Inserted highest id first, so a "whatever Neo4j returns first" pick would
    # tend to get it wrong.
    await _chain(clean_driver, vendor="Veeam", product="Veeam Agent for Linux",
                 source_id="src-b", source="Agent for Linux Guide", article="shared")
    await _chain(clean_driver, vendor="Veeam", product="Veeam Backup & Replication",
                 source_id="src-a", source="VBR User Guide", article="shared")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        got = await _ingest(clean_driver)._document_context("shared")
    assert got == ("Veeam", "Veeam Backup & Replication", "VBR User Guide")
    msgs = [r.getMessage() for r in caplog.records
            if r.name == _LOGGER and r.levelno == logging.WARNING]
    assert len(msgs) == 1
    assert "shared" in msgs[0] and "2 structural chains" in msgs[0]
    assert "lowest source id" in msgs[0]
