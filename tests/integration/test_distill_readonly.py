"""ReadOnlyNeo4jDriver against a real Neo4j (testcontainer): reads work, every
write path is refused -- client-side or by the server -- and construction builds
no index."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from neo4j import WRITE_ACCESS
from neo4j.exceptions import ClientError

from scripts.distill.readonly_driver import ReadOnlyNeo4jDriver, ReadOnlyViolation

pytestmark = pytest.mark.asyncio(loop_scope="module")

WRITE_REFUSED = (ReadOnlyViolation, ClientError)


def _server_refused(exc: pytest.ExceptionInfo) -> None:
    """The write reached Neo4j in a READ transaction and Neo4j itself refused it."""
    assert isinstance(exc.value, ClientError), exc.value
    assert "AccessMode" in (exc.value.code or ""), exc.value.code


async def _count(driver) -> int:
    async with driver.session() as s:
        rec = await (await s.run("MATCH (n) RETURN count(n) AS c")).single()
        return rec["c"]


async def _index_names(driver) -> set[str]:
    async with driver.session() as s:
        return {r["name"] async for r in await s.run("SHOW INDEXES YIELD name")}


async def _index_count(driver) -> int:
    return len(await _index_names(driver))


@pytest_asyncio.fixture(loop_scope="module")
async def seeded(extract_driver):
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (:Probe {name: 'seed'})")
    return extract_driver


@pytest_asyncio.fixture(loop_scope="module")
async def ro(extract_neo4j):
    uri, user, password = extract_neo4j
    driver = ReadOnlyNeo4jDriver(uri, user, password)
    yield driver
    await driver.close()


async def test_construction_runs_no_index_build(extract_neo4j, extract_driver):
    before = await _index_count(extract_driver)
    uri, user, password = extract_neo4j
    driver = ReadOnlyNeo4jDriver(uri, user, password)     # inside a running loop
    await asyncio.sleep(1.0)            # a scheduled build would have run by now
    assert driver._init_task is None
    assert await _index_count(extract_driver) == before
    await driver.close()


async def test_a_match_runs(seeded, ro):
    records, _, _ = await ro.execute_query("MATCH (p:Probe) RETURN p.name AS name")
    assert [r["name"] for r in records] == ["seed"]
    async with ro.session() as s:
        rec = await (await s.run("MATCH (p:Probe) RETURN count(p) AS c")).single()
    assert rec["c"] == 1


async def test_create_through_execute_query_is_refused(seeded, ro):
    with pytest.raises(WRITE_REFUSED) as exc:
        await ro.execute_query("CREATE (:Probe {name: 'x'})")
    _server_refused(exc)
    with pytest.raises(ReadOnlyViolation):
        await ro.execute_query("CREATE (:Probe {name: 'x'})", routing_="w")
    assert await _count(seeded) == 1


async def test_create_through_a_session_is_refused_by_the_server(seeded, ro):
    with pytest.raises(WRITE_REFUSED) as exc:
        async with ro.session() as s:
            await (await s.run("CREATE (:Probe {name: 'x'})")).consume()
    _server_refused(exc)
    with pytest.raises(WRITE_REFUSED) as exc:
        async with ro.transaction() as tx:
            await tx.run("CREATE (:Probe {name: 'x'})")
    _server_refused(exc)
    assert await _count(seeded) == 1


async def test_create_through_execute_read_is_refused_by_the_server(seeded, ro):
    async def write(tx):
        await (await tx.run("CREATE (:Probe {name: 'x'})")).consume()

    with pytest.raises(WRITE_REFUSED) as exc:
        async with ro.session() as s:
            await s.execute_read(write)
    _server_refused(exc)
    assert await _count(seeded) == 1


async def test_graphitis_final_bulk_save_is_refused(seeded, ro):
    """add_episode's last step: add_nodes_and_edges_bulk -> session.execute_write."""
    ep = EpisodicNode(name="e", group_id="backup-docs", source=EpisodeType.text,
                      source_description="d", content="c",
                      valid_at=datetime.now(timezone.utc), created_at=datetime.now(timezone.utc))
    with pytest.raises(ReadOnlyViolation):
        await add_nodes_and_edges_bulk(ro, [ep], [], [], [], embedder=None)  # type: ignore[arg-type]
    with pytest.raises(WRITE_REFUSED) as exc:
        await ep.save(ro)                 # execute_query, READ-routed
    _server_refused(exc)
    assert await _count(seeded) == 1


async def test_private_attributes_cannot_bypass_read_mode(seeded, ro):
    """The review probe: the neo4j session's private _run_transaction(WRITE_ACCESS)
    committed a node when __getattr__ delegated everything. Neither it nor the
    unwrapped driver is reachable through the wrappers now."""
    async def write(tx):
        await (await tx.run("CREATE (:Probe {name: 'x'})")).consume()

    async with ro.session() as s:
        with pytest.raises(AttributeError):
            await s._run_transaction(WRITE_ACCESS, None, write, (), {})
    with pytest.raises(AttributeError):
        ro.client._driver
    with pytest.raises(AttributeError):
        ro.session()._session
    assert await _count(seeded) == 1


async def test_explicit_index_build_is_refused(extract_driver, ro):
    before = await _index_count(extract_driver)
    with pytest.raises(ReadOnlyViolation):
        await ro.build_indices_and_constraints()
    assert await _index_count(extract_driver) == before


async def test_control_graphitis_plain_driver_does_build_indices(extract_neo4j, extract_driver):
    """The control for test_construction_runs_no_index_build: the same construction
    with graphiti's own Neo4jDriver DOES build indices, so an unchanged count above
    is evidence, not an artefact of timing. It writes indices, and drops them again."""
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    before = await _index_names(extract_driver)
    uri, user, password = extract_neo4j
    plain = Neo4jDriver(uri, user, password)
    try:
        assert plain._init_task is not None
        await plain._init_task
        assert len(await _index_names(extract_driver)) > len(before)
    finally:
        await plain.close()
        # Drop what it built, so the "no index build" assertions above stay
        # order-independent on this shared container.
        async with extract_driver.session() as s:
            for name in await _index_names(extract_driver) - before:
                await (await s.run(f"DROP INDEX `{name}` IF EXISTS")).consume()
        assert await _index_names(extract_driver) == before
