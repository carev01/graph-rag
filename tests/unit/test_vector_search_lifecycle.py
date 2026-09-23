"""ensure_vector_indexes' wait and state handling, against a fake driver.

The state a real index is in after a bounded wait cannot be steered on a
testcontainer without seeding ~10^5 vectors (minutes), so the POPULATING /
FAILED branches are pinned here; the ONLINE path runs for real in
tests/integration/test_vector_search.py."""
from __future__ import annotations

import logging
from typing import Any

import pytest
from neo4j.exceptions import ClientError

from graph_extract import vector_search as vs

pytestmark = pytest.mark.asyncio

DIM = 8


class _Result:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records


def _row(name: str, state: str, pct: float = 100.0, **over: Any) -> dict[str, Any]:
    entity_type, labels, props = vs._SCHEMA[name]
    row = {"name": name, "state": state, "populationPercent": pct,
           "entityType": entity_type, "labelsOrTypes": labels, "properties": props,
           "options": {"indexConfig": vs.expected_config(DIM)}}
    row.update(over)
    return row


class _FakeDriver:
    def __init__(self, rows: list[dict[str, Any]],
                 await_errors: dict[str, Exception] | None = None) -> None:
        self.rows = rows
        self.await_errors = await_errors or {}
        self.queries: list[tuple[str, dict[str, Any]]] = []

    async def execute_query(self, query: str, **params: Any) -> _Result:
        self.queries.append((query, params))
        if "db.awaitIndex(" in query and params.get("name") in self.await_errors:
            raise self.await_errors[params["name"]]
        if query.startswith("SHOW"):
            return _Result(self.rows)
        return _Result([])

    def awaited(self) -> list[tuple[str, dict[str, Any]]]:
        return [(q, p) for q, p in self.queries if "awaitIndex" in q]


def _timed_out(name: str) -> ClientError:
    # Verbatim shape of what Neo4j 2026.07.1 raises (measured, fix report F1).
    return ClientError._hydrate_neo4j(
        code="Neo.ClientError.Procedure.ProcedureTimedOut",
        message=f"Index on 'Index( id=5, name='{name}', type='VECTOR' )' "
                "did not come online within 1 SECONDS")


async def test_waits_on_this_modules_two_indexes_only_within_the_budget():
    driver = _FakeDriver([_row(vs.EDGE_INDEX, "ONLINE"), _row(vs.NODE_INDEX, "ONLINE")])
    await vs.ensure_vector_indexes(driver, DIM, wait_seconds=60.0)
    awaited = driver.awaited()
    assert all("db.awaitIndexes" not in q for q, _ in awaited), \
        "db.awaitIndexes waits on EVERY index in the database"
    assert sorted(p["name"] for _, p in awaited) == sorted([vs.EDGE_INDEX, vs.NODE_INDEX])
    assert all(0 <= p["timeout"] <= 60 for _, p in awaited)


async def test_default_wait_is_the_startup_budget_not_an_hour():
    driver = _FakeDriver([_row(vs.EDGE_INDEX, "ONLINE"), _row(vs.NODE_INDEX, "ONLINE")])
    await vs.ensure_vector_indexes(driver, DIM)
    assert all(p["timeout"] <= vs.DEFAULT_STARTUP_WAIT_SECONDS for _, p in driver.awaited())
    assert vs.DEFAULT_STARTUP_WAIT_SECONDS == 60.0


async def test_a_populating_index_warns_with_its_progress_and_does_not_raise(caplog):
    driver = _FakeDriver([_row(vs.EDGE_INDEX, "ONLINE"),
                          _row(vs.NODE_INDEX, "POPULATING", pct=41.5)],
                         await_errors={vs.NODE_INDEX: _timed_out(vs.NODE_INDEX)})
    with caplog.at_level(logging.WARNING, logger=vs.logger.name):
        await vs.ensure_vector_indexes(driver, DIM, wait_seconds=1)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert vs.NODE_INDEX in warnings[0]
    assert "41.5%" in warnings[0]
    assert "POPULATING" in warnings[0]
    # never tell an operator to drop a half-built index
    assert "rebuild" not in warnings[0].lower()


async def test_a_failed_index_raises_with_the_rebuild_hint():
    driver = _FakeDriver([_row(vs.EDGE_INDEX, "FAILED", pct=12.0),
                          _row(vs.NODE_INDEX, "ONLINE")])
    with pytest.raises(vs.VectorIndexMismatch) as exc:
        await vs.ensure_vector_indexes(driver, DIM, wait_seconds=1)
    assert vs.EDGE_INDEX in str(exc.value)
    assert "FAILED" in str(exc.value)
    assert vs.REBUILD_HINT in str(exc.value)


async def test_a_config_mismatch_raises_even_while_populating():
    bad = dict(vs.expected_config(DIM), **{"vector.hnsw.m": 16})
    driver = _FakeDriver([_row(vs.EDGE_INDEX, "ONLINE"),
                          _row(vs.NODE_INDEX, "POPULATING", pct=3.0,
                               options={"indexConfig": bad})])
    with pytest.raises(vs.VectorIndexMismatch, match="vector.hnsw.m"):
        await vs.ensure_vector_indexes(driver, DIM, wait_seconds=1)


async def test_an_unexpected_await_error_propagates():
    err = ClientError._hydrate_neo4j(code="Neo.ClientError.Security.Forbidden",
                                     message="not allowed")
    driver = _FakeDriver([_row(vs.EDGE_INDEX, "ONLINE"), _row(vs.NODE_INDEX, "ONLINE")],
                         await_errors={vs.NODE_INDEX: err})
    with pytest.raises(ClientError):
        await vs.ensure_vector_indexes(driver, DIM, wait_seconds=1)


async def test_index_status_reports_population_percent():
    driver = _FakeDriver([_row(vs.NODE_INDEX, "POPULATING", pct=7.25)])
    status = await vs.index_status(driver)
    assert status[vs.NODE_INDEX]["populationPercent"] == 7.25
    assert status[vs.NODE_INDEX]["state"] == "POPULATING"


async def test_an_await_error_on_a_failed_index_still_gets_the_rebuild_hint():
    err = ClientError._hydrate_neo4j(code="Neo.ClientError.Schema.IndexFailed",
                                     message="index failed to populate")
    driver = _FakeDriver([_row(vs.EDGE_INDEX, "FAILED"), _row(vs.NODE_INDEX, "ONLINE")],
                         await_errors={vs.EDGE_INDEX: err})
    with pytest.raises(vs.VectorIndexMismatch, match="FAILED"):
        await vs.ensure_vector_indexes(driver, DIM, wait_seconds=1)
