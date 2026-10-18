"""A graphiti `Neo4jDriver` that cannot write.

The distillation harness runs graphiti's real `add_episode` pipeline against the
LIVE production graph, so every write graphiti would make -- the index build it
schedules at construction, the final bulk save, anything else -- must fail before
it reaches the data. Three layers, outermost first:

1. **No index build.** `Neo4jDriver.__init__` schedules
   `build_indices_and_constraints()` as a task when an event loop is running. We
   cancel that task before it ever starts (a task cancelled before its first step
   never executes its coroutine body), and `build_indices_and_constraints` /
   `delete_all_indexes` raise `ReadOnlyViolation` when called explicitly.
2. **Client-side refusal.** `self.client` -- the neo4j `AsyncDriver` that every
   graphiti code path goes through (`execute_query`, `session`, `transaction`) --
   is wrapped in `_ReadOnlyAsyncDriver`:
   - every `execute_query` is sent with `routing_=RoutingControl.READ` (the neo4j
     default is WRITE); asking for WRITE raises `ReadOnlyViolation`;
   - every session opens with `default_access_mode=READ_ACCESS`; asking for
     WRITE access raises `ReadOnlyViolation`;
   - `session.execute_write` raises `ReadOnlyViolation`. This one matters most:
     the neo4j driver's `execute_write` forces WRITE mode whatever the session's
     default, and it is exactly what graphiti's final save
     (`add_nodes_and_edges_bulk`) calls.
3. **Server-side refusal.** Anything that still slips through (`session.run`,
   `begin_transaction`, `execute_read` with a write query) runs in a READ
   transaction, which Neo4j itself rejects for a write
   (`Neo.ClientError.Statement.AccessMode`).

`ReadOnlyViolation` subclasses `PermissionError`, never a neo4j error, so no
graphiti retry or "index already exists" handler mistakes it for something to
swallow.
"""
from __future__ import annotations

from typing import Any

from graphiti_core.driver.neo4j_driver import Neo4jDriver
from neo4j import READ_ACCESS, RoutingControl


class ReadOnlyViolation(PermissionError):
    """A caller asked a read-only driver to write."""


def _is_write_routing(routing: Any) -> bool:
    return routing == RoutingControl.WRITE or str(routing).lower() in ("w", "write")


class _ReadOnlySession:
    """A neo4j `AsyncSession` (opened in READ mode) that refuses write transactions.
    Everything else -- `run`, `begin_transaction`, `execute_read`, `close` -- is
    delegated and runs under the READ default access mode."""

    def __init__(self, session: Any) -> None:
        self._session = session

    async def __aenter__(self) -> _ReadOnlySession:
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return await self._session.__aexit__(exc_type, exc, tb)

    async def execute_write(self, *args: Any, **kwargs: Any) -> Any:
        raise ReadOnlyViolation("execute_write on a read-only session")

    async def write_transaction(self, *args: Any, **kwargs: Any) -> Any:
        raise ReadOnlyViolation("write_transaction on a read-only session")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._session, name)


class _ReadOnlyAsyncDriver:
    """Wraps a neo4j `AsyncDriver`: READ routing on every query, READ access on
    every session, a `ReadOnlyViolation` for any request to write."""

    def __init__(self, driver: Any) -> None:
        self._driver = driver

    async def execute_query(self, query_: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            # parameters_/routing_ passed positionally could smuggle WRITE routing.
            raise ReadOnlyViolation("pass execute_query options by keyword")
        if _is_write_routing(kwargs.get("routing_", RoutingControl.READ)):
            raise ReadOnlyViolation("execute_query with WRITE routing")
        kwargs["routing_"] = RoutingControl.READ
        return await self._driver.execute_query(query_, **kwargs)

    def session(self, **config: Any) -> _ReadOnlySession:
        mode = config.get("default_access_mode", READ_ACCESS)
        if mode != READ_ACCESS:
            raise ReadOnlyViolation(f"session with access mode {mode!r}")
        config["default_access_mode"] = READ_ACCESS
        return _ReadOnlySession(self._driver.session(**config))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._driver, name)


class ReadOnlyNeo4jDriver(Neo4jDriver):
    """graphiti's `Neo4jDriver`, read-only: never builds indices, refuses writes.
    See the module docstring for the three layers."""

    def __init__(self, uri: str, user: str | None, password: str | None,
                 database: str | None = None) -> None:
        # None -> graphiti's own default ('neo4j'), which is what production uses.
        super().__init__(uri, user, password, database=database or "neo4j")
        if self._init_task is not None:
            # Scheduled but not yet started: cancelling now means its body
            # (the index build) never runs.
            self._init_task.cancel()
            self._init_task = None
        self.client = _ReadOnlyAsyncDriver(self.client)  # type: ignore[assignment]

    async def execute_query(self, cypher_query_: Any, **kwargs: Any) -> Any:
        if _is_write_routing(kwargs.get("routing_", RoutingControl.READ)):
            raise ReadOnlyViolation("execute_query with WRITE routing")
        kwargs["routing_"] = RoutingControl.READ
        return await super().execute_query(cypher_query_, **kwargs)

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        raise ReadOnlyViolation("index/constraint build on a read-only driver")

    def delete_all_indexes(self) -> Any:
        raise ReadOnlyViolation("delete_all_indexes on a read-only driver")
