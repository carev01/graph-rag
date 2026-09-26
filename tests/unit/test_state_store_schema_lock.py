"""init_schema must take its advisory lock INSIDE the same transaction as the DDL,
so concurrent workers apply the schema one after another and the lock is released at
commit (a session lock on a pooled connection could outlive the call)."""
from __future__ import annotations

import pytest

from graph_sync import state_store as ss

pytestmark = pytest.mark.asyncio


class _Tx:
    def __init__(self, log):
        self.log = log

    async def __aenter__(self):
        self.log.append("BEGIN")

    async def __aexit__(self, *exc):
        self.log.append("COMMIT")


class _Conn:
    def __init__(self, log):
        self.log = log

    def transaction(self):
        return _Tx(self.log)

    async def execute(self, sql, *args):
        self.log.append(("LOCK", args) if "pg_advisory_xact_lock" in sql else "DDL")


class _Acquire:
    def __init__(self, conn):
        self.conn = conn

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *exc):
        return None


class _Pool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return _Acquire(self.conn)


async def test_init_schema_locks_then_runs_ddl_in_one_transaction(monkeypatch):
    log: list = []
    store = ss.StateStore("postgresql://unused")

    async def _pool():
        return _Pool(_Conn(log))
    monkeypatch.setattr(store, "_get_pool", _pool)

    await store.init_schema()

    assert log == ["BEGIN", ("LOCK", (ss._SCHEMA_LOCK_KEY,)), "DDL", "COMMIT"]
    assert ss._SCHEMA_LOCK_KEY not in (ss._LOCK_KEY, ss._WARMUP_LOCK_KEY)
