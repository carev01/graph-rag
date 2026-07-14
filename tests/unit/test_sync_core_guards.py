"""Unit tests for SyncCore's concurrency + cursor-safety guards (fakes only)."""
from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from graph_sync.sync_core import SyncCore

pytestmark = pytest.mark.asyncio


def _cursor(seq: int) -> str:
    return base64.b64encode(json.dumps({"seq": seq, "v": 1}).encode()).decode()


class _FakeStore:
    def __init__(self, *, cursor=None, watermarks=None):
        self._cursor = cursor
        self._watermarks = watermarks or []
        self.set_cursor_calls: list[str] = []
        self.lock_held = False

    async def try_lock(self) -> bool:
        self.lock_held = True
        return True

    async def unlock(self) -> None:
        self.lock_held = False

    async def get_cursor(self):
        return self._cursor

    async def set_cursor(self, c: str) -> None:
        self.set_cursor_calls.append(c)
        self._cursor = c

    async def upsert_bootstrap(self, *a) -> None:
        pass

    async def all_bootstrap_watermarks(self):
        return list(self._watermarks)


def _settings():
    return SimpleNamespace(webhook_debounce_seconds=0)


async def test_run_incremental_is_single_flight_in_process():
    # A blocking delta response keeps the first run_incremental in-flight; a
    # second concurrent call must skip (not run an overlapping sync).
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return httpx.Response(
            200, content=b'{"control":"cursor","next_since":"' + _cursor(9).encode() + b'","count":0}'
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x")
    store = _FakeStore(cursor=None)
    core = SyncCore(client, object(), object(), store, _settings())

    task_a = asyncio.create_task(core.run_incremental())
    await entered.wait()  # A is mid-stream, guard is set

    res_b = await core.run_incremental()  # must short-circuit (no overlapping run)
    assert res_b.advanced is False and res_b.applied == 0

    release.set()
    res_a = await task_a
    assert res_a.advanced is True  # A did the real work (advanced the cursor)
    # B's mid-sync nudge marked the run dirty, so A does ONE more pass rather
    # than dropping the trigger (spec §5.3/§7) -> two passes, two set_cursor calls.
    assert store.set_cursor_calls == [_cursor(9), _cursor(9)]
    await client.aclose()


def _bootstrap_client() -> httpx.AsyncClient:
    body = (
        b'{"control":"bootstrap_start","next_since":"' + _cursor(1).encode() + b'"}\n'
        b'{"control":"cursor","next_since":"' + _cursor(1).encode() + b'","count":0}'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x")


async def test_bootstrap_does_not_rewind_a_higher_cursor():
    # Steady-state cursor is high; a repair-bootstrap whose watermark decodes
    # lower must NOT regress it.
    client = _bootstrap_client()
    store = _FakeStore(cursor=_cursor(5000), watermarks=[_cursor(100)])
    core = SyncCore(client, object(), object(), store, _settings())
    await core.bootstrap(source_id="s1")
    assert store.set_cursor_calls == []  # cursor left untouched (no rewind)
    assert await store.get_cursor() == _cursor(5000)
    await client.aclose()


async def test_bootstrap_advances_a_lower_cursor():
    # The normal case still advances when the watermark is ahead of the cursor.
    client = _bootstrap_client()
    store = _FakeStore(cursor=_cursor(100), watermarks=[_cursor(5000)])
    core = SyncCore(client, object(), object(), store, _settings())
    await core.bootstrap(source_id="s1")
    assert store.set_cursor_calls == [_cursor(5000)]
    await client.aclose()
