import asyncio

import pytest
from httpx import ASGITransport, AsyncClient
from graph_sync.app import build_lifespan, create_app

pytestmark = pytest.mark.asyncio

class FakeStatusStore:
    async def get_cursor(self): return "cur-1"
    async def dead_letter_count(self): return 0

async def test_health_and_status():
    app = create_app(status_store=FakeStatusStore(), start_background=False)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        assert (await c.get("/health")).json() == {"status": "ok"}
        s = (await c.get("/status")).json()
        assert s["cursor"] == "cur-1" and s["dead_letter_count"] == 0


class FakeSchemaResource:
    """Stands in for Neo4jRepo/StateStore: records init_schema/close calls."""

    def __init__(self) -> None:
        self.init_schema_calls = 0
        self.close_calls = 0

    async def init_schema(self) -> None:
        self.init_schema_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


class FakeClient:
    """Stands in for httpx.AsyncClient: records aclose calls."""

    def __init__(self) -> None:
        self.aclose_calls = 0

    async def aclose(self) -> None:
        self.aclose_calls += 1


async def test_lifespan_shutdown_survives_a_failing_poll_trigger():
    """Pins the Important shutdown-robustness fix.

    A poll `trigger` that raises on every invocation must not stop shutdown
    from closing repo/store/client, and shutdown must not hang even though
    the poll task ended (or is ending) with an exception.
    """
    repo = FakeSchemaResource()
    store = FakeSchemaResource()
    client = FakeClient()

    async def failing_trigger() -> None:
        raise RuntimeError("boom: poll cycle always fails")

    lifespan = build_lifespan(
        repo=repo,
        store=store,
        client=client,
        trigger=failing_trigger,
        poll_interval_seconds=0,
    )

    async def drive() -> None:
        async with lifespan(None):
            assert repo.init_schema_calls == 1
            assert store.init_schema_calls == 1
            # Give the background poll task a chance to run `failing_trigger`
            # and end in an exception state before we trigger shutdown.
            await asyncio.sleep(0.05)

    try:
        # `wait_for` also pins "shutdown must not hang": without the fix,
        # `await poll_task` in the shutdown path re-raises the poll task's
        # RuntimeError immediately (no hang) but skips every close() call
        # below it -- caught here so the close-call assertions can run.
        await asyncio.wait_for(drive(), timeout=2)
    except asyncio.TimeoutError:
        pytest.fail("lifespan shutdown hung instead of completing promptly")
    except Exception:
        pass

    assert repo.close_calls == 1
    assert store.close_calls == 1
    assert client.aclose_calls == 1
