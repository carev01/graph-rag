import asyncio
import pytest
from graph_sync.poll_loop import run_poll_loop

pytestmark = pytest.mark.asyncio


async def test_run_poll_loop_calls_trigger_until_stopped():
    calls = 0
    stop = asyncio.Event()

    async def trigger() -> None:
        nonlocal calls
        calls += 1
        if calls >= 3:
            stop.set()

    await run_poll_loop(trigger, interval_seconds=0, stop=stop)
    assert calls == 3


async def test_run_poll_loop_noop_when_already_stopped():
    calls = 0
    stop = asyncio.Event()
    stop.set()

    async def trigger() -> None:
        nonlocal calls
        calls += 1

    await run_poll_loop(trigger, interval_seconds=0, stop=stop)
    assert calls == 0
