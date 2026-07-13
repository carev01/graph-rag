from __future__ import annotations
import asyncio
from typing import Awaitable, Callable

async def run_poll_loop(trigger: Callable[[], Awaitable[None]],
                        interval_seconds: int, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            await trigger()
