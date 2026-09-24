from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


async def run_poll_loop(trigger: Callable[[], Awaitable[None]],
                        interval_seconds: int, stop: asyncio.Event) -> None:
    """Call `trigger` every `interval_seconds` until `stop` is set.

    A failing poll (DocExtractor unreachable, Neo4j/Postgres blip) is logged and
    retried at the next interval -- it must never end the loop: the poll task
    would die silently while `/health` kept reporting ok. `CancelledError` is a
    `BaseException`, not an `Exception`, so shutdown still propagates.
    """
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            try:
                await trigger()
            except Exception:
                logger.exception("poll trigger failed; retrying in %ss", interval_seconds)
