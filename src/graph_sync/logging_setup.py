"""One logging setup for every graph-rag process entrypoint.

Used by `graph_sync.cli` (worker / sync-once / bootstrap), `graph_sync.app.main()`
and `answer_api.app.main()` (the uvicorn factories), so INFO lines from the
poller, the worker and the answer API all reach `kubectl logs`.
"""
from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    """Make INFO-level log lines (the worker's `semantic batch: ...` summary,
    dedup and vector-search counters, timings) actually reach stdout /
    `kubectl logs`.

    Nothing in this package ever calls `logging.basicConfig`: the root logger's
    default level is WARNING and it has no handler, so every `logger.info(...)`
    call is silently dropped -- not filtered on purpose, just nowhere to go.
    `LOG_LEVEL` overrides the default; an unrecognised value falls back to INFO
    rather than raising.

    Guarded against double-configuring: if a handler is already attached (a
    parent process, a test harness, or a previous call in this process already
    did it), do nothing rather than fight whatever level/format it chose.

    `httpx`/`httpcore` log one line per HTTP request at INFO -- every LLM,
    embedder, and DocExtractor call the worker makes during a bootstrap -- which
    would otherwise drown the handful of lines that actually matter. Quieted to
    WARNING unconditionally, independent of the guard above, since this is a
    noise fix, not a "did I configure the root logger" concern. The Neo4j
    driver's `neo4j.notifications` logger gets the same treatment: it logs every
    server notification at INFO (`IF NOT EXISTS` schema no-ops on each start, a
    cartesian-product hint per structural batch); its WARNING-level
    notifications still get through.
    """
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)

    root = logging.getLogger()
    if root.handlers:
        return
    level = logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO").upper())
    if not isinstance(level, int):
        level = logging.INFO
    # force=True: without it, basicConfig's own built-in "do nothing if the root
    # logger already has handlers" check would make the guard above redundant
    # (harmless, but untestable -- a mutation deleting it would change nothing
    # observable). With force=True, this call WOULD unconditionally strip any
    # existing handler and reset the level if reached, so the guard above is the
    # only thing standing between "leave it alone" and "reconfigure it".
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
