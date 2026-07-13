from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import APIRouter, FastAPI

from graph_sync.catalog import Catalog
from graph_sync.config import get_settings
from graph_sync.delta_client import make_client
from graph_sync.neo4j_repo import Neo4jRepo
from graph_sync.poll_loop import run_poll_loop
from graph_sync.state_store import StateStore
from graph_sync.sync_core import SyncCore
from graph_sync.webhook import build_router


def create_app(
    *,
    status_store: Any = None,
    webhook_router: APIRouter | None = None,
    start_background: bool = True,
    lifespan: Any = None,
) -> FastAPI:
    """Test-focused seam: accepts already-built deps rather than building its own.

    `status_store`/`webhook_router` let unit tests inject fakes; `lifespan` lets
    `main()` (below) supply the real startup/shutdown wiring without duplicating
    the route definitions here. `start_background` is accepted for interface
    parity with the brief -- the actual background poll-loop task is started by
    the `lifespan` passed in from `main()`, not by this function directly, so it
    is a no-op when `lifespan` is None (the unit-test path).
    """
    app = FastAPI(title="graph-sync", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/status")
    async def status() -> dict[str, Any]:
        if status_store is None:
            return {"cursor": None, "dead_letter_count": None}
        return {
            "cursor": await status_store.get_cursor(),
            "dead_letter_count": await status_store.dead_letter_count(),
        }

    if webhook_router is not None:
        app.include_router(webhook_router)
    return app


def main() -> FastAPI:
    """Production entrypoint: `uvicorn graph_sync.app:main --factory`.

    Builds the real dependency graph (httpx client, Catalog, Neo4jRepo,
    StateStore, SyncCore), wires the webhook router and the background poll
    loop through a FastAPI lifespan, and delegates route composition to
    `create_app` so the `/health` and `/status` wiring isn't duplicated.
    """
    settings = get_settings()
    client = make_client(settings, admin=False)
    catalog = Catalog(client)
    repo = Neo4jRepo(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    store = StateStore(settings.postgres_dsn)
    sync_core = SyncCore(client, catalog, repo, store, settings)

    async def trigger() -> None:
        await sync_core.run_incremental()

    router = build_router(store, settings, trigger)
    stop_event = asyncio.Event()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await catalog.load()
        await repo.init_schema()
        await store.init_schema()
        poll_task = asyncio.create_task(
            run_poll_loop(trigger, settings.poll_interval_seconds, stop_event)
        )
        try:
            yield
        finally:
            stop_event.set()
            await poll_task
            await repo.close()
            await store.close()
            await client.aclose()

    return create_app(
        status_store=store,
        webhook_router=router,
        start_background=True,
        lifespan=lifespan,
    )
