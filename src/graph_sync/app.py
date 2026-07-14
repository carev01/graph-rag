from __future__ import annotations

import asyncio
import logging
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

from fastapi import APIRouter, FastAPI

from graph_sync.catalog import Catalog
from graph_sync.config import get_settings
from graph_sync.delta_client import make_client
from graph_sync.neo4j_repo import Neo4jRepo
from graph_sync.poll_loop import run_poll_loop
from graph_sync.state_store import StateStore
from graph_sync.sync_core import SyncCore
from graph_sync.webhook import build_router

logger = logging.getLogger(__name__)


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


class _SchemaResource(Protocol):
    """Structural shape of `Neo4jRepo`/`StateStore`: shared by `build_lifespan`
    so unit tests can pass fakes without depending on the real classes."""

    async def init_schema(self) -> None: ...
    async def close(self) -> None: ...


class _AsyncCloser(Protocol):
    async def aclose(self) -> None: ...


class _CatalogLike(Protocol):
    async def load(self) -> None: ...


def build_lifespan(
    *,
    repo: _SchemaResource,
    store: _SchemaResource,
    client: _AsyncCloser,
    trigger: Callable[[], Awaitable[None]],
    poll_interval_seconds: int,
    catalog: _CatalogLike | None = None,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Build the FastAPI lifespan for the live service, given already-built deps.

    Kept separate from `main()` (which builds the *real* deps) so the
    startup/shutdown sequencing can be exercised in unit tests with fakes
    instead of real Neo4j/Postgres/httpx.

    Startup: optionally `catalog.load()` (main() passes a real Catalog; unit
    tests can omit it), then `repo.init_schema()` and `store.init_schema()`,
    then starts the background poll task.

    Shutdown must be robust even if a poll cycle is mid-flight or already
    failed:
    - The poll task is *cancelled* (not just awaited) so an in-flight
      `trigger()` call (e.g. a slow httpx request) can't block shutdown, and
      any exception it already raised (or raises on cancellation) is caught
      and logged rather than propagated -- a failing poll cycle must never
      prevent cleanup.
    - The three resources are closed independently via `asyncio.gather(...,
      return_exceptions=True)` so a failure closing one doesn't skip the
      others; any close failures are logged, not raised.
    """

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        stop_event = asyncio.Event()
        poll_task: asyncio.Task[None] | None = None
        try:
            # Startup init is inside the try so a failure here (e.g. Neo4j down
            # at boot) still runs the cleanup below instead of leaking the
            # already-opened resources.
            if catalog is not None:
                await catalog.load()
            await repo.init_schema()
            await store.init_schema()
            poll_task = asyncio.create_task(
                run_poll_loop(trigger, poll_interval_seconds, stop_event)
            )
            yield
        finally:
            if poll_task is not None:
                stop_event.set()
                poll_task.cancel()
                try:
                    await poll_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("poll loop raised during shutdown")

            results = await asyncio.gather(
                repo.close(), store.close(), client.aclose(), return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.exception(
                        "error closing a resource during shutdown", exc_info=result
                    )

    return lifespan


def main() -> FastAPI:
    """Production entrypoint: `uvicorn graph_sync.app:main --factory`.

    Builds the real dependency graph (httpx client, Catalog, Neo4jRepo,
    StateStore, SyncCore), wires the webhook router, and delegates the
    startup/shutdown sequencing to `build_lifespan` and route composition to
    `create_app` so neither is duplicated here.
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
    lifespan = build_lifespan(
        repo=repo,
        store=store,
        client=client,
        trigger=trigger,
        poll_interval_seconds=settings.poll_interval_seconds,
        catalog=catalog,
    )

    return create_app(
        status_store=store,
        webhook_router=router,
        start_background=True,
        lifespan=lifespan,
    )
