from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from neo4j import AsyncDriver, AsyncGraphDatabase

from answer_api import search as search_mod
from graph_extract.config import ExtractSettings, get_extract_settings
from graph_extract.graphiti_client import build_graphiti
from graphiti_core import Graphiti

logger = logging.getLogger(__name__)


async def _build_driver(settings: ExtractSettings) -> AsyncDriver:
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build one Graphiti + one Neo4j AsyncDriver on startup, store them on
    `app.state`, and close both on shutdown.

    Shutdown mirrors `graph_sync.app`'s lifespan: each close is attempted
    independently and a failure is logged rather than propagated, so a
    failure closing one resource never skips closing the other.
    """
    settings = get_extract_settings()
    graphiti = build_graphiti(settings)
    driver = await _build_driver(settings)
    app.state.settings = settings
    app.state.graphiti = graphiti
    app.state.driver = driver
    try:
        yield
    finally:
        for name, closer in (("graphiti", graphiti.close), ("driver", driver.close)):
            try:
                await closer()
            except Exception:
                logger.exception("error closing %s during shutdown", name)


def create_app() -> FastAPI:
    app = FastAPI(title="answer-api", lifespan=_lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/search/local")
    async def search_local(
        q: str, k: int = 10, vendor: str | None = None, include_invalid: bool = False
    ) -> dict[str, Any]:
        return await search_mod.search_local(
            app.state.graphiti,
            app.state.driver,
            q=q,
            k=k,
            vendor=vendor,
            include_invalid=include_invalid,
            group_id=app.state.settings.group_id,
        )

    return app


def main() -> FastAPI:
    """Production entrypoint: `uvicorn answer_api.app:main --factory`."""
    return create_app()


__all__ = ["create_app", "main", "Graphiti"]
