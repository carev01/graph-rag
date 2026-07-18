from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Query
from neo4j import AsyncDriver, AsyncGraphDatabase

from answer_api import drift as drift_mod
from answer_api import global_search as global_mod
from answer_api import search as search_mod
from answer_api import synthesize as synth_mod
from answer_api import timeline as timeline_mod
from graph_extract.config import ExtractSettings, get_extract_settings
from graph_extract.graphiti_client import build_embedder, build_graphiti
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
    # Guard the window between building graphiti and the driver: if
    # _build_driver raises (e.g. a malformed neo4j_uri, validated
    # synchronously), the already-built graphiti would otherwise leak
    # (mirrors graph_extract/cli.py's build-or-cleanup pattern).
    try:
        driver = await _build_driver(settings)
    except Exception:
        await graphiti.close()
        raise
    # Same guard, extended: if building the synthesis client fails (e.g.
    # judge_base_url unset), close both already-built graphiti and driver
    # before re-raising.
    try:
        from answer_api.synthesize import _synthesis_client_and_model

        synth_client, synth_model = _synthesis_client_and_model(settings)
    except Exception:
        await graphiti.close()
        await driver.close()
        raise
    # Same guard, extended again: if building the embedder fails, close
    # graphiti, driver, and synth_client before re-raising.
    try:
        embedder = build_embedder(settings)
    except Exception:
        await graphiti.close()
        await driver.close()
        await synth_client.close()
        raise
    # Same guard, extended once more: if building the map client fails,
    # close graphiti, driver, synth_client, and the already-built embedder's
    # dedicated client before re-raising.
    try:
        map_client, map_model = global_mod._map_client_and_model(settings)
    except Exception:
        await graphiti.close()
        await driver.close()
        await synth_client.close()
        await embedder.client.close()
        raise
    app.state.settings = settings
    app.state.graphiti = graphiti
    app.state.driver = driver
    app.state.synth_client = synth_client
    app.state.synth_model = synth_model
    app.state.embedder = embedder
    app.state.map_client = map_client
    app.state.map_model = map_model
    try:
        yield
    finally:
        for name, closer in (
            ("graphiti", graphiti.close),
            ("driver", driver.close),
            ("synth_client", synth_client.close),
            ("map_client", map_client.close),
            ("embedder", lambda: embedder.client.close()),
        ):
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
        q: str, k: int = Query(10, ge=1), vendor: str | None = None,
        include_invalid: bool = False
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

    @app.get("/answer")
    async def answer(
        q: str, k: int = Query(15, ge=1), vendor: str | None = None
    ) -> dict[str, Any]:
        return await synth_mod.answer_local(
            app.state.graphiti,
            app.state.driver,
            app.state.synth_client,
            app.state.synth_model,
            q=q,
            k=k,
            vendor=vendor,
            group_id=app.state.settings.group_id,
        )

    @app.get("/search/global")
    async def search_global(
        q: str, level: int | None = Query(None, ge=0), k: int | None = Query(None, ge=1)
    ) -> dict[str, Any]:
        st = app.state
        return await global_mod.global_search(
            st.driver, st.embedder, st.map_client, st.map_model,
            st.synth_client, st.synth_model,
            q=q, level=st.settings.global_default_level if level is None else level,
            k=st.settings.global_shortlist_k if k is None else k,
            group_id=st.settings.group_id,
            relevance_min=st.settings.global_map_relevance_min)

    @app.get("/search/drift")
    async def search_drift(
        q: str, level: int | None = Query(None, ge=0),
        iterations: int | None = Query(None, ge=1, le=2)
    ) -> dict[str, Any]:
        st = app.state
        s = st.settings
        return await drift_mod.drift_search(
            st.graphiti, st.driver, st.embedder, st.synth_client, st.synth_model,
            q=q,
            level=s.drift_primer_level if level is None else level,
            iterations=s.drift_iterations if iterations is None else iterations,
            primer_k=s.drift_primer_k, max_followups=s.drift_max_followups,
            followup_k=s.drift_followup_k, group_id=s.group_id)

    @app.get("/timeline")
    async def timeline(
        q: str, limit: int = Query(30, ge=1), vendor: str | None = None
    ) -> dict[str, Any]:
        return await timeline_mod.timeline_local(
            app.state.graphiti,
            app.state.driver,
            q=q,
            limit=limit,
            vendor=vendor,
            group_id=app.state.settings.group_id,
        )

    return app


def main() -> FastAPI:
    """Production entrypoint: `uvicorn answer_api.app:main --factory`."""
    return create_app()


__all__ = ["create_app", "main", "Graphiti"]
