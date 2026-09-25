from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

from fastapi import FastAPI, HTTPException, Query
from neo4j import AsyncDriver, AsyncGraphDatabase

from answer_api import drift as drift_mod
from answer_api import global_search as global_mod
from answer_api import router as router_mod
from answer_api import search as search_mod
from answer_api import timeline as timeline_mod
from answer_api.router import Mode
from answer_api.scope import Scope, ScopeResolver, UnknownScopeName
from graph_extract.config import ExtractSettings, get_extract_settings
from graph_extract.graphiti_client import build_embedder, build_graphiti
from graph_extract.vector_search import ensure_vector_indexes
from graph_sync.logging_setup import configure_logging
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
    # Same guard: the ScopeResolver is loaded from the structural layer over
    # the just-built driver, so a failure here closes both already-built
    # resources before re-raising.
    try:
        scope_resolver = await ScopeResolver.load(driver)
    except Exception:
        await graphiti.close()
        await driver.close()
        raise
    # Verify (and on a fresh database create) the tuned vector indexes before
    # serving: a mismatch refuses to start rather than serving full scans.
    if settings.vector_search_enabled:
        try:
            await ensure_vector_indexes(
                graphiti.driver, settings.embed_dim,
                wait_seconds=settings.vector_index_startup_wait_seconds)
        except Exception:
            await graphiti.close()
            await driver.close()
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
    cheap_client, cheap_model = router_mod._cheap_classify_client(settings)
    app.state.settings = settings
    app.state.graphiti = graphiti
    app.state.driver = driver
    app.state.scope_resolver = scope_resolver
    app.state.scope_loaded_at = time.monotonic()
    app.state.synth_client = synth_client
    app.state.synth_model = synth_model
    app.state.embedder = embedder
    app.state.map_client = map_client
    app.state.map_model = map_model
    app.state.cheap_client = cheap_client
    app.state.cheap_model = cheap_model
    try:
        yield
    finally:
        closers = [
            ("graphiti", graphiti.close),
            ("driver", driver.close),
            ("synth_client", synth_client.close),
            ("map_client", map_client.close),
            ("embedder", lambda: embedder.client.close()),
        ]
        if cheap_client is not None:
            closers.append(("cheap_client", cheap_client.close))
        for name, closer in closers:
            try:
                await closer()
            except Exception:
                logger.exception("error closing %s during shutdown", name)


async def _resolve_scope(app: FastAPI, q: str, vendor: list[str] | None,
                         product: list[str] | None, scope_mode: str) -> Scope:
    """Resolve once, at the route boundary, then pass the same Scope to every
    mode. The resolver reloads lazily on use when older than
    `settings.scope_reload_seconds` (plan ruling 4) rather than on a timer, so
    an idle service never reloads and a busy one reloads at most once per TTL.
    """
    settings = app.state.settings
    now = time.monotonic()
    if now - app.state.scope_loaded_at > settings.scope_reload_seconds:
        app.state.scope_resolver = await ScopeResolver.load(app.state.driver)
        app.state.scope_loaded_at = now
    try:
        return app.state.scope_resolver.resolve(
            q, vendors=vendor, products=product, disabled=(scope_mode == "none"))
    except UnknownScopeName as e:
        raise HTTPException(422, detail={"unknown": e.names}) from e


def create_app() -> FastAPI:
    app = FastAPI(title="answer-api", lifespan=_lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/search/local")
    async def search_local(
        q: str, k: int = Query(10, ge=1),
        vendor: list[str] | None = Query(None), product: list[str] | None = Query(None),
        scope: Literal["auto", "none"] = "auto",
        include_invalid: bool = False
    ) -> dict[str, Any]:
        resolved = await _resolve_scope(app, q, vendor, product, scope)
        result = await search_mod.search_local(
            app.state.graphiti,
            app.state.driver,
            q=q,
            k=k,
            scope=resolved,
            include_invalid=include_invalid,
            group_id=app.state.settings.group_id,
        )
        result["scope"] = resolved.as_dict()
        return result

    @app.get("/answer")
    async def answer(
        q: str, mode: Mode | None = None,
        vendor: list[str] | None = Query(None), product: list[str] | None = Query(None),
        scope: Literal["auto", "none"] = "auto",
    ) -> dict[str, Any]:
        st = app.state
        resolved = await _resolve_scope(app, q, vendor, product, scope)
        return await router_mod.answer_router(
            st.graphiti, st.driver, st.embedder, st.synth_client, st.synth_model,
            st.map_client, st.map_model, st.cheap_client, st.cheap_model,
            q=q, mode_override=mode, scope=resolved, settings=st.settings)

    @app.get(
        "/search/global",
        description="Community map-reduce search. Note: `k` only bounds the "
        "pre-rerank shortlist -- once a reranker is configured (rerank_base_url "
        "+ rerank_model), rerank_top_n decides how many communities actually "
        "reach the map step, not `k`.",
    )
    async def search_global(
        q: str, level: int | None = Query(None, ge=0), k: int | None = Query(None, ge=1),
        vendor: list[str] | None = Query(None), product: list[str] | None = Query(None),
        scope: Literal["auto", "none"] = "auto",
    ) -> dict[str, Any]:
        st = app.state
        resolved = await _resolve_scope(app, q, vendor, product, scope)
        result = await global_mod.global_search(
            st.driver, st.embedder, st.map_client, st.map_model,
            st.synth_client, st.synth_model,
            q=q, level=st.settings.global_default_level if level is None else level,
            k=st.settings.global_shortlist_k if k is None else k,
            group_id=st.settings.group_id,
            settings=st.settings, scope=resolved)
        result["scope"] = resolved.as_dict()
        return result

    @app.get(
        "/search/drift",
        description="DRIFT primer->follow-up->synthesis search. Note: the "
        "settings-configured drift_primer_k only bounds the pre-rerank primer "
        "shortlist -- once a reranker is configured, rerank_top_n decides how "
        "many communities feed the primer, not drift_primer_k.",
    )
    async def search_drift(
        q: str, level: int | None = Query(None, ge=0),
        iterations: int | None = Query(None, ge=1, le=2),
        vendor: list[str] | None = Query(None), product: list[str] | None = Query(None),
        scope: Literal["auto", "none"] = "auto",
    ) -> dict[str, Any]:
        st = app.state
        s = st.settings
        resolved = await _resolve_scope(app, q, vendor, product, scope)
        result = await drift_mod.drift_search(
            st.graphiti, st.driver, st.embedder, st.synth_client, st.synth_model,
            q=q,
            level=s.drift_primer_level if level is None else level,
            iterations=s.drift_iterations if iterations is None else iterations,
            primer_k=s.drift_primer_k, max_followups=s.drift_max_followups,
            followup_k=s.drift_followup_k, group_id=s.group_id, settings=s,
            scope=resolved)
        result["scope"] = resolved.as_dict()
        return result

    @app.get("/timeline")
    async def timeline(
        q: str, limit: int = Query(30, ge=1),
        vendor: list[str] | None = Query(None), product: list[str] | None = Query(None),
        scope: Literal["auto", "none"] = "auto",
    ) -> dict[str, Any]:
        resolved = await _resolve_scope(app, q, vendor, product, scope)
        result = await timeline_mod.timeline_local(
            app.state.graphiti,
            app.state.driver,
            q=q,
            limit=limit,
            scope=resolved,
            group_id=app.state.settings.group_id,
        )
        result["scope"] = resolved.as_dict()
        return result

    return app


def main() -> FastAPI:
    """Production entrypoint: `uvicorn answer_api.app:main --factory`."""
    configure_logging()
    return create_app()


__all__ = ["create_app", "main", "Graphiti"]
