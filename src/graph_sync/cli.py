from __future__ import annotations

import asyncio
import json
import logging
import secrets
import signal

import httpx
import typer
from graphiti_core import Graphiti
from neo4j import AsyncDriver

from graph_extract.cli import _build_ingest_driver
from graph_extract.config import get_extract_settings
from graph_extract.ingest_driver import IngestDriver
from graph_extract.warmup import WarmupGate
from graph_sync.catalog import Catalog
from graph_sync.config import Settings, get_settings
from graph_sync.delta_client import make_client
from graph_sync.neo4j_repo import Neo4jRepo
from graph_sync.semantic_worker import run_worker
from graph_sync.state_store import StateStore
from graph_sync.sync_core import SyncCore

app = typer.Typer()
logger = logging.getLogger(__name__)


def _dump(obj: object) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


async def _build_sync_core(
    settings: Settings,
) -> tuple[SyncCore, httpx.AsyncClient, Neo4jRepo, StateStore]:
    """Construct the real dependency graph a one-shot CLI command needs.

    Mirrors `app.main()`'s lifespan startup (load catalog, init schemas) so
    CLI runs and the live service see the same bootstrap sequence.

    If any step after opening a resource fails (including `catalog.load()`/
    `init_schema()`), every resource already opened here is closed before
    the exception propagates. Without this, a failure partway through
    construction would return nothing to the caller, so the caller's own
    `try/finally` (which only covers the code *after* this function
    returns) would never run and the already-open client/driver/pool would
    leak.
    """
    client = make_client(settings, admin=False)
    repo: Neo4jRepo | None = None
    store: StateStore | None = None
    try:
        catalog = Catalog(client)
        repo = Neo4jRepo(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        store = StateStore(settings.postgres_dsn)
        await catalog.load()
        await repo.init_schema()
        await store.init_schema()
    except Exception:
        for closer in (
            store.close if store is not None else None,
            repo.close if repo is not None else None,
            client.aclose,
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception:
                logger.exception(
                    "error closing a resource while cleaning up after a "
                    "_build_sync_core failure"
                )
        raise
    core = SyncCore(client, catalog, repo, store, settings)
    return core, client, repo, store


async def _build_worker_deps(
    settings: Settings,
) -> tuple[StateStore, IngestDriver, Graphiti, httpx.AsyncClient, AsyncDriver]:
    """Construct the real dependency graph the `worker` command needs.

    `_build_ingest_driver` already cleans up its own partial state (docext,
    driver, graphiti) on failure, but it has no knowledge of `store`. If
    `store.init_schema()` or `_build_ingest_driver` itself fails after
    `store`'s connection pool is already open, that pool must still be
    closed here before the exception propagates -- otherwise the caller's
    own `try/finally` (which only covers the code *after* this function
    returns) would never run and the already-open pool would leak.
    """
    store = StateStore(settings.postgres_dsn)
    try:
        await store.init_schema()
        ingest, graphiti, docext, driver = await _build_ingest_driver(get_extract_settings())
    except Exception:
        try:
            await store.close()
        except Exception:
            logger.exception(
                "error closing the state store while cleaning up after a "
                "_build_worker_deps failure"
            )
        raise
    return store, ingest, graphiti, docext, driver


@app.command("register-webhook")
def register_webhook() -> None:
    s = get_settings()
    secret = s.webhook_secret or secrets.token_urlsafe(32)
    async def _run() -> None:
        async with make_client(s, admin=True) as client:
            resp = await client.post("/api/webhooks", json={
                "url": s.webhook_public_url,
                "events": ["extraction_complete"],
                "secret": secret, "is_active": True})
            resp.raise_for_status()
            typer.echo(f"registered; secret={secret}")
    asyncio.run(_run())


@app.command("bootstrap")
def bootstrap(
    source_id: str | None = typer.Option(None, "--source-id"),
    vendor_id: str | None = typer.Option(None, "--vendor-id"),
) -> None:
    async def _run() -> None:
        settings = get_settings()
        core, client, repo, store = await _build_sync_core(settings)
        try:
            res = await core.bootstrap(source_id=source_id, vendor_id=vendor_id)
            typer.echo(
                f"bootstrap complete: applied={res.applied} skipped={res.skipped} "
                f"sources={len(res.sources)}"
            )
        finally:
            await repo.close()
            await store.close()
            await client.aclose()

    asyncio.run(_run())


@app.command("sync-once")
def sync_once() -> None:
    async def _run() -> None:
        settings = get_settings()
        core, client, repo, store = await _build_sync_core(settings)
        try:
            res = await core.run_incremental()
            typer.echo(
                f"sync-once complete: applied={res.applied} removed={res.removed} "
                f"skipped={res.skipped} advanced={res.advanced} sources={len(res.sources)}"
            )
        finally:
            await repo.close()
            await store.close()
            await client.aclose()

    asyncio.run(_run())


@app.command("refresh-toc")
def refresh_toc(source_id: str = typer.Argument(...)) -> None:
    async def _run() -> None:
        settings = get_settings()
        core, client, repo, store = await _build_sync_core(settings)
        try:
            await core.refresh_toc(source_id)
            typer.echo(f"refresh-toc complete for source_id={source_id}")
        finally:
            await repo.close()
            await store.close()
            await client.aclose()

    asyncio.run(_run())


@app.command("queue-status")
def queue_status() -> None:
    """Dead-letter / queue observability: print `semantic_jobs` counts by status."""

    async def _run() -> None:
        settings = get_settings()
        store = StateStore(settings.postgres_dsn)
        try:
            await store.init_schema()
            _dump(await store.job_status_counts())
        finally:
            await store.close()

    asyncio.run(_run())


@app.command("worker")
def worker(
    batch: int = typer.Option(10, "--batch"),
    poll_seconds: float = typer.Option(5.0, "--poll-seconds"),
    max_batches: int | None = typer.Option(
        None, "--max-batches",
        help="Stop after N batches instead of running until SIGINT/SIGTERM. "
             "Use this for bounded/test runs: `timeout` does not forward "
             "SIGTERM through `uv run`, so an unbounded worker can outlive "
             "its intended window and claim jobs you did not mean to process."),
) -> None:
    """Standalone semantic-ingestion worker: claims `semantic_jobs` rows and
    drives them through the real `IngestDriver` (upsert -> ingest_article,
    remove -> tombstone_article_episodes). Runs until SIGINT/SIGTERM."""

    async def _run() -> None:
        settings = get_settings()
        store, ingest, graphiti, docext, driver = await _build_worker_deps(settings)
        try:
            stop_event = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, stop_event.set)
            # The fan-out and warm-up knobs live on ExtractSettings, not
            # graph_sync's Settings: it is the same object `_build_worker_deps`
            # gave the driver (get_extract_settings is lru_cached), so the
            # worker and the driver path read one value.
            extract = get_extract_settings()
            # ingest_warmup_articles=0 is OFF: no predicate at all, so the
            # fan-out takes the byte-for-byte pre-warm-up path. The worker's
            # predicate is per ARTICLE (a claimed batch spans sources and has
            # no notion of a start); the gate resolves each to its source.
            is_cold = (
                WarmupGate(driver, extract.group_id, extract.ingest_warmup_articles)
                .is_cold_article
                if extract.ingest_warmup_articles > 0 else None)
            await run_worker(
                store, ingest, batch=batch, poll_seconds=poll_seconds, stop_event=stop_event,
                max_batches=max_batches,
                budget=settings.semantic_daily_token_budget,
                max_attempts=settings.semantic_max_attempts,
                backoff_base=settings.semantic_backoff_base_seconds,
                backoff_cap=settings.semantic_backoff_cap_seconds,
                lease=settings.semantic_reaper_lease_seconds,
                concurrency=extract.ingest_article_concurrency,
                is_cold=is_cold,
            )
        finally:
            await store.close()
            await driver.close()
            await docext.aclose()
            await graphiti.close()
            if ingest._cheap is not None:
                await ingest._cheap.graphiti.close()

    asyncio.run(_run())


if __name__ == "__main__":
    app()
