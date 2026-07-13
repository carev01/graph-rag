from __future__ import annotations

import asyncio
import secrets

import httpx
import typer

from graph_sync.catalog import Catalog
from graph_sync.config import Settings, get_settings
from graph_sync.delta_client import make_client
from graph_sync.neo4j_repo import Neo4jRepo
from graph_sync.state_store import StateStore
from graph_sync.sync_core import SyncCore

app = typer.Typer()


async def _build_sync_core(
    settings: Settings,
) -> tuple[SyncCore, httpx.AsyncClient, Neo4jRepo, StateStore]:
    """Construct the real dependency graph a one-shot CLI command needs.

    Mirrors `app.main()`'s lifespan startup (load catalog, init schemas) so
    CLI runs and the live service see the same bootstrap sequence.
    """
    client = make_client(settings, admin=False)
    catalog = Catalog(client)
    repo = Neo4jRepo(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    store = StateStore(settings.postgres_dsn)
    await catalog.load()
    await repo.init_schema()
    await store.init_schema()
    core = SyncCore(client, catalog, repo, store, settings)
    return core, client, repo, store


@app.command("register-webhook")
def register_webhook() -> None:
    s = get_settings()
    secret = s.webhook_secret or secrets.token_urlsafe(32)
    async def _run() -> None:
        async with make_client(s, admin=True) as client:
            resp = await client.post("/api/webhooks", json={
                "url": s.webhook_public_url,
                "events": "extraction_complete",
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


if __name__ == "__main__":
    app()
