from __future__ import annotations

import asyncio
import json
import logging

import httpx
import typer
from neo4j import AsyncDriver, AsyncGraphDatabase

from graph_extract.config import ExtractSettings, get_extract_settings
from graph_extract.eval import cost_report, dedup_report, fact_quality, provenance_report
from graph_extract.graphiti_client import build_graphiti, init_indices
from graph_extract.ingest_driver import IngestDriver
from graph_extract.probe import DEFAULT_MODES, run_probe
from graph_extract.provenance import Provenance
from graph_sync.delta_client import make_client

app = typer.Typer()
eval_app = typer.Typer()
app.add_typer(eval_app, name="eval")
logger = logging.getLogger(__name__)

# Task 11 spec: canonical dedup targets for the pilot eval -- concepts that
# should merge to a single :Entity node, and pairs that must stay distinct.
DEFAULT_CANON_MERGE = ["immutability", "cross-region copy", "Kubernetes", "RPO"]
DEFAULT_DISTINCT_PAIRS = [
    ("Amazon S3", "Azure Blob Storage"),
    ("AWS Backup", "Azure Backup"),
]


def _dump(obj: object) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str))


async def _build_ingest_driver(
    settings: ExtractSettings,
) -> tuple[IngestDriver, object, httpx.AsyncClient, AsyncDriver]:
    """Construct the real dependency graph the `ingest` command needs.

    If any step after opening a resource fails, every resource already
    opened here is closed before the exception propagates -- without this,
    a failure partway through construction would return nothing to the
    caller, so the caller's own `try/finally` (which only covers the code
    *after* this function returns) would never run and the already-open
    graphiti/client/driver would leak.
    """
    graphiti = build_graphiti(settings)
    docext: httpx.AsyncClient | None = None
    driver: AsyncDriver | None = None
    try:
        docext = make_client(settings, admin=False)
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        await init_indices(graphiti)
        provenance = Provenance(driver)
        ingest = IngestDriver(settings, graphiti, docext, provenance, driver)
    except Exception:
        for closer in (
            driver.close if driver is not None else None,
            docext.aclose if docext is not None else None,
            graphiti.close,
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception:
                logger.exception(
                    "error closing a resource while cleaning up after a "
                    "_build_ingest_driver failure"
                )
        raise
    return ingest, graphiti, docext, driver


async def _build_driver(settings: ExtractSettings) -> AsyncDriver:
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


@app.command("ingest")
def ingest(
    source_id: str = typer.Option(..., "--source-id"),
    limit: int | None = typer.Option(None, "--limit"),
) -> None:
    async def _run() -> None:
        settings = get_extract_settings()
        ingest_driver, graphiti, docext, driver = await _build_ingest_driver(settings)
        try:
            res = await ingest_driver.ingest_source(source_id, limit)
            typer.echo(
                f"ingest complete: articles={res.articles} "
                f"episodes_added={res.episodes_added} "
                f"episodes_skipped={res.episodes_skipped}"
            )
        finally:
            await driver.close()
            await docext.aclose()
            await graphiti.close()

    asyncio.run(_run())


@eval_app.command("dedup")
def eval_dedup() -> None:
    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            report = await dedup_report(
                driver, settings.group_id, DEFAULT_CANON_MERGE, DEFAULT_DISTINCT_PAIRS
            )
            _dump(report)
        finally:
            await driver.close()

    asyncio.run(_run())


@eval_app.command("provenance")
def eval_provenance(sample: int = typer.Option(20, "--sample")) -> None:
    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            report = await provenance_report(driver, sample)
            _dump(report)
        finally:
            await driver.close()

    asyncio.run(_run())


@eval_app.command("cost")
def eval_cost(
    episodes_processed: int = typer.Option(0, "--episodes-processed"),
) -> None:
    async def _run() -> None:
        report = await cost_report(episodes_processed)
        _dump(report)

    asyncio.run(_run())


@eval_app.command("quality")
def eval_quality(sample: int = typer.Option(20, "--sample")) -> None:
    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            report = await fact_quality(driver, settings, sample)
            _dump(report)
        finally:
            await driver.close()

    asyncio.run(_run())


@app.command("probe")
def probe(
    article_ids: str = typer.Option(..., "--article-ids", help="Comma-separated article IDs"),
    n: int = typer.Option(3, "--n", help="Chunks per article"),
) -> None:
    async def _run() -> None:
        settings = get_extract_settings()
        ids = [a.strip() for a in article_ids.split(",") if a.strip()]
        report = await run_probe(settings, ids, n, DEFAULT_MODES)
        _dump(report)

    asyncio.run(_run())


if __name__ == "__main__":
    app()
