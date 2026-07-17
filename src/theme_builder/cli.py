"""theme-build: detect communities -> per community assemble context + generate a
fact-cited report -> write the :Community subgraph. Full rebuild."""
from __future__ import annotations

import asyncio
import json
import logging

import typer
from neo4j import AsyncDriver, AsyncGraphDatabase

from graph_extract.config import ExtractSettings, get_extract_settings
from graph_extract.graphiti_client import build_embedder
from theme_builder.context import EntityRow, FactRow, assemble_context
from theme_builder.detect import detect_communities
from theme_builder.report import _report_client_and_model, generate_report
from theme_builder.writeback import write_communities

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.callback()
def _root() -> None:
    """Community-report layer (theme-builder) commands."""
    # Presence of a callback keeps typer in multi-command mode, so the single
    # `theme-build` command must be named explicitly on the CLI.


async def _fetch_members(driver: AsyncDriver, group_id: str, uuids: list[str]) -> list[EntityRow]:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) WHERE e.uuid IN $uuids "
            "OPTIONAL MATCH (e)-[r:RELATES_TO {group_id:$g}]-(m:Entity) WHERE m.uuid IN $uuids "
            "WITH e, count(r) AS degree "
            "RETURN e.uuid AS uuid, e.name AS name, "
            "coalesce([l IN labels(e) WHERE l<>'Entity'][0],'Entity') AS type, "
            "coalesce(e.summary,'') AS summary, degree",
            g=group_id, uuids=uuids)
        return [EntityRow(uuid=x["uuid"], name=x["name"], type=x["type"],
                          summary=x["summary"], degree=x["degree"]) async for x in r]


async def _fetch_facts(driver: AsyncDriver, group_id: str, uuids: list[str]) -> list[FactRow]:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (a:Entity {group_id:$g})-[f:RELATES_TO {group_id:$g}]->(b:Entity {group_id:$g}) "
            "WHERE a.uuid IN $uuids AND b.uuid IN $uuids "
            "RETURN f.uuid AS uuid, f.fact AS fact, toString(f.valid_at) AS valid_at, "
            "toString(f.invalid_at) AS invalid_at, f.name AS name",
            g=group_id, uuids=uuids)
        return [FactRow(uuid=x["uuid"], fact=x["fact"], valid_at=x["valid_at"],
                        invalid_at=x["invalid_at"], name=x["name"]) async for x in r]


async def _run_theme_build(settings: ExtractSettings, *, driver: AsyncDriver) -> dict:
    communities = await detect_communities(
        driver, settings.group_id,
        min_community_size=settings.leiden_min_community_size,
        max_levels=settings.leiden_max_levels)
    client, model = _report_client_and_model(settings)
    embedder = build_embedder(settings)
    reports: dict = {}
    skipped = 0
    try:
        for c in communities:
            try:
                members = await _fetch_members(driver, settings.group_id, c.member_uuids)
                facts = await _fetch_facts(driver, settings.group_id, c.member_uuids)
                ctx = assemble_context(members, facts,
                                       top_entities=settings.report_top_entities,
                                       token_budget=settings.report_token_budget)
                rep = await generate_report(client, model, ctx)
            except Exception:
                logger.exception("theme-build: community %s errored; skipping", c.community_id)
                rep = None
            if rep is None:
                logger.info("theme-build: community %s produced no report; skipping", c.community_id)
                skipped += 1
                continue
            reports[c.community_id] = rep
        res = await write_communities(driver, embedder, settings.group_id,
                                      communities, reports, corpus_cursor=None)
        res["communities_detected"] = len(communities)
        res["reports_skipped"] = skipped
        return res
    finally:
        # close both owned clients (report LLM + embedder) so a repeated caller
        # doesn't leak httpx pools; the injected driver is the caller's to close.
        await client.close()
        await embedder.client.close()


@app.command("theme-build")
def theme_build() -> None:
    """Full rebuild of the community-report layer for the configured group."""
    async def _main() -> None:
        settings = get_extract_settings()
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        try:
            res = await _run_theme_build(settings, driver=driver)
            typer.echo(json.dumps(res, indent=2, default=str))
        finally:
            await driver.close()

    asyncio.run(_main())


if __name__ == "__main__":
    app()
