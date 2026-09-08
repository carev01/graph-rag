"""theme-build: detect communities -> per community assemble context + generate a
fact-cited report -> write the :Community subgraph. Full rebuild."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

import typer
from neo4j import AsyncDriver, AsyncGraphDatabase

from graph_extract.config import ExtractSettings, get_extract_settings
from graph_extract.graphiti_client import build_embedder
from theme_builder.context import EntityRow, FactRow, assemble_context
from theme_builder.detect import detect_communities
from theme_builder.incremental import (
    classify, load_persisted, match_communities, prev_corpus_cursor, touched_entities)
from theme_builder.report import _report_client_and_model, generate_report
from theme_builder.writeback import write_communities, write_communities_incremental

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


async def _corpus_cursor(driver: AsyncDriver, group_id: str) -> str | None:
    """The 'corpus as-of' watermark stamped on every report: the latest write time
    across Episodic, Entity, AND RELATES_TO. It must cover all three because
    incremental refresh's dirty-detection (theme_builder.incremental.touched_entities)
    tests Entity/RELATES_TO `created_at` against this stored cursor — and graphiti
    creates entities/facts slightly AFTER their episode, so an Episodic-only max
    would leave those newer nodes forever 'after the cursor' and mark every
    community dirty on an unchanged graph."""
    async with driver.session() as s:
        r = await s.run(
            "CALL () { MATCH (e:Episodic {group_id:$g}) RETURN e.created_at AS t "
            "UNION ALL MATCH (n:Entity {group_id:$g}) RETURN n.created_at AS t "
            "UNION ALL MATCH ()-[f:RELATES_TO {group_id:$g}]->() RETURN f.created_at AS t "
            "UNION ALL MATCH ()-[f:RELATES_TO {group_id:$g}]->() "
            "WHERE f.expired_by_sweep = true RETURN f.invalid_at AS t } "
            "RETURN toString(max(t)) AS c",
            g=group_id)
        rec = await r.single()
        return rec["c"] if rec else None


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
                rep = await generate_report(client, model, ctx, settings.report_max_tokens)
            except Exception:
                logger.exception("theme-build: community %s errored; skipping", c.community_id)
                rep = None
            if rep is None:
                logger.info("theme-build: community %s produced no report; skipping", c.community_id)
                skipped += 1
                continue
            reports[c.community_id] = rep
        corpus_cursor = await _corpus_cursor(driver, settings.group_id)
        res = await write_communities(driver, embedder, settings.group_id,
                                      communities, reports, corpus_cursor=corpus_cursor)
        res["communities_detected"] = len(communities)
        res["reports_skipped"] = skipped
        return res
    finally:
        # close both owned clients (report LLM + embedder) so a repeated caller
        # doesn't leak httpx pools; the injected driver is the caller's to close.
        await client.close()
        await embedder.client.close()


async def _run_theme_build_incremental(settings: ExtractSettings, *, driver: AsyncDriver) -> dict:
    communities = await detect_communities(
        driver, settings.group_id,
        min_community_size=settings.leiden_min_community_size,
        max_levels=settings.leiden_max_levels)
    persisted = await load_persisted(driver, settings.group_id)
    prev_cursor = await prev_corpus_cursor(driver, settings.group_id)
    touched = await touched_entities(driver, settings.group_id, prev_cursor)
    # Snapshot the new watermark HERE (start of run, alongside `touched`), not
    # after report generation — otherwise a fact ingested during the run would be
    # stamped "already covered" (created_at <= a late cursor) yet was never in this
    # run's touched set, so its community would never regenerate. A start snapshot
    # leaves such writes strictly after the stored cursor -> caught next run.
    new_cursor = await _corpus_cursor(driver, settings.group_id)
    matches = match_communities(communities, persisted,
                                tau=settings.theme_refresh_jaccard_tau)
    dirty, clean = classify(communities, matches, touched)

    # stable id per fresh community + content-hash -> stable id map (to remap parents)
    stable_ids = {}
    for i, c in enumerate(communities):
        m = matches[i]
        stable_ids[i] = m.community_id if m is not None else c.community_id
    id_map = {communities[i].community_id: stable_ids[i] for i in range(len(communities))}

    client, model = _report_client_and_model(settings)
    embedder = build_embedder(settings)
    entries: list[dict] = []
    regenerated = 0
    reused = 0
    skipped = 0
    now = datetime.now(timezone.utc)
    try:
        for i, c in enumerate(communities):
            base = {"community_id": stable_ids[i], "level": c.level,
                    "member_uuids": c.member_uuids,
                    "parent_id": id_map.get(c.parent_id) if c.parent_id else None}
            if i in clean:
                p = matches[i]
                assert p is not None
                entries.append({**base, "title": p.title, "summary": p.summary,
                                "full_report": p.full_report, "rating": p.rating,
                                "rating_explanation": p.rating_explanation, "tags": p.tags,
                                "cited_fact_uuids": p.cited_fact_uuids,
                                "embedding": p.embedding, "generated_at": p.generated_at})
                reused += 1
                continue
            try:
                members = await _fetch_members(driver, settings.group_id, c.member_uuids)
                facts = await _fetch_facts(driver, settings.group_id, c.member_uuids)
                ctx = assemble_context(members, facts, top_entities=settings.report_top_entities,
                                       token_budget=settings.report_token_budget)
                rep = await generate_report(client, model, ctx, settings.report_max_tokens)
            except Exception:
                logger.exception("theme-build: community %s errored; skipping", stable_ids[i])
                rep = None
            if rep is None:
                skipped += 1
                continue
            emb = (await embedder.create_batch([f"{rep.title}\n{rep.summary}"]))[0]
            entries.append({**base, "title": rep.title, "summary": rep.summary,
                            "full_report": rep.full_report, "rating": rep.rating,
                            "rating_explanation": rep.rating_explanation, "tags": rep.tags,
                            "cited_fact_uuids": rep.cited_fact_uuids,
                            "embedding": emb, "generated_at": now})
            regenerated += 1
        res = await write_communities_incremental(driver, settings.group_id, entries,
                                                  corpus_cursor=new_cursor)
        matched_persisted = sum(1 for i in matches if matches[i] is not None)
        res.update({"communities_detected": len(communities),
                    "reports_regenerated": regenerated, "reports_reused": reused,
                    "reports_skipped": skipped,
                    "communities_dissolved": len(persisted) - matched_persisted})
        return res
    finally:
        await client.close()
        await embedder.client.close()


@app.command("theme-build")
def theme_build(full: bool = typer.Option(
        False, "--full", help="Full rebuild (regenerate every report) instead of incremental.")) -> None:
    """Refresh the community-report layer (incremental by default; --full rebuilds all)."""
    async def _main() -> None:
        settings = get_extract_settings()
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        try:
            runner = _run_theme_build if full else _run_theme_build_incremental
            res = await runner(settings, driver=driver)
            typer.echo(json.dumps(res, indent=2, default=str))
        finally:
            await driver.close()

    asyncio.run(_main())


if __name__ == "__main__":
    app()
