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
    PersistedCommunity, classify, load_persisted, match_communities, prev_corpus_cursor,
    touched_entities)
from theme_builder.report import (ReportStats, _regenerate_summary,
                                  _report_client_and_model, _verify_client_and_model,
                                  generate_report, verify_report)
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
    vclient, vmodel = _verify_client_and_model(settings)

    async def _verifier(findings, summary, fact_texts):
        return await verify_report(vclient, vmodel, findings, summary, fact_texts)

    embedder = build_embedder(settings)
    reports: dict = {}
    pending: dict = {}
    skipped = 0
    findings_dropped = reverified = unverified = 0
    try:
        for c in communities:
            try:
                members = await _fetch_members(driver, settings.group_id, c.member_uuids)
                facts = await _fetch_facts(driver, settings.group_id, c.member_uuids)
                ctx = assemble_context(members, facts,
                                       top_entities=settings.report_top_entities,
                                       token_budget=settings.report_token_budget)
                st = ReportStats()
                rep = await generate_report(client, model, ctx,
                                            settings.report_max_tokens,
                                            verifier=_verifier, stats=st)
                findings_dropped += st.findings_dropped
                reverified += 1 if st.reverified else 0
                unverified += 1 if st.unverified else 0
                if st.staged_report is not None:
                    pending[c.community_id] = st.staged_report
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
                                      communities, reports, corpus_cursor=corpus_cursor,
                                      pending=pending)
        res["communities_detected"] = len(communities)
        res["reports_skipped"] = skipped
        res["findings_dropped"] = findings_dropped
        res["reports_reverified"] = reverified
        res["reports_unverified"] = unverified
        res["reports_staged"] = len(pending)
        return res
    finally:
        # close both owned clients (report LLM + embedder) so a repeated caller
        # doesn't leak httpx pools; the injected driver is the caller's to close.
        await client.close()
        await vclient.close()
        await embedder.client.close()


def _is_staged(p: PersistedCommunity) -> bool:
    """A persisted community whose report was never verified. Its text lives in
    pending_*; `summary`/`full_report` read back empty, so it must never be
    carried over in the normal shape."""
    return not p.verified or not p.embedding


def _carry_over(base: dict, p: PersistedCommunity, *, stale: bool = False) -> dict:
    """Build an entry that hands a persisted community back to the writer
    unchanged, in whichever shape it already has.

    `stale` marks a report kept because a NEW attempt failed rather than because
    the community was clean. It keeps its embedding and stays retrievable -- it is
    a real verified report, and dropping it from retrieval is the harm being
    avoided -- but it describes the member set it was written against, and this
    run stamps a fresh corpus_cursor, so `classify` needs the flag to know to
    retry it. A staged community is already always-dirty, so the flag is not
    applied to it."""
    if _is_staged(p):
        return {**base, "title": p.title,
                "pending_summary": p.pending_summary,
                "pending_full_report": p.pending_full_report,
                "rating": p.rating, "rating_explanation": p.rating_explanation,
                "tags": p.tags, "cited_fact_uuids": p.cited_fact_uuids,
                "generated_at": p.generated_at, "staged": True}
    return {**base, "title": p.title, "summary": p.summary,
            "full_report": p.full_report, "rating": p.rating,
            "rating_explanation": p.rating_explanation, "tags": p.tags,
            "cited_fact_uuids": p.cited_fact_uuids,
            "embedding": p.embedding, "generated_at": p.generated_at,
            "stale": stale}


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
    vclient, vmodel = _verify_client_and_model(settings)

    async def _verifier(findings, summary, fact_texts):
        return await verify_report(vclient, vmodel, findings, summary, fact_texts)

    embedder = build_embedder(settings)
    entries: list[dict] = []
    regenerated = 0
    reused = 0
    skipped = 0
    staged = 0
    preserved = 0
    lost_by_level: dict[int, int] = {}
    findings_dropped = reverified = unverified = 0
    now = datetime.now(timezone.utc)
    try:
        for i, c in enumerate(communities):
            base = {"community_id": stable_ids[i], "level": c.level,
                    "member_uuids": c.member_uuids,
                    "parent_id": id_map.get(c.parent_id) if c.parent_id else None}
            if i in clean:
                p = matches[i]
                assert p is not None
                entries.append(_carry_over(base, p))
                reused += 1
                continue
            try:
                members = await _fetch_members(driver, settings.group_id, c.member_uuids)
                facts = await _fetch_facts(driver, settings.group_id, c.member_uuids)
                ctx = assemble_context(members, facts, top_entities=settings.report_top_entities,
                                       token_budget=settings.report_token_budget)
                st = ReportStats()
                rep = await generate_report(client, model, ctx,
                                            settings.report_max_tokens,
                                            verifier=_verifier, stats=st)
                findings_dropped += st.findings_dropped
                reverified += 1 if st.reverified else 0
                unverified += 1 if st.unverified else 0
                staged_report = st.staged_report
            except Exception:
                logger.exception("theme-build: community %s errored; skipping", stable_ids[i])
                rep = None
                staged_report = None
            if rep is None:
                if staged_report is not None:
                    # Verification could not COMPLETE. The report is generated and
                    # paid for; dropping this community from `entries` would let
                    # write_communities_incremental's DETACH DELETE take the
                    # community's PREVIOUSLY verified report with it. Stage it
                    # instead -- no embedding, so it stays unreachable from every
                    # answering path -- and recover it with --verify-pending.
                    entries.append({**base, "title": staged_report.title,
                                    "pending_summary": staged_report.summary,
                                    "pending_full_report": staged_report.full_report,
                                    "rating": staged_report.rating,
                                    "rating_explanation": staged_report.rating_explanation,
                                    "tags": staged_report.tags,
                                    "cited_fact_uuids": staged_report.cited_fact_uuids,
                                    "generated_at": now, "staged": True})
                    staged += 1
                    continue
                # Nothing was staged: generation itself failed (the `except` above,
                # or `_generate_once` returning None -- measured 3/29 empty-choices
                # replies on a real run). If this community already HAS a persisted
                # report, a failed NEW attempt must not take it: carry it over
                # rather than let the DETACH DELETE rebuild drop it. Only a
                # community with nothing persisted is a genuine loss.
                p = matches[i]
                if p is not None:
                    entries.append(_carry_over(base, p, stale=True))
                    logger.warning(
                        "theme-build: community %s produced no report; carrying over "
                        "its persisted %s report, flagged stale for the next run",
                        stable_ids[i], "staged" if _is_staged(p) else "verified")
                    if _is_staged(p):
                        staged += 1
                    else:
                        preserved += 1
                    continue
                logger.warning(
                    "theme-build: community %s (level %s) produced no report and has "
                    "nothing persisted; LOST", stable_ids[i], c.level)
                skipped += 1
                lost_by_level[c.level] = lost_by_level.get(c.level, 0) + 1
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
                    "reports_skipped": skipped, "reports_staged": staged,
                    "reports_preserved": preserved, "lost_by_level": lost_by_level,
                    "communities_dissolved": len(persisted) - matched_persisted,
                    "findings_dropped": findings_dropped,
                    "reports_reverified": reverified,
                    "reports_unverified": unverified})
        return res
    finally:
        await client.close()
        await vclient.close()
        await embedder.client.close()


async def _run_verify_pending(settings: ExtractSettings, *, driver: AsyncDriver) -> dict:
    """Re-verify only the reports staged by an earlier run and promote the ones
    that pass.

    Recovering from a verifier outage costs one verify call per staged community
    rather than regenerating the corpus (spec 4.7).

    A staged report is lost ONLY when no finding survives verification. The summary
    verdict is not a rejection reason (spec 4.4 / 4.5.5); the promoted summary is
    regenerated from the findings that survived, so it inherits their support.
    """
    vclient, vmodel = _verify_client_and_model(settings)
    # The promoted summary is REGENERATED from the findings that survived, so this
    # needs the report tier too -- promoting the staged summary would publish (and
    # embed) a paragraph written against findings that have just been dropped.
    rclient, rmodel = _report_client_and_model(settings)
    embedder = build_embedder(settings)
    promoted = rejected = still_pending = findings_dropped = 0
    try:
        async with driver.session() as s:
            r = await s.run(
                "MATCH (c:Community {group_id:$g}) WHERE c.pending_full_report IS NOT NULL "
                "RETURN c.community_id AS cid, c.title AS title, "
                "c.pending_summary AS summary, c.pending_full_report AS full_report, "
                "coalesce(c.cited_fact_uuids,[]) AS cited", g=settings.group_id)
            staged = [dict(x) async for x in r]
        for row in staged:
            findings = json.loads(row["full_report"] or "[]")
            async with driver.session() as s:
                fr = await s.run(
                    "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $u "
                    "RETURN f.uuid AS uuid, f.fact AS fact",
                    g=settings.group_id, u=list(row["cited"]))
                fact_texts = {x["uuid"]: x["fact"] async for x in fr}
            result = await verify_report(vclient, vmodel, findings,
                                         row["summary"] or "", fact_texts)
            if result is None:
                still_pending += 1
                continue
            kept = [f for i, f in enumerate(findings, 1) if i not in result.unsupported]
            findings_dropped += len(findings) - len(kept)
            # The SUMMARY verdict is deliberately NOT a rejection reason: judging an
            # inherently synthetic paragraph by "states nothing the facts do not
            # state" destroyed 15 of 41 communities. Only "no finding survived" --
            # a genuine content failure -- loses the staged report.
            if not kept:
                rejected += 1
                async with driver.session() as s:
                    await s.run(
                        "MATCH (c:Community {community_id:$cid, group_id:$g}) "
                        "REMOVE c.pending_summary, c.pending_full_report",
                        cid=row["cid"], g=settings.group_id)
                continue
            cited: list[str] = []
            for f in kept:
                for fid in f.get("fact_ids", []):
                    if fid not in cited:
                        cited.append(fid)
            summary = await _regenerate_summary(rclient, rmodel, kept, row["title"] or "")
            if not summary:
                # A summariser hiccup must not cost a verified report; keep the
                # staged summary, but say so -- the promoted paragraph may then
                # still reference a dropped finding.
                logger.warning(
                    "verify-pending: summary regeneration returned nothing usable for "
                    "community %s; keeping the staged summary", row["cid"])
                summary = row["summary"] or ""
            emb = (await embedder.create_batch([f"{row['title']}\n{summary}"]))[0]
            async with driver.session() as s:
                await s.run(
                    "MATCH (c:Community {community_id:$cid, group_id:$g}) "
                    "SET c.summary=$summary, c.full_report=$full_report, "
                    "c.cited_fact_uuids=$cited, c.embedding=$emb, c.verified=true "
                    "REMOVE c.pending_summary, c.pending_full_report",
                    cid=row["cid"], g=settings.group_id, summary=summary,
                    full_report=json.dumps(kept), cited=cited, emb=emb)
            promoted += 1
    finally:
        await vclient.close()
        await rclient.close()
        await embedder.client.close()
    return {"reports_promoted": promoted, "reports_rejected": rejected,
            "reports_still_pending": still_pending,
            "findings_dropped": findings_dropped}


@app.command("theme-build")
def theme_build(
        full: bool = typer.Option(
            False, "--full", help="Full rebuild (regenerate every report) instead of incremental."),
        verify_pending: bool = typer.Option(
            False, "--verify-pending",
            help="Re-verify only reports staged by an earlier run and promote the ones "
                 "that pass. Recovers from a transient verifier outage without "
                 "regenerating anything. Wins over --full if both are given.")) -> None:
    """Refresh the community-report layer (incremental by default; --full rebuilds all)."""
    async def _main() -> None:
        settings = get_extract_settings()
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
        try:
            if verify_pending:
                res = await _run_verify_pending(settings, driver=driver)
            else:
                runner = _run_theme_build if full else _run_theme_build_incremental
                res = await runner(settings, driver=driver)
            typer.echo(json.dumps(res, indent=2, default=str))
        finally:
            await driver.close()

    asyncio.run(_main())


if __name__ == "__main__":
    app()
