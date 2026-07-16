"""Deterministic post-ingest cleanup: prune noise entities from the graph.

Loads every :Entity in a group, flags names via noise_filter.is_noise (the
single source of truth for noise), and DETACH DELETEs the flagged nodes —
removing their RELATES_TO fact edges and MENTIONS edges with them. Returns
the pruned names as an audit trail so a human can inspect exactly what was
deleted before trusting it.

`retype_region_entities` is a second, complementary corrector: the cheap
extraction model sometimes mistypes a region (e.g. "Germany West Central")
as :Platform or another custom type. region_names.is_region is the single
source of truth for "what is a region" -- also used by the ontology
examples -- so relabeling here stays in lockstep with extraction guidance.
This is a label change, not a delete: :Entity and every edge are preserved.
"""

from __future__ import annotations

from neo4j import AsyncDriver

from graph_extract.article_filter import is_navigation_article
from graph_extract.noise_filter import is_noise
from graph_extract.region_names import is_region


async def prune_noise_entities(driver: AsyncDriver, group_id: str) -> dict:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) "
            "RETURN e.name AS name, [l IN labels(e) WHERE l<>'Entity'][0] AS type",
            g=group_id,
        )
        rows = [dict(rec) async for rec in r]
        noise = [row["name"] for row in rows if is_noise(row["name"] or "", row.get("type"))]
        if noise:
            await s.run(
                "MATCH (e:Entity {group_id:$g}) WHERE e.name IN $names DETACH DELETE e",
                g=group_id,
                names=noise,
            )
    return {"scanned": len(rows), "pruned": len(noise), "pruned_names": noise}


async def retype_region_entities(driver: AsyncDriver, group_id: str) -> dict:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) "
            "RETURN elementId(e) AS id, e.name AS name, "
            "[l IN labels(e) WHERE l<>'Entity' AND l<>'Region'] AS types, "
            "'Region' IN labels(e) AS is_region_lbl",
            g=group_id,
        )
        rows = [dict(rec) async for rec in r]
        targets = [
            row
            for row in rows
            if is_region(row["name"] or "") and not row["is_region_lbl"]
        ]
        for row in targets:
            # remove the wrong custom type labels, add :Region (labels can't
            # be parameterised; these come from labels(e), not user input,
            # so the f-string interpolation is safe -- still backtick-quoted)
            remove = "".join(f" REMOVE e:`{t}`" for t in row["types"])
            await s.run(
                f"MATCH (e:Entity {{group_id:$g}}) WHERE elementId(e)=$id SET e:Region{remove}",
                g=group_id,
                id=row["id"],
            )
    names = sorted(row["name"] for row in targets)
    return {"scanned": len(rows), "retyped": len(targets), "retyped_names": names}


async def tombstone_navigation_articles(driver: AsyncDriver, group_id: str) -> dict:
    """Mark the episodes of already-extracted navigation/index/changelog
    articles (article_filter.is_navigation_article) as removed=true.

    Deterministic cleanup counterpart to Task 1's ingest-time prevention:
    already-ingested junk gets tombstoned here, then the existing
    sweep_stale_facts expires facts whose only supporting episodes are now
    all removed. Marks, never deletes (temporal policy #3) -- the episode
    nodes and their HAS_EPISODE edges stay in place for provenance/history.
    """
    async with driver.session() as s:
        r = await s.run(
            "MATCH (a:Article)-[:HAS_EPISODE]->(:Episodic {group_id:$g}) "
            "RETURN DISTINCT a.id AS id, a.title AS title", g=group_id)
        rows = [dict(rec) async for rec in r]
        nav = [row for row in rows if is_navigation_article(row["title"] or "")]
        ids = [row["id"] for row in nav]
        episodes = 0
        if ids:
            rr = await s.run(
                "MATCH (a:Article)-[:HAS_EPISODE]->(e:Episodic {group_id:$g}) "
                "WHERE a.id IN $ids SET e.removed=true RETURN count(e) AS c",
                g=group_id, ids=ids)
            rr_record = await rr.single()
            assert rr_record is not None  # count() always returns exactly one row
            episodes = rr_record["c"]
    return {"scanned": len(rows), "tombstoned_articles": len(ids),
            "tombstoned_episodes": episodes, "titles": sorted(row["title"] for row in nav)}
