"""Deterministic post-ingest cleanup: prune noise entities from the graph.

Loads every :Entity in a group, flags names via noise_filter.is_noise (the
single source of truth for noise), and DETACH DELETEs the flagged nodes —
removing their RELATES_TO fact edges and MENTIONS edges with them. Returns
the pruned names as an audit trail so a human can inspect exactly what was
deleted before trusting it.
"""

from __future__ import annotations

from neo4j import AsyncDriver

from graph_extract.noise_filter import is_noise


async def prune_noise_entities(driver: AsyncDriver, group_id: str) -> dict:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) "
            "RETURN e.name AS name, [l IN labels(e) WHERE l<>'Entity'][0] AS type",
            g=group_id,
        )
        rows = [dict(rec) async for rec in r]
        noise = [row["name"] for row in rows if is_noise(row["name"], row.get("type"))]
        if noise:
            await s.run(
                "MATCH (e:Entity {group_id:$g}) WHERE e.name IN $names DETACH DELETE e",
                g=group_id,
                names=noise,
            )
    return {"scanned": len(rows), "pruned": len(noise), "pruned_names": noise}
