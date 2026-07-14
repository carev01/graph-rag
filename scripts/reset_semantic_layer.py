"""Reset the Graphiti SEMANTIC layer, keeping the slice-1 STRUCTURAL layer.

Use this before re-running the pilot with a DIFFERENT extraction model, so the
graph doesn't mix two models' extractions (which would confound the dedup /
quality eval).

Deletes: `:Episodic`, `:Entity`, `:Community` nodes and all their relationships
(`RELATES_TO`, `MENTIONS`, `HAS_EPISODE`, `IN_COMMUNITY`, ...).
Keeps: `:Vendor` `:Product` `:Source` `:Chapter` `:Article` (structural,
model-independent, from `graph_sync` bootstrap).

    uv run --extra dev python scripts/reset_semantic_layer.py
"""
from __future__ import annotations
import asyncio

from neo4j import AsyncGraphDatabase

from graph_extract.config import get_extract_settings


async def main() -> None:
    s = get_extract_settings()
    driver = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    try:
        async with driver.session() as sess:
            before = await (await sess.run(
                "MATCH (n) WHERE n:Episodic OR n:Entity OR n:Community "
                "RETURN count(n) AS c")).single()
            print(f"[reset] semantic nodes to delete: {before['c']}")
            # Batched detach-delete to avoid a huge single transaction.
            while True:
                r = await (await sess.run(
                    "MATCH (n) WHERE n:Episodic OR n:Entity OR n:Community "
                    "WITH n LIMIT 5000 DETACH DELETE n RETURN count(n) AS c")).single()
                if not r["c"]:
                    break
                print(f"[reset] deleted {r['c']}")
            struct = await (await sess.run(
                "MATCH (a:Article) RETURN count(a) AS c")).single()
            print(f"[reset] done. Structural :Article nodes kept: {struct['c']}")
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
