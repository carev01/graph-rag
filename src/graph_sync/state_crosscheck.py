"""Read-only cross-check of migrated sync state against Neo4j (docs/deploy/k3s.md).

    python -m graph_sync.state_crosscheck [--sample 50]

Free (no LLM tokens) and read-only on both stores: Postgres is read inside a
`READ ONLY` transaction, Neo4j through a READ-access session. It answers "does the
Postgres state that was just restored describe the graph this cluster points at?"
before the poller is allowed to act on it:

* every `bootstrap_progress` row with status `complete` must have at least one
  `:Article` in Neo4j for its shard. A shard is what `SyncCore.bootstrap` keyed it
  by: `source_id or vendor_id or "global"` -- so it is matched as a Source id
  (`Article.source_id`), else as a Vendor id (Vendor->Product->Source->Article),
  else `global` (every Article).
* a random sample of `semantic_jobs` rows with status `done` and op `upsert` must
  each have an `:Article` with at least one `HAS_EPISODE` edge -- except
  navigation pages, which the ingest driver completes without extracting
  (`graph_extract.article_filter.is_navigation_article`).

Exit status is non-zero on any mismatch. Output holds only shard/article ids and
counts -- no credential.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass

import asyncpg
from neo4j import READ_ACCESS, AsyncGraphDatabase

from graph_extract.article_filter import is_navigation_article
from graph_sync.config import get_settings

_SHARD_COUNT = """
OPTIONAL MATCH (a:Article {source_id: $shard})
WITH count(a) AS by_source
OPTIONAL MATCH (:Vendor {id: $shard})-[:HAS_PRODUCT]->(:Product)
               -[:HAS_SOURCE]->(:Source)-[:HAS_ARTICLE]->(b:Article)
RETURN by_source, count(DISTINCT b) AS by_vendor
"""
_ALL_ARTICLES = "MATCH (a:Article) RETURN count(a) AS n"
_EPISODES = """
UNWIND $ids AS id
OPTIONAL MATCH (a:Article {id: id})
OPTIONAL MATCH (a)-[r:HAS_EPISODE]->()
RETURN id, a IS NOT NULL AS found, a.title AS title, count(r) AS episodes
"""


@dataclass
class DoneJob:
    article_id: str
    exists: bool
    title: str | None
    episodes: int


def evaluate(shard_counts: dict[str, int], done: list[DoneJob]) -> tuple[list[str], bool]:
    """Pure verdict over what was read; kept separate so it is unit-testable."""
    lines: list[str] = []
    ok = True
    for shard, n in sorted(shard_counts.items()):
        good = n > 0
        ok &= good
        lines.append(f"{'OK  ' if good else 'FAIL'}  bootstrap shard {shard}: {n} :Article")
    if not shard_counts:
        lines.append("OK    no complete bootstrap_progress rows to check")
    missing = [d for d in done if not d.exists]
    no_ep = [d for d in done if d.exists and d.episodes == 0
             and not is_navigation_article(d.title or "")]
    nav = [d for d in done if d.exists and d.episodes == 0
           and is_navigation_article(d.title or "")]
    ok &= not missing and not no_ep
    lines.append(f"{'OK  ' if not missing and not no_ep else 'FAIL'}  sampled done jobs: "
                 f"{len(done)} checked, {len(missing)} article missing, "
                 f"{len(no_ep)} without HAS_EPISODE, {len(nav)} navigation (expected 0 episodes)")
    lines += [f"      missing :Article {d.article_id}" for d in missing]
    lines += [f"      no HAS_EPISODE   {d.article_id}" for d in no_ep]
    return lines, ok


async def _read(sample: int) -> tuple[dict[str, int], list[DoneJob]]:
    s = get_settings()
    conn = await asyncpg.connect(s.postgres_dsn, timeout=15)
    try:
        async with conn.transaction(readonly=True):
            shards = [r["shard"] for r in await conn.fetch(
                "SELECT shard FROM bootstrap_progress WHERE status='complete' ORDER BY shard")]
            ids = [r["article_id"] for r in await conn.fetch(
                "SELECT article_id FROM (SELECT DISTINCT article_id FROM semantic_jobs "
                "WHERE status='done' AND op='upsert') d ORDER BY random() LIMIT $1", sample)]
    finally:
        await conn.close()

    driver = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    try:
        async with driver.session(default_access_mode=READ_ACCESS) as session:
            counts: dict[str, int] = {}
            for shard in shards:
                if shard == "global":
                    rec = await (await session.run(_ALL_ARTICLES)).single()
                    counts[shard] = rec["n"] if rec else 0
                    continue
                rec = await (await session.run(_SHARD_COUNT, shard=shard)).single()
                counts[shard] = max(rec["by_source"], rec["by_vendor"]) if rec else 0
            done = [DoneJob(r["id"], r["found"], r["title"], r["episodes"])
                    async for r in await session.run(_EPISODES, ids=ids)]
    finally:
        await driver.close()
    return counts, done


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only cross-check of migrated sync state.")
    ap.add_argument("--sample", type=int, default=50)
    args = ap.parse_args()
    lines, ok = evaluate(*asyncio.run(_read(args.sample)))
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
