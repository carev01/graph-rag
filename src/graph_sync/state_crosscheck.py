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
* no `:Article` with content (`content_hash` set, not removed) may have neither a
  `HAS_EPISODE` edge nor ANY `semantic_jobs` row: such an article was never
  extracted and nothing will ever queue it (BACKLOG 50 -- a state reset, a lost
  row). Re-running `bootstrap --source-id` for its source re-queues it. Content-
  less TOC placeholder Articles are excluded; they are not ingestable yet.

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
_UNEXTRACTED = """
MATCH (a:Article)
WHERE a.content_hash IS NOT NULL AND coalesce(a.removed, false) = false
  AND NOT EXISTS { (a)-[:HAS_EPISODE]->() }
RETURN a.id AS id, a.source_id AS source_id
"""
_NO_JOB_ROW = (
    "SELECT x AS article_id FROM unnest($1::text[]) AS x "
    "WHERE NOT EXISTS (SELECT 1 FROM semantic_jobs j WHERE j.article_id = x)"
)
_CHUNK = 5000


@dataclass
class DoneJob:
    article_id: str
    exists: bool
    title: str | None
    episodes: int


def evaluate(shard_counts: dict[str, int], done: list[DoneJob], *,
             stranded: dict[str, list[str]] | None = None) -> tuple[list[str], bool]:
    """Pure verdict over what was read; kept separate so it is unit-testable.

    `stranded` maps source id -> article ids with content, no episodes and no job
    row of any status; `None` means the check was not run."""
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
    if stranded is not None:
        n = sum(len(v) for v in stranded.values())
        ok &= n == 0
        lines.append(f"{'OK  ' if n == 0 else 'FAIL'}  {n} stranded :Article (content, "
                     f"no episodes, no semantic_jobs row) across {len(stranded)} source(s)")
        for sid, ids in sorted(stranded.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"      {sid}: {len(ids)} (e.g. {', '.join(ids[:3])})")
        if n:
            lines.append("      repair: re-run `python -m graph_sync.cli bootstrap "
                         "--source-id <id>` for each source above (re-queues them)")
    return lines, ok


async def _read(sample: int) -> tuple[dict[str, int], list[DoneJob], dict[str, list[str]]]:
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
            unextracted = {r["id"]: r["source_id"]
                           async for r in await session.run(_UNEXTRACTED)}
    finally:
        await driver.close()

    stranded: dict[str, list[str]] = {}
    if unextracted:
        conn = await asyncpg.connect(s.postgres_dsn, timeout=15)
        try:
            async with conn.transaction(readonly=True):
                all_ids = sorted(unextracted)
                for i in range(0, len(all_ids), _CHUNK):
                    for r in await conn.fetch(_NO_JOB_ROW, all_ids[i:i + _CHUNK]):
                        aid = r["article_id"]
                        stranded.setdefault(unextracted[aid] or "<no source_id>", []).append(aid)
        finally:
            await conn.close()
    return counts, done, stranded


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only cross-check of migrated sync state.")
    ap.add_argument("--sample", type=int, default=50)
    args = ap.parse_args()
    counts, done, stranded = asyncio.run(_read(args.sample))
    lines, ok = evaluate(counts, done, stranded=stranded)
    print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
