"""D6 A/B: today's chunking vs packing to 1,200 tokens. PAID -- authorised 2026-09-23,
cap $5. Each arm writes into its OWN throwaway Neo4j testcontainer; the .env graph is
never touched. Articles run sequentially through IngestDriver.ingest_article (this
bypasses ingest_source's warm-up barrier and sibling spreading, which do not interact
with the chunking variable under test).

    uv run python scripts/chunk_ab.py --sample SAMPLE.json --out DIR [--articles 30] [--smoke-only]

SAMPLE.json is scripts/spikes/pre_bootstrap/d6_sample.py's output.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from testcontainers.neo4j import Neo4jContainer

from graph_extract.article_router import is_dense_matrix
from graph_extract.chonkie_client import Chunk
from graph_extract.cli import _build_ingest_driver
from graph_extract.config import ExtractSettings, get_extract_settings
from graph_extract.episode_builder import build_episodes
from graph_extract.usage import get_tally

PRICES = {"cheap": (0.09, 0.36), "strong": (0.25, 2.00)}  # $/1M in, out
SPEND_CAP_USD = 5.0
SMOKE_N = 2
ARMS: dict[str, dict[str, Any]] = {
    "today": {},
    "pack1200": {"pack_target_tokens": 1200, "cheap_max_chunk_tokens": 1200},
}
NEO4J_IMAGE = "neo4j:2026.07.1-community"


def today_episode_count(article: dict, s: ExtractSettings) -> int:
    chunks = [Chunk(text=c["t"], start_index=0, end_index=0, token_count=c["tok"])
              for c in article["chunks"]]
    md = "\n".join(c["t"] for c in article["chunks"])
    dense = is_dense_matrix(md, ratio_threshold=s.dense_table_line_ratio,
                            pipe_threshold=s.dense_pipe_count)
    cap = s.max_chunk_tokens if dense else s.cheap_max_chunk_tokens
    return len(build_episodes(article_id=article["id"], title=article["title"],
                              chapter_path="", content_hash="x" * 16, chunks=chunks,
                              max_chunk_tokens=cap, min_chunk_tokens=s.min_chunk_tokens))


def select_articles(sample: list[dict], n: int, rng: random.Random,
                    s: ExtractSettings, min_episodes: int = 3) -> list[dict]:
    """Articles with >= min_episodes episodes today, round-robin across vendors."""
    by_vendor: dict[str, list[dict]] = {}
    for a in sample:
        if today_episode_count(a, s) >= min_episodes:
            by_vendor.setdefault(a["vendor"] or "?", []).append(a)
    for group in by_vendor.values():
        rng.shuffle(group)
    out: list[dict] = []
    while len(out) < n and any(by_vendor.values()):
        for v in sorted(by_vendor):
            if by_vendor[v] and len(out) < n:
                out.append(by_vendor[v].pop())
    return out


def article_cost(prompt: int, completion: int, tier: str) -> float:
    pin, pout = PRICES[tier]
    return (prompt * pin + completion * pout) / 1_000_000


def summarise(rows: list[dict]) -> dict:
    keys = ("episodes", "facts", "entities", "seconds", "cost")
    return {k: sum(r[k] for r in rows) for k in keys} | {"articles": len(rows)}


async def _seed(driver, articles: list[dict]) -> None:
    await driver.execute_query(
        "UNWIND $rows AS r MERGE (a:Article {id: r.id}) "
        "SET a.source_url = 'https://example.invalid/' + r.id, a.title = r.title "
        "MERGE (c:Chapter {id: 'ch-' + r.id}) SET c.title = r.title "
        "MERGE (a)-[:IN_CHAPTER]->(c)",
        rows=[{"id": a["id"], "title": a["title"]} for a in articles])


async def _article_graph(driver, article_id: str) -> dict:
    r = await driver.execute_query(
        "MATCH (:Article {id: $a})-[:HAS_EPISODE]->(e:Episodic) "
        "OPTIONAL MATCH (e)-[:MENTIONS]->(n:Entity) "
        "WITH collect(DISTINCT e.uuid) AS eps, collect(DISTINCT n.name) AS ents "
        "OPTIONAL MATCH ()-[f:RELATES_TO]->() WHERE any(u IN f.episodes WHERE u IN eps) "
        "RETURN size(ents) AS entities, count(DISTINCT f) AS facts, "
        "collect(DISTINCT f.fact) AS fact_texts, ents",
        a=article_id)
    rec = r.records[0]
    return {"entities": rec["entities"], "facts": rec["facts"],
            "fact_texts": sorted(rec["fact_texts"]), "entity_names": sorted(rec["ents"])}


async def run_arm(name: str, articles: list[dict], spent_before: float) -> list[dict]:
    """Ingest `articles` in a fresh container under arm `name`'s settings."""
    rows: list[dict] = []
    with Neo4jContainer(NEO4J_IMAGE) as neo:
        s = get_extract_settings().model_copy(update={
            "neo4j_uri": neo.get_connection_url(), "neo4j_user": "neo4j",
            "neo4j_password": neo.password, "group_id": f"ab-{name}", **ARMS[name]})
        ingest, graphiti, docext, driver = await _build_ingest_driver(s)
        try:
            await _seed(driver, articles)
            for a in articles:
                t = get_tally()
                p0, c0 = t.prompt_tokens, t.completion_tokens
                start = time.perf_counter()
                res = await ingest.ingest_article(a["id"])
                seconds = time.perf_counter() - start
                t = get_tally()
                dp, dc = t.prompt_tokens - p0, t.completion_tokens - c0
                g = await _article_graph(driver, a["id"])
                rows.append({"arm": name, "article_id": a["id"], "vendor": a["vendor"],
                             "tier": res.tier, "episodes": res.episodes_added,
                             "facts": g["facts"], "entities": g["entities"],
                             "seconds": seconds, "prompt_tokens": dp,
                             "completion_tokens": dc, "cost": article_cost(dp, dc, res.tier),
                             "dedup": asdict(res.dedup), "fact_texts": g["fact_texts"],
                             "entity_names": g["entity_names"]})
                spent = spent_before + sum(r["cost"] for r in rows)
                print(f"[{name}] {len(rows)}/{len(articles)} {a['id']} episodes={res.episodes_added} "
                      f"facts={g['facts']} ${spent:.3f}", flush=True)
                if spent > SPEND_CAP_USD:
                    raise SystemExit(f"spend cap ${SPEND_CAP_USD} exceeded (${spent:.2f}); stopping")
        finally:
            await driver.close()
            await docext.aclose()
            await graphiti.close()
    return rows


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--articles", type=int, default=30)
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = get_extract_settings()
    articles = select_articles(json.loads(Path(args.sample).read_text()), args.articles,
                               random.Random(7), s)
    print(f"{len(articles)} articles selected", flush=True)

    # Smoke: prove the variable is engaged before the full spend (CLAUDE.md).
    smoke = articles[:SMOKE_N]
    tally_before = get_tally().prompt_tokens
    a_rows = await run_arm("today", smoke, 0.0)
    b_rows = await run_arm("pack1200", smoke, sum(r["cost"] for r in a_rows))
    a_eps, b_eps = sum(r["episodes"] for r in a_rows), sum(r["episodes"] for r in b_rows)
    spent = sum(r["cost"] for r in a_rows + b_rows)
    print(f"smoke: today episodes={a_eps} pack1200 episodes={b_eps} spent=${spent:.3f}", flush=True)
    if get_tally().prompt_tokens == tally_before:
        raise SystemExit("usage tally did not move: cost is not being measured -- refusing")
    if not b_eps < a_eps:
        raise SystemExit("packing did not reduce episodes on the smoke articles -- the "
                         "variable is not engaged; refusing to spend on the full run")
    (out / "smoke.json").write_text(json.dumps(a_rows + b_rows, indent=1))
    if args.smoke_only:
        return

    results: dict[str, Any] = {}
    for arm in ARMS:
        rows = await run_arm(arm, articles, spent)
        spent += sum(r["cost"] for r in rows)
        results[arm] = {"summary": summarise(rows), "rows": rows}
    results["spent_usd_including_smoke"] = spent
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    for arm in ARMS:
        print(arm, results[arm]["summary"], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
