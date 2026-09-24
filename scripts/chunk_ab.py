"""D6 A/B: today's chunking vs packing to 1,200 tokens. PAID -- authorised 2026-09-23,
cap $5. Each arm writes into its OWN throwaway Neo4j testcontainer; the .env graph is
never touched. Articles run sequentially through IngestDriver.ingest_article (this
bypasses ingest_source's warm-up barrier and sibling spreading, which do not interact
with the chunking variable under test).

    uv run python scripts/chunk_ab.py --sample SAMPLE.json --out DIR [--articles 30] [--smoke-only]
    uv run python scripts/chunk_ab.py --sample SAMPLE.json --out DIR --coverage --reuse PREV.json

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
from graph_extract.ontology import CHEAP_TIER_SALIENCE, EXTRACTION_INSTRUCTIONS
from graph_extract.usage import get_tally

PRICES = {"cheap": (0.09, 0.36), "strong": (0.25, 2.00)}  # $/1M in, out
SPEND_CAP_USD = 5.0
SMOKE_N = 2
ARMS: dict[str, dict[str, Any]] = {
    "today": {},
    "pack1200": {"pack_target_tokens": 1200, "cheap_max_chunk_tokens": 1200},
}
NEO4J_IMAGE = "neo4j:2026.07.1-community"

ARMS["pack1200_coverage"] = dict(ARMS["pack1200"])

_FIXED_COUNT = " A dense table chunk should yield roughly 10-25 facts total, not dozens."
assert _FIXED_COUNT in CHEAP_TIER_SALIENCE, "CHEAP_TIER_SALIENCE changed; update the variant"
CHEAP_TIER_SALIENCE_UNCAPPED = CHEAP_TIER_SALIENCE.replace(_FIXED_COUNT, "")

COVERAGE_MARKER = "COVERAGE (work through the whole message)"
COVERAGE_DIRECTIVE = (
    f"\n\n{COVERAGE_MARKER}: the CURRENT MESSAGE may be long and cover several "
    "sections. Go through it section by section, from the first line to the last, and "
    "extract EVERY product, feature, workload, platform, requirement, limitation and "
    "availability statement each section states -- do not stop after the first topics "
    "and do not condense the message into a few summary items. A longer message "
    "should yield proportionally more entities and facts. Never restate one fact in "
    "different words: each fact must add a claim or a concrete detail (a version, a "
    "limit, a region, a condition) that no other fact already states."
)
ARM_INSTRUCTIONS: dict[str, tuple[str, str]] = {
    "pack1200_coverage": (
        EXTRACTION_INSTRUCTIONS + COVERAGE_DIRECTIVE,
        EXTRACTION_INSTRUCTIONS + CHEAP_TIER_SALIENCE_UNCAPPED + COVERAGE_DIRECTIVE,
    ),
}


def directive_seen(capture_path: Path, marker: str) -> bool:
    """True iff graphiti actually SENT the directive in an extraction prompt --
    proof the override reached the model, not merely that an attribute was set."""
    if not capture_path.exists():
        return False
    for line in capture_path.read_text().splitlines():
        rec = json.loads(line)
        if not str(rec.get("prompt_name", "")).startswith(("extract_nodes.", "extract_edges.")):
            continue
        if any(marker in str(m.get("content", "")) for m in rec.get("messages") or []):
            return True
    return False


def reused_rows(prev: dict, arm: str, article_ids: list[str]) -> list[dict]:
    rows = prev[arm]["rows"]
    if [r["article_id"] for r in rows] != article_ids:
        raise SystemExit(f"reused arm {arm!r} does not match the selected articles "
                         "in order -- refusing to compare different samples")
    return rows


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


def _smoke_refusal(a_rows: list[dict], b_rows: list[dict], tally_moved: bool) -> str | None:
    """Check smoke gates: tally moved globally, and any row with episodes > 0 is metered.

    Returns error message if a gate fails, None if all gates pass.
    """
    if not tally_moved:
        return "usage tally did not move: cost is not being measured -- refusing"

    smoke_rows = a_rows + b_rows
    for row in smoke_rows:
        if row["episodes"] > 0 and row["prompt_tokens"] == 0:
            return ("smoke row has episodes > 0 but prompt_tokens == 0; "
                    "a tier whose usage was not tallied -- refusing")

    a_eps = sum(r["episodes"] for r in a_rows)
    b_eps = sum(r["episodes"] for r in b_rows)
    if not b_eps < a_eps:
        return ("packing did not reduce episodes on the smoke articles -- the "
                "variable is not engaged; refusing to spend on the full run")

    return None


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
        "WITH collect(DISTINCT e.uuid) AS eps, collect(DISTINCT n.uuid) AS entity_uuids, "
        "  collect(DISTINCT n.name) AS ent_names "
        "OPTIONAL MATCH ()-[f:RELATES_TO]->() WHERE any(u IN f.episodes WHERE u IN eps) "
        "RETURN size(entity_uuids) AS entities, count(DISTINCT f) AS facts, "
        "collect(DISTINCT f.fact) AS fact_texts, ent_names",
        a=article_id)
    rec = r.records[0]
    return {"entities": rec["entities"], "facts": rec["facts"],
            "fact_texts": sorted(rec["fact_texts"]), "entity_names": sorted(rec["ent_names"])}


async def run_arm(name: str, articles: list[dict], spent_before: float,
                  out: Path, *, capture_path: Path | None = None) -> list[dict]:
    """Ingest `articles` in a fresh container under arm `name`'s settings.

    Writes each completed row to out/rows.jsonl immediately (to avoid losing results
    on a crash or spend-cap exit).
    """
    rows: list[dict] = []
    rows_file = out / "rows.jsonl"
    with Neo4jContainer(NEO4J_IMAGE) as neo:
        update_dict = {
            "neo4j_uri": neo.get_connection_url(), "neo4j_user": "neo4j",
            "neo4j_password": neo.password, "group_id": f"ab-{name}", **ARMS[name]}
        if capture_path is not None:
            update_dict["llm_capture_path"] = str(capture_path)
        s = get_extract_settings().model_copy(update=update_dict)
        ingest, graphiti, docext, driver = await _build_ingest_driver(s)
        try:
            if name in ARM_INSTRUCTIONS:
                strong_instr, cheap_instr = ARM_INSTRUCTIONS[name]
                if ingest._cheap is None:
                    raise SystemExit("cheap tier not configured; the variant needs both tiers")
                ingest._strong.instructions = strong_instr
                ingest._cheap.instructions = cheap_instr
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
                row = {"arm": name, "article_id": a["id"], "vendor": a["vendor"],
                       "tier": res.tier, "episodes": res.episodes_added,
                       "facts": g["facts"], "entities": g["entities"],
                       "seconds": seconds, "prompt_tokens": dp,
                       "completion_tokens": dc, "cost": article_cost(dp, dc, res.tier),
                       "dedup": asdict(res.dedup), "fact_texts": g["fact_texts"],
                       "entity_names": g["entity_names"]}
                rows.append(row)
                rows_file.write_text(rows_file.read_text() + json.dumps(row) + "\n"
                                     if rows_file.exists() else json.dumps(row) + "\n")
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
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--reuse")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = get_extract_settings()
    articles = select_articles(json.loads(Path(args.sample).read_text()), args.articles,
                               random.Random(7), s)
    print(f"{len(articles)} articles selected", flush=True)

    if args.coverage:
        if not args.reuse:
            raise SystemExit("--coverage needs --reuse PREV_results.json")
        prev = json.loads(Path(args.reuse).read_text())
        ids = [a["id"] for a in articles]
        reused = {arm: reused_rows(prev, arm, ids) for arm in ("today", "pack1200")}
        cap = out / "capture-smoke.jsonl"
        tally_before = get_tally().prompt_tokens
        smoke_rows = await run_arm("pack1200_coverage", articles[:SMOKE_N], 0.0, out,
                                   capture_path=cap)
        (out / "smoke_coverage.json").write_text(json.dumps(smoke_rows, indent=1))
        if get_tally().prompt_tokens == tally_before:
            raise SystemExit("usage tally did not move -- refusing")
        if any(r["episodes"] > 0 and r["prompt_tokens"] == 0 for r in smoke_rows):
            raise SystemExit("an unmetered smoke row -- refusing")
        if not directive_seen(cap, COVERAGE_MARKER):
            raise SystemExit("coverage directive not found in any captured extraction "
                             "prompt -- the variable is not engaged; refusing")
        spent = sum(r["cost"] for r in smoke_rows)
        print(f"coverage smoke passed; spent=${spent:.3f}", flush=True)
        if args.smoke_only:
            return
        rows = await run_arm("pack1200_coverage", articles, spent, out)
        spent += sum(r["cost"] for r in rows)
        results = {arm: {"summary": summarise(r), "rows": r} for arm, r in reused.items()}
        results["pack1200_coverage"] = {"summary": summarise(rows), "rows": rows}
        results["spent_usd_new_arm"] = spent
        results["method_notes"] = prev.get("method_notes", []) + [
            "today and pack1200 reused from " + str(args.reuse)
            + "; pack1200_coverage run separately with overridden tier instructions"]
        (out / "results_coverage.json").write_text(json.dumps(results, indent=1, default=str))
        for arm in results:
            if isinstance(results[arm], dict) and "summary" in results[arm]:
                print(arm, results[arm]["summary"], flush=True)
        return

    # Smoke: prove the variable is engaged before the full spend (CLAUDE.md).
    smoke = articles[:SMOKE_N]
    tally_before = get_tally().prompt_tokens
    a_rows = await run_arm("today", smoke, 0.0, out)
    b_rows = await run_arm("pack1200", smoke, sum(r["cost"] for r in a_rows), out)
    spent = sum(r["cost"] for r in a_rows + b_rows)
    print(f"smoke: today episodes={sum(r['episodes'] for r in a_rows)} "
          f"pack1200 episodes={sum(r['episodes'] for r in b_rows)} spent=${spent:.3f}",
          flush=True)

    # Write smoke.json BEFORE gate checks (to preserve results if gates refuse)
    (out / "smoke.json").write_text(json.dumps(a_rows + b_rows, indent=1))

    # Check smoke gates
    refusal = _smoke_refusal(a_rows, b_rows, get_tally().prompt_tokens != tally_before)
    if refusal:
        raise SystemExit(refusal)

    if args.smoke_only:
        return

    results: dict[str, Any] = {}
    for arm in ARMS:
        rows = await run_arm(arm, articles, spent, out)
        spent += sum(r["cost"] for r in rows)
        results[arm] = {"summary": summarise(rows), "rows": rows}
    results["spent_usd_including_smoke"] = spent
    results["method_notes"] = [
        "articles run sequentially through IngestDriver.ingest_article; "
        "ingest_source's warm-up barrier and sibling spreading were bypassed "
        "(they do not interact with the chunking variable)"
    ]
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    for arm in ARMS:
        print(arm, results[arm]["summary"], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
