"""A/B: cheap tier with vs without the prompt cache layout. PAID (authorised 2026-09-25,
12 articles, cap $4). Each arm ingests the SAME prose articles, in the same order, into
its own throwaway Neo4j testcontainer; the production graph is never touched.

    uv run python scripts/cache_layout_ab.py --sample d6_sample.json --out DIR [--articles 12]

Measures per arm: cached / prompt tokens (overall and steady state = articles 3..N,
after the learner's first calls), $/article at the cheap tier's real rates (cached input
billed separately), and fact quality via the chunk-A/B judge (distinct supported ideas,
unsupported facts, against the source markdown). Refuses to spend past article 2 of the
cache arm unless a static block was learned AND tokens were served from cache.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from testcontainers.neo4j import Neo4jContainer

sys.path.insert(0, str(Path(__file__).parent))
from chunk_ab import NEO4J_IMAGE, _article_graph, _seed, select_articles  # noqa: E402
from chunk_ab_judge import _judge_one, build_prompt, labels_for, score  # noqa: E402

from answer_api.eval_router import _eval_judge_client_and_model  # noqa: E402
from docext.client import make_docext_client  # noqa: E402
from graph_extract.article_router import is_dense_matrix  # noqa: E402
from graph_extract.cli import _build_ingest_driver  # noqa: E402
from graph_extract.config import get_extract_settings  # noqa: E402
from graph_extract.content_fetch import fetch_article  # noqa: E402
from graph_extract.usage import get_tally  # noqa: E402

# solar-pro4 via OpenRouter/Upstage, $/1M tokens: input, cached input, output.
CHEAP = (0.09, 0.018, 0.36)
SPEND_CAP_USD = 4.0
ARMS = {"off": {}, "cache": {"cheap_llm_cache_layout": True}}
GATE_AFTER = 2


class _Learned(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.n = 0

    def emit(self, record):
        if "static block learned" in record.getMessage():
            self.n += 1


def cost(prompt: int, cached: int, completion: int) -> float:
    pin, pcached, pout = CHEAP
    return ((prompt - cached) * pin + cached * pcached + completion * pout) / 1e6


async def run_arm(name: str, articles: list[dict], spent_before: float, out: Path) -> list[dict]:
    learned = _Learned()
    logging.getLogger("graph_extract.cache_layout").addHandler(learned)
    logging.getLogger("graph_extract.cache_layout").setLevel(logging.INFO)
    rows: list[dict] = []
    with Neo4jContainer(NEO4J_IMAGE) as neo:
        s = get_extract_settings().model_copy(update={
            "neo4j_uri": neo.get_connection_url(), "neo4j_user": "neo4j",
            "neo4j_password": neo.password, "group_id": f"cl-{name}", **ARMS[name]})
        ingest, graphiti, docext, driver = await _build_ingest_driver(s)
        try:
            await _seed(driver, articles)
            for i, a in enumerate(articles, 1):
                t = get_tally()
                p0, c0, k0 = t.prompt_tokens, t.completion_tokens, t.cached_tokens
                start = time.perf_counter()
                res = await ingest.ingest_article(a["id"])
                seconds = time.perf_counter() - start
                t = get_tally()
                dp, dc, dk = t.prompt_tokens - p0, t.completion_tokens - c0, t.cached_tokens - k0
                g = await _article_graph(driver, a["id"])
                row = {"arm": name, "article_id": a["id"], "tier": res.tier,
                       "episodes": res.episodes_added, "facts": g["facts"],
                       "entities": g["entities"], "seconds": seconds, "prompt_tokens": dp,
                       "cached_tokens": dk, "completion_tokens": dc,
                       "cost": cost(dp, dk, dc), "dedup": asdict(res.dedup),
                       "fact_texts": g["fact_texts"]}
                rows.append(row)
                with (out / "rows.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                spent = spent_before + sum(r["cost"] for r in rows)
                print(f"[{name}] {i}/{len(articles)} tier={res.tier} eps={res.episodes_added} "
                      f"facts={g['facts']} cached={dk}/{dp} ${spent:.3f}", flush=True)
                if name == "cache" and i == GATE_AFTER:
                    served = sum(r["cached_tokens"] for r in rows)
                    if learned.n == 0 or served == 0:
                        raise SystemExit(f"cache arm not engaged after {i} articles "
                                         f"(blocks learned={learned.n}, cached={served}) -- "
                                         "refusing to spend on the rest")
                    print(f"[cache] engaged: {learned.n} static block(s) learned, "
                          f"{served} tokens served from cache", flush=True)
                if spent > SPEND_CAP_USD:
                    raise SystemExit(f"spend cap ${SPEND_CAP_USD} exceeded (${spent:.2f})")
        finally:
            await driver.close()
            await docext.aclose()
            await graphiti.close()
            logging.getLogger("graph_extract.cache_layout").removeHandler(learned)
    return rows


def summarise(rows: list[dict]) -> dict:
    tot = {k: sum(r[k] for r in rows) for k in
           ("episodes", "facts", "entities", "seconds", "prompt_tokens", "cached_tokens",
            "completion_tokens", "cost")}
    steady = rows[GATE_AFTER:]
    sp = sum(r["prompt_tokens"] for r in steady)
    tot["cache_rate"] = round(tot["cached_tokens"] / tot["prompt_tokens"], 3) if tot["prompt_tokens"] else 0
    tot["cache_rate_steady"] = round(sum(r["cached_tokens"] for r in steady) / sp, 3) if sp else 0
    return tot


async def judge(rows: dict[str, dict[str, dict]], out: Path) -> dict:
    s = get_extract_settings()
    client, model = _eval_judge_client_and_model(s)
    dx = make_docext_client(base_url=s.docext_base_url, read_key=s.docext_read_key,
                            admin_key=s.docext_admin_key, verify_tls=s.docext_verify_tls)
    arms = list(ARMS)
    per: list[dict] = []
    unscored: list[str] = []
    try:
        for aid in rows["off"]:
            md = (await fetch_article(dx, aid)).content_markdown
            lab = labels_for(aid, arms)
            facts = {lab[a]: rows[a][aid]["fact_texts"] for a in arms}
            obj = await _judge_one(client, model, build_prompt(md, facts),
                                   {k: len(v) for k, v in facts.items()})
            if obj is None:
                unscored.append(aid)
                continue
            row = {"article_id": aid, **{a: score(obj["labels"][lab[a]]) for a in arms}}
            per.append(row)
            with (out / "judged.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print("judged", aid[:8], {a: (row[a]["distinct"], row[a]["unsupported"]) for a in arms},
                  flush=True)
    finally:
        await dx.aclose()
        await client.close()
    ratios = [r["cache"]["distinct"] / r["off"]["distinct"] for r in per if r["off"]["distinct"]]
    return {"judge_model": model, "articles": len(per), "unscored": unscored,
            "distinct_ratio_median": statistics.median(ratios) if ratios else None,
            "totals": {a: {k: sum(r[a][k] for r in per)
                           for k in ("facts", "supported", "distinct", "redundant", "unsupported")}
                       for a in arms}}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--articles", type=int, default=12)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = get_extract_settings()
    sample = json.loads(Path(args.sample).read_text())
    # Prose only: the cache layout is a cheap-tier switch, and a dense article would
    # route to the strong tier and measure nothing.
    prose = [a for a in sample if not is_dense_matrix(
        "\n".join(c["t"] for c in a["chunks"]), ratio_threshold=s.dense_table_line_ratio,
        pipe_threshold=s.dense_pipe_count)]
    articles = select_articles(prose, args.articles, random.Random(20260925), s)
    print(f"{len(articles)} prose articles selected", flush=True)
    off = await run_arm("off", articles, 0.0, out)
    cache = await run_arm("cache", articles, sum(r["cost"] for r in off), out)
    results = {"off": {"summary": summarise(off), "rows": off},
               "cache": {"summary": summarise(cache), "rows": cache}}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    rows = {a: {r["article_id"]: r for r in results[a]["rows"]} for a in ARMS}
    verdict = await judge(rows, out)
    (out / "judge_summary.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps({"off": results["off"]["summary"], "cache": results["cache"]["summary"],
                      "judge": verdict}, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
