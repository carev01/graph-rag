"""A/B (BACKLOG 59): today's routed extraction vs Claude Haiku 5.5 as the only model.
PAID -- run only with the user's authorisation and --cap set to the authorised amount.

Arm `routed` is production: solar-pro4 (cheap tier, cache layout on) with dense tables
routed to gpt-5-mini. Arm `haiku` is one model for everything: Claude Haiku 5.5,
thinking disabled, cache layout on, routing off. Each arm ingests the SAME articles into
its own throwaway Neo4j testcontainer; the production graph is never touched.

    uv run python scripts/haiku_extract_ab.py --sample d6_sample.json --out DIR \
        --articles 200 --cap 12 [--concurrency 4] [--smoke]

Engagement gate (CLAUDE.md cost rule), run before anything else spends: the haiku arm's
first GATE_N articles run alone, and the run refuses to continue unless Claude served
calls, NO other LLM call happened, the system-block cache was read, and episodes were
written. `--smoke` stops after the gate.

Measured per arm: articles failed (recorded, never skipped silently), episodes, facts,
entities, dedup index validity (dedup_guard), integer-array overruns (haiku only: Claude
rejects maxItems), tokens, dollars, wall time; then the chunk-A/B judge scores both
arms' facts against the source markdown (distinct supported ideas, unsupported facts).
Articles run `--concurrency` at a time per arm, same order in both arms; production's
warm-up ordering is not reproduced (as in the earlier A/Bs), so duplicates are not a
measured outcome here.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

from testcontainers.neo4j import Neo4jContainer

sys.path.insert(0, str(Path(__file__).parent))
from chunk_ab import NEO4J_IMAGE, _article_graph, _seed  # noqa: E402
from chunk_ab_judge import _judge_one, build_prompt, labels_for, score  # noqa: E402

import graph_extract.anthropic_chat as ac  # noqa: E402
from answer_api.eval_router import _eval_judge_client_and_model  # noqa: E402
from docext.client import make_docext_client  # noqa: E402
from graph_extract.cli import _build_ingest_driver  # noqa: E402
from graph_extract.config import ExtractSettings, get_extract_settings  # noqa: E402
from graph_extract.content_fetch import fetch_article  # noqa: E402
from graph_extract.usage import CURRENT_USAGE_TALLY, UsageTally, get_tally  # noqa: E402

# $/1M tokens: input, cached input, output.
RATES = {"cheap": (0.09, 0.018, 0.36),       # solar-pro4 via OpenRouter/Upstage
         "strong": (0.25, 0.025, 2.00),      # gpt-5-mini on Azure
         "haiku": (0.10, 0.010, 0.50)}       # Claude Haiku 5.5 (< 100k-token prompts)
HAIKU_MODEL = "claude-haiku-5-5"
GATE_N = 2


def arm_settings(name: str, base: ExtractSettings) -> dict:
    if name == "routed":
        return {"cheap_llm_cache_layout": True}
    if not ac.is_anthropic(base.report_llm_base_url) or not base.report_llm_api_key:
        raise SystemExit("haiku arm needs the Anthropic key: REPORT_LLM_BASE_URL must be "
                         "api.anthropic.com with REPORT_LLM_API_KEY set")
    return {"llm_base_url": "https://api.anthropic.com",
            "llm_api_key": base.report_llm_api_key, "llm_model": HAIKU_MODEL,
            "llm_reasoning_effort": "none", "llm_cache_layout": True,
            "extraction_routing": False}


def article_cost(tier: str, prompt: int, cached: int, completion: int) -> float:
    pin, pcached, pout = RATES[tier]
    return ((prompt - cached) * pin + cached * pcached + completion * pout) / 1e6


class Spend:
    def __init__(self, cap: float):
        self.cap, self.total, self.stopped = cap, 0.0, False

    def add(self, dollars: float) -> None:
        self.total += dollars
        if self.total > self.cap and not self.stopped:
            self.stopped = True
            print(f"SPEND CAP ${self.cap} reached (${self.total:.2f}): no new articles start",
                  flush=True)


async def ingest_one(name: str, ingest, driver, a: dict, spend: Spend, out: Path) -> dict:
    tally = UsageTally()
    CURRENT_USAGE_TALLY.set(tally)        # this task's own context
    start = time.perf_counter()
    row: dict = {"arm": name, "article_id": a["id"], "vendor": a.get("vendor")}
    try:
        res = await ingest.ingest_article(a["id"])
        g = await _article_graph(driver, a["id"])
        tier = "haiku" if name == "haiku" else res.tier
        row.update(tier=tier, episodes=res.episodes_added, facts=g["facts"],
                   entities=g["entities"], dedup=asdict(res.dedup),
                   fact_texts=g["fact_texts"], error=None)
    except Exception as exc:              # recorded and counted, never dropped
        tier = "haiku" if name == "haiku" else "cheap"
        row.update(tier=tier, episodes=0, facts=0, entities=0, dedup=None,
                   fact_texts=[], error=f"{type(exc).__name__}: {exc}"[:500])
    row.update(seconds=round(time.perf_counter() - start, 1),
               prompt_tokens=tally.prompt_tokens, cached_tokens=tally.cached_tokens,
               completion_tokens=tally.completion_tokens, llm_calls=tally.calls,
               cost=article_cost(row["tier"], tally.prompt_tokens, tally.cached_tokens,
                                 tally.completion_tokens))
    spend.add(row["cost"])
    with (out / "rows.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{name}] {a['id'][:8]} tier={row['tier']} eps={row['episodes']} "
          f"facts={row['facts']} calls={row['llm_calls']} ${row['cost']:.4f} "
          f"{row['seconds']}s{' ERROR ' + row['error'] if row['error'] else ''}", flush=True)
    return row


async def run_arm(name: str, articles: list[dict], spend: Spend, out: Path,
                  concurrency: int, gate: str = "", gate_ok: asyncio.Event | None = None
                  ) -> list[dict]:
    """`gate` "gate" runs the engagement gate first (and sets `gate_ok` on a pass);
    "smoke" stops after it."""
    rows: list[dict] = []
    base = get_extract_settings()
    with Neo4jContainer(NEO4J_IMAGE) as neo:
        s = base.model_copy(update={
            "neo4j_uri": neo.get_connection_url(), "neo4j_user": "neo4j",
            "neo4j_password": neo.password, "group_id": f"hx-{name}",
            **arm_settings(name, base)})
        ingest, graphiti, docext, driver = await _build_ingest_driver(s)
        try:
            await _seed(driver, articles)
            todo = list(articles)
            if gate:
                calls0, claude0 = get_tally().calls, ac.CLAUDE_USAGE.calls
                for a in todo[:GATE_N]:
                    rows.append(await ingest_one(name, ingest, driver, a, spend, out))
                todo = todo[GATE_N:]
                check_gate(rows, get_tally().calls - calls0, ac.CLAUDE_USAGE.calls - claude0)
                if gate_ok is not None:
                    gate_ok.set()
                if gate == "smoke":
                    return rows
            sem = asyncio.Semaphore(concurrency)

            async def bounded(a: dict) -> dict | None:
                async with sem:
                    if spend.stopped:
                        return None
                    return await ingest_one(name, ingest, driver, a, spend, out)

            done = await asyncio.gather(*[bounded(a) for a in todo])
            rows += [r for r in done if r is not None]
        finally:
            await driver.close()
            await docext.aclose()
            await graphiti.close()
    return rows


def check_gate(rows: list[dict], llm_calls: int, claude_calls: int) -> None:
    episodes = sum(r["episodes"] for r in rows)
    reads = ac.CLAUDE_USAGE.cache_read_tokens
    problems = []
    if claude_calls == 0:
        problems.append("Claude served no calls")
    if llm_calls != claude_calls:
        problems.append(f"{llm_calls - claude_calls} LLM call(s) went to another provider")
    if reads == 0:
        problems.append("no cache read")
    if episodes == 0:
        problems.append("no episode written")
    errors = [r["error"] for r in rows if r["error"]]
    if errors:
        problems.append(f"errors: {errors}")
    if problems:
        raise SystemExit("ENGAGEMENT GATE FAILED -- refusing to spend: " + "; ".join(problems))
    print(f"GATE PASSED: {claude_calls} Claude calls, 0 elsewhere, {reads} tokens read "
          f"from cache, {episodes} episodes, ${ac.CLAUDE_USAGE.dollars(0.10, 0.50):.4f}",
          flush=True)


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error"]]
    dedup_calls = sum(r["dedup"]["calls"] for r in ok)
    invalid = sum(r["dedup"]["invalid_calls"] for r in ok)
    tot = {k: sum(r[k] for r in rows) for k in
           ("episodes", "facts", "entities", "prompt_tokens", "cached_tokens",
            "completion_tokens", "llm_calls", "cost")}
    tot.update(articles=len(rows), failed=len(rows) - len(ok),
               tiers={t: sum(1 for r in rows if r["tier"] == t)
                      for t in {r["tier"] for r in rows}},
               dedup_calls=dedup_calls, dedup_invalid_calls=invalid,
               dedup_invalid_rate=round(invalid / dedup_calls, 4) if dedup_calls else 0,
               cost_per_article=round(tot["cost"] / len(rows), 5) if rows else 0,
               cost_per_1k_facts=round(1000 * tot["cost"] / tot["facts"], 3)
               if tot["facts"] else None,
               seconds_median=statistics.median([r["seconds"] for r in rows]) if rows else 0)
    return tot


async def judge(by_arm: dict[str, dict[str, dict]], out: Path) -> dict:
    s = get_extract_settings()
    client, model = _eval_judge_client_and_model(s)
    dx = make_docext_client(base_url=s.docext_base_url, read_key=s.docext_read_key,
                            admin_key=s.docext_admin_key, verify_tls=s.docext_verify_tls)
    arms = list(by_arm)
    per: list[dict] = []
    unscored: list[str] = []
    try:
        for aid, r0 in by_arm["routed"].items():
            r1 = by_arm["haiku"].get(aid)
            if r1 is None or r0["error"] or r1["error"]:
                continue
            md = (await fetch_article(dx, aid)).content_markdown
            lab = labels_for(aid, arms)
            facts = {lab[a]: by_arm[a][aid]["fact_texts"] for a in arms}
            obj = await _judge_one(client, model, build_prompt(md, facts),
                                   {k: len(v) for k, v in facts.items()})
            if obj is None:
                unscored.append(aid)
                continue
            row = {"article_id": aid, **{a: score(obj["labels"][lab[a]]) for a in arms}}
            per.append(row)
            with (out / "judged.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
    finally:
        await dx.aclose()
        await client.close()
    ratios = [r["haiku"]["distinct"] / r["routed"]["distinct"]
              for r in per if r["routed"]["distinct"]]
    return {"judge_model": model, "articles": len(per), "unscored": unscored,
            "distinct_ratio_median": statistics.median(ratios) if ratios else None,
            "totals": {a: {k: sum(r[a][k] for r in per)
                           for k in ("facts", "supported", "distinct", "redundant",
                                     "unsupported")} for a in arms}}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--articles", type=int, default=200)
    ap.add_argument("--cap", type=float, required=True, help="authorised spend, USD")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="stop after the gate")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = json.loads(Path(args.sample).read_text())
    rng = random.Random(20261014)
    articles = rng.sample(sample, min(args.articles, len(sample)))
    print(f"{len(articles)} articles from {len({a.get('vendor') for a in articles})} "
          f"vendors", flush=True)
    spend = Spend(args.cap)
    if args.smoke:
        rows = await run_arm("haiku", articles[:GATE_N], spend, out, 1, gate="smoke")
        print(json.dumps({"haiku": summarise(rows),
                          "claude_usage": asdict(ac.CLAUDE_USAGE)}, indent=1))
        return
    started = time.perf_counter()
    # The gate runs inside the haiku arm before its fan-out; the routed arm starts only
    # once the gate has passed, so a failed gate spends nothing on it.
    gate_ok = asyncio.Event()
    haiku_task = asyncio.create_task(
        run_arm("haiku", articles, spend, out, args.concurrency, gate="gate",
                gate_ok=gate_ok))
    waiter = asyncio.create_task(gate_ok.wait())
    await asyncio.wait({haiku_task, waiter}, return_when=asyncio.FIRST_COMPLETED)
    if not gate_ok.is_set():
        waiter.cancel()
        await haiku_task                 # re-raises the gate's SystemExit
        raise SystemExit("haiku arm ended before its gate passed")
    routed, haiku = await asyncio.gather(
        run_arm("routed", articles, spend, out, args.concurrency), haiku_task)
    wall = time.perf_counter() - started
    results = {"wall_seconds": round(wall), "spend_cap": args.cap,
               "claude_usage": asdict(ac.CLAUDE_USAGE),
               "claude_dollars_exact": round(ac.CLAUDE_USAGE.dollars(0.10, 0.50), 4),
               "array_overruns": asdict(ac.ARRAY_OVERRUNS),
               "routed": summarise(routed), "haiku": summarise(haiku)}
    (out / "results.json").write_text(json.dumps(results, indent=1))
    by_arm = {"routed": {r["article_id"]: r for r in routed},
              "haiku": {r["article_id"]: r for r in haiku}}
    results["judge"] = await judge(by_arm, out)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items()}, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
