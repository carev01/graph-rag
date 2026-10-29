"""Extraction-model A/B (BACKLOG 59): today's routed extraction vs one model for
everything. PAID -- run only with the user's authorisation and --cap set to the
authorised amount (the cap counts extraction AND the judge).

Arms:
- `routed`: production -- solar-pro4 (cheap tier, cache layout on) with dense tables
  routed to gpt-5-mini.
- `haiku`: Claude Haiku 5.5 for everything, thinking disabled, cache layout on.
- `luna`: openai/gpt-6-luna on OpenRouter for everything, reasoning "none", strict
  schemas (OpenAI structured outputs reject graphiti's otherwise), cache layout on.
- `v2`: production's GPU-worker setup -- routed, with the cheap tier on the distilled
  qwen35-4b-graphrag served by vLLM (`V2_BASE_URL`, key in `V2_API_KEY`; never on the
  command line), cache layout on as in the distillation capture, dense tables still on
  gpt-5-mini. Its cheap-tier tokens cost $0 here: the GPU is billed by the hour.
Each arm ingests the SAME articles into its own throwaway Neo4j testcontainer; the
production graph is never touched. Arms already measured can be reused from a previous
run's rows.jsonl (`--reuse`): same sample, same code, so only the judge reruns.

    uv run python scripts/haiku_extract_ab.py --sample d6_sample.json --out DIR \
        --arms luna --reuse PREV/rows.jsonl --judge routed,luna --cap 6 [--smoke]

`--exclude SAMPLE.json` drops the articles a distillation sample trained on
(`articles[].split == "train"`), so a fine-tune is never scored on its own training data.

Engagement gate (CLAUDE.md cost rule), run before anything else spends: each new
single-model arm's first GATE_N articles run alone, and the run refuses to continue
unless that arm's model served every LLM call with the variable under test engaged
(haiku: Claude only, cache read; luna: every captured request to gpt-6-luna with
reasoning "none" and a closed schema; v2: vLLM's own success counter rose by exactly the
cheap-tier calls) and episodes were written. `--smoke` stops there.

Measured per arm: articles failed (recorded, never skipped silently), episodes, facts,
entities, dedup index validity (dedup_guard), tokens, dollars, wall time; then the
chunk-A/B judge scores the judged pair's facts against the source markdown (distinct
supported ideas, unsupported facts). Articles run `--concurrency` at a time per arm;
production's warm-up ordering is not reproduced, so duplicates are not measured here.

Document context (2026-10-21): each article's real Vendor->Product->Source chain is
written into every arm's Neo4j (vendor and product from DocExtractor's article endpoint,
source name from the sample), so extraction runs as in production: with the
per-article DOCUMENT CONTEXT sentence, and with the grounding gate scoped to the
documented product. Runs before that date had neither. `--no-chain` reproduces them.
Rows record `document_context`, and `--reuse` refuses rows from the other setting.

`--capture` writes every LLM call of each arm run now to `capture-<arm>.jsonl` (prompt as
sent, completion, prompt type, tier): the on-policy material a DPO round is built from
(`scripts/distill/dpo_pairs.py`). `--judge none` skips the judge, for a capture-only run;
`--arms none --reuse ROWS` reruns only the judge (a second pass over the same facts).
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

import httpx
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
         "haiku": (0.10, 0.010, 0.50),       # Claude Haiku 5.5 (< 100k-token prompts)
         "luna": (0.10, 0.010, 0.50),        # openai/gpt-6-luna on OpenRouter
         "local": (0.0, 0.0, 0.0)}           # vLLM on a rented GPU, billed per hour
HAIKU_MODEL = "claude-haiku-5-5"
LUNA_MODEL = "openai/gpt-6-luna"
V2_MODEL = "qwen35-graphrag"
GATE_N = 2
# Arms that keep production's routing: the article's tier comes from the router, and
# each tier is priced at that arm's rate.
ROUTED_ARMS = {"routed": {"cheap": "cheap", "strong": "strong"},
               "v2": {"cheap": "local", "strong": "strong"},
               "v2a": {"cheap": "local", "strong": "strong"},
               "v2av": {"cheap": "local", "strong": "strong"}}
# Arms whose cheap tier is our vLLM: the gate checks vLLM's own success counter.
# v2a/v2av (spec 2026-10-29-verifier-trial): one base + LoRA adapters `extractor`
# (ckpt100) and `verifier` (verifier1); v2av turns the edge verifier on.
VLLM_ARMS = {"v2", "v2a", "v2av"}
ADAPTER_EXTRACTOR, ADAPTER_VERIFIER = "extractor", "verifier"


CHAIN_WRITE = (
    "UNWIND $rows AS r "
    "MATCH (a:Article {id: r.article}) "
    "MERGE (v:Vendor {id: r.vendor.id}) SET v.name = r.vendor.name "
    "MERGE (p:Product {id: r.product.id}) SET p.name = r.product.name, "
    "  p.vendor_id = r.vendor.id "
    "MERGE (s:Source {id: r.source.id}) SET s.name = r.source.name "
    "MERGE (v)-[:HAS_PRODUCT]->(p) MERGE (p)-[:HAS_SOURCE]->(s) "
    "MERGE (s)-[:HAS_ARTICLE]->(a)")
CHAIN_COUNT = (
    "MATCH (:Vendor)-[:HAS_PRODUCT]->(:Product)-[:HAS_SOURCE]->(:Source)"
    "-[:HAS_ARTICLE]->(a:Article) WHERE a.id IN $ids RETURN count(DISTINCT a) AS n")


def chain_from_docext(article: dict, source_name: str) -> dict:
    """The structural chain graph-sync would write for this article."""
    vendor, product, source_id = (article.get("vendor"), article.get("product"),
                                  article.get("source_id"))
    if not (vendor and vendor.get("id") and product and product.get("id") and source_id
            and source_name):
        raise SystemExit(f"article {article.get('id')}: DocExtractor returned no complete "
                         "vendor/product/source -- cannot write its document context")
    return {"article": article["id"],
            "vendor": {"id": vendor["id"], "name": vendor["name"]},
            "product": {"id": product["id"], "name": product["name"]},
            "source": {"id": source_id, "name": source_name}}


def check_reuse_context(rows: list[dict], chain: bool) -> None:
    other = [r for r in rows if bool(r.get("document_context")) != chain]
    if other:
        raise SystemExit(
            f"--reuse: {len(other)} rows were extracted "
            f"{'without' if chain else 'with'} document context, this run "
            f"{'with' if chain else 'without'} it -- not comparable "
            f"({'drop --no-chain' if not chain else 'pass --no-chain'} or rerun those arms)")


async def fetch_chains(articles: list[dict]) -> None:
    """Sets `a["chain"]` on every article, before anything spends."""
    s = get_extract_settings()
    async with make_docext_client(base_url=s.docext_base_url, read_key=s.docext_read_key,
                                  admin_key=s.docext_admin_key,
                                  verify_tls=s.docext_verify_tls) as dx:
        for a in articles:
            resp = await dx.get(f"/api/articles/{a['id']}")
            resp.raise_for_status()
            a["chain"] = chain_from_docext(resp.json(), a.get("source") or "")


async def seed_chains(driver, articles: list[dict]) -> None:
    rows = [a["chain"] for a in articles if a.get("chain")]
    if not rows:
        return
    await driver.execute_query(CHAIN_WRITE, rows=rows)
    r = await driver.execute_query(CHAIN_COUNT, ids=[c["article"] for c in rows])
    n = r.records[0]["n"]
    if n != len(rows):
        raise SystemExit(f"structural chains written for {n} of {len(rows)} articles")


def rate_key(arm: str, tier: str) -> str:
    return ROUTED_ARMS[arm][tier] if arm in ROUTED_ARMS else tier


def train_article_ids(distill_sample: dict) -> set[str]:
    return {a["article_id"] for a in distill_sample["articles"] if a.get("split") == "train"}


def vllm_success_total(metrics_text: str) -> float:
    """Sum of vLLM's `vllm:request_success_total` over every label set."""
    total = 0.0
    for line in metrics_text.splitlines():
        if line.startswith("vllm:request_success_total"):
            total += float(line.rsplit(" ", 1)[1])
    return total


async def vllm_requests(base_url: str) -> float:
    root = base_url.rstrip("/").removesuffix("/v1")
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{root}/metrics")
        r.raise_for_status()
        return vllm_success_total(r.text)


def capture_settings(name: str, out: Path, capture: bool) -> dict:
    """`--capture`: the arm's calls go to capture-<arm>.jsonl (luna always captures)."""
    return {"llm_capture_path": str(out / f"capture-{name}.jsonl")} if capture else {}


def arm_list(value: str) -> list[str]:
    """`--arms`/`--judge` value -> arm names; "none" (or empty) is no arm."""
    return [] if value.strip() in ("", "none") else [a for a in value.split(",") if a]


def arm_settings(name: str, base: ExtractSettings, out: Path) -> dict:
    if name == "routed":
        return {"cheap_llm_cache_layout": True}
    if name == "haiku":
        if not ac.is_anthropic(base.report_llm_base_url) or not base.report_llm_api_key:
            raise SystemExit("haiku arm needs the Anthropic key: REPORT_LLM_BASE_URL must "
                             "be api.anthropic.com with REPORT_LLM_API_KEY set")
        return {"llm_base_url": "https://api.anthropic.com",
                "llm_api_key": base.report_llm_api_key, "llm_model": HAIKU_MODEL,
                "llm_reasoning_effort": "none", "llm_cache_layout": True,
                "extraction_routing": False}
    if name == "luna":
        if not base.cheap_llm_api_key:
            raise SystemExit("luna arm needs the OpenRouter key (CHEAP_LLM_API_KEY)")
        return {"llm_base_url": "https://openrouter.ai/api/v1",
                "llm_api_key": base.cheap_llm_api_key, "llm_model": LUNA_MODEL,
                "llm_client_mode": "generic_json_schema", "llm_strict_schema": True,
                "llm_openrouter_reasoning_effort": "none", "llm_send_temperature": False,
                "llm_cache_layout": True,
                "extraction_routing": False,
                "llm_capture_path": str(out / "capture-luna.jsonl")}
    if name in VLLM_ARMS:
        import os
        url, key = os.environ.get("V2_BASE_URL", ""), os.environ.get("V2_API_KEY", "").strip()
        if not url or not key:
            raise SystemExit(f"{name} arm needs V2_BASE_URL and V2_API_KEY")
        out = {"cheap_llm_base_url": url, "cheap_llm_api_key": key,
               "cheap_llm_model": V2_MODEL if name == "v2" else ADAPTER_EXTRACTOR,
               "cheap_llm_client_mode": "generic_json_schema",
               "cheap_llm_timeout_seconds": 900.0, "cheap_llm_cache_layout": True}
        if name == "v2av":
            out.update(edge_verifier_enabled=True, edge_verifier_base_url=url,
                       edge_verifier_api_key=key, edge_verifier_model=ADAPTER_VERIFIER)
        return out
    raise SystemExit(f"unknown arm {name!r}")


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
            print(f"SPEND CAP ${self.cap} reached (${self.total:.2f}): nothing new starts",
                  flush=True)


async def ingest_one(name: str, ingest, driver, a: dict, spend: Spend, out: Path) -> dict:
    tally = UsageTally()
    CURRENT_USAGE_TALLY.set(tally)        # this task's own context
    start = time.perf_counter()
    row: dict = {"arm": name, "article_id": a["id"], "vendor": a.get("vendor"),
                 "document_context": bool(a.get("chain"))}
    try:
        res = await ingest.ingest_article(a["id"])
        g = await _article_graph(driver, a["id"])
        tier = res.tier if name in ROUTED_ARMS else name
        row.update(tier=tier, episodes=res.episodes_added, facts=g["facts"],
                   entities=g["entities"], dedup=asdict(res.dedup),
                   fact_texts=g["fact_texts"], error=None)
    except Exception as exc:              # recorded and counted, never dropped
        tier = "cheap" if name in ROUTED_ARMS else name
        row.update(tier=tier, episodes=0, facts=0, entities=0, dedup=None,
                   fact_texts=[], error=f"{type(exc).__name__}: {exc}"[:500])
    row.update(seconds=round(time.perf_counter() - start, 1),
               prompt_tokens=tally.prompt_tokens, cached_tokens=tally.cached_tokens,
               completion_tokens=tally.completion_tokens, llm_calls=tally.calls,
               cost=article_cost(rate_key(name, row["tier"]), tally.prompt_tokens,
                                 tally.cached_tokens, tally.completion_tokens))
    spend.add(row["cost"])
    with (out / "rows.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{name}] {a['id'][:8]} tier={row['tier']} eps={row['episodes']} "
          f"facts={row['facts']} calls={row['llm_calls']} ${row['cost']:.4f} "
          f"{row['seconds']}s{' ERROR ' + row['error'] if row['error'] else ''}", flush=True)
    return row


async def run_arm(name: str, articles: list[dict], spend: Spend, out: Path,
                  concurrency: int, gate: str = "", gate_ok: asyncio.Event | None = None,
                  capture: bool = False) -> list[dict]:
    """`gate` "gate" runs the engagement gate first (and sets `gate_ok` on a pass);
    "smoke" stops after it."""
    rows: list[dict] = []
    base = get_extract_settings()
    with Neo4jContainer(NEO4J_IMAGE) as neo:
        s = base.model_copy(update={
            "neo4j_uri": neo.get_connection_url(), "neo4j_user": "neo4j",
            "neo4j_password": neo.password, "group_id": f"hx-{name}",
            **arm_settings(name, base, out), **capture_settings(name, out, capture)})
        ingest, graphiti, docext, driver = await _build_ingest_driver(s)
        try:
            await _seed(driver, articles)
            await seed_chains(driver, articles)
            todo = list(articles)
            if gate:
                calls0, claude0 = get_tally().calls, ac.CLAUDE_USAGE.calls
                vllm0 = await vllm_requests(s.cheap_llm_base_url) if name in VLLM_ARMS else 0.0
                for a in todo[:GATE_N]:
                    rows.append(await ingest_one(name, ingest, driver, a, spend, out))
                todo = todo[GATE_N:]
                vllm_delta = (await vllm_requests(s.cheap_llm_base_url) - vllm0
                              if name in VLLM_ARMS else 0.0)
                check_gate(name, rows, get_tally().calls - calls0,
                           ac.CLAUDE_USAGE.calls - claude0, out, vllm_delta)
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


def check_gate(name: str, rows: list[dict], llm_calls: int, claude_calls: int,
               out: Path, vllm_delta: float = 0.0) -> None:
    episodes = sum(r["episodes"] for r in rows)
    problems = []
    if llm_calls == 0:
        problems.append("no LLM call")
    if episodes == 0:
        problems.append("no episode written")
    errors = [r["error"] for r in rows if r["error"]]
    if errors:
        problems.append(f"errors: {errors}")
    detail = ""
    if name == "haiku":
        if claude_calls != llm_calls:
            problems.append(f"{llm_calls - claude_calls} LLM call(s) went elsewhere")
        if ac.CLAUDE_USAGE.cache_read_tokens == 0:
            problems.append("no cache read")
        detail = f"{claude_calls} Claude calls, {ac.CLAUDE_USAGE.cache_read_tokens} cached"
    elif name == "luna":
        path = out / "capture-luna.jsonl"
        recs = ([json.loads(x) for x in path.read_text().splitlines()]
                if path.exists() else [])
        if len(recs) != llm_calls:
            problems.append(f"{len(recs)} captured requests vs {llm_calls} LLM calls")
        wrong = [r.get("model") for r in recs if r.get("model") != LUNA_MODEL]
        if wrong:
            problems.append(f"requests to other models: {sorted(set(map(str, wrong)))}")
        no_effort = [r for r in recs if (r.get("extra_body") or {}).get("reasoning")
                     != {"effort": "none"}]
        if no_effort:
            problems.append(f"{len(no_effort)} request(s) without reasoning effort none")
        open_schema = [r for r in recs if ((r.get("response_format") or {}).get(
            "json_schema") or {}).get("schema", {}).get("additionalProperties") is not False]
        if open_schema:
            problems.append(f"{len(open_schema)} request(s) without a closed schema")
        cached = sum(r.get("cached_tokens", 0) for r in recs)
        detail = f"{len(recs)} gpt-6-luna requests, reasoning none, {cached} cached"
    elif name in VLLM_ARMS:
        problems += vllm_gate_problems(name, rows, vllm_delta)
        detail = f"{vllm_delta:.0f} vLLM requests, verifier {_verifier_totals(rows)}"
    if problems:
        raise SystemExit(f"ENGAGEMENT GATE FAILED ({name}) -- refusing to spend: "
                         + "; ".join(problems))
    print(f"GATE PASSED ({name}): {detail}, {episodes} episodes, "
          f"${sum(r['cost'] for r in rows):.4f}", flush=True)


def _verifier_totals(rows: list[dict]) -> dict[str, int]:
    keys = ("verifier_calls", "verifier_dropped", "verifier_failed_open")
    return {k: sum((r.get("dedup") or {}).get(k, 0) for r in rows) for k in keys}


def vllm_gate_problems(name: str, rows: list[dict], vllm_delta: float) -> list[str]:
    """vLLM's success counter must rise by exactly the cheap-tier graphiti calls plus the
    verifier's calls (a separate client, not in llm_calls): proof both adapters served
    what the arm claims. v2av must have fired, dropped something and never failed open;
    the other arms must not have called a verifier at all."""
    problems = []
    cheap_calls = sum(r["llm_calls"] for r in rows if r["tier"] == "cheap")
    v = _verifier_totals(rows)
    if cheap_calls == 0:
        problems.append("no cheap-tier article among the gate articles")
    elif vllm_delta != cheap_calls + v["verifier_calls"]:
        problems.append(f"vLLM served {vllm_delta:.0f} requests for {cheap_calls} cheap-tier "
                        f"calls + {v['verifier_calls']} verifier calls")
    if name == "v2av":
        if v["verifier_calls"] == 0:
            problems.append("the edge verifier never ran")
        elif v["verifier_dropped"] == 0:
            problems.append("the edge verifier dropped nothing")
        if v["verifier_failed_open"]:
            problems.append(f"{v['verifier_failed_open']} verifier call(s) failed open")
    elif v["verifier_calls"]:
        problems.append(f"{name} must not run the edge verifier ({v['verifier_calls']} calls)")
    return problems


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
               **_verifier_totals(ok),
               dedup_invalid_rate=round(invalid / dedup_calls, 4) if dedup_calls else 0,
               cost_per_article=round(tot["cost"] / len(rows), 5) if rows else 0,
               cost_per_1k_facts=round(1000 * tot["cost"] / tot["facts"], 3)
               if tot["facts"] else None,
               seconds_median=statistics.median([r["seconds"] for r in rows]) if rows else 0)
    return tot


def openrouter_rates(model: str) -> tuple[float, float, float]:
    """$/1M input, cached input, output for an OpenRouter model id, from its listing."""
    data = httpx.get("https://openrouter.ai/api/v1/models", timeout=30).json()["data"]
    p = next(m["pricing"] for m in data if m["id"] == model)
    return (float(p["prompt"]) * 1e6, float(p.get("input_cache_read") or p["prompt"]) * 1e6,
            float(p["completion"]) * 1e6)


async def judge(pair: list[str], by_arm: dict[str, dict[str, dict]], out: Path,
                spend: Spend, rates: tuple[float, float, float]) -> dict:
    s = get_extract_settings()
    client, model = _eval_judge_client_and_model(s)
    dx = make_docext_client(base_url=s.docext_base_url, read_key=s.docext_read_key,
                            admin_key=s.docext_admin_key, verify_tls=s.docext_verify_tls)
    a0, a1 = pair
    per: list[dict] = []
    unscored: list[str] = []
    spent = 0.0
    try:
        for aid, r0 in by_arm[a0].items():
            r1 = by_arm[a1].get(aid)
            if r1 is None or r0["error"] or r1["error"]:
                continue
            if spend.stopped:
                print("judge stopped by the spend cap", flush=True)
                break
            md = (await fetch_article(dx, aid)).content_markdown
            lab = labels_for(aid, pair)
            facts = {lab[a]: by_arm[a][aid]["fact_texts"] for a in pair}
            t = UsageTally()
            CURRENT_USAGE_TALLY.set(t)
            obj = await _judge_one(client, model, build_prompt(md, facts),
                                   {k: len(v) for k, v in facts.items()})
            cost = ((t.prompt_tokens - t.cached_tokens) * rates[0]
                    + t.cached_tokens * rates[1] + t.completion_tokens * rates[2]) / 1e6
            spent += cost
            spend.add(cost)
            if obj is None:
                unscored.append(aid)
                continue
            row = {"article_id": aid, **{a: score(obj["labels"][lab[a]]) for a in pair}}
            per.append(row)
            with (out / "judged.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
    finally:
        await dx.aclose()
        await client.close()
    ratios = [r[a1]["distinct"] / r[a0]["distinct"] for r in per if r[a0]["distinct"]]
    return {"judge_model": model, "pair": pair, "articles": len(per), "unscored": unscored,
            "judge_dollars": round(spent, 4),
            "distinct_ratio_median": statistics.median(ratios) if ratios else None,
            "totals": {a: {k: sum(r[a][k] for r in per)
                           for k in ("facts", "supported", "distinct", "redundant",
                                     "unsupported")} for a in pair}}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--articles", type=int, default=200)
    ap.add_argument("--cap", type=float, required=True,
                    help="authorised spend, USD (extraction + judge)")
    ap.add_argument("--arms", default="routed,haiku", help="arms to run now (none: judge only)")
    ap.add_argument("--reuse", help="a previous run's rows.jsonl for arms not run now")
    ap.add_argument("--judge", default="routed,haiku", help="the pair to judge (none: skip)")
    ap.add_argument("--capture", action="store_true",
                    help="write every LLM call of each arm run now to capture-<arm>.jsonl")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="stop after the gate")
    ap.add_argument("--no-chain", action="store_true",
                    help="no structural chain: extraction without document context, as "
                         "every run before 2026-10-21")
    ap.add_argument("--exclude", help="a distillation sample.json: its train articles are "
                                      "dropped from the sample")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample = json.loads(Path(args.sample).read_text())
    if args.exclude:
        train = train_article_ids(json.loads(Path(args.exclude).read_text()))
        kept = [a for a in sample if a["id"] not in train]
        print(f"--exclude: dropped {len(sample) - len(kept)} distillation train articles",
              flush=True)
        sample = kept
    rng = random.Random(20261014)
    articles = rng.sample(sample, min(args.articles, len(sample)))
    ids = {a["id"] for a in articles}
    print(f"{len(articles)} articles from {len({a.get('vendor') for a in articles})} "
          f"vendors", flush=True)
    if not args.no_chain:
        await fetch_chains(articles)
        print(f"document context: {len({a['chain']['product']['id'] for a in articles})} "
              "products from DocExtractor", flush=True)
    run_now = arm_list(args.arms)
    pair = arm_list(args.judge)
    if pair and len(pair) != 2:
        raise SystemExit(f"--judge takes two arms or none, not {pair}")
    spend = Spend(args.cap)
    gated = [a for a in run_now if a != "routed"]
    # Judge pricing is looked up before anything spends: a model missing from the
    # listing must stop the run here, not after the extraction it would judge.
    jclient, jmodel = _eval_judge_client_and_model(get_extract_settings())
    await jclient.close()
    judge_rates = openrouter_rates(jmodel)
    print(f"judge {jmodel}: ${judge_rates[0]:.3f} in / ${judge_rates[2]:.3f} out per 1M",
          flush=True)

    if args.smoke:
        name = gated[0]
        rows = await run_arm(name, articles[:GATE_N], spend, out, 1, gate="smoke",
                             capture=args.capture)
        print(json.dumps({name: summarise(rows)}, indent=1))
        return

    reused: dict[str, list[dict]] = {}
    if args.reuse:
        for line in Path(args.reuse).read_text().splitlines():
            r = json.loads(line)
            if r["arm"] not in run_now and r["article_id"] in ids:
                reused.setdefault(r["arm"], []).append(r)
        for arm, rows in reused.items():
            check_reuse_context(rows, chain=not args.no_chain)
            print(f"reusing {arm}: {len(rows)} rows from {args.reuse}", flush=True)
    missing = [a for a in pair if a not in run_now and len(reused.get(a, [])) != len(ids)]
    if missing:
        raise SystemExit(f"judge pair needs arms {missing}: run them or --reuse a "
                         "complete rows.jsonl for this sample")

    started = time.perf_counter()
    results: dict = {}
    # Every single-model arm passes its gate before any arm fans out, so a failed
    # gate spends nothing beyond its own GATE_N articles.
    tasks: dict[str, asyncio.Task] = {}
    for name in gated:
        ok = asyncio.Event()
        tasks[name] = asyncio.create_task(
            run_arm(name, articles, spend, out, args.concurrency, gate="gate", gate_ok=ok,
                    capture=args.capture))
        waiter = asyncio.create_task(ok.wait())
        await asyncio.wait({tasks[name], waiter}, return_when=asyncio.FIRST_COMPLETED)
        if not ok.is_set():
            waiter.cancel()
            await tasks[name]                # re-raises the gate's SystemExit
            raise SystemExit(f"{name} arm ended before its gate passed")
    if "routed" in run_now:
        tasks["routed"] = asyncio.create_task(
            run_arm("routed", articles, spend, out, args.concurrency, capture=args.capture))
    finished = dict(zip(tasks, await asyncio.gather(*tasks.values())))
    by_arm_rows = {**reused, **finished}
    results.update(wall_seconds=round(time.perf_counter() - started), spend_cap=args.cap,
                   document_context=not args.no_chain,
                   claude_usage=asdict(ac.CLAUDE_USAGE),
                   array_overruns=asdict(ac.ARRAY_OVERRUNS),
                   reused=sorted(reused),
                   **{arm: summarise(rows) for arm, rows in by_arm_rows.items()})
    (out / "results.json").write_text(json.dumps(results, indent=1))
    by_arm = {arm: {r["article_id"]: r for r in rows} for arm, rows in by_arm_rows.items()}
    if pair:
        results["judge"] = await judge(pair, by_arm, out, spend, judge_rates)
    results["spend_total"] = round(spend.total, 4)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
