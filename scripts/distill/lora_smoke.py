"""LoRA-serving smoke test (spec docs/superpowers/specs/2026-10-29-verifier-trial-design.md
section 3). Run against the rental's vLLM (one Qwen3.5-4B base, adapters `extractor` =
ckpt100 and `verifier` = verifier1) BEFORE any trial spend; exit 1 on any failed check.

    V2_BASE_URL=http://127.0.0.1:18080/v1 V2_API_KEY=$(cat key) uv run --extra dev \\
        python -m scripts.distill.lora_smoke --capture D/capture-run/capture-v2.jsonl \\
        --verifier-data V1 --verifier-predictions V1/test_predictions.jsonl --out smoke.json

1. the extractor adapter reproduces merged ckpt100's captured edge replies (same prompt,
   response_format, temperature 0): >= 45/50 byte-identical and every reply parses;
2. the served verifier's verdicts agree with the notebook's test predictions on >= 95% of
   facts (vLLM vs HF generation drift);
3. throughput, extractor adapter vs the bare base model on the same prompts (reported).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

from graph_extract.fact_verifier import max_tokens, parse_reply

EDGE_PROMPT = "extract_edges.edge"
MIN_IDENTICAL = 45          # of 50
MIN_VERIFIER_AGREEMENT = 0.95


def replay_requests(capture_lines: list[str], n: int, seed: int = 20261029) -> list[dict]:
    """A seeded sample of finished cheap-tier edge replies with their exact request."""
    pool = []
    for line in capture_lines:
        r = json.loads(line)
        if (r.get("prompt_name") == EDGE_PROMPT and r.get("tier") == "cheap"
                and r.get("finish_reason") == "stop" and r.get("completion")):
            pool.append({k: r[k] for k in ("call_id", "messages", "response_format", "completion")})
    return random.Random(seed).sample(pool, min(n, len(pool)))


def _triples(text: str | None) -> set | None:
    try:
        return {(e["source_entity_name"], e["target_entity_name"], e["relation_type"])
                for e in json.loads(text)["edges"]}
    except (ValueError, KeyError, TypeError):
        return None


def extractor_equivalence(pairs: list[tuple[str | None, str | None]]) -> dict:
    """pairs = (captured merged-model reply, regenerated adapter reply)."""
    identical, differing, unparseable = 0, [], 0
    for i, (want, got) in enumerate(pairs):
        if want == got:
            identical += 1
            continue
        a, b = _triples(want), _triples(got)
        unparseable += b is None
        jac = len(a & b) / len(a | b) if a is not None and b is not None and (a | b) else None
        differing.append({"index": i, "edge_jaccard": jac})
    return {"n": len(pairs), "identical": identical, "unparseable": unparseable,
            "differing": differing}


def extractor_passes(rep: dict) -> bool:
    return rep["identical"] >= MIN_IDENTICAL * rep["n"] / 50 and rep["unparseable"] == 0


NOISE_MARGIN = 5   # of 50: the pre-stated tolerance (verifier-trial OPS.md, 2026-10-29)


def adapter_equivalent(identical_adapter: int, identical_merged: int, unparseable: int) -> bool:
    """The adapter differs from merged ckpt100 (served now, same box) no more than merged
    ckpt100 differs from its own captured replies, give or take NOISE_MARGIN."""
    return identical_adapter >= identical_merged - NOISE_MARGIN and unparseable == 0


def verifier_agreement(n_facts: dict[str, int], notebook: dict[str, str],
                       served: dict[str, str]) -> dict:
    """Per-fact verdict agreement on the records whose notebook reply parses."""
    facts = agree = bad = 0
    for cid, n in n_facts.items():
        want = parse_reply(notebook.get(cid), n)
        if want is None:
            continue
        got = parse_reply(served.get(cid), n)
        facts += n
        if got is None:
            bad += 1
            continue
        agree += sum(a["verdict"] == b["verdict"] for a, b in zip(want, got))
    return {"facts": facts, "agree": agree, "share": agree / facts if facts else 0.0,
            "unparseable": bad}


def verifier_passes(rep: dict) -> bool:
    return rep["share"] >= MIN_VERIFIER_AGREEMENT and rep["unparseable"] == 0


async def _chat(client, sem, model: str, messages, max_tok: int,
                response_format=None) -> tuple[str | None, int, float]:
    async with sem:
        t0 = time.perf_counter()
        kw = {"response_format": response_format} if response_format else {}
        r = await client.chat.completions.create(model=model, messages=messages, temperature=0,
                                                 max_tokens=max_tok, **kw)
        # An empty reply stays None: parse_reply and the equivalence check count it.
        return (r.choices[0].message.content, r.usage.completion_tokens,
                time.perf_counter() - t0)


async def replay_only(a: argparse.Namespace, client) -> int:
    """Noise-floor mode: replay the extractor prompts on one served model, save the replies,
    and compare them with the captured ones (and with --compare-with's, if given)."""
    sem = asyncio.Semaphore(a.concurrency)
    reqs = replay_requests(Path(a.capture).read_text().splitlines(), a.n)
    outs = await asyncio.gather(*[_chat(client, sem, a.replay_only, r["messages"], 16384,
                                        r["response_format"]) for r in reqs])
    texts = [o[0] for o in outs]
    Path(a.save_replies).write_text(json.dumps(
        {r["call_id"]: t for r, t in zip(reqs, texts)}, ensure_ascii=False))
    report: dict = {"model": a.replay_only,
                    "vs_captured": extractor_equivalence([(r["completion"], t)
                                                          for r, t in zip(reqs, texts)])}
    if a.compare_with:
        other = json.loads(Path(a.compare_with).read_text())
        report["vs_compare_with"] = extractor_equivalence([(other.get(r["call_id"]), t)
                                                           for r, t in zip(reqs, texts)])
    Path(a.out).write_text(json.dumps(report, indent=2))
    for k in ("vs_captured", "vs_compare_with"):
        if k in report:
            print(f"{a.replay_only} {k}: {report[k]['identical']}/{report[k]['n']} byte-identical, "
                  f"{report[k]['unparseable']} unparseable")
    return 0


async def run(a: argparse.Namespace) -> int:
    import openai
    client = openai.AsyncOpenAI(base_url=os.environ["V2_BASE_URL"],
                                api_key=os.environ["V2_API_KEY"].strip(), timeout=900)
    if a.replay_only:
        return await replay_only(a, client)
    models = {m.id for m in (await client.models.list()).data}
    report: dict = {"models": sorted(models)}
    missing = {"extractor", "verifier"} - models
    if missing:
        print(f"SMOKE FAILED: vLLM does not serve {sorted(missing)} (has {sorted(models)})")
        Path(a.out).write_text(json.dumps(report, indent=2))
        return 1
    sem = asyncio.Semaphore(a.concurrency)
    reqs = replay_requests(Path(a.capture).read_text().splitlines(), a.n)

    async def replay(model: str) -> tuple[list[str], float, int]:
        t0 = time.perf_counter()
        out = await asyncio.gather(*[_chat(client, sem, model, r["messages"], 16384,
                                           r["response_format"]) for r in reqs])
        return [o[0] for o in out], time.perf_counter() - t0, sum(o[1] for o in out)

    texts, secs, toks = await replay("extractor")
    report["extractor"] = extractor_equivalence([(r["completion"], t) for r, t in zip(reqs, texts)])
    report["throughput"] = {"extractor_tokens_per_s": round(toks / secs, 1)}
    if a.base_model:
        _, bsecs, btoks = await replay(a.base_model)
        report["throughput"]["base_tokens_per_s"] = round(btoks / bsecs, 1)

    test = [json.loads(x) for x in (Path(a.verifier_data) / "test.jsonl").read_text().splitlines()
            if x.strip()][:a.n]
    nb = {json.loads(x)["call_id"]: json.loads(x)["reply"]
          for x in Path(a.verifier_predictions).read_text().splitlines() if x.strip()}
    outs = await asyncio.gather(*[_chat(client, sem, "verifier", r["messages"][:-1],
                                        max_tokens(len(r["meta"]["final"]))) for r in test])
    report["verifier"] = verifier_agreement(
        {r["meta"]["call_id"]: len(r["meta"]["final"]) for r in test}, nb,
        {r["meta"]["call_id"]: o[0] for r, o in zip(test, outs)})
    ok_e, ok_v = extractor_passes(report["extractor"]), verifier_passes(report["verifier"])
    report["passed"] = ok_e and ok_v
    Path(a.out).write_text(json.dumps(report, indent=2))
    e, v = report["extractor"], report["verifier"]
    print(f"extractor: {e['identical']}/{e['n']} byte-identical, {e['unparseable']} unparseable, "
          f"differing {e['differing']} -> {'PASS' if ok_e else 'FAIL'}")
    print(f"verifier: {v['agree']}/{v['facts']} facts agree ({v['share']:.3f}), "
          f"{v['unparseable']} unparseable -> {'PASS' if ok_v else 'FAIL'}")
    print(f"throughput: {report['throughput']}")
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lora_smoke")
    ap.add_argument("--capture", required=True, help="capture-v2.jsonl served by merged ckpt100")
    ap.add_argument("--verifier-data", help="dir with verifier1 test.jsonl")
    ap.add_argument("--verifier-predictions")
    ap.add_argument("--replay-only", default="",
                    help="noise-floor mode: replay the extractor prompts on this served model only")
    ap.add_argument("--save-replies", help="--replay-only: write {call_id: reply} here")
    ap.add_argument("--compare-with", help="--replay-only: another run's saved replies")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--base-model", default="qwen35-base",
                    help="the bare base's served name, for the throughput reference ('' skips)")
    a = ap.parse_args(argv)
    if a.replay_only and not a.save_replies:
        ap.error("--replay-only needs --save-replies")
    if not a.replay_only and not (a.verifier_data and a.verifier_predictions):
        ap.error("the smoke test needs --verifier-data and --verifier-predictions")
    return asyncio.run(run(a))


if __name__ == "__main__":
    sys.exit(main())
