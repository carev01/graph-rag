"""Load test for the vast.ai vLLM tier: throughput sweep + held-out validity.

    uv run python scripts/vast/loadtest.py --key-file ~/.config/graph-rag/vast/api_key \
        --conc 1,8,32 --per-prompt 8

Needs data/ft-12k/val.jsonl (gitignored; regenerate with scripts/build_extraction_dataset.py)
and the vLLM endpoint reachable at --base (an SSH tunnel, see docs/deploy/vast-gpu.md).

Replays held-out production prompts (data/ft-12k/val.jsonl) at increasing
concurrency, records req/s, output tok/s, p50/p95 latency and JSON validity
(scored by the same validate_capture the fine-tune eval used).
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.build_extraction_dataset import RESPONSE_MODELS, validate_capture  # noqa: E402

import httpx  # noqa: E402


def load(n_per_prompt: int, seed: int = 20260922):
    by = collections.defaultdict(list)
    for line in (REPO / "data/ft-12k/val.jsonl").open():
        d = json.loads(line)
        by[d["prompt_name"]].append(d)
    rng = random.Random(seed)
    out = []
    for name, recs in sorted(by.items()):
        rng.shuffle(recs)
        out += recs[:n_per_prompt]
    rng.shuffle(out)
    return out


def score(rec, txt):
    name = rec["prompt_name"]
    try:
        obj = RESPONSE_MODELS[name](**json.loads(txt))
    except Exception:  # noqa: BLE001
        return "bad_json"
    msgs = [m for m in rec["messages"] if m["role"] != "assistant"]
    return "defect" if validate_capture(name, msgs, obj, allow_overlong_summaries=True) else "clean"


async def run(base, key, model, recs, conc, timeout):
    sem = asyncio.Semaphore(conc)
    hdr = {"Authorization": f"Bearer {key}"} if key else {}
    res = []

    async with httpx.AsyncClient(headers=hdr, timeout=timeout) as c:
        async def one(rec):
            body = {"model": model,
                    "messages": [m for m in rec["messages"] if m["role"] != "assistant"],
                    "temperature": 0, "top_p": 1.0, "presence_penalty": 0, "max_tokens": 4096,
                    "chat_template_kwargs": {"enable_thinking": False}}
            async with sem:
                t = time.time()
                try:
                    r = await c.post(f"{base}/chat/completions", json=body)
                    out = r.json()
                    ch = out["choices"][0]
                    u = out.get("usage", {})
                    res.append((rec, ch["message"]["content"], ch.get("finish_reason"),
                                time.time() - t, u.get("prompt_tokens", 0),
                                u.get("completion_tokens", 0)))
                except Exception as e:  # noqa: BLE001
                    res.append((rec, None, f"{type(e).__name__}: {str(e)[:100]}",
                                time.time() - t, 0, 0))

        t0 = time.time()
        await asyncio.gather(*(one(r) for r in recs))
        wall = time.time() - t0
    return res, wall


def summarise(conc, res, wall):
    lat = sorted(r[3] for r in res)
    outcomes = collections.Counter(
        "call_error" if r[1] is None else score(r[0], r[1]) for r in res)
    trunc = sum(1 for r in res if r[1] is not None and r[2] != "stop")
    ptok = sum(r[4] for r in res)
    ctok = sum(r[5] for r in res)
    row = {"conc": conc, "n": len(res), "wall_s": round(wall, 1),
           "req_s": round(len(res) / wall, 2),
           "prompt_tok_s": round(ptok / wall), "out_tok_s": round(ctok / wall),
           "p50_s": round(lat[len(lat) // 2], 1), "p95_s": round(lat[int(len(lat) * .95)], 1),
           "truncated": trunc, **outcomes}
    print(json.dumps(row), flush=True)
    return row


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18080/v1")
    ap.add_argument("--key-file", type=Path)
    ap.add_argument("--model", default="qwen35-graphrag")
    ap.add_argument("--conc", default="1,4,8,16,32")
    ap.add_argument("--per-prompt", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    key = a.key_file.read_text().strip() if a.key_file else None
    recs = load(a.per_prompt)
    print(f"{len(recs)} records/level", flush=True)
    rows = []
    for c in [int(x) for x in a.conc.split(",")]:
        res, wall = await run(a.base, key, a.model, recs, c, a.timeout)
        rows.append(summarise(c, res, wall))
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
