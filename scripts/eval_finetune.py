"""Held-out evaluation of the fine-tuned model against the val split.

The val split is a real held-out set: 17 articles, zero overlap with the 60
train articles, and the model never saw it. Each record carries the PROMPT the
production pipeline actually sends and the reference TARGET.

Validity is scored with `build_extraction_dataset.validate_capture` -- the same
checker that measured solar-pro4 on the capture -- so the two rates are
comparable rather than merely similar. A defect here is a real rejection site in
graphiti 0.30.1 or in this project's own guard, not a style preference.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_extraction_dataset import (  # noqa: E402
    RESPONSE_MODELS,
    validate_capture,
)

BASE = "http://srv-llm.home.lan:8080/v1/chat/completions"


async def call(session, msgs, model, sem, timeout):
    body = {"model": model, "messages": msgs,
            # Explicit: the server defaults are temperature 0.7 / top_k 20 /
            # top_p 0.8 / presence_penalty 1.5. A presence penalty punishes
            # repeated tokens, and JSON is mostly repeated structural tokens --
            # this project has it on record that repetition penalties make the
            # index-array failure WORSE.
            "temperature": 0, "top_p": 1.0, "presence_penalty": 0,
            "max_tokens": 4096}
    async with sem:
        t = time.time()
        try:
            r = await session.post(BASE, json=body, timeout=timeout)
            out = r.json()
        except Exception as e:  # noqa: BLE001 - any failure is a result, not a crash
            return None, f"{type(e).__name__}: {str(e)[:120]}", time.time() - t
    ch = out.get("choices", [{}])[0]
    return ch.get("message", {}).get("content"), ch.get("finish_reason"), time.time() - t


async def main() -> int:
    import httpx
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path, default=Path("data/ft-12k/val.jsonl"))
    ap.add_argument("--per-prompt", type=int, default=25)
    ap.add_argument("--model", default="qwen35-graphrag")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    by_prompt = collections.defaultdict(list)
    for line in args.val.open():
        d = json.loads(line)
        by_prompt[d["prompt_name"]].append(d)
    rng = random.Random(20260922)
    picked = []
    for name, recs in sorted(by_prompt.items()):
        rng.shuffle(recs)
        picked += recs[:args.per_prompt]
    print(f"evaluating {len(picked)} held-out records, concurrency {args.concurrency}",
          flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async with httpx.AsyncClient() as session:
        async def one(rec):
            nonlocal done
            msgs = [m for m in rec["messages"] if m["role"] != "assistant"]
            txt, finish, dt = await call(session, msgs, args.model, sem, args.timeout)
            done += 1
            if done % 10 == 0:
                print(f"  [{done}/{len(picked)}]", flush=True)
            return rec, txt, finish, dt

        results = await asyncio.gather(*(one(r) for r in picked))

    stats = collections.defaultdict(lambda: collections.Counter())
    defects = collections.defaultdict(collections.Counter)
    lat = collections.defaultdict(list)
    rows = []
    for rec, txt, finish, dt in results:
        name = rec["prompt_name"]
        s = stats[name]
        s["n"] += 1
        lat[name].append(dt)
        if txt is None:
            s["call_error"] += 1
            rows.append({"prompt": name, "outcome": "call_error", "detail": finish})
            continue
        if finish != "stop":
            s["truncated"] += 1
        try:
            payload = json.loads(txt)
        except Exception:
            s["unparseable"] += 1
            rows.append({"prompt": name, "outcome": "unparseable"})
            continue
        try:
            obj = RESPONSE_MODELS[name](**payload)
        except Exception as e:  # noqa: BLE001
            s["schema_invalid"] += 1
            defects[name][f"schema:{type(e).__name__}"] += 1
            rows.append({"prompt": name, "outcome": "schema_invalid", "detail": str(e)[:200]})
            continue
        msgs = [m for m in rec["messages"] if m["role"] != "assistant"]
        found = validate_capture(name, msgs, obj, allow_overlong_summaries=True)
        if found:
            s["semantic_defect"] += 1
            for d in found:
                defects[name][d.split(":")[0]] += 1
        else:
            s["clean"] += 1
        rows.append({"prompt": name, "outcome": "clean" if not found else "semantic_defect",
                     "defects": found,
                     "exact_match": txt.strip() == rec["messages"][-1]["content"].strip()})

    print(f"\n{'='*78}\nHELD-OUT RESULT ({args.model})\n{'='*78}")
    print(f"{'prompt':42} {'n':>4} {'clean':>6} {'defect':>7} {'bad json':>9} {'err':>4} {'p50 s':>6}")
    for name in sorted(stats):
        s = stats[name]
        badjson = s["unparseable"] + s["schema_invalid"]
        p50 = sorted(lat[name])[len(lat[name]) // 2] if lat[name] else 0
        print(f"{name:42} {s['n']:>4} {s['clean']:>6} {s['semantic_defect']:>7} "
              f"{badjson:>9} {s['call_error']:>4} {p50:>6.1f}")
    tot = sum(s["n"] for s in stats.values())
    clean = sum(s["clean"] for s in stats.values())
    print(f"\noverall clean: {clean}/{tot} = {100*clean/max(tot,1):.1f}%")
    em = sum(1 for r in rows if r.get("exact_match"))
    print(f"byte-exact match with reference target: {em}/{tot}")
    for name in sorted(defects):
        if defects[name]:
            print(f"\n  {name} defects: {dict(defects[name])}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2)[:2_000_000])
        print(f"\nper-record detail -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
