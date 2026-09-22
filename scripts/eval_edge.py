"""Every held-out extract_edges.edge record, to tighten a 2/15 estimate.

118 is the ceiling: the val split is 17 articles and that is all the edge calls
they produced. Reported split by whether the prompt fits the 12,288-token cap the
model was trained under -- the 12 that exceed it are still inside the server's
32,768 context, but they are out of the training distribution and should not be
pooled with the rest without saying so.
"""
from __future__ import annotations

import collections
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.build_extraction_dataset import RESPONSE_MODELS, validate_capture  # noqa: E402

B = "http://srv-llm.home.lan:8080/v1/chat/completions"
NAME = "extract_edges.edge"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> int:
    with open("data/ft-captured/val.jsonl") as fh:
        recs = [json.loads(line) for line in fh]
    edge = [r for r in recs if r["prompt_name"] == NAME]
    def tok(r):
        return sum(len(m["content"]) for m in r["messages"]
                   if m["role"] != "assistant") // 3
    print(f"{len(edge)} held-out {NAME} records", flush=True)
    buckets = {"in_dist": [], "over_cap": []}
    for i, rec in enumerate(edge, 1):
        msgs = [m for m in rec["messages"] if m["role"] != "assistant"]
        body = {"model": "qwen35-graphrag", "messages": msgs, "temperature": 0,
                "top_p": 1.0, "presence_penalty": 0, "max_tokens": 4096}
        t = time.time()
        try:
            out = json.load(urllib.request.urlopen(urllib.request.Request(
                B, json.dumps(body).encode(), {"Content-Type": "application/json"}),
                timeout=900))
            txt = out["choices"][0]["message"]["content"]
            fin = out["choices"][0].get("finish_reason")
        except Exception as e:  # noqa: BLE001
            txt, fin = None, f"{type(e).__name__}"
        dt = time.time() - t
        b = "in_dist" if tok(rec) <= 12288 else "over_cap"
        if txt is None:
            buckets[b].append(("call_error", [fin], dt))
            continue
        try:
            obj = RESPONSE_MODELS[NAME](**json.loads(txt))
        except Exception as e:  # noqa: BLE001
            buckets[b].append(("bad_json", [type(e).__name__], dt))
            continue
        found = validate_capture(NAME, msgs, obj, allow_overlong_summaries=True)
        buckets[b].append(("defect" if found else "clean", found, dt))
        if i % 20 == 0:
            print(f"  [{i}/{len(edge)}]", flush=True)

    print(f"\n{'='*70}\nextract_edges.edge, all held-out records\n{'='*70}")
    for b, rows in buckets.items():
        if not rows:
            continue
        n = len(rows)
        bad = sum(1 for o, _, _ in rows if o != "clean")
        lo, hi = wilson(bad, n)
        p50 = sorted(d for _, _, d in rows)[n // 2]
        print(f"\n  {b}: n={n}  defects={bad} ({100*bad/n:.1f}%)  "
              f"Wilson 95% CI [{100*lo:.1f}%, {100*hi:.1f}%]  p50 {p50:.1f}s")
        cls = collections.Counter(d.split(':')[0] for _, ds, _ in rows for d in (ds or []))
        if cls:
            print(f"      defect classes: {dict(cls)}")
    allrows = buckets["in_dist"] + buckets["over_cap"]
    n = len(allrows)
    bad = sum(1 for o, _, _ in allrows if o != "clean")
    lo, hi = wilson(bad, n)
    print(f"\n  POOLED: n={n} defects={bad} ({100*bad/n:.1f}%) "
          f"CI [{100*lo:.1f}%, {100*hi:.1f}%]")
    print("  solar-pro4 on the same checker: 126/613 = 20.6%")
    print(f"  -> {'SEPARATED' if hi < 0.206 else 'NOT separated'} from solar-pro4 at 95%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
