"""Build DPO preference pairs from the student's own extract_edges replies. PAID (judge).

    uv run --extra dev python -m scripts.distill.dpo_pairs \\
        --capture RUN/capture-v2.jsonl --out DIR [--budget 4] [--val-fraction 0.1]

Input is a `haiku_extract_ab --capture` file of the student (cheap tier, served model
`qwen35-graphrag`). Every terminal, parseable `extract_edges.edge` reply is judged TWICE
(the judge tier from `_eval_judge_client_and_model`, facts shuffled differently per pass)
against the chunk it came from: stated / previous_only / unsupported, as the teacher
judge does. The judge's run-to-run noise is large (identical facts scored 20 unsupported
one run and 36 the next, 2026-10-21), so a fact counts as unsupported only when BOTH
passes say so.

A pair is built for each reply with at least one such fact and at least one fact left:
`rejected` is the reply as the student wrote it, `chosen` is the same reply with only
those edges removed -- on-policy, differing in nothing else, so DPO moves probability
away from the invented facts without teaching the model to write fewer of the others.
(A reply whose every fact is unsupported gives no pair: `{"edges": []}` as the preferred
answer would teach emptiness.) Both are serialised as the SFT targets were
(`json.dumps(..., ensure_ascii=False)`).

Writes, under --out: `judged-1.jsonl`, `judged-2.jsonl` (resumable, keyed on the capture
`call_id`), `train.jsonl` / `val.jsonl` (split by article group, so no article is in
both), and `summary.json`. Distillation data: keep --out outside the public repo.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

PROMPT = "extract_edges.edge"
STUDENT_MODEL = "qwen35-graphrag"
TERMINAL = {"stop"}
VERDICTS = {"stated", "previous_only", "unsupported"}
CATEGORIES = {"over_inference", "subject_substitution", "taxonomy_as_fact",
              "wrong_detail", "other"}

INSTRUCTIONS = """You are auditing knowledge-graph facts extracted from ONE chunk of a vendor
documentation page. CURRENT MESSAGE is the chunk. PREVIOUS MESSAGES are earlier chunks of
the same page, given only as context. DOCUMENT CONTEXT names the documented product.

For EVERY numbered fact below, in order, decide:
- "stated": the CURRENT MESSAGE states it, or implies it unambiguously (resolving a
  reference with PREVIOUS MESSAGES or DOCUMENT CONTEXT is fine).
- "previous_only": the claim is supported only by the PREVIOUS MESSAGES, not the CURRENT MESSAGE.
- "unsupported": neither supports it. Be strict: plausible general knowledge the text does
  not state is unsupported.
For an unsupported fact give a category:
- "over_inference": a claim the text does not make (generalisation, assumed capability, cause).
- "subject_substitution": a claim the text makes about something else, attributed to the wrong
  product, component or entity.
- "taxonomy_as_fact": a list/menu/heading membership turned into a relationship the text does not assert.
- "wrong_detail": right subject and gist, but a wrong number, version, condition or limit.
- "other".

Return only JSON: {"labels": [{"n": <fact number>, "verdict": "...", "category": "..." or null,
"why": "<= 15 words"}]} with exactly one entry per fact, in order.
"""


def tag(text: str, name: str) -> str:
    m = re.search(rf"<{name}>\n?(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else ""


CONTEXT_LINE = re.compile(r"DOCUMENT CONTEXT: This chunk comes from [^\n]*")


def document_context(messages: list[dict]) -> str:
    """The DOCUMENT CONTEXT line (graphiti_client.document_context), wherever cache
    layout left it."""
    for m in messages:
        hit = CONTEXT_LINE.search(m["content"])
        if hit:
            return hit.group(0)
    return ""


def group_of(user: str, context: str) -> str:
    """The article a chunk belongs to, as far as the prompt shows it: the title (first
    breadcrumb segment of the current chunk) plus the documented product. Used only to
    keep one article's chunks on one side of the train/val split."""
    cur = tag(user, "CURRENT_MESSAGE")
    m = re.search(r"\[([^\]›]*)", cur)
    return f"{context}|{(m.group(1) if m else cur[:80]).strip()}"


def edge_records(lines: list[str]) -> tuple[list[dict], Counter]:
    """The judgeable student edge replies, and why the others were skipped."""
    out: list[dict] = []
    skipped: Counter = Counter()
    for line in lines:
        r = json.loads(line)
        if r.get("prompt_name") != PROMPT or r.get("tier") != "cheap":
            continue
        if r.get("model") != STUDENT_MODEL:
            skipped["not_student"] += 1
            continue
        msgs = r.get("messages") or []
        if [m.get("role") for m in msgs] != ["system", "user"]:
            skipped["unexpected_messages"] += 1
            continue
        if r.get("finish_reason") not in TERMINAL:
            skipped["nonterminal"] += 1
            continue
        try:
            payload = json.loads(r.get("completion") or "")
        except json.JSONDecodeError:
            skipped["unparseable"] += 1
            continue
        edges = payload.get("edges") if isinstance(payload, dict) else None
        if not isinstance(edges, list) or not all(isinstance(e, dict) for e in edges):
            skipped["no_edge_list"] += 1
            continue
        if not edges:
            skipped["no_edges"] += 1
            continue
        messages = [{"role": m["role"], "content": str(m["content"])} for m in msgs]
        ctx = document_context(messages)
        out.append({"id": r["call_id"], "messages": messages, "payload": payload,
                    "response_format": r.get("response_format"), "context": ctx,
                    "group": group_of(messages[1]["content"], ctx)})
    return out, skipped


def fact_line(e: dict) -> str:
    return f"{e.get('source_entity_name')} -> {e.get('target_entity_name')}: {e.get('fact')}"


def judge_prompt(rec: dict, pass_no: int) -> tuple[str, list[int]]:
    """The prompt for one pass, and `order[n-1]` = the edge index fact n shows."""
    edges = rec["payload"]["edges"]
    order = list(range(len(edges)))
    random.Random(f"{rec['id']}:{pass_no}").shuffle(order)
    u = rec["messages"][1]["content"]
    lines = [f"{n}. {fact_line(edges[i])}" for n, i in enumerate(order, 1)]
    prompt = "\n\n".join([INSTRUCTIONS, "PREVIOUS MESSAGES:\n" + tag(u, "PREVIOUS_MESSAGES"),
                          "CURRENT MESSAGE:\n" + tag(u, "CURRENT_MESSAGE"), rec["context"],
                          "FACTS:\n" + "\n".join(lines)])
    return prompt, order


def labels_by_edge(obj: Any, order: list[int]) -> list[dict] | None:
    """The judge's labels re-indexed by edge, or None when the reply is unusable."""
    labels = obj.get("labels") if isinstance(obj, dict) else None
    if not isinstance(labels, list) or len(labels) != len(order):
        return None
    if not all(isinstance(e, dict) and e.get("verdict") in VERDICTS
               and (e["verdict"] != "unsupported" or e.get("category") in CATEGORIES)
               for e in labels):
        return None
    out: list[dict] = [{} for _ in order]
    for lab, i in zip(labels, order):
        out[i] = {"verdict": lab["verdict"], "category": lab.get("category"),
                  "why": lab.get("why")}
    return out


def unsupported_both(a: list[dict], b: list[dict]) -> list[int]:
    return [i for i, (x, y) in enumerate(zip(a, b))
            if x["verdict"] == "unsupported" and y["verdict"] == "unsupported"]


def serialise(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


def build_pair(rec: dict, bad: list[int], a: list[dict], b: list[dict]) -> dict | None:
    """The DPO record, or None when no fact is bad or none would be left."""
    edges = rec["payload"]["edges"]
    if not bad or len(bad) == len(edges):
        return None
    chosen = {**rec["payload"], "edges": [e for i, e in enumerate(edges) if i not in bad]}
    return {"prompt_name": PROMPT, "response_format": rec["response_format"],
            "messages": rec["messages"], "chosen": serialise(chosen),
            "rejected": serialise(rec["payload"]),
            "meta": {"call_id": rec["id"], "group": rec["group"], "n_edges": len(edges),
                     "removed": [{"fact": fact_line(edges[i]),
                                  "categories": [a[i]["category"], b[i]["category"]],
                                  "why": [a[i]["why"], b[i]["why"]]} for i in bad]}}


def split_of(group: str, val_fraction: float) -> str:
    h = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "train"


async def judge_pass(recs: list[dict], pass_no: int, out: Path, budget: float,
                     spend: dict, concurrency: int) -> dict[str, list[dict]]:
    from answer_api.eval_router import _eval_judge_client_and_model
    from graph_extract.config import get_extract_settings
    from graph_extract.usage import usable_content
    from scripts.haiku_extract_ab import openrouter_rates

    path = out / f"judged-{pass_no}.jsonl"
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r["labels"]
    todo = [r for r in recs if r["id"] not in done]
    client, model = _eval_judge_client_and_model(get_extract_settings())
    rates = openrouter_rates(model)
    print(f"pass {pass_no}: judge {model}; {len(done)} done, {len(todo)} to judge", flush=True)
    sem = asyncio.Semaphore(concurrency)
    failed: list[str] = []

    async def one(rec: dict) -> list[dict] | None:
        prompt, order = judge_prompt(rec, pass_no)
        async with sem:
            for _ in range(2):
                resp = await client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}, temperature=0,
                    max_tokens=16000)
                u = resp.usage
                if u:
                    spend["usd"] += (u.prompt_tokens * rates[0]
                                     + u.completion_tokens * rates[2]) / 1e6
                text = usable_content(resp)
                try:
                    obj = json.loads(text) if text else None
                except json.JSONDecodeError:
                    obj = None
                labels = labels_by_edge(obj, order) if obj is not None else None
                if labels is not None:
                    return labels
        return None

    try:
        with path.open("a") as fh:
            step = concurrency * 4
            for start in range(0, len(todo), step):
                if spend["usd"] > budget:
                    print(f"BUDGET ${budget} reached at ${spend['usd']:.3f}; stopping",
                          flush=True)
                    break
                batch = todo[start:start + step]
                for rec, labels in zip(batch, await asyncio.gather(*map(one, batch))):
                    if labels is None:
                        failed.append(rec["id"])
                        continue
                    done[rec["id"]] = labels
                    fh.write(json.dumps({"id": rec["id"], "labels": labels}) + "\n")
                fh.flush()
                print(f"  pass {pass_no}: {start + len(batch)}/{len(todo)} "
                      f"${spend['usd']:.3f} failed {len(failed)}", flush=True)
    finally:
        await client.close()
    return done


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budget", type=float, default=4.0, help="judge USD, both passes")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N records only (smoke)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lines = [ln for p in a.capture for ln in p.read_text().splitlines() if ln.strip()]
    recs, skipped = edge_records(lines)
    if a.limit:
        recs = recs[: a.limit]
    print(f"{len(recs)} student edge replies; skipped {dict(skipped)}", flush=True)
    if not recs:
        raise SystemExit("no judgeable student edge replies in the capture")
    no_ctx = sum(not r["context"] for r in recs)
    if no_ctx:
        print(f"WARNING: {no_ctx} replies have no DOCUMENT CONTEXT in their prompt", flush=True)
    spend = {"usd": 0.0}
    p1 = await judge_pass(recs, 1, a.out, a.budget, spend, a.concurrency)
    p2 = await judge_pass(recs, 2, a.out, a.budget, spend, a.concurrency)
    both = [r for r in recs if r["id"] in p1 and r["id"] in p2]
    stats: Counter = Counter()
    cats: Counter = Counter()
    pairs = {"train": [], "val": []}
    for r in both:
        l1, l2 = p1[r["id"]], p2[r["id"]]
        bad = unsupported_both(l1, l2)
        stats["edges"] += len(l1)
        stats["unsupported_pass1"] += sum(x["verdict"] == "unsupported" for x in l1)
        stats["unsupported_pass2"] += sum(x["verdict"] == "unsupported" for x in l2)
        stats["unsupported_both"] += len(bad)
        stats["previous_only_both"] += sum(x["verdict"] == y["verdict"] == "previous_only"
                                           for x, y in zip(l1, l2))
        cats.update(l1[i]["category"] for i in bad)
        pair = build_pair(r, bad, l1, l2)
        if pair is not None:
            pairs[split_of(r["group"], a.val_fraction)].append(pair)
        elif bad:
            stats["no_pair_all_unsupported"] += 1
    for split, rows in pairs.items():
        with (a.out / f"{split}.jsonl").open("w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {"replies": len(recs), "judged_both": len(both), "skipped": dict(skipped),
               "judge_usd_this_run": round(spend["usd"], 4), **stats,
               "categories_both": dict(cats),
               "pairs": {k: len(v) for k, v in pairs.items()},
               "groups": {k: len({p["meta"]["group"] for p in v}) for k, v in pairs.items()}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
