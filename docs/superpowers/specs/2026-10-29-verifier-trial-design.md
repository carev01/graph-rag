# ckpt100 + fact verifier on one GPU, end-to-end trial -- design

Status: draft for review, 2026-10-29. Follows `2026-10-27-fact-verifier-design.md`. The
Qwen 4B verifier missed that spec's offline bar, but verifier1 (ckpt200, policy P2)
removes about 20 unsupported facts per 1,000 for about 12 supported ones. The user asked
whether that is better than no verifier; this spec answers it end to end.

## 1. Question and decision rule

Does ckpt100 + verifier1 (P2) produce better graph facts than ckpt100 alone? It is
adopted when, on the same articles, with two judge passes summed:

- unsupported facts are **lower** than ckpt100 alone, and
- the distinct-fact share against ckpt100 alone is **>= 0.95**.

This is the adoption rule ckpt100 itself passed (2026-10-21). Anything else: the verifier
is not adopted, and ckpt100 stays as it is.

## 2. Serving: one base, two adapters

Both adapters are LoRA r=16 on `Qwen/Qwen3.5-4B`. ckpt100 is the
`opus-distill-v2/checkpoints/checkpoint-100` adapter that `export_checkpoint.ipynb`
merged. The user uploads it to the private repo `carev01/qwen35-4b-graphrag-verifier`,
revision `ckpt100-adapter`.

vLLM serves the bf16 base once with `--enable-lora --max-lora-rank 16` and two LoRA
modules, `extractor` (ckpt100) and `verifier` (verifier1-ckpt200). A request picks one by
model name.

Both adapter configs store `target_modules` as an Unsloth regex string. If vLLM rejects
it, the served copy's `adapter_config.json` gets the equivalent explicit list (the seven
projection names, language model only). That is a metadata-only change; the weights stay
untouched.

## 3. Smoke rental (about $1), before any trial spend

One RTX 4090, the same template and provision path as the capture. Hard checks; any
failure stops the run, and the rental is stopped:

1. **vLLM loads both adapters.** `/v1/models` lists `extractor` and `verifier`.
2. **The extractor adapter matches merged ckpt100.** Take 50 captured ckpt100
   `extract_edges.edge` prompts (`2026-10-24-dpo4/capture-run/capture-v2.jsonl`, which
   was served by the merged model), regenerate them greedily through `extractor`, and
   compare.
   - Passes when at least 45/50 replies are byte-identical and every reply parses.
   - The differing replies are reported with their edge-set overlap.
3. **The verifier adapter matches the notebook.** For 50 verifier1 test prompts, the
   served replies' verdicts agree with `test_predictions.jsonl` on at least 95% of facts.
   This catches any drift between vLLM and HF generation.
4. **Throughput.** Extractor tokens per second at concurrency 8, with and without
   adapters, against the merged model on the same box. Reported, not gated.

## 4. The verifier hook (production code, behind a flag)

- **`src/graph_extract/fact_verifier.py`:**
  - The verifier prompt: `VERIFIER_INSTRUCTIONS` plus the user text built from the
    `extract_edges.edge` call's messages, byte-identical to
    `scripts/distill/verifier_data.verifier_user`.
  - `parse_reply`, shared with `verifier_eval` (moved into `src`, imported by the
    script).
  - `keep_edges(reply, edges) -> (kept, dropped)` under P2.
- **`dedup_guard`:** on every `extract_edges.edge` reply, when
  `EDGE_VERIFIER_ENABLED=true`, call the verifier (`EDGE_VERIFIER_BASE_URL`,
  `EDGE_VERIFIER_MODEL=verifier`, `EDGE_VERIFIER_API_KEY`) and drop the edges it labels
  `unsupported` before graphiti sees them.
  - **Fails open:** a timeout, an error or an unparseable reply keeps every edge.
  - **Counters** on the worker batch line: `verifier_calls`, `verifier_dropped`,
    `verifier_failed_open`.
  - **Default off.**
- Never `content or ""`: an empty verifier reply is a failure, which keeps all edges and
  is counted.

## 5. The trial (about $3-5, authorised with this spec)

`haiku_extract_ab.py`, two new arms on the same 60 articles as the 2026-10-21 adoption
trial (`2026-10-24-dpo4/trial/d6_sample.json`), both on the smoke instance:

- **`v2a`:** the cheap tier on `extractor` (ckpt100 as an adapter), dense tables on
  gpt-5-mini as before.
- **`v2av`:** the same, plus `EDGE_VERIFIER_ENABLED=true`.

The engagement gate (CLAUDE.md cost rule) applies before the full run:

- `v2av` must show `verifier_calls` equal to its `extract_edges.edge` calls, with
  `verifier_dropped > 0` over the gate articles.
- `v2a` must show `verifier_calls == 0`.

The judge runs two passes on `v2a` against `v2av`. The rental is stopped at the end, and
destroyed after the user's decision.

## 6. Tests

- **`fact_verifier`:**
  - The prompt is byte-identical to `verifier_data` on a captured record.
  - `keep_edges` drops only `unsupported` and fails open on a bad, empty or wrong-count
    reply.
- **`dedup_guard`:**
  - Flag off: no verifier call.
  - Flag on: the verifier is called once per edge reply with a fake client, the dropped
    edges are gone from the reply graphiti parses, and the counters are right.
  - A raising client fails open.
- **Harness:**
  - `v2a`/`v2av` arm settings.
  - The engagement-gate predicate.
