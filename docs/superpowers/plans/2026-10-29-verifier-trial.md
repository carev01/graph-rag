# Verifier Trial Implementation Plan

> **For agentic workers:** executed natively (superpowers:executing-plans), TDD per task, ledger at `.superpowers/sdd/2026-10-29-verifier-trial/progress.md`.

**Goal:** Serve ckpt100 and verifier1 as two LoRA adapters on one Qwen3.5-4B base, add a flag-gated `extract_edges` verifier hook, and run the end-to-end trial `v2a` (ckpt100 adapter) vs `v2av` (+ verifier, policy P2).

**Spec:** `docs/superpowers/specs/2026-10-29-verifier-trial-design.md`

## Global Constraints

- The hook is default OFF (`EDGE_VERIFIER_ENABLED=false`) and fails open on any error, timeout, empty or unparseable reply. Never write `content or ""`.
- The verifier prompt is byte-identical to the training prompt (`scripts/distill/verifier_data.verifier_user` + `VERIFIER_INSTRUCTIONS`).
- `src/` never imports `scripts/`. The shared pieces move into `src/graph_extract/fact_verifier.py`, and the scripts import them.
- The verifier is installed on the CHEAP tier only. Strong-tier (gpt-5-mini) dense-table replies are not verified.
- Tokens are passed only via stdin or env. Paid steps only as authorised in the spec (smoke about $1, trial about $3-5).
- CI gate: `uv run ruff check src tests`, `uv run mypy src`, `uv run --extra dev pytest -m "not live"`.

## Review Focus

1. A verifier reply that parses but numbers facts differently from the edges sent must keep every edge (fail open), never misalign drops.
2. An `extract_edges` reply that is not a dict with an `edges` list (graphiti contract drift) must pass through untouched, counted as a failed-open call.
3. The hook must see the prompt as authored. graphiti's clients append schema text to the messages in place, so the verifier is built from a pristine copy.
4. With the flag off, the dedup guard must make zero verifier calls, and its behaviour must be byte-identical to today's.
5. The trial's engagement gate must refuse to run `v2av` when the verifier never fired or never dropped anything, and refuse `v2a` when it fired at all.

### Task 1: `src/graph_extract/fact_verifier.py` (prompt, parser, P2 filter, client)

- **Produces:**
  - `VERIFIER_INSTRUCTIONS`;
  - `verifier_user(messages: list[dict], facts: list[str]) -> str`;
  - `fact_line(edge) -> str`;
  - `parse_reply(text, n) -> list[dict] | None`;
  - `keep_edges(parsed, edges) -> tuple[list, list]` (P2: drop `unsupported` only; `None` keeps all);
  - `class EdgeVerifier(client, model, timeout)` with `async check(messages, response, stats) -> response`.
- **Scripts:**
  - `verifier_data` takes `VERIFIER_INSTRUCTIONS`/`fact_line` from it, and its `verifier_user(rec, facts)` delegates.
  - `verifier_eval.parse_reply` is re-exported from it.
- **Tests (`tests/unit/test_fact_verifier.py`):**
  - `VERIFIER_INSTRUCTIONS` equals the scripts' original construction.
  - The prompt is byte-identical to a verifier1 test record's user message, rebuilt from its capture record.
  - `keep_edges` cases: drop only unsupported; `None` keeps all.
  - `check()`:
    - counts `verifier_calls`/`verifier_dropped`;
    - fails open on exception, empty content, wrong count, or a non-dict reply, counting `verifier_failed_open`;
    - sends `temperature=0` and `max_tokens=64+160n`;
    - uses pristine messages.

### Task 2: settings, dedup_guard hook, CLI wiring

- **`ExtractSettings`:** `edge_verifier_enabled=False`, `edge_verifier_base_url=""`, `edge_verifier_api_key=""`, `edge_verifier_model="verifier"`, `edge_verifier_timeout_seconds=120.0`.
- **`DedupIndexStats`:** gains `verifier_calls`, `verifier_dropped`, `verifier_failed_open`.
- **`install_dedup_guard(..., edge_verifier=None)`:** for `extract_edges.edge`, make a pristine copy, call `_primary`, then `edge_verifier.check(...)`.
- **`cli._build_ingest_driver`:** builds the `EdgeVerifier` for the cheap guard only, when enabled. It raises at startup if enabled without a base URL or key.
- **Tests:**
  - Flag off: no call, response untouched.
  - Flag on: edges dropped before graphiti sees them, counters right.
  - A raising verifier fails open.
  - The strong guard never gets a verifier.
  - Startup validation.

### Task 3: trial harness arms `v2a` / `v2av`

- **`arm_settings`:** `v2a` = the `v2` settings with `cheap_llm_model="extractor"`; `v2av` = `v2a` + `edge_verifier_*` (same `V2_BASE_URL`/`V2_API_KEY`, model `verifier`).
- **Gate:**
  - vLLM's success counter must rise by the cheap calls + `verifier_calls`.
  - `v2av` needs `verifier_calls` = the cheap `extract_edges.edge` calls (counted by the guard timings), `verifier_dropped > 0` and `verifier_failed_open == 0`.
  - `v2a` needs `verifier_calls == 0`.
  - The gate predicate is a pure function, unit-tested.
- **Summary:** totals of `verifier_calls`/`dropped`/`failed_open` per arm.

### Task 4: LoRA serving + smoke checker

- **`scripts/vast/remote_setup.sh`:** `LORA_ADAPTERS="name=repo@rev ..."`. With the base `MODEL_REPO=Qwen/Qwen3.5-4B`, it downloads each adapter (HF token from stdin) to `/workspace/hf/lora-<name>`, rewrites the served copy's regex `target_modules` to the seven-name list, and adds `--enable-lora --max-lora-rank 16 --max-loras 2 --lora-modules name=dir ...`.
- **`scripts/distill/lora_smoke.py`** (pure comparison functions, unit-tested; the HTTP calls are thin):
  - `extractor_equivalence(captured, regenerated)`: >= 45/50 byte-identical, all parse;
  - `verifier_agreement(test_records, notebook_preds, served)`: >= 95% of facts;
  - throughput: extractor adapter vs the base model name.
- **Rental script (scratchpad):**
  1. rent;
  2. provision in LoRA mode;
  3. smoke (stop on fail);
  4. trial gate;
  5. full trial `--arms v2a,v2av --judge v2a,v2av`;
  6. second judge pass (`--arms none --reuse`);
  7. stop the instance on every path.

### Operational

O1: CI gate, gitleaks, push. O2: smoke + trial on one rental (authorised). O3: report against the adoption rule and STOP for the user; destroy the instance after their decision.
