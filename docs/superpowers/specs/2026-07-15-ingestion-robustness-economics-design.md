# Ingestion Robustness & Economics — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Pipeline hardening — Sub-slice B (robustness & economics). Makes the incremental pipeline (Sub-slice A) safe to run at corpus scale.
**Date:** 2026-07-15
**Status:** Approved design — ready for implementation planning

Builds on Sub-slice A ([`2026-07-15-incremental-semantic-ingestion-design.md`](2026-07-15-incremental-semantic-ingestion-design.md)). Closes the two prerequisites A's final review flagged: the **bootstrap-flood blocker** (the enqueue trigger fires during bootstrap, so a full ~105k-article bootstrap would flood `semantic_jobs` unthrottled) and the **orphaned-job gap** (a worker crash between claim and complete leaves a job `in_progress` forever, with no reaper). Adds priority lanes, a token budget, retry/backoff, dead-lettering, and an in-progress reaper. Maintenance jobs (staleness sweep, `SAME_AS` reconciliation) remain **Sub-slice C**.

---

## 1. Problem & scope

Sub-slice A delivered a working incremental pipeline (Postgres `semantic_jobs` queue, sync→queue trigger, standalone worker, temporal update policy). But it is not yet safe to operate at scale:
- A full bootstrap enqueues one job per article (~105k) with no rate/cost control — the token spend is unbounded and would starve live incremental updates behind a huge backlog.
- A crashed worker orphans its claimed jobs (`in_progress` never reclaimed); a one-time update whose article never changes again is silently dropped.
- A poison job is marked `failed` once and never retried (transient failures — a timeout, a rate-limit — permanently lose the update).

**In scope (B):** priority lanes (incremental preempts bootstrap), a daily token budget gating bootstrap work, retry with exponential backoff, dead-lettering after max attempts, and an in-progress reaper. All are changes to `semantic_jobs`/`StateStore`, the enqueue call, and the worker.
**Out of scope:** the residual-staleness sweep and structural↔semantic `SAME_AS` reconciliation (Sub-slice C); the answer-api (Phase 2).

## 2. Budget policy (decided)

**Incremental is unbounded; bootstrap is capped by a configurable daily token budget.** Live updates must stay fresh, so incremental-lane jobs are never budget-gated. Bootstrap-lane jobs are backfill and yield to the cap: once the day's recorded token spend reaches `daily_token_budget`, the worker stops claiming bootstrap jobs (but keeps draining incremental) until the next day. The cap is a config value (`SEMANTIC_DAILY_TOKEN_BUDGET`) with a sensible default, tunable as a business input.

## 3. Schema changes — `semantic_jobs` + `token_ledger`

Additive migration in `StateStore.init_schema` (the table is new from A, low-risk): `ALTER TABLE semantic_jobs ADD COLUMN IF NOT EXISTS …`.

`semantic_jobs` gains:
| column | type | notes |
|---|---|---|
| `lane` | text not null default `'incremental'` | `'incremental'` \| `'bootstrap'` |
| `next_attempt_at` | timestamptz not null default now() | claim gate for backoff |

New status value `'dead'` (dead-lettered; no schema change, `status` is free text). New table:
```sql
CREATE TABLE IF NOT EXISTS token_ledger (day date PRIMARY KEY, tokens bigint NOT NULL DEFAULT 0);
```

The partial-unique `ux_semantic_jobs_pending` (pending-only) from A is unchanged — retry keeps a job in `pending`, so a new delta for the same article still collapses onto it (latest wins), which is correct.

## 4. Priority lanes

- **Enqueue with a lane.** `enqueue_semantic_job(article_id, op, content_hash, lane)` — `sync_core._apply_record` infers the lane from the `res` argument it already holds: `BootstrapResult → 'bootstrap'`, `IncrementalResult → 'incremental'`. (Tombstones only occur on incremental streams → `'incremental'`.) On the idempotent `ON CONFLICT` update, an incremental enqueue **upgrades** a pending bootstrap job to `'incremental'` (a live change to an article still being backfilled should jump the lane); a bootstrap enqueue never downgrades an incremental one.
- **Claim orders by lane then age:** `ORDER BY (lane='bootstrap'), next_attempt_at` (incremental sorts first because the boolean is false=0). So incremental always drains ahead of bootstrap.

## 5. Token budget gate

- **Accounting:** the worker wraps each `ingest_article` with a read of `usage.get_tally()` before/after and records the delta into `token_ledger` for `current_date` via an upsert-increment (`INSERT … ON CONFLICT (day) DO UPDATE SET tokens = token_ledger.tokens + $delta`). `tombstone_article_episodes` consumes no LLM tokens (Cypher only) → records nothing. Only the extraction LLM is metered (embeddings are local/untracked, consistent with `cost_report`).
- **Gate:** before a claim, the worker reads `today_tokens = SELECT tokens FROM token_ledger WHERE day=current_date`. If `today_tokens >= daily_token_budget`, it claims with a **lane filter excluding bootstrap** (incremental only); otherwise it claims across both lanes. Implemented as a `claim_semantic_jobs(batch, include_bootstrap: bool)` — the worker computes `include_bootstrap = today_tokens < budget`.
- **Note (honest):** the gate is checked per claim-batch, not per token — a batch already claimed will finish even if it tips over budget. This is a soft cap (bounded overshoot of one batch), sufficient for backfill pacing; a hard per-token cap is unnecessary.

## 6. Retry, backoff, dead-letter

Replaces A's terminal `fail_semantic_job` with a retry-aware transition:
- On job exception, the worker calls `fail_semantic_job(job_id, error, attempts, max_attempts)` (or the store computes it): if `attempts+1 < max_attempts` → `status='pending'`, `attempts=attempts+1`, `last_error=$e`, `next_attempt_at = now() + exp_backoff(attempts)`; else → `status='dead'`, `attempts+1`, `last_error`.
- **Backoff:** `exp_backoff(n) = min(base * 2^n, cap)` (e.g. base 30s, cap 1h) — config values.
- **Claim gate:** only `status='pending' AND next_attempt_at <= now()` is claimable, so a backed-off job is invisible until its time.
- **Dead-letter:** `status='dead'` retains `last_error`/`attempts` in place (queryable for ops); no separate table. A `dead_semantic_job_count()` helper surfaces the count for monitoring.

## 7. In-progress reaper

A job `in_progress` with `claimed_at < now() - lease` means the worker that claimed it died. `reap_stale_jobs(lease_seconds) -> int` resets such jobs to `status='pending'`, `attempts=attempts+1`, `next_attempt_at=now()` (so they're immediately re-claimable, subject to backoff on the next failure), returns the count. The worker calls it at the top of each `run_worker_once` (cheap; a single UPDATE). `lease` is a config value (default e.g. 30 min) — must exceed the longest expected single-article ingest.

## 8. Worker changes

`run_worker_once(store, ingest, batch, budget)`:
1. `await store.reap_stale_jobs(lease)`.
2. `today = await store.today_token_total()`; `include_bootstrap = today < budget`.
3. `jobs = await store.claim_semantic_jobs(batch, include_bootstrap)`.
4. Per job: read tally, dispatch (`upsert`→`ingest_article`, `remove`→`tombstone_article_episodes`), record token delta, `complete`; on exception → the retry-aware fail.
5. Return count.

The `worker` CLI passes `daily_token_budget`, `max_attempts`, `backoff_base/cap`, and `lease` from config (typer options / `Settings`). Existing shutdown/resource-cleanup behavior unchanged.

## 9. Testing

Postgres testcontainer (all queue logic is DB; ingest is a stub recording tokens):
- **Lanes:** enqueue a bootstrap + an incremental job; claim returns incremental first. An incremental enqueue upgrades a pending bootstrap job's lane; a bootstrap enqueue doesn't downgrade.
- **Budget:** with `today_tokens >= budget`, a claim returns only incremental jobs (bootstrap withheld); under budget, both lanes claim. Token delta from a stub-ingest is recorded to `token_ledger`.
- **Retry/backoff:** a failing job goes back to `pending` with `next_attempt_at` in the future (not immediately re-claimable), `attempts` incremented; after `max_attempts` it becomes `dead` (not re-claimable); `dead_semantic_job_count` reflects it.
- **Reaper:** a job manually set `in_progress` with an old `claimed_at` is reset to `pending` by `reap_stale_jobs`; a fresh `in_progress` job is left alone.
- **Worker loop:** `run_worker_once` reaps, gates on budget, records tokens, and retries — one integration test wiring a stub ingest that fails once then succeeds.

## 10. Acceptance criteria

1. Bootstrap-lane jobs never starve incremental: a claim always returns available incremental jobs before any bootstrap job.
2. When the day's recorded tokens reach `daily_token_budget`, bootstrap jobs stop being claimed while incremental jobs continue.
3. A transient job failure retries with increasing backoff and succeeds on a later attempt; after `max_attempts` it is dead-lettered (`status='dead'`, error retained), not retried forever.
4. A job orphaned `in_progress` past the lease is reclaimed and reprocessed.
5. Per-job extraction tokens are recorded to `token_ledger`; `tombstone` jobs record none.
6. Unit + integration tests green (Postgres testcontainer); ruff/mypy clean. No change to `graph_extract`'s public behavior beyond what the worker consumes.

## 11. Deferred (Sub-slice C and beyond)

- Residual-staleness sweep (Cypher expiry of facts whose supporting episodes are all superseded/removed — must handle A's flagged "byte-identical shrink-then-restore leaves an active episode flagged superseded" edge case).
- Structural↔semantic `SAME_AS` reconciliation.
- The `graph_sync ↔ graph_extract` package-cycle cleanup (move `delta_client.make_client` to a neutral module) — a standing follow-up, orthogonal to B.
- A monitoring dashboard over `dead`/lane/budget metrics (ops; the count helpers ship here, the dashboard doesn't).
