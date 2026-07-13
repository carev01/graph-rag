# Slice 1 (Ingestion Foundation) — Deferred Follow-ups

These are known, non-blocking items from the whole-branch review and per-task
reviews. The merge-blocking findings were fixed in commit `27df18f`; everything
below was consciously deferred. Roughly priority-ordered.

## Quality gate
- **mypy `strict = true` is configured but does not pass.** ~14+ strict errors
  accumulated (asyncpg/neo4j `Any` returns, `Record | None` indexing like
  `(await r.single())["n"]`, untyped params, bare `dict`). No CI enforces it
  today, so the gate is effectively off. Decide: fix the handful of real ones
  (the `Record | None` indexings are latent None-index bugs, safe only because
  the queries always return one row) and relax to a passing baseline, or drop
  the strict claim. Make the gate honest.
- **ruff not clean** on `tests/unit/test_webhook.py` (E401/E702 — multi-import
  and semicolon style, verbatim from the plan's example code).
- Add CI that actually runs `ruff check`, `mypy`, and `pytest -m "not live"`.

## Correctness / robustness (non-blocking)
- **Malformed NDJSON line handling.** `parse_delta_line` has no `try/except`, so
  a bad line aborts the stream (cursor safely stays put — good) but is never
  recorded to `dead_letter` and the stream doesn't "continue" as spec §7 asks.
  `StateStore.record_dead_letter` is now unused in `sync_core` (the unknown-source
  path no longer dead-letters) — wire it here for genuinely unparseable input.
- **Missing "dirty flag" re-pass.** Spec §5.3/§7: a nudge arriving mid-sync should
  trigger one more pass; currently a locked-out trigger just no-ops. Extra latency
  is bounded by the 30-min poll, so acceptable, but it's a documented gap.
- **`app.build_lifespan` leaks on startup-init failure.** If `catalog.load()` /
  `init_schema()` raises before the lifespan `yield`, the cleanup never runs. The
  CLI path (`_build_sync_core`) got the equivalent fix; the app lifespan did not.
  Low impact (a boot failure usually terminates the process anyway).
- **`webhook.py` unguarded `json.loads`** on an already-authenticated body → 500
  instead of a clean 400. Post-auth only, so low risk.
- **`catalog.refresh()` unbounded fan-out.** ~79 concurrent `?product_id=` calls
  with no semaphore and fail-fast `gather`; add a bounded semaphore. Also add a
  guard/warn if any single product's source page hits the 200 cap (the same
  silent-drop failure mode the per-product enumeration exists to avoid).
- **`StateStore.close()` not idempotent** (doesn't null `_pool`/`_lock_conn`); only
  called once in practice.
- **`should_run_source` first-call race** (two concurrent first-time callers can
  both pass the `FOR UPDATE` on a non-existent row); mitigated by the advisory
  lock / in-process single-flight in the intended usage.

## Performance (later slices / scale)
- Per-record `get_content_hash` + `apply_structural` are separate autocommit
  sessions (N+1 round-trips per article). Fine for slice-1 scale; batch later.

## Tests
- **Integration tests are order-dependent** due to module-scoped testcontainer
  fixtures (`test_sync_core.py` resets via `delete_source_articles`). A random
  test-order plugin could expose this; consider per-test cleanup or function
  scope.
- `test_verify_signature_accepts_and_rejects` is sync but under an `asyncio`
  `pytestmark` (pytest warning). Drop the mark for that one test.

## Environment / ops
- **Success-criterion #4 (real webhook delivery) is unproven end-to-end.** The
  receiver logic is unit-tested and locally verified, but cluster→host delivery
  was never confirmed: the DocExtractor extraction-trigger endpoint wasn't found
  (`POST /api/extraction/runs` → 405) and cluster→host reachability/TLS is
  untested. Revisit with the correct trigger endpoint.
- `DOCEXT_VERIFY_TLS=false` default disables TLS verification against the
  cluster — acceptable for the documented homelab/internal-CA posture, but call
  it out and prefer trusting the internal CA where possible.
- If local Postgres `5432` is occupied, remap the compose host port (the repo
  default is now the standard `5432:5432`).
