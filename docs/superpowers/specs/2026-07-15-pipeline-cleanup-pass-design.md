# Pipeline Cleanup Pass — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Cleanup pass — clears the accumulated group-A Minors before Sub-slice C.
**Date:** 2026-07-15
**Status:** Approved design — ready for implementation planning

A focused hygiene slice: five independent fixes triaged from the merged slices' final reviews (see `.superpowers/sdd/progress.md`). Group **B** items (shrink-restore-superseded edge case; should_distinct silent-merge) are folded into **Sub-slice C**; the **Region-magnet** residual rides the next extraction change; group **D** stays deferred. No new subsystems.

---

## 1. Scope

Five fixes, each self-contained and independently testable:

1. **Claim fencing token** — prevent a reaped-and-reclaimed job from being resurrected by a slow original worker's late `complete`/`fail`.
2. **Dead-letter observability** — a `queue-status` CLI surfacing job counts by status (incl. `dead`).
3. **Migration ALTER-path test** — lock the existing-DB `ALTER … ADD COLUMN IF NOT EXISTS` branch that testcontainers never exercise (they always start fresh).
4. **Package-cycle refactor** — move the DocExtractor client builder to a neutral module so `graph_extract` no longer imports `graph_sync`.
5. **Test/doc polish** — assert error-code markers in the noise-exclusion test; clarify VMware Platform-vs-Workload docstring phrasing.

## 2. Fix 1 — claim fencing token (queue)

**Problem:** if an ingest legitimately runs longer than the reaper lease, the reaper resets the job to `pending` and another worker re-claims it; the original worker's late `complete`/`fail` (currently keyed only on `id`) then flips the re-claimed job's state — double-processing / double token-spend (idempotent, so cost-only, not corruption).

**Fix:** fence `complete`/`fail` on the claim identity. A claim stamps `claimed_at=now()`; a reap + re-claim produces a *new* `claimed_at`, so the stale worker's write no longer matches.
- `claim_semantic_jobs` RETURNING gains `claimed_at`.
- `complete_semantic_job(job_id, claimed_at)` and `fail_semantic_job(job_id, error, *, max_attempts, retry_delay_seconds, claimed_at)` add `AND claimed_at = $claimed_at` to their `WHERE`.
- The worker passes `job["claimed_at"]` to both. `reap_stale_jobs` is unchanged (it already re-pends; the new `claimed_at` on the subsequent claim is what fences the stale worker).

**Acceptance:** a job claimed (claimed_at=T1), then reaper-reset + re-claimed (claimed_at=T2), then `complete`/`fail`-ed by the *original* worker with `claimed_at=T1` → no-op (the T2 claim's state is preserved). The normal path (same claimed_at) still completes/fails.

## 3. Fix 2 — dead-letter observability

**Problem:** a job that exhausts `max_attempts` becomes `status='dead'` (a permanent gap in the semantic graph for that article) but nothing surfaces it.

**Fix:** `StateStore.job_status_counts() -> dict[str, int]` (`SELECT status, count(*) FROM semantic_jobs GROUP BY status`) and a `queue-status` typer command in `graph_sync/cli.py` that `_dump`s the counts (build store, `init_schema`, close in `finally` — mirror existing commands). Reuse the existing `dead_semantic_job_count` or subsume it in the grouped counts.

**Acceptance:** seeding jobs across `pending`/`in_progress`/`done`/`failed`/`dead` yields the correct grouped counts; the CLI command is registered and runs.

## 4. Fix 3 — migration ALTER-path test

**Problem:** the idempotent migration's existing-DB branch (`ALTER TABLE semantic_jobs ADD COLUMN IF NOT EXISTS lane / next_attempt_at`) is never exercised — testcontainers always start empty, so only the `CREATE TABLE`-with-columns path runs.

**Fix:** an integration test that (a) creates an **A-era** `semantic_jobs` table by hand (the columns as they were before this migration — no `lane`/`next_attempt_at`) plus a row, (b) runs `StateStore.init_schema()`, (c) asserts `lane` (default `'incremental'`) and `next_attempt_at` now exist on the pre-existing row and that a lane-aware `enqueue_semantic_job`/`claim_semantic_jobs(…, True)` round-trips. Uses the real Postgres fixture.

**Acceptance:** the test fails against a hypothetical CREATE-only schema and passes with the shipped ALTER branch; running `init_schema` twice stays a no-op.

## 5. Fix 4 — package-cycle refactor (neutral DocExtractor client)

**Problem:** `graph_extract/cli.py` and `graph_extract/probe.py` import `graph_sync.delta_client.make_client`, while `graph_sync` now imports `graph_extract` (the worker) — a package-level cycle (a smell, not a deadlock).

**Fix:** introduce a neutral module `src/docext/client.py` with a primitive-arg builder:
```python
def make_docext_client(*, base_url: str, read_key: str, admin_key: str,
                       verify_tls: bool, admin: bool = False) -> httpx.AsyncClient: ...
```
(the current `make_client` body, but taking the four connection fields directly instead of a `graph_sync.config.Settings` object — this also removes the `type: ignore` that `graph_extract` needs today). Then:
- `graph_sync/delta_client.make_client(settings, *, admin=False)` becomes a thin wrapper calling `make_docext_client(base_url=settings.docext_base_url, …)` — its existing callers (`app`, `cli`, `sync_core` via `DeltaStream`) are untouched.
- `graph_extract/cli.py` and `probe.py` call `docext.client.make_docext_client(base_url=settings.docext_base_url, …)` directly — **no more `from graph_sync…` import**.
- `DeltaStream`/`build_delta_params` stay in `graph_sync/delta_client.py` (genuinely sync-internal; `graph_extract` doesn't use them).

**Acceptance:** `grep -rn "graph_sync" src/graph_extract/` returns **zero** matches; all existing tests still pass (the sync path and the extract path both build a working client); ruff/mypy clean.

## 6. Fix 5 — test/doc polish

- **noise-exclusion test** (`tests/unit/test_ontology.py`): extend the existing "instructions exclude noise" assertion to also require the error-code markers (`Failed`, `RequestId`) that `EXTRACTION_INSTRUCTIONS` names — closing the coverage gap flagged in the 2b-quality T6 review.
- **VMware phrasing** (`ontology.py`): a one-line docstring/comment clarification distinguishing `VMware` (a hypervisor **Platform**) from `VMware vSphere` (a **Workload** example), so a future contributor doesn't read the two examples as contradictory.

**Acceptance:** the extended test asserts the markers and passes; the docstring reads unambiguously; public ontology surface unchanged (still 9 types / 8 edges).

## 7. Testing & constraints

- Fixes 1–3 use the Postgres testcontainer; Fix 4 is covered by the existing suite + the layering grep; Fix 5 is a unit test + docstring.
- **Signature ripple:** `complete_semantic_job`/`fail_semantic_job` gain `claimed_at`; the worker and any tests calling them are updated in the same fix (Fix 1). `claim_semantic_jobs` RETURNING gains a field — additive, existing callers unaffected.
- Layering: `graph_sync → graph_extract` only (Fix 4 makes it strictly one-directional).
- No behavior change to extraction/temporal-policy; no secrets; ruff/mypy clean throughout.

## 8. Out of scope (per triage)

- **Sub-slice C** (next): residual-staleness sweep (must handle the shrink-restore-superseded edge case) + structural↔semantic `SAME_AS` reconciliation (+ the should_distinct silent-merge concern).
- **Region-magnet** (#8): a `Region`-docstring negative-example pass — deferred to ride the next live re-extraction.
- **Group D**: `_API_ERR Id$` watch, canonicalization marginality, `resolve_chain` superseded-episode gap (answer-api), non-atomic record+complete (subsumed by the fencing fix's cost-only framing), cross-encoder reranker / `heading_path` / N+1 batching (later phases).
