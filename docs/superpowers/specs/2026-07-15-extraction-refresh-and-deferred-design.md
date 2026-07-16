# Extraction Refresh & Deferred Items — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Extraction-model switch (gpt-5-mini → gpt-oss-120b) + Region-magnet fix + small deferred cleanups + full re-validation.
**Date:** 2026-07-15
**Status:** Approved design — ready for implementation planning

Clears the remaining small deferred items and refreshes the semantic graph on a cheaper extraction model. The model swap to **gpt-oss-120b** (same Azure endpoint/key, `reasoning=low`) is already **validated**: a one-episode smoke test produced 9 clean entities / 5 facts (incl. a `Limits` fact), no reasoning leak, working token capture — via the existing `generic_json_schema` client path. Since a new extraction model re-writes the whole semantic layer, the full quality metrics are re-validated (not just the Region-magnet).

---

## 1. Scope

**Code (no re-extraction):**
1. **Config default** — set `ExtractSettings.llm_model` default to `gpt-oss-120b` (committed default is still the stale `gpt-oss-20b`); update its comment. `.env` already carries the real value; this documents the chosen production model in-repo.
2. **Region-magnet fix** — add negative examples to the `Region` docstring so API operations, policies, scenarios, `PII`, and `Availability Zone` are not typed `Region`.
3. **Reconcile polish** — make `unmatched_structural` kind-aware; replace the `python -O`-strippable `assert` with a `raise`; add the missing `:Product`-path test.
4. **Maintenance runbook** — a `maintenance` CLI command that runs `prune`+`retype`+`sweep`+`reconcile` in order, plus a short runbook doc noting the recommended cadence. An actual cron/scheduler stays a deployment concern (out of scope).

**Live (the re-extraction):**
5. Re-extract with gpt-oss-120b + the ontology fix (sample → full 39-article run), then re-validate the quality metrics (noise / dedup / type-precision / Region-magnet) and write a short before/after report vs the gpt-5-mini baseline.

**Out of scope:** cron/scheduler wiring; Phase-2 retrieval; the group-D deferrals (`_API_ERR Id$` watch, canonicalization marginality, `resolve_chain` superseded gap, cross-encoder reranker, `heading_path`, N+1 batching).

## 2. Config default (item 1)

`src/graph_extract/config.py`: `llm_model: str = "gpt-oss-120b"` with a comment that this is the chosen extraction tier (validated 2026-07; runs via `generic_json_schema`; `reasoning_effort=low`), and that Azure endpoint/version/key come from `.env`. No secret in the repo. The `llm_reasoning_effort` comment's "gpt-5 family" note is extended to include gpt-oss (`low|medium|high`).

## 3. Region-magnet fix (item 2)

`src/graph_extract/ontology.py` — the `Region` docstring currently says a specific geo/cloud region/jurisdiction. Add an explicit negative list: **NOT** an API operation or field (`DescribeKey`, `RetireGrant`), a policy/permission name (`Managed policies for AWS Backup`), a scenario/section name, a data-classification term (`PII`), or a bare `Availability Zone` — those are Tool/Requirement/Concept/Workload/noise as appropriate. Keep it tight; public surface unchanged (still 9 types / 8 edges — a test asserts this). The deterministic `noise_filter` (numeric/org-id/Invalid patterns) already prunes the clear noise; this docstring fix reduces the model *typing* non-regions as `Region` at the source.

## 4. Reconcile polish (item 3)

`src/graph_extract/reconcile.py`:
- **Kind-aware unmatched:** `unmatched_structural` currently `sorted(set(names))`, so an unmatched `:Vendor` and `:Product` with the same name collapse to one entry. Return `[{"kind": ..., "name": ...}]` (or `"Vendor:AWS"`-style keys) so both are reported. (Choose the shape at plan time; keep the dict's other keys stable.)
- **`assert`→`raise`:** the `assert record is not None` guards on the aggregate queries become `if record is None: raise RuntimeError(...)` (assert is stripped under `python -O`).
- **`:Product` path test:** add a reconcile test seeding a structural `:Product` + an aliased semantic `:Entity:Product` (extend `vendor_aliases` with a product alias if needed, or use a name that matches by normalization) → `SAME_AS` linked. Currently only the `:Vendor` path is exercised.

## 5. Maintenance runbook (item 4)

- **`maintenance` CLI command** (`src/graph_extract/cli.py`): builds the driver once and runs, in order, `prune_noise_entities` → `retype_region_entities` → `sweep_stale_facts` → `reconcile_same_as`, `_dump`ing a combined `{prune, retype, sweep, reconcile}` audit; closes in `finally`. This is the single "housekeeping" entry point.
- **Runbook doc** (`docs/superpowers/maintenance-runbook.md`): what each job does, the recommended cadence (post-ingest `cleanup`/prune+retype after every incremental batch; a **weekly** `sweep`, a **weekly** `reconcile`; or just `maintenance` weekly for a simple deployment), and that scheduling (cron/systemd-timer/K8s CronJob) is a deployment concern the ops team wires. No behavior beyond composing the existing jobs.

## 6. Re-extraction & re-validation (item 5)

- **Baseline for comparison:** the current graph (gpt-5-mini, ontology-v5, ~552 entities / ~2094 facts) and the `slice-2b-followups-report.md` metrics are the "before."
- **Sample validation:** reset the semantic layer, ingest `scripts/pilot-ids-sample.txt` (8 articles) with gpt-oss-120b + the ontology fix, run `cleanup`, and spot-check: extraction is non-empty and sensible; the Region type no longer swallows API-ops/policies/PII; noise ≈ 0%; `AvailableIn`/`Limits`/`Tool` still populated. Iterate the Region docstring on the sample if needed (cheap).
- **Full run:** reset, ingest the full `scripts/pilot-ids.txt` (39 articles) with gpt-oss-120b, run `cleanup`, then `quality-report` (noise, dedup tri-state, type-precision) + the Region breakdown + `sweep`/`reconcile` sanity.
- **Report:** `docs/superpowers/extraction-refresh-report.md` — gpt-oss-120b vs gpt-5-mini on the headline metrics (entity/fact counts, noise, type-precision at N=100, Region-magnet count, Tool/Limits/AvailableIn/Region counts, cost/tokens per episode), a human spot-check, and a GO/NO-GO note on keeping gpt-oss-120b as the extraction tier.

## 7. Testing

- Config/ontology/reconcile changes: unit + integration (Neo4j testcontainer) — the reconcile `:Product`-path and kind-aware-unmatched tests; the ontology public-surface test stays green; a Region-docstring negative-example assertion.
- `maintenance` CLI: `--help` registration + a light integration test that it composes the four jobs (a stubbed/seeded run asserting the combined audit shape).
- The re-extraction (item 5) is controller-run live work; validated by the report, not CI.
- Full non-live suite + ruff/mypy clean.

## 8. Acceptance criteria

1. `llm_model` default is `gpt-oss-120b`; `.env` unchanged, no secret committed.
2. The `Region` docstring carries the negative examples; public ontology surface unchanged; tests green.
3. `reconcile_same_as` reports unmatched structural nodes kind-aware; no `assert` in the production path; the `:Product` reconcile path is tested.
4. `maintenance` CLI runs prune→retype→sweep→reconcile and is documented in the runbook.
5. The full re-extraction on gpt-oss-120b completes; the refresh report shows the quality metrics vs gpt-5-mini with a GO/NO-GO; the Region-magnet is measurably reduced.
6. Unit + integration tests green; ruff/mypy clean.

## 9. Deferred

- Cron/scheduler wiring (deployment).
- If gpt-oss-120b underperforms on the full run (a real risk despite the smoke test), the report's NO-GO path reverts `.env`/default to gpt-5-mini — the config supports both interchangeably.
- Group-D items and Phase-2/3 remain out of scope.
