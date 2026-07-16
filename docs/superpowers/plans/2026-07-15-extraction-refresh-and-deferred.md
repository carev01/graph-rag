# Extraction Refresh & Deferred Items Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the extraction model to gpt-oss-120b, fix the Region-magnet, clear small reconcile deferrals, add a `maintenance` command + runbook, then re-extract and re-validate the full quality metrics.

**Architecture:** Small code changes in `graph_extract` (config, ontology, reconcile, cli) + a runbook doc, then a controller-run live re-extraction and quality report.

**Tech Stack:** Python 3.12, Azure gpt-oss-120b (Responses endpoint, `generic_json_schema` client path, `reasoning=low`), Neo4j 5.26, pytest + testcontainers, ruff, mypy.

## Global Constraints

- **No secrets committed.** The model is set in the untracked `.env` (`LLM_MODEL=gpt-oss-120b`, `LLM_REASONING_EFFORT=low`, endpoint/key unchanged); the committed change is only the in-repo *default* + comments.
- **Public ontology surface unchanged** (9 entity types / 8 edges — a test asserts this). Region change is docstring-only.
- **Reconcile stays link-not-merge / idempotent**; only the reporting shape + the `assert`→`raise` change.
- Determinism of the maintenance jobs preserved (they compose existing jobs; no new LLM in the composition).
- Tests green (Neo4j/Postgres testcontainers); ruff/mypy clean.

---

## File Structure

- `src/graph_extract/config.py` — default `llm_model` → `gpt-oss-120b`; comments.
- `src/graph_extract/ontology.py` — `Region` docstring negative examples.
- `src/graph_extract/reconcile.py` — kind-aware `unmatched_structural`; `assert`→`raise`.
- `src/graph_extract/cli.py` — `maintenance` command.
- `docs/superpowers/maintenance-runbook.md` — new runbook.
- Tests: `tests/unit/test_ontology.py`, `tests/integration/test_reconcile.py`, `tests/unit/test_config.py` (if present), a `maintenance` `--help` check.
- Report (T5): `docs/superpowers/extraction-refresh-report.md`.

---

### Task 1: Config default → gpt-oss-120b

**Files:**
- Modify: `src/graph_extract/config.py`
- Test: `tests/unit/test_config.py` (if it asserts on `llm_model`; otherwise none)

- [ ] **Step 1: Change the default + comment.** `llm_model: str = "gpt-oss-120b"`. Update the nearby comment to note this is the chosen extraction tier (validated 2026-07 on Azure; runs via the `generic_json_schema` client path; `reasoning_effort=low`), Azure endpoint/version/key come from `.env`. Extend the `llm_reasoning_effort` comment to mention gpt-oss uses `low|medium|high` too.

- [ ] **Step 2: Check for a config test.** `grep -rn "gpt-oss-20b\|llm_model" tests/`. If a unit test asserts the old default, update it to `gpt-oss-120b`.

- [ ] **Step 3: Run gate.** `uv run --extra dev pytest -q -m "not live" && uv run --extra dev ruff check src/graph_extract/config.py && uv run --extra dev mypy src/graph_extract`.

- [ ] **Step 4: Commit.** `git add src/graph_extract/config.py tests/unit/test_config.py 2>/dev/null; git commit -m "config(extract): default extraction model gpt-oss-120b (validated)"`

---

### Task 2: Region docstring negative examples (Region-magnet fix)

**Files:**
- Modify: `src/graph_extract/ontology.py`
- Test: `tests/unit/test_ontology.py`

**Interfaces:** unchanged public surface (still 9 types / 8 edges); only `Region.__doc__` text changes.

- [ ] **Step 1: Write a failing assertion** in `tests/unit/test_ontology.py` — assert `ontology.Region.__doc__` mentions the new negatives (e.g. contains `"Availability Zone"` and a phrase excluding API operations/policies). Pick literal substrings you will add in Step 2.

- [ ] **Step 2: Extend the `Region` docstring.** Current: `"A specific geographic or cloud region, or a jurisdiction … NOT a Platform …, NOT a generic relative term …, NOT a redundancy tier …"`. Add: `— NOT an API operation or field (DescribeKey, ...Arn), NOT a policy/permission or role name, NOT a scenario/section/page name, NOT a data-classification term (PII), NOT a bare "Availability Zone"; those are Tools/Requirements/Concepts/Workloads or noise.` Keep it one tight docstring. Do NOT touch `ENTITY_TYPES`/`EDGE_TYPES`/`EDGE_TYPE_MAP`.

- [ ] **Step 3: Run `test_ontology.py`** — the new assertion passes AND the public-surface test (9 types/8 edges) still passes. `uv run --extra dev pytest tests/unit/test_ontology.py -v`.

- [ ] **Step 4: Gate + commit.** `uv run --extra dev ruff check src/graph_extract/ontology.py && uv run --extra dev mypy src/graph_extract`; `git add src/graph_extract/ontology.py tests/unit/test_ontology.py && git commit -m "feat(extract): Region docstring negatives (API-ops/policies/PII/AZ are not Regions)"`

---

### Task 3: Reconcile polish (kind-aware unmatched + assert→raise + Product test)

**Files:**
- Modify: `src/graph_extract/reconcile.py`
- Test: `tests/integration/test_reconcile.py`

**Interfaces:** `reconcile_same_as` output `unmatched_structural` becomes a `list[str]` of `"{kind}:{name}"` entries (kind-aware); `structural_scanned`/`linked` unchanged.

- [ ] **Step 1: Write failing tests.**
  - `test_reconcile_links_aliased_product`: seed structural `(:Product {name:'AWS Backup'})` + semantic `(:Entity:Product {group_id:g, name:'AWS Backup'})` (normalized-name match — no alias needed) → after reconcile a `(:Product)-[:SAME_AS]->(:Entity:Product)` exists and `res["linked"]>=1`.
  - `test_reconcile_unmatched_is_kind_aware`: seed an unmatched structural `(:Vendor {name:'Zeta'})` and `(:Product {name:'Zeta'})` (same name, different kind, no semantic match) → `res["unmatched_structural"]` contains BOTH `"Vendor:Zeta"` and `"Product:Zeta"` (they don't collapse). Use unique names to stay order-independent in the shared container.

- [ ] **Step 2: Run, confirm fail** (product path un-tested passes trivially today; the kind-aware test fails because current `sorted(set(unmatched))` collapses same-name entries).

- [ ] **Step 3: Implement.** In `reconcile.py`:
  - Replace `assert link_record is not None` with `if link_record is None: raise RuntimeError("reconcile: count() returned no row")`.
  - Track unmatched as `(kind, name)`: change `unmatched: list[str]` to collect `f"{kind}:{st['name']}"` (or a `(kind, name)` tuple list), and return `sorted(set(unmatched))` over those kind-qualified strings.

- [ ] **Step 4: Run tests green** (all reconcile tests incl the new two). Gate on `reconcile.py`.

- [ ] **Step 5: Commit.** `git add src/graph_extract/reconcile.py tests/integration/test_reconcile.py && git commit -m "fix(extract): reconcile kind-aware unmatched + raise-not-assert + product-path test"`

---

### Task 4: `maintenance` CLI command + runbook

**Files:**
- Modify: `src/graph_extract/cli.py`
- Create: `docs/superpowers/maintenance-runbook.md`
- Test: `tests/unit/test_extract_cli.py` (a `--help` smoke test if that file exists; else the registration check inline)

**Interfaces:** a `maintenance` command running `prune_noise_entities` → `retype_region_entities` → `sweep_stale_facts` → `reconcile_same_as`, dumping `{prune, retype, sweep, reconcile}`.

- [ ] **Step 1: Add the command** in `cli.py`, mirroring the `cleanup` command (read it: builds `_build_driver(settings)`, runs the correctors, `_dump`, closes in `finally`). Import `sweep_stale_facts` and `reconcile_same_as`. The `maintenance` command runs all four in order and `_dump`s the combined audit. Keep `cleanup` as-is (prune+retype only).

```python
@app.command("maintenance")
def maintenance() -> None:
    """Full housekeeping pass: prune noise, retype regions, sweep stale facts, reconcile SAME_AS."""
    async def _run() -> None:
        settings = get_extract_settings()
        driver = await _build_driver(settings)
        try:
            prune = await prune_noise_entities(driver, settings.group_id)
            retype = await retype_region_entities(driver, settings.group_id)
            sweep = await sweep_stale_facts(driver, settings.group_id)
            reconcile = await reconcile_same_as(driver, settings.group_id)
            _dump({"prune": prune, "retype": retype, "sweep": sweep, "reconcile": reconcile})
        finally:
            await driver.close()
    asyncio.run(_run())
```
(Add the two imports at the top: `from graph_extract.staleness_sweep import sweep_stale_facts` and `from graph_extract.reconcile import reconcile_same_as`.)

- [ ] **Step 2: Write the runbook** `docs/superpowers/maintenance-runbook.md`: one paragraph per job (what it does, why), the recommended cadence (`cleanup` = prune+retype after each incremental batch; `sweep` weekly; `reconcile` weekly; or `maintenance` weekly for a simple deployment), and a note that cron/systemd-timer/K8s-CronJob scheduling is a deployment concern. Include the exact CLI invocations.

- [ ] **Step 3: Verify registration + gate.** `uv run --extra dev python -c "from typer.testing import CliRunner; from graph_extract.cli import app; print('maintenance' in CliRunner().invoke(app,['--help']).output)"` prints True; add a `--help` smoke test if `test_extract_cli.py` has that pattern; `ruff`/`mypy` clean; full non-live suite green.

- [ ] **Step 4: Commit.** `git add src/graph_extract/cli.py docs/superpowers/maintenance-runbook.md tests/unit/test_extract_cli.py 2>/dev/null; git commit -m "feat(extract): maintenance CLI (prune+retype+sweep+reconcile) + runbook"`

---

### Task 5: Live re-extraction + full re-validation (controller-run)

**Files:**
- Create: `docs/superpowers/extraction-refresh-report.md`

This is controller-run live work (real Azure gpt-oss-120b extraction). No new unit tests.

- [ ] **Step 1: Sample validation.** Confirm `.env` has `LLM_MODEL=gpt-oss-120b`, `LLM_REASONING_EFFORT=low`. Then:
```bash
cd /home/openclaw/Documents/Shared/projects/graph-rag
uv run --extra dev python scripts/reset_semantic_layer.py
PILOT_IDS_FILE=scripts/pilot-ids-sample.txt uv run --extra dev python scripts/run_pilot.py
uv run --extra dev python -m graph_extract.cli cleanup
```
Spot-check: extraction non-empty/sensible; the Region type no longer swallows API-ops/policies/PII (query `:Region` names, count gazetteer-recognized vs not); noise ≈ 0%; Tool/Limits/AvailableIn still populated. If the Region-magnet persists, iterate the Region docstring (Task 2) on the sample before the full run.

- [ ] **Step 2: Full 39-article run.**
```bash
uv run --extra dev python scripts/reset_semantic_layer.py
uv run --extra dev python scripts/run_pilot.py
uv run --extra dev python -m graph_extract.cli maintenance
```

- [ ] **Step 3: Re-validate the quality metrics.** `uv run --extra dev python -m graph_extract.cli quality-report` (noise, dedup tri-state + silent_merge_suspects, type-precision at N=100). Capture: entity/fact counts; noise rate; type-precision + per-type; Region total vs gazetteer-recognized (the magnet count); Tool/Limits/AvailableIn/Region counts; cost/tokens per episode (from `run_pilot`'s cost report). Human spot-check ~20 facts + the type_precision misclassifications.

- [ ] **Step 4: Write the report** `docs/superpowers/extraction-refresh-report.md` — gpt-oss-120b vs the gpt-5-mini baseline (`slice-2b-followups-report.md`) on the headline metrics side-by-side, the human read, cost delta, the Region-magnet before/after, and a **GO/NO-GO** on keeping gpt-oss-120b. If NO-GO, note the revert (`.env` + config default back to gpt-5-mini) and DO NOT change the committed default back in this task (surface to the user).

- [ ] **Step 5: Commit the report.** `git add docs/superpowers/extraction-refresh-report.md && git commit -m "docs: extraction-refresh report (gpt-oss-120b vs gpt-5-mini, GO/NO-GO)"`

---

## Self-Review Notes

- **Spec coverage:** config → T1; Region fix → T2; reconcile polish → T3; maintenance+runbook → T4; re-extract+re-validate+report → T5. All §-items covered.
- **Deps:** T1 and T2 must land before T5 (the re-run uses the new model + ontology). T3, T4 independent. Order: T1, T2, T3, T4, T5.
- **Type consistency:** `unmatched_structural` is `list[str]` of `"kind:name"`; `maintenance` audit `{prune,retype,sweep,reconcile}`; all existing job signatures reused unchanged.
- **Risk:** T5 may reveal gpt-oss-120b underperforms on the full corpus despite the smoke test — the report's GO/NO-GO + revert path handles it; the config supports both models interchangeably.
- **No-placeholder check:** each code step carries actual code/exact commands.
