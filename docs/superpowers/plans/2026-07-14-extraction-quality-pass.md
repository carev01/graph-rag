# Extraction Quality Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce extraction noise and improve dedup/type precision in the `graph_extract` pipeline, and demonstrate the improvement quantitatively against the 2a baseline.

**Architecture:** Extraction-side prevention (ontology-v2 + prompt-v2) plus a deterministic post-filter. A pure `noise_filter` is the single definition of noise (drives both a metric and a cleanup pass). Quality is measured by three metrics — noise (deterministic), dedup vs a labelled set (deterministic), and type precision (judged by an independent GLM-5.2, never self-judged) — captured as a 2a baseline then re-run after tuning.

**Tech Stack:** Python 3.12, `graph_extract` (existing), Neo4j 5.26, Azure `gpt-5-mini` (extraction, unchanged), GLM-5.2 via Ollama Cloud (judge), pytest + testcontainers, ruff, mypy.

Spec: [`../specs/2026-07-14-extraction-quality-pass-design.md`](../specs/2026-07-14-extraction-quality-pass-design.md).

## Global Constraints

- Python 3.12. **The quality gate is enforced now**: `ruff check src tests` and `mypy src` must stay clean; default test lane `pytest -m "not live"`.
- **`noise_filter` is the single source of truth for "what is noise"** — the metric and the cleanup pass both call it; they can never disagree.
- **No self-judging.** `type_precision`'s judge is GLM-5.2 (`judge_base_url=https://ollama.com/v1`, `judge_model=glm-5.2:cloud`), distinct from the extraction model. GLM-5.2 is a **reasoning** model (reasoning in a separate field, final answer in `content`) → judge calls must allow an adequate token budget (~500), not a tiny cap, or the answer truncates. The two deterministic metrics (noise, dedup) use **no LLM**.
- **Baseline is captured from the intact 2a graph** (693 entities, still in the compose Neo4j) BEFORE any ontology/prompt change. `slice-2b-quality-baseline.json` is committed.
- `group_id = "backup-docs"`. Secrets (judge key) in `.env`, never committed.
- Pure units get unit tests; graph/judge units get integration/`@live` tests. Full metric runs + the pilot re-extraction are manual report-producing steps.

---

## File Structure

```
src/graph_extract/
  noise_filter.py        # new, pure: is_noise(name, type) -> bool
  quality_labels.py      # new: SHOULD_MERGE / SHOULD_DISTINCT / VENDOR_TOKENS
  graph_cleanup.py       # new: prune_noise_entities(driver, group_id)
  eval.py                # extend: noise_report, dedup_report v2, type_precision
  config.py              # (judge_* already present; verify)
  ontology.py            # v2: type boundaries + examples, vendor-scoping, noise exclusions
  cli.py                 # quality-baseline / quality-report commands
tests/unit/ tests/integration/ ...
docs/superpowers/slice-2b-quality-baseline.json   # committed baseline
docs/superpowers/slice-2b-quality-report.md        # final before/after
```

---

### Task 1: noise_filter (pure) + quality_labels

**Files:**
- Create: `src/graph_extract/noise_filter.py`, `src/graph_extract/quality_labels.py`, `tests/unit/test_noise_filter.py`

**Interfaces:**
- `noise_filter.is_noise(name: str, type: str | None = None) -> bool` — True for high-confidence noise (ARNs, error/exception codes, `…RequestId`, CLI commands, bare resource-ids); False for real domain terms. Conservative.
- `quality_labels.SHOULD_MERGE: list[str]`, `SHOULD_DISTINCT: list[tuple[str, str]]`, `VENDOR_TOKENS: list[str]`.

- [ ] **Step 1: Write the failing test** — `tests/unit/test_noise_filter.py` (fixtures are REAL 2a entity names)

```python
import pytest
from graph_extract.noise_filter import is_noise

NOISE = [
    "arn:aws:backup:us-east-1:123456789012:recovery-point:2FC4C6F8-0FC1-546B-B91C-209C599C1D56",
    "arn:aws:ec2:us-east-1::snapshot/snap-00a129455bdbc9d99",
    "arn:aws:iam::123456789012:root",
    "CreatorRequestId",
    "AssociateBackupVaultMpaApprovalTeamFailed",
    "CreateRestoreAccessBackupVaultFailed",
    "Install-Module -Name Az.RecoveryServices -Force",
    "snap-07ce8c3141d361233",
    "vol-00a422a05b9c6asd3",
]
KEEP = [
    "immutability", "Amazon S3", "Azure Blob Storage", "RPO", "recovery point",
    "cross-region copy", "AWS Backup Vault Lock", "Kubernetes", "SEC 17a-4",
    "AWS Backup", "Azure Backup", "soft delete", "retention policy",
]

@pytest.mark.parametrize("name", NOISE)
def test_flags_noise(name):
    assert is_noise(name) is True

@pytest.mark.parametrize("name", KEEP)
def test_keeps_domain_terms(name):
    assert is_noise(name) is False
```

- [ ] **Step 2: Run to verify it fails** — `uv run --extra dev pytest tests/unit/test_noise_filter.py -v` → FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write `src/graph_extract/noise_filter.py`**

```python
from __future__ import annotations
import re

_ARN = re.compile(r"^arn:aws:", re.IGNORECASE)
_RESOURCE_ID = re.compile(r"^(snap|vol|i|ami|vpc|subnet|sg|eni)-[0-9a-f]{6,}$", re.IGNORECASE)
_CLI = re.compile(r"(^|\s)(aws|az|Install-Module|kubectl|Get-|Set-|New-)\b|--[a-z]", re.IGNORECASE)
# CamelCase API/operation or error identifiers ending in a status/verb word.
_API_ERR = re.compile(r"^[A-Z][A-Za-z0-9]*(Failed|Error|Exception|RequestId|Id)$")


def is_noise(name: str, type: str | None = None) -> bool:
    """High-confidence, pattern-matchable noise. Conservative: borderline domain
    terms are kept (return False) to avoid dropping useful entities."""
    n = name.strip()
    if not n:
        return True
    if _ARN.search(n):
        return True
    if _RESOURCE_ID.match(n):
        return True
    if _CLI.search(n):
        return True
    if _API_ERR.match(n) and " " not in n:
        return True
    return False
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS (all NOISE flagged, all KEEP kept).

- [ ] **Step 5: Write `src/graph_extract/quality_labels.py`**

```python
"""Acceptance criteria as data (reviewed at spec sign-off)."""
# Generic concepts that appear in BOTH vendors' docs and SHOULD resolve to one
# shared entity node (cross-vendor merge is correct here).
SHOULD_MERGE: list[str] = [
    "immutability", "cross-region copy", "Kubernetes", "RPO", "RTO",
    "encryption", "recovery point", "retention policy", "soft delete",
    "backup vault", "restore",
]
# Pairs that must stay DISTINCT (vendor-specific; collapsing them is a false merge).
SHOULD_DISTINCT: list[tuple[str, str]] = [
    ("Amazon S3", "Azure Blob Storage"),
    ("AWS Backup", "Azure Backup"),
    ("AWS Backup Vault Lock", "Azure immutable vault"),
]
# Name tokens that mark an entity as vendor-branded (used by the
# suspect-false-merge detector).
VENDOR_TOKENS: list[str] = ["aws", "amazon", "azure", "microsoft"]
```

- [ ] **Step 6: Commit**

```bash
git add src/graph_extract/noise_filter.py src/graph_extract/quality_labels.py tests/unit/test_noise_filter.py
git commit -m "feat(extract): pure noise_filter + quality_labels (acceptance data)"
```

---

### Task 2: graph_cleanup.prune_noise_entities

**Files:**
- Create: `src/graph_extract/graph_cleanup.py`, `tests/integration/test_graph_cleanup.py`

**Interfaces:**
- `async def prune_noise_entities(driver, group_id: str) -> dict` — loads `:Entity {group_id}` names, flags via `noise_filter.is_noise`, `DETACH DELETE`s the flagged ones (dropping their `RELATES_TO`/`MENTIONS`), returns `{"scanned": int, "pruned": int, "pruned_names": list[str]}`.

- [ ] **Step 1: Write the integration test** — `tests/integration/test_graph_cleanup.py` (uses the existing `extract_driver` testcontainer fixture)

```python
import pytest
pytestmark = pytest.mark.asyncio(loop_scope="module")

async def test_prune_removes_noise_keeps_real(extract_driver):
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {name:'arn:aws:ec2:us-east-1::snapshot/snap-1', group_id:'g'})")
        await s.run("CREATE (:Entity {name:'CreatorRequestId', group_id:'g'})")
        await s.run("CREATE (:Entity {name:'immutability', group_id:'g'})")
        await s.run("CREATE (:Entity {name:'Amazon S3', group_id:'g'})")
        # a fact edge on a noise node -> must go with it
        await s.run("MATCH (a:Entity {name:'arn:aws:ec2:us-east-1::snapshot/snap-1'}), "
                    "(b:Entity {name:'immutability'}) "
                    "CREATE (a)-[:RELATES_TO {group_id:'g'}]->(b)")
    from graph_extract.graph_cleanup import prune_noise_entities
    res = await prune_noise_entities(extract_driver, "g")
    assert res["pruned"] == 2
    async with extract_driver.session() as s:
        r = await s.run("MATCH (e:Entity {group_id:'g'}) RETURN count(e) AS n")
        assert (await r.single())["n"] == 2                 # only the 2 real kept
        r = await s.run("MATCH ()-[x:RELATES_TO {group_id:'g'}]->() RETURN count(x) AS n")
        assert (await r.single())["n"] == 0                 # noise fact gone
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL.

- [ ] **Step 3: Write `src/graph_extract/graph_cleanup.py`**

```python
from __future__ import annotations
from neo4j import AsyncDriver
from graph_extract.noise_filter import is_noise

async def prune_noise_entities(driver: AsyncDriver, group_id: str) -> dict:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) "
            "RETURN e.name AS name, [l IN labels(e) WHERE l<>'Entity'][0] AS type",
            g=group_id)
        rows = [dict(rec) async for rec in r]
        noise = [row["name"] for row in rows if is_noise(row["name"], row.get("type"))]
        if noise:
            await s.run(
                "MATCH (e:Entity {group_id:$g}) WHERE e.name IN $names DETACH DELETE e",
                g=group_id, names=noise)
    return {"scanned": len(rows), "pruned": len(noise), "pruned_names": noise}
```

- [ ] **Step 4: Run to verify it passes** — Expected: PASS (2 pruned, 2 kept, 0 noise facts).

- [ ] **Step 5: Commit**

```bash
git add src/graph_extract/graph_cleanup.py tests/integration/test_graph_cleanup.py
git commit -m "feat(extract): prune_noise_entities deterministic noise cleanup"
```

---

### Task 3: eval v2 — noise_report + dedup_report v2 + type_precision

**Files:**
- Modify: `src/graph_extract/eval.py`
- Create: `tests/integration/test_eval_quality.py`

**Interfaces:**
- `async def noise_report(driver, group_id) -> dict` — `{entities: {total, noise, rate, by_flag_sample}, facts: {total, noise, rate}}` using `noise_filter` over `:Entity` names and `RELATES_TO` endpoints.
- `async def dedup_report_v2(driver, group_id, labels) -> dict` — for each `SHOULD_MERGE`: node_count (case-insensitive) + cross-vendor support (via `HAS_EPISODE`→Article→Vendor); each `SHOULD_DISTINCT` pair: collapsed bool; `suspect_false_merge`: entities whose name contains a `VENDOR_TOKEN` and have cross-vendor support (list + count).
- `async def type_precision(driver, settings, sample: int) -> dict` — sample N entities `(name, type)`, ask GLM-5.2 to pick the best type from the v2 definitions, compare, return overall + per-type precision + misclassifications. Uses `_judge_client_and_model` (must resolve to GLM-5.2, NOT the extraction model); judge call allows `max_tokens>=500` (reasoning model).

- [ ] **Step 1: Write the integration test** — `tests/integration/test_eval_quality.py` (seed a tiny graph; the deterministic reports need no LLM)

```python
import pytest
pytestmark = pytest.mark.asyncio(loop_scope="module")

async def _seed_cross_vendor(driver):
    # immutability mentioned by an AWS article AND an Azure article -> cross-vendor.
    async with driver.session() as s:
        await s.run("""
        CREATE (vA:Vendor {id:'vA', name:'AWS'})-[:HAS_PRODUCT]->(pA:Product {id:'pA'})
              -[:HAS_SOURCE]->(sA:Source {id:'sA'})-[:HAS_ARTICLE]->(aA:Article {id:'aA'})
              -[:HAS_EPISODE]->(eA:Episodic {uuid:'epA'})
        CREATE (vM:Vendor {id:'vM', name:'Microsoft'})-[:HAS_PRODUCT]->(pM:Product {id:'pM'})
              -[:HAS_SOURCE]->(sM:Source {id:'sM'})-[:HAS_ARTICLE]->(aM:Article {id:'aM'})
              -[:HAS_EPISODE]->(eM:Episodic {uuid:'epM'})
        CREATE (imm:Entity:Capability {name:'immutability', group_id:'g'})
        CREATE (eA)-[:MENTIONS]->(imm)
        CREATE (eM)-[:MENTIONS]->(imm)
        CREATE (s3:Entity:Workload {name:'Amazon S3', group_id:'g'})
        CREATE (blob:Entity:Workload {name:'Azure Blob Storage', group_id:'g'})
        // vendor-branded entity with cross-vendor support -> a suspect false merge
        CREATE (vl:Entity:Product {name:'AWS Backup Vault Lock', group_id:'g'})
        CREATE (eA)-[:MENTIONS]->(vl)
        CREATE (eM)-[:MENTIONS]->(vl)
        """)

async def test_dedup_report_v2(extract_driver):
    await _seed_cross_vendor(extract_driver)
    from graph_extract.eval import dedup_report_v2
    from graph_extract import quality_labels
    rep = await dedup_report_v2(extract_driver, "g", quality_labels)
    assert rep["should_merge"]["immutability"]["node_count"] == 1
    assert rep["should_merge"]["immutability"]["cross_vendor"] is True
    pair = next(p for p in rep["should_distinct"] if p["pair"] == ["Amazon S3", "Azure Blob Storage"])
    assert pair["collapsed"] is False
    assert any("AWS Backup Vault Lock" in nm for nm in rep["suspect_false_merge"]["names"])

async def test_noise_report(extract_driver):
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {name:'arn:aws:x', group_id:'g2'})")
        await s.run("CREATE (:Entity {name:'immutability', group_id:'g2'})")
    from graph_extract.eval import noise_report
    rep = await noise_report(extract_driver, "g2")
    assert rep["entities"]["total"] == 2 and rep["entities"]["noise"] == 1
```

- [ ] **Step 2: Run to verify it fails** — Expected: FAIL.

- [ ] **Step 3: Implement the three functions in `eval.py`.** `noise_report` and `dedup_report_v2` are Cypher + `noise_filter`/`quality_labels` (no LLM). `type_precision` samples entities and prompts the GLM judge (a `_TYPE_JUDGE_PROMPT` embedding the v2 type definitions; `max_tokens=600`; parse the final `content` word). Cross-vendor support Cypher:

```python
async def _cross_vendor_names(session, group_id) -> set[str]:
    r = await session.run(
        "MATCH (v:Vendor)-[:HAS_PRODUCT]->(:Product)-[:HAS_SOURCE]->(:Source)"
        "-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(:Episodic)-[:MENTIONS]->"
        "(e:Entity {group_id:$g}) "
        "WITH e, count(DISTINCT v.name) AS nv WHERE nv > 1 RETURN e.name AS name",
        g=group_id)
    return {rec["name"] async for rec in r}
```
`suspect_false_merge` = cross-vendor names whose lowercased name contains any `VENDOR_TOKENS`.

- [ ] **Step 4: Run to verify it passes** — Expected: PASS (deterministic tests).

- [ ] **Step 5: Commit**

```bash
git add src/graph_extract/eval.py tests/integration/test_eval_quality.py
git commit -m "feat(extract): eval v2 (noise_report, dedup_report_v2, type_precision)"
```

---

### Task 4: judge config (GLM-5.2) + CLI quality commands

**Files:**
- Modify: `src/graph_extract/cli.py`; (verify `config.py` judge fields); `.env` (not committed)

**Interfaces:**
- `.env`: `JUDGE_BASE_URL=https://ollama.com/v1`, `JUDGE_MODEL=glm-5.2:cloud`, `JUDGE_API_KEY=<key>`.
- `cli.py`: `quality-baseline` (runs the three reports, writes `docs/superpowers/slice-2b-quality-baseline.json`) and `quality-report` (runs them, diffs against the baseline JSON, writes `docs/superpowers/slice-2b-quality-report.md`).

- [ ] **Step 1: Set the judge in `.env`** (idempotent set-or-append; never commit the key). Confirm `_judge_client_and_model(get_extract_settings())` resolves to GLM-5.2, not the extraction model.

- [ ] **Step 2: Live-verify the judge** — a one-off call proving GLM-5.2 answers type questions (with adequate max_tokens):
```bash
uv run --extra dev python -c "
import asyncio
from graph_extract.config import get_extract_settings
from graph_extract.eval import _judge_client_and_model
async def main():
    c, m = _judge_client_and_model(get_extract_settings.__wrapped__())
    r = await c.chat.completions.create(model=m, max_tokens=600, temperature=0,
        messages=[{'role':'user','content':'Best type for \"AWS CLI\": Platform or Tool? One final word.'}])
    print(m, '->', (r.choices[0].message.content or '')[-40:])
    await c.close()
asyncio.run(main())"
```
Expected: model `glm-5.2:cloud`, answer contains "Tool".

- [ ] **Step 3: Write the `quality-baseline` / `quality-report` CLI commands** (build a neo4j driver + `_judge_client`, run `noise_report`+`dedup_report_v2`+`type_precision`, dump JSON/markdown; `quality-report` loads the baseline JSON and prints deltas). Mirror the existing eval command wiring; close resources in `finally`.

- [ ] **Step 4: Commit** (code only; `.env` stays untracked)
```bash
git add src/graph_extract/cli.py
git commit -m "feat(extract): GLM-5.2 judge wiring + quality-baseline/report CLI"
```

---

### Task 5: Capture the 2a baseline

**Files:**
- Create: `docs/superpowers/slice-2b-quality-baseline.json`

- [ ] **Step 1: Confirm the 2a graph is intact** — `MATCH (n:Entity {group_id:'backup-docs'}) RETURN count(n)` → ~693. (If not, re-extract once with the 2a-era config from git before capturing.)

- [ ] **Step 2: Run the baseline** (live — hits GLM for type_precision):
```bash
uv run --extra dev python -m graph_extract.cli quality-baseline
```
Expected: writes `slice-2b-quality-baseline.json` with noise_report + dedup_report_v2 + type_precision for the current (2a) graph.

- [ ] **Step 3: Sanity-check + commit the baseline** — the numbers should match 2a's observations (noise ~4% entities; several suspect_false_merge incl. `AWS Backup Vault Lock`; type_precision showing the AWS-CLI/SEC-17a-4 confusions).
```bash
git add docs/superpowers/slice-2b-quality-baseline.json
git commit -m "docs: capture 2a extraction-quality baseline"
```

---

### Task 6: ontology-v2 (prompt + type boundaries)

**Files:**
- Modify: `src/graph_extract/ontology.py`; `tests/unit/test_ontology.py` (keep passing; add assertions for the new guidance)

**Interfaces:** unchanged public surface (`ENTITY_TYPES`, `EDGE_TYPES`, `EDGE_TYPE_MAP`, `EXTRACTION_INSTRUCTIONS`); the docstrings and the instructions text change.

- [ ] **Step 1: Update the entity-type docstrings** with hard boundaries + negative examples (Platform ≠ tools/CLIs/regulations; Capability ≠ accounts/roles/resources; Concept holds RPO/RTO/regulations; Requirement = prerequisites). These feed Graphiti's extraction.

- [ ] **Step 2: Rewrite `EXTRACTION_INSTRUCTIONS`** to add:
  - a **noise-exclusion block**: "Do NOT extract as entities: ARNs (`arn:...`), resource IDs (`snap-...`, `vol-...`), error/exception codes (`...Failed`, `...RequestId`), CLI commands (`aws ...`, `Install-Module ...`), or example/placeholder values. Extract the concept, not the example (extract `recovery point`, not the ARN)."
  - a **vendor-scoping block**: "Use canonical GENERIC names for cross-vendor concepts so AWS and Azure converge (`immutability`, `cross-region copy`, `RPO`, `Kubernetes`). Keep VENDOR-BRANDED features vendor-specific and DISTINCT (`AWS Backup Vault Lock`, `Azure immutable vault`, `Amazon S3`, `Azure Blob Storage`) — do not merge these across vendors."

- [ ] **Step 3: Extend `tests/unit/test_ontology.py`** — assert the instructions mention the noise exclusions (e.g. "arn:", "Do NOT extract") and the vendor-scoping rule; existing tests still pass.

- [ ] **Step 4: Run tests + gate** — `uv run --extra dev pytest tests/unit/test_ontology.py -v`; `ruff check`; `mypy src`. Expected: PASS/clean.

- [ ] **Step 5: Commit**
```bash
git add src/graph_extract/ontology.py tests/unit/test_ontology.py
git commit -m "feat(extract): ontology-v2 (type boundaries, noise exclusions, vendor-scoping)"
```

---

### Task 7: Iterate on the small sample, then the final full run + report

**Files:**
- Create: `docs/superpowers/slice-2b-quality-report.md`; (a small `scripts/pilot-ids-sample.txt` for the iteration sample)

- [ ] **Step 1: Prepare a ~6–8 article high-overlap sample** — pick from `scripts/pilot-ids.txt` the strongest AWS↔Azure overlap articles (Vault Lock, cross-region/soft-delete/restore, S3/Blob) into `scripts/pilot-ids-sample.txt`.

- [ ] **Step 2: Iterate** (live; each cycle ~20–30 min):
```bash
uv run --extra dev python scripts/reset_semantic_layer.py
# point run_pilot at the sample ids, ingest, then prune + report
uv run --extra dev python scripts/run_pilot.py            # (sample file)
uv run --extra dev python -c "import asyncio;from neo4j import AsyncGraphDatabase as G;from graph_extract.config import get_extract_settings as S;from graph_extract.graph_cleanup import prune_noise_entities as P;s=S();d=G.driver(s.neo4j_uri,auth=(s.neo4j_user,s.neo4j_password));print(asyncio.run(P(d,s.group_id)))"
uv run --extra dev python -m graph_extract.cli quality-report
```
Adjust `ontology.py`/`noise_filter.py` and repeat until the sample metrics beat baseline (noise classes gone, suspect_false_merge down, types corrected).

- [ ] **Step 3: Final full 39-article run** — reset, ingest the full `scripts/pilot-ids.txt`, `prune_noise_entities`, `quality-report`.

- [ ] **Step 4: Human spot-check** — review ~20–30 facts + the type_precision misclassifications; confirm noise gone and the false-merge (`AWS Backup Vault Lock`) resolved.

- [ ] **Step 5: Write `docs/superpowers/slice-2b-quality-report.md`** — baseline-vs-after for all three metrics with deltas, the human read, and PASS/FAIL vs the §8 acceptance bar. Commit it + the sample-ids file.
```bash
git add docs/superpowers/slice-2b-quality-report.md scripts/pilot-ids-sample.txt
git commit -m "docs: slice-2b extraction-quality report (before/after)"
```

---

## Self-Review Notes

- **Spec coverage:** §3 components → Tasks 1–4,6; §4 noise → Tasks 1,2,6; §5 dedup/type → Tasks 1,3,6; §6 harness+baseline → Tasks 3,4,5; §7 testing → Tasks 1–3; §7 iteration + §8 acceptance → Task 7; §2 judge (independent, reasoning, token budget) → Task 4.
- **No self-judging** enforced: Task 4 Step 2 live-verifies the judge resolves to GLM-5.2, and `type_precision`'s call uses `max_tokens>=500` for the reasoning model.
- **noise_filter is the single source of truth**: Task 2 cleanup and Task 3 metric both import `is_noise` — no divergence.
- **Baseline before tuning**: Task 5 (baseline capture from the intact 2a graph) precedes Task 6 (ontology-v2), so the comparison is honest.
- **Naming consistency:** `is_noise`, `prune_noise_entities`, `noise_report`, `dedup_report_v2`, `type_precision`, `quality_labels.{SHOULD_MERGE,SHOULD_DISTINCT,VENDOR_TOKENS}` — used identically across tasks.
- **Deferred (not gaps):** pipeline hardening, false-merge *repair* (prevention-only here) — per spec §9.
