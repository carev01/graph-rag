# Quality-Metrics Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the 2b-quality measurements trustworthy (tri-state `should_distinct`, steadier `type_precision`) and promote regions to a first-class `Region` entity type + `AvailableIn` edges so region/residency questions ("which vendors offer SaaS backups in Germany?") are answerable, plus a canonicalization nudge.

**Architecture:** Edits to the existing `graph_extract` package — no new subsystems. New data module `region_names.py` (a curated gazetteer, single source of truth for "what is a region"). Two-layer region typing mirrors the noise two-layer design: the ontology prompt types regions as `Region`, and a deterministic `retype_region_entities` corrector guarantees it. Items 2–3 validate against the graph already in Neo4j; Items 1 & 4 change extraction and validate on a sample then full re-run.

**Tech Stack:** Python 3.12, Neo4j 5.26 (async driver), graphiti-core 0.29.2, Azure gpt-5-mini extraction, glm-5.2:cloud judge, pytest, ruff, mypy.

## Global Constraints

- **Single source of truth, per concern.** `noise_filter.is_noise` is the only definition of *noise*; `region_names.REGION_NAMES` is the only definition of *what is a region* — used by BOTH the extraction guidance and the `retype_region_entities` corrector. No divergence.
- **Ontology public surface is consumed downstream.** `ENTITY_TYPES`, `EDGE_TYPES`, `EDGE_TYPE_MAP`, `EXTRACTION_INSTRUCTIONS`, `EXCLUDED_ENTITY_TYPES` keep their names. `eval.type_precision` enumerates `ENTITY_TYPES` names+docstrings dynamically. Adding a type is safe only if added to `ENTITY_TYPES`; every `EDGE_TYPE_MAP` pair must reference types in `ENTITY_TYPES` and edges in `EDGE_TYPES`. Do NOT rename/remove the existing 8 types or 7 edges.
- **Custom type labels are ours** (design-invariant #5): relabeling an `:Entity`'s custom type label (e.g. `:Platform` → `:Region`) is allowed; never touch Graphiti's own `:Entity`/`:Episodic`/`RELATES_TO`/`MENTIONS` structure, and preserve all edges.
- **No-self-judge** stays enforced (`_judge_client_and_model` raises if judge==extraction). Judge is `glm-5.2:cloud`, `max_tokens>=500`, `temperature=0`.
- **Prevention-first dedup:** never split false-merges. `should_distinct` metric only *reports*; it does not mutate the graph.
- **Secrets only in untracked `.env`.** Never commit keys. `region_names.py` is public data (no secrets).
- **Determinism:** `dedup_report_v2`, `noise_report`, `retype_region_entities`, `region_names` contain NO LLM calls.

---

## File Structure

- `src/graph_extract/region_names.py` — **new.** `REGION_NAMES: frozenset[str]` (normalised) + `normalize_region(name) -> str` + `is_region(name) -> bool`. Pure data + helpers.
- `src/graph_extract/quality_labels.py` — **modify.** `SHOULD_DISTINCT` members gain alias support.
- `src/graph_extract/eval.py` — **modify.** `dedup_report_v2` tri-state; `type_precision` larger-N cap + retry; harden `_parse_type`.
- `src/graph_extract/ontology.py` — **modify.** Add `Region` type + `AvailableIn` edge (Product/Capability/Workload→Region); rework `EXTRACTION_INSTRUCTIONS` (regions now extracted, availability capture, canonicalization nudge).
- `src/graph_extract/graph_cleanup.py` — **modify.** Add `retype_region_entities`.
- `src/graph_extract/cli.py` — **modify.** Add a `cleanup` command running prune + retype; bump `type_precision` sample default.
- Tests: `tests/unit/test_region_names.py` (new), `tests/unit/test_ontology.py`, `tests/unit/test_eval_parse.py` (new or extend), `tests/integration/test_eval_quality.py`, `tests/integration/test_graph_cleanup.py`.

---

### Task 1: `should_distinct` tri-state + alias tolerance (Item 2)

**Files:**
- Modify: `src/graph_extract/quality_labels.py`
- Modify: `src/graph_extract/eval.py:110-145` (`dedup_report_v2`)
- Test: `tests/integration/test_eval_quality.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `dedup_report_v2` output where each `should_distinct` entry is `{"pair":[a,b], "state": "distinct"|"absent"|"merged", "collapsed": bool, "a_nodes": int, "b_nodes": int}`. `collapsed == (state=="merged")` for back-compat.

**Details:** `SHOULD_DISTINCT` members become `str | list[str]` (first = canonical, rest = aliases). Resolve each member to the SET of node element-ids whose lowercased name equals any of its forms. `state`: `absent` if either side has 0 nodes; `merged` if the two id-sets intersect (a shared node); else `distinct`.

- [ ] **Step 1: Write the failing integration test.** Add to `tests/integration/test_eval_quality.py` a test seeding three pairs — distinct (two different nodes), absent (only one member present), merged (a single node whose name matches member A, and B aliased to that same name). Assert `state` is `distinct`/`absent`/`merged` respectively and `collapsed == (state=="merged")`.

```python
async def test_dedup_v2_distinct_tristate(extract_driver):
    from graph_extract.eval import dedup_report_v2
    from types import SimpleNamespace
    g = "tri"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {group_id:$g, name:'AWS Backup'})", g=g)
        await s.run("CREATE (:Entity {group_id:$g, name:'Azure Backup'})", g=g)   # distinct pair
        await s.run("CREATE (:Entity {group_id:$g, name:'Amazon S3'})", g=g)       # S3 present, Blob absent
        await s.run("CREATE (:Entity {group_id:$g, name:'Shared Vault'})", g=g)    # one node for a merged pair
    labels = SimpleNamespace(
        SHOULD_MERGE=[],
        SHOULD_DISTINCT=[
            ["AWS Backup", "Azure Backup"],
            ["Amazon S3", "Azure Blob Storage"],
            ["Shared Vault", ["Shared Vault", "Also Shared Vault"]],  # B aliases onto the same node
        ],
        VENDOR_TOKENS=["aws", "amazon", "azure"],
    )
    rep = await dedup_report_v2(extract_driver, g, labels)
    states = [e["state"] for e in rep["should_distinct"]]
    assert states == ["distinct", "absent", "merged"]  # order matches SHOULD_DISTINCT
    for e in rep["should_distinct"]:
        assert e["collapsed"] == (e["state"] == "merged")
```

(For the merged case, member A `"Shared Vault"` and member B alias `"Shared Vault"` both resolve to the one seeded node, so their id-sets intersect → `merged`.)

- [ ] **Step 2: Run it, confirm it fails** (`state`/alias unsupported).

Run: `cd /home/openclaw/Documents/Shared/projects/graph-rag && uv run --extra dev pytest tests/integration/test_eval_quality.py -k tristate -v`
Expected: FAIL (KeyError `state` or alias not resolved).

- [ ] **Step 3: Add alias support to `quality_labels.SHOULD_DISTINCT`.** Change the known Azure-immutable-vault member to carry its observed alias; keep the others as plain strings (a plain `str` is treated as a single-form member).

```python
SHOULD_DISTINCT: list[tuple[str | list[str], str | list[str]]] = [
    ("Amazon S3", "Azure Blob Storage"),
    ("AWS Backup", "Azure Backup"),
    ("AWS Backup Vault Lock", ["Azure immutable vault", "Azure Backup Immutable vault"]),
]
```

- [ ] **Step 4: Rewrite the `should_distinct` loop in `dedup_report_v2`** (replace lines 130-137). Resolve members to node-id sets and classify.

```python
        def _forms(member):
            return [member] if isinstance(member, str) else list(member)

        async def _node_ids(forms):
            r = await s.run(
                "MATCH (e:Entity {group_id:$g}) "
                "WHERE toLower(e.name) IN $forms "
                "RETURN collect(DISTINCT elementId(e)) AS ids",
                g=group_id, forms=[f.lower() for f in forms])
            return set((await r.single())["ids"])

        for a, b in labels.SHOULD_DISTINCT:
            a_ids = await _node_ids(_forms(a))
            b_ids = await _node_ids(_forms(b))
            if not a_ids or not b_ids:
                state = "absent"
            elif a_ids & b_ids:
                state = "merged"
            else:
                state = "distinct"
            out["should_distinct"].append({
                "pair": [a, b],
                "state": state,
                "collapsed": state == "merged",
                "a_nodes": len(a_ids), "b_nodes": len(b_ids),
            })
```

- [ ] **Step 5: Run the new test + the existing dedup tests.**

Run: `uv run --extra dev pytest tests/integration/test_eval_quality.py -v`
Expected: PASS (new tristate test + existing dedup_v2/noise tests).

- [ ] **Step 6: Validate against the live graph** (no re-extraction; the 579-entity graph is still in Neo4j).

Run: `uv run --extra dev python -m graph_extract.cli quality-report` then inspect `should_distinct` — expected: no pair reports `state=="merged"`; the S3/Blob and Vault-Lock pairs read `distinct` (alias now resolves) or `absent`, never a false `merged`.

- [ ] **Step 7: Commit.**

```bash
git add src/graph_extract/quality_labels.py src/graph_extract/eval.py tests/integration/test_eval_quality.py
git commit -m "feat(eval): should_distinct tri-state (distinct/absent/merged) + alias tolerance"
```

---

### Task 2: `type_precision` larger sample + parse robustness + retry (Item 3)

**Files:**
- Modify: `src/graph_extract/eval.py` (`_parse_type` ~334-349; `type_precision` ~352-405; sampling cap)
- Modify: `src/graph_extract/cli.py` (bump the `type_precision` sample default to 100)
- Test: `tests/unit/test_eval_parse.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `type_precision(driver, settings, sample)` unchanged signature; internally caps `sample` at the available typed-entity count and retries one unparseable judgment before counting it unparseable. `_parse_type` more robust.

- [ ] **Step 1: Write failing unit tests for `_parse_type`.** Cover: think-block stripped; trailing punctuation/quotes tolerated; a bare final-line type; unknown text → None.

```python
import pytest
from graph_extract.eval import _parse_type

@pytest.mark.parametrize("text,expected", [
    ("<think>maybe Platform or Tool</think>\nTool", "Tool"),
    ("The answer is: **Workload**.", "Workload"),
    ("Region", "Region"),
    ("i cannot decide", None),
    ("", None),
])
def test_parse_type(text, expected):
    assert _parse_type(text) == expected
```

- [ ] **Step 2: Run, confirm current behavior** (some may already pass; the punctuation/`**bold**` case is the likely failure).

Run: `uv run --extra dev pytest tests/unit/test_eval_parse.py -v`
Expected: at least the bold/punctuation case FAILS.

- [ ] **Step 3: Harden `_parse_type`.** Strip markdown emphasis/punctuation before matching; keep last-match semantics.

```python
def _parse_type(content: str | None) -> str | None:
    if content is None:
        return None
    text = _THINK_TAG_RE.sub("", content.strip())
    text = text.replace("*", "").replace("`", "")
    matches = _TYPE_NAME_RE.findall(text)
    if not matches:
        return None
    return _CANON_TYPE[matches[-1].lower()]
```

- [ ] **Step 4: Add a single retry-on-unparseable + sample cap in `type_precision`.** Cap the sample query at available typed entities; when the first judge answer is unparseable, make one terse retry before counting it unparseable. Extract the judge call into a small local helper so the retry reuses it. Keep `max_tokens=600`, `temperature=0`.

```python
    # cap requested sample at what's available (avoids over-requesting on small graphs)
    async with driver.session() as s:
        cap = (await (await s.run(
            "MATCH (e:Entity {group_id:$g}) WHERE size([l IN labels(e) WHERE l<>'Entity'])>0 "
            "RETURN count(e) AS c", g=settings.group_id)).single())["c"]
        n = min(sample, cap)
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) "
            "UNWIND labels(e) AS l WITH e, l WHERE l <> 'Entity' "
            "WITH e, collect(l)[0] AS type "
            "RETURN e.name AS name, type AS type ORDER BY rand() LIMIT $n",
            g=settings.group_id, n=n)
        rows = [dict(rec) async for rec in r]

    async def _judge(name: str, terse: bool) -> str:
        prompt = _TYPE_JUDGE_PROMPT.format(definitions=definitions, name=name)
        if terse:
            prompt += "\n\nAnswer with exactly ONE type word, nothing else."
        resp = await client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=600,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content or ""
```

Then in the loop: `raw = await _judge(row["name"], terse=False); judged = _parse_type(raw)` and `if judged is None: raw = await _judge(row["name"], terse=True); judged = _parse_type(raw)` before the `if judged is None: unparseable += 1` accounting.

- [ ] **Step 5: Run unit tests + confirm nothing else broke.**

Run: `uv run --extra dev pytest tests/unit/test_eval_parse.py -v && uv run --extra dev pytest -q -m "not live"`
Expected: PASS.

- [ ] **Step 6: Bump CLI default sample to 100.** In `cli.py`, change the `type_precision`/`quality-*` sample option default constant from 40 to `100`.

- [ ] **Step 7: Live validation** (existing graph). Run `type_precision` twice and record the spread + unparseable rate.

Run: `uv run --extra dev python -m graph_extract.cli quality-baseline` is NOT needed; instead a quick two-run check via a one-off script or `quality-report` twice. Expected: unparseable rate materially below ~20%; the two overall numbers closer than the 0.31 spread seen at N=40 (record actuals; no hard threshold).

- [ ] **Step 8: Commit.**

```bash
git add src/graph_extract/eval.py src/graph_extract/cli.py tests/unit/test_eval_parse.py
git commit -m "feat(eval): type_precision larger sample cap + parse hardening + retry-on-unparseable"
```

---

### Task 3: `region_names.py` gazetteer + helpers (Item 1a)

**Files:**
- Create: `src/graph_extract/region_names.py`
- Test: `tests/unit/test_region_names.py`

**Interfaces:**
- Produces: `REGION_NAMES: frozenset[str]` (normalised, lowercased, whitespace-collapsed); `normalize_region(name: str) -> str` (lowercase, collapse spaces, strip a leading vendor word `aws|amazon|azure|microsoft|google|gcp|oracle|oci|ibm|alibaba` and a trailing `region`/`regions`); `is_region(name: str) -> bool` (`normalize_region(name) in REGION_NAMES`).

- [ ] **Step 1: Write failing unit tests** enumerating must-match regions across providers/forms and must-NOT-match domain terms.

```python
import pytest
from graph_extract.region_names import is_region

@pytest.mark.parametrize("name", [
    "India West", "Israel Central", "East Asia", "West Europe", "North Europe",
    "South Central US", "Germany West Central", "Australia East", "UK South",
    "Asia Pacific (Malaysia)", "Asia Pacific (Tokyo)", "US East (N. Virginia)",
    "us-east-1", "eu-west-2", "ap-southeast-4", "il-central-1", "us-gov-west-1",
    "us-central1", "europe-west4", "asia-northeast1",
    "Azure East US", "AWS us-east-1",           # vendor-prefixed forms
])
def test_is_region_true(name):
    assert is_region(name) is True

@pytest.mark.parametrize("name", [
    "immutability", "Amazon S3", "Azure Blob Storage", "soft delete",
    "cross-region copy", "AWS Backup", "recovery point", "retention policy",
    "Central India",  # <- this IS a region; MOVE to the true-list. (placeholder note)
])
def test_is_region_false(name):
    assert is_region(name) is False
```

(The implementer MUST fix the list: `Central India` is a real Azure region → it belongs in the true-list; the false-list is only genuine domain terms. The point of the test is: every KEEP domain term is False, every enumerated region is True.)

- [ ] **Step 2: Run, confirm fail** (module missing).

Run: `uv run --extra dev pytest tests/unit/test_region_names.py -v`
Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `region_names.py`.** A dated module. Build `REGION_NAMES` from the providers' published region tables (AWS regions codes + display names; Azure display names; GCP region codes; common OCI/IBM/Alibaba). Store each entry pre-normalised. **Only full multi-word names or codes — no bare tokens** (`central`, `east`, `standard`). Provide `normalize_region`/`is_region`.

```python
"""Region gazetteer — single source of truth for 'what is a region'.
Snapshot: cloud provider region tables as of 2026-07. Static data; extend via PR.
Entries are normalised (lowercased, whitespace-collapsed). NO bare directional
tokens — only full region display names or provider region codes — so domain
terms are never shadowed.
"""
from __future__ import annotations
import re

_VENDOR_PREFIX = re.compile(
    r"^(aws|amazon|azure|microsoft|google|gcp|oracle|oci|ibm|alibaba)\s+", re.IGNORECASE)
_TRAIL_REGION = re.compile(r"\s+regions?$", re.IGNORECASE)

def normalize_region(name: str) -> str:
    n = " ".join(name.strip().split()).lower()
    n = _VENDOR_PREFIX.sub("", n)
    n = _TRAIL_REGION.sub("", n)
    return n.strip()

_RAW: tuple[str, ...] = (
    # Azure (display names)
    "east us", "east us 2", "west us", "west us 2", "west us 3", "central us",
    "north central us", "south central us", "west central us",
    "canada central", "canada east", "brazil south", "brazil southeast",
    "north europe", "west europe", "uk south", "uk west", "france central",
    "germany west central", "germany north", "switzerland north", "norway east",
    "sweden central", "poland central", "italy north", "spain central",
    "east asia", "southeast asia", "australia east", "australia southeast",
    "australia central", "japan east", "japan west", "korea central",
    "central india", "south india", "west india", "jio india west",
    "israel central", "uae north", "qatar central", "south africa north",
    # AWS (codes)
    "us-east-1", "us-east-2", "us-west-1", "us-west-2", "ca-central-1", "ca-west-1",
    "eu-west-1", "eu-west-2", "eu-west-3", "eu-central-1", "eu-central-2",
    "eu-north-1", "eu-south-1", "eu-south-2", "ap-south-1", "ap-south-2",
    "ap-southeast-1", "ap-southeast-2", "ap-southeast-3", "ap-southeast-4",
    "ap-northeast-1", "ap-northeast-2", "ap-northeast-3", "ap-east-1",
    "sa-east-1", "me-south-1", "me-central-1", "af-south-1", "il-central-1",
    "us-gov-west-1", "us-gov-east-1", "cn-north-1", "cn-northwest-1",
    # AWS (display names)
    "us east (n. virginia)", "us east (ohio)", "us west (n. california)",
    "us west (oregon)", "asia pacific (tokyo)", "asia pacific (singapore)",
    "asia pacific (malaysia)", "eu (ireland)", "canada (central)",
    # GCP (codes)
    "us-central1", "us-east1", "us-east4", "us-west1", "us-west2",
    "europe-west1", "europe-west2", "europe-west4", "europe-north1",
    "asia-east1", "asia-northeast1", "asia-southeast1", "southamerica-east1",
    "me-central1", "me-central2", "africa-south1", "australia-southeast1",
    # ... implementer completes each provider's current table.
)
REGION_NAMES: frozenset[str] = frozenset(normalize_region(x) for x in _RAW)

def is_region(name: str) -> bool:
    return normalize_region(name) in REGION_NAMES
```

- [ ] **Step 4: Run tests to green** (after moving `Central India` etc. to the true-list and completing the tables).

Run: `uv run --extra dev pytest tests/unit/test_region_names.py -v`
Expected: PASS. Also `uv run --extra dev ruff check src/graph_extract/region_names.py && uv run --extra dev mypy src/graph_extract/region_names.py`.

- [ ] **Step 5: KEEP-guard cross-check.** Assert (in the same test file) that `is_region` returns False for every name in `noise_filter` KEEP-style domain terms and every `quality_labels.SHOULD_MERGE`/`SHOULD_DISTINCT` concept name — proving the gazetteer shadows no domain term.

- [ ] **Step 6: Commit.**

```bash
git add src/graph_extract/region_names.py tests/unit/test_region_names.py
git commit -m "feat(extract): region_names gazetteer (single source of truth for regions)"
```

---

### Task 4: ontology-v5 — `Region` type + `AvailableIn` edge + instruction rework + canonicalization (Items 1b + 4)

**Files:**
- Modify: `src/graph_extract/ontology.py`
- Test: `tests/unit/test_ontology.py`

**Interfaces:**
- Produces: `ENTITY_TYPES` gains `"Region"`; `EDGE_TYPES` gains `"AvailableIn"`; `EDGE_TYPE_MAP` gains `("Product","Region")`, `("Capability","Region")`, `("Workload","Region")` each `["AvailableIn"]`. `EXTRACTION_INSTRUCTIONS` no longer excludes regions; adds availability-capture + canonicalization lines.

- [ ] **Step 1: Write failing ontology tests.** Assert `Region` in `ENTITY_TYPES`; `AvailableIn` in `EDGE_TYPES`; the three `AvailableIn` pairs present in `EDGE_TYPE_MAP`; every `EDGE_TYPE_MAP` pair still references valid types/edges; `EXTRACTION_INSTRUCTIONS` mentions availability ("available in") and NO LONGER contains the region-noise-exclusion phrase ("specific geographic regions").

```python
def test_region_type_and_edge():
    from graph_extract import ontology as o
    assert "Region" in o.ENTITY_TYPES
    assert "AvailableIn" in o.EDGE_TYPES
    for pair in [("Product","Region"),("Capability","Region"),("Workload","Region")]:
        assert "AvailableIn" in o.EDGE_TYPE_MAP[pair]
    for (a,b),edges in o.EDGE_TYPE_MAP.items():
        assert a in o.ENTITY_TYPES and b in o.ENTITY_TYPES
        assert all(e in o.EDGE_TYPES for e in edges)
    instr = o.EXTRACTION_INSTRUCTIONS.lower()
    assert "available in" in instr
    assert "specific geographic regions" not in instr  # regions are now extracted
```

- [ ] **Step 2: Run, confirm fail.**

Run: `uv run --extra dev pytest tests/unit/test_ontology.py -k region -v`
Expected: FAIL.

- [ ] **Step 3: Add the `Region` class + `AvailableIn` class** and register them. `Region` docstring: *"A specific geographic or cloud region, or a jurisdiction, where a product operates or stores backup data (e.g. Germany West Central, East US, us-east-1, Germany, EU) — NOT a Platform (a region runs on a platform), NOT a generic relative term (primary/secondary region are Concepts), NOT a redundancy tier (LRS/ZRS are Concepts)."* `AvailableIn` docstring: *"A product, capability, or workload is available in, operates in, or stores data in a region."* Add `Region` to `ENTITY_TYPES` (after `Platform` or `Concept`), `AvailableIn` to `EDGE_TYPES`, and the three pairs to `EDGE_TYPE_MAP`.

- [ ] **Step 4: Rework `EXTRACTION_INSTRUCTIONS`** (in `ontology.py:70-115`):
  - In the "Do NOT extract as entities" block, **remove** `specific geographic regions or availability zones ("Australia East", "us-east-1", "West Central US", "primary region")` — keep only `time zones (UTC)` and the non-region noise. Regions are now first-class.
  - In "Type boundaries", add: *"A specific region/jurisdiction (Germany West Central, East US, us-east-1, Germany) is a Region, not a Platform; a generic 'primary/secondary region' remains a Concept."*
  - Add an availability line: *"DO capture availability/residency statements (\"available in <region>\", \"data resides in\", \"supported regions\", \"not available in\") as AvailableIn facts from the product, capability, or workload to the region."*
  - **Canonicalization (Item 4):** extend the Capabilities/Concepts canonical lists so the observed fragmenting cases converge — e.g. add `"immutability" (not "immutable vault"/"immutable backups"/"immutability policy")` and `"retention policy" (not "retention"/"retention rule"/"retention settings")`.

- [ ] **Step 5: Run ontology tests + full non-live suite.**

Run: `uv run --extra dev pytest tests/unit/test_ontology.py -v && uv run --extra dev pytest -q -m "not live"`
Expected: PASS. Then `uv run --extra dev ruff check src/graph_extract/ontology.py && uv run --extra dev mypy src/graph_extract`.

- [ ] **Step 6: Commit.**

```bash
git add src/graph_extract/ontology.py tests/unit/test_ontology.py
git commit -m "feat(extract): ontology-v5 (Region type + AvailableIn edge, region/availability instructions, canonicalization)"
```

---

### Task 5: `retype_region_entities` deterministic corrector + `cleanup` CLI (Item 1c)

**Files:**
- Modify: `src/graph_extract/graph_cleanup.py`
- Modify: `src/graph_extract/cli.py`
- Test: `tests/integration/test_graph_cleanup.py`

**Interfaces:**
- Consumes: `region_names.is_region`.
- Produces: `async def retype_region_entities(driver, group_id) -> dict` returning `{"scanned": int, "retyped": int, "retyped_names": list[str]}`. Relabels any `:Entity` whose `is_region(name)` is True and that is not already `:Region`: remove its current custom type label(s) (any label other than `Entity`/`Region`) and `SET e:Region`. Preserves `:Entity` and all edges. CLI `cleanup` command runs `prune_noise_entities` then `retype_region_entities` and `_dump`s both audits.

- [ ] **Step 1: Write the failing integration test.** Seed a graph: a mistyped region (`(:Entity:Platform {name:'Germany West Central'})`), a correct non-region (`(:Entity:Workload {name:'Amazon S3'})`), and an already-correct region (`(:Entity:Region {name:'East US'})`). After `retype_region_entities`: the Platform region is now `:Region` and no longer `:Platform`; the Workload is untouched; the already-Region is unchanged; count `retyped == 1`.

```python
async def test_retype_region_entities(extract_driver):
    from graph_extract.graph_cleanup import retype_region_entities
    g = "rgn"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity:Platform {group_id:$g, name:'Germany West Central'})", g=g)
        await s.run("CREATE (:Entity:Workload {group_id:$g, name:'Amazon S3'})", g=g)
        await s.run("CREATE (:Entity:Region {group_id:$g, name:'East US'})", g=g)
    res = await retype_region_entities(extract_driver, g)
    assert res["retyped"] == 1 and res["retyped_names"] == ["Germany West Central"]
    async with extract_driver.session() as s:
        lbls = {r["name"]: set(r["l"]) async for r in await s.run(
            "MATCH (e:Entity {group_id:$g}) RETURN e.name AS name, labels(e) AS l", g=g)}
    assert "Region" in lbls["Germany West Central"] and "Platform" not in lbls["Germany West Central"]
    assert lbls["Amazon S3"] == {"Entity", "Workload"}
    assert "Region" in lbls["East US"]
```

- [ ] **Step 2: Run, confirm fail.**

Run: `uv run --extra dev pytest tests/integration/test_graph_cleanup.py -k retype -v`
Expected: FAIL (function missing).

- [ ] **Step 3: Implement `retype_region_entities`.**

```python
from graph_extract.region_names import is_region

async def retype_region_entities(driver: AsyncDriver, group_id: str) -> dict:
    async with driver.session() as s:
        r = await s.run(
            "MATCH (e:Entity {group_id:$g}) "
            "RETURN elementId(e) AS id, e.name AS name, "
            "[l IN labels(e) WHERE l<>'Entity' AND l<>'Region'] AS types, "
            "'Region' IN labels(e) AS is_region_lbl",
            g=group_id)
        rows = [dict(rec) async for rec in r]
        targets = [row for row in rows
                   if row["name"] and is_region(row["name"]) and not row["is_region_lbl"]]
        for row in targets:
            # remove the wrong custom type labels, add :Region (labels can't be parameterised)
            remove = "".join(f" REMOVE e:`{t}`" for t in row["types"])
            await s.run(
                f"MATCH (e:Entity {{group_id:$g}}) WHERE elementId(e)=$id SET e:Region{remove}",
                g=group_id, id=row["id"])
    names = sorted(row["name"] for row in targets)
    return {"scanned": len(rows), "retyped": len(targets), "retyped_names": names}
```

(Label names come from `labels(e)`, not user input, so the f-string interpolation is safe; still backtick-quote them.)

- [ ] **Step 4: Run the test to green.**

Run: `uv run --extra dev pytest tests/integration/test_graph_cleanup.py -v`
Expected: PASS (retype + existing prune tests).

- [ ] **Step 5: Add the `cleanup` CLI command** in `cli.py`, mirroring the existing `eval` command wiring (`_build_driver`, close in `finally`). It runs `prune_noise_entities` then `retype_region_entities` and `_dump`s a combined `{"prune": ..., "retype": ...}`.

- [ ] **Step 6: Verify CLI wiring** (no live calls needed).

Run: `uv run --extra dev python -c "from typer.testing import CliRunner; from graph_extract.cli import app; print(CliRunner().invoke(app, ['--help']).output)"` — confirm `cleanup` appears. Then `uv run --extra dev ruff check src/graph_extract/graph_cleanup.py src/graph_extract/cli.py && uv run --extra dev mypy src/graph_extract`.

- [ ] **Step 7: Commit.**

```bash
git add src/graph_extract/graph_cleanup.py src/graph_extract/cli.py tests/integration/test_graph_cleanup.py
git commit -m "feat(extract): retype_region_entities corrector + cleanup CLI (prune + retype)"
```

---

### Task 6: Live validation — sample then full re-run + report update (Items 1 & 4 end-to-end)

**Files:**
- Modify: `docs/superpowers/slice-2b-quality-report.md` (append a follow-ups section) OR create `docs/superpowers/slice-2b-followups-report.md`.

This task is controller-run (live extraction). No new unit tests.

- [ ] **Step 1: Sample re-run with ontology-v5.**

```bash
cd /home/openclaw/Documents/Shared/projects/graph-rag
uv run --extra dev python scripts/reset_semantic_layer.py
PILOT_IDS_FILE=scripts/pilot-ids-sample.txt uv run --extra dev python scripts/run_pilot.py
uv run --extra dev python -m graph_extract.cli cleanup   # prune + retype
```

- [ ] **Step 2: Verify the region model works on the sample.** Query for `:Region` nodes, `AvailableIn` facts, and specifically that a Germany/EU region node exists if the sample mentions one; confirm no region remains typed `:Platform` (post-retype); confirm `immutability`/`retention policy` resolve to canonical names (Item 4). Record counts.

```bash
uv run --extra dev python -c "
import asyncio; from neo4j import AsyncGraphDatabase as G; from graph_extract.config import get_extract_settings as S
async def m():
    s=S.__wrapped__(); d=G.driver(s.neo4j_uri,auth=(s.neo4j_user,s.neo4j_password))
    async with d.session() as x:
        for q,lbl in [
          (\"MATCH (r:Region {group_id:'backup-docs'}) RETURN count(r) AS c\",'Region nodes'),
          (\"MATCH ()-[e:RELATES_TO {group_id:'backup-docs'}]->() WHERE e.name='AvailableIn' RETURN count(e) AS c\",'AvailableIn facts'),
          (\"MATCH (r:Region:Platform {group_id:'backup-docs'}) RETURN count(r) AS c\",'region-still-Platform (want 0)')]:
            print(lbl, (await (await x.run(q)).single())['c'])
    await d.close()
asyncio.run(m())"
```

If regions aren't being extracted well or AvailableIn is sparse, iterate the ontology-v5 instructions (Task 4) on the sample before the full run.

- [ ] **Step 3: Full 39-article re-run** for the headline.

```bash
uv run --extra dev python scripts/reset_semantic_layer.py
uv run --extra dev python scripts/run_pilot.py
uv run --extra dev python -m graph_extract.cli cleanup
```

- [ ] **Step 4: Validate the target query traversal.** Confirm a path exists for "vendors offering a SaaS-backup capability available in a Germany region": e.g. `MATCH (p:Product)-[a:RELATES_TO {name:'AvailableIn'}]->(r:Region) WHERE toLower(r.name) CONTAINS 'germany' RETURN p.name, r.name` returns rows (or document that the 39-article AWS/Azure pilot corpus lacks Germany-specific availability text, so validate with whatever region IS present, e.g. a US/Europe region).

- [ ] **Step 5: Re-check the metrics** (`quality-report`) and the type histogram — confirm `Region` is populated, `Platform` no longer holds regions, `should_distinct` shows no false `merged`, and record the steadier `type_precision`.

- [ ] **Step 6: Write the follow-ups report** — a short before/after for the four items (tri-state fixed the false collapses; type_precision steadier + lower unparseable; Region type + AvailableIn counts + the Germany/region traversal; canonicalization effect). Commit.

```bash
git add docs/superpowers/slice-2b-followups-report.md
git commit -m "docs: 2b-quality follow-ups report (region modelling, metric fixes)"
```

---

## Notes for execution

- Tasks 1–5 are subagent-implementable (deterministic, unit/integration-tested). Task 6 is controller-run live work (extraction + judge), like the slice's Task 7.
- Order: 1 and 2 (metric fixes, no re-extraction) can land first and be validated immediately against the existing graph. 3 → 4 → 5 build the region model. 6 validates 1/4/5 end-to-end and produces the headline. (2 and 3 are independent; either order.)
- Each ontology/instruction change is prompt-sensitive — Task 6 may loop Task 4's wording on the sample before the full run (expected, cheap).
