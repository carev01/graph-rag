# Maintenance Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the residual-staleness sweep (folding in Minor #6), structural↔semantic `SAME_AS` reconciliation, and a `dedup_report_v2` silent-merge signal (Minor #7).

**Architecture:** All in `graph_extract`, Cypher-only maintenance jobs over the Neo4j graph (no LLM). New `staleness_sweep.py`, `reconcile.py`, `vendor_aliases.py`; extend `eval.py` and `cli.py`.

**Tech Stack:** Python 3.12, Neo4j 5.26 (async), pytest + testcontainers (Neo4j), ruff, mypy.

## Global Constraints

- **Temporal policy:** the sweep MARKS facts invalid (`invalid_at` + `expired_by_sweep=true`); it never deletes facts/episodes.
- **#6 — sweep keys on `HAS_EPISODE` linkage, NOT the `superseded` flag.** Dead episode = `removed=true` OR no `HAS_EPISODE` edge from a non-removed `:Article`. A live-but-`superseded`-flagged episode (shrink-restore) keeps its edge → alive → its facts are NOT expired.
- **Reconciliation links, never merges** (design-decision #5): `MERGE (:Vendor|:Product)-[:SAME_AS]->(:Entity)`, idempotent, additive; conservative curated matching, never links noise.
- **Determinism:** all three jobs are pure Cypher / data — NO LLM calls.
- Never touch graphiti's own `invalid_at` (only expire facts where `invalid_at IS NULL`).
- Secrets only in untracked `.env`; DB tests use testcontainers; ruff/mypy clean.

---

## File Structure

- `src/graph_extract/staleness_sweep.py` — **new:** `sweep_stale_facts(driver, group_id) -> dict`.
- `src/graph_extract/vendor_aliases.py` — **new:** `VENDOR_ALIASES` map + `normalize`/`accepted_forms`.
- `src/graph_extract/reconcile.py` — **new:** `reconcile_same_as(driver, group_id) -> dict`.
- `src/graph_extract/eval.py` — **modify:** `dedup_report_v2` gains `silent_merge_suspects`.
- `src/graph_extract/cli.py` — **modify:** `sweep` + `reconcile` commands.
- Tests: `tests/integration/test_staleness_sweep.py` (new), `tests/integration/test_reconcile.py` (new), `tests/unit/test_vendor_aliases.py` (new), `tests/integration/test_eval_quality.py` (extend).

---

### Task 1: Residual-staleness sweep (#6)

**Files:**
- Create: `src/graph_extract/staleness_sweep.py`
- Modify: `src/graph_extract/cli.py` (`sweep` command)
- Test: `tests/integration/test_staleness_sweep.py`

**Interfaces:** `async def sweep_stale_facts(driver, group_id) -> dict` → `{"scanned": int, "expired": int, "expired_sample": list[str]}`.

- [ ] **Step 1: Write failing integration tests** (Neo4j testcontainer — use the `extract_driver` fixture from `tests/integration/test_provenance_rekey.py`). Seed by hand (no LLM); a "fact" is a `RELATES_TO` edge with an `episodes` list; a "supporting episode" is an `:Episodic {uuid}` in that list, optionally linked `(:Article)-[:HAS_EPISODE]->(:Episodic)`.

```python
async def _fact(s, g, uuid, eps):
    # a RELATES_TO fact between two throwaway endpoint nodes, carrying an episodes list
    await s.run(
        "CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:$u, episodes:$eps}]->(y:Entity)",
        g=g, u=uuid, eps=eps)

async def test_sweep_expires_fact_with_no_live_episodes(extract_driver):
    from graph_extract.staleness_sweep import sweep_stale_facts
    g = "swp1"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Episodic {uuid:'e_dead'})")   # exists, NO HAS_EPISODE edge -> dead
        await _fact(s, g, "f1", ["e_dead"])
    res = await sweep_stale_facts(extract_driver, g)
    async with extract_driver.session() as s:
        row = await (await s.run(
            "MATCH ()-[f:RELATES_TO {uuid:'f1'}]->() RETURN f.invalid_at AS inv, f.expired_by_sweep AS ex")).single()
    assert res["expired"] == 1 and row["inv"] is not None and row["ex"] is True

async def test_sweep_keeps_fact_with_live_but_superseded_episode(extract_driver):
    # THE #6 CASE: episode has a HAS_EPISODE edge (alive) but superseded=true -> fact NOT expired
    from graph_extract.staleness_sweep import sweep_stale_facts
    g = "swp2"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Article {removed:false})-[:HAS_EPISODE]->(:Episodic {uuid:'e_live', superseded:true})")
        await _fact(s, g, "f2", ["e_live"])
    await sweep_stale_facts(extract_driver, g)
    async with extract_driver.session() as s:
        inv = (await (await s.run("MATCH ()-[f:RELATES_TO {uuid:'f2'}]->() RETURN f.invalid_at AS i")).single())["i"]
    assert inv is None   # kept alive by the HAS_EPISODE edge, superseded flag ignored

async def test_sweep_expires_removed_episode_fact(extract_driver): ...      # (:Article)-[:HAS_EPISODE]->(:Episodic{removed:true}) -> dead
async def test_sweep_ignores_already_invalid(extract_driver): ...           # fact with invalid_at set -> untouched
async def test_sweep_keeps_fact_with_any_live_episode(extract_driver): ...  # episodes:['e_live','e_dead'] -> kept
```
(Use unique `group_id`s per test so the shared module container stays order-independent. The remaining `...` tests follow the same pattern as the two shown.)

- [ ] **Step 2: Run, confirm fail** (module missing). `uv run --extra dev pytest tests/integration/test_staleness_sweep.py -v`.

- [ ] **Step 3: Implement `sweep_stale_facts`:**

```python
from __future__ import annotations
from neo4j import AsyncDriver

_SWEEP = """
MATCH ()-[f:RELATES_TO {group_id:$g}]->()
WHERE f.invalid_at IS NULL AND f.episodes IS NOT NULL AND size(f.episodes) > 0
WITH f, f.episodes AS eps
CALL {
  WITH eps
  UNWIND eps AS epu
  MATCH (e:Episodic {uuid: epu})
  OPTIONAL MATCH (a:Article)-[:HAS_EPISODE]->(e) WHERE coalesce(a.removed,false)=false
  WITH e, count(a) AS live_links
  RETURN sum(CASE WHEN coalesce(e.removed,false)=false AND live_links>0 THEN 1 ELSE 0 END) AS alive
}
WITH f WHERE alive = 0
SET f.invalid_at = datetime(), f.expired_by_sweep = true
RETURN count(f) AS expired, collect(f.uuid)[..20] AS sample
"""

async def sweep_stale_facts(driver: AsyncDriver, group_id: str) -> dict:
    async with driver.session() as s:
        scanned = (await (await s.run(
            "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.invalid_at IS NULL "
            "RETURN count(f) AS c", g=group_id)).single())["c"]
        rec = await (await s.run(_SWEEP, g=group_id)).single()
    return {"scanned": scanned, "expired": rec["expired"], "expired_sample": rec["sample"]}
```
(Note: the CALL subquery imports `eps` and returns `alive`; `f` stays bound across the CALL. If a Neo4j 5.26 syntax nuance requires `CALL(eps) { ... }` or a variable-scope tweak, adjust — verify against the container.)

- [ ] **Step 4: Run tests green** (all 5, esp. the #6 case). Gate on `staleness_sweep.py`.

- [ ] **Step 5: Add the `sweep` CLI command** in `cli.py`, mirroring the `cleanup` command (`_build_driver(settings)`, run, `_dump`, close in `finally`).

- [ ] **Step 6: Verify wiring** — `sweep` appears in `--help`; ruff + mypy clean.

- [ ] **Step 7: Commit.** `git add src/graph_extract/staleness_sweep.py src/graph_extract/cli.py tests/integration/test_staleness_sweep.py && git commit -m "feat(extract): residual-staleness sweep (expire facts with no live episodes; #6-safe)"`

---

### Task 2: `vendor_aliases` data module

**Files:**
- Create: `src/graph_extract/vendor_aliases.py`
- Test: `tests/unit/test_vendor_aliases.py`

**Interfaces:** `VENDOR_ALIASES: dict[str, frozenset[str]]` (canonical structural name → accepted normalized semantic forms); `normalize(name) -> str`; `accepted_forms(structural_name) -> set[str]` (the structural name's own normalized form + its alias forms).

- [ ] **Step 1: Write failing unit tests.**

```python
from graph_extract.vendor_aliases import normalize, accepted_forms
def test_aws_forms():
    forms = accepted_forms("AWS")
    assert "amazon web services" in forms and "aws" in forms
def test_microsoft_forms():
    assert "azure" in accepted_forms("Microsoft")
def test_normalize():
    assert normalize("  Amazon Web Services ") == "amazon web services"
def test_unknown_vendor_forms_is_just_itself():
    assert accepted_forms("Veeam") == {"veeam"}   # no aliases yet -> its own normalized name
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement:**

```python
"""Structural-vendor -> accepted semantic-name aliases (snapshot 2026-07).
Static, conservative: only high-confidence equivalences so reconciliation
never links a structural node to a wrong/noise semantic entity."""
from __future__ import annotations

def normalize(name: str) -> str:
    return " ".join(name.strip().split()).lower()

# canonical structural name -> extra accepted semantic surface forms (pre-normalised)
VENDOR_ALIASES: dict[str, frozenset[str]] = {
    "AWS": frozenset({"aws", "amazon web services", "amazon"}),
    "Microsoft": frozenset({"microsoft", "azure", "microsoft azure"}),
    # extend per vendor as the corpus widens (Veeam, Commvault, Rubrik, ...)
}

def accepted_forms(structural_name: str) -> set[str]:
    forms = {normalize(structural_name)}
    forms |= set(VENDOR_ALIASES.get(structural_name, frozenset()))
    return forms
```

- [ ] **Step 4: KEEP-guard test** — assert no alias form of one vendor equals the normalized name (or any alias) of a *different* vendor in the map (no cross-vendor collision).

- [ ] **Step 5: Run green + gate.**

- [ ] **Step 6: Commit.** `git add src/graph_extract/vendor_aliases.py tests/unit/test_vendor_aliases.py && git commit -m "feat(extract): vendor alias map for SAME_AS reconciliation"`

---

### Task 3: `SAME_AS` reconciliation

**Files:**
- Create: `src/graph_extract/reconcile.py`
- Modify: `src/graph_extract/cli.py` (`reconcile` command)
- Test: `tests/integration/test_reconcile.py`

**Interfaces:** `async def reconcile_same_as(driver, group_id) -> dict` → `{"structural_scanned": int, "linked": int, "unmatched_structural": list[str]}`. Consumes `vendor_aliases` (Task 2).

- [ ] **Step 1: Write failing integration tests** (Neo4j testcontainer). Seed structural `(:Vendor {name:'AWS'})` (NOT `:Entity`) + semantic `(:Entity:Vendor {group_id:g, name:'Amazon Web Services'})` + a noise `(:Entity:Vendor {group_id:g, name:'vaults'})` + an unmatched structural `(:Vendor {name:'Veeam'})`.

```python
async def test_reconcile_links_aliased_vendor(extract_driver):
    from graph_extract.reconcile import reconcile_same_as
    g = "rec"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Vendor {name:'AWS'})")
        await s.run("CREATE (:Entity:Vendor {group_id:$g, name:'Amazon Web Services'})", g=g)
        await s.run("CREATE (:Entity:Vendor {group_id:$g, name:'vaults'})", g=g)   # noise
        await s.run("CREATE (:Vendor {name:'Veeam'})")                              # unmatched
    res = await reconcile_same_as(extract_driver, g)
    async with extract_driver.session() as s:
        n = (await (await s.run(
            "MATCH (:Vendor {name:'AWS'})-[:SAME_AS]->(e:Entity {name:'Amazon Web Services'}) "
            "RETURN count(*) AS c")).single())["c"]
    assert n == 1 and res["linked"] >= 1
    assert "Veeam" in res["unmatched_structural"]
    # noise not linked
    async with extract_driver.session() as s:
        m = (await (await s.run("MATCH (:Vendor)-[:SAME_AS]->(e:Entity {name:'vaults'}) RETURN count(*) AS c")).single())["c"]
    assert m == 0

async def test_reconcile_is_idempotent(extract_driver): ...   # run twice -> still 1 SAME_AS
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement `reconcile_same_as`** — for each structural `:Vendor`/`:Product` (a node with that label but NOT `:Entity`), compute `accepted_forms(name)`, find semantic `:Entity` (same label kind, group_id) whose `normalize(name)` is in the forms, `MERGE` the `SAME_AS`. Track linked count + unmatched (structural nodes that matched nothing).

```python
from __future__ import annotations
from neo4j import AsyncDriver
from graph_extract.vendor_aliases import accepted_forms, normalize

async def reconcile_same_as(driver: AsyncDriver, group_id: str) -> dict:
    linked = 0
    unmatched: list[str] = []
    async with driver.session() as s:
        for kind in ("Vendor", "Product"):
            r = await s.run(f"MATCH (v:{kind}) WHERE NOT v:Entity RETURN elementId(v) AS id, v.name AS name")
            structs = [dict(rec) async for rec in r]
            for st in structs:
                forms = [f for f in accepted_forms(st["name"])]
                r = await s.run(
                    f"MATCH (v:{kind}) WHERE elementId(v)=$id "
                    f"MATCH (e:Entity:{kind} {{group_id:$g}}) WHERE toLower(e.name) IN $forms "
                    "MERGE (v)-[:SAME_AS]->(e) RETURN count(e) AS c",
                    id=st["id"], g=group_id, forms=forms)
                c = (await r.single())["c"]
                linked += c
                if c == 0:
                    unmatched.append(st["name"])
    return {"structural_scanned": len(structs), "linked": linked,
            "unmatched_structural": sorted(set(unmatched))}
```
(Fix `structural_scanned` to total across both kinds — accumulate a counter rather than `len(structs)` of the last kind.)

- [ ] **Step 4: Run tests green** (link, noise-not-linked, unmatched, idempotent). Gate.

- [ ] **Step 5: Add the `reconcile` CLI command** (mirror `cleanup`/`sweep`).

- [ ] **Step 6: Verify wiring + commit.** `git add src/graph_extract/reconcile.py src/graph_extract/cli.py tests/integration/test_reconcile.py && git commit -m "feat(extract): structural<->semantic SAME_AS reconciliation (alias-matched, link-not-merge)"`

---

### Task 4: `dedup_report_v2` silent-merge detection (#7)

**Files:**
- Modify: `src/graph_extract/eval.py` (`dedup_report_v2`)
- Test: `tests/integration/test_eval_quality.py`

**Interfaces:** `dedup_report_v2` output gains `silent_merge_suspects: {"names": [...], "count": int}`.

- [ ] **Step 1: Write the failing integration test** — seed a `SHOULD_DISTINCT` member node with cross-vendor episode support (a node whose name matches a member, with `MENTIONS` from episodes whose articles span two vendors — reuse the seeding pattern the existing `test_dedup_report_v2` / `_cross_vendor_names` tests use; read them first). Assert the node appears in `silent_merge_suspects`; a cleanly-distinct pair yields none.

- [ ] **Step 2: Run, confirm fail** (`silent_merge_suspects` KeyError).

- [ ] **Step 3: Implement** — read the current `dedup_report_v2` (`eval.py:110-165`). It already computes `cross_vendor_lower` (from `_cross_vendor_names`) and has `_forms(member)`. After the `should_distinct` loop, add:

```python
    silent = sorted({
        form_name
        for a, b in labels.SHOULD_DISTINCT
        for member in (a, b)
        for form_name in _forms(member)
        if form_name.lower() in cross_vendor_lower
    })
    out["silent_merge_suspects"] = {"names": silent, "count": len(silent)}
```
(A labelled-distinct member with cross-vendor episode support is a suspected silent merge. Deterministic; reuses the existing `cross_vendor_lower` set — no new query.)

- [ ] **Step 4: Run tests green** (new + existing dedup tests). Gate on `eval.py`.

- [ ] **Step 5: Commit.** `git add src/graph_extract/eval.py tests/integration/test_eval_quality.py && git commit -m "feat(eval): dedup_report_v2 silent_merge_suspects (labelled-pair cross-vendor signal, #7)"`

---

## Self-Review Notes

- **Spec coverage:** sweep (+#6) → T1; alias data → T2; reconcile → T3; silent-merge (#7) → T4. All §-components covered.
- **Deps:** T3 needs T2 (`accepted_forms`). T1, T4 independent. Order: T1, T2, T3, T4.
- **Type consistency:** `sweep_stale_facts`, `reconcile_same_as`, `accepted_forms`/`normalize`/`VENDOR_ALIASES`, `silent_merge_suspects` used identically across tasks.
- **#6 fidelity:** T1's tests include the explicit "live-but-superseded episode → fact kept" case; the sweep Cypher consults `HAS_EPISODE`/`removed`, never `superseded`.
- **No-LLM:** all three jobs are pure Cypher/data; T1/T3 tests seed graphs by hand (no extraction).
- **No-placeholder check:** each code step carries actual code; the T1 seed-Cypher note and the `structural_scanned` accumulation fix are explicit implementer instructions, not vague TODOs.
