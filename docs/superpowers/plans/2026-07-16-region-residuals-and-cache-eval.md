# Region Residuals + Prompt-Caching Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prune the cleanly-noise Region residuals (doc-titles, API-id fields) deterministically, close the reconcile alias gap where real, instrument prompt-cache hits, and measure whether extraction prompts are cached — then a findings doc.

**Architecture:** `graph_extract` only. `noise_filter` gains two patterns; `usage`/`eval.cost_report` gain `cached_tokens`. Two controller-run steps (re-prune validation + reconcile alias; caching measurement) produce docs. No re-extraction of the corpus.

**Tech Stack:** Python 3.12, Neo4j, Azure gpt-5-mini (unchanged), pytest, ruff, mypy.

## Global Constraints

- **Deterministic + safe:** the new `noise_filter` patterns must not flag real backup terms. In particular the API-id pattern requires a lowercase char before the terminal `ID`/`Identifier` so acronyms (`RAID`, `GRID`, `UUID`) are KEPT.
- **No re-extraction / no model change.** Part A validates by re-pruning the *current* graph; Part B only observes (and at most keeps our static prompt block stable).
- **`noise_filter` stays the single source of truth** (used by `noise_report` + `prune_noise_entities`).
- Secrets only in untracked `.env`; tests pass; ruff/mypy clean.

---

## File Structure

- `src/graph_extract/noise_filter.py` — `_DOC_TITLE` + `_API_ID` patterns.
- `src/graph_extract/usage.py` — `cached_tokens` in `UsageTally` + `_tally_usage`.
- `src/graph_extract/eval.py` — `cost_report` surfaces cached tokens + hit rate.
- `src/graph_extract/vendor_aliases.py` — (only if a real alias is warranted, T3).
- Docs: `docs/superpowers/prompt-caching-findings.md` (T4).
- Tests: `tests/unit/test_noise_filter.py`, `tests/unit/test_usage.py` (new or extend), `tests/integration/test_ingest_driver.py` cost assertions if present.

---

### Task 1: `noise_filter` — doc-titles + API-id fields

**Files:**
- Modify: `src/graph_extract/noise_filter.py`
- Test: `tests/unit/test_noise_filter.py`

**Interfaces:** `is_noise` additionally flags documentation titles and CamelCase API-id fields.

- [ ] **Step 1: Add failing test cases** to `test_noise_filter.py`'s `NOISE` and `KEEP` lists:

```python
# NOISE additions:
    "Amazon Elastic Compute Cloud User Guide",
    "Amazon Redshift Developer Guide",
    "Amazon Redshift Getting Started Guide",
    "AccountID",
    "DBInstanceIdentifier",
# KEEP additions (must NOT be flagged):
    "RAID",            # acronym ending in ID, not a CamelCase field
    "GRID",
    "Availability Zone",
    "Veeam Backup & Replication",   # a real product with no doc-title/id shape
```

- [ ] **Step 2: Run, confirm the new NOISE cases fail** (`is_noise` returns False today). `uv run --extra dev pytest tests/unit/test_noise_filter.py -q`.

- [ ] **Step 3: Add the patterns** to `noise_filter.py`:

```python
# Documentation / reference titles extracted as entities.
_DOC_TITLE = re.compile(
    r"\b(User Guide|Developer Guide|Getting Started Guide|Reference Guide|"
    r"Administration Guide|Administrator Guide|API Reference|Documentation)$")

# CamelCase API id / identifier field names ("AccountID", "DBInstanceIdentifier").
# Require a LOWERCASE char before the terminal ID/Identifier so real acronyms
# (RAID, GRID, UUID) are kept.
_API_ID = re.compile(r"^[A-Z][A-Za-z0-9]*[a-z](ID|Identifier)$")
```

And wire into `is_noise` (after the existing `_API_ERR` check):
```python
    if _DOC_TITLE.search(n):
        return True
    if _API_ID.match(n):
        return True
```

- [ ] **Step 4: Run tests green** (new NOISE flagged, all KEEP incl RAID/GRID/Availability Zone/products not flagged). Gate on `noise_filter.py`.

- [ ] **Step 5: Commit.** `git add src/graph_extract/noise_filter.py tests/unit/test_noise_filter.py && git commit -m "feat(extract): noise_filter catches doc-titles + CamelCase API-id fields (region residuals)"`

---

### Task 2: `usage`/`cost_report` — capture `cached_tokens`

**Files:**
- Modify: `src/graph_extract/usage.py`, `src/graph_extract/eval.py` (`cost_report`)
- Test: `tests/unit/test_usage.py` (new)

**Interfaces:** `UsageTally` gains `cached_tokens: int`; `_tally_usage` reads it; `cost_report` returns `cached_tokens` + `cache_hit_rate`.

- [ ] **Step 1: Write failing unit tests** in `tests/unit/test_usage.py` using stub usage objects:

```python
from types import SimpleNamespace
from graph_extract import usage

def _resp(prompt, completion, cached=None):
    u = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion)
    if cached is not None:
        u.prompt_tokens_details = SimpleNamespace(cached_tokens=cached)
    return SimpleNamespace(usage=u)

def test_tally_captures_cached_tokens():
    usage.reset_tally()
    usage._tally_usage(_resp(1000, 200, cached=768))
    t = usage.get_tally()
    assert t.prompt_tokens == 1000 and t.cached_tokens == 768

def test_tally_cached_defaults_zero_when_absent():
    usage.reset_tally()
    usage._tally_usage(_resp(500, 100))          # no prompt_tokens_details
    assert usage.get_tally().cached_tokens == 0

def test_tally_responses_api_cached():
    usage.reset_tally()
    u = SimpleNamespace(input_tokens=800, output_tokens=100,
                        input_tokens_details=SimpleNamespace(cached_tokens=640))
    usage._tally_usage(SimpleNamespace(usage=u))
    assert usage.get_tally().cached_tokens == 640
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement.** In `usage.py`:
  - Add `cached_tokens: int = 0` to `UsageTally`; add a `cached` kwarg to `add` (default 0) that increments `self.cached_tokens` and the per-call bucket.
  - In `_tally_usage`, after computing prompt/completion, read cached:
```python
    details = getattr(u, "prompt_tokens_details", None) or getattr(u, "input_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) if details is not None else 0
    _TALLY.add("llm", prompt=prompt or 0, completion=completion or 0, cached=cached or 0)
```
  In `eval.cost_report`, add to the returned dict:
```python
        "cached_tokens": t.cached_tokens,
        "cache_hit_rate": (t.cached_tokens / t.prompt_tokens) if t.prompt_tokens else 0.0,
```

- [ ] **Step 4: Run tests green** + the full non-live suite (nothing else breaks). Gate on `usage.py`/`eval.py`.

- [ ] **Step 5: Commit.** `git add src/graph_extract/usage.py src/graph_extract/eval.py tests/unit/test_usage.py && git commit -m "feat(extract): capture cached_tokens + cache_hit_rate in usage/cost_report"`

---

### Task 3: Region re-prune validation + reconcile alias gap (controller-run)

**Files:**
- Modify (maybe): `src/graph_extract/vendor_aliases.py` (+ its test) — only if a real alias is warranted.

Controller-run against the current graph (no re-extraction).

- [ ] **Step 1: Re-prune to validate Part A.** `uv run --extra dev python -m graph_extract.cli cleanup` — confirm `pruned_names` now includes the A1 doc-titles + API-id fields (`Amazon … User Guide`, `AccountID`, `DBInstanceIdentifier`). Record the before/after `:Region` junk count (was 13; expect ~8 after — the A2 phrases remain).

- [ ] **Step 2: Investigate the reconcile alias gap.** Query the current semantic vendor entities:
```bash
uv run --extra dev python -c "
import asyncio; from neo4j import AsyncGraphDatabase as G; from graph_extract.config import get_extract_settings as S
async def m():
    s=S.__wrapped__(); d=G.driver(s.neo4j_uri,auth=(s.neo4j_user,s.neo4j_password))
    async with d.session() as x:
        r=await x.run(\"MATCH (e:Entity:Vendor {group_id:'backup-docs'}) RETURN e.name AS n ORDER BY n\")
        print('semantic Vendor entities:', [y['n'] async for y in r])
    await d.close()
asyncio.run(m())"
```
  - If a genuine AWS/Microsoft vendor entity exists under a name not in `VENDOR_ALIASES` (e.g. `Amazon`/`Azure`/`Microsoft Azure` variants already covered, or a new form), ADD that normalized form to `VENDOR_ALIASES` in `vendor_aliases.py`, keep the KEEP-guard test green, and re-run `reconcile` to confirm `AWS`/`Microsoft` now link (`linked` up, `unmatched_structural` no longer lists them). Commit that change.
  - If the semantic layer produced **no** vendor entity matching AWS/Microsoft (they were only extracted as products or not at all), that's correct behaviour — do NOT invent aliases. Record the finding for the T4 doc / ledger.

- [ ] **Step 3: Record** the re-prune result + the alias finding (in the ledger; the caching doc in T4 can also note the reconcile outcome). If `vendor_aliases.py` changed, commit it: `git add src/graph_extract/vendor_aliases.py tests/unit/test_vendor_aliases.py && git commit -m "feat(extract): reconcile alias for <vendor> (close SAME_AS gap)"` — otherwise no commit.

---

### Task 4: Prompt-caching measurement + findings (controller-run)

**Files:**
- Create: `docs/superpowers/prompt-caching-findings.md`

Controller-run live measurement (a few articles; ~10 min).

- [ ] **Step 1: Measure current cache behaviour.** Reset the semantic layer, ingest ~3 fresh articles (a 3-line `scripts/pilot-ids-cache.txt` subset of `pilot-ids.txt`) with the instrumented tally, then read the cost report:
```bash
uv run --extra dev python scripts/reset_semantic_layer.py
PILOT_IDS_FILE=scripts/pilot-ids-cache.txt uv run --extra dev python scripts/run_pilot.py 2>&1 | tail -30
```
`run_pilot` prints the `cost_report` at the end — capture `prompt_tokens`, `cached_tokens`, `cache_hit_rate`, `calls`. (The first article is a cold cache; later articles/calls should show non-zero `cached_tokens` IF Graphiti's static prefix is stable.)

- [ ] **Step 2: Inspect the prompt layout.** Determine whether Graphiti places the static block (our `EXTRACTION_INSTRUCTIONS` + entity/edge type definitions) at the FRONT of the extraction prompt (cacheable prefix) or interleaves it with the episode. Options: read `graphiti_core`'s prompt library for the extraction node (`grep -rn "custom" .venv/lib/python*/site-packages/graphiti_core/prompts/` and the extract-nodes/edges prompt builders), or add a one-off debug log of the first extraction request's messages. Note where the dynamic (episode) content sits relative to the static block.

- [ ] **Step 3: Write `docs/superpowers/prompt-caching-findings.md`:** the measured hit rate + cached/total tokens; whether the prompt layout is cache-friendly (static-prefix-first) or not; the estimated cost impact at corpus scale; and a recommendation:
  - caching already effective → document, no change;
  - a low-risk win we control (keep `custom_extraction_instructions` large + stable, no per-call dynamic content in it) → apply + note;
  - ordering is Graphiti-controlled + unfavourable → document the limitation + the future option to patch/configure Graphiti (out of scope here).
  Also record the Part-A re-prune result and the Task-3 reconcile-alias finding for a complete picture.

- [ ] **Step 4: Commit.** `git add docs/superpowers/prompt-caching-findings.md scripts/pilot-ids-cache.txt && git commit -m "docs: prompt-caching findings (measured hit rate + recommendation)"`

---

## Self-Review Notes

- **Spec coverage:** A1 noise patterns → T1; cached-tokens instrument → T2; re-prune validation + reconcile alias → T3; caching measurement + findings → T4. A2 residual + Graphiti-patch explicitly deferred.
- **Safety:** the API-id pattern requires a lowercase before `ID`/`Identifier` (RAID/GRID/UUID kept) — T1's KEEP tests lock this.
- **Deps:** T2 before T4 (measurement needs the cached-tokens instrument). T1 before T3 (re-prune validates T1). T3/T4 controller-run.
- **Type consistency:** `cached_tokens`/`cache_hit_rate` in `UsageTally`/`cost_report`; `_DOC_TITLE`/`_API_ID` in `noise_filter`.
- **No-placeholder check:** each code step carries actual code; T3/T4 carry exact commands; the only "maybe" (vendor alias) is explicitly conditional on a real matching entity existing.
