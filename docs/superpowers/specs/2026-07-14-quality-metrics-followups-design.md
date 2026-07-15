# Quality-Metrics Follow-ups — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** 2b-quality follow-ups (metric trustworthiness + two extraction nudges)
**Date:** 2026-07-14
**Status:** Draft for review

Follows the merged slice 2b-quality ([`slice-2b-quality-report.md`](../slice-2b-quality-report.md) §5). Fixes the four residuals so the quality measurements can be trusted as gates, plus two extraction nudges. No new subsystems — edits to `eval.py`, `noise_filter.py`, `quality_labels.py`, `ontology.py`, `cli.py`.

## Motivation

The slice hit its goals but three *measurements* were shown unreliable and two extraction residuals remain. The metrics are what future slices gate on, so making them honest is the priority.

## Scope & environment

- The **full-run 2a-successor graph is still in Neo4j** (579 entities / 1,935 facts, group `backup-docs`). Items 1–3 are validated against it with **no re-extraction**. Item 4 is a prompt change validated on the 8-article overlap sample (`scripts/pilot-ids-sample.txt`, ~20 min).
- Judge stays `glm-5.2:cloud` (independent; self-judge guard now enforced).

## Item 2 — `should_distinct` tri-state + synonym tolerance (deterministic)

**Problem:** `dedup_report_v2` reports `collapsed = node_count < 2` for a labelled distinct pair. That conflates three different situations, producing false failures: it fired for `Amazon S3`/`Azure Blob Storage` (Blob Storage simply wasn't extracted) and `AWS Backup Vault Lock`/`Azure immutable vault` (extracted as `Azure Backup Immutable vault` — a spelling variant).

**Fix:**
1. **Alias tolerance.** In `quality_labels.py`, allow each `SHOULD_DISTINCT` member to carry accepted surface forms. Represent a member as `str | list[str]` (first is canonical; rest are aliases). `dedup_report_v2` resolves a member to the set of nodes whose lowercased name equals *any* of its forms. (Add the known alias `"Azure immutable vault" -> ["Azure immutable vault", "Azure Backup Immutable vault"]`.)
2. **Tri-state result** per pair, replacing the boolean `collapsed`:
   - `distinct` — both members resolve to node sets that are **non-empty and disjoint** (different node ids). GOOD.
   - `absent` — at least one member resolves to **zero** nodes this run. NOT a failure (coverage gap); reported separately.
   - `merged` — the two members resolve to a **shared** node id (one node represents both). The only real failure.
   Keep a back-compat `collapsed` boolean = `state == "merged"` so existing callers/tests still read.
3. **Acceptance:** on the current graph, none of the three labelled pairs report `merged`; the two that previously read `collapsed` now read `absent`.

**Testing:** integration test on a seeded graph covering all three states (distinct, absent via missing member, merged via a single shared node).

## Item 3 — `type_precision` larger sample + parse robustness (chosen: larger N, no majority-vote)

**Problem:** at N=40 with a non-deterministic reasoning judge the metric swung 0.472/0.625/0.784 on the identical graph, and ~20% of judgments came back unparseable (discarded).

**Fix:**
1. **Larger default sample.** Raise the default from 40 toward **100**, capped at the number of typed entities available (`min(requested, typed_count)`), so sampling variance shrinks. CLI `--sample` default bumped accordingly.
2. **Parse robustness (complements larger N — the unparseable rate is otherwise unchanged):**
   - Harden `_parse_type` (strip `<think>…</think>`, tolerate trailing punctuation/quotes, match a type word anywhere in the final line).
   - **One retry** when the first judge response is unparseable — a second single call with a terse "answer with ONE type word only" nudge. Still a single authoritative judgment per entity (not majority-vote), so cost stays ~1–1.2× calls, not 3×.
   - Continue to report `unparseable` count separately from disagreements, so a parsing regression stays visible.
3. **Acceptance:** on the current graph, unparseable rate materially below the ~20% baseline, and two back-to-back runs land within a tighter band than the 0.31 spread seen at N=40 (report the observed spread, don't hard-assert a threshold).

**Testing:** unit tests for `_parse_type` (think-block, punctuation, empty→None). The larger-N/retry behaviour is validated live against the current graph (report-producing, not CI).

## Item 1 — bare place-name regions (best-effort, conservative)

**Problem:** `_REGION` (`\bregions?$`) catches trailing-"region" names but not display-name regions still typed `Platform`: `India West`, `Israel Central`, `Asia Pacific (Malaysia)`.

**Fix (accepted as brittle/non-exhaustive):** add a conservative `noise_filter` pattern for the common cloud-region *display-name* shapes:
- a Title-Case name whose final word is a cardinal/ordinal direction (`East`, `West`, `North`, `South`, `Central`, `Northeast`, `Northwest`, `Southeast`, `Southwest`) — e.g. `India West`, `Israel Central`, `East US`, `West Europe`; **and**
- a name ending in a parenthesised place, e.g. `Asia Pacific (Malaysia)`.
Guarded hard against domain false-positives: unit tests must prove KEEP terms survive (`cross-region copy`, `Amazon S3`, `soft delete`, product names). If any realistic domain term collides, the pattern is narrowed rather than shipped. This stays a **reversible** decision (documented alongside `_REGION`).

**Acceptance:** the three observed escapees are pruned; every existing KEEP/NOISE test still holds; no new false positive on the current graph's non-region entities (manual scan of what the new pattern would additionally flag).

## Item 4 — canonicalization nudge (prompt; lowest priority)

**Problem:** `immutability` and `retention policy` appeared under near-synonym names in the full run rather than the canonical label (hurting `should_merge` node counts).

**Fix:** extend the canonical-names guidance in `EXTRACTION_INSTRUCTIONS` with the specific fragmenting cases (map common variants → the canonical `immutability`, `retention policy`, etc.). Prompt-only; no schema change.

**Acceptance:** on a sample re-run, `immutability`/`retention policy` resolve to their canonical `SHOULD_MERGE` names (node_count ≥ 1 under the canonical spelling). This is the one item needing re-extraction; if the effect is marginal, it is logged rather than chased.

## Non-goals / deferred

- No majority-vote for `type_precision` (chosen against — cost).
- No region gazetteer beyond the two display-name shapes above.
- Broader pipeline hardening (temporal policy, queue, SAME_AS, staleness sweep) stays in the later 2b sub-slice per `slice-2-followups.md`.

## Acceptance summary

Done when: `should_distinct` reports tri-state and no false `merged` on the current graph; `type_precision` unparseable rate is materially reduced and the metric is visibly steadier at the larger N; the three region escapees are pruned with all filter tests green; the canonicalization nudge is shipped and its sample-run effect recorded; unit + integration tests green; ruff/mypy clean.
