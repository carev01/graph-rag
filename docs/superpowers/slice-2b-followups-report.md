# Slice 2b Quality-Metrics Follow-ups — Report

**Date:** 2026-07-15
**Branch:** `quality-metrics-followups`
**Baseline:** the 2a graph metrics in `slice-2b-quality-baseline.json` (693 entities / 2,185 facts).
**After:** full 39-article re-run with ontology-v5 + tri-state metric + region model, post-`cleanup` (prune + retype). ~552 entities / ~2,094 facts.
**Judge:** `glm-5.2:cloud` (independent; no-self-judge guard enforced), `type_precision` now N=100.

---

## Verdict

All four follow-ups landed. The two metric fixes (Items 2, 3) make the measurements trustworthy; the region model (Item 1) delivers the headline capability — **region/residency questions are now answerable** — with one documented residual (a `Region`-type magnet). The canonicalization nudge (Item 4) was marginal and is logged, as its spec allowed.

---

## Item 2 — `should_distinct` tri-state + alias tolerance — DONE

`dedup_report_v2` now classifies each distinct pair as `distinct` / `absent` / `merged` on **node identity** (shared `elementId` = merged), with alias tolerance, replacing the old `node_count < 2` heuristic that produced false "collapsed" readings.

| Pair | 2a baseline | After |
|---|---|---|
| `AWS Backup` / `Azure Backup` | not collapsed | **distinct** |
| `AWS Backup Vault Lock` / `Azure immutable vault` | **collapsed=True** (false alarm) | **distinct** ✅ |
| `Amazon S3` / `Azure Blob Storage` | not collapsed | **distinct** |

The Vault-Lock pair was the 2a "false merge" that triggered this whole quality pass — it now reads correctly, because the alias (`Azure Backup Immutable vault`, the spelling the model actually produces) resolves and the identity check confirms the two are **separate nodes**. No pair reports `merged`. A regression test (`test_dedup_v2_distinct_tristate`) locks all three states, and a CLI-render test guards the `quality-report` path that a mid-review crash exposed.

**Known limitation (documented):** the `merged` state matches on entity name, so a *silent* merge (Azure mentions attaching to an AWS-named node without a distinct node) reads as `absent`, not `merged`. `suspect_false_merge` remains the primary silent-merge signal.

## Item 3 — `type_precision` stability — DONE

Default sample raised to **N=100** (capped at available typed entities), `_parse_type` hardened, and a **single retry-on-unparseable** (terse "one type word" nudge) reclaims judgments the reasoning model would otherwise drop.

| | Baseline (N=40, single run) | After (N=100) |
|---|---|---|
| Overall precision | 0.571 | **0.677** |
| Run-to-run spread (same graph) | 0.472 / 0.625 / 0.784 (0.31) | tightened at larger N |

Per-type gains on the exact 2a confusions: **Platform 0.20 → 0.50**, **Product 0.125 → 0.50**, **Concept 0.50 → 0.74** (plus new `Tool` 0.78, `Region` 0.56). The metric is now a usable gate rather than a ±0.15-noise number.

## Item 1 — `Region` first-class type + `AvailableIn` edge — DONE (with residual)

Regions were promoted from mistyped-`Platform` noise to a first-class `Region` entity type with an `AvailableIn` edge (Product / Capability / Workload → Region), backed by an exhaustive cross-provider gazetteer (`region_names.py`, ~230 entries) and a deterministic `retype_region_entities` corrector.

**The headline capability works.** On the full run:
- **33 `Region` nodes**, **91 `AvailableIn` facts**.
- **Residency traversal validated** — e.g. `AWS Backup -[AvailableIn]-> us-east-1 / US East (Ohio) / US West (Oregon) / Africa (Cape Town) / Asia Pacific (Hyderabad)`. The query class *"which vendors offer their SaaS backups in <region>?"* is now a graph traversal.
- **`Platform` dropped 63 → 31** — regions left it (that was the 2a Platform-precision drag).
- The gazetteer flags **0** recognised regions left mistyped; the `retype` corrector was a tested safety net that the model didn't actually need (it self-typed regions correctly under ontology-v5).

**Residual — `Region`-type magnet:** the model over-applies `Region` to some non-region entities (analogous to the earlier `Tool`-magnet). Of 33 `Region` nodes, ~20 are genuine regions and ~13 are type-confusion — API operations (`…DescribeKey`), phrases (`Managed policies for AWS Backup`, `Kubernetes service account annotations`), `Availability Zone`, `PII`. Two deterministic mitigations shipped this task: new `noise_filter` patterns pruned the clear noise that had also leaked in as `Region` (AWS account ids `112233445566`, Organizations/root ids `o-…`/`r-…`, `Invalid…` codes — 5 nodes), and the gazetteer gained the regions the initial tables missed (Taipei, Beijing, Ningxia, New Zealand, Mexico Central, `eastus2euap`). The remaining ~13 are real concepts wrongly typed; **fixing them needs a `Region`-docstring negative-example pass validated on a re-run — deferred as a follow-up** rather than chased here.

## Item 4 — canonicalization nudge — MARGINAL (logged)

The prompt nudge to converge `immutability` / `retention policy` variants had limited effect: the model produced `immutable`, `retention period`, `soft-delete retention` — partly reflecting the source docs' own vocabulary. Per the spec (Item 4 was lowest-priority, "log rather than chase"), this is logged, not iterated. The one real risk it introduced — folding the branded `Azure immutable vault` into generic `immutability` — was caught in review and fixed with an explicit brand carve-out.

## Noise (unchanged goal, still clean)

Entity noise 4.9% → **0%**, fact noise 6.7% → **0%** post-cleanup. The two-layer design holds: `noise_filter` is the single source of truth for both the metric and the prune, now covering numeric/org ids and `Invalid…` codes.

## Follow-ups logged for later

1. **`Region`-magnet:** tighten the `Region` docstring with negative examples (API ops, policies, scenarios, `Availability Zone` are not Regions); validate on a re-run.
2. **`should_distinct` silent-merge blind spot** (name-based `merged` detection) — pair with `suspect_false_merge`.
3. **Canonicalization** of `immutability`/`retention` is source-vocabulary-limited; revisit only if it hurts a real query.
4. Minor `retype`/gazetteer test-coverage gaps (multi-wrong-label, idempotency assertion, double-vendor-prefix) noted in the ledger.

## Acceptance

| Item | Result |
|---|---|
| 2 `should_distinct` tri-state, no false `merged` | **PASS** — all pairs distinct; Vault-Lock artifact fixed |
| 3 `type_precision` steadier + up | **PASS** — 0.571→0.677 at N=100 |
| 1 `Region` type + `AvailableIn`, residency answerable | **PASS** (with documented Region-magnet residual) |
| 4 canonicalization | **MARGINAL** — logged per spec |
| Tests + gate green | **PASS** |
