# Extraction Refresh Report — gpt-oss-120b Evaluation (NO-GO) + gpt-5-mini Re-run

**Date:** 2026-07-16
**Question:** Switch the extraction tier from Azure `gpt-5-mini` to `gpt-oss-120b` (same endpoint/key, cheaper) while re-extracting to fix the Region-magnet.
**Verdict: NO-GO on gpt-oss-120b.** Kept `gpt-5-mini` as the production tier; kept the Region docstring fix + code cleanups (which are model-independent).

---

## 1. gpt-oss-120b evaluation

A one-episode smoke test passed (9 clean entities / 5 facts, no reasoning leak, token capture works, via the `generic_json_schema` client path). But an **8-article sample re-run on the identical articles gpt-5-mini uses** exposed two regressions:

| Metric (same 8-article sample) | gpt-5-mini (v5) | gpt-oss-120b |
|---|---|---|
| Wall time | ~20 min | **~37 min (~2×)** → full 39-run ≈ 4 hr |
| Entities | ~103 | 123 |
| **`AvailableIn` facts** | 6 | **0** |
| **Region nodes** | 4 | **1** |
| Region-magnet (junk `:Region`) | ~2–3 | 0 |
| Limits facts | ~37 | 17 |
| Noise (post-prune) | 0% | 0% |

gpt-oss-120b's "0 junk Region" is an **artifact of under-extraction** (it typed only 1 region at all), not a real improvement. The decisive problem is that it produced **zero `AvailableIn` facts** — the residency-query capability ("which vendors offer backups in region X?") that the ontology was extended to support — and far fewer regions/`Limits`. Combined with ~2× latency for only "slightly lower cost," this is a poor trade.

**Decision (user):** NO-GO. Revert `.env` + the committed config default to `gpt-5-mini`; the pipeline still supports gpt-oss-120b via `.env` if revisited. The Region docstring fix and the reconcile/maintenance cleanups are decoupled from the model and kept.

## 2. gpt-5-mini re-run (with the Region docstring fix) — full 39 articles

Re-extracted on gpt-5-mini (348 episodes, ~2.3 hr), then `maintenance` (prune → retype → sweep → reconcile).

| Metric | Result | Note |
|---|---|---|
| Entities | 564 | (v5 baseline ~552) |
| Facts | 2004 | |
| **`AvailableIn` facts** | **124** | **restored** (gpt-oss-120b had 0); residency traversal works: `AWS Backup → ca-central-1`, `Azure Virtual machines → Australia East`, `archive tier → Spain Central` |
| `Limits` facts | 148 | |
| `Tool` entities | 62 | |
| Noise (post-prune) | **0%** entities / 0% facts | |
| `should_distinct` | `AWS Backup`/`Azure Backup` **distinct**, `AWS Backup Vault Lock`/`Azure immutable vault` **distinct**, `Amazon S3`/`Azure Blob Storage` **absent** (Blob not extracted) | no false merges — tri-state/alias fix holds |
| `suspect_false_merge` | 19 | heuristic upper bound |
| `silent_merge_suspects` (#7) | 2 | the new labelled-pair cross-vendor signal is populated |

All the query-enabling models (`AvailableIn`, `Limits`, `Tool`, `Region`) are healthy on gpt-5-mini, and dedup shows no real merges.

## 3. Region-magnet — partially improved, residual remains

The `Region` docstring negatives (API-ops, policies, PII, `Availability Zone` are not Regions) were **kept** and help some categories, but on gpt-5-mini the magnet is **not fully closed**: of 43 `:Region` nodes, 30 are gazetteer-recognized and **13 are junk** — roughly the same junk count as the v5 baseline. The residual junk this run: doc-guide titles (`Amazon … User Guide`), API fields (`AccountID`, `DBInstanceIdentifier`, `RestoreLatestVersionsUpTo`), `subscriptions`/`subscription S1`, `private IP address`, `Availability Zone`, `Protected resources`.

The docstring alone can't fully suppress this; a proper fix needs either (a) broader deterministic `noise_filter` patterns for the leaked classes (doc-titles ending "Guide", `…Identifier`/`…ID` fields, `subscription(s)`), or (b) a `Region`-type guard in `retype_region_entities` that demotes non-gazetteer `:Region` entities matching junk patterns (risky — the gazetteer isn't exhaustive). **Deferred as a follow-up.**

## 4. Outcome

- **Production tier stays `gpt-5-mini`** — restores `AvailableIn`/regions, ~2× faster, all features working.
- **Durable wins kept (model-independent):** the `Region` docstring negatives (modest help), reconcile polish (kind-aware unmatched, raise-not-assert, `:Product`-path test), the `maintenance` CLI + runbook, and the config now documents the gpt-oss-120b NO-GO for future reference.
- **The evaluation was cheap:** a 37-min sample surfaced the regressions before a 4-hr full run was spent on the wrong model.
- **Follow-up:** the Region-magnet residual (~13 junk `:Region`) needs broader noise patterns or a Region guard — a separate deterministic pass, no re-extraction of the model needed.
- **Reconcile note:** `reconcile` left structural Vendors `AWS`/`Microsoft` unmatched this run — the semantic Vendor entity names this extraction produced aren't in `vendor_aliases`; add them (the runbook's documented workflow).

## 5. Addendum (2026-07-16) — both follow-ups RESOLVED

Investigated, then fixed (strategy vetted by a `fable` subagent). Summary of the
decisions; details in the two paragraphs below.

- **Region-magnet → RESOLVED (code).** `retype_region_entities` now demotes
  self-typed non-gazetteer `:Region` to bare `:Entity` + a `demoted_from_region`
  audit stamp (reversible, self-healing, guard-limited). Verified against the
  live graph to target exactly the 8 junk `:Region`, 0 genuine regions; applied
  via `cleanup`/`maintenance`.
- **Reconcile → RESOLVED (reframed).** Not a matching bug: the valuable
  Product-level bridges already link; only the two Vendors are unmatched because
  their semantic twins don't exist. Matching left unchanged; added
  `unmatched_detail` classification (`no_candidate` vs `wrong_type_candidate`) so
  the expected misses stop looking like a bug.

**Region-magnet — prune vs. demote is a real design choice.** The current 38
`:Region` nodes are ~30 genuine regions + ~8 mistyped: `Availability Zone`,
`private IP address`, `Protected resources`, `subscriptions`, `subscription S1`,
`requester comment`, `RestoreLatestVersionsUpTo`, `Backup Fairfax Microsoft Entra
application`. Only `RestoreLatestVersionsUpTo`/`subscription S1` are true junk;
the rest are **valid domain concepts merely mis-typed as Region** — so the
`noise_filter` route would *delete legitimate entities*. The alternative, a
"Region guard" that demotes non-gazetteer `:Region`, risks demoting genuine
regions absent from the gazetteer (`Australia Central 2`, `Poland Central`,
`Israel Central`, `Norway West`, …). The right fix is a **demote-not-delete**
retype pass gated on a verified-complete gazetteer. **Shipped** as the
promote+demote `retype_region_entities` (see §6 for the fraction-dominant guard
and the production apply result).

**Reconcile — the aliases are already present; the real gap is upstream typing.**
`vendor_aliases` already maps `aws`/`amazon`/`microsoft`/`azure`. The production
semantic graph has **no `:Vendor` entity named AWS/Microsoft at all** (only
`Sysinternals` and a junk `awsbackup Amazon Resource Names (ARNs)`). AWS/Microsoft
are extracted as other types (Platform), so a `:Vendor`-scoped `reconcile` match
finds nothing. Fixing this is a reconcile-matching-strategy or extraction-typing
question, not an alias addition — deeper than the original note implied.
**Shipped** as read-only `unmatched_detail` classification (matching unchanged;
no false cross-type link) — see §6.

## 6. Applied to production (2026-07-17)

Both fixes landed on `main` (commits `6d28a26` region demote, `4230661` reconcile
classification, `2ae40d1` guard hardening) and were applied to the live
`backup-docs` graph.

**Region demote (surgical `retype_region_entities`, not full `cleanup`):**
`38 → 30 :Region`. Exactly the **8 junk demoted** to bare `:Entity` +
`demoted_from_region=true` (`Availability Zone`, `private IP address`,
`Protected resources`, `subscriptions`, `subscription S1`, `requester comment`,
`RestoreLatestVersionsUpTo`, `Backup Fairfax Microsoft Entra application`).
**0 genuine regions lost**, guard not tripped (8 ≤ `max(2, 50%·38)=19`).
Reversible via the audit stamp; self-heals if any name is later gazetteered.

**Demote-guard hardening (post-review):** the guard is now fraction-dominant —
`skip demotions if > max(2, 50% of :Region count)` — replacing a fixed floor of
10 that could never trip below ~34 regions (so it now protects small
bootstrap/per-vendor graphs from an `is_region` regression). A `cleanup
--force-demote` bypass handles a confirmed large legitimate backlog so cleanup is
never permanently stuck.

**Reconcile classification (live):** `linked=2` (the Product bridges `AWS Backup`,
`Azure Backup`, idempotent), `unmatched_detail = {Vendor:AWS → no_candidate
(expected), Vendor:Microsoft → wrong_type_candidate [Azure:Platform]}`. No
cross-type SAME_AS was created. The permanent false-alarm is now a self-labelling
report: `no_candidate` needs no action; a `wrong_type_candidate` is the only
actionable signal (an upstream extraction-typing gap for a future session).
