# Slice 2b — Extraction Quality Pass: Before/After Report

**Date:** 2026-07-14
**Baseline:** 2a graph (ontology-v1), captured in `slice-2b-quality-baseline.json` (693 entities / 2,185 facts).
**After:** full 39-article re-run with ontology-v4 + extended `noise_filter`, post-`prune_noise_entities`.
**Judge:** `glm-5.2:cloud` (Ollama Cloud), independent of the extraction model (`gpt-5-mini`). No self-judging.

> This is the analytical headline report. The `quality-report` CLI regenerates a
> raw numeric-delta version at this same path; the committed version is this
> narrative one.

---

## Verdict

**PASS on the two mandated goals** (noise reduction, dedup quality). Type
disambiguation: the **specific 2a confusions are corrected**, but the aggregate
type-precision *metric* proved too noisy to quote a delta (see §3). One residual
(geographic regions typed `Platform`) is documented as a follow-up.

The two mid-slice user directives — a **`Tool` entity type** and modelling
**limitations as broadened `Limits` fact-edges** — are implemented and verified
in the graph (50 `Tool` entities, 154 `Limits` facts).

---

## 1. Noise — PASS (unambiguous)

| Metric | Baseline (2a) | After | Delta |
|---|---|---|---|
| Entity noise rate | 4.9% (34/693) | **0.0%** (0/579) | −4.9pp |
| Fact noise rate | 6.7% (147/2185) | **0.0%** (0/1935) | −6.7pp |

The full re-run extracted 623 raw entities; `prune_noise_entities` removed **44**
and left **579 clean**. Everything pruned was genuine noise — a clean sweep of
every class the slice targeted:

- **ARNs / resource URIs:** `arn:aws:backup:*:…:backup-vault:cab-*`, `arn:aws:ec2:*:*:snapshot/*`, `arn:aws:kms:…:key/…`
- **Error/status codes:** `ScanJobFailed`, `RevokeRestoreAccessBackupVaultFailed`, `CreatorRequestId`
- **IAM/service actions** (`service:Action`): `iam:PassRole`, `kms:GetKeyPolicy`, `backup:CopyIntoBackupVault`, `rds:RestoreDBInstanceToPointInTime`, `ec2:CreateTags`
- **API request-field names** (CamelCase `…Arn`/`…Name`): `BackupVaultArn`, `EncryptionKeyArn`, `HypervisorArn`, `VMName`, `ComputeResourceName`
- **CLI / PowerShell commands:** `Get-AzRecoveryServicesBackupProtectionPolicy`, `Disable-AzRecoveryServicesBackupProtection`, `Remove-OBPolicy with -DeleteBackup flag`
- **Geographic regions:** `AWS Regions`, `Azure paired region`, `US government regions`, `secondary region`

Noise reduction is a *two-layer* design: the prompt (ontology v2–v4) discourages
these, and the deterministic `noise_filter`/prune guarantees removal. The prune
count itself (44) shows the model still emits some noise, so the deterministic
layer is load-bearing — and it is the single source of truth shared by the
metric and the cleanup, so "what we measure as noise" is exactly "what we
remove."

## 2. Dedup quality — PASS (in substance)

| Signal | Baseline | After |
|---|---|---|
| `suspect_false_merge` (heuristic upper bound) | 23 | **19** |
| Real cross-vendor node merges (direct inspection) | — | **0 found** |

**No entity node conflates two vendors.** Direct inspection confirmed the
vendor-branded features stay distinct nodes:
- `AWS Backup Vault Lock` and `Azure Backup Immutable vault` are **separate** nodes.
- `Amazon S3`, `AWS Backup`, `Azure Backup` each stand alone.

**On the `should_distinct` metric "collapses":** the metric keys on the *exact*
label names in `quality_labels`, so it reports a false "collapsed" when (a) one
member of the pair wasn't extracted this run (`Azure Blob Storage` — the sampled
Azure articles cover soft-delete/cross-region/encryption/restore, not Blob
Storage), or (b) the concept was extracted under a near-synonym (`Azure Backup
Immutable vault` vs the label's `Azure immutable vault`). These are
**measurement artifacts, not merges** — the underlying nodes are correctly
distinct. *Follow-up:* make `should_distinct` resolve label synonyms / tolerate
absent members, and align `quality_labels` spellings with observed extractions.

The `suspect_false_merge` heuristic (vendor-token name + cross-vendor episode
support) is an *over-approximation*: e.g. `AWS Backup` name-dropped inside an
Azure comparison doc counts as "cross-vendor support" without being a merge.
Its drop 23→19 is directional; the authoritative check (no real merges) passed.

## 3. Type disambiguation — MIXED (targeted fixes land; aggregate metric unreliable)

**The specific 2a confusions are corrected (verified in-graph):**

| Entity | 2a type | After |
|---|---|---|
| `AWS CLI` | Platform ❌ | **Tool** ✅ |
| `SEC 17a-4` (regulation) | Platform ❌ | **Concept** ✅ |
| API params / ARNs (`BackupVaultArn`, `arn:…`) | Product/entity ❌ | **pruned as noise** ✅ |
| Storage classes | Platform ❌ | Concept (ontology-v4) ✅ |

**The aggregate `type_precision` number is too noisy to quote a delta.** Three
runs of the judge against the *identical* final graph returned **0.472 / 0.625 /
0.784** — a 0.31 spread driven by 40-entity random sampling, non-deterministic
GLM reasoning, and 3–8/40 responses coming back unparseable. The baseline's
single 0.571 measurement sits inside that band. **We therefore do not claim a
precision delta**; the honest evidence is the qualitative spot-check above.

**Human spot-check of the judge's disagreements** (representative run) shows they
are almost entirely defensible, not degradation:
- **Regions typed `Platform`** (`India West`, `Israel Central`, `Asia Pacific (Malaysia)`) → judge says `Concept`. These are bare place-names that escape the `…region$` filter — the dominant Platform-precision drag and the one real residual (see §5).
- **Genuinely ambiguous boundaries** where two competent models disagree: `Amazon EC2` (Platform vs Workload), `SNS event notifications` (Concept vs Capability), `Backup Storage` (Workload vs Concept).

**Final entity-type distribution (579 typed + 165 Graphiti-untyped):**
Workload 95 · Concept 89 · Platform 63 · Requirement 56 · **Tool 50** · Capability 43 · Product 17 · Vendor 1.

## 4. User-directed additions — delivered

- **`Tool` entity type** (ontology-v3, tightened in v4): CLIs, SDKs, APIs, and
  named consoles now type as `Tool` instead of polluting `Product`/`Platform`.
  50 `Tool` entities in the full graph (`AWS CLI`, `Azure PowerShell`,
  `AWS Backup console`, `REST API`, `*BackupVault* API`). v4 tightened the
  docstring so UI panes/buttons/actions no longer leak in.
- **Limitations as `Limits` fact-edges** (broadened to Product→Workload,
  Product→Capability, Product→Platform + an extraction instruction to capture
  "not supported / except / does not" statements). **154 `Limits` facts** in the
  full graph. Kept as *temporal edges*, not an entity type, so a limitation can
  be invalidated when a later doc says a vendor added support — the right shape
  for questions like "key limitations of Commvault protecting Kubernetes."

## 5. Residuals & follow-ups (for the next sub-slice)

1. **Bare-place-name regions typed `Platform`** (`India West`, `Israel Central`).
   The `noise_filter` catches trailing `…region(s)` but not arbitrary geographic
   names; a full gazetteer across 40 vendors is out of scope here. *Reversible
   product decision:* region entities are currently treated as low-value and
   suppressed where matchable — remove `_REGION` from `noise_filter` if
   region-scoped questions become in scope.
2. **`should_distinct` metric** should resolve label synonyms and tolerate
   absent pair-members (it currently exact-matches names).
3. **`type_precision`** needs a larger sample and/or majority-vote over repeats
   to be a stable gate — single N=40 runs swing ±0.15.
4. `immutability` / `retention policy` appeared under near-synonym names this
   run rather than the canonical label — canonicalization tuning.

## 6. Acceptance bar (spec §8)

| # | Criterion | Result |
|---|---|---|
| 1 | Noise classes eliminated; rate well below 2a's ~4% | **PASS** — 0% post-prune |
| 2 | `SHOULD_MERGE` concepts merge with cross-vendor support | **PASS** (no merge failures; coverage-limited) |
| 3 | `SHOULD_DISTINCT` pairs don't collapse | **PASS** in substance (no real merges; metric artifacts explained) |
| 4 | `suspect_false_merge` reduced | **PASS** — 23→19, zero real merges |
| 5 | Type precision up; 2a confusions corrected | **PARTIAL** — confusions corrected ✅; aggregate metric too noisy to quantify; 1 documented residual |
| 6 | Report written; tests + gate green | **PASS** |

**Net:** the slice delivers on the 2a sign-off conditions (noise + dedup) and
both user directives, with type-precision characterized honestly rather than
over-claimed.
