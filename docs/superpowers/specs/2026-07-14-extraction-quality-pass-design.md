# Extraction Quality Pass — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** 2b-quality (first sub-slice of slice 2b) — extraction noise reduction + dedup/type precision
**Date:** 2026-07-14
**Status:** Approved design — ready for implementation planning

Builds on slice 2a: [`2026-07-13-semantic-extraction-core-design.md`](2026-07-13-semantic-extraction-core-design.md) and its verdict `slice-2a-viability.md` (GO-WITH-CHANGES). Mandated by the 2a sign-off: see [`../slice-2-followups.md`](../slice-2-followups.md) "Slice 2b ENTRY REQUIREMENTS".

---

## 1. Scope

The 2a viability verdict was approved **on condition** that slice 2b first improves two extraction-quality issues, re-measured against the 2a baseline. This slice does exactly that, on the *existing* `graph_extract` pipeline — **no new subsystems**:

1. **Reduce noise** — ARNs, error/exception codes, CLI commands, and bare example identifiers are being extracted as entities/facts (~4% of entities, ~20–30% of facts in 2a).
2. **Improve dedup precision + type disambiguation** — false cross-vendor merges (an Azure vault/immutability concept merged into `AWS Backup Vault Lock`) and entity type confusion (`AWS CLI`→Platform, `SEC 17a-4`→Platform, `AWS account`→Capability).

**Explicitly deferred to a later 2b sub-slice** (pipeline hardening): temporal update policy, durable queue, budget metering, `SAME_AS` reconciliation, staleness sweep, delta-driven incremental ingestion. See the followups ledger.

**Approach (chosen): A — extraction-side tuning + deterministic post-filter.** Fix it at the source (prompt-v2 + ontology-v2) *and* guarantee the clear-cut cases with post-processing.

## 2. Environment (established)

- **Extraction model:** Azure OpenAI `gpt-5-mini` (Responses API, structured mode, reasoning=minimal), unchanged. Embeddings local TEI/Jina (768-dim). Chunking local Chonkie neural. Neo4j 5.26.
- **Judge model (NEW, independent):** `glm-5.2:cloud` via **Ollama Cloud** — OpenAI-compatible at `https://ollama.com/v1`, key in `.env` (untracked). Verified: it correctly classified `AWS CLI` as a Tool (not Platform) — i.e. it catches gpt-5-mini's *own* systematic errors. It is a **reasoning** model (reasoning in a separate field, final answer in `content`), so judge calls need an adequate token budget (~400–500, not truncated). **Never self-judge** (extraction and judge models are distinct).
- **The 2a result graph** is still in the compose Neo4j (693 entities / 2,185 facts) and serves as the baseline snapshot.

## 3. Architecture & components

Modifies the existing `graph_extract` package.

| Component | Change | Kind |
|---|---|---|
| `ontology.py` | **v2**: crisp per-type boundaries + positive/negative examples (fix type confusion); expanded canonicalization; vendor-scoping guidance; explicit noise-exclusion block in `EXTRACTION_INSTRUCTIONS`. | edit |
| `noise_filter.py` | **new, pure**: `is_noise(name, type) -> bool` + the pattern set. Single source of truth for "what is noise", used by BOTH the eval metric and the cleanup pass (they can never disagree). | new, unit-tested |
| `graph_cleanup.py` | **new**: `async def prune_noise_entities(driver, group_id) -> PruneResult` — `DETACH DELETE`s `:Entity` nodes flagged by `noise_filter` (removing their `RELATES_TO`/`MENTIONS` edges). Deterministic guarantee on top of prompt-v2. | new, integration |
| `quality_labels.py` | **new**: `SHOULD_MERGE`, `SHOULD_DISTINCT`, `VENDOR_TOKENS` — the acceptance criteria as data (reviewed as part of sign-off). | new |
| `eval.py` | **extend**: `noise_report`, `dedup_report` v2 (labelled set + suspect-false-merge), `type_precision` (GLM-5.2 judge). | edit |
| `config.py` | ensure `judge_base_url`/`judge_model`/`judge_api_key` are wired (they exist); no self-judge default. | edit |
| `cli.py` / `run_pilot.py` | `quality-baseline` + `quality-report` commands; wire `prune_noise_entities` as a post-ingest step in the pilot. | edit |

**Key seam:** `noise_filter` is pure — the *one* definition of noise — so "what we measure as noise" is exactly "what we remove".

## 4. Noise reduction

**Noise classes (from 2a's real output):** ARNs/resource-URIs (`arn:aws:…`), error/exception codes (`InvalidOrganizationBackupPlan`, `…Failed`, `…RequestId`), CLI commands/code (`Install-Module …`, `aws backup start-restore-job …`), bare example identifiers (`snap-…`, `vol-…`, `restored-aurora-cluster`).

**Layer 1 — prompt-v2** (`EXTRACTION_INSTRUCTIONS`): an explicit "do NOT extract as entities" block covering these classes with examples, plus "extract the *concept*, not the example value" (extract `recovery point`, not `arn:aws:…:recovery-point:2FC4…`). Reduces model noise output + saves tokens.

**Layer 2 — `noise_filter.is_noise(name, type)`** (pure, conservative — only high-confidence noise; borderline domain terms kept):
- `arn:` prefix / ARN shape.
- CamelCase API/error tokens: endswith `Failed`/`Error`/`Exception`, `Invalid…`, `…RequestId`.
- CLI/shell markers (`--flag`, `Install-Module`, leading `aws `/`az `/`$`/backticks).
- bare resource-id shapes (`snap-`, `vol-`, `i-`, and similar hex-id patterns not matching a real concept).

**`prune_noise_entities`** runs post-ingest: flag every `:Entity` via `is_noise`, `DETACH DELETE` the flagged ones. Returns the count (a high prune count itself signals prompt-v2 is still weak). `excluded_entity_types` is NOT the lever (noise lives inside valid types), so it stays empty.

**Metric — `noise_report`:** `% entities` and `% facts` flagged by `noise_filter`, by class, + a raw flagged sample. Baseline vs after.

**Honesty note:** the filter is conservative and pattern-based; it won't catch every over-extracted vaguely-scoped Concept — those are a softer, prompt-side concern. It targets the clear-cut, pattern-matchable noise, which is the bulk of 2a's problem.

## 5. Dedup precision + type disambiguation

**Type confusion → ontology-v2 with hard boundaries + negative examples:**
- `Platform` = OS or cloud/infra platform (Windows, Linux, Azure, AWS, Hyper-V, VMware) — **not** tools/CLIs/SDKs/regulations.
- `Capability` = a backup feature/mechanism — **not** accounts/roles/resources.
- `Concept` = domain concept (RPO, RTO, 3-2-1, retention); regulations (`SEC 17a-4`) land here or `Requirement`, **not** Platform.
- `Requirement` = prerequisite/constraint (IAM permission, min version, port, license).

**Dedup false-merge → prevention via vendor-scoping** (prevention-first; splitting a wrongly-merged node is not attempted — re-attributing mentions is unreliable). Prompt-v2 rule distinguishes:
- **Generic concepts SHOULD merge across vendors** — `immutability`, `cross-region copy`, `Kubernetes`, `RPO`, `encryption`, `recovery point` → canonical generic names so AWS/Azure converge on one node.
- **Vendor-branded features stay distinct** — `AWS Backup Vault Lock`, `Azure immutable vault`, `Amazon S3`, `Azure Blob Storage` keep vendor-specific names → graphiti's resolver won't conflate them.

**Safe post-hoc only:** may add `SAME_AS`/merge for should-merge-that-didn't; do NOT split false-merges.

**Metric — `dedup_report` v2 against `quality_labels`:**
- `SHOULD_MERGE`: each canonical concept resolves to a small node-count (ideally 1) **with cross-vendor episode support**.
- `SHOULD_DISTINCT` (incl. `AWS Backup Vault Lock`↔`Azure` equivalent): confirm **not collapsed** (node_count ≥ 2).
- **`suspect_false_merge`** (new): entities whose name carries a vendor token (`aws`/`amazon`/`azure`/`microsoft`) yet have cross-vendor episode support → likely false merges. Count + names, vs baseline. Target: **reduced**.

## 6. Measurement harness

- **Baseline capture (impl step 1, before any tuning):** run the new metrics against the current 2a graph → save `docs/superpowers/slice-2b-quality-baseline.json` (committed). If the graph is altered, re-extract once with the 2a-era config (in git) first.
- **Three metrics:** `noise_report` (deterministic), `dedup_report` v2 (deterministic), `type_precision` (sample ~40 entities, judged by the independent GLM-5.2 + human spot-check; report overall + per-type precision + misclassifications).
- **`quality-report`** (`docs/superpowers/slice-2b-quality-report.md`): the three metrics side-by-side with the baseline, deltas, and PASS/FAIL vs the relative acceptance bar.
- **No self-judging:** noise + dedup use no LLM; type_precision uses GLM-5.2 (distinct from gpt-5-mini) + your human read as authoritative.

## 7. Testing & iteration workflow

**Testing:**
- **Pure unit:** `noise_filter.is_noise` — every noise class flagged; domain terms (`immutability`, `Amazon S3`, `RPO`) kept.
- **Integration:** `prune_noise_entities` (seeded graph: only noise deleted, edges gone); `dedup_report` v2 (seeded labelled graph counting).
- **`@live`:** small `type_precision` smoke against GLM-5.2.
- Full metrics runs are report-producing scripts (manual), not CI.

**Iteration workflow:**
1. Build metrics + `noise_filter` + cleanup + labels → capture 2a baseline.
2. Loop on a **~6–8 article overlap sample**: edit ontology-v2/prompt/noise-patterns → reset semantic layer → re-extract sample → `prune_noise_entities` → `quality-report` vs baseline → repeat.
3. **Final full 39-article re-run** → cleanup → the headline `slice-2b-quality-report.md`.

## 8. Acceptance criteria (relative, per the 2a sign-off)

Done when, vs the committed baseline:
1. Noise classes (ARN/error-code/command) largely eliminated; total noise-entity rate well below 2a's ~4%.
2. `SHOULD_MERGE` concepts merge (small node-count) with cross-vendor support.
3. `SHOULD_DISTINCT` pairs still don't collapse.
4. `suspect_false_merge` count reduced.
5. Type precision up (GLM-5.2 judge + human spot-check), with the 2a-observed confusions (AWS CLI/SEC 17a-4/AWS account) corrected on the sample.
6. `quality-report.md` written with baseline-vs-after deltas; unit + integration tests green; gate (mypy/ruff) clean.

## 9. Deferred

- **Pipeline hardening** (later 2b sub-slice): temporal update policy, durable queue, budget metering, `SAME_AS` reconciliation, staleness sweep, delta-driven incremental ingestion.
- **Heavier dedup** (only if the re-measure shows prevention is insufficient): a custom entity resolver, or embedding-threshold tuning — an escalation informed by this slice's suspect-false-merge result.
- Per-chunk `heading_path`, Jina task-prefix tuning — see `slice-2-followups.md`.
