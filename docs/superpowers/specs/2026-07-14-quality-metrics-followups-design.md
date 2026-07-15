# Quality-Metrics Follow-ups — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** 2b-quality follow-ups (metric trustworthiness + region modelling + extraction nudges)
**Date:** 2026-07-14
**Status:** Draft for review

Follows the merged slice 2b-quality ([`slice-2b-quality-report.md`](../slice-2b-quality-report.md) §5). Fixes the four residuals so the quality measurements can be trusted as gates, promotes regions to a first-class entity so region/residency questions are answerable, and adds a canonicalization nudge. No new subsystems — edits to `eval.py`, `noise_filter.py`, `quality_labels.py`, `ontology.py`, `graph_cleanup.py`, `cli.py`, plus a new `region_names.py` data module.

## Motivation

The slice hit its goals but three *measurements* were shown unreliable and two extraction residuals remain. Crucially, the earlier plan to *suppress* regions as noise would have made region/data-residency questions — e.g. **"which vendors offer their SaaS backups in Germany?"** — unanswerable. Regions were never noise; they were **mistyped as `Platform`**. So Item 1 gives them a correct home (a `Region` type + availability edge) instead of deleting them, which both fixes the type pollution and *enables* a valuable query class.

## Scope & environment

- The **full-run 2a-successor graph is still in Neo4j** (579 entities / 1,935 facts, group `backup-docs`). **Items 2–3** (metric/parse changes) are validated against it with **no re-extraction**. **Items 1 and 4** change the ontology/prompt, so their extracted output (the `Region` nodes, `AvailableIn` facts, canonical names) is validated on the 8-article overlap sample (`scripts/pilot-ids-sample.txt`, ~20 min) and then the full 39-article run for the headline.
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

## Item 1 — `Region` as a first-class entity type (Option A) + exhaustive gazetteer

**Problem:** regions were typed `Platform`, which both polluted `Platform` and — under the earlier suppress-as-noise plan — would have deleted the very data needed to answer region/residency questions. The right fix is a correct type, not deletion.

**Design — ontology-v5 (the fifth and final ontology addition, alongside `Tool`):**
1. **New entity type `Region`** (`ontology.py`): *a specific geographic or cloud region, or a jurisdiction, where a product operates or stores backup data* — a cloud region (`Germany West Central`, `East US`, `us-east-1`), a country/geo (`Germany`, `EU`), or a named availability zone. Explicitly **NOT** a `Platform` (a region runs *on* a platform), **NOT** a generic relative term (`primary region`, `secondary region` are `Concept`s), **NOT** a data-redundancy tier (LRS/ZRS stay `Concept`). Add to `ENTITY_TYPES`.
2. **New edge `AvailableIn` (Product→Region)** (`ontology.py`): *a product/service is available in, operates in, or stores data in a region.* Add to `EDGE_TYPES` and `EDGE_TYPE_MAP` as `("Product","Region"): ["AvailableIn"]`. Add an `EXTRACTION_INSTRUCTIONS` line to capture availability/residency statements ("available in…", "data resides in…", "supported regions…", "not available in…"). This is what makes *"which vendors offer their SaaS backups in Germany?"* answerable: `(:Product)-[AvailableIn]->(:Region "Germany …")`, with the vendor reachable via the structural layer / the product's `Provides` capability facts.
3. **`region_names.py` gazetteer** — `REGION_NAMES: frozenset[str]`, the normalised (lowercased, whitespace-collapsed) set of published region **display names and codes** across the providers this corpus covers — AWS (codes `us-east-1`…`il-central-1`…`us-gov-west-1`…`cn-north-1` + display names `US East (N. Virginia)`, `Asia Pacific (Tokyo)`, `EU (Ireland)`…), Azure (`East US`, `West Europe`, `North Europe`, `Southeast Asia`, `East Asia`, `Australia East`, `India West`, `Central India`, `Israel Central`, `Germany West Central`, `Sweden Central`, `UK South`, `Brazil South`, `Qatar Central`, `Jio India West`…), GCP (`us-central1`, `europe-west4`, `asia-northeast1`, `southamerica-east1`, `me-central2`, `africa-south1`…), and the common OCI / IBM Cloud / Alibaba Cloud regions. Dated (`2026-07`) static data file, like `quality_labels.py`.
4. **Deterministic retype guarantee** — the two-layer pattern (prompt + deterministic corrector), mirroring `noise_filter`/`prune_noise_entities`. `region_names.REGION_NAMES` is the **single source of truth** for "what is a region", used by both the extraction examples and a new `graph_cleanup.retype_region_entities(driver, group_id) -> dict`: for every `:Entity` whose normalised name (vendor-prefix stripped) is in `REGION_NAMES` but is not already `:Region`, **relabel** it — remove the mistaken custom type label (e.g. `:Platform`) and `SET e:Region`, leaving the Graphiti-owned `:Entity` label and all edges intact (allowed under design-invariant #5 because these custom type labels are *ours*, defined in `ENTITY_TYPES`). Returns `{scanned, retyped, retyped_names}` for audit. This guarantees regions are correctly typed even when the model still slips (we observed it mistype regions before).

**`noise_filter` changes:** regions are **no longer noise**. Remove nothing that deletes a *specific* region. Keep `_REGION` (`\bregions?$`) only for **generic/relative** region *words* (`AWS Regions`, `Azure paired region`, `primary/secondary region`) — non-specific, not answerable, low value (reversible, documented). The gazetteer is NOT wired into `noise_filter` for deletion; it feeds typing.

**False-positive guard (critical):** the gazetteer must not shadow domain terms. Entries are full multi-word region names or codes only — **no bare tokens** (`central`, `east`, `standard`, `archive`) that double as domain words. Unit tests assert every KEEP term survives and that `retype_region_entities` only relabels genuine regions on a seeded graph. Any collision → drop that entry.

**Granularity / rollup:** named cloud regions embed the country/geo token (`Germany West Central` contains `Germany`), so a country-level question resolves by retrieval-time containment against `Region` nodes; the model is also instructed to extract the bare country/geo as its own `Region` where stated, so `Germany` can exist as a node in its own right.

**Acceptance:** on a sample/full re-run, `Germany`/`Germany West Central`/major-provider regions exist as `:Region` nodes (not `:Platform`); `AvailableIn` facts connect products to regions; the retype pass relabels any residual mistyped region deterministically (audited); `Platform` no longer contains regions; every existing KEEP/NOISE filter test holds; and the target query — vendors offering a SaaS-backup capability `AvailableIn` a `Germany` region — is traversable in the graph.

## Item 4 — canonicalization nudge (prompt; lowest priority)

**Problem:** `immutability` and `retention policy` appeared under near-synonym names in the full run rather than the canonical label (hurting `should_merge` node counts).

**Fix:** extend the canonical-names guidance in `EXTRACTION_INSTRUCTIONS` with the specific fragmenting cases (map common variants → the canonical `immutability`, `retention policy`, etc.). Prompt-only; no schema change. Folds into the same ontology edit as Item 1 (both touch `EXTRACTION_INSTRUCTIONS`) and rides the same re-run.

**Acceptance:** on a sample re-run, `immutability`/`retention policy` resolve to their canonical `SHOULD_MERGE` names (node_count ≥ 1 under the canonical spelling). If the effect is marginal, it is logged rather than chased.

## Non-goals / deferred

- No majority-vote for `type_precision` (chosen against — cost).
- Region gazetteer covers the major cloud/backup providers' published regions (AWS/Azure/GCP/OCI/IBM/Alibaba) as of the 2026-07 snapshot; niche/private-cloud region naming and future new regions fall through to the structural patterns / retype pass until added.
- `Region` availability is modelled via the `AvailableIn` edge only (Product→Region). No region hierarchy (region→country→continent) or geo-reasoning; country rollup is retrieval-time containment. No `Vendor→Region` edge (vendor reached via the product).
- Broader pipeline hardening (temporal policy, queue, SAME_AS, staleness sweep) stays in the later 2b sub-slice per `slice-2-followups.md`.

## Acceptance summary

Done when: `should_distinct` reports tri-state and no false `merged` on the current graph; `type_precision` unparseable rate is materially reduced and the metric is visibly steadier at the larger N; `Region` is a first-class type with the gazetteer + deterministic retype pass, regions leave `Platform`, `AvailableIn` facts exist, and the "vendors offering SaaS backup in Germany" traversal works on a re-run; the canonicalization nudge is shipped and its sample-run effect recorded; unit + integration tests green; ruff/mypy clean.
