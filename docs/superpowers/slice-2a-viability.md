# Slice 2a — Semantic Extraction Core: Viability Verdict

**Date:** 2026-07-14
**Pilot:** AWS Backup (146) + Azure Backup (442) → curated 39-article overlapping-topic sample.
**Extraction model (final):** Azure OpenAI **`gpt-5-mini`** (Responses API, structured mode, reasoning=minimal). Embeddings: local TEI/Jina (768-dim). Chunking: local Chonkie neural.

## Verdict: **GO — WITH CHANGES**

Graphiti + a capable hosted instruct model produces a clean, provenance-linked, cross-vendor knowledge graph over the backup corpus. The pipeline is viable end-to-end. Before full-corpus rollout, three tuning items are needed: **noise suppression, entity-type disambiguation, and dedup precision.** None are blockers; all are prompt/ontology/config refinements, not architectural.

## What works (proven on real data)

- **Full pipeline end-to-end:** DocExtractor content → Chonkie neural chunking → Graphiti `add_episode` (7-type ontology) → `:Entity`/`RELATES_TO` semantic graph → `HAS_EPISODE` provenance, over both vendors in one `group_id`.
- **Provenance is exact:** sampled facts resolve fact → episode → article → `source_url` deterministically (e.g. an Azure ADE-encryption fact → its `learn.microsoft.com/.../backup-azure-vms-encryption` page). This is the citation backbone and it holds.
- **Fact quality (core) is good:** accurate, well-formed, correctly vendor-attributed facts, including nuanced ones ("a cross-Region copy of a continuous backup becomes a snapshot backup; PITR is not available for the copy").
- **Distinct entities stay distinct:** `Amazon S3` ≠ `Azure Blob Storage`; `AWS Backup` ≠ `Azure Backup` (no catastrophic over-merge).
- **Real cross-vendor merges happen:** genuinely shared concepts resolve to one shared node across both vendors' docs (e.g. `recovery point` — 1 AWS + many Azure articles).
- **Model/infra is production-shaped:** `gpt-5-mini` is fast (~27 s/episode), reliable (0 hard failures after the embedder-batch fix), cheap, and cost-tracked. Embeddings stay on-prem (no doc content leaves for embedding).

## Numbers

| Metric | Value |
|---|---|
| Articles ingested | 39 / 39 (0 dropped after batch-cap fix) |
| Episodes | 218 added (321 incl. resumes) |
| Entities | 693 (Workload 120, Requirement 78, Capability 72, Platform 66, Concept 46, Product 43, Vendor 6) |
| Fact edges (`RELATES_TO`) | 2,185 |
| Cross-vendor entities | 88 / 693 (~13%) — a mix of correct shared concepts and false merges (see below) |
| Tokens (extraction LLM) | 8.46M (8.1M prompt / 357K completion) over 2,861 calls |
| ≈ per article | ~217K tokens |
| Full-corpus extrapolation (~105k articles) | ~4B tokens (order-of-magnitude ~$1–2k at gpt-5-mini pricing — within the plan's budget prior) |
| Wall (resumed run) | ~95 min for 39 articles (~2.4 min/article) |

## Issues to fix before full-corpus rollout

1. **Noise entities/facts (~20–30% of facts are low-value).** ARNs (`arn:aws:ec2:...`), error codes (`InvalidOrganizationBackupPlan`, `CreatorRequestId`), specific example IDs, and shell commands (`Install-Module …`) are being extracted as entities/facts (~27 clearly-noise entities / 693 ≈ 4%, more at the fact level). The `EXTRACTION_INSTRUCTIONS` noise-suppression didn't fully hold with `gpt-5-mini`. **Fix:** strengthen the suppression prompt (explicit "do not extract ARNs, resource IDs, error codes, CLI commands, or example values"), add `excluded_entity_types`, and consider a post-filter.
2. **Entity type confusion.** `AWS CLI` / `AWS Command Line Interface` labeled `Platform`; `SEC 17a-4` labeled `Platform`; `AWS account` labeled `Capability`. The 7-type ontology's boundaries aren't crisp to the model. **Fix:** tighten type descriptions with positive/negative examples (ontology v2), informed by this run's per-type errors.
3. **Dedup precision — false cross-vendor merges.** `AWS Backup Vault Lock` (AWS-specific) picked up mentions from Azure articles — graphiti merged an Azure vault/immutability concept into the AWS-branded node. So the "88 cross-vendor" count overstates real sharing. **Fix:** tune the canonicalization (it currently pushes toward merging), add vendor-scoping hints, and/or add targeted `SAME_AS`/split fix-ups (plan §4.3). The distinct-pair guard passing shows the failure is precision (over-merge of mid-tier concepts), not the headline entities.

## Model recommendation

- **Use a capable hosted instruct/reasoning model via a managed API** (Azure `gpt-5-mini` validated). It gives reliability, low latency, high concurrency, structured-output support (Responses API), and low cost.
- **Local models were not viable here:** `gpt-oss-20b` (llama-server) — reasoning can't be disabled → ~4–6 min/episode + empty-response failures; `qwen3-235b` (OpenRouter) — quality good but cost/latency didn't fit target scale, plus provider-routing fragility.
- The pipeline now supports **local / OpenRouter / Azure** backends interchangeably via config, with cost captured in both chat-completions and Responses-API modes.

## Recommendation for slice 2b

Proceed to slice 2b (semantic pipeline hardening) with `gpt-5-mini` as the extraction model, and fold the three tuning items above into an **ontology/prompt v2 + dedup-tuning** pass early in 2b. Slice-2b scope (temporal update policy, durable queue, budget metering, `SAME_AS` reconciliation, staleness sweep) is unchanged and is where the dedup `SAME_AS` fix-ups naturally live. See `slice-2-followups.md`.
