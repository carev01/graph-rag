# Slice 2 — Open Follow-ups

Forward-looking only. The slice-2a should-fixes originally logged here were
resolved in the follow-ups cleanup pass (commits `e7c99fc`/`83795a4`):
`build_graphiti` now closes all AsyncOpenAI clients (LLM + embedder + reranker);
`eval` is `group_id`-scoped and `fact_quality` judges all supporting episodes;
the live-e2e re-run fragility is fixed. The slice-1 quality gate (mypy/ruff/CI)
is also done — see [`slice-1-followups.md`](slice-1-followups.md) for the few
remaining slice-1 items.

## Environment / decisions locked in 2a (context, not TODO)
- **Extraction model = Azure OpenAI `gpt-5-mini`** (Responses API, structured
  `OpenAIClient`, reasoning=minimal). Local models were unsuitable (gpt-oss
  reasoning can't be disabled; qwen-235b cost/latency). The pipeline supports
  local / OpenRouter / Azure backends interchangeably via config, with token+cost
  capture in both chat-completions and Responses-API modes. Embeddings stay local
  (TEI/Jina, 768-dim).
- **Neo4j 5.26** (dynamic-label Cypher graphiti-core 0.29.2 emits; 5.22 rejects).

## Slice 2b ENTRY REQUIREMENTS (user-mandated at the 2a GO-WITH-CHANGES sign-off)

The 2a verdict (`slice-2a-viability.md`) was approved **on condition** that 2b
addresses these two extraction-quality issues up front (an ontology/prompt-v2 +
dedup-tuning pass before scaling). Requirements, not nice-to-haves:

1. **Reduce noise.** ~20–30% of extracted facts (and ~4% of entities) are
   low-value: ARNs (`arn:aws:...`), error codes (`InvalidOrganizationBackupPlan`),
   example IDs/values, and CLI commands (`Install-Module …`). Strengthen the
   `EXTRACTION_INSTRUCTIONS` suppression (explicitly exclude ARNs/resource-IDs/
   error-codes/CLI-commands/example-values), populate `excluded_entity_types`, and
   consider a post-extraction filter. Re-measure noise rate.
2. **Improve deduplication quality.** Dedup precision misses cause false
   cross-vendor merges (e.g. an Azure vault/immutability concept merged into
   `AWS Backup Vault Lock`). Tighten canonicalization (currently over-merges
   mid-tier concepts), add vendor-scoping hints, and add targeted `SAME_AS`/split
   fix-ups (plan §4.3). Also address entity **type confusion** (`AWS CLI`→Platform,
   `SEC 17a-4`→Platform) via sharper type descriptions (ontology v2). Re-measure the
   dedup report (correct-merge vs false-merge) against a labelled overlap set.

## Deferred to slice 2b (semantic pipeline hardening)
- **Append/invalidate temporal update policy** for re-extracted changed articles;
  the residual-staleness sweep; structural↔semantic `SAME_AS` reconciliation;
  durable work queue + dead-letter; token-budget metering + priority lanes;
  webhook/delta-driven incremental semantic ingestion wired to the slice-1 sync.
- **`provenance.link` + ingest atomicity** (CANNOT trigger in 2a's first-extraction
  + hash-gate single run, but must be closed when re-extraction lands):
  - `link()`'s MERGE keys on the episode node, so re-linking a chunk to a
    *different* episode (content changed) creates a second `HAS_EPISODE` edge. Key
    the MERGE on `(article_id, chunk_index)` and re-point/detach the old edge.
  - `ingest_article` does add-then-link non-atomically: a `link` failure after
    `add_text_episode` succeeds leaves a duplicate orphan episode on re-run. Use an
    upsert-style link keyed by episode uuid, or a single transactional write.
- **`HAS_EPISODE` name collision:** graphiti-core 0.29.2 reserves `HAS_EPISODE` for
  its `Saga` feature. Safe now (we never pass `saga`; provenance always anchors on
  `(:Article)-[:HAS_EPISODE]->`), but revisit if sagas are used.

## Deferred to later phases
- Full-corpus rollout (40 vendors, phased, business-priority order).
- **Community-report layer** (theme-builder, hierarchical Leiden, Neo4j GDS,
  MS-GraphRAG-style reports).
- Retrieval / answer-api (local/global/drift/timeline + citation resolver); MCP
  server + Copilot Studio.
- **Cross-encoder for query-time search:** graphiti's default `OpenAIRerankerClient`
  ranks via OpenAI tiktoken token-IDs for True/False, which won't map on non-OpenAI
  backends — search-time reranking may be random there. Not used during ingestion
  (verified: `cross_encoder.rank` is search-only), so it does not affect 2a. When
  building retrieval, validate reranker scores; if broken, use a local BGE
  cross-encoder or an embedding reranker.

## Refinements (revisit as the corpus widens)
- Per-chunk `heading_path` derived from markdown positions (using the article
  chapter-path for now). Jina query/passage task-prefix tuning (one mode now).
- `cost_report` covers extraction-LLM tokens only (excludes TEI embeddings, not
  token-metered; and rerank, not called during ingestion) — labeled as such.
