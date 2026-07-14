# Slice 2 — Deferred Follow-ups

Deferred items from slice 2a (semantic extraction core) — the spec §11 ledger
plus everything surfaced during execution and the final whole-branch review.
Merge-blocking issues were fixed on the branch; everything here was consciously
deferred. Roughly priority-ordered.

## Environment / decisions locked during 2a (not deferred — recorded for context)
- **LLM client mode = `generic_json_schema`.** `gpt-oss-20b` via llama-server fails the
  `OpenAIClient` "structured" (Responses API) path (markdown-fenced/malformed JSON, 0
  entities). `OpenAIGenericClient(json_schema)` extracts cleanly. See `slice-2a-probe.md`.
- **Neo4j bumped 5.22 → 5.26** (docker-compose): graphiti-core 0.29.2 emits `SET n:$(labels)`
  dynamic-label Cypher that 5.22 rejects. Slice-1 testcontainer pins stay 5.22 (no Graphiti).
- **Latency ~6 min/episode** on one GPU (serialized). Full-corpus extraction is a
  background/batch concern; the 2a pilot runs a sample.

## Slice 2b ENTRY REQUIREMENTS (user-mandated at the 2a GO-WITH-CHANGES sign-off)

The 2a verdict (`slice-2a-viability.md`) was approved **on condition** that slice 2b
addresses these two extraction-quality issues up front (an ontology/prompt-v2 +
dedup-tuning pass before scaling). These are requirements, not nice-to-haves:

1. **Reduce noise.** ~20–30% of extracted facts (and ~4% of entities) are low-value:
   ARNs (`arn:aws:...`), error codes (`InvalidOrganizationBackupPlan`), specific
   example IDs/values, and CLI commands (`Install-Module …`). Strengthen the
   `EXTRACTION_INSTRUCTIONS` suppression (explicitly exclude ARNs/resource-IDs/error-
   codes/CLI-commands/example-values), populate `excluded_entity_types`, and consider a
   post-extraction filter. Re-measure noise rate.
2. **Improve deduplication quality.** Dedup precision misses cause false cross-vendor
   merges (e.g. an Azure vault/immutability concept merged into `AWS Backup Vault Lock`).
   Tighten canonicalization (it currently over-merges mid-tier concepts), add vendor-
   scoping hints to extraction/resolution, and add targeted `SAME_AS`/split fix-ups
   (plan §4.3). Also address entity **type confusion** (e.g. `AWS CLI`→Platform,
   `SEC 17a-4`→Platform) via sharper type descriptions (ontology v2). Re-measure the
   dedup report (correct-merge vs false-merge) against a labelled overlap set.

## Deferred to slice 2b (semantic pipeline hardening)
- **Append/invalidate temporal update policy** for re-extracted changed articles; the
  residual-staleness sweep; structural↔semantic `SAME_AS` reconciliation; durable work
  queue + dead-letter; token-budget metering + priority lanes; webhook/delta-driven
  incremental semantic ingestion wired to the slice-1 sync. (Spec §11.)
- **`provenance.link` + ingest atomicity (from final review + T8/T9 reviews).** Two related
  gaps that CANNOT trigger in 2a (first-extraction + hash-gate, single run) but must be
  closed when re-extraction lands:
  - `link()`'s MERGE keys on the episode node, so re-linking a chunk to a *different*
    episode (content changed) creates a second `HAS_EPISODE` edge. Fix: key the MERGE on
    `(article_id, chunk_index)` and re-point / detach the old edge.
  - `ingest_article` does add-then-link non-atomically: if `link` fails after
    `add_text_episode` succeeds, a re-run re-extracts (duplicate orphan episode). Fix:
    upsert-style link keyed by episode uuid, or a single transactional write.
- **`HAS_EPISODE` name collision:** graphiti-core 0.29.2 reserves `HAS_EPISODE` for its
  `Saga` feature (`(:Saga)-[:HAS_EPISODE]->(:Episodic)`). Safe now (we never pass `saga`,
  and provenance always anchors on `(:Article)-[:HAS_EPISODE]->`), but revisit if sagas
  are ever used.

## Deferred to later phases
- Full-corpus rollout (40 vendors, phased); the community-report layer (theme-builder,
  hierarchical Leiden, GDS, MS-GraphRAG reports); retrieval/answer-api; MCP + Copilot.
- **Cross-encoder for query-time search (retrieval slice):** graphiti's default
  `OpenAIRerankerClient` ranks via OpenAI *tiktoken* token-IDs for True/False in
  `logit_bias`/`logprobs`, which won't map correctly on `gpt-oss-20b`/llama-server — so
  cross-encoder reranking at *search* time may be effectively random/unsupported. Not used
  during ingestion (verified: `cross_encoder.rank` is search-only), so it does not affect
  2a. When building retrieval, validate reranker scores and if broken swap in a local BGE
  cross-encoder (`graphiti_core.cross_encoder.bge_reranker_client`) or an embedding reranker.

## Should-fix (revisit after the pilot / before scaling)
- **`build_graphiti` leaks AsyncOpenAI clients** — `Graphiti.close()` only closes the driver,
  not the llm/embedder/reranker clients. One-time per CLI run (bounded), multiplied per-mode
  in `probe.py`. Close them explicitly if graph_extract ever runs long-lived.
- **`fact_quality` judges each fact only against `episodes[0].content`** — for a fact derived
  from a later supporting episode this yields a false "unsupported", understating precision.
  Judge against all supporting episodes (or the best-matching one) when interpreting the
  verdict; treat the human spot-check as authoritative.
- **`cost_report` covers extraction-LLM `chat.completions` tokens only** — excludes TEI
  embeddings (not token-metered) and rerank (not called during ingestion). Adequate for the
  verdict; label it as such.
- **`provenance_report`/`fact_quality` don't filter `RELATES_TO` by `group_id`** (dedup_report
  does). Fine for the single-group spike DB; fix if the DB ever holds multiple groups.
- **Live e2e re-run fragility:** `test_ingest_driver.py` asserts `episodes_added >= 1` against
  a persistent DB with no cleanup — green only on a clean DB. Add teardown or branch the
  assertion for re-runs.
- **Chapter-path prefix uses the article's chapter title only** (per-chunk heading path from
  markdown positions deferred — spec §11). Jina query/passage task-prefix tuning deferred.
- **Ontology v2** from pilot learnings (`excluded_entity_types` tuning, per-type precision from
  the eval); watch for type-confusion (`soft delete` Capability-vs-Concept, `cross-region`
  Platform-vs-Capability).

## Carried from slice 1 (still open)
See [`slice-1-followups.md`](slice-1-followups.md) — mypy `--strict` gate not enforced + no CI
(now spans both packages: bare `dict` return types etc.), real webhook cluster→host delivery
verification, catalog fan-out semaphore, malformed-line dead-lettering, etc. Referenced, not
duplicated. Standing up CI that runs `ruff`, `mypy`, and `pytest -m "not live"` would cover the
quality-gate debt for both slices.
