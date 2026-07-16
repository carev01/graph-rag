# Region Residuals + Prompt-Caching Evaluation — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Deterministic region-residual cleanup + prompt-caching measurement.
**Date:** 2026-07-16
**Status:** Approved design — ready for implementation planning

Two independent pieces the user asked for: (A) chip away at the Region-magnet residual **deterministically** (no re-extraction), and (B) **measure** whether extraction prompts are benefiting from Azure/OpenAI prompt caching, and only then decide if there's an actionable structural win.

---

## Part A — Region-residual refinement (deterministic, no re-run)

The gpt-5-mini re-run left 13 junk `:Region` nodes. They split into two classes:

**A1. Cleanly pattern-matchable noise (fix in `noise_filter`):**
- **Doc/reference titles ending "Guide":** `Amazon Elastic Compute Cloud User Guide`, `Amazon Redshift Database Developer Guide`, `Amazon Redshift Getting Started Guide`.
- **CamelCase API-id / field names:** `AccountID` (ends `ID`), `DBInstanceIdentifier` (ends `Identifier`).

These are noise **regardless of type**, so pruning them removes them from `:Region` (and anywhere else). Add to `noise_filter`:
- `_DOC_TITLE = re.compile(r"\b(User Guide|Developer Guide|Getting Started Guide|Reference Guide|Guide|Reference|Documentation)$")` — anchored to the end so only titles, not phrases containing "guide". Guarded by tests: no real backup concept name ends in these.
- Extend `_API_ERR`'s suffix group to include `ID` and `Identifier`: `(Failed|Error|Exception|RequestId|Id|ID|Arn|Name|Identifier)$` (still single-token via the existing `" " not in n` guard, so `Availability Zone` etc. are untouched).

`RestoreLatestVersionsUpTo` has no safe pattern (1 node) — left.

**A2. Model over-application (documented residual, not fixed here):** `subscriptions`, `subscription S1`, `Availability Zone`, `Protected resources`, `private IP address`, `requester comment`, `Backup Fairfax Microsoft Entra application`. These are generic phrases the model mis-typed as `:Region` — not noise, and a demotion guard would risk demoting unlisted-but-legit regions (the gazetteer isn't exhaustive). Left as a documented residual; the docstring lever was already shown weak.

**Validation:** unit tests for the new `noise_filter` patterns (the A1 names flagged; KEEP terms — incl. `Availability Zone`, `Amazon S3`, real product names — not flagged), then run `cleanup`/`maintenance` on the current graph and confirm the A1 names appear in `pruned_names` (no re-extraction).

## Part A' — Reconcile alias gap

The gpt-5-mini `reconcile` left structural Vendors `AWS`/`Microsoft` unmatched. **Investigate first:** inspect the current semantic `:Entity:Vendor` names. If a genuine AWS/Microsoft vendor entity exists under a name not in `vendor_aliases` (e.g. a variant spelling), add that form to `VENDOR_ALIASES` and re-run `reconcile` to confirm linkage. If the semantic layer simply produced **no** vendor entity for AWS/Microsoft this run (they were extracted only as products, or not at all), then "unmatched" is correct — document that and add nothing (don't invent aliases for entities that don't exist).

## Part B — Prompt-caching evaluation (measure-first)

**Goal:** find out whether the multi-call Graphiti extraction is getting prompt-cache hits on Azure, and whether there's a low-risk structural change worth making — without guessing.

**Grounding:** Azure/OpenAI auto-cache prompt **prefixes** ≥1024 tokens (no explicit cache-control needed); usage reports `prompt_tokens_details.cached_tokens` (chat-completions) / `input_tokens_details.cached_tokens` (Responses API). Our static content (`EXTRACTION_INSTRUCTIONS` + entity/edge type definitions) is handed to Graphiti via `custom_extraction_instructions`/`entity_types`/`edge_types`; Graphiti builds the actual prompts internally, so a cacheable prefix exists only if Graphiti places that static block first with only the episode text varying at the end.

**B1. Instrument `cached_tokens`.** `usage.py`: add `cached_tokens` to `UsageTally` and read it in `_tally_usage` from `usage.prompt_tokens_details.cached_tokens` (chat) or `usage.input_tokens_details.cached_tokens` (Responses), defaulting to 0 when absent. Surface it in `cost_report` (total cached, and a cache-hit-rate = cached / prompt tokens). Unit-testable with a stub usage object.

**B2. Measure.** Controller-run: reset the semantic layer, ingest ~3 fresh articles with the instrumented tally, and record: total prompt tokens, cached tokens, hit rate; and the trend across calls (a cold first call vs later calls — if the static prefix is stable, later calls should show a large `cached_tokens`). Also inspect one actual Graphiti extraction prompt (log/capture the messages) to see whether the static instructions/type-defs land at the **front** (cacheable) or are interleaved with the episode.

**B3. Findings + decision.** Write `docs/superpowers/prompt-caching-findings.md`: the measured hit rate, whether Graphiti's prompt layout is cache-friendly, the estimated cost impact of the current caching, and a recommendation:
- If caching already works well → document it, no change.
- If a low-risk structural win exists that we control (e.g. our `custom_extraction_instructions` block is large and stable and lands early → keep it stable; avoid per-call dynamic content in it) → apply it.
- If the ordering is entirely Graphiti-controlled and unfavorable → document the limitation + the option to patch/configure Graphiti later (out of scope to patch the library here).

**Constraint:** the extraction model (gpt-5-mini) and behavior are unchanged; this part only *observes* and, at most, keeps our static prompt block stable — no behavior change to what gets extracted.

## Testing

- `noise_filter` new patterns: unit tests (A1 names flagged; KEEP terms incl. `Availability Zone`/products not flagged).
- `usage` cached-tokens: unit test with a stub usage object exposing `prompt_tokens_details.cached_tokens`, and one without (→ 0).
- `cost_report`: asserts the new cached fields.
- Reconcile alias (if an alias is added): the existing reconcile tests stay green; a targeted test only if a real alias is added.
- Full non-live suite + ruff/mypy clean. Parts A-validation (re-prune) and B-measurement are controller-run.

## Acceptance criteria

1. `noise_filter` flags the A1 doc-titles + API-id fields; KEEP terms (incl. `Availability Zone`) untouched; `cleanup` on the current graph prunes the A1 names (shown in `pruned_names`).
2. The reconcile alias gap is either closed (a real vendor alias added + linkage confirmed) or documented as "no matching semantic vendor entity this run."
3. `usage`/`cost_report` capture `cached_tokens` + a cache-hit-rate.
4. A measured prompt-caching findings doc with a clear recommendation (and any low-risk change applied).
5. Unit tests green; ruff/mypy clean.

## Deferred

- The A2 Region-magnet phrases (model over-application) — no safe deterministic fix; documented residual.
- Patching/configuring Graphiti's internal prompt construction for caching (library change) — only if B's findings show it's the bottleneck and worth it.
- Any re-extraction — this slice is deterministic + measurement only.
