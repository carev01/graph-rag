# Vendor and product end to end — report (2026-09-25)

**What:** every answer names the vendors/products each claim applies to, and a question
that names vendors/products is answered from their documentation only.
**Spec:** `specs/2026-09-25-vendor-product-end-to-end-design.md` · **Plan:**
`plans/2026-09-25-vendor-product-end-to-end.md` · **Branch:** `bootstrap-resilience`.

## What shipped

- `answer_api/scope.py` — `ScopeResolver` from STRUCTURAL Vendor/Product nodes only (graphiti
  entities also carry those labels: 56 `:Vendor` nodes vs 40 vendors) plus
  `scope_aliases.json`; whole-word, case-insensitive, longest-match-first detection
  ("Veeam Backup for Microsoft 365" is one product, not Veeam + Microsoft); a vendor wins a
  vendor/product name collision (11 in the catalog, e.g. Keepit); cross-vendor wording with a
  cross-vendor noun keeps a question unscoped; `scope_episode_uuids(candidates=)` checks only
  the retrieved edges' episodes via the Episodic uuid index.
- `answer_api/attribution.py` — `(Vendor · Product)` labels on every fact line an LLM sees
  (local, global reduce, DRIFT, timeline render) and `applies_to` in every envelope, built from
  cited facts only. Labels come from `Provenance.resolve_citations`; the LLM never writes one.
- Global: in-scope share per candidate community from ONE `resolve_citations` call (a first
  custom Cypher full-scanned per fact: 20.6 s / 24M dbHits vs 1.3 s / 18k), communities below
  `global_scope_min_share` (0.5) dropped, out-of-scope facts removed before reduce.
- Router/API: one `Scope` to every mode and fallback; repeatable `vendor`/`product`,
  `scope=auto|none` on every route; a detected scope that grounds nothing is relaxed
  (`scope.source="detected-relaxed"`), an explicit one never; global falls back to scoped
  local when it cites nothing; resolver reload is lazy, single-flight and fail-soft.
- Eval: attribution judge calibrated on 8 known-count probes (8/8 on 5 runs; the first prompt
  counted components/tools named inside correctly labelled facts, 6 on a deterministic
  timeline render), grounding split scoped vs cross-vendor, routing split classifier vs
  answer path.
- Along the way: `merged_at` stamped as a string (a Neo4j DateTime broke extraction for every
  episode meeting a merged entity); bootstrap resumes a dropped stream (transport error,
  5xx/429) from the last applied id.

## Acceptance (final head `d946540`, paid run 2026-09-25)

| | target | result |
|---|---|---|
| scoped grounding | ≥ 0.8 | **0.96** (n=23) |
| classifier routing | ≥ 0.97 | **0.97** |
| faithfulness | ≥ 4.8 | **5.00** (0 unscored) |
| misattributed claims | reported | 6, all in one cross-vendor global answer |
| cross-vendor grounding (informational) | — | 0.33 (n=3; golden answers predate the Tier 1 vendors) |
| answer-path routing (informational) | — | 0.83 (scoped comparisons fall back to local by design) |

The first acceptance run failed (global grounding 0.17): scoped global has only ~3 AWS/Azure
communities, a comparison lost one vendor's community to a map-step failure, and the router
never fell back because a community had been used. Fixed by the cited-nothing fallback; the
criterion was re-baselined with the user (spec §6).

## Reviews

Every task passed a task-scoped review (Tasks 1, 3, 4, 6, 7 needed one fix round each); the
final whole-branch review (FIX FIRST: over-eager detection, unbounded scope query, docs)
and its scoped re-review (FIX FIRST: cross-vendor regex too greedy) were both resolved.

## Follow-ups

- Tier 1 golden questions once Veeam and Commvault are ingested (spec §6).
- BACKLOG 53 — superseded/removed episodes still count toward labels.
- BACKLOG 54 — generic product names could over-scope detection.
- Watch misattribution on cross-vendor global answers (the only non-zero answer).
