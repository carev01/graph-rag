# Prompt-Caching Findings

**Date:** 2026-07-16
**Question:** Is the Azure `gpt-5-mini` extraction pipeline benefiting from prompt caching, and is there a low-risk way to structure prompts to get more of it?
**Answer:** Caching is **already active** (~13% of prompt tokens cached, automatically). Our static content (extraction instructions + ontology) is already in the cached prefix. Materially increasing the cached fraction would require patching Graphiti's internal prompt assembly — **not worth it now**; no code change recommended beyond the new monitoring we added.

---

## 1. Measurement (controller-run)

Instrumented `usage.py` to capture `cached_tokens` (Azure reports `prompt_tokens_details.cached_tokens`), then ran 5 distinct short episodes through the real extraction pipeline (scratch `group_id`, `gpt-5-mini`, so the analyzed corpus graph was untouched):

| Episode | prompt total | cached total | Δ cached | calls |
|---|---|---|---|---|
| 1 (cold) | 5,629 | 0 | 0 | 2 |
| 2 | 15,087 | 1,536 | +1,536 | 8 |
| 3 | 24,713 | 3,072 | +1,536 | 14 |
| 4 | 35,945 | 4,608 | +1,536 | 21 |
| 5 | 46,318 | 6,144 | +1,536 | 27 |
| **Final** | **46,318** | **6,144** | — | 27 |

**Cache hit rate = 6,144 / 46,318 = 13.3%.** Episode 1 is a cold cache (0); every subsequent episode adds a **constant ~1,536 cached tokens** — a stable ~1,536-token prefix is being cached and reused. (Azure caches prompt prefixes ≥1024 tokens in 128-token blocks; 1,536 = 12×128.)

## 2. Why 1,536 and not more — the prompt layout

Graphiti builds each extraction prompt as a short static **system** message plus a long **user** message ordered:

```
[static extraction rules / task]      <- static, identical every call
<ENTITY TYPES> {our ontology} </>     <- static (our EXTRACTION_INSTRUCTIONS + type defs)
<PREVIOUS MESSAGES> {recent episodes} <- DYNAMIC (varies per episode)
<CURRENT MESSAGE> {episode content}   <- DYNAMIC
[more task instructions]
```

Azure caches the **longest identical prefix across calls** — here, everything up to `<PREVIOUS MESSAGES>`: the system prompt + the static rules + **our `<ENTITY TYPES>` block (the ontology + extraction instructions)** ≈ 1,536 tokens. So **our static content is already in the cached prefix** — that's the good news. The prefix is *capped* by the first dynamic block (`previous_episodes`) appearing mid-user-prompt, and the multi-call pipeline (~5.4 calls/episode across different prompt types — extract-nodes, extract-edges, dedup, attributes) means only the prompt type(s) with a ≥1024-token stable prefix cache; the rest don't.

## 3. Levers, and why we're not pulling most of them

- **What we control (already optimal):** our `EXTRACTION_INSTRUCTIONS` + ontology type definitions are large and **stable** (no per-call dynamic content), and Graphiti places them *before* the dynamic episode content — so they cache. Keeping them stable across runs is the one thing to preserve: any edit to the instructions/ontology invalidates the cached prefix until it warms again (a one-episode cost, negligible).
- **What we do NOT control (would need a Graphiti library patch):**
  - The dynamic `previous_episodes` block sits *before* `<CURRENT MESSAGE>`, capping the cacheable prefix. Moving all static content to a strict prefix (system message) and pushing *all* dynamic content to the very end would extend the cached prefix — but that is Graphiti's prompt-assembly, not ours.
  - The multi-prompt-type pipeline: each type caches independently; some have short prefixes.
- **Not worth patching now:** 13% is caching working as designed for this prompt shape. The saving is real but modest (cached input tokens bill at a discount), automatic, and requires no maintenance. Patching a vendored library's prompt assembly is a maintenance liability that only pays off at full-corpus scale — revisit then.

## 4. What shipped

- `usage.py` / `cost_report` now capture `cached_tokens` + `cache_hit_rate`, so cache effectiveness is **monitorable** on every run (watch it if instructions/ontology change, or at corpus scale).
- **No prompt restructuring** — our static block is already cached; the remaining gains are Graphiti-internal and deferred.

## 5. Related findings from this slice

- **Region residuals (Part A):** the new `noise_filter` doc-title + API-id-field patterns pruned 9 junk entities from the current graph (doc "User Guide"s, `AccountID`, `DBInstanceIdentifier`, `DBClusterIdentifier`, `PrincipalOrgID`); the Region-magnet junk dropped 13 → 8. The remaining 8 are generic phrases the model mis-typed as `:Region` (`Availability Zone`, `subscriptions`, `RestoreLatestVersionsUpTo`, …) — no safe deterministic fix; documented residual.
- **Reconcile alias gap:** the `AWS`/`Microsoft` structural vendors are unmatched not because of a missing alias but because this extraction produced **no `:Entity:Vendor` denoting them** (the only semantic Vendor entities were `Sysinternals` and an ARN-junk node) — a Vendor-type *under-extraction* symptom, not an alias-map gap. No alias added, correctly.

## 6. Recommendation

**No action beyond the monitoring already shipped.** Prompt caching is active and our controllable static content is cached. Re-evaluate a Graphiti prompt-assembly patch (static-prefix-first, all-dynamic-last) only if, at full-corpus scale, the ~13%→higher delta justifies vendoring/patching the library.
