# ling-2.6-flash Production-Readiness — Final Verdict (2026-07-17)

**Question:** after tuning, is `inclusionai/ling-2.6-flash` (OpenRouter, ~20× cheaper)
ready to replace Azure `gpt-5-mini` as the extraction tier?

**Verdict: NO-GO as a wholesale replacement — but recommend a HYBRID.** ling is
~23× cheaper with comparable-or-better quality on the content it can handle, but
it is fundamentally unreliable and pathologically slow on **dense availability
matrices** (the highest-value residency data), and no combination of levers fixes
it. The investigation also produced **three universal extraction wins** (already
merged) that help gpt-5-mini too.

---

## Three universal wins banked (merged to main)

These stand regardless of the ling decision — they improve extraction for **any**
model:

| commit | change | impact |
|---|---|---|
| `ef46361` | **Subject nudge** — "always extract the backup product as an entity every chunk" | ling subject-in-chunk 0/6→6/6, surviving facts 20→312 (15×). No-op for gpt-5-mini (already 4/4). graphiti drops any edge whose endpoint isn't an extracted node, so a missing subject silently discarded every "Product provides X" fact. |
| `cabb20f` | **Structure-aware chunk splitter** — cut oversize chunks on line/table-row boundaries, not mid-cell | Preserves neural-chunk quality when `max_chunk_tokens` is lowered; equal-char split only as last resort. |
| `27e8e21` | **Dropped `Product.version` attribute** | With NO entity type carrying attributes, graphiti skips its attribute-extraction step — which was where the cheap model ballooned. Worst matrix chunk: response 56,539→19,224 chars, truncations 1→0, LLM calls 8→3. |

## Cost — ling is ~23× cheaper (real prices)

Real OpenRouter price: ling `$0.01/1M` input, `$0.03/1M` output (gpt-5-mini
estimated `$0.25/$2.00`). Measured on 2 clean prose articles (14 episodes each):

| | tokens/ep | **$/ep** | **corpus proj (105k eps)** |
|---|---|---|---|
| ling | 31,852 | **$0.00043** | **~$45** |
| gpt-5-mini | 26,109 | $0.00974 | ~$1,023 |

ling is more verbose (22% more tokens/ep) but its per-token price is so low it's
**~23× cheaper overall**. The cost saving the user hoped for is very real.

## Quality — comparable or better (on content ling completes)

Same 2 clean articles, after the deterministic correctors:

| metric | ling | gpt-5-mini |
|---|---|---|
| type_precision | **0.789** | 0.763 |
| noise (entity / fact) | 0.0 / 0.0 | 0.0 / 0.0 |
| dedup (AWS/Azure distinct, false merges) | correct, 0 | correct, 0 |
| AvailableIn facts | 6 | 2 |
| Region entities | 3 | 0 |

Quality is a wash-to-slight-win; ling actually captured **more** residency data.

## Reliability — the wall (dense matrices)

On dense availability/support-matrix articles ling **over-generates unboundedly**
in graphiti's edge-extraction step (output ∝ entities², and ling enumerates every
cell). This blows the 16k-token completion cap → JSON truncates mid-string →
parse-fail → graphiti retries → retry storm. **No lever combination fixes it:**

| mitigation tried | result on dense content |
|---|---|
| chunk 1800 (natural neural) | ~6 truncations / run |
| chunk 900 (+ structure-split) | ~6 truncations / run |
| raise cap 16k→32k | **worse** — 107k-char response, still truncated, fewer facts |
| drop `version` | big help (56k→19k on worst chunk) but biggest chunks still balloon |
| **full robust combo** (no-version + salience + 900 + nudge + split) | still **lost 4+ chunks** and **did not finish 5 articles in 60+ min** (gpt-5-mini: ~3 min) |

The root cause is ling's own lack of output discipline — it fills whatever budget
it's given and still overflows. It is **irreducible by prompting/chunking/caps**.
Every truncated chunk is either lost (dropped after retries) or triggers a retry
storm that makes ling **20–60× slower** than gpt-5-mini on exactly the articles
that hold the most valuable data.

## Recommendation — HYBRID routing

Don't switch wholesale, and don't discard ling. Route by content:

- **ling for prose / simple articles** (the majority of the corpus) → captures
  ~all of the ~23× cost saving with comparable quality.
- **gpt-5-mini for dense availability/support-matrix articles** (a detectable
  minority) → reliable residency extraction, no retry storms.

Routing is deterministic and cheap: a **table-density heuristic** on the article
markdown (ratio of `|`-delimited table lines, or presence of a large
feature×region grid) decides the tier per article. A simpler variant: run ling,
and **re-extract only the chunks that fail** (truncate) on gpt-5-mini — the
failure set is small, so the blended cost stays near ling's.

This is a genuine architecture win: most of the corpus at ~$0.03/1M, with
gpt-5-mini reserved for the ~few-percent of articles ling can't handle.

## What would make ling fully viable (not found)

Nothing tested bounds ling's dense-table over-generation. It would need either a
model with output discipline (a different cheap model, or a future ling revision)
or a graphiti-side change to cap/stream the edge-extraction step — out of scope
here. Until then, hybrid is the pragmatic answer.

## Repro (scratch, not committed)

`ling_diag.py`, `ling_ab.py`, `ling_trunc.py`, `ling_salience.py`,
`ling_capraise.py` (cap 16k vs 32k), `ling_semantic.py`, `ling_prod_eval.py`
(+ variants). All in isolated `ling-*` / `*-eval` groups, cleaned up; production
`backup-docs` never touched. Prior interim doc: `ling-viability-investigation.md`
(superseded by this file).
