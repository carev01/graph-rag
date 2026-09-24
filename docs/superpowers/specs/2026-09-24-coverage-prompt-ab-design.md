# Coverage-prompt A/B — can a prompt recover packing's fact loss? — design

**Date:** 2026-09-24. **Approved:** 2026-09-24 (user: "set it up"; paid, ~$2–3).
**Follows:** `docs/superpowers/chunk-packing-ab-2026-09-24.md` (packing to 1,200 tokens:
−36% cost, −40% time, −28% facts).

## Question

Packing loses facts because each extraction call yields a roughly constant number of
items. Two causes are addressable in the prompt: our cheap-tier instruction caps the
count ("roughly 10-25 facts total", written for ~250-token chunks), and graphiti's entity
prompt is conservative, while facts are only formed between extracted entities. Does a
coverage instruction, plus removing the fixed cap, recover **today's distinct content**
at packing's cost?

## Arms (same 30 articles as the packing A/B)

| arm | chunking | instructions | source |
|---|---|---|---|
| `today` | production | production | reused from `ab_full/results.json` |
| `pack1200` | pack 1,200, cheap cap 1,200 | production | reused |
| `pack1200_coverage` | pack 1,200, cheap cap 1,200 | variant (below) | new run |

Reusing two arms saves ~$1.2; run-to-run noise is present in every arm anyway.

**Variant instructions** (harness-only constants; production untouched until adopted):
- strong tier: `EXTRACTION_INSTRUCTIONS + COVERAGE_DIRECTIVE`
- cheap tier: `EXTRACTION_INSTRUCTIONS + CHEAP_TIER_SALIENCE_UNCAPPED + COVERAGE_DIRECTIVE`

`CHEAP_TIER_SALIENCE_UNCAPPED` is `CHEAP_TIER_SALIENCE` with its final fixed-count
sentence ("A dense table chunk should yield roughly 10-25 facts total, not dozens.")
removed; the anti-per-cell rules stay. `COVERAGE_DIRECTIVE` tells the model to work
through the whole message section by section, extract every stated product, feature,
workload, requirement, limitation and availability statement in each section, scale
output with the message's length, and never restate one fact in different words.

The override is applied by the harness after `_build_ingest_driver`, by setting the
`instructions` of the IngestDriver's strong and cheap `ExtractionTier`s.

**Engaged-variable proof (CLAUDE.md):** the new arm's 2-article smoke runs with LLM
capture on; the harness refuses to continue unless a captured `extract_nodes` or
`extract_edges` prompt contains the directive's marker string — proof graphiti actually
sent it, not merely that a Python attribute was set.

## Scoring: distinct facts, not raw facts

Raw fact counts reward redundancy (a third of `today`'s surplus was restatement). A judge
LLM — the eval judge (`_eval_judge_client_and_model`: DeepSeek V4 Flash, a different
family from both extraction models) — gets, per article, the article's markdown and the
three arms' fact lists under shuffled anonymous labels. It returns an inventory of
distinct ideas and, for every fact, the idea id it expresses and whether the source
supports it. Per arm: `distinct` = unique idea ids among supported facts; `redundant` =
supported facts − distinct; `unsupported` = facts the source does not support.

A reply that does not label every fact exactly once is retried once, then recorded as
unscored (`None`, never 0). Articles whose largest arm exceeds 150 facts are excluded
from judging and listed (the 461-fact Commvault page would not fit one judge call).

## Decision rule

Adopt packing + the variant for the bootstrap if `pack1200_coverage`'s distinct facts
per article are ≥ 0.95 × `today`'s (paired median) with unsupported facts no higher than
`today`'s, and its cost stays ≤ 0.75 × `today`'s. Otherwise keep today's configuration.
If adopted, the variant moves into `ontology.py` and the local fine-tune's prompt layout
needs re-evaluation before the incremental lane uses it.

Spend cap: $3 for the new arm plus judging (the harness's cap is per process; the judge
reports its own token spend).
