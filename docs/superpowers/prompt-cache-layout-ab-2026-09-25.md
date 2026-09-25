# Prompt cache layout A/B — 2026-09-25

**Question:** does moving each prompt type's static tail into the system message
(`graph_extract.cache_layout`, `CHEAP_LLM_CACHE_LAYOUT`) raise provider cache hits and
lower cheap-tier cost without hurting extraction?
**Run:** `scripts/cache_layout_ab.py`, 12 prose articles from the D6 sample, solar-pro4
(OpenRouter/Upstage: $0.09 input, $0.018 cached input, $0.36 output per 1M), same
articles and order per arm, each arm in its own throwaway Neo4j; facts judged by the
chunk-A/B judge (DeepSeek V4 Flash). Spend $0.45. Engagement gate passed after article 2
(4 static blocks learned, 130,560 tokens served from cache).

| | layout off | layout on |
|---|---|---|
| cache hit rate, overall | 4.8% | **17.5%** |
| cache hit rate, steady state (articles 3–12) | 6.1% | **25.4%** |
| cost, 11 typical articles | $0.103 | **$0.086 (−16%)** |
| cost, one 47-episode article | $0.129 | $0.130 (11% cached) |
| cost, all 12 | $0.232 | $0.216 (−7%) |
| distinct facts (judge, 11 articles) | 153 | 190 |
| unsupported facts (judge) | 85 | 20 |
| wall time | 2,525 s | 2,659 s (+5%, within run-to-run noise) |

Quality does not regress. Most of the judged gap is one article whose OFF run produced
66 unsupported facts and no distinct ones; excluding it, 161 vs 153 distinct and 14 vs
19 unsupported. One article (47 episodes, >150 facts) was too large to judge.

**Why not 5x:** only ~25% of prompt tokens are static even after the move — the bulk of
each prompt is per-call content (the chunk, previous episodes, candidate entity/fact
lists) — and output tokens (~a fifth of cost) are not discounted.

**Decision (user, 2026-09-25):** enable in production (`CHEAP_LLM_CACHE_LAYOUT=true`).
The worker's batch line (`semantic batch llm tokens: prompt= cached= (…%)`) is the
production check.
