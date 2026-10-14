# Claude Haiku 5.5 as the only extraction model — A/B, 2026-10-14 (BACKLOG 59)

**Question:** can Claude Haiku 5.5, alone, replace today's routed extraction (solar-pro4
cheap tier with cache layout + gpt-5-mini for dense tables) without losing quality?
**Run:** `scripts/haiku_extract_ab.py`, the D6 sample regenerated (197 articles, 13
vendors, 19 sources), each arm in its own throwaway Neo4j, 4 articles at a time per arm.
Haiku arm: thinking disabled (`LLM_REASONING_EFFORT=none`), cache layout on, routing off.
Facts judged by the chunk-A/B judge (DeepSeek V4 Flash) against the source markdown.
Engagement gate passed before any fan-out (5 Claude calls, 0 to any other provider,
cache read, 2 episodes). Authorised cap $10; spend ≈ $6.05 (Anthropic $1.78, Azure
$1.12, OpenRouter $3.15 = solar $0.91 + judge ~$2.2 + smoke).

**Decision: do not switch.** Haiku keeps 77% of the distinct supported facts.

| | routed (today) | Haiku 5.5 only |
|---|---|---|
| failed articles | 0 | 0 |
| episodes | 685 | 669 |
| facts written | 3,857 | 3,037 (−21%) |
| **distinct supported facts (judge, 193 articles)** | **2,057** | **1,580 (77%; per-article median 0.75)** |
| unsupported facts (judge) | 113 | **88 (−22%)** |
| articles where Haiku had more / fewer / equal distinct facts | | 56 / 104 / 33 |
| extraction cost | $2.03 | **$1.78 (−12%)** |
| cost per 1,000 distinct facts | $0.99 | $1.13 (+14%) |
| median seconds per article | 26.5 | **15.3** |
| invalid dedup indices | 0 of 621 calls | 0 of 317 calls |
| integer-array overruns (`maxItems` unenforceable on Claude) | — | 0 |
| prompt tokens served from cache | 20% | 32% |

By article type (the routed arm's tier):

| | routed distinct | Haiku distinct | Haiku share | routed $ | Haiku per-episode $ vs routed |
|---|---|---|---|---|---|
| prose → solar-pro4 (176 judged) | 1,611 | 1,302 | 81% | $0.91 / 550 eps | +61% ($0.00266 vs $0.00165) |
| dense tables → gpt-5-mini (17 judged) | 446 | 278 | 62% | $1.12 / 135 eps | far cheaper |

**What it shows.** Haiku is reliable — no failures, no invalid indices, no runaway index
arrays even without `maxItems`, fewer unsupported facts, and twice as fast — but it is a
more conservative extractor: a quarter fewer real facts, worst on dense tables. On prose
it is also *more expensive per episode* than solar-pro4 at near-identical list prices,
because its prompts run larger (19.7M vs 12.9M prompt tokens over fewer episodes);
caching recovers part of that. The whole-arm saving comes only from not paying
gpt-5-mini for dense tables.

**Not tested here:** Haiku with thinking on. Thinking added 83% output tokens on a probe
extraction with the same entities, so it would raise cost; whether it recovers fact
yield is unmeasured. Anthropic rate limits at 8-worker scale are also untested (no 429s
at 4 concurrent articles).

**Consequences.** BACKLOG 59 closed (keep routed). BACKLOG 60 (batch API) was conditional
on 59 and is shelved for Haiku; it could be revisited for a different Claude model. The
adapter work stays: Claude can serve any extraction tier by configuration, with system
caching, structured outputs, `effort: none`, overrun counting and exact cost tallying.
