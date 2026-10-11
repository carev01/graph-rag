# Report writer: Claude Haiku 5.5 vs GLM-5.3-flash (2026-10-11)

**Decision:** move the theme-report writer to `claude-haiku-5-5` (default effort) and keep
DeepSeek V4 Flash (`~deepseek/deepseek-v4-flash-latest`, OpenRouter) as the verifier.
Quality is a tie; Haiku was more reliable; it costs ~1.8x per report and is billed to
Anthropic API credits instead of the OpenRouter balance.

## Method

25 communities sampled at random (seed 7) from the 853 verified live communities
(10–3000 members). Each writer ran the production pipeline unchanged
(`theme_builder.report.generate_report`: generate -> verify -> one retry naming the
offenders -> drop still-unsupported findings -> regenerate the summary), on the same
assembled context. All three arms shared one verifier, Claude Sonnet 5.5 (effort
medium), so no writer graded itself. A blind pairwise judge (Sonnet 5.5, A/B order
randomised) compared each Haiku arm's final report with GLM's against the facts.

Caveat: the verifier and judge are the same family as Haiku. Read a narrow Haiku win
as a tie.

## Results

| | GLM-5.3-flash | Haiku 5.5 `effort: low` | Haiku 5.5 default effort |
|---|---|---|---|
| First-pass findings supported | 64% (139/217) | 56% (157/280) | 57% (170/299) |
| Reports produced | 24/25 (one lost every finding) | 25/25 | 25/25 |
| Final findings / report | 10.5 | 10.4 | 10.8 |
| Cited facts / report | 27.7 | 23.4 | 24.2 |
| Blind judge vs GLM | — | 11 wins, 12 losses, 1 tie | 13 wins, 11 losses |
| Faithfulness / coverage / usefulness (1–5) | 4.29 / 4.29 / 2.92 | 4.25 / 4.04 / 3.12 | 4.33 / 4.08 / 3.12 |
| Median seconds / report | 20 | 34 | 36 |
| Empty replies / errors | — | 0 / 0 (75 calls) | 0 / 0 (74 calls) |

Haiku is slightly more faithful and more analytical (it more often flags conflicting
facts); GLM covers more facts. Both occasionally add a product name the facts do not
state ("Veeam Backup & Replication"); the verifier catches most.

## Cost

| | Per report | Full rebuild (~857) |
|---|---|---|
| GLM-5.3-flash ($0.15 / $0.50 per M) | ~$0.0036 (measured: balance -$0.091 over 25) | ~$3.10 |
| Haiku 5.5 default effort ($0.10 / $0.50 per M, <100k prompt) | ~$0.0064 (20.8k in + 8.6k out) | ~$5.50 |
| Verifier: DeepSeek V4 Flash ($0.018 / $0.64 per M) | ~$0.0015 | ~$1.30 |
| Verifier: Sonnet 5.5 ($2 / $10 per M) | ~$0.05 | ~$45 |

Report prompts are ~21k tokens, well under Haiku's 100k pricing threshold.

## Haiku writer + DeepSeek verifier (5 communities, seed 11)

Haiku 5.5 through the production client (`REPORT_LLM_BASE_URL=https://api.anthropic.com`),
each first-pass report verified by DeepSeek and by Sonnet 5.5 on the same findings:

- Verdict agreement 29/36 findings (0.81); DeepSeek rejected 18, Sonnet 15 — DeepSeek is
  at least as strict, so it does not wave through what Sonnet catches.
- Full pipeline with DeepSeek: 4/4 reports produced. One community's 18-finding report
  got no usable JSON from DeepSeek at either budget and would be staged for
  `theme-build --verify-pending` — the existing recovery path; 3 other calls needed the
  built-in retry.

## Configuration

```
REPORT_LLM_BASE_URL=https://api.anthropic.com
REPORT_LLM_MODEL=claude-haiku-5-5
REPORT_LLM_API_KEY=<Anthropic API key>
REPORT_REASONING_EFFORT=high      # Claude's default; "low" scored slightly worse
REPORT_MAX_TOKENS=64000           # thinking shares the budget: a 776-fact hub report used 47k
# verifier unchanged: VERIFY_LLM_* unset -> EVAL_JUDGE_* (DeepSeek V4 Flash)
```

Any synthesis-side tier built by `bounded_llm_client` (report, verify, map, eval judge,
synthesis, router classifier) moves to Claude the same way; only the report writer was
validated here.

## First live build: verifier budget (2026-10-11)

The first incremental build (658 communities) staged 8 of its first 10 reports
unverified. Cause: DeepSeek's reasoning length depends on the OpenRouter provider
serving it. Re-measured on one 16-finding report, 4 calls per budget:

| max_tokens | outcome |
|---|---|
| 4000 | 4/4 empty (`length`), all Cohere |
| 8000 | 3/4 verdicts (BaseTen, ~2,600 tokens); Cohere still `length` |
| 16000 | 4/4 verdicts (Cohere 7,000–9,000 tokens, ignoring `effort: low`) |

Haiku's first drafts carry more findings than GLM's (~12 vs ~9), which pushed Cohere
past the old 4000/12000 budgets; a GLM report hit the 4000 cap on 1 of 3 calls too.
`VERIFY_MAX_TOKENS` (default 8000; the retry gets 3x = 24000) replaces the hardcoded
4000. Only tokens used are billed. The same run exposed `AnthropicChat` lacking
`close()`, which theme-build calls on exit; both are fixed.

## Hub communities: context selection and output budget (2026-10-11)

Tier 1 run 4 (after the AWS/Azure ingest) cited only Veeam sources on every cross-vendor
global question. The AWS/Azure data sits in one ~2,150–2,400-entity hub community
(~7,000 facts) at every level, and `assemble_context` kept its 42 most *recent* facts:
0 of the 82 about `AWS Backup Vault Lock` (a top-30 entity), so no report mentioned
Vault Lock. Fixed by choosing facts by the top entities (taking turns) and growing
the context budget with the fact count (12k -> 48k tokens): the hub context now holds
~776 facts, 26 on Vault Lock, and its report carries two Vault Lock findings.

The larger contexts produce longer reports, and Haiku's thinking shares
`max_tokens`: at 16000, 37 of the first 80 regenerated communities came back
truncated (unparseable JSON, report carried over). The adapter now streams every
request and logs a `max_tokens` stop; the hub report used 87k input / 47k output
tokens (~$0.03).
