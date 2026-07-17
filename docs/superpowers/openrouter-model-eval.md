# OpenRouter Extraction-Model Evaluation — laguna-xs-2.1 + ling-2.6-flash (both NO-GO)

**Date:** 2026-07-16
**Question:** Are `poolside/laguna-xs-2.1:free` or `inclusionai/ling-2.6-flash`
(via OpenRouter) viable as the extraction tier, replacing Azure `gpt-5-mini`?
**Verdict: NO-GO on both.** Production tier stays `gpt-5-mini`. Same decision
discipline as the [gpt-oss-120b evaluation](extraction-refresh-report.md): a small
sample surfaced disqualifying regressions before any full-corpus run was spent.

---

## Method

Two layers, mirroring the established probe approach:

- **Layer 0 — capability smoke** (raw OpenRouter API, no graph writes): does the
  model respond, honor a `json_schema` `response_format` (the pipeline's
  `generic_json_schema` path requires it), avoid reasoning-leak, and extract
  sensibly from a representative AWS-Backup+regions chunk?
- **Layer 1 — full-pipeline probe** (only for a model that passes Layer 0): real
  DocExtractor article chunks through graphiti's actual extraction path, isolated
  in a throwaway `group_id` (deleted after). Measures the discriminating
  capability that decided the gpt-oss-120b NO-GO: `AvailableIn` facts + `Region`
  entities. Includes a **same-article `gpt-5-mini` control** on "AWS Backup
  feature availability" (a feature×region matrix article) for an apples-to-apples
  comparison on the decisive capability.

## `poolside/laguna-xs-2.1:free` — NO-GO (fails Layer 0)

- **No structured output.** A `json_schema` `response_format` request returns
  **HTTP 404** — "No endpoints found that can handle the requested parameters."
  The pipeline's extraction path (`generic_json_schema`) depends on it. Fatal.
- **Empty content in plain mode.** Without a schema, it's a reasoning/code
  completion model that spent its whole token budget on hidden reasoning and
  returned **empty content** (`finish_reason=length`, 0 chars) in 13.8s.
- Wrong tool for the job (it's poolside's code-completion model), and the `:free`
  tier is rate-limited. Not worth a Layer-1 probe.

## `inclusionai/ling-2.6-flash` — NO-GO (passes Layer 0, fails Layer 1)

Layer 0 looked promising — clean `json_schema` output in 5.5s, no reasoning leak,
correctly typed entities (Region/Product/Vendor/Capability) and `is available in`
facts on the synthetic chunk. The **full pipeline on real articles** told a
different story.

### Same-article comparison — "AWS Backup feature availability"

| | ling-2.6-flash | gpt-5-mini (control) |
|---|---|---|
| Entities | 13 | 51 |
| **Facts** | **1** | **149** |
| `AvailableIn`-class facts | ~0 (on this article) | 39 |

**ling extracted ONE fact from the single most availability-rich article in the
corpus.** This is a feature×region matrix; `gpt-5-mini` parsed it into 149
structured facts (39 residency/availability facts). ling collapsed it — the exact
tabular-residency failure mode that killed gpt-oss-120b, but more severe.

### Full 3-article probe

| Metric | ling (3 articles) | gpt-5-mini (1 article, control) |
|---|---|---|
| Wall time | 798s (**266s/article**) | 121s/article |
| Entities / Facts | 136 / 115 | 51 / 149 |
| Region nodes | 12 (several junk) | 2 |
| `AvailableIn` facts | 17 (**highly duplicated**) | 39 |
| Tokens (prompt/completion) | 208k / 84k | 90k / 8k |

Three independent disqualifiers:

1. **Severe under-extraction on tabular/availability content** — 1 fact on the
   decisive article (above). The high-value residency data is exactly what's lost.
2. **Prompt-instruction leakage into entity names.** ling emitted a **749-character
   `:Region` entity** whose text is the extraction ontology itself —
   `"…limit requirement platform tool concept capability workload region vendor
   product service ui pane tab button wizard page section document navigation
   … pronoun antecedent vague reference"`. The model confuses its own instructions
   with document content and writes them into the graph. Graph-poisoning that no
   deterministic noise filter should have to clean up.
3. **~2.2× slower per article** (266s vs 121s) and **4× the completion tokens**;
   its `AvailableIn` facts are the same 3 sentences repeated
   ("AWS Backup supports cross-Region backup" ×N).

ling's better-looking articles ("Encryption…" 95 entities/85 facts) skew toward
over-extraction/noise, and that article is where the garbage entity appeared.

## Outcome

- **Production tier stays Azure `gpt-5-mini`** — it remains the only evaluated
  model that reliably extracts the `AvailableIn`/`Region` residency layer and
  parses feature-matrix tables, at ~half the latency and a quarter of the
  completion tokens.
- **No re-extraction run** — the sample was disqualifying, so no full-corpus run
  was spent (Layer-1 cost ~13 min total).
- **Pipeline still supports OpenRouter** via `.env` (`LLM_BASE_URL`/`LLM_MODEL`/
  `LLM_API_KEY`, `generic_json_schema`) if a stronger OpenRouter model is tried
  later. The `_inject_openrouter_provider` routing (require_parameters=True →
  Novita for ling) worked; the failure is model quality, not plumbing.
- **What "viable" would take:** a model that (a) supports `json_schema` structured
  output, (b) parses feature×region tables into per-cell facts, (c) does not leak
  instructions into entity names, and (d) is within ~1.5× gpt-5-mini latency.
  Neither candidate met (b)–(d); laguna failed (a).

## Reproduction

Scratch scripts (not committed): `or_smoke.py` (Layer 0), `ling_probe.py`
(Layer 1, isolated `model-eval-*` groups, auto-cleaned). Probe articles:
`85ca0827…` (feature availability), `78e2b9e5…` (Cross-Region backup),
`06059606…` (Encryption). OpenRouter key was passed at runtime, never committed.
