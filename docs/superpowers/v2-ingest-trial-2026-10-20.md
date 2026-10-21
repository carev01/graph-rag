# Distilled v2 as the cheap extraction tier — end-to-end ingest trial, 2026-10-20

**Question:** with the Opus-distilled qwen3.5-4b (`carev01/qwen35-4b-graphrag@opus-distill-2026-10`,
"v2") as the cheap tier, the GPU-worker setup of `deploy/helm` (routing on, dense tables still on
gpt-5-mini), does extraction match today's routed setup (solar-pro4 + gpt-5-mini)?
**Run:** `scripts/haiku_extract_ab.py --arms routed,v2 --judge routed,v2 --articles 60 --cap 3
--exclude <distill sample.json>`. 60 articles from the regenerated D6 sample (13 vendors; none of
them in the distillation data, and 9 of the 13 vendors absent from it entirely). Each arm ran in
its own throwaway Neo4j at 4 articles at a time. v2 was served by vLLM on a vast.ai RTX 4090
($0.416/h). Cache layout was on in both arms, as in the distillation capture.
The engagement gate passed before any fan-out: vLLM's own success counter rose by 46 for 46
cheap-tier calls, and 8 episodes were written. Facts were judged by the chunk-A/B judge (DeepSeek
V4 Flash) against the source markdown.
**Spend:** $2.66 in total: extraction $1.70, judge $0.17, unsupported-fact re-judge ~$0.05,
GPU $0.75.

**Decision: do not switch yet.** v2 holds most of the facts, but it doubles the unsupported ones,
and it brings no speed gain.

## Results

Extraction over all 60 articles (51 cheap-tier, 9 dense → gpt-5-mini in both arms):

| | routed (today) | routed, v2 cheap tier |
|---|---|---|
| failed articles | 0 | 0 |
| episodes | 267 | 267 |
| facts written | 1,641 | 1,553 (−5%) |
| entities | 1,372 | 1,167 (−15%) |
| invalid dedup indices | 0 of 172 calls | 0 of 174 calls |
| median seconds per article | 31.3 | 32.0 |
| cheap-tier API cost (51 articles) | $0.29 | $0 (GPU billed per hour) |

Judge (49 articles scored; 11 large articles came back with a malformed judge reply, as in the
2026-10-14 Haiku run):

| | n | routed distinct / unsupported | v2 distinct / unsupported | v2 share (total, median) |
|---|---|---|---|---|
| **cheap tier: the model comparison** | 41 | 509 / 13 | 453 / **32** | 0.89, 0.96 |
| cheap, vendors seen in distillation | 6 | 111 / 1 | 107 / 11 | 0.96, 1.07 |
| cheap, unseen vendors | 35 | 398 / 12 | 346 / 21 | 0.87, 0.88 |
| dense tier: same model, both arms (noise floor) | 8 | 126 / 12 | 132 / 19 | 1.05, 0.98 |

The dense-tier row is the run-to-run noise floor: the same model in both arms still moved
unsupported counts by 12 → 19. The cheap tier's 13 → 32 goes past that, and a re-judge of the 15
articles where v2 had unsupported facts confirmed it (v2 34, routed 11).

**What the unsupported facts look like.** They are not wild hallucinations. They are
over-inference and subject substitution: a requirement restated for a sibling product
("Veeam Software Appliance provides unstructured data backup", repeated for the Infrastructure
Appliance), plausible platform facts the page never states ("Azure Resource Manager controls
whether an operation can be performed through Azure RBAC"), and taxonomy children promoted into
facts ("Client Tags are a type of query tag"). In a citation system that is worse than a missing
fact.

## What it shows

- **Held-out validity did not predict end-to-end quality.** The 96.5% held-out validity
  (same day, `opus-distillation-run-2026-10-18.md`) checks that the JSON is usable. It says
  nothing about faithfulness, and faithfulness is where v2 falls short.
- **Coverage generalises unevenly.** v2 keeps 96% of the distinct facts on the 4 distilled
  vendors, but only 87% on the 9 unseen ones. The distillation sample covered Veeam, Microsoft,
  AWS and Cohesity.
- **No speed gain at this concurrency.** Per-article latency is the same. The GPU's advantage
  is throughput: at 4 articles per arm it is mostly idle.
- **Cost only pays at saturation.** solar-pro4 cost $0.0057 per cheap-tier article here. A $0.42/h
  GPU breaks even at about 74 articles/h, against ~420/h measured at concurrency 48 in the
  2026-10-08 pilot. That makes the GPU about 4× cheaper when kept busy, roughly $0.005 saved per
  article.

## Next, if v2 is pursued

1. **Test `checkpoint-100` first; it is the cheapest lever.** Val loss bottomed there, and the
   final model's over-generation fits the overfitting seen in the loss curve. The test needs a
   Colab export, then `--arms v2 --reuse run/rows.jsonl` (routed is reused): about $0.2 judge
   plus ~$0.5 GPU.
2. **For a v3, broaden the vendors and teach abstention.** Capture Opus on more vendors (the
   unseen-vendor gap), and keep the per-example mix. The notes in
   `distill-colab-log-notes` (1 epoch or repeats ≤2, `load_best_model_at_end`) also apply.
3. **Re-run this trial as the gate.** Adopt when the cheap tier reaches distinct share ≥ 0.95
   (median) and unsupported facts at or below routed's, the chunk-A/B judge's adoption rule.

## Follow-up, 2026-10-21: is the teacher the source?

The same judge (DeepSeek V4 Flash, strict: plausible knowledge the text doesn't state counts
as unsupported) scored every Opus `extract_edges` target against the chunk it came from, and,
on the 45 held-out chunks, Opus, v2 and the old fine-tune blind and shuffled in one list.
Spend $0.12.

| | facts | unsupported | rate |
|---|---|---|---|
| Opus, train targets (243 chunks) | 1,378 | 29 | 2.1% |
| Opus, held-out (45 chunks) | 256 | 2 | 0.8% |
| v2, same 45 held-out chunks | 247 | 11 | 4.5% |
| old fine-tune, 26 of them | 108 | 8 | 7.4% |

Per chunk, v2 had more unsupported facts than Opus on 10 and fewer on 1 (34 tied); a sign
test gives p ≈ 0.006. The gap comes from the student, not from the teacher. Opus's targets
are already clean, so filtering them would change little (29 facts).

One case shows where the student's extra facts come from. v2 wrote the same fact ("At the
Check Configuration step of the Configuration Restore wizard, Veeam Backup & Replication
analyzes...") on two unrelated Veeam chunks. Neither chunk, nor its previous messages, nor
any training record contains it. It appears in a third held-out page. The base model knows
the vendor docs from pretraining and writes them in as if the page said them.

Consequences for another round:
- Cleaning the teacher data won't fix it.
- Early stopping (`checkpoint-100`) is still worth testing: an overfitted student is more
  confident, so it may write in more remembered knowledge. The 45-chunk comparison above
  is a cheap pre-gate before the end-to-end trial.
- If `checkpoint-100` doesn't close the gap, more of the same training is unlikely to. A
  fact-level grounding check in the pipeline (as the entity grounding gate does for entity
  names) would address it for any cheap model, solar-pro4 included.

## Follow-up, 2026-10-21: `checkpoint-100` (the val-loss minimum)

Exported as `carev01/qwen35-4b-graphrag@opus-distill-2026-10-ckpt100`
(`notebooks/export_checkpoint.ipynb`) and run like v2. The routed rows were reused, so
the extracted facts are identical; only the judge reran. Spend $1.96: Vast $0.89 (two
relaunches, one host whose image pull hit Docker Hub's pull limit), extraction $0.76
(the dense articles on gpt-5-mini), judges $0.31.

**Held-out eval:** 98.8% clean (252/255), against v2's 96.5%. On the 45 held-out edge
chunks, unsupported facts: Opus 3, v2 14, ckpt100 13.

**End to end (56 articles scored; 49 cheap tier):**

| cheap tier | n | routed distinct / unsupported | ckpt100 distinct / unsupported | share (total, median) |
|---|---|---|---|---|
| all scored | 49 | 764 / 31 | 761 / 40 | 1.00, 1.00 |
| vendors seen in distillation | 6 | 108 / 3 | 103 / 4 | 0.95, 1.00 |
| unseen vendors | 43 | 656 / 28 | 658 / 36 | 1.00, 1.00 |
| the 40 also scored for v2 | 40 | 475 / 26 | 474 / 25 | 1.00, 1.00 |
| (v2, same 40, 2026-10-20 judge) | 40 | 499 / 11 | 446 / 30 | 0.89, 1.00 |

Dense tier (same model in both arms): 107 / 10 against 127 / 10.

**What it shows:**
- Stopping at step 100 recovers coverage. ckpt100 matches routed's distinct facts,
  including on the 9 vendors absent from training (share 1.00, against v2's 0.87). The
  overfitted final model was what lost them.
- **Unsupported facts are at parity or slightly above, and the judge cannot tell which.**
  The routed facts are byte-identical in both runs, yet the judge scored them 20
  unsupported on 2026-10-20 and 36 on 2026-10-21 (on the 47 articles both scored). Its
  run-to-run noise is larger than the difference under test. Within this run's paired
  calls, ckpt100 has 25 against 26 on the common 40 and 40 against 31 on all 49.
- Speed is unchanged: median 29 s per article against 31 s.
- **Both trials ran without document context:** see the harness fix below.
  Production and the distillation capture have it.

**Verdict:** ckpt100 is a clear improvement on v2 and close to the adoption rule
(median distinct 1.00 ≥ 0.95; unsupported 50 against 41 over all scored, 35 against 36
on the common articles). It is not yet proven. The next run should have document
context, both arms, and two judge passes.

**Harness fix (`cebaa43`):** `haiku_extract_ab.py` wrote only Article and Chapter nodes.
Every arm of every trial so far therefore extracted with no DOCUMENT CONTEXT sentence
and with the entity grounding gate unscoped. Production has both, and the fine-tunes
were trained on prompts that always had the sentence. The harness now writes each
article's real Vendor→Product→Source chain: vendor and product from DocExtractor,
source name from the sample. A free check confirmed all 60 trial articles now resolve a
documented product through the production `_document_context` lookup. `--no-chain`
reproduces the old runs, and `--reuse` refuses rows from the other setting.

## Follow-up, 2026-10-21 (evening): the adoption test, with document context

The same 60 articles, routed vs ckpt100. Both arms extracted afresh with each article's
structural chain written (`cebaa43`), so every prompt carried the DOCUMENT CONTEXT
sentence and the grounding gate was scoped. All 171 captured ckpt100 edge prompts
carried it. The judge ran twice over the same facts. Extraction cost $1.77; the two
judge passes cost $0.29.

**Results (53 articles scored by both passes; each cell is pass 1 + pass 2):**

| | n | routed distinct / unsupported | ckpt100 distinct / unsupported | share (total, median) |
|---|---|---|---|---|
| all scored by both passes | 53 | 768+788 / 67+38 | 757+756 / 51+40 | 0.97, 1.00 |
| cheap tier | 45 | 620+639 / 62+35 | 603+603 / 49+38 | 0.96, 1.00 |
| cheap, vendors seen in distillation | 4 | 28+33 / 4+1 | 33+33 / 1+1 | 1.08, 0.78 |
| cheap, unseen vendors | 41 | 592+606 / 58+34 | 570+570 / 48+37 | 0.95, 1.00 |
| dense tier (gpt-5-mini in both arms) | 8 | 148+149 / 5+3 | 154+153 / 2+2 | 1.03, 1.02 |

- **Unsupported facts: ckpt100 91 against routed 105 over the two passes.** Pass 1 was
  51 against 67; pass 2 was 40 against 38. Per article, ckpt100 had fewer on 15, more on
  18, and the same on 20. That is parity, not a proven gain. The judge noise shows again:
  routed's identical facts scored 67 unsupported in one pass and 38 in the next.
- **Distinct facts: median share 1.00** (adoption needs ≥ 0.95); the total share is 0.97
  overall and 0.96 on the cheap tier.
- **Speed:** median 35.7 s per cheap-tier article against 32.5 s. Cost: the cheap tier
  that ckpt100 replaces cost $0.30 across these 51 articles.
- **One failure (1 of 51 cheap-tier articles):** `0c9570dd`, a 26-chunk Microsoft
  licensing page. On one chunk, a long licence list, ckpt100 enumerated
  service-plan/licence edges until it hit `max_tokens` (16,384). graphiti retried the
  identical call, it truncated again each time, and the article failed with a
  `JSONDecodeError`. routed extracted the page, 53 facts. The article-level dense-table
  router did not catch this page because the table is a small share of a long article.

**Verdict: ckpt100 meets the adoption rule**: median distinct share 1.00, and
unsupported facts at parity (91 vs 105 summed over two passes). Before it takes the
production cheap tier, one fix is needed. A cheap-tier reply that truncates
(`finish_reason == "length"`) should be retried once on the strong tier, as
`dedup_guard` already does for dedup replies, instead of failing the whole article.

**DPO data for a further round** was captured on the same rental: ckpt100 alone on 700
fresh articles (`scripts/distill/dpo_sample.py`, none from the distillation or trial
samples, dense matrices excluded), every call recorded (`--capture`). See
`docs/superpowers/distill-runbook.md` §8 for the pair builder and the Colab notebook.
