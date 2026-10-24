# dpo4: coverage-balanced DPO pairs, Claude labelling, gate on Vast -- design

**Date:** 2026-10-24. **Status:** approved in conversation, section by section.
**Builds on:** `scripts/distill/dpo_pairs.py`, `scripts/distill/dpo_sample.py`,
`notebooks/distill_dpo.ipynb` (dpo3, commits `b4c4629`, `1cdd707`, `e0f2f16`),
`docs/superpowers/distill-runbook.md` section 8.

## 1. Why another round, and why this is the last one

dpo3 (lr 5e-5, deletion-only pairs) learned to stop writing unsupported edges, but it got
there partly by writing less. On the continuation gate, checkpoint-10 wrote 0.71 of
ckpt100's removed edges (pass, bar <= 0.75) and 0.85 of its kept edges (fail, bar >= 0.90).
After step 10, `logps/chosen` collapsed (-45 -> -158 by step 30) while val loss kept
falling. The cause is the data: every pair's `chosen` was `rejected` minus edges, so
"write fewer edges" scored on every pair.

Three fixes:
1. pairs that reward coverage as well as precision, balanced so length carries no signal;
2. a val set big enough to resolve a 15-20% effect (~150 A pairs, up from 39);
3. labels from Claude subagents instead of the DeepSeek judge. Plan usage, not API spend,
   and a check on dpo3's labels.

**Stop rule, fixed now:** if dpo4 fails the gate (section 6), DPO stops and ckpt100
(`carev01/qwen35-4b-graphrag@opus-distill-2026-10-ckpt100`) stays the student. There is
no dpo5.

## 2. Pair types

Every pair is built from ckpt100's own reply to a captured `extract_edges.edge` prompt.
That keeps the data on-policy. Each pair differs from its reply in one kind of change only.

| Type | `rejected` | `chosen` | Teaches |
|---|---|---|---|
| **A: drop unsupported** | the reply as written | the reply minus its confirmed-unsupported edges | don't invent |
| **B: keep supported** | a clean reply minus one verified `stated` edge | the clean reply | don't drop real edges |
| **C: add missed** | a clean reply as written | the reply plus Opus's verified missed edges, appended after its last edge | keep going while more is supported |

Definitions:
- **Confirmed unsupported:** labelled `unsupported` (section 3) AND confirmed by the Opus
  re-check.
- **Disputed:** labelled `unsupported` but the re-check calls it supported. It stays in
  both `chosen` and `rejected`, so it is neutral, just as a single-pass `unsupported` was
  under the old both-passes rule.
- **Clean reply:** no confirmed-unsupported and no disputed edge.
- **Verified:** the edge carries a quote that appears word for word in the prompt's
  CURRENT MESSAGE (whitespace runs normalised to one space). The check is code, not a model.

Construction rules:
- **One pair per reply,** so no prompt appears twice. Replies with a confirmed-unsupported
  edge go to A, as `dpo_pairs.build_pair` does today: no pair when every edge would be
  removed. A clean reply with at least one verified missed edge goes to C. Every other
  clean reply with at least 2 verified `stated` edges is a B candidate.
- **B removal:** the dropped edge is chosen by a seeded RNG among the reply's verified
  `stated` edges, keyed on `call_id`. Any position is allowed, including the last.
- **C, what counts as missed:** an Opus edge is missed when ckpt100's reply has no edge
  between the same two entities, in either direction. Both names must appear in the
  prompt's entity list exactly. The edge must validate against the call's
  `response_format` schema. Edges that fail any check are dropped. A reply left with none
  is not a C pair.
- **Serialisation:** `json.dumps(payload, ensure_ascii=False)`, edge order kept, as now.
- **Meta:** each pair's `meta` gains `type` (`"A"|"B"|"C"`), `labeller` (`"sonnet"|"opus"`),
  and the changed edges with their quotes and labels.

**Length balance (fixed before training).** Per split, the mean of
`len(chosen edges) - len(rejected edges)` over all pairs must be within +/-0.1 edge. That
works out as follows:
- an A pair counts -k for k removed edges;
- a B pair counts +1;
- a C pair counts +m for m added edges.

To balance:
1. Keep every A and C pair.
2. If the sum is still negative, add B candidates in seeded order until the mean reaches
   the band.
3. If C alone overshoots, drop C pairs in seeded order instead.

The builder asserts the band and reports the final mean.

**Split.** By article group (`group_of` + `split_of`, sha256), with `--val-fraction 0.2`,
up from 0.1. That gives about 150 val A pairs from the old and new capture together. The
hash is unchanged, so dpo3's 39 val articles stay in val. The builder also writes
`val_eval.jsonl`: a fixed, seeded subset of 120 val pairs, split across types in
proportion, for the trainer's evals (section 5).

**Expected scale** (reported exactly at step 5 of section 8): about 630 A, plus B and C,
about 1,250 training pairs in all. Val is about 150 A plus the B and C pairs from val
articles.

## 3. Labelling with Claude subagents

The DeepSeek judge (`judge_pass`, two passes, paid) is replaced by Claude Code subagents
working in batches of about 10 tasks under a task brief, as the Opus teacher run did.
Answers are recorded in an `AnswerStore` (`scripts/distill/teacher.py`): append-only,
crash-safe, and resumable.

Three task kinds, each with its own brief and schema:

| Task | Input | Output | Model | Volume |
|---|---|---|---|---|
| `dpo_label` | prompt text (PREVIOUS / CURRENT MESSAGE, DOCUMENT CONTEXT) + ckpt100's edges, shuffled with a per-task seed | per edge: `stated` / `previous_only` / `unsupported` (+ category, as in `INSTRUCTIONS`), a quote from CURRENT MESSAGE for `stated`, `why` <= 15 words | **Sonnet**, if the calibration passes; else Opus | every reply: 2,053 existing + about 2,900 new |
| `dpo_recheck` | prompt text + ONLY the edges labelled `unsupported`, with no first label shown | per edge: supported (with quote) / unsupported (+ category) | **Opus**, a fresh subagent | about 5% of edges |
| `dpo_missed` | prompt text + entity list + ckpt100's edges | edges ckpt100 missed, in the call's edge schema, each with a quote from CURRENT MESSAGE | **Opus** | about 600 clean replies, seeded sample over train and val |

Rules:
- A label's quote is checked the same way as an added edge's. A `stated` label whose quote
  fails the check is downgraded to "unverified": that edge can't be the dropped edge in a
  B pair, and the reply can't be a C pair.
- An answer that fails schema validation is re-asked once, then dropped and logged, never
  guessed.
- All existing replies are re-labelled, not just the new ones. That puts train and val on
  one standard, and gives an Opus-vs-DeepSeek comparison (section 7).

**Calibration, fixed before running:**
1. Sonnet and Opus each label the same 100 existing replies: 50 drawn from replies with
   at least one DeepSeek both-pass `unsupported` edge, 50 from the rest, seeded.
2. Recall = |Sonnet `unsupported` ∩ Opus `unsupported`| / |Opus `unsupported`|.
3. **Sonnet labels the full set only if recall >= 0.90.** Otherwise Opus labels everything.

False positives from Sonnet are tolerated, because the Opus re-check sees every one. The
calibration report also gives the per-verdict agreement matrix.

Distillation data stays outside the public repo, under
`graph-rag-distill/2026-10-24-dpo4/`.

## 4. Capture

- **Sample:** `dpo_sample.py --articles 1000` with a new `--seed`, excluding the
  distillation `sample.json`, the trial `d6_sample.json` and
  `2026-10-21-dpo/dpo_sample.json`.
- **Capture:** `haiku_extract_ab.py --arms v2 --capture --judge none`, with ckpt100 on vLLM
  on a Vast GPU, the same as runbook section 8 step 2. Expect about 3.3 h of wall time,
  about $3 of GPU, and about 2,900 judgeable edge replies.
- **Smoke first** (CLAUDE.md cost rule): run 2 articles, then assert that the capture file
  has `extract_edges.edge` lines from model `qwen35-graphrag` and that none of the
  articles is in an exclude file. Refuse the full run otherwise.
- **Authorisation:** given in advance for this capture, about $3 (2026-10-24).
- **Afterwards:** the Vast instance is **stopped, not destroyed** (section 8, step 9).

## 5. Training notebook (`notebooks/distill_dpo.ipynb`, dpo4)

Unchanged: causal_conv1d kernel off for training, `torch._dynamo.config.disable = True`,
GradGuard, `eval_on_start`, loss `["sigmoid", "sft"]`, BETA 0.1, 2 epochs, warmup 0.1,
cosine schedule, effective batch 8, EVAL_STEPS 10 (equal to the save interval).

Changed:
- `HF_DATASET_REVISION = "dpo4"`, `HF_MODEL_REVISION = "opus-distill-2026-10-dpo4"`.
- **`LEARNING_RATE = 2e-5`.** At 5e-5, dpo3 started displacing chosen replies by step 20.
  At 5e-6, dpo1 had no effect. One value only, no sweep.
- **`NLL_ALPHA = 2.0`.** The `sft` term is a per-token mean, while DPO's log-probs are sums
  over the reply. At 1.0 it was too weak to stop the displacement.
- **Eval set:** `val_eval.jsonl` (120 pairs), not the full val set.
- **Checkpoint selection by likelihood:** `load_best_model_at_end = False`,
  `SAVE_TOTAL_LIMIT = None`. The selection rule, fixed before training:
  - **use the last checkpoint whose `eval_logps/chosen` is >= its step-0 value - 1.0 nat;**
  - a `TrainerCallback` stops training at the first eval below that line;
  - if no checkpoint after step 0 qualifies, the round fails with no gate.
- **Hand-off:** a cell applies the rule, prints the step it chose and the logps trajectory,
  and uploads only that checkpoint's adapter (`adapter_model.safetensors`,
  `adapter_config.json`) to the **private** HF repo `carev01/qwen35-4b-graphrag-dpo-adapters`
  (created `private=True` if missing), revision `dpo4-ckpt<step>`.
- **The in-notebook gate and export are removed.** The gate runs on Vast (section 6), and
  nothing goes to the public model repo from Colab.

Size: about 1,250 pairs × 2 epochs / 8 ≈ 310 steps, against dpo3's 71. The likelihood
callback is expected to stop the run earlier.

## 6. Gate on Vast (`scripts/distill/dpo_gate.py`)

Runs on the restarted Vast instance. Its output is `gate.json`; its exit code is 0 on a
pass and 1 on a fail.

1. **Merge:** download ckpt100 (bf16) and the adapter (the token comes from the
   environment and is never written to disk). Merge with plain `peft` (`merge_and_unload`)
   and save the result locally.
   - vLLM's LoRA support for Qwen3.5's hybrid layers is unconfirmed, so the adapter is not
     served directly.
   - The merged bf16 model is also what would ship.
2. **Prove the adapter is engaged:** assert that some LoRA target-module weight differs
   between the merged model and ckpt100. Then smoke-test 2 val pairs and refuse to continue
   if both models' continuations are token-identical.
3. **Generate:** vLLM offline, bf16, one process per model, one after the other.
   - For each val pair, from the token prefix that `chosen` and `rejected` share (prompt
     rendered by `chat_format`, the serving prompt).
   - Greedy (temperature 0), `max_tokens = min(len(rejected tail) + 256, 4096)`.
4. **Score each pair:** by exact (source, target, relation) triples, using the dpo3
   `rest_from_split` logic, generalised.
   - `removed` = rejected − chosen;
   - `added` = chosen − rejected;
   - `kept` = chosen minus the edges already complete before the split;
   - plus whether the reply is valid JSON.
5. **Pass bar, unchanged from dpo3, on the A val pairs:**
   - the DPO model's removed edges written <= 0.75 × ckpt100's;
   - its kept edges written >= 0.90 × ckpt100's;
   - valid JSON replies >= ckpt100's;
   - ckpt100's removed edges written >= 10.
6. **Reported, not part of the bar:**
   - bootstrap 95% CIs (pairs resampled, seeded) for the removed ratio, the kept ratio and
     their quotient (selectivity), plus the down/same/up counts and sign test, as in the
     dpo3 analysis;
   - B: the dropped edge written back, DPO vs ckpt100;
   - C: added edges written, DPO vs ckpt100.
7. **One checkpoint only:** the one section 5's rule picked.
8. **Export, after a pass and only with authorisation** (`--push REVISION`): push the
   already-merged weights to `carev01/qwen35-4b-graphrag@opus-distill-2026-10-dpo4`, with
   the export notebook's checks:
   - refuse a branch that already has its own commits;
   - 3 held-out prompts must return JSON;
   - the Hub's shards must equal the merged index.

ckpt100's bf16 numbers are measured fresh. The 4-bit Colab numbers from dpo3 are not
reused.

## 7. Reports

- **`labels_report.json`:**
  - calibration recall and agreement matrix;
  - Opus-vs-DeepSeek agreement on the 2,053 existing replies, in particular how many of
    DeepSeek's 554 both-pass `unsupported` edges Opus confirms;
  - verdict and category counts;
  - quote-check failure rates per task kind.
- **`summary.json`** (pair builder):
  - pairs per type per split;
  - the length-balance mean per split;
  - the val sizes;
  - how many replies were skipped, and why.
- **`gate.json`:** totals, per-pair rows, and the section 6 statistics.

## 8. Order of work, cost, stopping points

| # | Step | Cost | Stops for the user? |
|---|---|---|---|
| 1 | Code: sampler exclusion run, labelling briefs and task files on `AnswerStore`, quote check, A/B/C builders with balance, split and `val_eval`, notebook changes, `dpo_gate.py`; unit tests | $0 | no |
| 2 | Sonnet-vs-Opus calibration (100 replies); the 0.90 bar decides the labeller | plan usage | report only |
| 3 | Capture about 1,000 articles (smoke first), then **stop** the instance | about $3 | no (authorised 2026-10-24) |
| 4 | Labelling (about 5,000 replies), Opus re-check, Opus missed edges (about 600) | plan usage | no |
| 5 | Build pairs; reports from section 7 | $0 | **yes: review before upload and training** |
| 6 | Package and push pairs (`ft-opus-distill@dpo4`, private); user trains on Colab | Colab | user hands back the result |
| 7 | Gate on Vast | about $1 | **yes: pass/fail** |
| 8a | Pass: export, then the adoption trial on the same 60 articles as ckpt100 (runbook section 7), priced before it runs | priced then | **yes: authorise** |
| 8b | Fail: DPO stops; ckpt100 stays | $0 | -- |
| 9 | Destroy the Vast instance, only when the round is over (after 8a's extraction, or at 8b), confirmed with the Vast CLI | -- | -- |

Cash before the adoption trial: about $4, plus a few cents a day of disk while the
instance is stopped.

## 9. Testing

Unit tests, hermetic, in `tests/unit/test_distill_dpo_pairs.py` and a new
`tests/unit/test_distill_dpo_gate.py`. They cover:
- the quote check (exact, whitespace-normalised, a miss);
- A pairs with disputed edges left in both sides;
- B pairs: seeded removal, no unverified edges, at least 2 verified edges;
- C pairs: the missed-edge test in both directions, entity names not in the list, schema
  failures, appended order;
- the balance assertion and seeded fill;
- the split with `--val-fraction 0.2`, including dpo3's val groups staying in val;
- `val_eval` proportions;
- calibration recall;
- the gate's triple scoring for A, B and C pairs, and the pass rule;
- the selection rule over a synthetic `log_history`.

The notebook is exercised by the existing CPU harnesses (`scratchpad/nbcheck/`). Labelling
briefs are smoke-tested on 2 tasks per kind before any batch run.
