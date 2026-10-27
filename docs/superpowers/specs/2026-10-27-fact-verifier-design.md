# Fact verifier: a fine-tuned 4B model that checks extracted facts -- design

Status: draft for review, 2026-10-27. Agreed in conversation the same day: a fine-tuned
Qwen verifier only (a Gemma arm only if Qwen disappoints), offline measurement before any
production code.

## 1. Why

ckpt100 (`carev01/qwen35-4b-graphrag@opus-distill-2026-10-ckpt100`) already meets the
adoption rule against the routed cheap tier, but it still writes facts its chunk does not
support: Opus confirmed 1,958 of about 27,800 of its edges as unsupported (over-inference
913, subject substitution 456, wrong detail 438, taxonomy-as-fact 137). Four DPO rounds
failed to make it more selective; the last (dpo4, 2026-10-26) cut supported edges as much
as unsupported ones.

A verifier separates the two jobs. Checking one fact against a chunk is easier than
writing every fact, and the dpo4 labels are a ready training set. A strong-model verifier
would cost more than the extraction it checks; a fine-tuned 4B verifier runs on the same
rented GPU as ckpt100, so it costs GPU time only.

## 2. Goal and non-goals

Goal: a LoRA adapter on `Qwen/Qwen3.5-4B` that, given one `extract_edges.edge` call's chunk
and reply, labels every fact, and passes a pre-registered offline bar (section 6) on
articles it never saw.

Non-goals here: the production hook, serving, and the end-to-end trial. They follow only
on a pass, in a separate spec (section 8). No Gemma arm. No API spend.

## 3. Data

**Source** (all in `graph-rag-distill/`, outside the public repo): both ckpt100 captures
(`2026-10-21-dpo/capture-run/capture-v2.jsonl`, `2026-10-24-dpo4/capture-run/capture-v2.jsonl`;
4,641 edge replies) and the dpo4 Opus stores `store/label-opus` and `store/recheck-opus`.

**Final label per edge**, exactly as `dpo_coverage.classify` / `final_label` compute it:
`stated` (24,711), `confirmed` = unsupported after the recheck (1,958), `previous_only`
(627), `disputed` = first label unsupported, recheck supported (463).

**One verifier record per reply:**
- **Prompt**: `VERIFIER_INSTRUCTIONS` (below) + PREVIOUS MESSAGES + CURRENT MESSAGE +
  DOCUMENT CONTEXT + numbered FACTS, built with the same `_context_prompt` layout the
  labellers saw, so production can build it from the same call. Facts are listed in
  reply order (production order), not shuffled.
- **Excluded facts**: `disputed` ones, and `stated` ones whose Opus quote fails
  `quote_ok` (0.09%). They are left out of the FACTS list, so the model never trains on
  an uncertain label. A reply left with no facts is skipped.
- **Target**: `{"labels": [{"n": 1, "verdict": "stated", "quote": "..."}, ...]}`, one
  entry per listed fact, in order. `stated` carries Opus's quote; `unsupported` carries
  `category` (the recheck's when present, else the first label's); `previous_only`
  carries neither. No "why" field: it costs tokens and nothing reads it.

`VERIFIER_INSTRUCTIONS` is `LABEL_INSTRUCTIONS` with the output paragraph changed to the
target format above. The definitions of the verdicts and categories stay word for word.

**Split**: by article group, with dpo4's hash (`group_of` + `split_of`, val fraction 0.2).
The 20% becomes **test**, scored once at the end. Of the remaining groups, a seeded 5%
becomes **dev**, for eval loss during training. Test articles are the dpo4 val articles,
so neither the verifier nor any DPO checkpoint trained on them.

**Builder**: `scripts/distill/verifier_data.py` writes `train.jsonl`, `dev.jsonl`,
`test.jsonl` (chat records: system + user + assistant, as `distill_train.ipynb` reads) and
`summary.json` (counts per split and per final label). It packages through
`package_dataset.py` to `carev01/ft-opus-distill@verifier1` (private).

## 4. Training (Colab notebook `notebooks/distill_verifier.ipynb`)

Derived from `distill_train.ipynb` (SFT), with every guard learned since:
- **Install cell** from `distill_dpo.ipynb`: the hub causal_conv1d kernel is NOT used
  for training (its backward is broken at real sizes), and `torch._dynamo` is disabled.
- **Serving-prompt check**: the training text equals the thinking-off serving prompt +
  answer + end of turn (section 8 of `distill_train.ipynb`), and the loss mask covers
  only the answer.
- **Gradient guard** from `distill_dpo.ipynb`: stop on a non-finite gradient before the
  optimiser step.

Settings: base `Qwen/Qwen3.5-4B`, 4-bit, LoRA r=16/alpha 16 on the same seven modules,
lr 2e-4, cosine, warmup 0.03, 2 epochs, completion-only loss, `MAX_SEQ_LENGTH` 20,480.
Eval and save every 25 steps, `save_total_limit=None` (SFT v2's best checkpoint was
rotated out). The selected checkpoint is the lowest dev loss; SFT v2 overfitted after
about half an epoch, so the run may be stopped once dev loss has risen for three evals in
a row.

**Prediction**: the notebook's last section loads the selected checkpoint, generates
greedy, thinking-off replies for every test prompt, and writes `test_predictions.jsonl`
(call id + raw reply) to Drive and, with the adapter, to the private repo
`carev01/qwen35-4b-graphrag-verifier` (revision `verifier1-ckpt<step>`). Scoring is local.

## 5. Scoring (`scripts/distill/verifier_eval.py`, local, free)

Each prediction is parsed. A reply that does not parse, or has the wrong number of
entries, keeps every fact (production fails open) and is counted.

**Drop policies**, both pre-registered:
- **P1 (primary):** drop a fact when its verdict is `unsupported`, or when it is
  `stated` and its quote fails `quote_ok` against the CURRENT MESSAGE.
- **P2:** drop only on `unsupported`.

`previous_only` is kept under both. Those facts are true, and their article is the same,
so citations stay correct.

**Metrics** against Opus's final labels:
- **catch rate**: share of `confirmed` facts dropped;
- **false-drop rate**: share of `stated` facts dropped;
- `previous_only` dropped share and the parse-failure share, reported;
- catch rate per category;
- bootstrap 95% CIs, resampling replies.

**Baseline**, report-only: a code-only lexical check. A fact is dropped when fewer than a
threshold share of its distinctive words (numbers, versions, words outside the entity
names and a stopword list) appear in the CURRENT MESSAGE. The threshold is tuned on train
and reported on test.

## 6. Pre-registered bar

A policy passes on test when all of these hold:
- catch rate >= 0.50;
- false-drop rate <= 0.02;
- parse failures <= 1% of replies.

If both pass, P1 is used. The point estimate decides, and the CIs are reported next to
it.

If neither passes, one follow-up run is allowed, and only for a recall miss (catch < 0.50
with false drops <= 0.02): replies containing a `confirmed` fact are upsampled x3 in
train. Any other failure, or a second miss, ends the Qwen arm; the Gemma arm is then the
next decision for the user.

## 7. Cost

- No API spend.
- Colab: about 3,500 train examples per epoch (80% of 4,641 replies, less dev) across 2
  epochs, comparable to SFT v2 (about 2,900 per epoch, 7 h on its GPU); early stopping is
  expected to cut it.
- Packaging and scoring are local.

## 8. After a pass (separate spec, not built here)

- Serve the adapter as a second LoRA on the vLLM process that serves ckpt100.
- Add an `extract_edges.edge` hook in `dedup_guard` beside the entity grounding gate:
  build the verifier prompt from the call, drop per the passing policy, fail open on a
  bad reply, and add counters on the worker batch line.
- Behind a flag, re-run the end-to-end adoption trial against ckpt100 alone.

## 9. Tests

- `verifier_data`: final-label mapping; disputed and failed-quote facts excluded and the
  numbering compacted; reply order kept; the target format; the split matches dpo4's
  `split_of`; dev drawn only from train groups.
- `verifier_eval`: parsing (good, bad JSON, wrong count -> keep all); P1 versus P2 on a
  fabricated quote; the metrics on a hand-built set; the bar decision; seeded bootstrap.
- Notebook: the scratchpad harness runs its data, chat-format and trainer cells on CPU
  with a tiny model, as for the DPO notebook.
