# Opus-distillation runbook

Spec: `docs/superpowers/specs/2026-10-17-opus-distillation-design.md`. Plan:
`docs/superpowers/plans/2026-10-17-opus-distillation.md`. Run every command from the
repo root (`.env` supplies production's Neo4j/LLM settings). Everything through
**export** is READ-ONLY on the live graph and costs nothing but teacher-subagent time
(spec section 1: "no API spend... Claude Opus answers the prompts as subagents"). Nothing
in this repo rents a GPU — training happens on the user's own Colab (section 3).

**Keep production's semantic workers idle while rounds run** (pause the
`graph-sync semantic-worker` processes from the first `round` to the last). Every round
reads the LIVE graph -- previous episodes, dedup candidates, the self-match footprint --
and the answer keys hash those prompts' bytes. A worker writing new episodes or entities
between rounds changes the prompts of chunks still in flight (live-graph drift): their
keys no longer match the stored answers, so they come back as new pending prompts and the
teacher answers them twice.

## 1. Pick the sample (Task 3, built)

```bash
uv run --extra dev python -m scripts.distill.harness select --out DIR/sample.json \
    [--n-total 100] [--n-heldout 20] [--seed 20261017] \
    [--vendors Veeam Microsoft AWS Cohesity]
```

`--vendors` must match structural `Vendor.name` values case-insensitively — check the
real names first (`select` fails with the list it did find if a requested vendor has no
article with live episodes; for example it may be "Amazon Web Services", not "AWS").
`DIR` is the per-run store root, e.g. `data/distill-runs/2026-10-pilot/`. Keep
`DIR/sample.json` — the dataset builder's `--articles-from` and the packager's
`sample.json` both come from it later.

## 2. Round / batches / teacher / intake loop

**Pilot:** restrict the round runner to the first few train articles with `--articles N`
(cross-vendor by construction — `select_sample`'s output round-robins vendors, so the
first 5 are a spread, not all one vendor):

```bash
uv run --extra dev python -m scripts.distill.harness round \
    --sample DIR/sample.json --store DIR/store --articles 5
```

This replays every sampled chunk once through graphiti's real `add_episode` pipeline on
a read-only driver; chunks that raise `TeacherNeeded` become `pending`, those are the
prompts a teacher has to answer. `round` exits 1 if any chunk ended `error` in that
round — check `harness status` before going further:

```bash
uv run --extra dev python -m scripts.distill.harness status --store DIR/store
```

Then:

```bash
# Write the pending prompts as batch files, grouped by prompt type:
uv run --extra dev python -m scripts.distill.harness batches \
    --store DIR/store --size 10 --out DIR/batches/round-N/

# Dispatch one Claude-Opus teacher subagent per batch file (Agent tool), briefed with
# scripts/distill/TEACHER_BRIEF.md. Each teacher writes DIR/batches/round-N/batch-XXX.answers.jsonl,
# echoing every line's `ordinal` and `check` next to its key.

# Validate and store the answers (accepted -> store.put; a first rejection is re-queued
# with its reason; a second rejection is dropped). --sample is required: it is how
# intake resolves each key's documented product/vendor for the grounding check.
uv run --extra dev python -m scripts.distill.harness intake \
    --store DIR/store --answers DIR/batches/round-N/ --sample DIR/sample.json
```

`intake` prints its totals and the same counts per prompt type (accepted / requeued /
dropped / unanswered). An answer whose echoed `ordinal`/`check` does not match its key's
batch line is rejected as "answer filed under the wrong prompt". Re-running `intake` on
the same answers directory is safe: a rejection becomes a drop only for the answer to a
re-ask, or for an answer that differs from the one already rejected -- the same rejected
answer seen again leaves the key pending (`already rejected`).

Repeat **round -> batches -> teacher -> intake** until `status` shows every chunk
`complete` or `dropped` (no `pending`, no `error`). graphiti 0.30.1's `add_episode` is
sequential (extract_text -> dedupe_nodes, when any entity needs the LLM -> extract_edges ->
resolve_edge, one per edge, in parallel -> extract_summaries_batch), so a chunk advances one
stage per round: about five rounds, with the parallelism across chunks, not stages.
The CLI pins `PYTHONHASHSEED=0` (re-executing itself once): graphiti renders entity
labels from a set, and under another seed the stage-2 prompts change bytes and answers
never replay. A round also unqueues any pending prompt that no replayed chunk asked for
again ("stale prompts unqueued"), so a teacher is never sent a prompt that can't replay. Re-asked prompts carry
`"previous_answer"`/`"rejection"` in their next batch line, so a teacher fixing a
rejected answer sees exactly what was wrong. `status` (and `report`) also show
**"answered but not replayed in the latest round"**: accepted keys no round has replayed
yet, i.e. with no capture record -- every answer accepted since the last `round`. It must
be 0 before `export` (section 3).

**Pilot report.** After the pilot's last round, write the report the approval gate reads:

```bash
uv run --extra dev python -m scripts.distill.harness report \
    --store DIR/store --batches DIR/batches/ --out DIR/pilot-report.md
```

Per prompt type: distinct keys asked, accepted, re-asked and dropped; the validity rate
(accepted / answered) and re-ask rate (re-asked / answered); prompt and answer sizes
(p50 / p95 / max, in characters and in estimated tokens, chars / 4). Then chunk-state
totals, "answered but not replayed in the latest round", the dedup self-match share, and
a "Controller notes" table the controller fills in by hand: wall time and Agent usage per
batch (the harness cannot see either).

**Full run:** the same loop with `--articles` omitted (all of `sample.json`'s train
articles), driven by a Workflow over rounds with stage-parallel batches, resumable from
the store — a controller concern, not this runbook's; see the plan's "Controller steps"
section. Approval gate: the pilot's validity/re-ask rates and reviewer findings, before
spending teacher time on the full sample.

## 3. Review and export

Run one more `round` first. `export` reads `<store>/capture.jsonl`, and a capture
record for a key only gets written when a round REPLAYS it through the transport
(`teacher.py`'s module docstring) -- `intake` accepting a key does not itself produce
one. Exporting right after `intake`, with no further round, under-counts (`export`'s
`missing` tally catches this and the command exits non-zero unless `--allow-missing`
is passed):

```bash
uv run --extra dev python -m scripts.distill.harness round \
    --sample DIR/sample.json --store DIR/store
```

```bash
# ~5% of accepted answers, stratified by prompt type (at least one per prompt type that
# has any), for one reviewer subagent briefed with scripts/distill/REVIEWER_BRIEF.md:
uv run --extra dev python -m scripts.distill.harness review-sample \
    --store DIR/store --fraction 0.05 --seed 20261017 --out DIR/reviews/

# Copy the capture records whose keys were accepted (production capture format,
# tier=cheap), dropping dropped keys:
uv run --extra dev python -m scripts.distill.harness export \
    --store DIR/store --out DIR/capture.jsonl
```

Read the reviewer's `{"key", "verdict", "issues"}` lines before treating `export`'s
output as final — a reviewer `"issue"` verdict on a prompt type is a reason to re-ask
that type's batches, not to export anyway.

## 4. Build the dataset (Task 5, built)

```bash
uv run --extra dev python scripts/build_extraction_dataset.py \
    --from-capture DIR/capture.jsonl \
    --articles-from DIR/sample.json \
    --loss-basis completion \
    --upsample --upsample-basis example --max-repeat 12 \
    --max-prompt-tokens 20480 --chars-per-token 3.0 \
    --no-repair \
    --out data/ft-opus
```

Notes on these flags, specific to this pipeline (not the graph-derived build's
defaults):

- **`--articles-from DIR/sample.json`** scopes the graph load to exactly the sampled
  articles instead of the whole `group_id` (required at production scale — see
  `load_graph`'s module-docstring note on the ~240k-edge `RELATES_TO` scan) and implies
  `with_embeddings=False` for the capture path (it matches on text, not vectors). It
  also fixes the **train/val split**: each record goes where `sample.json`'s per-article
  `split` says (`heldout` -> `val.jsonl`, `train` -> `train.jsonl`), so the held-out
  articles `eval_finetune.py` scores are exactly the ones `select` held out.
  `--val-fraction` is ignored (an INFO line says so), and a record whose article is not
  in the sample fails the build with a count and examples.
- **`--no-repair`**: intake already validated every accepted answer with zero defects,
  so the builder must not need to repair anything. With repair off, any defect the
  builder finds shows up as a DROP in its capture report instead of being silently
  corrected -- a non-zero drop count here means intake's guarantee did not hold and is
  worth investigating before training.
- **`--loss-basis completion`**, not the builder's default `full`: the Colab notebook
  trains with prompt tokens masked out of the loss (completion-only), and
  `assign_loss_weights`'s docstring is explicit that the weight values depend on which
  masking regime they assume. Mismatched basis doesn't break the upsample-based mix (see
  below), but keeps `meta.loss_basis` honest about what the file actually assumes.
- **`--upsample-basis example`** (v2, 2026-10-20): aim the repeats at EXAMPLE share,
  not token share. Unsloth cannot count `num_items_in_batch` for qwen3.5, so each
  micro-batch loss is a mean over its own tokens; at the notebook's batch size of 1
  every example is one equal unit of gradient, whatever its length. The v1 build (no
  flag: the upsample followed `--loss-basis completion`) hit the mix in completion
  tokens, which under this trainer gave `extract_edges.edge` 2.6% of the gradient
  against 34% and `resolve_edge` 82%. v2 from the same capture: 2,898 train rows
  (edges x4, summaries x4, extract_text x2, dedupe_nodes x2, resolve_edge x1) at
  34/24/16/17/9% of examples; val unchanged (byte-identical, 255 records). Compare
  the report's `n%` column, not `tokens%`, to `target%`.
- **`--upsample --max-repeat 12`**: the notebook does not implement a custom
  weighted-loss trainer (TRL's `SFTTrainer`, which Unsloth wraps, has no per-example
  weight it would honour) — it trains plain SFT on the **upsampled** train split, which
  is `upsample_train`'s whole purpose: repeat TRAIN records so `TARGET_MIX` is
  approached in token share. Check the printed upsample report for any prompt type that
  hit the `--max-repeat` cap and stayed below target; raise the cap or accept the gap.
- **`--max-prompt-tokens 20480`** must equal the notebook's `MAX_SEQ_LENGTH` exactly — an
  example the trainer would silently truncate is dropped here instead (see
  `drop_overlong`'s docstring on why silent truncation is worse than dropping).

Output: `data/ft-opus/train.jsonl`, `data/ft-opus/val.jsonl`, `data/ft-opus/schemas.json`.
The builder does not write `sample.json` itself — copy Task 3's into the same directory
before packaging:

```bash
cp DIR/sample.json data/ft-opus/sample.json
```

## 5. Package and hand off

```bash
uv run --extra dev python -m scripts.distill.package_dataset \
    --in data/ft-opus --out dist/ft-opus.zip \
    [--hf-repo carev01/ft-opus-distill [--hf-revision v2]]
```

Zips `train.jsonl`, `val.jsonl`, `sample.json` (warns and continues if absent — it is
provenance, not training data) and a freshly-written `MANIFEST.json` (counts per
`prompt_name`, the loss-weight mix, token estimates, and the git commit this was
packaged at). With `--hf-repo`, pushes those files to an HF **dataset** repo via
`huggingface_hub` (not a base dependency of this project — install it first, e.g.
`uv pip install huggingface_hub`); the token comes only from `HF_TOKEN` /
`HUGGING_FACE_HUB_TOKEN` in the environment and is never printed. **Private by default
(fail closed)** — `--public` is the explicit opt-out; pass it only when you deliberately
want a public dataset repo (it logs a warning when you do). `--hf-revision` pushes to a
branch (created if missing) instead of `main`: v1 stays on `main`, v2 is on `v2`, and the
notebook's `HF_DATASET_REVISION` picks one.

Hand the user either `dist/ft-opus.zip` (for Google Drive) or the HF dataset repo id —
the notebook's section 4 ("Get the dataset") takes either.

## 6. Train on Colab

Open `notebooks/distill_train.ipynb` in Colab (A100 recommended; an L4 works with the
smaller batch size the notebook's config cell documents). Steps, matching the
notebook's own section numbers:

**The run is resumable (v2, 2026-10-20; v1's Colab runtime died mid-training and lost
everything).** Checkpoints (every `SAVE_STEPS` = 25 optimizer steps, last
`SAVE_TOTAL_LIMIT` = 3 kept), the final adapter and the eval metrics live on Google
Drive under `MyDrive/graph-rag-distill/<RUN_NAME>/`. After a runtime death: reconnect,
**Runtime -> Run all**. Section 9 resumes from the newest checkpoint; if the final
adapter is already there it skips training; section 10 always reloads the adapter from
Drive, so a death during export costs only the export. Change neither the dataset nor
the training config between a crash and its resume. Rotated-out checkpoints land in the
Drive trash, which counts against quota.

1. **Install** (section 1) — the current `pip install unsloth` (which brings its own
   compatible TRL/PEFT/transformers/datasets), then `trl`, `datasets` and
   `huggingface_hub` unpinned. Every pip command's output is shown and a failed install
   raises; the installed versions are printed.
2. **HF token** (section 2) — set a Colab secret named `HF_TOKEN`, or export it in the
   runtime's environment; the notebook never hardcodes one and never prints it.
3. **Configure** (section 3) — pick `DATA_SOURCE` (`"hf_dataset"` with
   `HF_DATASET_REVISION`, or `"drive"` with `DRIVE_DATA_DIR` pointing at the unzipped
   package), and `RUN_NAME` (a new name starts a new run; the same name resumes).
   **Mount Drive** (section 3a) — always, for the checkpoints; it lists what is already
   on Drive for this run.
4. **Get the dataset** (section 4) — downloads or reads `train.jsonl`/`val.jsonl`.
5. **Mix report** (section 5) — prints the realised per-prompt-type EXAMPLE share
   (this trainer's gradient share) and `meta.loss_weight`/`repeat_index`, warns if
   `meta.loss_basis` isn't `"completion"` and if `meta.upsample_basis` isn't `"example"`
   (see step 4's notes above).
6. **Load model + LoRA** (section 6) — base `Qwen/Qwen3.5-4B` (the BASE model, not the
   current production fine-tune `carev01/qwen35-4b-graphrag` — that fine-tune was
   distilled from the OLD, leak-prone extraction instructions and would carry the
   leakage forward; spec section 1). LoRA config, documented in-notebook as **conservative
   Unsloth defaults, CHECKED not recovered (2026-10-17)** — the controller inspected the
   private HF repo `carev01/qwen35-4b-graphrag` directly (README and pushed files): no
   adapter config, no trainer config, no recorded rank/lr/epoch/max_seq for the original
   solar-pro4-distilled run. There is nothing to recover; these are a stated starting
   point:
   - `r=16`, `lora_alpha=16`, `lora_dropout=0.0`, targets = the standard
     q/k/v/o/gate/up/down projections, `use_gradient_checkpointing="unsloth"`.
   - `learning_rate=2e-4`, `num_train_epochs=2`, cosine schedule, `warmup_ratio=0.03`,
     `adamw_8bit`.
   - Batch: `per_device_train_batch_size=1`, `gradient_accumulation_steps=16` (A100) or
     `32` (L4) at `max_seq_length=20480` with gradient checkpointing on — what fits;
     raise/lower on an OOM or on spare headroom.
   Tune these against the held-out eval (this runbook's section 7, "Evaluate", below),
   not on faith.
7. **Format for training, aligned with the thinking-off SERVING prompt** (section 7) —
   qwen3.5's chat template, called with `enable_thinking=False`, renders the GENERATION
   PROMPT for an assistant turn as `"<|im_start|>assistant\n<think>\n\n</think>\n\n"`
   (checked against `carev01/qwen35-4b-graphrag`'s own `chat_template.jinja`,
   2026-10-17) — matching the production server's
   `--default-chat-template-kwargs '{"enable_thinking":false}'`. The empty think block is
   already part of the serving PROMPT, not something the model emits, so training text is
   built as `apply_chat_template(messages[:-1], add_generation_prompt=True,
   enable_thinking=False) + answer + "<|im_end|>\n"` — the SERVING prompt plus the
   answer — never from a plain whole-conversation render, because the same template can
   render a HISTORICAL assistant turn differently from the live generation prompt for
   that turn. This mirrors `scripts/distill/chat_format.py` (also in this repo, covered
   by `tests/unit/test_distill_chat_format.py` so the construction logic is tested
   without a GPU); the notebook duplicates it inline rather than importing it, so Colab
   needs no clone of this private repo.
8. **Verify, fail loudly** (section 8) — before training starts, renders one example's
   serving prompt independently and asserts the constructed training text equals
   `serving_prompt + answer + "<|im_end|>\n"` exactly; an `AssertionError` here means
   this tokenizer's real template does not match section 7's assumption and must be
   investigated before spending any GPU time.
9. **Train** (section 9) — `train_on_responses_only` with a `response_part` marker
   derived from the tokenizer's own template (a live probe render), not hardcoded, so it
   is exactly the serving-prompt suffix section 8 just verified; `packing=False`, eval on
   val during training. The trainer gets the tokenizer as `processing_class=` or
   `tokenizer=`, whichever the installed TRL takes (feature-detected). Before training,
   the cell decodes one example's unmasked labels and asserts they equal the answer plus
   `<|im_end|>` (a trailing newline allowed); on a mismatch it prints both strings and
   raises.
10. **Export** (section 10) — reloads the final adapter from Drive, smoke-tests three
    short held-out prompts (each reply must parse as a JSON object), then saves merged
    safetensors + Q8 GGUF locally and uploads them with `HfApi` to
    `carev01/qwen35-4b-graphrag` revision `opus-distill-2026-10` (a NEW revision — the
    current production revision is untouched and stays available for comparison).


**Exporting an intermediate checkpoint instead of the final adapter:** use
`notebooks/export_checkpoint.ipynb`. It takes `checkpoint-<step>` from Drive, shows that
step's eval loss from the checkpoint's own trainer state, smoke-tests it, and pushes merged
safetensors to its own revision (`opus-distill-2026-10-ckpt<step>`). It never overwrites a
revision that already holds weights.

## 7. Evaluate (ask before any paid GPU serving step)

Two evaluations, spec section 8:

**1. Held-out agreement** — validity rate and per-prompt-type agreement with the
teacher, on the 20 held-out articles:

```bash
uv run --extra dev python scripts/eval_finetune.py \
    --val data/ft-opus/val.jsonl --model <served-name> \
    --base-url <server>/v1/chat/completions --per-prompt 0
```

`--per-prompt 0` scores EVERY held-out record; the default (`--per-prompt 25`) scores a
seeded subsample of 25 per prompt type, which is not the held-out set the pass criterion
names. `--base-url` is the chat-completions endpoint serving the exported revision (the
user's Colab runtime over a tunnel, or a GGUF server, with the user's approval for any
paid GPU); it defaults to the in-house `srv-llm` endpoint the old fine-tune was scored on.

**2. End to end** — `scripts/extraction_proof.py` with the student serving the cheap
tier. **This is a PAID run: get the user's explicit authorisation before running it**
(it calls the production LLM tiers -- its own docstring puts the full sample at about
$0.30 per run -- and the three runs below are on top of any GPU serving):

```bash
CHEAP_LLM_BASE_URL=<student-server-url> CHEAP_LLM_MODEL=<served-name> \
    uv run --extra dev python scripts/extraction_proof.py \
    --sample-affected 40 --out docs/superpowers/opus-distill-proof-<date>.md
```

Run it three ways — against the student, against solar-pro4 (current `.env` cheap
tier), and against the old fine-tune (`carev01/qwen35-4b-graphrag`'s current, non-`opus-
distill-2026-10` revision) — and compare: foreign-unnamed-subject facts, grounding
calls/dropped/injected/renamed, facts per article, and `RestoresTo` edges to each restore
page's destination.

### Pass criteria (spec section 8)

- **Validity**: at least 99% on held-out prompts (`eval_finetune.py`).
- **At least as good as solar-pro4** on every `extraction_proof.py` metric above.

Write the results to a report under `docs/superpowers/` (not this runbook — this is the
procedure, the report is the evidence).
