# Distilling the extraction tier from Claude Opus, via subagents — design

**Status:** approved in conversation 2026-10-17. Teacher: Claude Opus, run as subagents in Claude
Code, not through the API. Approach A (chained prompts built from the live graph). Student:
qwen3.5-4b. Sample: stratified plus hard cases. Pilot first, then the full run on approval.

## 1. Why

The local extraction model (qwen3.5-4b, private HF `carev01/qwen35-4b-graphrag`) was distilled
from `upstage/solar-pro4` captures made with the pre-2026-10-17 extraction instructions. Those
instructions leaked their own example entities into facts: forced "AWS Backup" subjects and
invented vocabulary, in about 16% of articles (`docs/superpowers/extraction-grounding-proof-2026-10-17.md`).
The student learned that behaviour.

The new ontology needs fresh targets. It adds a per-chunk DOCUMENT CONTEXT, leak-proof
instructions, NAMING_RULES, RestoresTo and the grounding gate. The user wants the strongest
teacher and no API spend, so Claude Opus answers the prompts as subagents in this environment.

## 2. Scope

The student serves all five graphiti prompt types of the cheap tier. Every one is distilled, with
the old target mix as the starting weights:

| prompt | old token share |
|---|---|
| `extract_edges.edge` | 34% |
| `dedupe_edges.resolve_edge` | 25% |
| `extract_nodes.extract_summaries_batch` | 18% |
| `extract_nodes.extract_text` | 13% |
| `dedupe_nodes.nodes` | 10% |

**Out of scope:**

- changing the production pipeline;
- the held re-extraction of ~3,300 articles;
- serving changes beyond swapping the model behind the cheap tier;
- a 9B student.

## 3. Data flow

1. **Harness** (`scripts/distill/`). It reads the sample's chunks read-only from production
   `:Episodic.content`, the exact strings production extracted, so chunking matches. It also reads
   each article's structural chain for `document_context`.
2. **Prompt construction.** Every prompt is built with graphiti 0.30.1's real `prompt_library`
   functions, the live `EXTRACTION_INSTRUCTIONS` (plus `CHEAP_TIER_SALIENCE`, as the cheap tier
   sends it) and `document_context(...)`. The harness then replays the client-side message changes
   (`OpenAIGenericClient` suffix and `_clean_input`) and the `cache_layout` split, exactly as the
   existing dataset builder does, so training prompts are what the student receives in production.
   The builder's byte-exactness checks carry over as tests.
3. **Stages**, run breadth-first over all chunks so each stage parallelises:
   - **stage 1:** `extract_nodes.extract_text`;
   - **stage 2:** `dedupe_nodes.nodes` and `extract_edges.edge`, built from the validated stage-1
     answer. Dedup candidates come from the live graph by the same similarity search production
     runs. The edge prompt carries the teacher's entity list;
   - **stage 3:** `dedupe_edges.resolve_edge` and `extract_nodes.extract_summaries_batch`. Related
     edges and existing summaries come from the live graph. The invalidation-candidate search stays
     off, as in production (`ingest_detect_contradictions=False`).

   Only entity → fact is chained on teacher output. Everything else uses production state
   read-only. The live graph is never written.
4. **Teacher batches.** Each stage writes JSONL batch files of about 10 prompts. A subagent answers
   one batch into a response file.
5. **Validation.** Every answer is checked, and a failure is re-asked once with the reason, then
   dropped and counted:
   - it parses against graphiti's response model;
   - its index ranges pass (the existing `validate_capture`);
   - for `extract_text`, the grounding gate runs as a **validator**: an answer the gate would
     filter (an ungrounded entity) or need to fix (a missing documented product, an alias that
     should be renamed) is rejected, never silently corrected. The student learns targets that need
     no repair.
6. **Output.** Validated pairs are written in the existing capture format, so
   `build_extraction_dataset.py --from-capture`, its repair audit, `assign_loss_weights`,
   `--upsample`, `refilter_dataset.py` and `eval_finetune.py` are reused.

## 4. The teacher brief

A fixed brief, versioned in the repo, is given to every subagent:

- answer each prompt ONLY from its own text, with no outside knowledge;
- return exactly the JSON schema graphiti expects for that prompt, and nothing else;
- follow every rule in the prompt (naming rules, the documented-product rule, RestoresTo direction,
  name entities as the text writes them);
- each answer is a training target, so it must be the best answer the prompt allows, not a
  minimal one.

**Reviewer:** a separate subagent checks a random 5% of validated answers against their chunk for
missed or invented facts, wrong RestoresTo direction and wrong subjects. Its findings are recorded
and can refine the brief. They are reported, not auto-applied.

## 5. Sample

About 100 articles: about 80 train and about 20 held out, split by article with no overlap.

- **Strata:** vendor and product family across Veeam, Microsoft, AWS and Cohesity, in proportion
  to ingested articles but with a floor of 5 articles per vendor.
- **Hard cases, oversampled:** chunks with no product named (subjectless lists), restore and
  destination pages, dense tables, cross-product pages, and articles from the 2026-10-17 affected
  set.

Selection is deterministic and recorded, so the dataset can be rebuilt.

## 6. Pilot, then scale

**Pilot:** 5 articles (about 40 chunks, about 20 subagent runs). It reports:

- validity and re-ask rates per prompt type;
- the reviewer's findings;
- prompt and answer token sizes;
- wall time and usage per batch.

The user approves the full run from these numbers. If the pilot is not clean, the harness and
brief are fixed and the pilot is repeated.

**Full run:** the remaining articles, staged as in §3. It is orchestrated as a subagent workflow
with resume: answers already validated are never re-asked. That is about 650 chunks, 3,000–4,000
teacher answers and 300–400 subagent runs, which costs plan usage but no API money.

## 7. Train

qwen3.5-4b, Unsloth/TRL, run by the user on **Google Colab** (user decision, 2026-10-17), with
context of at least 20k tokens. Starting hyperparameters come from the previous run. The
deliverables from this repo are:

- a Colab notebook (`notebooks/distill_train.ipynb`) that mounts the dataset, trains with the
  dataset's loss weights, and exports merged safetensors (plus a Q8 GGUF) to a new private HF
  repo revision;
- the train/val JSONL packaged for upload: a private HF dataset repo, or Google Drive, as the user
  chooses at run time;
- a short runbook.

Nothing in this repo rents a GPU. The old model stays available for comparison.

## 8. Evaluate

Two evaluations:

1. **Held-out agreement.** On the 20 held-out articles, compared with the teacher: validity rate
   (`eval_finetune.py`) and entity/fact agreement per prompt type.
2. **End to end.** `scripts/extraction_proof.py` with the student serving the cheap tier (vLLM,
   thinking off). The serving host is chosen at evaluation time: the user's Colab runtime exposed
   over a tunnel, or the existing GGUF server, with user approval for any paid GPU. It is compared with the current cheap tier (solar-pro4) and the
   old fine-tune on the same articles. Metrics: foreign-unnamed-subject facts, grounding
   calls/dropped/injected/renamed, facts per article, and RestoresTo to each restore page's
   destination.

**Pass:** validity of at least 99% on held-out prompts, and at least as good as solar-pro4 on
every proof metric. Results go into a report under `docs/superpowers/`.

## 9. Risks

- **Teacher drift across subagents.** A fixed brief, deterministic validation and the reviewer
  sample limit it.
- **Prompt fidelity.** The builder's byte-exact replay plus tests on the new instruction and
  document-context path.
- **Context length.** Prompts reach about 20k tokens (`previous_episodes`). `refilter_dataset.py`
  can cap them if training memory requires it.
- **Live-graph candidates still contain pre-fix contaminated entities.** That is acceptable: the
  teacher decides duplicates on their merits, and the candidates are only context.
