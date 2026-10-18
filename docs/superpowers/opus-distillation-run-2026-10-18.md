# Opus distillation run — 2026-10-18

Spec: `specs/2026-10-17-opus-distillation-design.md`. Plan: `plans/2026-10-17-opus-distillation.md`.
Runbook: `distill-runbook.md`. The store, batches, reviews and dataset live outside this
public repo (`~/Documents/Shared/projects/graph-rag-distill/2026-10-18/`).

## Result

| | pilot (5 articles) | full run (95 more) | total |
|---|---:|---:|---:|
| chunks | 18 | 270 | 288 |
| accepted answers | 117 | 1,578 | 1,695 |
| re-asked | 0 | 7 (all extract_text, all fixed on re-ask) | 7 |
| dropped | 0 | 0 | 0 |
| teacher subagent runs | 11 | 116 | 127 |
| teacher tokens (subagent, approx.) | ~1.0M | ~11M | ~12M |

Answers per prompt type: resolve_edge 819, extract_text 288, extract_edges 288, dedupe_nodes 157,
extract_summaries_batch 143. Export: 1,695 records, 0 missing; 1,415 in the "moved" cache
layout (production steady state), 280 "unmoved" (each prompt type's first calls in a process).

Production workers were at 0 replicas for the whole run (no live-graph drift). Wall time of
the full run: ~3h15m — rounds of 10–44 min of harness replay, then up to 20 parallel teachers.

## Review

- Pilot: 20% sample (24 answers), 24 ok.
- Full: 5% sample (84 answers, three reviewers), 82 ok, 2 issues: a pilot-era resolve_edge
  answer that called "existing fact + a command name" a duplicate (before the brief's rule
  existed), and an extract_text answer that took "PostgreSQL" from a section heading only.
  Both left in place (2 of ~1,700 targets).
- Checked specifically and clean: fact subjects when DOCUMENT CONTEXT and the chunk name
  different products; RestoresTo direction (all sampled restore edges point at the
  destination); invented entities; summary length.
- Residual noise the validator cannot catch: entity typing varies across chunks for a few
  entities ("MARS agent" Tool vs Product), and a few node-dedup calls split between near
  candidates ("Security Officer account" vs "role").

## Harness fixes the run found

- `e3eea46` pin `PYTHONHASHSEED`: graphiti renders entity labels from a set, so stage-2
  prompt bytes changed every round and answers could not replay.
- `07240d2` a round unqueues prompts no replayed chunk asked for again.
- `49c837d` cache layout forced ON as production runs it (the local `.env` had it off, so
  every capture was "unmoved"), plus `round --replay-final`.

## Teacher brief, refined during the run

Judgement calls settled from pilot review and observed cross-teacher drift (`3e39e1b`,
`db774e5`, `dcbdd2b`, `ad298e7`): click-paths stay out of fact text; edge dedup —
restatement is a duplicate, fact + detail is not, a subset is a duplicate unless the existing
fact narrows scope; summaries drop sentences about other entities; breadcrumb-only entities
need a body mention; always include the documented product in extract_text; a product the
chunk names is the fact subject, the documented product only for unnamed facts.

## Veeam Agent chunks

The structural chain files the Veeam Agent for Microsoft Windows / Linux document sets under
the product "Veeam Backup & Replication", so DOCUMENT CONTEXT names VBR for Agent chunks.
That is correct: the Agents are VBR components (standalone use is rare), so attributing those
chunks to VBR and the grounding gate adding VBR as an entity are intended. The brief's
subject rule still makes a product the chunk names (the Agent) the subject of its facts.

## Dataset

`build_extraction_dataset.py --from-capture … --articles-from sample.json --loss-basis
completion --upsample --max-repeat 12 --max-prompt-tokens 20480 --chars-per-token 3.0
--no-repair`: 1,695 usable of 1,695 (0 rejected, 0 over-length). Split from sample.json:
train 1,440 unique → 9,517 after upsampling, val 255 (20 held-out articles, never upsampled).
The upsample targets completion-token share (the notebook trains completion-only); after it
the completion mix is ~35/26/13/17/10% (edges/resolve_edge/summaries/extract_text/
dedupe_nodes) against 34/25/18/13/10%. The builder's "tokens%" column reports full-token
share, which is why it looks far off. Train is ~16.3M tokens per epoch; `resolve_edge`
records are repeated 11×.

Pushed to the private HF dataset repo `carev01/ft-opus-distill` (train/val/sample/MANIFEST; anonymous access 401). Package: `dist/ft-opus.zip` (sha256 `53e77ae05f1bf41760251036eb897d0793ef4460d54fe4440968e8e145a08e91`),
MANIFEST at commit `ad298e7`. Next: training on Colab (`notebooks/distill_train.ipynb`),
then the held-out and end-to-end evaluations of spec §8.
