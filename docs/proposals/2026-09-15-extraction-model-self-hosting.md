# Replacing Solar Pro 4 on the extraction tier — first-pass assessment

**Date:** 2026-09-15
**Status:** Not started. Assessment only — no experiment run, no decision taken.
**Trigger:** Solar Pro 4 has a substantial price increase on 2026-10-10. The goal
stated with it matters as much as the trigger: *settle on one extraction model and
stop changing it.*

---

## Where Solar Pro 4 actually sits

| setting | model | role |
|---|---|---|
| `LLM_MODEL` | `gpt-5-mini` | strong tier — dense availability/support matrices only |
| `CHEAP_LLM_MODEL` | **`upstage/solar-pro4`** | **cheap tier — everything else** |
| `MAP_LLM_MODEL` | `upstage/solar-pro4` | global-search map step |

`article_router.is_dense_matrix` routes dense tables to the strong tier and
everything else to the cheap one, so Solar Pro 4 carries the **majority** of
extraction volume. This is the tier the price increase acts on.

## The reframe: "settle and never change" argues for self-hosting, not fine-tuning

A fine-tune would make the system **more** coupled, not less. It would be trained on
graphiti 0.30.1's exact prompt templates (`extract_nodes`, `extract_edges`,
`dedupe_edges.resolve_edge`). Upgrading graphiti changes those prompts and staleness
the fine-tune against them. This project deliberately pins graphiti and never forks
it, extending by patching imported-by-name references; a fine-tune adds a second
artefact that must move in lockstep with a library we do not control.

Self-hosted **base** weights give the stability actually being asked for. Weights on
disk do not get repriced and do not get deprecated — and this project has already
been burned by exactly that: `inclusionai/ling-2.6-flash` died mid-project, which is
why the cheap tier is Solar Pro 4 today.

## The binding constraint is schema conformance, not model intelligence

Every recorded cheap-tier failure in this project was a **format** failure:

- *"graphiti unbounded index arrays — the real cause of every cheap-model extraction
  failure; `maxItems` fixes it, repetition penalties make it worse"*
- *"ling is dead; require `structured_outputs` in `supported_parameters` and probe
  with a strict `json_schema` before trusting any cheap model"*

Neither is a comprehension problem. That changes the calculus: self-hosting with
**grammar-constrained decoding** (vLLM + xgrammar/outlines) makes malformed output
structurally impossible rather than merely unlikely — strictly stronger than any
API's best-effort structured-output mode.

**So the first experiment is not a fine-tune.** It is a base instruct model behind a
hard grammar constraint, measured on the harness that already exists.

## What the cheap tier is actually asked to do

Measured LLM-time share, from the 2026-09-14 N=4 A/B:

| prompt | share | nature |
|---|---|---|
| `extract_edges.edge` | 33.6% | relation extraction |
| `dedupe_edges.resolve_edge` | 22.9% | pick an index from candidates |
| `extract_nodes.extract_summaries_batch` | 19.8% | short summaries |
| `extract_nodes.extract_text` | 12.7% | entity extraction |
| `dedupe_nodes.nodes` | 10.9% | pick an index |

**A third of the work is index-picking** — close to classification, and well within a
constrained 9B. Entity and relation extraction from vendor documentation is the
harder half, but it is extraction from provided text, not open-ended reasoning.

No claim is made here about Qwen-3.5-9B's specific quality on this task; that is
what the evaluation is for.

## The gate: hardware

`nvidia-smi` finds no GPU on the host this was assessed from. A 9B wants ~24 GB at
bf16 with batching headroom; ~16 GB works quantized. **If no GPU is available in the
k3s cluster and none is planned, this entire line collapses** and the question
becomes "which API model, with `structured_outputs` verified" instead.

Resolve this first. Everything below depends on it.

## Proposed sequence

1. **Confirm the GPU story.** Gate on it.
2. **Serve the candidate on vLLM with guided decoding**, pointed at graphiti's real
   JSON schemas. Probe per the standing rule: strict `json_schema`, and specifically
   check `maxItems` behaviour on the index arrays that killed ling.
3. **Run the A/B that already exists.** Same 83 pilot articles, cheap tier swapped,
   against the 999-entity sequential baseline: entity yield, fact yield, out-of-range
   dedup indices, plus `eval_quality` and the router golden set. Method and runner:
   `.superpowers/sdd/ab-w8-validation.py` and `ab-revert.py` (supersede-and-restore,
   never a reset of the baseline). Near-zero marginal cost once the model is local.
4. **Fine-tune only if step 3 falls short**, and only on the prompts that fail.
   Distillation data is a capture job rather than a labelling job — `usage.py`
   already instruments every prompt/response pair, so harvesting a few thousand
   examples per prompt type from the strong tier is mechanical. LoRA on a 9B is
   hours on one GPU.

## The number that frames the decision

The A/B measured **~$0.12/article** across both tiers. At 126,405 articles that is
**~$15k of extraction at current pricing** — and that total, not the current monthly
spend, is what the October increase acts on. Self-hosting converts it to a fixed
hardware cost. That is where the money is, independent of whether anything is ever
fine-tuned.

## Open questions for the user

1. Is there a GPU in the cluster, or a budget for one?
2. Is the map tier (`MAP_LLM_MODEL`, also Solar Pro 4) in scope, or extraction only?
3. Does the strong tier (`gpt-5-mini`, dense matrices) stay as-is? It is a small
   share of volume and a different quality bar.
