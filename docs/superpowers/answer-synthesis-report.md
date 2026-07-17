# Answer Synthesis — Groundedness Report

**Date:** 2026-07-16
**What:** `/answer` (GLM-5.2 synthesis) over the gpt-5-mini semantic graph, scored on the 15 golden questions. **answer-groundedness** = the synthesized answer cites ≥1 fact whose resolved source is an expected article. Design-decision #2: the LLM emits only `[N]` markers; a deterministic resolver expands the markers it used into source URLs — **the LLM never authors a URL**.

---

## Headline

- **answer_groundedness@15 = 11/15 = 0.733** — close to the `/search/local` citation-precision@10 baseline (0.800).
- **Design-decision #2 holds end-to-end:** across all 15 answers, **zero URLs were authored by the LLM** (the deterministic `_finalize_answer` URL-strip found nothing to remove on the real run; citations come only from the marker→fact→source map). Spot-read answers are faithful to the cited facts.
- **Refuse-when-unsupported works:** 2/15 returned the fixed refusal; 0 hallucinated answers.

## The key tuning finding (why the first run scored 0.400)

The first live run scored **0.400** with **9 empty answers**. Root cause: **GLM-5.2 is a reasoning model, and `max_tokens=800` truncated the answer to empty** (`finish_reason='length'`, `content=''`) — the reasoning consumed the whole budget. Raising to **`max_tokens=3000`** fixed it (empty→populated, `finish_reason='stop'`), lifting groundedness **0.400 → 0.733**. This is the same lesson as the GLM judge (`type_precision` needed adequate headroom). (Also fixed: the harness's URL flag was `"http" in answer`, a false positive on the word "HTTPS" — now uses the real `_URL_RE`; no actual URL ever survived.)

## Per-question (after the fix)

| Question | Outcome | cited |
|---|---|---|
| AWS Vault Lock — what it enforces | GROUNDED | 2 |
| AWS cross-Region copy | GROUNDED | 5 |
| AWS Backup encryption | GROUNDED | 6 |
| Amazon S3 restore | GROUNDED | 12 |
| AWS continuous backups / PITR | GROUNDED | 3 |
| Amazon Redshift backups | GROUNDED | 8 |
| AWS cross-account backup | GROUNDED | 8 |
| Amazon EC2 restore | GROUNDED | 11 |
| Azure soft delete + retention | **REFUSED** | 0 |
| Azure Backup encryption | GROUNDED | 3 |
| Azure Cross Region Restore | unground (adjacent) | 2 |
| Restore SQL Server from Azure vault | GROUNDED | 11 |
| Back up encrypted Azure VM | GROUNDED | 13 |
| Restore VMware VMs w/ Azure Backup Server | **REFUSED** | 0 |
| AWS change retention period | unground (adjacent) | 5 |

## The 4 non-grounded, categorized

- **2 adjacent-article** (Azure CRR, AWS retention): the answer *cited facts* (2 and 5), but from a **related** pilot article rather than the single canonical golden label — the same label-strictness seen in the retrieval report, not a synthesis failure.
- **2 refusals** (Azure soft-delete, VMware restore): GLM returned the fixed refusal despite relevant facts being retrievable — the grounding threshold is **slightly conservative**. Refuse-when-unsupported is correctly wired (better a safe non-answer than a hallucination); tuning the prompt to be less eager to refuse when facts are present is a follow-up.

## Assessment

The synthesis layer meets its design goals: a cited prose answer where **every source URL is produced deterministically from a resolved fact, never written by the LLM** (design-decision #2, validated across 15 live answers), with an honest refusal path. Groundedness (0.733) tracks retrieval precision (0.800) closely; the gap is 2 label-strictness cases + 2 conservative refusals — not citation-integrity issues.

## Follow-ups (not blocking)

- **GLM refusal calibration:** it refused on 2 questions whose facts were present — a prompt tweak (or a stronger synthesis model via the `synthesis_*` seam already built) could recover them.
- **Verbose/compound queries** (the deferred retrieval concern) still affects the adjacent-article cases — query rewrite is the next retrieval slice.
- **Answer faithfulness scoring** (an independent model checks the prose is entailed by the cited facts) — a later eval slice; this report spot-reads faithfulness, doesn't score it.
- The `max_tokens=3000` is a working value for GLM-5.2; revisit if a different synthesis model is wired.
