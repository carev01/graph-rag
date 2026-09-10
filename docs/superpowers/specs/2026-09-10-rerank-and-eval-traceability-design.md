# Reranked Community Selection + Eval Traceability — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Date:** 2026-09-10
**Status:** Approved design — ready for implementation planning
**Backlog:** implements 3a-bis, 3b, 3d; partially addresses the eval-harness half of 5b's observability complaint

Two independent parts. **Part 1 is small and ships first**, because it makes Part 2's own
validation runs observable instead of another two-hour black box.

---

# Part 1 — Eval traceability

## 1.1 The defect

`answer_api/eval_router.py` prints nothing until it finishes. On 2026-09-10 a run took
**2h09m**; a hung run and a working run were indistinguishable, and confirming it was alive
required inspecting CPU time and open sockets.

Worse, `run_eval` catches per-question failures, logs a warning, and rewrites the record as
`chosen: "error", routing_hit: False`. An earlier run swallowed **14 `APIConnectionError`s**
this way and would have produced a plausible-looking report whose routing accuracy was
depressed purely by network failures. The summary does not report how many questions failed.

This is the same swallow-and-continue pattern that has cost this project repeatedly.

## 1.2 The fix

1. **Per-question progress.** As each question completes, print one line: ordinal and total,
   intent, chosen mode, routing hit, grounding, faithfulness, elapsed seconds.
2. **Failures are counted and visible.** Add `questions_failed` to the summary. A failed
   question renders as `-` in the per-question table, never as a legitimate `0` or `False`.
3. **A failed question must not silently depress routing accuracy.** Exclude failures from
   the routing-accuracy denominator and say so in the report, rather than scoring a network
   error as a routing miss.

## 1.3 Testing

Hermetic unit tests: a run with one failing question reports `questions_failed == 1`;
routing accuracy is computed over the questions that actually ran; the failed row renders
as `-`. Existing eval tests must keep passing.

---

# Part 2 — Reranked community selection

## 2.1 The defect

Observed 2026-09-10 on *"What should I consider for backup encryption across cloud
providers?"*:

| Community | LLM relevance | rerank-3 |
|---|---|---|
| KMS Key Policy Management for AWS Backup | **3** | rank 8, 0.4453 |
| Resiliency in Azure: Unified BCDR Platform | **6** | rank 10, 0.4316 |

The off-topic BCDR overview outranked the encryption community, and returned **25
`fact_ids`** none of which concern encryption. `_MAP_PROMPT` asks for
`"relevance": 0-10 (how useful for the question)` with **no anchors and no definition of
relevance**, so a large, information-dense report reads as "useful". An LLM asked to rate
0-10 is improvising a scale; a cross-encoder is trained and calibrated for exactly this.

**A second, larger problem surfaced while sizing this work.** Global search runs at
`global_default_level = 1`, which holds **11** retrievable communities, and
`global_shortlist_k = 10`. The cosine shortlist admits 10 of 11 — **it is not selecting at
all.** With `relevance_min = 2` as the only other gate, essentially everything reaches the
reduce step. The reranker does not merely re-order the shortlist; it supplies selection
that currently does not happen.

**Note the asymmetry:** local search already uses a reranker — `graphiti_client.py:212`
passes `OpenAIRerankerClient` as graphiti's cross-encoder — and local scores 4.79
faithfulness. `global_search.py` has none, and global scores 2.33.

## 2.2 Live evidence the fix works

`rerank-3` over all 37 retrievable communities for that question, verified before designing:

```
0.6992  Azure Backup: Encryption, Soft Delete, and Cross-Region Resiliency
0.6250  AWS Backup: Cross-Account and Cross-Region Copy, Vaults, and Encryption
0.5781  AWS Backup: Capabilities, Workload Coverage, Operational Considerations
0.4453  KMS Key Policy Management for AWS Backup        <- above the BCDR overview
0.4316  Resiliency in Azure: Unified BCDR Platform      <- correctly demoted
```

The inversion is corrected, the scores are calibrated across a smooth 0.70-0.38 spread, and
one rerank call replaced what would be ten LLM calls, returning in seconds.

## 2.3 Architecture

```
shortlist_communities
  -> cosine over community embeddings, bounded to `rerank_candidates` (default 50)
  -> rerank-3 scores those candidates
  -> keep top `rerank_top_n` with score >= `rerank_score_floor`
  -> map_report EXTRACTS ONLY (key_points + fact_ids); no relevance scoring
  -> reduce (unchanged)
```

The cosine stage is retained deliberately. At today's 11 communities it is a no-op, but it
bounds what is posted to an external API if a level ever holds thousands.

### 2.3.1 The rerank client

New `src/answer_api/rerank.py`:

```python
async def rerank(query: str, documents: list[str], *, top_k: int) -> list[tuple[int, float]] | None
```

Voyage AI, Cohere-compatible: `POST {rerank_base_url}/rerank` with
`{"model", "query", "documents", "top_k"}`, `Authorization: Bearer`. Returns
`(index, relevance_score)` pairs, highest first.

**Returns `None` when relevance could not be scored** — non-200, transport error, or a
malformed payload. `None` means "could not score", never "nothing is relevant". This is the
same discipline as `verify_report`, and the reason is the same: four separate defects in
this codebase came from a failure being coerced into a legitimate-looking value.

Documents are `f"{title}: {summary}"`. Cross-encoders cap near 512 tokens and a community
summary is one paragraph, so it fits — and this is exactly the form tested in §2.2.

### 2.3.2 The cut

```python
keep = [(i, s) for i, s in reranked[:rerank_top_n] if s >= rerank_score_floor]
```

Both bounds matter. The floor alone could admit all 11 on a broad question, losing the cost
saving. `top_n` alone always fills its slots, which is today's failure — junk gets in
because something must fill the shortlist. With a floor, an off-topic question yields zero
survivors and takes the existing `_REFUSAL` path rather than synthesising junk.

**Defaults are measured, not guessed.** Implementation includes running the golden set's ten
global/DRIFT questions, recording the score distributions, and setting `rerank_top_n` and
`rerank_score_floor` from that data, with the §2.2 examples as the acceptance check. Picking
a floor from a single question would repeat the mistake that produced the summary rule this
project just had to undo.

### 2.3.3 `map_report` loses relevance scoring

`_MAP_PROMPT` drops the `relevance` field and asks only for `key_points` and `fact_ids`.
The rubric problem disappears rather than needing a rubric written for it.

`MapResult.relevance` is retained but now carries the **rerank score as a float**, so
`communities_used[].relevance` in the envelope becomes a calibrated number instead of an
improvised integer. `global_map_relevance_min` retires with the scoring it gated.

### 2.3.4 Degradation is visible to the machine AND the reader

When `rerank` returns `None`: log a warning, fall back to cosine ordering with `top_n`, and
set `degraded: "rerank-unavailable"` on the envelope — the pattern `drift.py` already uses
for `no-primer-communities`.

**The answer text must also carry a disclaimer**, because the envelope field is
machine-readable only and a reader would never see it:

> *Note: relevance ranking was unavailable for this answer, so the sources it draws on may
> be less relevant than usual. Please verify against the cited sources.*

Constraints on the disclaimer:

- It is added **after** `_finalize_answer`, or marker-stripping and whitespace repair would
  treat it as answer prose.
- It contains **no `[N]` markers and no URL**, so it cannot be mistaken for cited content
  and cannot violate design decision #2.
- It is added only when a real answer is produced — never appended to a refusal.
- It is implemented as a `degraded reason -> disclaimer text` mapping so other degraded
  modes can adopt it, but **only `rerank-unavailable` is wired up in this slice.**
  `drift.py`'s `no-primer-communities` degradation is a different situation — the reader
  gets a valid local answer, not a less accurate one — and disclaiming it is a separate
  decision, deliberately not taken here.

## 2.4 Components

| File | Change |
|---|---|
| `src/answer_api/rerank.py` | **New.** Voyage/Cohere-compatible client; returns `None` when it cannot score. |
| `src/answer_api/global_search.py` | Rerank between shortlist and map; `top_n` + floor cut; `_MAP_PROMPT` drops relevance; degraded fallback + disclaimer. |
| `src/graph_extract/config.py` | `rerank_base_url`, `rerank_model`, `rerank_api_key`, `rerank_candidates`, `rerank_top_n`, `rerank_score_floor`. Retire `global_map_relevance_min`. |
| `src/answer_api/eval_router.py` | Part 1: per-question progress, `questions_failed`, failures excluded from routing accuracy. |
| `src/answer_api/drift.py` | None — it consumes `shortlist_communities` and inherits reranking. |

## 2.5 Error handling and edge cases

| Case | Behaviour |
|---|---|
| Rerank returns a normal result | Top-N above the floor proceed to extraction. |
| Every candidate below the floor | Zero survivors; existing `_REFUSAL` path. Correct for an off-topic question. |
| Rerank non-200 / transport error / malformed payload | `None` → cosine fallback, warning logged, `degraded` set, disclaimer added. |
| Rerank returns fewer results than requested | Use what came back; not an error. |
| Rerank returns an out-of-range index | Ignore that entry; never index a candidate list with it. |
| No communities at the level | Unchanged: existing refusal path, no rerank call. |
| A single candidate | Still reranked; the floor can legitimately reject it. |
| Refusal produced while degraded | No disclaimer — there is no answer to qualify. |

## 2.6 Testing

- **Unit (hermetic, fake HTTP):** the client parses a well-formed response; returns `None`
  on non-200, on a transport exception, and on malformed JSON; out-of-range indices are
  ignored; `top_n` + floor selects correctly, including the zero-survivor case; the
  disclaimer is added on degradation, is absent on the happy path, is absent on a refusal,
  and contains no `[N]` marker.
- **Regression:** existing `global_search` and DRIFT tests pass. Tests referencing
  `relevance_min` or `MapResult.relevance` as a 0-10 integer are updated honestly to the new
  contract, not weakened.
- **Live measurement:** the ten golden global/DRIFT questions, recording score
  distributions, to set `rerank_top_n` and `rerank_score_floor` from data.
- **Live acceptance:** the two §2.2 examples must come out in the corrected order.
- **Eval re-run** with Part 1's traceability active, reported against global faithfulness
  **2.33** and grounding **0.86**, whatever it shows.
- Full non-live suite, `uv run ruff check src tests`, `uv run mypy src` clean.

## 2.7 Success criteria

1. A rerank failure never reads as "nothing is relevant".
2. A degraded answer carries a reader-visible disclaimer with no `[N]` markers.
3. `map_report` no longer scores relevance; `_MAP_PROMPT` has no relevance field.
4. `rerank_top_n` and `rerank_score_floor` are set from measured distributions, and the
   measurement is recorded.
5. The two §2.2 examples come out in the corrected order.
6. The eval is re-run with per-question progress and `questions_failed` visible, and global
   faithfulness reported against 2.33 whatever it shows.

## 2.8 What this slice does not claim

**Reranking changes what is selected, not whether content is invented.** That was the
report-verification slice. A reranker would have surfaced the KMS community correctly, but
would not have stopped "1-second PITR precision" being fabricated. Do not let a faithfulness
movement here be attributed to reranking without checking which communities were selected.

**Narrower selection may produce shorter answers, or refusals on thin questions.** That is
the correct outcome, the same principle as report verification, and will be reported as such
rather than tuned away.

**Accepted, with a backlog item:** nothing aggregates `routing.degraded`, so a prolonged
Voyage outage would serve weaker answers and only the per-answer disclaimer would show it.
Aggregating degradation signals is out of scope here and belongs on the backlog.

**The API key** is in the untracked `.env` only. It was shared in plaintext in a chat
transcript and should be rotated once this work is settled.
