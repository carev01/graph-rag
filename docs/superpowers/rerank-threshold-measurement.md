# Rerank threshold measurement

**Date:** 2026-09-10
**Purpose:** set `rerank_top_n` and `rerank_score_floor` from measured score
distributions rather than from the single example that motivated the work.
**Setup:** Voyage `rerank-3`, all **11** retrievable communities at
`global_default_level = 1`, scored against each of the **10** golden global/DRIFT
questions. Documents are `f"{title}: {summary}"`.

## Chosen values

| setting | value | previously (provisional) |
|---|---|---|
| `rerank_top_n` | **4** | 4 |
| `rerank_score_floor` | **0.40** | 0.45 |

## Top-6 scores per question

| intent | top 6 scores | question |
|---|---|---|
| global | 0.613 0.463 0.449 0.449 0.414 0.402 | Compare how AWS Backup and Azure Backup encrypt backup data |
| global | 0.652 0.539 0.451 0.436 0.352 0.346 | Compare cross-region restore between AWS and Azure Backup |
| global | 0.590 0.471 0.465 0.449 0.439 0.398 | How do AWS/Azure Backup each protect recovery points from deletion |
| global | 0.594 0.586 0.543 0.449 0.410 0.371 | Which backup capabilities do AWS and Azure share for compliance retention |
| global | 0.547 0.441 0.432 0.379 0.373 0.371 | Compare AWS Backup and Azure Backup database restore workflows |
| drift | 0.523 0.500 0.453 0.451 0.377 0.342 | Long-term backup retention across cloud vendors |
| drift | 0.625 0.467 0.445 0.424 0.422 0.383 | Backup encryption across cloud providers |
| drift | 0.629 0.527 0.490 0.486 0.373 0.359 | Cross-region disaster recovery of cloud backups |
| drift | 0.645 0.512 0.512 0.363 0.354 0.330 | Immutability and ransomware protection |
| drift | 0.504 0.479 0.479 0.426 0.402 0.391 | Restoring databases from cloud backups |

## Survivors per candidate floor

| floor | survivors per question | questions with ZERO survivors |
|---|---|---|
| 0.35 | 7 5 7 8 7 5 7 8 5 8 | 0/10 |
| **0.40** | **6 4 5 5 3 4 5 4 3 5** | **0/10** |
| 0.45 | 2 3 3 3 1 4 2 4 3 3 | 0/10 |
| 0.50 | 1 2 1 3 1 2 1 2 3 1 | 0/10 |
| 0.55 | 1 1 1 2 0 0 1 1 1 0 | **3/10** |
| 0.60 | 1 1 0 0 0 0 1 1 1 0 | **5/10** |

## Why 0.40 and top_n 4

**Scores are compressed, so the floor cannot do fine-grained work.** Top scores sit
between 0.504 and 0.652, and below rank 1 the values collapse into a dense band around
0.40-0.49. In the acceptance case the two communities the whole exercise is about —
"KMS Key Policy Management" and "Resiliency in Azure BCDR" — are **0.023 apart** (0.4453
vs 0.4219). No threshold separates those cleanly. The reranker fixes the **ordering**; the
selection must therefore come from `top_n`, with the floor doing only one job: removing the
clear tail.

**The tail is clear.** On the acceptance question the bottom four score 0.330, 0.328,
0.285, 0.279 — visibly detached from the pack. 0.40 removes those and the marginal
0.383/0.369 pair without touching the contested band.

**0.55 and above are disqualified by evidence, not taste.** They produce zero survivors on
3/10 and 5/10 golden questions. These are questions the corpus is expected to answer, so a
refusal there is a false negative, not honest restraint.

**0.45 and 0.50 select too narrowly for what global mode is for.** At 0.50 the median is 1
surviving community. Global mode exists to synthesise *across* communities; feeding the
reduce step a single community makes it local search with extra steps. 0.45 would also cut
KMS (0.4453) on the acceptance question — the very community the inversion complaint was
about.

**Net effect at 0.40 + top_n 4:** every question keeps 3-4 communities (median 4, minimum
3, no refusals), and the LLM extraction step runs on ~4 communities instead of 10 — a ~60%
reduction in map-step calls.

## Acceptance check — the observed inversion is corrected

Question: *"What should I consider for backup encryption across cloud providers?"*

| rank | score | community |
|---|---|---|
| 1 | 0.6250 | AWS Backup: Cross-Account and Cross-Region Copy, Vaults, and Encryption |
| 2 | 0.4668 | AWS Backup Vault Lock: Compliance Modes, Retention Controls |
| **3** | **0.4453** | **KMS Key Policy Management for AWS Backup** — Solar Pro scored this **3** |
| 4 | 0.4238 | AWS Key Management Service KMS Actions in AWS Backup |
| **5** | **0.4219** | **Resiliency in Azure: Unified BCDR Posture Management** — Solar Pro scored this **6** |
| 6-11 | 0.383 → 0.279 | tail |

KMS now ranks **above** the BCDR overview, reversing Solar Pro's inversion. With
`top_n = 4` the BCDR overview is excluded while both KMS communities are kept — the
discrimination the inversion complaint asked for, achieved by ordering plus the cap rather
than by the floor.

## Honest caveats

- **Production never sees the survivor counts this table shows.** `rerank()` is called
  with `top_k=rerank_top_n` (4), so the API itself only ever returns 4 scored rows per
  question — the "survivors per floor" table above (5-8 survivors at 0.35, 3-6 at 0.40)
  describes a distribution measured with `top_k` wide open for this exercise, which
  production never encounters. That weakens the floor-selection argument as written: the
  real question the floor answers in production is "does it ever cut into the already-tiny
  top-4", not "how many of 11 candidates clear it." The chosen values are unaffected —
  0.40 still measurably fixes the acceptance-case ordering and the compression argument
  above still holds — but the survivor-count table should not be read as evidence about
  what production sees.
- **The floor sits just below a dense band.** At 0.40 the nearest excluded scores are
  0.383 and 0.369, and several questions have multiple communities within 0.02 of the
  cut. Small score shifts — a reranker version change, an edited community summary — could
  move survivor counts noticeably. This threshold is not robust, it is merely *measured*.
- **This measures selection, not answer quality.** Nothing here shows that 4 communities
  produce better answers than 10, only which communities survive. The eval is what tests
  that, and it may yet argue for a different value.
- **Corpus size is small.** 11 communities at level 1. Both the compression and the tail's
  clarity may look different at corpus scale, so these values deserve re-measuring after
  broad ingestion.
- Reproduce with the script in the implementation plan's Task 5, Step 1.
