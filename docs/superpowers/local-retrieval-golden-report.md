# Local Retrieval — Golden-Set Citation-Precision Report

**Date:** 2026-07-16
**What:** `GET /search/local` end-to-end over the gpt-5-mini semantic graph (555 entities / ~2004 facts, the 39-article pilot), scored against 15 golden questions grounded in known pilot articles. Retrieval = Graphiti `EDGE_HYBRID_SEARCH_RRF` (semantic + BM25 + RRF, **no cross-encoder**); citations = deterministic graph traversal (no LLM at query time).

---

## Headline

- **Citation precision@10 = 11/15 = 0.733** (strict: the *labeled canonical* article must appear in a returned fact's citations).
- **MRR = 0.477**; **7 of 11 hits rank in the top 2** (six at rank 1).
- **Topic-relevant@10 ≈ 14/15 ≈ 0.93** (looser: a returned fact correctly answers the question, even if cited to an *adjacent* pilot article — see §Misses).
- **The retrieval stack works end-to-end** on the Azure/TEI backend: RRF hybrid search runs cleanly, returns 10 ranked facts/query, and every fact resolves to a source URL by traversal. The reranker caveat is avoided (no cross-encoder dependency).

## Per-question

| # | Question | Result | First-hit rank |
|---|---|---|---|
| 1 | AWS Backup Vault Lock — what it enforces | HIT | 1 |
| 2 | AWS cross-Region copy | miss (adjacent) | — |
| 3 | AWS Backup encryption | HIT | 2 |
| 4 | Amazon S3 restore | HIT | 1 |
| 5 | AWS continuous backups / PITR | HIT | 6 |
| 6 | Amazon Redshift backups | HIT | 1 |
| 7 | AWS cross-account backup | HIT | 7 |
| 8 | Amazon EC2 restore | HIT | 4 |
| 9 | Azure soft delete + retention | **miss (genuine)** | — |
| 10 | Azure Backup encryption | HIT | 10 |
| 11 | Azure Cross Region Restore | miss (adjacent) | — |
| 12 | Restore SQL Server from Azure vault | HIT | 1 |
| 13 | Back up encrypted Azure VM | HIT | 1 |
| 14 | Restore VMware VMs with Azure Backup Server | HIT | 1 |
| 15 | AWS change retention period | miss (adjacent) | — |

## Misses — categorized (inspected the actual returned citations)

**3 "adjacent-article" misses** — retrieval found the correct *topic*; the answering fact was cited to a **different pilot article** than the single canonical label:
- **AWS cross-Region copy (Q2):** returned facts like *"A copy of a backup to another AWS Region is encrypted using…"* cited to **Encryption for backups in AWS Backup** — correct cross-region-copy content, extracted into the encryption article's facts, not the labeled *Cross-Region backup* article.
- **Azure CRR (Q11):** *"You configure Cross Region Restore for a Recovery Services vault…"* cited to **Migrate to availability zone support** — the CRR answer, in an adjacent article.
- **AWS retention (Q15):** *"A retention period recommendation notes a warm storage retention…"* cited to **Controls and remediation** — retention content, adjacent to the labeled *Changing your retention period*.
These reflect golden-label strictness (one canonical article) vs. how the corpus distributes facts across related articles — not a retrieval failure of the topic.

**1 genuine miss** — **Azure soft delete (Q9):** the top-10 returned unrelated AWS blog/vault facts; **no Azure soft-delete content surfaced**. A real gap: soft-delete facts either embed poorly for this query phrasing or are underrepresented. Candidate follow-ups: check the soft-delete articles' extracted facts; consider query expansion or a cross-encoder rerank *only if* the golden set shows a systematic pattern (it doesn't yet — this is 1/15).

## Assessment vs the Phase-2 exit

The Phase-2 exit criterion is "local questions answered with correct URLs at target citation precision." This slice **meets the mechanism** (deterministic cited retrieval, validated end-to-end) and gives a **first measured number: 0.733 strict / ~0.93 topic-relevant**. That is a healthy baseline for an untuned RRF hybrid over a 39-article pilot with strict single-article labels — most hits rank 1–2, and 3 of 4 misses are label-strictness, not topic failure.

**No cross-encoder needed yet** — RRF ranking is sensible on this backend, sidestepping the tiktoken-reranker risk. Revisit a local BGE / embedding reranker only if a larger golden set shows systematic ranking failures.

## Follow-ups (not blocking)

- Investigate the Azure soft-delete gap (the one genuine miss) — inspect the soft-delete articles' facts / embeddings.
- Grow the golden set (more vendors/topics) once the corpus expands past the pilot; consider allowing multiple canonical articles per question to reduce label-strictness noise.
- Synthesis layer (LLM writes prose citing fact-IDs → resolver expands to URLs) — the deferred next retrieval slice; it consumes exactly the `results` this endpoint returns.
- `search_local` uses Graphiti's `_search` (RRF recipe), which is marked deprecated in graphiti-core 0.29.2 (delegates to `search_`); switch the single call site to `search_` on the next graphiti bump.
