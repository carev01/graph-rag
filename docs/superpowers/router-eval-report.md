# /answer Router Golden-Set Eval

> **Re-run after reranked community selection (2026-09-10).** Graph identical to the
> previous run — 385 episodes, 2,173 facts, 41 communities (37 retrievable) — so this is
> like-for-like.
>
> | | prev (2026-09-10) | this run |
> |---|---|---|
> | Routing accuracy | 0.97 | 0.97 |
> | Grounding precision | 0.88 | 0.88 |
> | — global grounding | 0.86 | 0.86 |
> | Faithfulness mean | 4.04 | 4.07 |
> | — **global faithfulness** | **2.33** | **2.30** |
> | questions failed | (not reported) | **0/29** |
> | unscored | 1/29 | 0/29 |
>
> ## Reranking did NOT improve faithfulness
>
> Global faithfulness went 2.33 → 2.30. That is flat on the headline, but the headline is
> not the whole story: the previous 2.33 averaged **9** scored questions (1 was unscored),
> and this run's 2.30 averages **10** (the previously-unscored question now scores 5). On
> the **9 questions common to both runs, the mean fell 2.33 → 2.00** — a decline, held flat
> in the headline only by the newly-scored question pulling the 10-question average back
> up. Per-question global scores were 5, 1, 1, 5, 2 against a previous 5, 1, 4, –, 2:
> individual questions moved in both directions and cancelled out. Comparative global rose
> 2.4 → 2.625 and comparative DRIFT fell 4.3 → 4.2 — both within noise on ten questions.
>
> This does not make the conclusion below a regression claim — ten questions (nine, for the
> like-for-like comparison) is too small a base to call 2.33 → 2.00 anything more than
> "did not improve, and arguably drifted down within noise." It does mean "flat" undersells
> what the like-for-like arithmetic actually shows.
>
> **This was pre-committed as the honest outcome and is reported as such.** The slice's own
> spec (§2.8) states that reranking changes what is *selected*, not whether content is
> invented, and warns against attributing any faithfulness movement to it without checking
> which communities were chosen. There is no movement to attribute.
>
> ## What the slice did deliver
>
> - **The observed relevance inversion is corrected.** On *"What should I consider for
>   backup encryption across cloud providers?"*, "KMS Key Policy Management" now ranks 3rd
>   (0.4453) against the Azure BCDR overview at 5th (0.4219). The previous LLM scorer rated
>   them 3 and 6 respectively — backwards. See `rerank-threshold-measurement.md`.
> - **~60% fewer map-step LLM calls.** Extraction now runs on ~4 communities per question
>   instead of 10, because relevance is scored by one cheap cross-encoder call rather than
>   ten LLM calls.
> - **The eval is observable.** Per-question progress with timings, and `failed: 0` proving
>   no question was silently swallowed. The previous run hid 14 connection errors inside
>   `routing_hit: False` records.
>
> ## What this tells us about where global's weakness actually lives
>
> Global faithfulness across three consecutive slices:
>
> | change | global faithfulness |
> |---|---|
> | baseline | 1.4 (mismeasured; the judge scored unmeasurable answers 0) |
> | reduce-prompt binding | 1.6 — flat |
> | **community-report verification** | **2.33 — moved** |
> | reranked selection | 2.30 — flat |
>
> Only fixing *invented content* moved the number. Neither constraining the reduce step nor
> improving *selection* did. Global remains 2.3 against local's 5.0.
>
> **The next step should be to check the metric before changing the system again.** BACKLOG
> item 2: `_cited_fact_texts` returns facts in arbitrary Neo4j order and the judge prompt
> lists them unnumbered, so the judge can only ask "is this claim supported by the fact set
> *collectively*?" — a misattributed-but-plausible marker passes. Two interventions have now
> failed to move this number. Before a third, it is worth establishing that the number
> measures what we believe it measures.
>
> ## Harness note
>
> Individual global/DRIFT questions took 246s, 456s, 478s, 502s and 528s. That is BACKLOG
> 5b — the answer-path clients have no timeout and no throughput routing, unlike the report
> tier which was fixed for exactly this on 2026-09-08. Now visible per-question rather than
> as one unexplained multi-hour total.

Questions: 29 (failed: 0)

**Routing accuracy: 0.97**
by intent: {'local': 0.9333333333333333, 'global': 1.0, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.8846153846153846
by mode: {'local': 0.9285714285714286, 'timeline': 0.8, 'global': 0.8571428571428571}
Faithfulness mean: 4.07 (unscored: 0/29)
by mode: {'local': 5.0, 'timeline': 5.0, 'global': 2.3}

Comparative (broad): {'local': {'faithfulness': 4.9, 'grounding': 0.8571428571428571}, 'global': {'faithfulness': 2.625, 'grounding': 0.8571428571428571}, 'drift': {'faithfulness': 4.2, 'grounding': 1.0}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | question |
|---|---|---|---|---|---|
| local | local | True | True | 5 | What does AWS Backup Vault Lock enforce on recovery points? |
| local | local | True | False | 5 | How does AWS Backup copy backups to another AWS Region? |
| local | local | True | True | 5 | How are backups encrypted in AWS Backup? |
| local | local | True | True | 5 | How do you restore an Amazon S3 backup? |
| local | local | True | True | 5 | How does AWS Backup support continuous backups and point-in-time recovery? |
| local | local | True | True | 5 | How are Amazon Redshift clusters backed up? |
| local | local | True | True | 5 | How does AWS Backup perform cross-account backup? |
| local | local | True | True | 5 | How do you restore an Amazon EC2 instance from a backup? |
| local | local | True | True | 5 | What is soft delete in Azure Backup and how long does it retain deleted items? |
| local | local | True | True | 5 | How does Azure Backup encrypt backup data? |
| local | local | True | True | 5 | How do you configure and run Cross Region Restore in Azure Backup? |
| local | local | True | True | 5 | How do you restore a SQL Server database from an Azure Backup vault? |
| local | local | True | True | 5 | How do you back up an encrypted Azure VM? |
| local | local | True | True | 5 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | True | 5 | How do you change the backup retention period in AWS Backup? |
| global | global | True | True | 5 | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | global | True | True | 1 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | global | True | True | 1 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | global | True | None | 5 | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | global | True | False | 2 | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 1 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | True | 2 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 3 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | True | 1 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | global | True | None | 2 | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | True | 5 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | True | 5 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | False | 5 | How has Azure Backup encryption support changed over time? |
