# /answer Router Golden-Set Eval

> **Run 2026-09-11 11:28 → 13:12, after binding reduce claims to their facts (BACKLOG 0b,
> branch `bind-claims-to-facts`). Full account: `bind-claims-to-facts-2026-09-11.md`.**
>
> Same corpus as the previous three runs (level 1 12/12 retrievable, no `theme-build`).
> Code changes between the two runs: the reduce block is now `[N] <fact>` lines instead
> of key points plus a marker bag (one batched fact-text read; numbering unchanged), the
> reduce prompt explains the format, and this report's new `mps` column (markers per
> sentence, mean/max).
>
> | | 09-10 re-baseline | 0c run | 0d run (prev) | this run |
> |---|---|---|---|---|
> | Routing accuracy | 0.97 | 0.97 | 0.97 | 0.97 |
> | Grounding precision | 0.85 | 0.88 | 0.88 | **0.92** |
> | Faithfulness mean | 3.92 | 4.11 | 3.75 | **4.90** |
> | — global | 1.5 (8/10) | 2.44 (9/10) | 1.6 (10/10) | **4.7 (10/10)** |
> | **unscored** | **3/29** | **2/29** | **1/29** | **0/29** |
> | ranges in any answer | — | — | 0/29 | **0/29** |
> | comparative global / drift | — | 1.6 / 4.5 | 2.5 / 4.7 | **4.9 / 4.8** |
>
> **What moved is CITATION CORRECTNESS, not the truth of the content.** Claims correctly
> cited went 24% → 80%; claims supported by the facts given went 95% → 86% (slightly down).
> The judge is finally handed the facts an answer rests on. Read the jump below as exact
> provenance arriving, not as answers becoming more accurate.
>
> **Global faithfulness 1.6 → 4.7 over all ten global-mode questions**; per question
> 5,2,2,0,2 / 1,1,1,1,1 → 4,5,5,5,4 / 5,4,5,5,5. The per-claim audit behind it (same
> deletion question: 0–2 of 9 claims correctly cited before, 9–24 of 13–25 after) is in
> the slice report. The encryption control read 4 this run (5 last run, 17/17 cited):
> one "Both use KMS-integrated encryption" over-reach, a synthesis error, not a
> citation one. The `mps` column shows two large-pool answers still carrying a bag
> sentence (compliance retention 16 markers on one sentence; cross-region DR 19) despite
> scoring 5 — bag-pasting is now visible where before it was indistinguishable from
> binding. 8 synthesis empty-content retries across the run's ~40 reduce calls (all
> recovered, 0 gave up): the reduce prompt is 2–3× larger — BACKLOG 5b. Q10 (local)
> answered this run, so nothing is unscored.

Questions: 29 (failed: 0)

**Routing accuracy: 0.97**
by intent: {'local': 0.9333333333333333, 'global': 1.0, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.9230769230769231
by mode: {'local': 0.9285714285714286, 'timeline': 0.8, 'global': 1.0}
Faithfulness mean: 4.90 (unscored: 0/29)
by mode: {'local': 5.0, 'timeline': 5.0, 'global': 4.7}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.7142857142857142, 'max': 5}, 'timeline': {'mean': 1.0, 'max': 1}, 'global': {'mean': 3.8200000000000003, 'max': 19}}

Comparative (broad): {'local': {'faithfulness': 4.888888888888889, 'grounding': 0.7142857142857143}, 'global': {'faithfulness': 4.9, 'grounding': 1.0}, 'drift': {'faithfulness': 4.8, 'grounding': 1.0}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | question |
|---|---|---|---|---|---|---|---|---|
| local | local | True | True | 5 | 5 | 0 | 2.3/3 | What does AWS Backup Vault Lock enforce on recovery points? |
| local | local | True | False | 5 | 8 | 0 | 2.7/5 | How does AWS Backup copy backups to another AWS Region? |
| local | local | True | True | 5 | 2 | 0 | 1.0/1 | How are backups encrypted in AWS Backup? |
| local | local | True | True | 5 | 11 | 0 | 2.2/3 | How do you restore an Amazon S3 backup? |
| local | local | True | True | 5 | 10 | 0 | 1.8/3 | How does AWS Backup support continuous backups and point-in-time recovery? |
| local | local | True | True | 5 | 11 | 0 | 1.8/4 | How are Amazon Redshift clusters backed up? |
| local | local | True | True | 5 | 9 | 0 | 1.3/2 | How does AWS Backup perform cross-account backup? |
| local | local | True | True | 5 | 14 | 0 | 1.4/3 | How do you restore an Amazon EC2 instance from a backup? |
| local | local | True | True | 5 | 5 | 0 | 1.0/1 | What is soft delete in Azure Backup and how long does it retain deleted items? |
| local | local | True | True | 5 | 1 | 0 | 1.0/1 | How does Azure Backup encrypt backup data? |
| local | local | True | True | 5 | 5 | 0 | 1.7/2 | How do you configure and run Cross Region Restore in Azure Backup? |
| local | local | True | True | 5 | 11 | 0 | 3.0/4 | How do you restore a SQL Server database from an Azure Backup vault? |
| local | local | True | True | 5 | 11 | 0 | 1.6/2 | How do you back up an encrypted Azure VM? |
| local | local | True | True | 5 | 5 | 0 | 1.2/2 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | True | 5 | 30 | 0 | 1.0/1 | How do you change the backup retention period in AWS Backup? |
| global | global | True | True | 4 | 14 | 0 | 3.0/7 | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | global | True | True | 5 | 14 | 0 | 2.6/6 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | global | True | True | 5 | 28 | 0 | 3.1/7 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | global | True | None | 5 | 23 | 0 | 7.7/16 | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | global | True | True | 4 | 18 | 0 | 3.0/9 | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 5 | 89 | 0 | 3.8/9 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | True | 4 | 22 | 0 | 2.9/10 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 5 | 62 | 0 | 5.2/19 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | True | 5 | 8 | 0 | 4.0/4 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | global | True | None | 5 | 48 | 0 | 2.9/5 | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | False | 5 | 30 | 0 | 1.0/1 | How has Azure Backup encryption support changed over time? |
