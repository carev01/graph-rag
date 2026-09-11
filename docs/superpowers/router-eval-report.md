# /answer Router Golden-Set Eval

> **Run 2026-09-11 09:00 → 10:40, after the reduce-prompt range ban (BACKLOG 0d, branch
> `no-range-citations`). Full account: `no-range-citations-2026-09-11.md`.**
>
> Same corpus as the previous two runs (level 1 12/12 retrievable, no `theme-build`).
> Code changes between the two runs: one rule in `_REDUCE_PROMPT` (cite markers
> individually, never as a range), and this report's new `cited` / `ranges` columns.
>
> | | pre-repair | 09-10 re-baseline | 0c run (prev) | this run |
> |---|---|---|---|---|
> | Routing accuracy | 0.97 | 0.97 | 0.97 | 0.97 |
> | Grounding precision | 0.88 | 0.85 | 0.88 | 0.88 |
> | Faithfulness mean | 4.07 | 3.92 | 4.11 | 3.75 |
> | — global | 2.30 (10/10) | 1.5 (8/10) | 2.44 (9/10) | **1.6 (10/10)** |
> | **unscored** | **0/29** | **3/29** | **2/29** | **1/29** |
> | ranges in any answer | — | — | — | **0/29** |
>
> **The model complied: 0 ranges in 29 answers.** The encryption control recovered 2 → 5
> (17/17 cited). **Global faithfulness did not improve** — 1.6 over all ten global-mode
> questions (1.67 like-for-like over last run's nine); the fourth flat-or-down result.
> The new columns show the low scores resting on 5–28 citations with no range
> (compliance retention: 0 on 15 citations), i.e. facts that do not back the sentences
> they are attached to — BACKLOG 0b, not 0d. The DRIFT-intent database-restore question
> that refused last run answers (1); Q10 (local) unscored as before.

Questions: 29 (failed: 0)

**Routing accuracy: 0.97**
by intent: {'local': 0.9333333333333333, 'global': 1.0, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.8846153846153846
by mode: {'local': 0.8571428571428571, 'timeline': 0.8, 'global': 1.0}
Faithfulness mean: 3.75 (unscored: 1/29)
by mode: {'local': 4.923076923076923, 'timeline': 5.0, 'global': 1.6}

Comparative (broad): {'local': {'faithfulness': 5.0, 'grounding': 0.8571428571428571}, 'global': {'faithfulness': 2.5, 'grounding': 1.0}, 'drift': {'faithfulness': 4.7, 'grounding': 1.0}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | question |
|---|---|---|---|---|---|---|---|
| local | local | True | True | 5 | 6 | 0 | What does AWS Backup Vault Lock enforce on recovery points? |
| local | local | True | False | 5 | 1 | 0 | How does AWS Backup copy backups to another AWS Region? |
| local | local | True | True | 5 | 2 | 0 | How are backups encrypted in AWS Backup? |
| local | local | True | True | 5 | 11 | 0 | How do you restore an Amazon S3 backup? |
| local | local | True | True | 5 | 9 | 0 | How does AWS Backup support continuous backups and point-in-time recovery? |
| local | local | True | True | 5 | 12 | 0 | How are Amazon Redshift clusters backed up? |
| local | local | True | True | 5 | 13 | 0 | How does AWS Backup perform cross-account backup? |
| local | local | True | True | 5 | 13 | 0 | How do you restore an Amazon EC2 instance from a backup? |
| local | local | True | True | 5 | 4 | 0 | What is soft delete in Azure Backup and how long does it retain deleted items? |
| local | local | True | False | - | 0 | 0 | How does Azure Backup encrypt backup data? |
| local | local | True | True | 5 | 5 | 0 | How do you configure and run Cross Region Restore in Azure Backup? |
| local | local | True | True | 5 | 13 | 0 | How do you restore a SQL Server database from an Azure Backup vault? |
| local | local | True | True | 4 | 11 | 0 | How do you back up an encrypted Azure VM? |
| local | local | True | True | 5 | 5 | 0 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | True | 5 | 30 | 0 | How do you change the backup retention period in AWS Backup? |
| global | global | True | True | 5 | 17 | 0 | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | global | True | True | 2 | 5 | 0 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | global | True | True | 2 | 8 | 0 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | global | True | None | 0 | 15 | 0 | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | global | True | True | 2 | 12 | 0 | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 1 | 28 | 0 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | True | 1 | 12 | 0 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 1 | 15 | 0 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | True | 1 | 24 | 0 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | global | True | None | 1 | 16 | 0 | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | True | 5 | 30 | 0 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | 30 | 0 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | True | 5 | 30 | 0 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | False | 5 | 30 | 0 | How has Azure Backup encryption support changed over time? |
