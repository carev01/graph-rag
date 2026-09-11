# /answer Router Golden-Set Eval

> **Re-baseline on the REPAIRED corpus (2026-09-10) — NOT COMPARABLE, see below.**
>
> Level 1 was restored before this run: 19/19, 12/12, 9/9 retrievable, zero staged. The
> "Azure Backup: Encryption, Soft Delete, and Cross-Region Resiliency" community (219
> entities, 705 facts) is back, so global search can see both vendors for the first time —
> it now ranks #1 on the encryption question (0.7344) and the database-restore question
> (0.5781).
>
> | | prev (damaged corpus) | this run (repaired) |
> |---|---|---|
> | Routing accuracy | 0.97 | 0.97 |
> | Grounding precision | 0.88 | 0.85 |
> | Faithfulness mean | 4.07 | 3.92 |
> | — global | 2.30 | 1.5 |
> | **unscored** | **0/29** | **3/29** |
>
> **Do not read the global figure as a decline.** Three questions came back unscored
> because they **refused**, so 1.5 is an average over 8 of 10 global-mode questions rather
> than 10. Two global questions that previously answered now refuse.
>
> **Traced, not inferred.** It is neither selection nor missing evidence. On *"Compare AWS
> Backup and Azure Backup database restore workflows"*: the shortlist returns 4 survivors
> with the restored Azure community ranked #1; the map step returns 4 results carrying **29
> fact_ids** of substantive AWS *and* Azure content; and the reduce step refuses anyway.
>
> The likely trigger is map-step meta-commentary. Among the key points handed to reduce:
> *"The provided report contains no information about Azure Backup."* The reduce prompt was
> taught to refuse when findings do not support an answer, so a finding that literally says
> "no information" invites exactly that — while 29 cited facts sit beside it. The
> meta-commentary ban written into `_REDUCE_PROMPT` was never applied to `_MAP_PROMPT`.
>
> See BACKLOG 0c. **No faithfulness comparison is valid until that is fixed and this is
> re-run.**

Questions: 29 (failed: 0)

**Routing accuracy: 0.97**
by intent: {'local': 0.9333333333333333, 'global': 1.0, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.8461538461538461
by mode: {'local': 0.8571428571428571, 'timeline': 0.8, 'global': 0.8571428571428571}
Faithfulness mean: 3.92 (unscored: 3/29)
by mode: {'local': 5.0, 'timeline': 5.0, 'global': 1.5}

Comparative (broad): {'local': {'faithfulness': 5.0, 'grounding': 0.7142857142857143}, 'global': {'faithfulness': 1.5555555555555556, 'grounding': 1.0}, 'drift': {'faithfulness': 4.7, 'grounding': 1.0}}
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
| local | local | True | False | - | How does Azure Backup encrypt backup data? |
| local | local | True | True | 5 | How do you configure and run Cross Region Restore in Azure Backup? |
| local | local | True | True | 5 | How do you restore a SQL Server database from an Azure Backup vault? |
| local | local | True | True | 5 | How do you back up an encrypted Azure VM? |
| local | local | True | True | 5 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | True | 5 | How do you change the backup retention period in AWS Backup? |
| global | global | True | True | 5 | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | global | True | True | 1 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | global | True | True | 1 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | global | True | None | - | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | global | True | False | - | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 1 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | True | 1 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 1 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | True | 1 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | global | True | None | 1 | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | True | 5 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | True | 5 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | False | 5 | How has Azure Backup encryption support changed over time? |
