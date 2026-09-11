# /answer Router Golden-Set Eval

> **Run 2026-09-10 22:59 → 2026-09-11 01:04, after the map-prompt meta-commentary fix
> (BACKLOG 0c, branch `map-meta-commentary`). Full account:
> `map-meta-commentary-fix-2026-09-10.md`.**
>
> Same repaired corpus as the previous run (level 1 12/12 retrievable, 40 communities,
> no `theme-build`). The only code change between the two runs is `_MAP_PROMPT`.
>
> | | pre-repair | prev (repaired, 09-10) | this run |
> |---|---|---|---|
> | Routing accuracy | 0.97 | 0.97 | 0.97 |
> | Grounding precision | 0.88 | 0.85 | 0.88 |
> | Faithfulness mean | 4.07 | 3.92 | 4.11 |
> | — global | 2.30 (10/10) | 1.5 (8/10) | **2.44 (9/10)** |
> | **unscored** | **0/29** | **3/29** | **2/29** |
>
> **The two refusing global questions now answer** (compliance retention 1, database
> restore 2), so all five global-intent questions are scored. **Global faithfulness is
> flat against the pre-repair 2.30, not improved** — 2.44 over nine questions is within
> the run-to-run noise of this question set; do not read "1.5 → 2.44" as a gain, the 1.5
> averaged eight questions with the two hardest missing. The encryption control dropped
> 5 → 2 (it answers and cites both vendors; see the fix report §6.3 for the likely
> mechanism). One **new** refusal: "What should I think about when restoring databases
> from cloud backups?" (drift intent → global), untraced. Q10 (local) unscored as before.

Questions: 29 (failed: 0)

**Routing accuracy: 0.97**
by intent: {'local': 0.9333333333333333, 'global': 1.0, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.8846153846153846
by mode: {'local': 0.8571428571428571, 'timeline': 0.8, 'global': 1.0}
Faithfulness mean: 4.11 (unscored: 2/29)
by mode: {'local': 4.923076923076923, 'timeline': 5.0, 'global': 2.4444444444444446}

Comparative (broad): {'local': {'faithfulness': 4.888888888888889, 'grounding': 0.7142857142857143}, 'global': {'faithfulness': 1.6, 'grounding': 1.0}, 'drift': {'faithfulness': 4.5, 'grounding': 1.0}}
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
| local | local | True | True | 4 | How do you back up an encrypted Azure VM? |
| local | local | True | True | 5 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | True | 5 | How do you change the backup retention period in AWS Backup? |
| global | global | True | True | 2 | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | global | True | True | 4 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | global | True | True | 5 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | global | True | None | 1 | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | global | True | True | 2 | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 1 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | True | 2 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 1 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | True | 4 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | global | True | None | - | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | True | 5 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | True | 5 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | False | 5 | How has Azure Backup encryption support changed over time? |
