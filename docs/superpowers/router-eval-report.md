# /answer Router Golden-Set Eval

> **Run 2026-09-25, after the k3s bootstrap rehearsal (+ Cohesity FortKnox and Veeam VSPC,
> `merge-duplicates --apply`, `theme-build --full`).** First eval on the graph rebuilt
> since 2026-09-14, so not like-for-like with the 2026-09-11 run (grounding 0.92 there).
> Global grounding 0.29 is cross-vendor community crowding, traced in
> `bootstrap-rehearsal-2026-09-25.md` §7 (BACKLOG 52). Raw answers for this run were lost
> to a serialisation crash, since fixed.

Questions: 29 (failed: 0)

**Routing accuracy: 0.97**
by intent: {'local': 0.9333333333333333, 'global': 1.0, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.6923076923076923
by mode: {'local': 0.8571428571428571, 'timeline': 0.8, 'global': 0.2857142857142857}
Faithfulness mean: 5.00 (unscored: 1/29)
by mode: {'local': 5.0, 'timeline': 5.0, 'global': 5.0}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.6785714285714286, 'max': 6}, 'timeline': {'mean': 1.0, 'max': 1}, 'global': {'mean': 3.5777777777777775, 'max': 20}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.0, 'timeline': 0.0, 'global': 0.20333333333333334}

Comparative (broad): {'local': {'faithfulness': 5.0, 'grounding': 0.8571428571428571}, 'global': {'faithfulness': 4.8, 'grounding': 0.42857142857142855}, 'drift': {'faithfulness': 4.6, 'grounding': 0.7142857142857143}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | question |
|---|---|---|---|---|---|---|---|---|---|
| local | local | True | True | 5 | 3 | 0 | 1.5/2 | 0.0 | What does AWS Backup Vault Lock enforce on recovery points? |
| local | local | True | False | 5 | 2 | 0 | 1.0/1 | 0.0 | How does AWS Backup copy backups to another AWS Region? |
| local | local | True | True | 5 | 3 | 0 | 1.0/1 | 0.0 | How are backups encrypted in AWS Backup? |
| local | local | True | True | 5 | 11 | 0 | 2.2/4 | 0.0 | How do you restore an Amazon S3 backup? |
| local | local | True | True | 5 | 8 | 0 | 2.0/3 | 0.0 | How does AWS Backup support continuous backups and point-in-time recovery? |
| local | local | True | True | 5 | 11 | 0 | 1.4/2 | 0.0 | How are Amazon Redshift clusters backed up? |
| local | local | True | True | 5 | 12 | 0 | 2.4/6 | 0.0 | How does AWS Backup perform cross-account backup? |
| local | local | True | True | 5 | 14 | 0 | 2.0/5 | 0.0 | How do you restore an Amazon EC2 instance from a backup? |
| local | local | True | True | 5 | 3 | 0 | 1.5/2 | 0.0 | What is soft delete in Azure Backup and how long does it retain deleted items? |
| local | local | True | False | 5 | 2 | 0 | 2.0/2 | 0.0 | How does Azure Backup encrypt backup data? |
| local | local | True | True | 5 | 7 | 0 | 1.2/2 | 0.0 | How do you configure and run Cross Region Restore in Azure Backup? |
| local | local | True | True | 5 | 8 | 0 | 1.3/2 | 0.0 | How do you restore a SQL Server database from an Azure Backup vault? |
| local | local | True | True | 5 | 12 | 0 | 2.0/3 | 0.0 | How do you back up an encrypted Azure VM? |
| local | local | True | True | 5 | 4 | 0 | 2.0/2 | 0.0 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | True | 5 | 30 | 0 | 1.0/1 | 0.0 | How do you change the backup retention period in AWS Backup? |
| global | global | True | False | - | 0 | 0 | - | - | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | global | True | True | 5 | 6 | 0 | 2.7/3 | 0.0 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | global | True | False | 5 | 10 | 0 | 1.7/4 | 0.0 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | global | True | None | 5 | 19 | 0 | 4.8/10 | 0.53 | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | global | True | False | 5 | 19 | 0 | 2.3/6 | 0.0 | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 5 | 81 | 0 | 4.6/20 | 0.24 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | False | 5 | 52 | 0 | 5.6/20 | 0.5 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 5 | 53 | 0 | 3.4/10 | 0.19 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | False | 5 | 54 | 0 | 4.2/20 | 0.37 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | global | True | None | 5 | 40 | 0 | 2.9/6 | 0.0 | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | 0.0 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | 0.0 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | 0.0 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | False | 5 | 30 | 0 | 1.0/1 | 0.0 | How has Azure Backup encryption support changed over time? |
