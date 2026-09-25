# /answer Router Golden-Set Eval

> **Run 2026-09-25 on `d946540` (vendor/product end to end, acceptance).** Re-baselined
> metrics: scoped grounding and classifier routing are the acceptance measures;
> cross-vendor grounding and answer-path routing are informational. See
> `vendor-product-end-to-end-report.md`.

Questions: 29 (failed: 0)

**Routing accuracy: 0.83** (answer path); classifier: 0.97
by intent: {'local': 0.9333333333333333, 'global': 0.2, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.8846153846153846
**Grounding, scoped questions: 0.9565217391304348 (n=23)** | cross-vendor, informational (golden answers predate the Tier 1 vendors): 0.3333333333333333 (n=3)
by mode: {'local': 0.9411764705882353, 'timeline': 1.0, 'drift': 1.0, 'global': 0.3333333333333333}
Faithfulness mean: 5.00 (unscored: 0/29)
by mode: {'local': 5.0, 'timeline': 5.0, 'drift': 5.0, 'global': 5.0}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.7722222222222221, 'max': 8}, 'timeline': {'mean': 1.0, 'max': 1}, 'drift': {'mean': 2.3, 'max': 7}, 'global': {'mean': 3.18, 'max': 20}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.016666666666666666, 'timeline': 0.0, 'drift': 0.0, 'global': 0.182}

Misattributed claims: 6 across 1 answers (unscored: 0)
Comparative (broad): {'local': {'faithfulness': 4.888888888888889, 'grounding': 0.7142857142857143}, 'global': {'faithfulness': 4.9, 'grounding': 0.5714285714285714}, 'drift': {'faithfulness': 4.6, 'grounding': 0.5714285714285714}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | misattributed | question |
|---|---|---|---|---|---|---|---|---|---|---|
| local | local | True | True | 5 | 2 | 0 | 1.0/1 | 0.0 | 0 | What does AWS Backup Vault Lock enforce on recovery points? |
| local | local | True | True | 5 | 7 | 0 | 1.8/3 | 0.0 | 0 | How does AWS Backup copy backups to another AWS Region? |
| local | local | True | True | 5 | 4 | 0 | 1.0/1 | 0.0 | 0 | How are backups encrypted in AWS Backup? |
| local | local | True | True | 5 | 10 | 0 | 1.7/3 | 0.0 | 0 | How do you restore an Amazon S3 backup? |
| local | local | True | True | 5 | 14 | 0 | 2.0/4 | 0.0 | 0 | How does AWS Backup support continuous backups and point-in-time recovery? |
| local | local | True | True | 5 | 12 | 0 | 1.3/2 | 0.0 | 0 | How are Amazon Redshift clusters backed up? |
| local | local | True | True | 5 | 13 | 0 | 3.2/6 | 0.0 | 0 | How does AWS Backup perform cross-account backup? |
| local | local | True | True | 5 | 12 | 0 | 1.7/3 | 0.0 | 0 | How do you restore an Amazon EC2 instance from a backup? |
| local | local | True | True | 5 | 3 | 0 | 1.5/2 | 0.0 | 0 | What is soft delete in Azure Backup and how long does it retain deleted items? |
| local | local | True | False | 5 | 3 | 0 | 1.5/2 | 0.0 | 0 | How does Azure Backup encrypt backup data? |
| local | local | True | True | 5 | 12 | 0 | 1.7/4 | 0.0 | 0 | How do you configure and run Cross Region Restore in Azure Backup? |
| local | local | True | True | 5 | 10 | 0 | 1.4/2 | 0.0 | 0 | How do you restore a SQL Server database from an Azure Backup vault? |
| local | local | True | True | 5 | 15 | 0 | 1.7/3 | 0.0 | 0 | How do you back up an encrypted Azure VM? |
| local | local | True | True | 5 | 4 | 0 | 2.0/2 | 0.0 | 0 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | True | 5 | 30 | 0 | 1.0/1 | 0.0 | 0 | How do you change the backup retention period in AWS Backup? |
| global | local | False | True | 5 | 13 | 0 | 2.0/7 | 0.0 | 0 | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | local | False | True | 5 | 12 | 0 | 2.0/4 | 0.0 | 0 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | drift | True | True | 5 | 25 | 0 | 2.3/7 | 0.0 | 0 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | local | False | None | 5 | 3 | 0 | 1.4/2 | 0.0 | 0 | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | local | False | True | 5 | 15 | 0 | 3.0/8 | 0.3 | 0 | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 5 | 7 | 0 | 2.3/4 | 0.0 | 0 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | False | 5 | 40 | 0 | 3.6/16 | 0.4 | 0 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 5 | 62 | 0 | 2.7/10 | 0.16 | 0 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | False | 5 | 57 | 0 | 4.8/20 | 0.35 | 0 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | global | True | None | 5 | 48 | 0 | 2.5/6 | 0.0 | 6 | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | 0.0 | 0 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | 0.0 | 0 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | 0.0 | 0 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | 0.0 | 0 | How has Azure Backup encryption support changed over time? |
