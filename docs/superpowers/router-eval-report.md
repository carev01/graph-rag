# /answer Router Golden-Set Eval

> **Re-run after the community-report-verification slice (2026-09-10).**
>
> | | prev (2026-09-09) | this run |
> |---|---|---|
> | Routing accuracy | 0.97 | 0.97 |
> | Grounding precision | 0.88 | 0.88 |
> | — global grounding | 1.00 | 0.86 |
> | Faithfulness mean | 3.71 | **4.04** |
> | — **global faithfulness** | **1.6** | **2.33** |
> | unscored | 1/29 | 1/29 |
>
> **Global faithfulness rose 1.6 -> 2.33, a real measured gain** from stopping the
> community-report writer inventing specifics its own cited facts do not support. The
> question behind the original trace ("database restore workflows") went 0 -> 2; the
> encryption question went 4 -> 5. Comparative global went 1.8 -> 2.4.
>
> **Two caveats, stated plainly:**
> - **Not like-for-like.** The community layer is now 37 retrievable communities (of 41,
>   with 2 staged and 2 rejected) carrying only verified findings, against 41 carrying
>   unverified ones. Fewer, truer inputs.
> - **Global grounding dipped 1.00 -> 0.86** on a 7-question base — one question flipped.
>   That is within noise at this sample size, but it is a dip, not an improvement.
>
> **Global is still the weak mode: 2.33 against local's 4.79.** The remaining gap looks
> like a SELECTION problem rather than an invention problem. Observed 2026-09-10 on
> "What should I consider for backup encryption across cloud providers?": a community
> literally about KMS key policies scored relevance **3**, while an Azure BCDR platform
> overview with no encryption facts among its 25 cited ids scored **6**. `_MAP_PROMPT`
> asks for "relevance 0-10 (how useful)" with no anchors and no definition of relevance,
> so a dense off-topic report outranks a narrow on-topic one. See BACKLOG 3a-bis and 3d.
>
> **Harness note:** this run took over 2 hours against ~20 minutes previously, with only
> 2 retry warnings in the log — so it is provider route latency, not retries. The
> answer-path clients have no timeout and no throughput routing, unlike the report tier
> which was fixed for exactly this on 2026-09-08. See BACKLOG 5b.

Questions: 29

**Routing accuracy: 0.97**
by intent: {'local': 0.9333333333333333, 'global': 1.0, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.8846153846153846
by mode: {'local': 0.9285714285714286, 'timeline': 0.8, 'global': 0.8571428571428571}
Faithfulness mean: 4.04 (unscored: 1/29)
by mode: {'local': 4.785714285714286, 'timeline': 5.0, 'global': 2.3333333333333335}

Comparative (broad): {'local': {'faithfulness': 4.9, 'grounding': 0.8571428571428571}, 'global': {'faithfulness': 2.4, 'grounding': 0.8571428571428571}, 'drift': {'faithfulness': 4.3, 'grounding': 1.0}}
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
| local | local | True | True | 2 | How does Azure Backup encrypt backup data? |
| local | local | True | True | 5 | How do you configure and run Cross Region Restore in Azure Backup? |
| local | local | True | True | 5 | How do you restore a SQL Server database from an Azure Backup vault? |
| local | local | True | True | 5 | How do you back up an encrypted Azure VM? |
| local | local | True | True | 5 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | True | 5 | How do you change the backup retention period in AWS Backup? |
| global | global | True | True | 5 | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | global | True | True | 1 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | global | True | True | 4 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | global | True | None | - | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | global | True | False | 2 | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 1 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | True | 2 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 2 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | True | 2 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | global | True | None | 2 | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | True | 5 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | True | 5 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | False | 5 | How has Azure Backup encryption support changed over time? |
