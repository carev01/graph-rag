# /answer Router Golden-Set Eval

> **Findings (2026-07-18, `backup-docs` = AWS + Azure, 29 questions).** Raw
> harness output below; this header interprets it.
>
> - **Routing accuracy 0.93** — strong. Per intent: local 0.93, global 0.80,
>   drift 1.0, timeline 1.0. The two misroutes are understandable: *"How do you
>   change the retention period…"* → timeline (the temporal heuristic caught
>   "change"), and *"How do AWS and Azure each protect recovery points…"* → local
>   (the classifier read it as specific rather than cross-vendor).
> - **Grounding precision 0.73** — local 0.80, global 1.00, timeline 0.20.
>   Timeline routing + faithfulness are perfect, but timeline retrieval often cites
>   change-event facts from *different* articles than the labeled ones → low
>   grounding-precision (a retrieval-targeting observation, not a wrong answer).
> - **Faithfulness mean 3.79** (judge 0–5). local 4.4, timeline 5.0, global 2.375.
>   The `drift: 0.0` by-mode value is a **single-sample artifact** — only ONE
>   drift-intent question actually *chose* drift in the main pass (the other four
>   routed to `global`, both acceptable per `expected_modes`), and that one scored
>   0. The comparative pass below is the real drift signal.
> - **Comparative (10 broad questions forced through each mode):** local 4.6 /
>   global 2.6 / **drift 3.5** faithfulness. So **`drift_wins = False`**: DRIFT
>   beats *global* but not *local* on broad questions at this corpus scale — the
>   weak-signal outcome the spec anticipated for a 2-vendor graph. Re-run once more
>   vendors are ingested.
> - **Bug found + fixed during this run:** the faithfulness judge used
>   `max_tokens=8`, which GLM-5.2 (a reasoning model) returns as EMPTY content →
>   every score 0. Raised to 2000 (bare integer now); these are the post-fix numbers.

Questions: 29

**Routing accuracy: 0.93**
by intent: {'local': 0.9333333333333333, 'global': 0.8, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.7307692307692307
by mode: {'local': 0.8, 'timeline': 0.2, 'global': 1.0}
Faithfulness mean: 3.79
by mode: {'local': 4.4, 'timeline': 5.0, 'global': 2.375, 'drift': 0.0}

Comparative (broad): {'local': {'faithfulness': 4.6, 'grounding': 0.8571428571428571}, 'global': {'faithfulness': 2.6, 'grounding': 0.8571428571428571}, 'drift': {'faithfulness': 3.5, 'grounding': 0.7142857142857143}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | question |
|---|---|---|---|---|---|
| local | local | True | True | 5 | What does AWS Backup Vault Lock enforce on recovery points? |
| local | local | True | True | 5 | How does AWS Backup copy backups to another AWS Region? |
| local | local | True | True | 5 | How are backups encrypted in AWS Backup? |
| local | local | True | True | 5 | How do you restore an Amazon S3 backup? |
| local | local | True | True | 5 | How does AWS Backup support continuous backups and point-in-time recovery? |
| local | local | True | True | 5 | How are Amazon Redshift clusters backed up? |
| local | local | True | True | 4 | How does AWS Backup perform cross-account backup? |
| local | local | True | True | 5 | How do you restore an Amazon EC2 instance from a backup? |
| local | local | True | False | 0 | What is soft delete in Azure Backup and how long does it retain deleted items? |
| local | local | True | True | 5 | How does Azure Backup encrypt backup data? |
| local | local | True | False | 5 | How do you configure and run Cross Region Restore in Azure Backup? |
| local | local | True | True | 3 | How do you restore a SQL Server database from an Azure Backup vault? |
| local | local | True | True | 4 | How do you back up an encrypted Azure VM? |
| local | local | True | False | 5 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | False | 5 | How do you change the backup retention period in AWS Backup? |
| global | global | True | True | 3 | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | global | True | True | 2 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | local | False | True | 5 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | global | True | None | 4 | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | global | True | True | 2 | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 2 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | True | 2 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 2 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | True | 2 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | drift | True | None | 0 | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | False | 5 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | False | 5 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | False | 5 | How has Azure Backup encryption support changed over time? |
