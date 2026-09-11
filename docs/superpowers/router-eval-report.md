# /answer Router Golden-Set Eval

> **Run 2026-09-11 14:04 → 14:30 (26 min), after hardening every answer-path client
> (BACKLOG 5b + 5, branch `answer-path-client-hardening`). Full account:
> `answer-path-hardening-2026-09-11.md`.**
>
> Same corpus as the previous four runs (no `theme-build`). Code changes between the two
> runs: every answer-path client (synthesis/reduce, map, eval judge, classifier) now has a
> bounded timeout, OpenRouter throughput routing, and a per-tier reasoning bound
> (`effort: low` on synthesis and the eval judge; none on the `solar-pro4` map and
> classifier tiers, by measurement). No prompt, retrieval or reduce-format change.
>
> | | 0d run | 0b run (prev) | this run |
> |---|---|---|---|
> | Wall-clock | — | 1h44m | **26 min** |
> | Slowest question | — | 528s | **184s** |
> | Synthesis empty-content retries | — | 8 in ~40 reduce calls | **0** |
> | Routing accuracy | 0.97 | 0.97 | 0.97 |
> | Grounding precision | 0.88 | 0.92 | 0.92 |
> | Faithfulness mean | 3.75 | 4.90 | **5.00** |
> | — global | 1.6 (10/10) | 4.7 (10/10) | 5.0 (10/10) |
> | **unscored** | 1/29 | 0/29 | 0/29 |
> | ranges in any answer | 0/29 | 0/29 | 0/29 |
> | comparative global / drift | 2.5 / 4.7 | 4.9 / 4.8 | 5.0 / 4.8 |
>
> **This run is about latency and retries, not quality — read the 4.90 → 5.00 with care.**
> The three global-mode answers that scored 4 last run (the encryption comparison's "Both
> use KMS" over-reach, database restore workflows, drift encryption) scored 5. The eval
> judge itself changed in this run (`effort: low`), so it was probed before reading this:
> on a fixed answer set it still scores faithful=5, one-invented-claim=4, mostly-invented=0,
> wholly-unsupported=0 — the same as the unbounded judge (5, 3–4, 0, 0) — so the judge is
> not a rubber stamp. What DID change is the synthesis output: `cited` moved in 8 of 14
> local answers (8→1, 9→3, 5→3, 1→2, 10→8, 11→13 …) and in 8 of 10 global ones (89→64,
> 8→65, 22→37, 28→21 …). Different provider routes and a bounded reasoning budget produce different
> answers at `temperature=0`; three 4→5 moves on n=10 are within that variance and are
> not claimed as an improvement. The bag-sentence problem is unchanged: global `mps` max
> is still 18 (was 19), two answers still carry ≥11-marker sentences.

Questions: 29 (failed: 0)

**Routing accuracy: 0.97**
by intent: {'local': 0.9333333333333333, 'global': 1.0, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.9230769230769231
by mode: {'local': 0.9285714285714286, 'timeline': 0.8, 'global': 1.0}
Faithfulness mean: 5.00 (unscored: 0/29)
by mode: {'local': 5.0, 'timeline': 5.0, 'global': 5.0}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.6500000000000001, 'max': 5}, 'timeline': {'mean': 1.0, 'max': 1}, 'global': {'mean': 3.3299999999999996, 'max': 18}}

Comparative (broad): {'local': {'faithfulness': 5.0, 'grounding': 0.8571428571428571}, 'global': {'faithfulness': 5.0, 'grounding': 1.0}, 'drift': {'faithfulness': 4.8, 'grounding': 1.0}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | question |
|---|---|---|---|---|---|---|---|---|
| local | local | True | True | 5 | 5 | 0 | 1.8/2 | What does AWS Backup Vault Lock enforce on recovery points? |
| local | local | True | False | 5 | 1 | 0 | 1.0/1 | How does AWS Backup copy backups to another AWS Region? |
| local | local | True | True | 5 | 2 | 0 | 1.0/1 | How are backups encrypted in AWS Backup? |
| local | local | True | True | 5 | 11 | 0 | 2.2/3 | How do you restore an Amazon S3 backup? |
| local | local | True | True | 5 | 8 | 0 | 1.6/3 | How does AWS Backup support continuous backups and point-in-time recovery? |
| local | local | True | True | 5 | 11 | 0 | 1.8/3 | How are Amazon Redshift clusters backed up? |
| local | local | True | True | 5 | 3 | 0 | 1.0/1 | How does AWS Backup perform cross-account backup? |
| local | local | True | True | 5 | 13 | 0 | 3.0/5 | How do you restore an Amazon EC2 instance from a backup? |
| local | local | True | True | 5 | 3 | 0 | 1.0/1 | What is soft delete in Azure Backup and how long does it retain deleted items? |
| local | local | True | True | 5 | 2 | 0 | 2.0/2 | How does Azure Backup encrypt backup data? |
| local | local | True | True | 5 | 5 | 0 | 1.7/2 | How do you configure and run Cross Region Restore in Azure Backup? |
| local | local | True | True | 5 | 13 | 0 | 1.9/3 | How do you restore a SQL Server database from an Azure Backup vault? |
| local | local | True | True | 5 | 13 | 0 | 1.9/3 | How do you back up an encrypted Azure VM? |
| local | local | True | True | 5 | 5 | 0 | 1.2/2 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | True | 5 | 30 | 0 | 1.0/1 | How do you change the backup retention period in AWS Backup? |
| global | global | True | True | 5 | 14 | 0 | 2.9/6 | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | global | True | True | 5 | 14 | 0 | 2.9/6 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | global | True | True | 5 | 21 | 0 | 3.0/5 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | global | True | None | 5 | 18 | 0 | 4.5/11 | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | global | True | True | 5 | 37 | 0 | 3.4/9 | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 5 | 64 | 0 | 3.0/5 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | True | 5 | 37 | 0 | 3.0/8 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 5 | 64 | 0 | 4.3/18 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | True | 5 | 65 | 0 | 4.1/11 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | global | True | None | 5 | 28 | 0 | 2.2/5 | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | True | 5 | 30 | 0 | 1.0/1 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | False | 5 | 30 | 0 | 1.0/1 | How has Azure Backup encryption support changed over time? |
