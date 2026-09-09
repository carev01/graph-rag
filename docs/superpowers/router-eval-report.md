# /answer Router Golden-Set Eval

> **Re-run after the answer-citation-integrity slice (2026-09-08).** Corpus unchanged
> from the previous run — 385 episodes, 2,173 facts, 41 communities, verified before
> the run — so this is like-for-like on retrieval. It is NOT like-for-like on the
> judge; see the health warning below.
>
> **The result is split. Grounding improved a lot. Faithfulness did not improve.**
>
> | | prev (2026-09-08) | this run |
> |---|---|---|
> | Routing accuracy | 0.97 | 0.97 |
> | Grounding precision | 0.69 | **0.88** |
> | — global grounding | 0.29 | **1.00** |
> | Faithfulness mean | 3.76 | 3.62 |
> | — global faithfulness | 1.4 | **1.1** |
> | unscored | (not reported) | **0/29** |
>
> - **Global grounding went 0.29 → 1.00.** Every global answer's citations now resolve
>   to the expected articles. This is the slice working: unresolvable markers are
>   stripped from the answer text, and the reduce prompt requires a marker per claim.
>   Traced live: the question that previously produced 3,516 characters carrying 4
>   citations and opening with meta-commentary now produces ~1,900 characters with 21
>   citations, no meta-commentary, and zero orphan markers.
> - **Global faithfulness did not improve: 1.4 → 1.1.** Stated plainly, prompting did
>   not fix the reduce step. Marker stripping *cannot* raise faithfulness by
>   construction — the same claims remain with fewer visible markers — so the prompt
>   change was the only lever here, and it was not enough. The next attempt has to be
>   structural, not another prompt revision.
> - **Why the two moved in opposite directions, traced not inferred.** Grounding 1.00
>   with faithfulness 1.1 means the citations point at the right articles while the
>   prose is not supported by them. Inspecting the worst case: the answer asserts "PITR
>   at 1-second precision for up to 35 days", "retention ranges from 1-35 days", "not
>   supported for RDS Multi-AZ clusters", and Azure coverage of "SAP HANA and
>   PostgreSQL" — **none of these specifics appear in any of the 21 cited facts**, which
>   say only general things like "AWS Backup provides continuous backups for Amazon
>   Aurora." The reduce step invents numbers and service lists and attaches genuine
>   markers to them. The judge scoring this ~0-1 is correct behaviour.
>
> **Health warning on the comparison.** The previous run's numbers were measured with a
> broken judge, so treat the "prev" faithfulness column as indicative only:
>
> 1. `_faithfulness_judge` used `max_tokens=2000` against a reasoning model. On heavy
>    inputs it spent the whole budget on reasoning and returned `content=None`
>    (`finish_reason='length'`); `_parse_judge_score` then scored that **0**. So "I could
>    not measure this" was recorded as "this answer is completely unfaithful". The
>    causality was perverse: better-cited answers make the judge's prompt bigger, which
>    made it truncate, which *lowered* the score. An intermediate run of this slice
>    scored global faithfulness 0.0 for exactly this reason, not because the answers got
>    worse.
> 2. A blank answer scored **5** by vacuous truth ("no claims, so every claim is
>    supported"), inflating the metric in the opposite direction.
>
> Both are fixed. The judge now returns *unscored* rather than 0 when it cannot measure,
> refuses to score a blank answer, and the harness reports the unscored count — **0/29
> here**, so every number above was actually measured.
>
> **Known limitation, not yet addressed.** `_cited_fact_texts` returns facts in
> arbitrary Neo4j order and the judge prompt lists them unnumbered, so the judge can
> only check claims against the fact set *collectively* — it cannot verify that marker
> `[N]` is backed by the fact `[N]` actually points to. A misattributed-but-plausible
> marker would pass. Worth fixing before faithfulness is used to gate anything.

Questions: 29

**Routing accuracy: 0.97**
by intent: {'local': 0.9333333333333333, 'global': 1.0, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.8846153846153846
by mode: {'local': 0.8571428571428571, 'timeline': 0.8, 'global': 1.0}
Faithfulness mean: 3.62 (unscored: 0/29)
by mode: {'local': 4.928571428571429, 'timeline': 5.0, 'global': 1.1}

Comparative (broad): {'local': {'faithfulness': 5.0, 'grounding': 0.8571428571428571}, 'global': {'faithfulness': 1.8, 'grounding': 1.0}, 'drift': {'faithfulness': 4.7, 'grounding': 0.8571428571428571}}
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
| local | local | True | False | 5 | How does Azure Backup encrypt backup data? |
| local | local | True | True | 5 | How do you configure and run Cross Region Restore in Azure Backup? |
| local | local | True | True | 4 | How do you restore a SQL Server database from an Azure Backup vault? |
| local | local | True | True | 5 | How do you back up an encrypted Azure VM? |
| local | local | True | True | 5 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | True | 5 | How do you change the backup retention period in AWS Backup? |
| global | global | True | True | 4 | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | global | True | True | 1 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | global | True | True | 2 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | global | True | None | 0 | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | global | True | True | 0 | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 0 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | True | 1 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 1 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | True | 1 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | global | True | None | 1 | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | True | 5 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | True | 5 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | False | 5 | How has Azure Backup encryption support changed over time? |
