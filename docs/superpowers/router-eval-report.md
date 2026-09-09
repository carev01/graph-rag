# /answer Router Golden-Set Eval

> **Re-run after the answer-citation-integrity slice (2026-09-08).** Corpus unchanged
> from the previous run — 385 episodes, 2,173 facts, 41 communities, verified before the
> run — so this is like-for-like on retrieval. It is NOT like-for-like on the judge; see
> the health warning.
>
> **The result is split: grounding improved substantially, faithfulness did not.**
>
> | | prev (2026-09-08) | this run |
> |---|---|---|
> | Routing accuracy | 0.97 | 0.97 |
> | Grounding precision | 0.69 | **0.88** |
> | — global grounding | 0.29 | **1.00** |
> | Faithfulness mean | 3.76 | 3.71 |
> | — global faithfulness | 1.4 | **1.6** |
> | unscored | (not reported) | **1/29** |
>
> - **Global grounding went 0.29 → 1.00**, and held at 1.00 across both trustworthy
>   runs. Every global answer's citations resolve to the expected articles. This is the
>   slice working: unresolvable markers are stripped from the answer text, and the reduce
>   prompt requires a marker per claim. Traced live — the question behind the original
>   report (3,516 characters, 4 citations, opening with meta-commentary) now yields
>   ~1,900 characters with 21 citations, no meta-commentary and no orphan markers.
> - **Global faithfulness did not improve.** It reads 1.6 here against a 1.4 baseline,
>   and 1.1 on the immediately preceding run of the same code path. That spread is
>   run-to-run variance on a 10-question sample, not a trend: the honest summary is that
>   global faithfulness sits around 1-2 out of 5 and **prompting did not move it**.
>   Marker stripping cannot raise faithfulness by construction — the same claims remain
>   with fewer visible markers — so the prompt was the only lever, and it was not enough.
>   The next attempt must be structural, not another prompt revision.
> - **Why grounding and faithfulness diverge, traced not inferred.** Grounding 1.00 with
>   faithfulness ~1.6 means the citations point at the right articles while the prose is
>   not supported by them. In the worst case the answer asserts "PITR at 1-second
>   precision for up to 35 days", "retention ranges from 1-35 days", "not supported for
>   RDS Multi-AZ clusters" and Azure coverage of "SAP HANA and PostgreSQL" — **none of
>   these specifics appear in any of its 21 cited facts**, which state only general
>   things like "AWS Backup provides continuous backups for Amazon Aurora." The reduce
>   step invents numbers and service lists and attaches genuine markers to them. The
>   judge scoring this 1 is correct behaviour, not a judge problem.
>
> **Health warning: the "prev" faithfulness column was measured with a broken judge** and
> is indicative only. Two defects, both fixed in this branch:
>
> 1. `_faithfulness_judge` used `max_tokens=2000` against a reasoning model. On heavy
>    inputs it spent the whole budget on reasoning and returned `content=None`
>    (`finish_reason='length'`); `_parse_judge_score` scored that **0**, so "I could not
>    measure this" was recorded as "completely unfaithful". The causality was perverse —
>    better-cited answers enlarge the judge's prompt, which made it truncate, which
>    *lowered* the score. An intermediate run scored global faithfulness 0.0 for exactly
>    this reason, not because the answers got worse.
> 2. A claim-free answer scored **5** by vacuous truth ("no claims, so every claim is
>    supported"), inflating the metric in the opposite direction. This hit both blank
>    answers and refusals.
>
> The judge now returns *unscored* rather than 0 when it cannot measure, refuses to score
> a blank answer or a refusal, and the harness reports the count. The single `-` in the
> table below is that fix working: a local refusal reported as unmeasured instead of
> silently scoring 5.
>
> **Also fixed here, found while running this eval:** all three answer paths turned an
> empty LLM reply into a blank answer served with HTTP 200 and zero citations. The guard
> fired twice during this very run (`finish_reason=length`, then `finish_reason=error`)
> and recovered both times by retrying at a larger budget.
>
> **Known limitation, not addressed.** `_cited_fact_texts` returns facts in arbitrary
> Neo4j order and the judge prompt lists them unnumbered, so the judge can only check
> claims against the fact set *collectively* — it cannot verify that marker `[N]` is
> backed by the fact `[N]` actually points to. A misattributed-but-plausible marker would
> pass. Worth fixing before faithfulness gates anything.

Questions: 29

**Routing accuracy: 0.97**
by intent: {'local': 0.9333333333333333, 'global': 1.0, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.8846153846153846
by mode: {'local': 0.8571428571428571, 'timeline': 0.8, 'global': 1.0}
Faithfulness mean: 3.71 (unscored: 1/29)
by mode: {'local': 4.846153846153846, 'timeline': 5.0, 'global': 1.6}

Comparative (broad): {'local': {'faithfulness': 4.666666666666667, 'grounding': 0.8571428571428571}, 'global': {'faithfulness': 1.8, 'grounding': 0.8571428571428571}, 'drift': {'faithfulness': 4.3, 'grounding': 0.8571428571428571}}
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
| local | local | True | True | 4 | How does AWS Backup perform cross-account backup? |
| local | local | True | True | 5 | How do you restore an Amazon EC2 instance from a backup? |
| local | local | True | True | 5 | What is soft delete in Azure Backup and how long does it retain deleted items? |
| local | local | True | False | - | How does Azure Backup encrypt backup data? |
| local | local | True | True | 5 | How do you configure and run Cross Region Restore in Azure Backup? |
| local | local | True | True | 5 | How do you restore a SQL Server database from an Azure Backup vault? |
| local | local | True | True | 4 | How do you back up an encrypted Azure VM? |
| local | local | True | True | 5 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | True | 5 | How do you change the backup retention period in AWS Backup? |
| global | global | True | True | 4 | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | global | True | True | 1 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | global | True | True | 2 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | global | True | None | 2 | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | global | True | True | 1 | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 1 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | True | 1 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 1 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | True | 2 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | global | True | None | 1 | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | True | 5 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | True | 5 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | False | 5 | How has Azure Backup encryption support changed over time? |
