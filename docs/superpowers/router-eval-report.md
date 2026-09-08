# /answer Router Golden-Set Eval

> **Findings (2026-09-08, 54 articles / 385 episodes / 2,173 facts / 41 communities).**
> First run since the corpus was rebuilt, the community layer created, and the judge
> made independent. Raw harness output below; this header interprets it.
>
> **Comparable to the 2026-07-18 baseline** (routing 0.93 / grounding 0.73 /
> faithfulness 3.79): the same 29 labelled questions, and the 16 articles they are
> labelled against were deliberately re-ingested first — without that, grounding would
> have scored 0.00 for reasons unrelated to retrieval.
>
> **NOT comparable: faithfulness.** The old 3.79 came from a judge that *was* the
> synthesis model, grading its own answers. The judge is now `deepseek-v4-flash`,
> a different family. Today's 3.76 is the more trustworthy number even though it
> looks flat.
>
> - **Routing accuracy 0.97, up from 0.93.** global, drift and timeline all 1.0;
>   local 0.93. The single miss is the same understandable one as before.
> - **Timeline grounding 0.20 → 0.80.** The largest movement in the run, and it
>   tracks the temporal-coherence slice: timeline no longer surfaces facts whose
>   citations cannot resolve.
> - **Global mode is the weak spot: faithfulness 1.4, grounding 0.29.** Local and
>   timeline both score 5.0, so this is specific to the map-reduce path.
> - **`drift_wins = False` — and this time the question was actually testable.** The
>   July run reached the same verdict with ZERO communities in the graph, so DRIFT
>   could not have won. With 41 real communities it still loses to local
>   (5.0 vs 1.4 faithfulness on the broad subset).
>
> **Diagnosis of the global failure (traced, not inferred).** The pipeline itself
> works: for *"Compare AWS Backup and Azure Backup database restore workflows"* the
> shortlist returned 10 communities, the map step kept 2 (relevance 6 and 3) and
> surfaced 28 facts, and the envelope carried 4 resolved citations. The defect is in
> the REDUCE step's output — a 3,516-character answer supported by 4 citations, opening
> with meta-commentary about what the reports do not contain, and emitting marker
> *ranges* (`[31]-[60]`) that do not resolve to fact ids. The judge scores that 0
> correctly: the prose makes far more claims than the cited evidence supports.
>
> Two contributing causes, both worth a follow-up slice: the reduce prompt does not
> constrain answer length to its evidence, and marker ranges are not handled by the
> citation resolver. At this corpus scale a cross-vendor comparison genuinely has thin
> evidence (2 communities), and the model hedges verbosely rather than saying so.

Questions: 29

**Routing accuracy: 0.97**
by intent: {'local': 0.9333333333333333, 'global': 1.0, 'drift': 1.0, 'timeline': 1.0}
Grounding precision: 0.6923076923076923
by mode: {'local': 0.8571428571428571, 'timeline': 0.8, 'global': 0.2857142857142857}
Faithfulness mean: 3.76
by mode: {'local': 5.0, 'timeline': 5.0, 'global': 1.4}

Comparative (broad): {'local': {'faithfulness': 5.0, 'grounding': 0.7142857142857143}, 'global': {'faithfulness': 1.0, 'grounding': 0.7142857142857143}, 'drift': {'faithfulness': 1.4, 'grounding': 0.2857142857142857}}
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
| local | local | True | True | 5 | How do you restore a SQL Server database from an Azure Backup vault? |
| local | local | True | True | 5 | How do you back up an encrypted Azure VM? |
| local | local | True | True | 5 | How do you restore VMware VMs with Azure Backup Server? |
| local | timeline | False | True | 5 | How do you change the backup retention period in AWS Backup? |
| global | global | True | True | 4 | Compare how AWS Backup and Azure Backup encrypt backup data. |
| global | global | True | False | 5 | Compare cross-region restore between AWS Backup and Azure Backup. |
| global | global | True | False | 0 | How do AWS Backup and Azure Backup each protect recovery points from deletion? |
| global | global | True | None | 0 | Which backup capabilities do AWS and Azure share for compliance retention? |
| global | global | True | False | 0 | Compare AWS Backup and Azure Backup database restore workflows. |
| drift | global | True | None | 0 | What should I consider when planning long-term backup retention across cloud vendors? |
| drift | global | True | False | 0 | What should I consider for backup encryption across cloud providers? |
| drift | global | True | True | 0 | What are the key considerations for cross-region disaster recovery of cloud backups? |
| drift | global | True | False | 5 | How should I approach immutability and ransomware protection for cloud backups? |
| drift | global | True | None | 0 | What should I think about when restoring databases from cloud backups? |
| timeline | timeline | True | True | 5 | How has Azure Backup soft-delete retention changed over time? |
| timeline | timeline | True | True | 5 | How has AWS Backup vault lock behavior evolved? |
| timeline | timeline | True | True | 5 | How has the AWS Backup retention period configuration changed? |
| timeline | timeline | True | False | 5 | How has Azure Backup encryption support changed over time? |
