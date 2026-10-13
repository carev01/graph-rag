# Tier 1 (Veeam) golden-set eval

## Run 8 -- 2026-10-13, facet-balanced answers, final (branch facet-answers)

After the run 7 fixes: product-named side questions for enumerations, rewrite fallback never
re-checks the cross-vendor question, faceted global keeps local-grounded answers, single rerank
per pooled search, enumeration marker/clause rules, cross-site ground truth includes Cohesity.

| | run 5 | run 6 | run 7 | **run 8** | target |
|---|---|---|---|---|---|
| routing | 1.00 | 1.00 | 1.00 | **1.00** | 1.00 |
| scoped grounding | 0.70 | 0.83 | 1.00 | **1.00 (12/12)** | 1.00 |
| faithfulness | 5.00 | 4.50 | 4.88 | **4.56** (4.87 excl. the probe) | 5.0 |
| misattributed claims | 0 | 1 | 1 | **0** | 0 |
| facet coverage | -- | 0.83 | 1.00 | **1.00 (6/6)** | 1.00 |
| enumeration completeness | -- | 0.00 | 0.33 | **0.67 (2/3)** | 1.00 |
| eval cost | | $0.15 | $0.24 | **$0.20** | |

Open: (1) restore verification documents Microsoft + Veeam, not AWS ("Restore testing" not in
AWS's gathered evidence); (2) the probe "Which vendors does Veeam integrate with?" is classified
as an enumeration (any "vendors" wording is, per spec §2) and answers with each vendor's own
integrations -- faithfulness 0. A question whose subject is a named vendor needs a different shape.

## Run 8 detail


Questions: 16 (failed: 0)

**Routing accuracy: 1.00** (answer path); classifier: 1.00
by intent: {'local': 1.0, 'drift': 1.0, 'global': 1.0}
Grounding precision: 0.8666666666666667
**Grounding, scoped questions: 1.0 (n=12)** | cross-vendor, informational (golden answers predate the Tier 1 vendors): 0.3333333333333333 (n=3)
by mode: {'local': 1.0, 'global': 0.7777777777777778}
Faithfulness mean: 4.56 (unscored: 0/16)
by mode: {'local': 5.0, 'global': 4.3}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.4333333333333333, 'max': 2}, 'global': {'mean': 2.83, 'max': 11}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.0, 'global': 0.086}

Misattributed claims: 0 across 0 answers (unscored: 0)
**Facet coverage (comparisons): 1.0 (n=6)** | **Enumeration completeness: 0.67 (n=3)**
Comparative (broad): {'local': {'faithfulness': 4.4, 'grounding': 0.7777777777777778}, 'global': {'faithfulness': 4.6, 'grounding': 0.8888888888888888}, 'drift': {'faithfulness': 4.8, 'grounding': 0.6666666666666666}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | misattributed | facet_coverage | enumeration_complete | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| local | local | True | True | 5 | 7 | 0 | 1.2/2 | 0.0 | 0 | - | - | How does a Veeam Backup & Replication hardened repository keep backups immutable? |
| local | local | True | True | 5 | 5 | 0 | 1.7/2 | 0.0 | 0 | - | - | What does a Veeam SureBackup job verify, and how is it processed? |
| local | local | True | True | 5 | 9 | 0 | 1.5/2 | 0.0 | 0 | - | - | How does Veeam Backup & Replication detect malware in backups? |
| local | local | True | True | 5 | 15 | 0 | 1.5/2 | 0.0 | 0 | - | - | In Veeam ONE, what happens when an alarm is triggered and how are remediation actions approved? |
| local | local | True | True | 5 | 14 | 0 | 1.4/2 | 0.0 | 0 | - | - | How do you halt and resume plan testing in Veeam Recovery Orchestrator? |
| drift | local | True | True | 5 | 8 | 0 | 1.3/2 | 0.0 | 0 | - | - | What should I consider when enabling backup immutability for Veeam Agent backups? |
| drift | global | True | True | 5 | 24 | 0 | 2.3/4 | 0.0 | 0 | True | - | Compare how Veeam hardened repositories and AWS Backup Vault Lock protect backups from deletion. |
| global | global | True | True | 4 | 29 | 0 | 3.7/8 | 0.22 | 0 | True | - | Compare Veeam backup copy jobs with AWS Backup cross-Region copy for keeping an off-site copy. |
| drift | global | True | True | 5 | 41 | 0 | 3.7/5 | 0.0 | 0 | True | - | How do Veeam Backup & Replication and Azure Backup each protect backups against ransomware? |
| local | global | True | True | 5 | 19 | 0 | 2.5/4 | 0.0 | 0 | True | - | How does immutability in Veeam Backup for Microsoft 365 compare with soft delete in Azure Backup? |
| global | global | True | False | 5 | 20 | 0 | 3.1/11 | 0.35 | 0 | - | True | Which backup vendors support immutable backups, and how do their approaches differ? |
| drift | global | True | False | 5 | 17 | 0 | 2.8/4 | 0.0 | 0 | - | False | What approaches do backup vendors take to verify that backups can actually be restored? |
| global | global | True | True | 5 | 15 | 0 | 1.5/2 | 0.0 | 0 | - | True | Which backup vendors can keep a copy of backups in another region or at another site? |
| global | global | True | True | 5 | 18 | 0 | 2.7/6 | 0.0 | 0 | True | - | Compare AWS Backup Vault Lock with Cohesity DataLock. |
| global | global | True | True | 4 | 26 | 0 | 3.5/8 | 0.29 | 0 | True | - | How does Cohesity FortKnox compare with Veeam hardened repositories for ransomware protection? |
| global | global | True | None | 0 | 28 | 0 | 2.5/7 | 0.0 | 0 | - | - | Which vendors does Veeam integrate with? |

## Run 7 -- 2026-10-13, facet-balanced answers (branch facet-answers)

16 questions (4 added 2026-10-13; ransomware question's expected articles widened, user-approved).
Shape-aware retrieval: comparisons retrieve per facet (rewritten side questions, scoped + reranked
local facts, per-facet community shortlist with a local supplement); enumerations check every
content vendor. Answer models unchanged.

| | run 5 | run 6 | **run 7** | target |
|---|---|---|---|---|
| questions | 12 | 16 | **16** | |
| routing | 1.00 | 1.00 | **1.00** | 1.00 |
| scoped grounding | 0.70 | 0.83 | **1.00 (12/12)** | 1.00 |
| faithfulness | 5.00 | 4.50 | **4.88** | 5.0 |
| misattributed claims | 0 | 1 | **1** | 0 |
| facet coverage (comparisons) | -- | 0.83 | **1.00 (6/6)** | 1.00 |
| enumeration completeness | -- | 0.00 | **0.33 (1/3)** | 1.00 |
| eval cost | | $0.15 | **$0.24** | |

Enumerations: immutability -- AWS, Cohesity, Microsoft, Veeam (complete). Cross-region / off-site
copies -- answer lists all four; ground truth AWS/Microsoft/Veeam (Cohesity FortKnox vaulted
off-site copies arguably qualify; pending user decision). Restore verification -- only Veeam
documented; AWS "Restore testing" and Microsoft "test restores" were not in the gathered evidence.

## Run 7 detail


Questions: 16 (failed: 0)

**Routing accuracy: 1.00** (answer path); classifier: 1.00
by intent: {'local': 1.0, 'drift': 1.0, 'global': 1.0}
Grounding precision: 0.8666666666666667
**Grounding, scoped questions: 1.0 (n=12)** | cross-vendor, informational (golden answers predate the Tier 1 vendors): 0.3333333333333333 (n=3)
by mode: {'local': 1.0, 'global': 0.7777777777777778}
Faithfulness mean: 4.88 (unscored: 0/16)
by mode: {'local': 5.0, 'global': 4.8}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.7, 'max': 4}, 'global': {'mean': 2.45, 'max': 14}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.0, 'global': 0.064}

Misattributed claims: 1 across 1 answers (unscored: 0)
**Facet coverage (comparisons): 1.0 (n=6)** | **Enumeration completeness: 0.33 (n=3)**
Comparative (broad): {'local': {'faithfulness': 4.7, 'grounding': 0.7777777777777778}, 'global': {'faithfulness': 4.8, 'grounding': 0.8888888888888888}, 'drift': {'faithfulness': 4.6, 'grounding': 0.6666666666666666}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | misattributed | facet_coverage | enumeration_complete | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| local | local | True | True | 5 | 7 | 0 | 1.2/2 | 0.0 | 0 | - | - | How does a Veeam Backup & Replication hardened repository keep backups immutable? |
| local | local | True | True | 5 | 7 | 0 | 2.3/3 | 0.0 | 0 | - | - | What does a Veeam SureBackup job verify, and how is it processed? |
| local | local | True | True | 5 | 9 | 0 | 1.5/2 | 0.0 | 0 | - | - | How does Veeam Backup & Replication detect malware in backups? |
| local | local | True | True | 5 | 15 | 0 | 1.9/3 | 0.0 | 0 | - | - | In Veeam ONE, what happens when an alarm is triggered and how are remediation actions approved? |
| local | local | True | True | 5 | 8 | 0 | 1.6/2 | 0.0 | 0 | - | - | How do you halt and resume plan testing in Veeam Recovery Orchestrator? |
| drift | local | True | True | 5 | 12 | 0 | 1.7/4 | 0.0 | 0 | - | - | What should I consider when enabling backup immutability for Veeam Agent backups? |
| drift | global | True | True | 4 | 31 | 0 | 2.8/6 | 0.0 | 0 | True | - | Compare how Veeam hardened repositories and AWS Backup Vault Lock protect backups from deletion. |
| global | global | True | True | 5 | 30 | 0 | 3.4/5 | 0.0 | 1 | True | - | Compare Veeam backup copy jobs with AWS Backup cross-Region copy for keeping an off-site copy. |
| drift | global | True | True | 5 | 42 | 0 | 3.2/6 | 0.0 | 0 | True | - | How do Veeam Backup & Replication and Azure Backup each protect backups against ransomware? |
| local | global | True | True | 5 | 19 | 0 | 2.4/4 | 0.0 | 0 | True | - | How does immutability in Veeam Backup for Microsoft 365 compare with soft delete in Azure Backup? |
| global | global | True | False | 4 | 27 | 0 | 3.3/14 | 0.64 | 0 | - | True | Which backup vendors support immutable backups, and how do their approaches differ? |
| drift | global | True | False | 5 | 5 | 0 | 1.2/2 | 0.0 | 0 | - | False | What approaches do backup vendors take to verify that backups can actually be restored? |
| global | global | True | True | 5 | 21 | 0 | 1.4/2 | 0.0 | 0 | - | False | Which backup vendors can keep a copy of backups in another region or at another site? |
| global | global | True | True | 5 | 21 | 0 | 1.6/3 | 0.0 | 0 | True | - | Compare AWS Backup Vault Lock with Cohesity DataLock. |
| global | global | True | True | 5 | 23 | 0 | 3.2/6 | 0.0 | 0 | True | - | How does Cohesity FortKnox compare with Veeam hardened repositories for ransomware protection? |
| global | global | True | None | 5 | 10 | 0 | 2.0/3 | 0.0 | 0 | - | - | Which vendors does Veeam integrate with? |

## Run 6 -- 2026-10-13 (facet-balanced, before the run 6 fixes)


Questions: 16 (failed: 0)

**Routing accuracy: 1.00** (answer path); classifier: 1.00
by intent: {'local': 1.0, 'drift': 1.0, 'global': 1.0}
Grounding precision: 0.8
**Grounding, scoped questions: 0.8333333333333334 (n=12)** | cross-vendor, informational (golden answers predate the Tier 1 vendors): 0.6666666666666666 (n=3)
by mode: {'local': 1.0, 'global': 0.6666666666666666}
Faithfulness mean: 4.50 (unscored: 0/16)
by mode: {'local': 4.833333333333333, 'global': 4.3}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.7166666666666666, 'max': 4}, 'global': {'mean': 2.977777777777778, 'max': 30}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.0, 'global': 0.12333333333333332}

Misattributed claims: 1 across 1 answers (unscored: 0)
**Facet coverage (comparisons): 0.83 (n=6)** | **Enumeration completeness: 0.0 (n=3)**
Comparative (broad): {'local': {'faithfulness': 4.8, 'grounding': 0.7777777777777778}, 'global': {'faithfulness': 5.0, 'grounding': 0.6666666666666666}, 'drift': {'faithfulness': 4.8, 'grounding': 0.8888888888888888}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | misattributed | facet_coverage | enumeration_complete | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| local | local | True | True | 5 | 5 | 0 | 1.0/1 | 0.0 | 0 | - | - | How does a Veeam Backup & Replication hardened repository keep backups immutable? |
| local | local | True | True | 4 | 7 | 0 | 2.3/3 | 0.0 | 0 | - | - | What does a Veeam SureBackup job verify, and how is it processed? |
| local | local | True | True | 5 | 9 | 0 | 1.5/2 | 0.0 | 0 | - | - | How does Veeam Backup & Replication detect malware in backups? |
| local | local | True | True | 5 | 15 | 0 | 1.9/3 | 0.0 | 0 | - | - | In Veeam ONE, what happens when an alarm is triggered and how are remediation actions approved? |
| local | local | True | True | 5 | 13 | 0 | 1.9/3 | 0.0 | 0 | - | - | How do you halt and resume plan testing in Veeam Recovery Orchestrator? |
| drift | local | True | True | 5 | 12 | 0 | 1.7/4 | 0.0 | 0 | - | - | What should I consider when enabling backup immutability for Veeam Agent backups? |
| drift | global | True | True | 4 | 24 | 0 | 3.4/7 | 0.0 | 0 | True | - | Compare how Veeam hardened repositories and AWS Backup Vault Lock protect backups from deletion. |
| global | global | True | True | 5 | 20 | 0 | 3.3/8 | 0.4 | 1 | True | - | Compare Veeam backup copy jobs with AWS Backup cross-Region copy for keeping an off-site copy. |
| drift | global | True | True | 5 | 32 | 0 | 2.7/5 | 0.0 | 0 | True | - | How do Veeam Backup & Replication and Azure Backup each protect backups against ransomware? |
| local | global | True | False | 5 | 3 | 0 | 3.0/3 | 0.0 | 0 | False | - | How does immutability in Veeam Backup for Microsoft 365 compare with soft delete in Azure Backup? |
| global | global | True | False | 0 | 0 | 0 | - | - | 0 | - | False | Which backup vendors support immutable backups, and how do their approaches differ? |
| drift | global | True | True | 5 | 8 | 0 | 1.3/2 | 0.0 | 0 | - | False | What approaches do backup vendors take to verify that backups can actually be restored? |
| global | global | True | True | 5 | 25 | 0 | 2.3/4 | 0.0 | 0 | - | False | Which backup vendors can keep a copy of backups in another region or at another site? |
| global | global | True | False | 5 | 5 | 0 | 1.8/4 | 0.0 | 0 | True | - | Compare AWS Backup Vault Lock with Cohesity DataLock. |
| global | global | True | True | 5 | 55 | 0 | 7.9/30 | 0.71 | 0 | True | - | How does Cohesity FortKnox compare with Veeam hardened repositories for ransomware protection? |
| global | global | True | None | 4 | 13 | 0 | 1.1/2 | 0.0 | 0 | - | - | Which vendors does Veeam integrate with? |

## Run 5 -- 2026-10-11, Haiku 5.5 reports over AWS/Azure + Veeam, entity-aware context

Graph: AWS Backup / Azure Backup ingested (620 articles), 17 exact-name duplicates merged.
Community reports: all 883 regenerated by Claude Haiku 5.5 (effort medium), verified by
DeepSeek V4 Flash, with report context chosen by top entities and sized to the community
(`docs/superpowers/haiku-report-tier-ab-2026-10-11.md`). Answer-side models unchanged.

| | run 3 | run 4 | **run 5** | target |
|---|---|---|---|---|
| routing | 1.00 | 1.00 | **1.00** | >= 0.97 |
| misattributed claims | 0 | 2 | **0** | reported |
| faithfulness | 4.92 | 4.92 | **5.00** (1 unscored) | >= 4.8 |
| scoped grounding | 1.00 (10/10) | 0.60 | **0.70** | >= 0.8 |
| global-mode grounding | 0.67 | 0.20 | **0.50** | -- |
| cross-vendor grounding (informational) | 1/2 | 1/2 | **1/2** | -- |

Run 4 (not kept as a result) was the same graph with Haiku reports over the OLD
recency-only context: the AWS/Azure hub community's context held 42 facts, 0 on Vault
Lock, and every cross-vendor global answer cited Veeam only. In run 5 the Vault Lock
comparison cites AWS sources and names Vault Lock; the Azure ransomware comparison
cites Microsoft sources; "Which backup vendors support immutable backups?" cites
Cohesity and Veeam but still neither AWS nor Azure.

Below target: scoped grounding 0.70. Misses -- the Veeam Agent immutability question
(LOCAL mode, which reads no community reports: it regressed with the AWS/Azure ingest
and merge in run 4, not with the reports), the Vault Lock comparison (cites AWS, but
not the golden Veeam hardened-repository articles), and the Microsoft 365 immutability
vs Azure soft-delete question (routed global, cited 2 Microsoft facts, no Veeam).

## Run 5 detail


Questions: 12 (failed: 0)

**Routing accuracy: 1.00** (answer path); classifier: 1.00
by intent: {'local': 1.0, 'drift': 1.0, 'global': 1.0}
Grounding precision: 0.6666666666666666
**Grounding, scoped questions: 0.7 (n=10)** | cross-vendor, informational (golden answers predate the Tier 1 vendors): 0.5 (n=2)
by mode: {'local': 0.8333333333333334, 'global': 0.5}
Faithfulness mean: 5.00 (unscored: 1/12)
by mode: {'local': 5.0, 'global': 5.0}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.8, 'max': 4}, 'global': {'mean': 3.033333333333333, 'max': 12}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.0, 'global': 0.07166666666666667}

Misattributed claims: 0 across 0 answers (unscored: 0)
Comparative (broad): {'local': {'faithfulness': 5.0, 'grounding': 0.16666666666666666}, 'global': {'faithfulness': 5.0, 'grounding': 0.5}, 'drift': {'faithfulness': 4.666666666666667, 'grounding': 0.5}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | misattributed | question |
|---|---|---|---|---|---|---|---|---|---|---|
| local | local | True | True | 5 | 6 | 0 | 1.2/2 | 0.0 | 0 | How does a Veeam Backup & Replication hardened repository keep backups immutable? |
| local | local | True | True | 5 | 7 | 0 | 2.3/3 | 0.0 | 0 | What does a Veeam SureBackup job verify, and how is it processed? |
| local | local | True | True | 5 | 9 | 0 | 2.2/3 | 0.0 | 0 | How does Veeam Backup & Replication detect malware in backups? |
| local | local | True | True | 5 | 14 | 0 | 1.3/2 | 0.0 | 0 | In Veeam ONE, what happens when an alarm is triggered and how are remediation actions approved? |
| local | local | True | True | 5 | 13 | 0 | 2.6/4 | 0.0 | 0 | How do you halt and resume plan testing in Veeam Recovery Orchestrator? |
| drift | local | True | False | 5 | 6 | 0 | 1.2/2 | 0.0 | 0 | What should I consider when enabling backup immutability for Veeam Agent backups? |
| drift | global | True | False | - | 18 | 0 | 2.9/5 | 0.0 | 0 | Compare how Veeam hardened repositories and AWS Backup Vault Lock protect backups from deletion. |
| global | global | True | True | 5 | 38 | 0 | 2.8/8 | 0.17 | 0 | Compare Veeam backup copy jobs with AWS Backup cross-Region copy for keeping an off-site copy. |
| drift | global | True | True | 5 | 47 | 0 | 3.9/12 | 0.26 | 0 | How do Veeam Backup & Replication and Azure Backup each protect backups against ransomware? |
| local | global | True | False | 5 | 2 | 0 | 2.0/2 | 0.0 | 0 | How does immutability in Veeam Backup for Microsoft 365 compare with soft delete in Azure Backup? |
| global | global | True | False | 5 | 23 | 0 | 3.3/7 | 0.0 | 0 | Which backup vendors support immutable backups, and how do their approaches differ? |
| drift | global | True | True | 5 | 43 | 0 | 3.3/6 | 0.0 | 0 | What approaches do backup vendors take to verify that backups can actually be restored? |

## Run 3 -- 2026-10-10, widened golden set + per-source scope

Same 12 questions, expected-article lists widened (approved), per-source scope deployed
(sha-e997c48: "Veeam Agent" now scopes to the six Agent guides).

| | run 1 (gamma 1) | run 2 (gamma 5) | **run 3** | target |
|---|---|---|---|---|
| routing | 1.00 | 1.00 | **1.00** | >= 0.97 |
| misattributed claims | 0 | 0 | **0** | reported |
| faithfulness | 4.67 | 4.83 | **4.92** | >= 4.8 |
| scoped grounding | 0.40 | 0.50 (0.80 widened) | **1.00 (10/10)** | >= 0.8 |
| cross-vendor grounding (informational) | 0/2 | 0/2 | **1/2** | -- |

Corrected split: the harness counted a question "scoped" only when its scope named a
vendor or product, so the source-scoped Agent question landed under cross-vendor and the
raw output below reads "1.0 (n=9) | 0.67 (n=3)"; with sources counted (fixed in
eval_router.py) it is 10/10 scoped and 1/2 cross-vendor. The one remaining miss is
"Which backup vendors support immutable backups?" -- cross-vendor, with only AWS/Azure
pilot data beside Veeam.

## Run 3 detail

### /answer Router Golden-Set Eval

Questions: 12 (failed: 0)

**Routing accuracy: 1.00** (answer path); classifier: 1.00
by intent: {'local': 1.0, 'drift': 1.0, 'global': 1.0}
Grounding precision: 0.9166666666666666
**Grounding, scoped questions: 1.0 (n=9)** | cross-vendor, informational (golden answers predate the Tier 1 vendors): 0.6666666666666666 (n=3)
by mode: {'local': 1.0, 'global': 0.6666666666666666}
Faithfulness mean: 4.92 (unscored: 0/12)
by mode: {'local': 5.0, 'global': 4.666666666666667}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.6555555555555557, 'max': 5}, 'global': {'mean': 3.3000000000000003, 'max': 8}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.0, 'global': 0.14333333333333334}

Misattributed claims: 0 across 0 answers (unscored: 0)
Comparative (broad): {'local': {'faithfulness': 4.5, 'grounding': 0.6666666666666666}, 'global': {'faithfulness': 4.833333333333333, 'grounding': 0.6666666666666666}, 'drift': {'faithfulness': 4.5, 'grounding': 0.8333333333333334}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | misattributed | question |
|---|---|---|---|---|---|---|---|---|---|---|
| local | local | True | True | 5 | 6 | 0 | 1.2/2 | 0.0 | 0 | How does a Veeam Backup & Replication hardened repository keep backups immutable? |
| local | local | True | True | 5 | 7 | 0 | 1.8/2 | 0.0 | 0 | What does a Veeam SureBackup job verify, and how is it processed? |
| local | local | True | True | 5 | 9 | 0 | 1.8/3 | 0.0 | 0 | How does Veeam Backup & Replication detect malware in backups? |
| local | local | True | True | 5 | 15 | 0 | 1.5/2 | 0.0 | 0 | In Veeam ONE, what happens when an alarm is triggered and how are remediation actions approved? |
| local | local | True | True | 5 | 13 | 0 | 1.6/3 | 0.0 | 0 | How do you halt and resume plan testing in Veeam Recovery Orchestrator? |
| drift | local | True | True | 5 | 7 | 0 | 1.4/2 | 0.0 | 0 | What should I consider when enabling backup immutability for Veeam Agent backups? |
| drift | local | True | True | 5 | 7 | 0 | 1.9/5 | 0.0 | 0 | Compare how Veeam hardened repositories and AWS Backup Vault Lock protect backups from deletion. |
| global | local | True | True | 5 | 14 | 0 | 2.3/4 | 0.0 | 0 | Compare Veeam backup copy jobs with AWS Backup cross-Region copy for keeping an off-site copy. |
| drift | global | True | True | 5 | 34 | 0 | 4.2/8 | 0.24 | 0 | How do Veeam Backup & Replication and Azure Backup each protect backups against ransomware? |
| local | local | True | True | 5 | 14 | 0 | 1.4/3 | 0.0 | 0 | How does immutability in Veeam Backup for Microsoft 365 compare with soft delete in Azure Backup? |
| global | global | True | False | 4 | 10 | 0 | 2.1/5 | 0.0 | 0 | Which backup vendors support immutable backups, and how do their approaches differ? |
| drift | global | True | True | 5 | 43 | 0 | 3.6/8 | 0.19 | 0 | What approaches do backup vendors take to verify that backups can actually be restored? |

---

# /answer Router Golden-Set Eval

Questions: 12 (failed: 0)

**Routing accuracy: 1.00** (answer path); classifier: 1.00
by intent: {'local': 1.0, 'drift': 1.0, 'global': 1.0}
Grounding precision: 0.9166666666666666
**Grounding, scoped questions: 1.0 (n=9)** | cross-vendor, informational (golden answers predate the Tier 1 vendors): 0.6666666666666666 (n=3)
by mode: {'local': 1.0, 'global': 0.6666666666666666}
Faithfulness mean: 4.92 (unscored: 0/12)
by mode: {'local': 5.0, 'global': 4.666666666666667}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.6555555555555557, 'max': 5}, 'global': {'mean': 3.3000000000000003, 'max': 8}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.0, 'global': 0.14333333333333334}

Misattributed claims: 0 across 0 answers (unscored: 0)
Comparative (broad): {'local': {'faithfulness': 4.5, 'grounding': 0.6666666666666666}, 'global': {'faithfulness': 4.833333333333333, 'grounding': 0.6666666666666666}, 'drift': {'faithfulness': 4.5, 'grounding': 0.8333333333333334}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | misattributed | question |
|---|---|---|---|---|---|---|---|---|---|---|
| local | local | True | True | 5 | 6 | 0 | 1.2/2 | 0.0 | 0 | How does a Veeam Backup & Replication hardened repository keep backups immutable? |
| local | local | True | True | 5 | 7 | 0 | 1.8/2 | 0.0 | 0 | What does a Veeam SureBackup job verify, and how is it processed? |
| local | local | True | True | 5 | 9 | 0 | 1.8/3 | 0.0 | 0 | How does Veeam Backup & Replication detect malware in backups? |
| local | local | True | True | 5 | 15 | 0 | 1.5/2 | 0.0 | 0 | In Veeam ONE, what happens when an alarm is triggered and how are remediation actions approved? |
| local | local | True | True | 5 | 13 | 0 | 1.6/3 | 0.0 | 0 | How do you halt and resume plan testing in Veeam Recovery Orchestrator? |
| drift | local | True | True | 5 | 7 | 0 | 1.4/2 | 0.0 | 0 | What should I consider when enabling backup immutability for Veeam Agent backups? |
| drift | local | True | True | 5 | 7 | 0 | 1.9/5 | 0.0 | 0 | Compare how Veeam hardened repositories and AWS Backup Vault Lock protect backups from deletion. |
| global | local | True | True | 5 | 14 | 0 | 2.3/4 | 0.0 | 0 | Compare Veeam backup copy jobs with AWS Backup cross-Region copy for keeping an off-site copy. |
| drift | global | True | True | 5 | 34 | 0 | 4.2/8 | 0.24 | 0 | How do Veeam Backup & Replication and Azure Backup each protect backups against ransomware? |
| local | local | True | True | 5 | 14 | 0 | 1.4/3 | 0.0 | 0 | How does immutability in Veeam Backup for Microsoft 365 compare with soft delete in Azure Backup? |
| global | global | True | False | 4 | 10 | 0 | 2.1/5 | 0.0 | 0 | Which backup vendors support immutable backups, and how do their approaches differ? |
| drift | global | True | True | 5 | 43 | 0 | 3.6/8 | 0.19 | 0 | What approaches do backup vendors take to verify that backups can actually be restored? |
