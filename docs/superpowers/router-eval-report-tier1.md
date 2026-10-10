# Tier 1 (Veeam) golden-set eval

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
