# /answer Router Golden-Set Eval

Questions: 21 (failed: 0)

**Routing accuracy: 1.00** (answer path); classifier: 1.00
by intent: {'local': 1.0, 'drift': 1.0, 'global': 1.0}
Grounding precision: 0.7333333333333333
**Grounding, scoped questions: 0.8333333333333334 (n=12)** | cross-vendor, informational (golden answers predate the Tier 1 vendors): 0.3333333333333333 (n=3)
by mode: {'local': 0.6666666666666666, 'global': 0.7777777777777778}
Faithfulness mean: 4.76 (unscored: 0/21)
by mode: {'local': 4.583333333333333, 'global': 5.0}
Markers per sentence by mode (mean/max): {'local': {'mean': 2.0, 'max': 28}, 'global': {'mean': 2.266666666666667, 'max': 8}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.07916666666666666, 'global': 0.02}

Misattributed claims: 24 across 3 answers (unscored: 1)
**Facet coverage (comparisons): 1.0 (n=6)** | **Enumeration completeness: 1.0 (n=3)** | **Item coverage (list questions): 0.99 (n=5)** | Items flagged / dropped (list answers): 79 / 7 (n=6)
Comparative (broad): {'local': {'faithfulness': 4.888888888888889, 'grounding': 0.7777777777777778}, 'global': {'faithfulness': 5.0, 'grounding': 0.7777777777777778}, 'drift': {'faithfulness': 4.8, 'grounding': 0.5555555555555556}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | misattributed | facet_coverage | enumeration_complete | item_coverage | flagged/dropped | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| local | local | True | False | 5 | 5 | 0 | 1.7/2 | 0.0 | 0 | - | - | - | - | How does a Veeam Backup & Replication hardened repository keep backups immutable? |
| local | local | True | False | 5 | 9 | 0 | 1.3/2 | 0.0 | 0 | - | - | - | - | What does a Veeam SureBackup job verify, and how is it processed? |
| local | local | True | True | 5 | 15 | 0 | 1.7/3 | 0.0 | 0 | - | - | - | - | How does Veeam Backup & Replication detect malware in backups? |
| local | local | True | True | 4 | 14 | 0 | 2.0/3 | 0.0 | 0 | - | - | - | - | In Veeam ONE, what happens when an alarm is triggered and how are remediation actions approved? |
| local | local | True | True | 5 | 15 | 0 | 1.5/3 | 0.0 | 0 | - | - | - | - | How do you halt and resume plan testing in Veeam Recovery Orchestrator? |
| drift | local | True | True | 5 | 15 | 0 | 1.9/7 | 0.0 | 0 | - | - | - | - | What should I consider when enabling backup immutability for Veeam Agent backups? |
| drift | global | True | True | 5 | 23 | 0 | 2.2/3 | 0.0 | 0 | True | - | - | - | Compare how Veeam hardened repositories and AWS Backup Vault Lock protect backups from deletion. |
| global | global | True | True | 5 | 19 | 0 | 2.1/4 | 0.0 | 1 | True | - | - | - | Compare Veeam backup copy jobs with AWS Backup cross-Region copy for keeping an off-site copy. |
| drift | global | True | True | 5 | 55 | 0 | 2.7/7 | 0.0 | 0 | True | - | - | - | How do Veeam Backup & Replication and Azure Backup each protect backups against ransomware? |
| local | global | True | True | 5 | 17 | 0 | 2.0/4 | 0.0 | 0 | True | - | - | - | How does immutability in Veeam Backup for Microsoft 365 compare with soft delete in Azure Backup? |
| global | global | True | False | 5 | 31 | 0 | 2.2/4 | 0.0 | 0 | - | True | - | - | Which backup vendors support immutable backups, and how do their approaches differ? |
| drift | global | True | False | 5 | 12 | 0 | 2.0/3 | 0.0 | 0 | - | True | - | - | What approaches do backup vendors take to verify that backups can actually be restored? |
| global | global | True | True | 5 | 18 | 0 | 1.6/3 | 0.0 | 0 | - | True | - | - | Which backup vendors can keep a copy of backups in another region or at another site? |
| global | global | True | True | 5 | 20 | 0 | 1.8/2 | 0.0 | 0 | True | - | - | - | Compare AWS Backup Vault Lock with Cohesity DataLock. |
| global | global | True | True | 5 | 38 | 0 | 3.8/8 | 0.18 | 0 | True | - | - | - | How does Cohesity FortKnox compare with Veeam hardened repositories for ransomware protection? |
| global | local | True | None | 4 | 233 | 0 | 2.7/28 | 0.41 | 0 | - | - | - | 1/2 | Which vendors does Veeam integrate with? |
| local | local | True | None | 5 | 197 | 0 | 2.3/12 | 0.17 | - | - | - | 1.0 | 0/1 | Which workloads does Veeam support immutability for? |
| local | local | True | None | 5 | 219 | 0 | 1.7/9 | 0.17 | 22 | - | - | 0.94 | 66/4 | Where can I restore a Windows agent backup? |
| local | local | True | None | 3 | 93 | 0 | 2.8/7 | 0.0 | 0 | - | - | 1.0 | 0/0 | Which hypervisors does Veeam Backup & Replication support? |
| local | local | True | None | 4 | 96 | 0 | 2.2/6 | 0.0 | 0 | - | - | 1.0 | 0/0 | Which resource types can AWS Backup protect? |
| local | local | True | None | 5 | 199 | 0 | 2.2/13 | 0.2 | 1 | - | - | 1.0 | 12/0 | What recovery options does Azure Backup offer for Azure VMs? |
