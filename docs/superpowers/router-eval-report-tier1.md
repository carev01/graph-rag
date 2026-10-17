# /answer Router Golden-Set Eval

Questions: 21 (failed: 0)

**Routing accuracy: 1.00** (answer path); classifier: 1.00
by intent: {'local': 1.0, 'drift': 1.0, 'global': 1.0}
Grounding precision: 0.8666666666666667
**Grounding, scoped questions: 0.9166666666666666 (n=12)** | cross-vendor, informational (golden answers predate the Tier 1 vendors): 0.6666666666666666 (n=3)
by mode: {'local': 0.8333333333333334, 'global': 0.8888888888888888}
Faithfulness mean: 4.78 (unscored: 3/21)
by mode: {'local': 4.666666666666667, 'global': 4.888888888888889}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.8083333333333333, 'max': 15}, 'global': {'mean': 2.2555555555555555, 'max': 6}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.06833333333333334, 'global': 0.0}

Misattributed claims: 0 across 0 answers (unscored: 4)
**Facet coverage (comparisons): 1.0 (n=6)** | **Enumeration completeness: 0.67 (n=3)** | **Item coverage (list questions): 0.94 (n=5)** | Items flagged / dropped (list answers): 67 / 5 (n=6)
Comparative (broad): {'local': {'faithfulness': 4.9, 'grounding': 0.7777777777777778}, 'global': {'faithfulness': 4.8, 'grounding': 0.8888888888888888}, 'drift': {'faithfulness': 5.0, 'grounding': 0.7777777777777778}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | misattributed | facet_coverage | enumeration_complete | item_coverage | flagged/dropped | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| local | local | True | True | 5 | 7 | 0 | 1.8/2 | 0.0 | 0 | - | - | - | - | How does a Veeam Backup & Replication hardened repository keep backups immutable? |
| local | local | True | False | 5 | 9 | 0 | 1.5/3 | 0.0 | 0 | - | - | - | - | What does a Veeam SureBackup job verify, and how is it processed? |
| local | local | True | True | 5 | 15 | 0 | 1.5/2 | 0.0 | 0 | - | - | - | - | How does Veeam Backup & Replication detect malware in backups? |
| local | local | True | True | 5 | 14 | 0 | 1.6/3 | 0.0 | 0 | - | - | - | - | In Veeam ONE, what happens when an alarm is triggered and how are remediation actions approved? |
| local | local | True | True | 5 | 14 | 0 | 1.4/2 | 0.0 | 0 | - | - | - | - | How do you halt and resume plan testing in Veeam Recovery Orchestrator? |
| drift | local | True | True | 5 | 15 | 0 | 1.9/8 | 0.53 | 0 | - | - | - | - | What should I consider when enabling backup immutability for Veeam Agent backups? |
| drift | global | True | True | 5 | 31 | 0 | 2.6/5 | 0.0 | 0 | True | - | - | - | Compare how Veeam hardened repositories and AWS Backup Vault Lock protect backups from deletion. |
| global | global | True | True | 4 | 23 | 0 | 2.0/4 | 0.0 | 0 | True | - | - | - | Compare Veeam backup copy jobs with AWS Backup cross-Region copy for keeping an off-site copy. |
| drift | global | True | True | 5 | 44 | 0 | 3.5/6 | 0.0 | 0 | True | - | - | - | How do Veeam Backup & Replication and Azure Backup each protect backups against ransomware? |
| local | global | True | True | 5 | 17 | 0 | 1.7/3 | 0.0 | 0 | True | - | - | - | How does immutability in Veeam Backup for Microsoft 365 compare with soft delete in Azure Backup? |
| global | global | True | False | 5 | 26 | 0 | 2.0/3 | 0.0 | 0 | - | True | - | - | Which backup vendors support immutable backups, and how do their approaches differ? |
| drift | global | True | True | 5 | 38 | 0 | 2.4/6 | 0.0 | 0 | - | False | - | - | What approaches do backup vendors take to verify that backups can actually be restored? |
| global | global | True | True | 5 | 26 | 1 | 2.1/5 | 0.0 | 0 | - | True | - | - | Which backup vendors can keep a copy of backups in another region or at another site? |
| global | global | True | True | 5 | 17 | 0 | 1.7/2 | 0.0 | 0 | True | - | - | - | Compare AWS Backup Vault Lock with Cohesity DataLock. |
| global | global | True | True | 5 | 19 | 0 | 2.3/4 | 0.0 | 0 | True | - | - | - | How does Cohesity FortKnox compare with Veeam hardened repositories for ransomware protection? |
| global | local | True | None | - | 254 | 0 | 1.4/6 | 0.0 | - | - | - | - | 0/2 | Which vendors does Veeam integrate with? |
| local | local | True | None | - | 190 | 0 | 2.0/15 | 0.14 | - | - | - | 1.0 | 0/0 | Which workloads does Veeam support immutability for? |
| local | local | True | None | - | 110 | 0 | 3.2/8 | 0.07 | - | - | - | 0.82 | 9/0 | Where can I restore a Windows agent backup? |
| local | local | True | None | 5 | 47 | 0 | 1.4/4 | 0.0 | 0 | - | - | 0.86 | 0/0 | Which hypervisors does Veeam Backup & Replication support? |
| local | local | True | None | 4 | 58 | 0 | 2.2/7 | 0.0 | 0 | - | - | 1.0 | 11/0 | Which resource types can AWS Backup protect? |
| local | local | True | None | 3 | 244 | 0 | 1.8/11 | 0.08 | - | - | - | 1.0 | 47/3 | What recovery options does Azure Backup offer for Azure VMs? |
