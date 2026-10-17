# /answer Router Golden-Set Eval

Questions: 21 (failed: 0)

**Routing accuracy: 1.00** (answer path); classifier: 1.00
by intent: {'local': 1.0, 'drift': 1.0, 'global': 1.0}
Grounding precision: 0.8
**Grounding, scoped questions: 0.8333333333333334 (n=12)** | cross-vendor, informational (golden answers predate the Tier 1 vendors): 0.6666666666666666 (n=3)
by mode: {'local': 0.6666666666666666, 'global': 0.8888888888888888}
Faithfulness mean: 4.62 (unscored: 0/21)
by mode: {'local': 4.416666666666667, 'global': 4.888888888888889}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.6583333333333332, 'max': 16}, 'global': {'mean': 2.2777777777777777, 'max': 7}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.014166666666666668, 'global': 0.0}

Misattributed claims: 11 across 4 answers (unscored: 1)
**Facet coverage (comparisons): 1.0 (n=6)** | **Enumeration completeness: 0.67 (n=3)** | **Item coverage (list questions): 0.98 (n=5)** | Items flagged / dropped (list answers): 52 / 7 (n=6)
Comparative (broad): {'local': {'faithfulness': 5.0, 'grounding': 0.7777777777777778}, 'global': {'faithfulness': 4.6, 'grounding': 0.8888888888888888}, 'drift': {'faithfulness': 4.9, 'grounding': 0.7777777777777778}}
drift_wins: False

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | misattributed | facet_coverage | enumeration_complete | item_coverage | flagged/dropped | question |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| local | local | True | False | 5 | 4 | 0 | 1.3/2 | 0.0 | 0 | - | - | - | - | How does a Veeam Backup & Replication hardened repository keep backups immutable? |
| local | local | True | False | 5 | 7 | 0 | 1.4/3 | 0.0 | 0 | - | - | - | - | What does a Veeam SureBackup job verify, and how is it processed? |
| local | local | True | True | 5 | 14 | 0 | 1.6/2 | 0.0 | 0 | - | - | - | - | How does Veeam Backup & Replication detect malware in backups? |
| local | local | True | True | 5 | 14 | 0 | 1.7/3 | 0.0 | 0 | - | - | - | - | In Veeam ONE, what happens when an alarm is triggered and how are remediation actions approved? |
| local | local | True | True | 5 | 15 | 0 | 1.7/3 | 0.0 | 0 | - | - | - | - | How do you halt and resume plan testing in Veeam Recovery Orchestrator? |
| drift | local | True | True | 5 | 15 | 0 | 1.9/7 | 0.0 | 0 | - | - | - | - | What should I consider when enabling backup immutability for Veeam Agent backups? |
| drift | global | True | True | 5 | 19 | 0 | 2.2/3 | 0.0 | 0 | True | - | - | - | Compare how Veeam hardened repositories and AWS Backup Vault Lock protect backups from deletion. |
| global | global | True | True | 5 | 25 | 0 | 2.5/4 | 0.0 | 2 | True | - | - | - | Compare Veeam backup copy jobs with AWS Backup cross-Region copy for keeping an off-site copy. |
| drift | global | True | True | 5 | 50 | 0 | 2.8/5 | 0.0 | 0 | True | - | - | - | How do Veeam Backup & Replication and Azure Backup each protect backups against ransomware? |
| local | global | True | True | 5 | 18 | 0 | 1.8/4 | 0.0 | 0 | True | - | - | - | How does immutability in Veeam Backup for Microsoft 365 compare with soft delete in Azure Backup? |
| global | global | True | False | 5 | 29 | 0 | 2.3/4 | 0.0 | 0 | - | True | - | - | Which backup vendors support immutable backups, and how do their approaches differ? |
| drift | global | True | True | 5 | 33 | 0 | 2.1/4 | 0.0 | - | - | False | - | - | What approaches do backup vendors take to verify that backups can actually be restored? |
| global | global | True | True | 5 | 33 | 0 | 2.2/5 | 0.0 | 0 | - | True | - | - | Which backup vendors can keep a copy of backups in another region or at another site? |
| global | global | True | True | 5 | 22 | 0 | 1.8/3 | 0.0 | 0 | True | - | - | - | Compare AWS Backup Vault Lock with Cohesity DataLock. |
| global | global | True | True | 4 | 40 | 0 | 2.8/7 | 0.0 | 0 | True | - | - | - | How does Cohesity FortKnox compare with Veeam hardened repositories for ransomware protection? |
| global | local | True | None | 3 | 226 | 0 | 2.1/16 | 0.17 | 6 | - | - | - | 1/0 | Which vendors does Veeam integrate with? |
| local | local | True | None | 4 | 137 | 0 | 1.0/2 | 0.0 | 0 | - | - | 1.0 | 4/2 | Which workloads does Veeam support immutability for? |
| local | local | True | None | 4 | 97 | 0 | 1.7/4 | 0.0 | 0 | - | - | 0.88 | 11/0 | Where can I restore a Windows agent backup? |
| local | local | True | None | 5 | 106 | 0 | 2.0/5 | 0.0 | 2 | - | - | 1.0 | 0/0 | Which hypervisors does Veeam Backup & Replication support? |
| local | local | True | None | 5 | 113 | 0 | 1.9/6 | 0.0 | 0 | - | - | 1.0 | 0/0 | Which resource types can AWS Backup protect? |
| local | local | True | None | 2 | 237 | 0 | 1.6/7 | 0.0 | 1 | - | - | 1.0 | 36/5 | What recovery options does Azure Backup offer for Azure VMs? |
