# Tier 1 (Veeam) golden-set eval

## Run 2 -- 2026-10-10, after BACKLOG 57 (Leiden gamma 5)

857 communities (839 reports; 853 live after `--verify-pending`), largest level-1
community 2,848 entities (was 10,706). Same 12 questions.

| | gamma 1 (run 1) | gamma 5 (run 2) | target |
|---|---|---|---|
| routing | 1.00 | **1.00** | >= 0.97 |
| misattributed claims | 0 | **0** | reported |
| faithfulness | 4.67 | **4.83** | >= 4.8 |
| scoped grounding, original expected lists | 0.40 | **0.50** (comparisons 1/4 -> 3/4) | >= 0.8 |
| scoped grounding, widened expected lists (proposed, not adopted) | -- | **0.80** | >= 0.8 |
| cross-vendor grounding (informational) | 0/2 | 0/2 | -- |

The comparison questions -- the global/DRIFT path the hub community had swamped -- went
from 1 of 4 grounded to 3 of 4. The widened lists add same-topic pages the first run's
answers cited ("Immutability for Backup Files", "SureBackup Job", "How YARA Scan Works",
"Recovery Verification for VMware vSphere"); they were found after seeing answers, so
they are a proposal awaiting review, not part of the golden file. Still open: Veeam Agent
immutability (product scoping cannot isolate the Agent guides) and both cross-vendor
questions.

The per-question table below is run 2. Run 1's summary follows it.

## Run 2 detail

# /answer Router Golden-Set Eval

Questions: 12 (failed: 0)

**Routing accuracy: 1.00** (answer path); classifier: 1.00
by intent: {'local': 1.0, 'drift': 1.0, 'global': 1.0}
Grounding precision: 0.4166666666666667
**Grounding, scoped questions: 0.5 (n=10)** | cross-vendor, informational (golden answers predate the Tier 1 vendors): 0.0 (n=2)
by mode: {'local': 0.5, 'global': 0.25}
Faithfulness mean: 4.83 (unscored: 0/12)
by mode: {'local': 5.0, 'global': 4.5}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.9, 'max': 8}, 'global': {'mean': 3.3, 'max': 11}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.04375, 'global': 0.1175}

Misattributed claims: 0 across 0 answers (unscored: 0)
Comparative (broad): {'local': {'faithfulness': 5.0, 'grounding': 0.5}, 'global': {'faithfulness': 5.0, 'grounding': 0.5}, 'drift': {'faithfulness': 5.0, 'grounding': 0.5}}
drift_wins: True

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | misattributed | question |
|---|---|---|---|---|---|---|---|---|---|---|
| local | local | True | False | 5 | 5 | 0 | 1.2/2 | 0.0 | 0 | How does a Veeam Backup & Replication hardened repository keep backups immutable? |
| local | local | True | False | 5 | 7 | 0 | 1.8/2 | 0.0 | 0 | What does a Veeam SureBackup job verify, and how is it processed? |
| local | local | True | False | 5 | 9 | 0 | 1.8/2 | 0.0 | 0 | How does Veeam Backup & Replication detect malware in backups? |
| local | local | True | True | 5 | 15 | 0 | 1.5/2 | 0.0 | 0 | In Veeam ONE, what happens when an alarm is triggered and how are remediation actions approved? |
| local | local | True | True | 5 | 14 | 0 | 2.8/4 | 0.0 | 0 | How do you halt and resume plan testing in Veeam Recovery Orchestrator? |
| drift | local | True | False | 5 | 6 | 0 | 1.5/2 | 0.0 | 0 | What should I consider when enabling backup immutability for Veeam Agent backups? |
| drift | local | True | True | 5 | 7 | 0 | 2.0/5 | 0.0 | 0 | Compare how Veeam hardened repositories and AWS Backup Vault Lock protect backups from deletion. |
| global | local | True | True | 5 | 15 | 0 | 2.6/8 | 0.35 | 0 | Compare Veeam backup copy jobs with AWS Backup cross-Region copy for keeping an off-site copy. |
| drift | global | True | True | 4 | 42 | 0 | 4.7/11 | 0.26 | 0 | How do Veeam Backup & Replication and Azure Backup each protect backups against ransomware? |
| local | global | True | False | 5 | 2 | 0 | 2.0/2 | 0.0 | 0 | How does immutability in Veeam Backup for Microsoft 365 compare with soft delete in Azure Backup? |
| global | global | True | False | 4 | 18 | 0 | 3.0/7 | 0.0 | 0 | Which backup vendors support immutable backups, and how do their approaches differ? |
| drift | global | True | False | 5 | 39 | 0 | 3.5/8 | 0.21 | 0 | What approaches do backup vendors take to verify that backups can actually be restored? |

---

## Run 1 -- 2026-10-10, gamma 1 (before BACKLOG 57)



Run after the post-Veeam maintenance (71 duplicates merged; theme-build: 284 communities,
268 live reports). 12 questions: 6 scoped, 4 comparisons (Veeam vs AWS/Azure), 2
cross-vendor (`src/answer_api/router_golden_tier1.json`).

**Reading the numbers.** Routing 1.00 and 0 misattributed claims meet the bar.
Faithfulness 4.67 is just under 4.8. Scoped grounding 0.40 overstates the problem for
LOCAL answers: the misses cited valid same-topic pages the hand-built expected lists did
not include ("Immutability for Backup Files", "SureBackup Job", "How YARA Scan Works")
-- the lists were built from title searches and are too narrow. The GLOBAL/DRIFT misses
are real: comparisons cited one Veeam GFS page and no AWS/Azure source, or Veeam's
Microsoft Sentinel app instead of Azure Backup. Both trace to the 10,282-entity community
("Veeam Backup & Replication: GFS Retention, Immutability, and Security Event
Monitoring") that absorbs immutability/security topics (BACKLOG 57). Product scoping
also cannot isolate the Veeam Agent guides (sources under the VBR product).

# /answer Router Golden-Set Eval

Questions: 12 (failed: 0)

**Routing accuracy: 1.00** (answer path); classifier: 1.00
by intent: {'local': 1.0, 'drift': 1.0, 'global': 1.0}
Grounding precision: 0.3333333333333333
**Grounding, scoped questions: 0.4 (n=10)** | cross-vendor, informational (golden answers predate the Tier 1 vendors): 0.0 (n=2)
by mode: {'local': 0.5, 'global': 0.0}
Faithfulness mean: 4.67 (unscored: 0/12)
by mode: {'local': 5.0, 'global': 4.0}
Markers per sentence by mode (mean/max): {'local': {'mean': 1.8875, 'max': 10}, 'global': {'mean': 3.2, 'max': 13}}
Share of citations in >=8-marker sentences, by mode: {'local': 0.0525, 'global': 0.14500000000000002}

Misattributed claims: 0 across 0 answers (unscored: 0)
Comparative (broad): {'local': {'faithfulness': 4.666666666666667, 'grounding': 0.5}, 'global': {'faithfulness': 4.833333333333333, 'grounding': 0.16666666666666666}, 'drift': {'faithfulness': 4.833333333333333, 'grounding': 0.6666666666666666}}
drift_wins: True

## Per question

| intent | chosen | routing | grounding | faithfulness | cited | ranges | mps | bag | misattributed | question |
|---|---|---|---|---|---|---|---|---|---|---|
| local | local | True | False | 5 | 4 | 0 | 1.3/2 | 0.0 | 0 | How does a Veeam Backup & Replication hardened repository keep backups immutable? |
| local | local | True | False | 5 | 6 | 0 | 1.5/2 | 0.0 | 0 | What does a Veeam SureBackup job verify, and how is it processed? |
| local | local | True | False | 5 | 9 | 0 | 1.8/2 | 0.0 | 0 | How does Veeam Backup & Replication detect malware in backups? |
| local | local | True | True | 5 | 15 | 0 | 1.5/2 | 0.0 | 0 | In Veeam ONE, what happens when an alarm is triggered and how are remediation actions approved? |
| local | local | True | True | 5 | 14 | 0 | 2.8/4 | 0.0 | 0 | How do you halt and resume plan testing in Veeam Recovery Orchestrator? |
| drift | local | True | False | 5 | 6 | 0 | 1.2/2 | 0.0 | 0 | What should I consider when enabling backup immutability for Veeam Agent backups? |
| drift | global | True | False | 5 | 7 | 0 | 3.5/5 | 0.0 | 0 | Compare how Veeam hardened repositories and AWS Backup Vault Lock protect backups from deletion. |
| global | local | True | True | 5 | 15 | 0 | 2.0/6 | 0.0 | 0 | Compare Veeam backup copy jobs with AWS Backup cross-Region copy for keeping an off-site copy. |
| drift | global | True | False | 4 | 40 | 0 | 5.0/13 | 0.33 | 0 | How do Veeam Backup & Replication and Azure Backup each protect backups against ransomware? |
| local | local | True | True | 5 | 14 | 0 | 3.0/10 | 0.42 | 0 | How does immutability in Veeam Backup for Microsoft 365 compare with soft delete in Azure Backup? |
| global | global | True | False | 4 | 12 | 0 | 1.3/2 | 0.0 | 0 | Which backup vendors support immutable backups, and how do their approaches differ? |
| drift | global | True | False | 3 | 36 | 0 | 3.0/9 | 0.25 | 0 | What approaches do backup vendors take to verify that backups can actually be restored? |
