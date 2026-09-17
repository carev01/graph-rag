# Sample records — readable digest
Generated from `docs/proposals/samples/sample.jsonl` (20 records). This file is a
REVIEW AID: long bodies are truncated. The JSONL is the actual deliverable
and is verbatim.

**6 of these 20 records have a REPAIRED target** (`meta.repaired`).
That is deliberate over-representation: repaired records are ~3% of the build, so a
uniform draw of 20 would show none, and the point of the sample is that the repair
policy can be inspected per example. **Do not read a repair rate off this file** —
the build report is where the rates live.

Rebuild both with:

```
uv run --extra dev python scripts/build_extraction_dataset.py \
    --from-capture /tmp/claude-1000/-home-openclaw-Documents-Shared-projects-graph-rag/964252d0-58b1-4131-93c7-53185320cfde/scratchpad/llm-capture-20260917.jsonl \
    --out docs/proposals/samples --limit 20 --sample-only --mix
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.2551` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `d0098594-34ad-46bc-a581-7ec06a7117ac` chunk `2` episode `c31ea5cb-fb49-496a-a44f-0774d1b9377a`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* **TARGET REPAIRED** (1 item(s)):
    * `entity_name_not_in_list` on edges[6] 'Backup center' -IntegratesWith-> 'Azure Business Continuity Center' — omitted from edges
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: TARGET CORRECTED: this completion failed validate_capture and was repaired per item to the DERIVABLE correct answer -- see meta.repaired for the defect class and the specific item changed. The prompt is untouched and is still byte-exact wire traffic; the target is no longer purely what the model emitted. It was re-validated after repair with zero residual defects.

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Delete › Delete]\nYou're prompted to enter a security personal identification number (PIN), which you must generate manually:\n\n\t1. Sign in to the Azure portal.\n\t2. Go to **Recovery Services vault** \\> **Settings** \\> **Properties**.\n\t3. Under **Security PIN**, select **Generate**. Copy this PIN. The PIN is valid for only five minutes.\n4. In the management console, paste the PIN, and then select **OK**.\n\n![Screenshot that shows generating a security PIN.](/media/3e86238a-0b5a-4f2e-b384-c73ce87bbaf5/de4e51d88c51.png)\n\n> **Figure:** Screenshot of the 'Stop a Scheduled Backup' dialog in the Schedule Backup Wizard, showing a prompt to enter a Security PIN to complete the operation. The PIN field is filled with masked characters, and the 'Finish' button is highlighted.\n5. On the **Modify Backup Progress** page, the following message appears: **Deleted backup data will be retained for 14 days. After that time, backup data will be permanently deleted.**\n\n![Screenshot that shows deleting the backup infrastructure.](/media/3e86238a-0b5a-4f2e-b384-c73ce87bbaf5/102056509c08.png)\n\n> **Figure:** Screenshot of the 'Modify Backup Progress' screen in the Schedule Backup Wizard, showing the successful stop of the backup schedule for Files and Folders. The status indicates deleted backup data will be retained for 14 days before permanent deletion.\n\nAfter you delete the on\\-premises backup items, follow the next steps from the portal.\n\n### Delete backup items from the MABS or DPM management console\n\nIf you deleted or lost the source machine without stopping the backup, the next scheduled backup fails. The old recovery point expires according to the policy, but the last single recovery point is always retained until you stop the 
  [...32724 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "Backup center", "target_entity_name": "Azure Backup", "relation_type": "Operates", "fact": "Backup center governs the Azure Backup estate and is used to manage backup compliance and protectable datasources.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Policies for Backup", "target_entity_name": "Azure Backup", "relation_type": "Provides", "fact": "Azure Policies for Backup provides Azure Policy definitions for backup that can be assigned to subscriptions and resource groups.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Policy definitions for backup", "target_entity_name": "Azure Backup", "relation_type": "Provides", "fact": "Azure Policy definitions for backup are available for assignment to subscriptions and resource groups to govern backup compliance.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Backup Compliance", "target_entity_name": "Azure Backup", "relation_type": "Provides", "fact": "Backup Compliance helps view the compliance of resources according to the various built-in policies assigned to the Azure environment.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Protectable Datasources", "target_entity_name": "Azure Backup", "relation_type": "Provides", "fact": "Protectable Datasources allows viewing all datasources that haven't been configured for backup and lets you configure backup for a resource by right-clicking the grid item and selecting Backup.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Virtual Machines", "target_entity_name": "Azure Backup", "relation_type": "Supports", "fact": "Azure 
  [...212 more chars...]
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.2551` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `d4d25628-1f6e-42ea-a5c5-e66b2e1bf2f8` chunk `9` episode `8b88c448-b02f-4822-ba7c-d12660446a81`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* **TARGET REPAIRED** (6 item(s)):
    * `episode_index_out_of_range` on edges[0] 'AWS Backup' -Supports-> 'Amazon RDS' — episode_indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] -> [0]
    * `episode_index_out_of_range` on edges[1] 'AWS Backup' -Supports-> 'Aurora' — episode_indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] -> [0]
    * `episode_index_out_of_range` on edges[2] 'AWS Backup' -Limits-> 'Amazon RDS' — episode_indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] -> [0]
    * `episode_index_out_of_range` on edges[3] 'AWS Backup CLI' -Operates-> 'AWS Backup' — episode_indices [24] -> [0]
    * `episode_index_out_of_range` on edges[4] 'AWS Backup' -Provides-> 'Amazon RDS' — episode_indices [24] -> [0]
    * `episode_index_out_of_range` on edges[5] 'Amazon RDS' -AvailableIn-> 'Amazon RDS User Guide' — episode_indices [24] -> [0]
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: TARGET CORRECTED: this completion failed validate_capture and was repaired per item to the DERIVABLE correct answer -- see meta.repaired for the defect class and the specific item changed. The prompt is untouched and is still byte-exact wire traffic; the target is no longer purely what the model emitted. It was re-validated after repair with zero residual defects.

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Assign resources with AWS CLI › Assign resources with AWS CLI]\n\n###### Example: Select all FSx file systems, the Aurora cluster \"my\\-aurora\\-cluster\", and all resources tagged with \"backup\":\"true\", except for resources tagged with \"stage\":\"test\"\n\n```\n{\n  \"BackupSelection\":{\n    \"Resources\":[\n      \"arn:aws:fsx:*\",\n      \"arn:aws:rds:*:*:cluster:my-aurora-cluster\"\n    ],\n    \"ListOfTags\":[\n      {\n        \"ConditionType\":\"StringEquals\",\n        \"ConditionKey\":\"backup\",\n        \"ConditionValue\":\"true\"\n      }\n    ],\n    \"Conditions\":{\n      \"StringNotEquals\":[\n        {\n          \"ConditionKey\":\"aws:ResourceTag/stage\",\n          \"ConditionValue\":\"test\"\n        }\n      ]\n    }\n  }\n}\n```\n\n\n###### Example: Select all resources tagged with tag \"backup\":\"true\" except for EBS volumes tagged with \"stage\":\"test\"\n\nUse two CLI commands to create two selections to select this group of resources. The\n first selection applies to all resources except for EBS volumes. The second selection\n applies to EBS volumes.\n\n\n```\n{\n  \"BackupSelection\":{\n    \"Resources\":[\n      \"*\"\n    ],\n    \"NotResources\":[\n      \"arn:aws:ec2:*:*:volume/*\"\n    ],\n    \"Conditions\":{\n      \"StringEquals\":[\n        {\n          \"ConditionKey\":\"aws:ResourceTag/backup\",\n          \"ConditionValue\":\"true\"\n        }\n      ]\n    }\n  }\n}\n```\n\n\n```\n{\n  \"BackupSelection\":{\n    \"Resources\":[\n      \"arn:aws:ec2:*:*:volume/*\"\n    ],\n    \"Conditions\":{\n      \"StringEquals\":[\n        {\n          \"ConditionKey\":\"aws:ResourceTag/backup\",\n          \"ConditionValue\":\"true\"\n        }\n      ],\n      \"StringNotEquals\":[\n        {\n    
  [...27440 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "AWS Backup", "target_entity_name": "Amazon RDS", "relation_type": "Supports", "fact": "AWS Backup supports continuous backups and point-in-time recovery for Amazon RDS instances and Aurora that are supported by the native Amazon RDS service.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "Aurora", "relation_type": "Supports", "fact": "AWS Backup supports continuous backups and point-in-time recovery for Aurora that is supported by the native Amazon RDS service.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "Amazon RDS", "relation_type": "Limits", "fact": "AWS Backup does not support continuous backups or point-in-time recovery for Amazon RDS Multi-AZ clusters.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup CLI", "target_entity_name": "AWS Backup", "relation_type": "Operates", "fact": "AWS Backup CLI can be used to restore RDS and Aurora databases to a point in time using metadata parameters.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "Amazon RDS", "relation_type": "Provides", "fact": "AWS Backup provides point-in-time restore capability for Amazon RDS databases.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Amazon RDS", "target_entity_name": "Amazon RDS User Guide", "relation_type": "AvailableIn", "fact": "Amazon RDS User Guide contains instructions for restoring a DB Instance to a specified time.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}]}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.2551` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `ea2a35ad-6628-414d-bb83-0fac5a4d6882` chunk `3` episode `963dd5af-fbca-438c-a046-6b60eabad5e8`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* **TARGET REPAIRED** (2 item(s)):
    * `self_edge` on edges[0] 'Azure Backup' -Provides-> 'Azure Backup' — omitted from edges
    * `all_edges_invalid` on <ENTITIES> offered 1 distinct name(s) — edges -> [] (no valid edge is expressible with <2 entities)
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: TARGET CORRECTED: this completion failed validate_capture and was repaired per item to the DERIVABLE correct answer -- see meta.repaired for the defect class and the specific item changed. The prompt is untouched and is still byte-exact wire traffic; the target is no longer purely what the model emitted. It was re-validated after repair with zero residual defects.

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Review security posture › Review security posture]\nTutorial: Review security posture\n=================================\n\nThis tutorial describes how to review and modify the security level for protected items in Resiliency in Azure.\n\nImportant\n\nAzure Business Continuity Center is now Resiliency in Azure, a unified platform for Zone Resiliency, High Availability, Backup \\& Disaster Recovery, and Ransomware Protection. offers integrated experiences to ensure ongoing resilience of your applications. To try out the Zonal feature in Resiliency, fill out [this enrollment form](https://forms.office.com/pages/responsepage.aspx?id=v4j5cvGGr0GRqy180BHbR9NW9IkjD2RCnDQwsmIfABFUNU5MWUVaN1FWWDYxMFY1VTNBM1FPVDM3OC4u&route=shorturl).\n\nAzure Backup provides security features at the vault level to safeguard the backup data stored in it. The vault security measures encompass the settings associated with the Azure Backup solution for the vault and apply to the protected data sources contained within the vault.\n\nResiliency in Azure allows you to view the security level for each protected item from the security posture view.\n\nView security level\n-------------------\n\nTo view the security level for protected items, follow these steps:\n\n1. On **Resiliency**, go to **Security \\+ Threat management** \\> **Security posture**.\n\n![Screenshot shows the security posture selection.](/media/d386db8d-41e7-4d5e-bdcb-d933eb2c7c9d/f31226d6ef15.png)\n\n> **Figure:** Screenshot of the 'Security posture' page in the Resiliency section of a Microsoft Azure service, showing the navigation menu with 'Security posture' highlighted and the main content area displaying security level details for Azure managed Active resources.", "timestamp": "2026-07-14T21:59
  [...19312 more chars...]
```

**assistant (target)**

```json
{"edges": []}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.2551` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `bf5b97e5-55ca-4136-9bbc-1dcf7da6221d` chunk `13` episode `6719affc-ab3f-400c-bac9-245e867af00f`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* **TARGET REPAIRED** (4 item(s)):
    * `episode_index_out_of_range` on edges[0] 'Azure Backup' -Supports-> 'Azure Database for PostgreSQL servers' — episode_indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] -> [0]
    * `episode_index_out_of_range` on edges[1] 'Azure Backup' -AvailableIn-> 'Secondary Region' — episode_indices [18, 19, 20, 21, 22, 23, 24] -> [0]
    * `episode_index_out_of_range` on edges[2] 'Azure Backup' -Provides-> 'cross-region copy' — episode_indices [18, 19, 20, 21, 22, 23, 24] -> [0]
    * `episode_index_out_of_range` on edges[3] 'cross-region copy' -AppliesTo-> 'Azure Database for PostgreSQL servers' — episode_indices [18, 19, 20, 21, 22, 23, 24] -> [0]
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: TARGET CORRECTED: this completion failed validate_capture and was repaired per item to the DERIVABLE correct answer -- see meta.repaired for the defect class and the specific item changed. The prompt is untouched and is still byte-exact wire traffic; the target is no longer purely what the model emitted. It was re-validated after repair with zero residual defects.

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Manage › Manage]\n\nOptimize backup and recovery with Resiliency Copilot\n----------------------------------------------------\n\nResiliency Copilot introduces new capabilities in the Backup vaults pane, enabling you to configure/manage a secure, resilient backup and recovery environment:\n\n* [Increase security level](../resiliency/tutorial-manage-data-using-copilot#increase-security-level-of-recovery-service-vault-and-backup-vault) to strengthen protection for backup data and disaster recovery operations.\n* [Analyze job failures](../resiliency/tutorial-manage-data-using-copilot#analyze-job-failures-for-recovery-service-vault-and-backup-vault) to gain insights into failures for precise analysis and faster resolution.\n* [Configure protection for unprotected resources](../resiliency/tutorial-manage-data-using-copilot#configure-protection-for-resources-in-recovery-services-vault-and-backup-vault).\n* [Delete a vault](../resiliency/tutorial-manage-data-using-copilot#delete-recovery-services-vault-and-backup-vault-using-copilot).\n* [Troubleshoot common errors efficiently](../resiliency/tutorial-manage-data-using-copilot#troubleshoot-error-codes-for-recovery-service-vaults-and-backup-vaults).\n\nLearn more about [managing the Business Continuity and Disaster Recovery estate using Copilot (preview)](../resiliency/tutorial-manage-data-using-copilot).\n\nMove a Backup vault across Azure subscriptions/resource groups\n--------------------------------------------------------------\n\nThis section explains how to move a Backup vault (configured for Azure Backup) across Azure subscriptions and resource groups using the Azure portal.\n\nNote\n\nYou can also move Backup vaults to a different resource group or subscription using [PowerShell](/en-
  [...24325 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "Azure Backup", "target_entity_name": "Azure Database for PostgreSQL servers", "relation_type": "Supports", "fact": "Azure Backup supports Azure Database for PostgreSQL servers.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "Secondary Region", "relation_type": "AvailableIn", "fact": "Azure Backup enables Cross Region Restore to a Secondary Region for Azure Database for PostgreSQL servers.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "cross-region copy", "relation_type": "Provides", "fact": "Azure Backup provides cross-region copy capability through Cross Region Restore for Azure Database for PostgreSQL servers.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "cross-region copy", "target_entity_name": "Azure Database for PostgreSQL servers", "relation_type": "AppliesTo", "fact": "Cross Region Restore (cross-region copy) applies to Azure Database for PostgreSQL servers.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}]}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.2551` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `78e2b9e5-bf0c-4e76-8b83-fc54e76f52cd` chunk `3` episode `8198b3e1-1465-4c8d-872f-0cda562e4cbb`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* **TARGET REPAIRED** (5 item(s)):
    * `episode_index_out_of_range` on edges[0] 'AWS Backup' -Provides-> 'cross-Region backup' — episode_indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] -> [0]
    * `episode_index_out_of_range` on edges[1] 'AWS Backup' -Requires-> 'retention period' — episode_indices [25] -> [0]
    * `episode_index_out_of_range` on edges[2] 'AWS Backup' -Requires-> 'IAM role' — episode_indices [25] -> [0]
    * `episode_index_out_of_range` on edges[3] 'cross-Region backup' -Limits-> 'Transition to cold storage' — episode_indices [23] -> [0]
    * `episode_index_out_of_range` on edges[4] 'AWS Backup' -Limits-> 'cross-Region backup' — episode_indices [22] -> [0]
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: TARGET CORRECTED: this completion failed validate_capture and was repaired per item to the DERIVABLE correct answer -- see meta.repaired for the defect class and the specific item changed. The prompt is untouched and is still byte-exact wire traffic; the target is no longer purely what the model emitted. It was re-validated after repair with zero residual defects.

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Continuous backups and PITR › Continuous backups and PITR]\n\n**Backup coverage during transition:**\n\n* Snapshot backups provide protection while waiting for maintenance window\n* Continuous recovery points become available when the backup job runs after PITR is enabled\n* No gap in backup protection occurs during the transition period\n* Recovery granularity may be limited to snapshot intervals until PITR is fully active\n\nNote: [Stopping the RDS instance](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_StopInstance.html) will remove pending changes. PITR configuration changes will be requeued by the next backup job and applied during a subsequent maintenance window.\n\n**Copies of Amazon RDS continuous backups:**\n\n* **Creating copies of Amazon RDS continuous backups**\n — You can't create copies of Amazon RDS continuous backups because AWS Backup for Amazon RDS\n does not allow copying transaction logs. Instead, AWS Backup creates a snapshot and copies\n it with the frequency specified in the backup plan.\n", "timestamp": "2026-08-17T23:53:18.361640+00:00"}, {"content": "[Continuous backups and PITR › Continuous backups and PITR]\n\n**Restores:** You can perform a point\\-in\\-time restore using either\n AWS Backup or Amazon RDS. For AWS Backup console instructions, see [Restoring an Amazon RDS Database](https://docs.aws.amazon.com/aws-backup/latest/devguide/restoring-rds.html).\n For Amazon RDS instructions, see [Restoring a DB Instance to a specified\n time](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_PIT.html) in the *Amazon RDS User Guide*.\n\n###### Tip\n\nA multi AZ (availability zone) database instance set to `Always On`\n should not have a backup retention set to zero. If errors occur, use AW
  [...24280 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "AWS Backup", "target_entity_name": "cross-Region backup", "relation_type": "Provides", "fact": "AWS Backup provides cross-Region backup, allowing backups to be copied to multiple AWS Regions on demand or automatically as part of a scheduled backup plan.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "retention period", "relation_type": "Requires", "fact": "AWS Backup requires the retention period for a cross-Region copy to be greater than 90 days beyond the Transition to cold storage value.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "IAM role", "relation_type": "Requires", "fact": "AWS Backup requires an IAM role to create a cross-Region copy; the role must have AWS Backup listed as a trusted entity. If Default is chosen and the default role is absent, one is created with correct permissions.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "cross-Region backup", "target_entity_name": "Transition to cold storage", "relation_type": "Limits", "fact": "Cross-Region backup copies transitioned to cold storage must be stored there for a minimum of 90 days, and this value cannot be changed after transition.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "cross-Region backup", "relation_type": "Limits", "fact": "AWS Backup does not support cross-Region copies for storage in cold tiers.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}]}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.2551` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `a6481417-ece0-481d-a73e-c4279e5eec5f` chunk `9` episode `6a4c36fa-7dd9-4db9-9ab0-9361b38d3a81`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* **TARGET REPAIRED** (3 item(s)):
    * `entity_name_not_in_list` on edges[7] 'Azure Backup' -Provides-> 'Security Alerts and Reports' — omitted from edges
    * `entity_name_not_in_list` on edges[16] 'Azure Policies' -Provides-> 'Private endpoints' — omitted from edges
    * `entity_name_not_in_list` on edges[17] 'Security Alerts and Reports' -Operates-> 'Azure Monitor' — omitted from edges
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: TARGET CORRECTED: this completion failed validate_capture and was repaired per item to the DERIVABLE correct answer -- see meta.repaired for the defect class and the specific item changed. The prompt is untouched and is still byte-exact wire traffic; the target is no longer purely what the model emitted. It was re-validated after repair with zero residual defects.

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Azure Backup architecture › Azure Backup architecture]\n\nData isolation with Azure Backup\n--------------------------------\n\nWith Azure Backup, the vaulted backup data is stored in Microsoft\\-managed Azure subscription and tenant. External users or guests have no direct access to this backup storage or its contents, ensuring the isolation of backup data from the production environment where the data source resides.\n\nIn Azure, all communications and data in transit is securely transferred with *HTTPS* and *TLS 1\\.2\\+* protocols. This data remains on the Azure backbone network ensuring reliable and efficient data transmission. The backup data at rest is encrypted by default using *Microsoft\\-managed keys*. You can also bring your own keys for encryption if you require greater control over the data. To enhance protection, you can use [immutability](backup-azure-immutable-vault-concept), which prevents data from being altered or deleted before its retention period. Azure Backup gives you diverse options such as [soft delete](backup-azure-enhanced-soft-delete-about), stop backup and delete data or retain data if you need to stop backups at any time. To protect critical operations, you can add [Multi\\-User Authorization (MUA)](multi-user-authorization-concept) that adds additional layer of protection by using an Azure resource called Azure Resource Guard.\n\nThis robust approach ensures that even in a compromised environment, existing backups cannot be tampered with or deleted by unauthorized users.\n\nLearn more about vaulted backup for the following datasources:\n\n* [Azure Files](azure-file-share-backup-overview?tabs=vault-standard#architecture-for-azure-files-backup)\n* [Azure Blob](blob-backup-overview?tabs=vaulted-backup#how
  [...41322 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "Azure Backup", "target_entity_name": "Azure virtual machines", "relation_type": "Supports", "fact": "Azure Backup supports restoring Azure virtual machines by creating new VMs or restoring disks from Azure Backup restore points to any permitted subscription.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "cross-subscription restore", "target_entity_name": "Azure virtual machines", "relation_type": "AppliesTo", "fact": "Cross-subscription restore applies to Azure virtual machines, allowing restoration of VMs or disks from Azure Backup restore points to any permitted subscription.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "cross-subscription restore", "target_entity_name": "Multi-user authorization", "relation_type": "Requires", "fact": "Cross-subscription restore actions should be classified as critical operations under Multi-user authorization (MUA) to ensure security admins can access and restore backup data.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "cross-subscription restore", "target_entity_name": "Recovery Services vault", "relation_type": "AvailableIn", "fact": "Cross-subscription restore can be enabled in a Recovery Services vault.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "cross-subscription restore", "target_entity_name": "Backup vault", "relation_type": "AvailableIn", "fact": "Cross-subscription restore can be enabled in a Backup vault.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "Business continuity and disaster recovery security posture", "relation_type": "Provides",
  [...3543 more chars...]
```

---

## `extract_nodes.extract_summaries_batch`
* provenance: **captured**
* loss_weight: `0.9842` (basis `full`)  |  response_schema: `SummarizedEntities`
* article `49d019c4-ceb8-4a0a-b908-cd35602cab20` chunk `6` episode `177f06c6-8134-4b63-af7b-9e06a03f3d4a`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are a helpful assistant that generates concise entity summaries from provided context.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

Given the MESSAGES and a list of ENTITIES, generate an updated summary for each entity that needs one.
Each summary must be under 1000 characters.

Guidelines:
        1. Output only factual content. Never explain what you're doing, why, or mention limitations or constraints.
        2. Only use the provided messages, entity, and entity context to set attribute values.
        3. Keep the summary information-dense and entity-specific. STATE FACTS DIRECTLY IN UNDER 1000 CHARACTERS.
        4. Preserve all materially relevant names, roles, places, dates, counts, and temporal qualifiers that are explicitly supported.
        5. Prefer compact factual sentences over vague thematic phrasing or meta-language.
        6. When the durable fact is the content of what was said, state the content directly instead of narrating that it was said.
        7. Use communication verbs only when the act of speaking, asking, sharing, presenting, announcing, or telling is itself the important fact.
        8. Never use filler verbs like "mentioned", "described", "stated", "reported", "noted", "discussed", "referenced", or "indicated" unless the communication act itself is the fact.
        9. Include temporal anchors when the messages provide them and they help ground the fact.
        10. Begin with the entity name or a direct fact, not with "A", "An", "The", or "This is" unless that wording is part of the entity name.

        Example summary:
        BAD: "The context shows John ordered pizza. Due to length constraints, other details are omitted from this summary."
        GOOD: "John ordered pepperoni pizza from Mario's at 7:30 PM and had it delivered to the office."
        

<MESSAGES>
[{"content": "[Cross-account backup › Cross-account backup]\nCreating backup copies across AWS acco
  [...18366 more chars...]
```

**assistant (target)**

```json
{"summaries": [{"name": "SSE-KMS", "summary": "SSE-KMS is an encryption type option for restoring objects in S3 buckets via AWS Backup. During restore, users can choose AWS Key Management Service key (SSE-KMS) and select an AWS managed key (aws/s3), choose from existing AWS KMS keys, or enter an AWS KMS key ARN. When using SSE-KMS, the KMSKey metadata parameter must be specified to indicate the SSE-KMS key to use. Cross-Region copy AWS KMS encryption is not available for Amazon S3 resources; a different encryption type must be used for restore jobs involving cross-Region copies. These encryption settings apply only to objects in the S3 bucket, not to the bucket itself."}]}
```

---

## `dedupe_edges.resolve_edge`
* provenance: **captured**
* loss_weight: `2.5234` (basis `full`)  |  response_schema: `EdgeDuplicate`
* article `56af70d3-1433-410b-b5f5-cb164cbacd89` chunk `3` episode `b470f595-a261-482e-9b09-18fcc50db8b6`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `new_fact_via_capture`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are a fact deduplication assistant. NEVER mark facts with key differences as duplicates.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
[{'idx': 0, 'fact': 'Azure Backup uses Resource Guard for multiuser authorization (MUA) to ensure critical operations are performed only with applicable authorization.'}]
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
[]
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
Azure Backup uses Resource Guard to enforce Multi-User Authorization for critical operations.
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT contradicts from either list.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
   - Return all contradicted idx values in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=[0], contradicted_facts=[] (identical factual informati
  [...486 more chars...]
```

**assistant (target)**

```json
{"duplicate_facts": [0], "contradicted_facts": []}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.2551` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `82ed2eb9-24a3-4322-bccb-958dd88c0f31` chunk `1` episode `adb1dc0c-cda2-443a-9156-698b2d79fe11`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Continuous backups and PITR › Continuous backups and PITR]\n\n###### Metadata parameters for RDS PITR restore\n\nThe following metadata parameters are supported for RDS and Aurora PITR restores:\n\n* **DBInstanceIdentifier** (RDS) or **DBClusterIdentifier** (Aurora) \\- Required. The name for the restored database.\n* **Engine** \\- Required. The database engine (e.g., mysql, postgres, aurora\\-mysql, aurora\\-postgresql).\n* **UseLatestRestorableTime** \\- Optional. Set to \"true\" to restore to the latest restorable time, or \"false\" to specify a RestoreTime.\n* **RestoreTime** \\- Optional. The date and time to restore to (ISO 8601 format). Required if UseLatestRestorableTime is \"false\".\n\n\n###### Copy tags to restored resource\n\nUse the `--copy-source-tags-to-restored-resource` flag to copy tags from the source database to the restored database. This ensures tag\\-based access controls and cost allocation tags are preserved.\n\n\nFor complete details on RDS PITR restore parameters, see:\n\n* [RestoreDBInstanceToPointInTime](https://docs.aws.amazon.com/AmazonRDS/latest/APIReference/API_RestoreDBInstanceToPointInTime.html) in the Amazon RDS API Reference\n* [RestoreDBClusterToPointInTime](https://docs.aws.amazon.com/AmazonRDS/latest/APIReference/API_RestoreDBClusterToPointInTime.html) in the Amazon RDS API Reference\n\n", "timestamp": "2026-08-17T23:53:18.361640+00:00"}, {"content": "[Continuous backups and PITR › Continuous backups and PITR]\n### Aurora\n\nTo enable continuous backup of your Aurora resources, see the steps in the first\n section of this page.\n\nThe procedure to restore an Aurora cluster to a point in time is a [variation of\n the steps to restore a snapshot of an aurora cluster](https://docs.aws.amazon.com/a
  [...25087 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "AWS Backup", "target_entity_name": "Amazon EC2", "relation_type": "Supports", "fact": "AWS Backup restores Amazon EC2 instances by creating an AMI, an instance, the Amazon EBS root volume, Amazon EBS data volumes, and Amazon EBS snapshots.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "Amazon EC2", "relation_type": "Limits", "fact": "AWS Backup configures the restored Amazon EC2 instance to use the same key pair that the protected resource used originally and does not allow specifying a different key pair during the restore process.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "Amazon EC2", "relation_type": "Limits", "fact": "AWS Backup does not back up and restore user-data that is used while launching an Amazon EC2 instance.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "instance profile", "relation_type": "Limits", "fact": "When configuring the restored Amazon EC2 instance, AWS Backup allows choosing between using the same instance profile that the protected resource used originally or launching without an instance profile to prevent privilege escalations.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "IAM role", "relation_type": "Requires", "fact": "If AWS Backup uses the original instance profile for the restored Amazon EC2 instance, it must be granted the iam:PassRole permission for the IAM role associated with the instance profile.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Am
  [...262 more chars...]
```

---

## `dedupe_edges.resolve_edge`
* provenance: **captured**
* loss_weight: `2.5234` (basis `full`)  |  response_schema: `EdgeDuplicate`
* article `49d019c4-ceb8-4a0a-b908-cd35602cab20` chunk `5` episode `34957b29-9467-46dd-b1cf-16e0b2a5f29e`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `new_fact_via_capture`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are a fact deduplication assistant. NEVER mark facts with key differences as duplicates.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
[{'idx': 0, 'fact': 'AWS Backup restore of Amazon S3 objects does not delete objects (without delete markers) from a bucket during a restore'}, {'idx': 1, 'fact': 'AWS Backup restore of Amazon S3 objects does not perform a destructive restore to the original bucket'}, {'idx': 2, 'fact': 'AWS Backup provides the capability to restore all objects in an Amazon S3 bucket or specific objects.'}, {'idx': 3, 'fact': 'AWS Backup restore of Amazon S3 objects with public ACLs is not restored if Block Public Access is enabled on the destination bucket'}, {'idx': 4, 'fact': 'AWS Backup provides the capability to restore Amazon S3 data to an existing or new bucket.'}, {'idx': 5, 'fact': 'AWS Backup provides the ability to restore up to the 10 latest versions of Amazon S3 objects using the AWS Backup console.'}, {'idx': 6, 'fact': 'AWS Backup restore of Amazon S3 objects is skipped if the destination bucket has an object with the same name or version ID'}, {'idx': 7, 'fact': 'AWS Backup restores only the latest version of Amazon S3 objects by default.'}, {'idx': 8, 'fact': 'AWS Backup restore of Amazon S3 data requires the destination bucket to have versioning turned on'}, {'idx': 9, 'fact': 'AWS Backup cannot restore ACLs for an Amazon S3 bucket if the backup was created with the BackupACLs option disabled.'}]
</EXISTIN
  [...1643 more chars...]
```

**assistant (target)**

```json
{"duplicate_facts": [0], "contradicted_facts": []}
```

---

## `extract_nodes.extract_summaries_batch`
* provenance: **captured**
* loss_weight: `0.9842` (basis `full`)  |  response_schema: `SummarizedEntities`
* article `1f433037-fd50-482c-b6c5-9f9c4480d2f2` chunk `10` episode `53b2cb0f-b75f-4bae-b7cc-c734a21c62be`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are a helpful assistant that generates concise entity summaries from provided context.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

Given the MESSAGES and a list of ENTITIES, generate an updated summary for each entity that needs one.
Each summary must be under 1000 characters.

Guidelines:
        1. Output only factual content. Never explain what you're doing, why, or mention limitations or constraints.
        2. Only use the provided messages, entity, and entity context to set attribute values.
        3. Keep the summary information-dense and entity-specific. STATE FACTS DIRECTLY IN UNDER 1000 CHARACTERS.
        4. Preserve all materially relevant names, roles, places, dates, counts, and temporal qualifiers that are explicitly supported.
        5. Prefer compact factual sentences over vague thematic phrasing or meta-language.
        6. When the durable fact is the content of what was said, state the content directly instead of narrating that it was said.
        7. Use communication verbs only when the act of speaking, asking, sharing, presenting, announcing, or telling is itself the important fact.
        8. Never use filler verbs like "mentioned", "described", "stated", "reported", "noted", "discussed", "referenced", or "indicated" unless the communication act itself is the fact.
        9. Include temporal anchors when the messages provide them and they help ground the fact.
        10. Begin with the entity name or a direct fact, not with "A", "An", "The", or "This is" unless that wording is part of the entity name.

        Example summary:
        BAD: "The context shows John ordered pizza. Due to length constraints, other details are omitted from this summary."
        GOOD: "John ordered pepperoni pizza from Mario's at 7:30 PM and had it delivered to the office."
        

<MESSAGES>
[{"content": "[FAQ- Recovery Services vault › FAQ- Recovery Services vault]\nAzure Backup \\- Frequ
  [...15619 more chars...]
```

**assistant (target)**

```json
{"summaries": [{"name": "Volume", "summary": "Volume is a data source in Azure Backup. Its size is determined by the amount of data being backed up from a single volume of the VM being backed up."}, {"name": "Exchange", "summary": "Exchange is a data source in Azure Backup. Its size is determined by the sum of all Exchange databases in an Exchange server being backed up."}, {"name": "BMR", "summary": "BMR (or BMR/System state) is a data source in Azure Backup. Its size is determined by each individual copy of BMR or system state of the machine being backed up."}]}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.2551` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `dad69d7c-189b-4d53-8d70-40697eae685a` chunk `5` episode `d0d4b44f-d559-45c8-802e-218ef528e30e`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Backup vaults › Backup vaults]\nBackup vaults\n=============\n\nIn AWS Backup, a *backup vault* is a container that stores and organizes your\n backups.\n\n\nWhen creating a backup vault, you must specify the AWS Key Management Service (AWS KMS) encryption key that\n encrypts some of the backups placed in this vault\n. Encryption for other backups is managed by\n their source AWS services.\n For more information about encryption, see the chart in [Encryption for backups\n in AWS](https://docs.aws.amazon.com/aws-backup/latest/devguide/encryption.html).\n\n\nThe following sections provide an overview of how to manage your backup vaults in AWS Backup.\n\n\n###### Topics\n\n* [Backup vault creation and deletion](./create-a-vault.html)\n* [Logically air\\-gapped vault](./logicallyairgappedvault.html)\n* [Vault access policies](./create-a-vault-access-policy.html)\n* [AWS Backup Vault Lock](./vault-lock.html)\n", "timestamp": "2026-06-25T23:20:38.079166+00:00"}, {"content": "[Backup vault creation and deletion › Backup vault creation and deletion]\nBackup vault creation and deletion\n==================================\n\nYou must create at least one vault before creating a backup plan or starting a backup\n job.\n\nWhen you first use the **Backup Vaults** page of the AWS Backup console in an\n AWS Region, the console automatically creates a default vault for the Region.\n\n###### Note\n\nIf a Default vault exists in a Region, AWS Backup periodically identifies orphaned AMIs \\- \n AMIs created by AWS Backup that are no longer associated with a recovery point \\- and places them \n as expired recovery points in the Default vault. You can then delete these expired recovery \n points from the Default vault, which also deletes the associated AM
  [...29902 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "AWS Backup", "target_entity_name": "AWS Backup Vault Lock", "relation_type": "Provides", "fact": "AWS Backup provides AWS Backup Vault Lock as an optional feature of a backup vault.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup Vault Lock", "target_entity_name": "us-east-1", "relation_type": "AvailableIn", "fact": "AWS Backup Vault Lock is available in the us-east-1 region.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "DescribeBackupVault", "target_entity_name": "AWS Backup", "relation_type": "Operates", "fact": "DescribeBackupVault operates on AWS Backup to review AWS Backup Vault Lock configuration details.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "ListBackupVaults", "target_entity_name": "AWS Backup", "relation_type": "Operates", "fact": "ListBackupVaults operates on AWS Backup to review AWS Backup Vault Lock configuration details.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}]}
```

---

## `dedupe_nodes.nodes`
* provenance: **captured**
* loss_weight: `0.3004` (basis `full`)  |  response_schema: `NodeResolutions`
* article `424b1222-b049-4e5a-82b3-cd5735f60f79` chunk `3` episode `e9f26ba1-64b9-4654-bde6-4ce735cb2617`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are an entity deduplication assistant. NEVER fabricate entity names or mark distinct entities as duplicates.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS MESSAGES>
[{"content": "[FAQ- Recovery Services vault › FAQ- Recovery Services vault]\n\n### Are there size limits for data backup?\n\nSizes limits are as follows:\n\n| OS/machine | Size limit of data source |\n| --- | --- |\n| Windows 8 or later | 54,400 GB |\n| Windows 7 | 1700 GB |\n| Windows Server 2012 or later | 54,400 GB |\n| Windows Server 2008, Windows Server 2008 R2 | 1700 GB |\n| Azure VM | See the [support matrix for Azure VM backup](backup-support-matrix-iaas#vm-storage-support) |\n\nNote\n\nWindows Server 2008, 2008 R2, 2012 and 2012 R2 have reached End of Support (EOS). Review your usage and plan OS upgrades and migrations accordingly. For more information, see End of support for:\n\n* [Windows Server 2008 and Windows Server 2008 R2](/en-us/troubleshoot/windows-server/windows-server-eos-faq/end-of-support-windows-server-2008-2008r2)\n* [Windows Server 2012](/en-us/windows/release-health/status-windows-server-2012)\n* [Windows Server 2012 R2](/en-us/windows/release-health/status-windows-8.1-and-windows-server-2012-r2)\n\n[Perform in\\-place upgrade to Windows Server 2016, 2019, 2022, or 2025](/en-us/azure/virtual-machines/windows-in-place-upgrade#perform-in-place-upgrade-to-windows-server-2016-2019-2022-or-2025).\n", "timestamp": "2026-07-14T22:02:42.343231+00:00"}, {"content": "[FAQ- Recovery Services vault › FAQ- Recovery Services vault]\n\n### How is the data source size determined?\n\nThe following table explains how each data source size is determined.\n\n| **Data source** | **Details** |\n| --- | --- |\n| Volume | The amount of data being backed up from single volume VM being backed up. |\n| SQL Server database | Size of single database size being backed up. |\n| SharePoint | Sum of the content and configuration databases within a SharePoi
  [...22776 more chars...]
```

**assistant (target)**

```json
{"entity_resolutions": [{"id": 0, "name": "Azure immutable vault", "duplicate_candidate_id": 14}]}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.2551` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `e2f3ee74-04f0-4d6f-9925-83dc0b43b193` chunk `0` episode `e8710215-a884-4391-9e21-d1be93640fea`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Monitor jobs › Monitor jobs]\n\n[\n\n![Screenshot showing the Jobs homepage.](/media/ea2a35ad-6628-414d-bb83-0fac5a4d6882/5d1a431ee514.png)\n\n> **Figure:** Screenshot of the Jobs homepage in Azure Backup, showing a list of backup operations with their status (In progress, Completed), item names, vaults, start times, and durations. The interface includes filters for solution, data source type, subscription, and time range, with summary counts for job outcomes.\n\n](media/tutorial-monitor-operate/job-homepage.png#lightbox)\n\n3. To change the scope for **Jobs** view from the scope\\-picker, select **Change** corresponding to **Currently showing: Protected items job details of Azure managed resources**.\n", "timestamp": "2026-07-14T21:59:26.617768+00:00"}, {"content": "[Monitor jobs › Monitor jobs]\n\n[\n\n![Screenshot showing the scope picker option.](/media/ea2a35ad-6628-414d-bb83-0fac5a4d6882/bbefc643686e.png)\n\n> **Figure:** Screenshot of the Azure Backup Jobs page showing the 'Change scope' dialog. The dialog allows selection of job source ('Protected items' or 'Other') and resource scope ('Azure' or 'Non-Azure'). The main page displays job details filtered to 'Protected items' for Azure managed resources.\n\n](media/tutorial-monitor-operate/scope-picker.png#lightbox)\n\n4. On the **Change scope** blade, select the following options as required, and then select **Update**.", "timestamp": "2026-07-14T21:59:26.617768+00:00"}, {"content": "[Monitor jobs › Monitor jobs]\n\n\n\t* **Resource managed by**:\n\t\t+ **Azure resource:** Resources that are under the direct management and control of Azure. Azure resources are provisioned, configured, and monitored through Azure's services and tools. They're fully integrated into the Azure ecos
  [...32479 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "Azure Backup", "target_entity_name": "Cross Region Restore", "relation_type": "Provides", "fact": "Azure Backup provides the Cross Region Restore capability.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Cross Region Restore", "target_entity_name": "Azure database for PostgreSQL", "relation_type": "AppliesTo", "fact": "Cross Region Restore applies to Azure database for PostgreSQL.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Cross Region Restore", "target_entity_name": "Azure VM", "relation_type": "AppliesTo", "fact": "Cross Region Restore applies to SQL databases hosted on Azure VMs.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Cross Region Restore", "target_entity_name": "Azure paired region", "relation_type": "AvailableIn", "fact": "Cross Region Restore allows restoring data in a secondary Azure paired region.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "Geo-redundant Storage", "relation_type": "Limits", "fact": "A Geo-redundant Storage (GRS) vault with Cross Region Restore enabled cannot be changed to Zone-redundant Storage (ZRS) or Locally redundant Storage (LRS) after protection starts for the first time.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Cross Region Restore", "target_entity_name": "Recovery Point Objective", "relation_type": "Limits", "fact": "The secondary region Recovery Point Objective (RPO) for Cross Region Restore is currently 36 hours.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}]}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.2551` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `a6481417-ece0-481d-a73e-c4279e5eec5f` chunk `8` episode `d685258e-4cde-45f2-ab23-64631a067da3`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Azure Backup architecture › Azure Backup architecture]\n### Back up and restore Azure VMs with premium storage\n\nYou can back up Azure VMs by using premium storage with Azure Backup:\n\n* During the process of backing up VMs with premium storage, the Backup service creates a temporary staging location, named *AzureBackup\\-*, in the storage account. The size of the staging location equals the size of the recovery point snapshot.\n* Make sure that the premium storage account has adequate free space to accommodate the temporary staging location. For more information, see [Scalability targets for premium page blob storage accounts](../storage/blobs/scalability-targets-premium-page-blobs). Don't modify the staging location.\n* After the backup job finishes, the staging location is deleted.\n* The price of storage used for the staging location is consistent with [premium storage pricing](/en-us/azure/virtual-machines/disks-types#billing).\n\nWhen you restore Azure VMs by using premium storage, you can restore them to premium or standard storage. Typically, you would restore them to premium storage. But if you need only a subset of files from the VM, it might be cost effective to restore them to standard storage.\n\n### Back up and restore managed disks\n\nYou can back up Azure VMs with managed disks:\n\n* You back up VMs with managed disks in the same way that you do any other Azure VM. You can back up the VM directly from the virtual machine settings, or you can enable backup for VMs in the Recovery Services vault.\n* You can back up VMs on managed disks through RestorePoint collections built on top of managed disks.\n* Azure Backup also supports backing up VMs with managed disks that were encrypted by using Azure Disk Encryption.\n\nWhe
  [...38547 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "Azure Backup", "target_entity_name": "Azure Advisor", "relation_type": "IntegratesWith", "fact": "Azure Backup integrates with Azure Advisor, which recommends configuration of backup-related security for production environments.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "Microsoft Defender for Cloud", "relation_type": "IntegratesWith", "fact": "Azure Backup integrates with Microsoft Defender for Cloud, which recommends configuration of backup-related security for production environments.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "Recovery Services vaults", "relation_type": "Provides", "fact": "Azure Backup provides storage redundancy types that can be found in Recovery Services vaults.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "Backup vault", "relation_type": "Provides", "fact": "Azure Backup provides storage redundancy types that can be found in Backup vault.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}]}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.2551` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `554ebdbf-e818-43e8-9d1d-55fbd28a983b` chunk `3` episode `322d7280-a60f-43bc-a789-476932d7afa5`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Getting started › Getting started]\n\nOpt\\-in choices apply to the specific account and AWS Region. When an account uses AWS Backup\n (creates a backup vault or backup plan) in a Region, the account automatically is opted into\n all resource types supported by AWS Backup in the Region at that time. Supported services added to\n that Region at a later date will not be automatically included in a backup plan. You can\n choose to opt into those resource types once they become supported.\n\nAs AWS Backup supports more and more AWS services and third\\-party applications, you might\n need to revisit this step to opt in to those newly\\-supported resources.\n\nAWS Backup does not govern or manage backups taken in AWS environments other than \n AWS Backup.\n\n\n###### Note\n\nFor more information on how service opt\\-ins work for management and delegated administrator accounts, see [Managing AWS Backup resources across multiple AWS accounts](./manage-cross-account.html).\n\n\n###### To opt in to use AWS Backup to protect all supported resource types\n\n1. Sign in to the AWS Management Console, and open the AWS Backup console at <https://console.aws.amazon.com/backup>.\n2. In the left navigation pane, choose **Settings**.\n3. Under **Service opt\\-in**, choose **Configure\n resources**.\n4. Opt in to all AWS Backup\\-supported **Resources** by moving all the toggles\n to the right.\n5. Choose **Confirm**.\n", "timestamp": "2026-06-25T23:20:36.805845+00:00"}, {"content": "[Backup plans › Backup plans]\nBackup plans\n============\n\nIn AWS Backup, a *backup plan* is a policy expression that defines when and\n how you want to back up your AWS resources, such as Amazon DynamoDB tables or Amazon Elastic File System (Amazon EFS) file\n systems.\n 
  [...26607 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "AWS Backup", "target_entity_name": "retention policy", "relation_type": "Provides", "fact": "AWS Backup retains the backup under the rule with the longer retention period when start windows of different rules overlap.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "start window", "relation_type": "Provides", "fact": "During the start window, the backup job status remains in CREATED status until it has successfully begun or until the start window time has run out.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "start window", "relation_type": "Provides", "fact": "If within the start window time AWS Backup receives an error that allows the job to be retried, AWS Backup will automatically retry to begin the job at least every 10 minutes until the backup successfully begins or until the job status changes to EXPIRED.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}]}
```

---

## `dedupe_edges.resolve_edge`
* provenance: **captured**
* loss_weight: `2.5234` (basis `full`)  |  response_schema: `EdgeDuplicate`
* article `a6481417-ece0-481d-a73e-c4279e5eec5f` chunk `0` episode `4a404847-ef55-4bc9-8036-99a5513ae5b8`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `new_fact_via_capture`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are a fact deduplication assistant. NEVER mark facts with key differences as duplicates.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
[{'idx': 0, 'fact': 'Azure Backup provides multiuser authorization (MUA) for authorizing critical protected operations using Azure Active Directory Privileged Identity Management.'}, {'idx': 1, 'fact': 'Azure Backup provides Multi-user authorization (MUA), which adds an additional layer of protection to critical operations on Recovery Services vaults and Backup vaults using Resource Guard.'}, {'idx': 2, 'fact': 'Azure Backup provides Multi-user authorization (MUA) as a security feature for Recovery Service vaults and Backup vaults.'}]
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
[]
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
Azure Backup includes multiuser authorization.
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT contradicts from either list.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).

  [...809 more chars...]
```

**assistant (target)**

```json
{"duplicate_facts": [0, 1, 2], "contradicted_facts": []}
```

---

## `dedupe_edges.resolve_edge`
* provenance: **captured**
* loss_weight: `2.5234` (basis `full`)  |  response_schema: `EdgeDuplicate`
* article `e845095d-d22b-47c1-90e2-c60629331d66` chunk `7` episode `71f60bb1-2b2a-4ce3-9cfb-d22ce57b59b8`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `new_fact_via_capture`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are a fact deduplication assistant. NEVER mark facts with key differences as duplicates.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
[{'idx': 0, 'fact': 'Azure Backup limits Incremental backup for SAP HANA databases to at most one per day, and both differential and incremental backups cannot be scheduled on a database (only one delta backup type can be scheduled), and a full backup and a differential backup cannot be configured on the same day.'}, {'idx': 1, 'fact': 'Azure Backup uses incremental backups, which store only the blocks of data that changed since the previous backup and offer high storage and network efficiency without needing to supplement with full backups, in all backups to Azure.'}, {'idx': 2, 'fact': "Azure Backup provides Incremental backup for SAP HANA databases, which is based on the most recent previous full/differential/incremental-data backup and captures only the data that's changed since that previous data backup."}, {'idx': 3, 'fact': "Azure Backup's incremental backups mark and transfer only the blocks that changed since the previous backup in each subsequent month."}]
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
[]
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
Azure Backup provides Incremental Backup as a backup type.
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DET
  [...1261 more chars...]
```

**assistant (target)**

```json
{"duplicate_facts": [0, 1, 2], "contradicted_facts": []}
```

---

## `extract_nodes.extract_text`
* provenance: **captured**
* loss_weight: `0.5655` (basis `full`)  |  response_schema: `ExtractedEntities`
* article `17fb9911-e0c1-47eb-8055-694293196941` chunk `1` episode `7480c5a9-cc52-4899-b44c-41fdc5608b5f`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are an entity extraction specialist for unstructured text. NEVER extract abstract concepts, feelings, or generic words.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

NEVER extract:
- Pronouns (you, me, he, she, they, it, them, him, her, we, us, this, that, those)
- Abstract concepts (joy, balance, growth, resilience, passion, motivation)
- Generic common nouns or bare object words (day, life, people, work, stuff, things, food, time,
  tickets, supplies, clothes, keys, gear)
- Generic media/content nouns unless uniquely identified in the node name itself (photo, pic, picture,
  image, video, post, story)
- Generic event/activity nouns unless uniquely identified in the node name itself (event, game, meeting,
  class, workshop, competition)
- Broad institutional nouns unless explicitly named or uniquely qualified (government, school, company,
  team, office)
- Ambiguous bare nouns whose meaning depends on sentence context rather than the node name itself
- Sentence fragments or clauses as entity names
- Bare relational or kinship terms (dad, mom, sister, brother, spouse, friend, boss, pet, dog,
  cat) unless qualified with a possessor (e.g., "Nisha's dad" is acceptable, "dad" alone is not)
- Bare generic objects that cannot be meaningfully qualified with a possessor, brand, or
  distinguishing detail (e.g., NEVER extract "supplies" from "I picked up some supplies")

Extract entities from the TEXT that are **explicitly mentioned**.
For each entity, classify it using the ENTITY TYPES above.
Only extract entities specific enough to be uniquely identifiable — ask: "Could this have its own Wikipedia article or database entry?"

<ENTITY TYPES>
[{'entity_type_id': 0, 'entity_type_name': 'Entity', 'entity_type_description': 'A specific, identifiable entity that does not fit any of the other listed types. Must still be a concrete, meaningful thing — specific enough to be uniquely identifiable. GOOD: a named entity not covered by the other type
  [...11887 more chars...]
```

**assistant (target)**

```json
{"extracted_entities": [{"name": "AWS Backup", "entity_type_id": 2, "episode_indices": []}, {"name": "CloudFormation", "entity_type_id": 1, "episode_indices": []}, {"name": "AWS::Backup::BackupVault", "entity_type_id": 3, "episode_indices": []}, {"name": "AWS::Backup::BackupPlan", "entity_type_id": 3, "episode_indices": []}, {"name": "Amazon RDS", "entity_type_id": 4, "episode_indices": []}, {"name": "Amazon Aurora", "entity_type_id": 4, "episode_indices": []}, {"name": "Amazon Neptune", "entity_type_id": 4, "episode_indices": []}, {"name": "Amazon DocumentDB", "entity_type_id": 4, "episode_indices": []}]}
```

---

## `dedupe_edges.resolve_edge`
* provenance: **captured**
* loss_weight: `2.5234` (basis `full`)  |  response_schema: `EdgeDuplicate`
* article `186a0012-43f6-499d-b271-d44f4372331e` chunk `None` episode `None`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `new_fact_via_capture`, target_normalised `True`
* repaired: none — target is exactly what the model emitted
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason).

**system**

```
You are a fact deduplication assistant. NEVER mark facts with key differences as duplicates.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

<EXISTING FACTS>
[{'idx': 0, 'fact': 'Backup center supports SAP HANA on Azure VM backup.'}, {'idx': 1, 'fact': 'Backup center supports executing cross-region restore jobs for SAP HANA in Azure VM.'}]
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
[]
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
Backup center supports SAP HANA in Azure VM backup.
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Determine which facts the NEW FACT contradicts from either list.
   - A fact from EXISTING FACTS can be both a duplicate AND contradicted (e.g., semantically the same but the new fact updates/supersedes it).
   - Return all contradicted idx values in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "Alice joined Acme Corp in 2020"
NEW FACT: "Alice joined Acme Corp in 2020"
Result: duplicate_facts=[0], contradicted_facts=[] (identical factual information)

EXISTING FACT: idx=1, "A
  [...457 more chars...]
```

**assistant (target)**

```json
{"duplicate_facts": [0, 1], "contradicted_facts": []}
```

