# Sample records — readable digest
Generated from `docs/proposals/samples/sample.jsonl` (20 records). This file is a
REVIEW AID: long bodies are truncated. The JSONL is the actual deliverable
and is verbatim.

Rebuild both with:

```
uv run --extra dev python scripts/build_extraction_dataset.py \
    --from-capture /tmp/claude-1000/-home-openclaw-Documents-Shared-projects-graph-rag/964252d0-58b1-4131-93c7-53185320cfde/scratchpad/llm-capture-20260917.jsonl \
    --out docs/proposals/samples --limit 20 --sample-only --mix
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.3129` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `49d019c4-ceb8-4a0a-b908-cd35602cab20` chunk `3` episode `3ca31b5c-6b61-4610-a81e-2834b080af67`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Cross-account backup › Cross-account backup]\n\n\n###### Limit destination accounts using account numbers and vault names\n\nWhen attached to an AWS Organizations root, OU, or individual account, this policy limits\n copies originating from that root, OU, or account to only two destination accounts. The\n permission `\"backup:CopyFromBackupVault\"` controls how a recovery point in\n the backup vault behaves, and, in this case, the destinations where you can copy that\n recovery point to. The source vault will only permit copies to the first destination\n account (112233445566\\) if one or more destination backup vault names begin with\n `cab-`. The source vault will only permit copies to the second destination\n account (123456789012\\) if the destination is the single backup vault named\n `fort-knox`.\n", "timestamp": "2026-06-25T23:20:45.858260+00:00"}, {"content": "[Cross-account backup › Cross-account backup]\n\nJSON\n\n```\n\n{\n  \"Version\":\"2012-10-17\",\n  \"Statement\": [\n    {\n      \"Sid\": \"DenyCopyFromBackupVault\",\n      \"Effect\": \"Deny\",\n      \"Action\": \"backup:CopyFromBackupVault\",\n      \"Resource\": \"arn:aws:ec2:*:*:snapshot/*\",\n      \"Condition\": {\n        \"ForAllValues:ArnNotLike\": {\n          \"backup:CopyTargets\": [\n            \"arn:aws:backup:*:112233445566:backup-vault:cab-*\",\n            \"arn:aws:backup:us-east-1:123456789012:backup-vault:fort-knox\"\n          ]\n        }\n      }\n    }\n  ]\n}\n\n```\n\n\n###### Limit destination accounts using organizational units in AWS Organizations\n\nWhen attached to an AWS Organizations root or OU that contains your source account, or when\n attached to your source account, the following policy limits the destination accounts to\n those
  [...23626 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "AWS Backup", "target_entity_name": "Amazon S3", "relation_type": "Supports", "fact": "AWS Backup supports backing up Amazon S3 data.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "Restore ACLs", "relation_type": "Provides", "fact": "AWS Backup provides a Restore ACLs setting to choose whether ACLs are part of the restore.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Restore ACLs", "target_entity_name": "Amazon S3", "relation_type": "AppliesTo", "fact": "The Restore ACLs setting applies to restoring Amazon S3 data.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "object tags", "relation_type": "Provides", "fact": "AWS Backup automatically restores object tags if they were included in the original backup.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "Amazon S3", "relation_type": "Limits", "fact": "AWS Backup cannot restore ACLs for an Amazon S3 bucket if the backup was created with the BackupACLs option disabled.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "Amazon S3", "relation_type": "Limits", "fact": "AWS Backup restores only the latest version of Amazon S3 objects by default.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "Amazon S3", "relation_type": "Provides", "fact": "AWS Backup provides the choice to restore additional or all versions of Amazon S3 objects.", "valid_at": null, "invalid_at": null, "episod
  [...911 more chars...]
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.3129` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `e845095d-d22b-47c1-90e2-c60629331d66` chunk `3` episode `c67fc735-3f4a-4575-aead-29e149f3424c`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Azure Backup documentation › Azure Backup documentation]\n\nBack up with Microsoft Azure Recovery Services (MARS) agent\n-----------------------------------------------------------\n\n### Concept\n\n* [Back up Windows machines with the Azure Backup MARS agent](backup-windows-with-mars-agent)\n* [Manage MARS Agent backups by using the Azure Backup service](backup-azure-manage-mars)\n* [Cross Region Restore support for MARS agent (preview)](about-restore-microsoft-azure-recovery-services#cross-region-restore-for-mars)\n* [MARS agent support matrix](backup-support-matrix-mars-agent)\n* [FAQ \\- MARS agent](backup-azure-file-folder-backup-faq)\n\n### How\\-To Guide\n\n* [Offline backup for MARS with customer\\-owned disks](backup-azure-backup-import-export)\n\n\nBack up with Microsoft Azure Backup Server (MABS) and DPM\n---------------------------------------------------------\n\n### Concept\n\n* [MABS protection matrix](backup-mabs-protection-matrix)\n* [MABS / DPM support matrix](backup-support-matrix-mabs-dpm)\n* [FAQ \\- MABS and DPM](backup-azure-dpm-azure-server-faq)\n* [Install and upgrade MABS](backup-azure-microsoft-azure-backup)", "timestamp": "2026-06-25T23:42:01.248767+00:00"}, {"content": "[Azure Backup documentation › Azure Backup documentation]\n\n\n### How\\-To Guide\n\n* [Offline backup using Azure Data Box](offline-backup-azure-data-box)\n* [Manage backup to Azure for DPM servers via PowerShell](backup-dpm-automation)\n* [Offline backup using Azure Data Box for DPM and MABS](offline-backup-azure-data-box-dpm-mabs)\n* [Manage telemetry settings in MABS](manage-telemetry)\n* [Back up an Exchange server with DPM](backup-azure-backup-exchange-server)\n", "timestamp": "2026-06-25T23:42:01.248767+00:00"}, {"content": "[Azure B
  [...26411 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "Azure Backup", "target_entity_name": "Full backup", "relation_type": "Provides", "fact": "Azure Backup uses full backups, which contain the entire data source and take more network bandwidth than differential or incremental backups, for initial backups.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "Differential backup", "relation_type": "Limits", "fact": "Azure Backup does not use differential backups, which store blocks changed since the initial full backup and are inefficient because unchanged data blocks between later backups are transferred and stored.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "Incremental backup", "relation_type": "Provides", "fact": "Azure Backup uses incremental backups, which store only the blocks of data that changed since the previous backup and offer high storage and network efficiency without needing to supplement with full backups, in all backups to Azure.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Incremental backup", "target_entity_name": "DPM", "relation_type": "AppliesTo", "fact": "Incremental backups are used by DPM for disk backups.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Incremental backup", "target_entity_name": "MABS", "relation_type": "AppliesTo", "fact": "Incremental backups are used by MABS for disk backups.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Incremental backup", "target_entity_name": "SQL Server", "relation_type": "Limits", "fact": "Incremental backups are not used for SQL Server backu
  [...68 more chars...]
```

---

## `extract_nodes.extract_summaries_batch`
* provenance: **captured**
* loss_weight: `0.4523` (basis `full`)  |  response_schema: `SummarizedEntities`
* article `8f869443-b48d-4a45-a9fb-7d0df772e632` chunk `4` episode `d021840b-a077-4787-8d76-af53e5a69d07`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

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
[{"content": "[Manage › Manage]\nEnable Immutable vault\n----------------------\n\nYou can enable i
  [...18887 more chars...]
```

**assistant (target)**

```json
{"summaries": [{"name": "SQL Server Management Studio", "summary": "SQL Server Management Studio is used to recover backup files restored via the 'Restore as files' option into a database on any machine where the files are present."}]}
```

---

## `extract_nodes.extract_summaries_batch`
* provenance: **captured**
* loss_weight: `0.4523` (basis `full`)  |  response_schema: `SummarizedEntities`
* article `49d019c4-ceb8-4a0a-b908-cd35602cab20` chunk `2` episode `cba7cafe-27e2-45e8-9608-8af2d3d19a7a`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

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
[{"content": "[Cross-account backup › Cross-account backup]\n\n###### Limit destination accounts us
  [...14060 more chars...]
```

**assistant (target)**

```json
{"summaries": [{"name": "AWS KMS encryption", "summary": "AWS KMS encryption is used to encrypt backups in destination accounts for cross-account backup operations. For resources fully managed by AWS Backup, either a customer managed key or the AWS Backup managed key (aws/backup) can be used for the destination vault. For all other resource types, a customer managed key is required because AWS managed key policies are immutable and cannot be shared cross-account. The destination vault cannot be the default vault because the default vault is encrypted with a key that cannot be shared with other accounts. Cross-Region copy AWS KMS encryption is not available for Amazon S3 resources; instead, a different encryption type must be used for a restore job. The original encryption method cannot be used to restore a recovery point copied from another Region. AWS Backup re-encrypts cross-account copies using the encryption key of the destination vault."}]}
```

---

## `dedupe_edges.resolve_edge`
* provenance: **captured**
* loss_weight: `2.9471` (basis `full`)  |  response_schema: `EdgeDuplicate`
* article `f8bd3e5d-ae81-417b-8dae-84875c95ff7b` chunk `9` episode `3a69019c-e907-47d8-868f-067d292aeb81`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `new_fact_via_capture`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

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
[{'idx': 0, 'fact': 'Backup center supports viewing backup policies for Azure Virtual Machine.'}, {'idx': 1, 'fact': 'Backup center allows triggering on-demand backups and restores for Azure Virtual Machines backup instances.'}, {'idx': 2, 'fact': 'Backup center provides a summarized view of backup instances for Azure Virtual Machines.'}, {'idx': 3, 'fact': 'Backup center allows stopping backup for Azure Virtual Machines backup instances.'}, {'idx': 4, 'fact': 'Backup center supports viewing datasources not configured for backup for Azure Virtual Machine.'}, {'idx': 5, 'fact': 'Backup center provides the capability to view Azure Monitor alerts at scale for Azure Virtual Machine.'}, {'idx': 6, 'fact': 'Backup center provides the capability to view Azure Backup metrics and write metric alert rules for Azure VM.'}, {'idx': 7, 'fact': "Backup center's Backup jobs screen shows a list of completed backup operations for Azure Virtual Machines."}, {'idx': 8, 'fact': 'Backup center shows active alerts for Azure Virtual machines on the Overview screen.'}, {'idx': 9, 'fact': 'Backup center can monitor a maximum of 1000 subscriptions and 1000 vaults across tenants for Azure Virtual Machine monitoring scenarios.'}]
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
[]
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
Backup Ce
  [...1531 more chars...]
```

**assistant (target)**

```json
{"duplicate_facts": [], "contradicted_facts": []}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.3129` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `f8bd3e5d-ae81-417b-8dae-84875c95ff7b` chunk `8` episode `ac5d788a-d0ad-456f-8013-10e7a41c3953`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Create a vault › Create a vault]\n\n2. On **Vaults**, select **\\+Vault**.\n\n[\n\n![Screenshot showing **+Vault** options.](/media/f7237de0-fae1-4216-a42f-06d35fe46aa9/a1f7e50ccf96.png)\n\n> **Figure:** Screenshot of the Vaults page in the Resiliency section of Azure portal, showing a list of recovery and backup vaults with columns for vault name, subscription, resource group, location, type, redundancy, security level, and cross-region replication. The '+ Vault' button is highlighted, indicating the option to create a new vault.\n\n](media/backup-vaults/create-vault.png#lightbox)\n", "timestamp": "2026-07-14T22:06:43.199911+00:00"}, {"content": "[Create a vault › Create a vault]\n3. On **Start: Create Vault**, select the type of vault you want to create. You have two options:\n\n\t* [Recovery Services vaults](../backup/backup-azure-recovery-services-vault-overview) to hold backup data for various Azure services such as IaaS virtual machines (Linux or Windows) and SQL Server in Azure Virtual Machines (VMs). Recovery Services vaults support System Center Data protection Manager (DPM), Windows Server, Azure Backup Server, and replication data for Azure Site Recovery.\n\t* [Backup vaults](../backup/backup-vault-overview) to hold backup data for various Azure services, such Azure Database for PostgreSQL servers and newer workloads that Azure Backup supports.\n\t\n\t[\n\n![Screenshot showing different vault options.](/media/f7237de0-fae1-4216-a42f-06d35fe46aa9/758c37afb3be.png)\n\n> **Figure:** Screenshot of the 'Create Vault' screen showing two vault type options: 'Recovery Services vault' and 'Backup vault', each with a list of supported data sources. The 'Recovery Services vault' option is selected.\n\n](media/backup-vaults/select-vaul
  [...25068 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "Backup center", "target_entity_name": "Azure Backup", "relation_type": "Operates", "fact": "Backup center operates Azure Backup, allowing users to monitor and manage backups and disaster recovery.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Backup center", "target_entity_name": "Recovery Services vault", "relation_type": "Operates", "fact": "Backup center allows viewing a list of Recovery Services vaults that the user has access to.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Backup center", "target_entity_name": "Backup vault", "relation_type": "Operates", "fact": "Backup center allows viewing a list of Backup vaults that the user has access to.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}]}
```

---

## `dedupe_nodes.nodes`
* provenance: **captured**
* loss_weight: `0.3066` (basis `full`)  |  response_schema: `NodeResolutions`
* article `f1932f08-72a5-4be0-aff5-0e4bcbac1990` chunk `2` episode `ba7f7cb4-fc18-470f-b478-1971ca054b2a`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

**system**

```
You are an entity deduplication assistant. NEVER fabricate entity names or mark distinct entities as duplicates.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS MESSAGES>
[{"content": "[What is AWS Backup? › What is AWS Backup?]\n\n### Backup activity monitoring\n\nAWS Backup provides a dashboard that makes it simple to audit backup and restore activity\n across AWS services. With just a few clicks on the AWS Backup console, you can view the status\n of recent backup jobs. You can also restore jobs across AWS services to ensure that your\n AWS resources are properly protected.\n\nAWS Backup integrates with Amazon CloudWatch and Amazon EventBridge. CloudWatch allows you to track metrics and create\n alarms. EventBridge allows you to view and monitor AWS Backup events. For more information, see [Monitoring AWS Backup\n events using EventBridge](https://docs.aws.amazon.com/aws-backup/latest/devguide/eventbridge.html) and [Monitoring AWS Backup metrics with\n CloudWatch](https://docs.aws.amazon.com/aws-backup/latest/devguide/cloudwatch.html).\n\nAWS Backup integrates with AWS CloudTrail. CloudTrail gives you a consolidated view of backup activity\n logs that make it quick and easy to audit how your resources are backed up. AWS Backup also\n integrates with Amazon Simple Notification Service (Amazon SNS), providing you with backup activity notifications, such as\n when a backup succeeds or a restore has been initiated. For more information, see [Logging AWS Backup API calls with CloudTrail](https://docs.aws.amazon.com/aws-backup/latest/devguide/logging-using-cloudtrail.html) and [Amazon SNS and AWS Backup events](https://docs.aws.amazon.com/aws-backup/latest/devguide/backup-notifications.html#backup-notifications-section).\n", "timestamp": "2026-06-25T23:20:34.561897+00:00"}, {"content": "[What is AWS Backup? › What is AWS Backup?]\n\n### Secure your data in backup vaults\n\nThe content of each AWS Backup backup is immuta
  [...25123 more chars...]
```

**assistant (target)**

```json
{"entity_resolutions": [{"id": 0, "name": "AWS Identity and Access Management", "duplicate_candidate_id": -1}, {"id": 1, "name": "AWS account root user", "duplicate_candidate_id": -1}, {"id": 2, "name": "administrator", "duplicate_candidate_id": -1}]}
```

---

## `extract_nodes.extract_summaries_batch`
* provenance: **captured**
* loss_weight: `0.4523` (basis `full`)  |  response_schema: `SummarizedEntities`
* article `67a20d54-b82e-49a0-b6b8-2a9a8beb86e2` chunk `0` episode `e08899f0-026e-477b-93af-514e22b6fbdb`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

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
[{"content": "[Manage vault operations › Manage vault operations]\n\n4. To look for specific vaults
  [...11749 more chars...]
```

**assistant (target)**

```json
{"summaries": [{"name": "Enhanced tier", "summary": "Migration from Enhanced to Standard tier is not allowed when reconfiguring backup for data sources in an alternate vault."}, {"name": "Standard tier", "summary": "Migration from Enhanced to Standard tier is not allowed when reconfiguring backup for data sources in an alternate vault."}, {"name": "re-registration", "summary": "Re-registration is the process of reconfiguring backup for data sources in Resiliency by switching to an alternate Recovery Services vault. It allows suspending backup in one vault and reconfiguring in another without losing existing recovery points. The feature is available for all datasources supported in Recovery Services vault, but is not supported for SQL Always On Availability Groups (AG) and SAP HANA System Replication (HSR). Recovery Services vault does not allow active multi-protection. The alternate vault has no limitations on redundancy, security settings, or policy retention, including for immutable vaults."}]}
```

---

## `dedupe_edges.resolve_edge`
* provenance: **captured**
* loss_weight: `2.9471` (basis `full`)  |  response_schema: `EdgeDuplicate`
* article `1cde879f-5152-4b0e-a9e8-9d4e762ee9ed` chunk `2` episode `5beeb3e9-73aa-40ce-9ea3-9f76e4687451`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `new_fact_via_capture`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

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
[{'idx': 0, 'fact': 'Soft delete retains deleted backup data for a configurable period of up to 180 days (default 14 days) in Recovery Services vaults.'}, {'idx': 1, 'fact': 'Soft Delete applies to Recovery Services vault.'}, {'idx': 2, 'fact': 'Soft delete is enforced by default for Recovery Services vaults and is in public preview across all public regions.'}]
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
[]
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
Soft delete applies to Recovery Services vaults, allowing them to be moved into a soft-deleted state and recovered within the configured retention period.
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
   - 
  [...741 more chars...]
```

**assistant (target)**

```json
{"duplicate_facts": [1], "contradicted_facts": []}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.3129` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `e0545ce0-40cb-49be-99c8-55b6b387fdf2` chunk `4` episode `05ee8ccd-984f-408c-bdbd-88dc7dbf5c6c`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Support Matrices › Support Matrices]\n\n| Reporting | View historical data of your Backup and Site Recovery estate in Reports | **Azure Backup**:  \\- Azure Files  \\- Azure Virtual Machine  \\- SAP HANA in Azure VM  \\- SQL in Azure VM  \\- Azure Disk  \\- Azure Database for PostgreSQL Server  \\- Azure Blob  \\- Azure Backup Agent  \\- Azure Backup Server  \\- Data Protection Manager (DPM)  **Azure Site Recovery**:  \\- Azure to Azure disaster recovery  \\- VMware and Physical to Azure disaster recovery | [Shows data based on the LA workspace retention](tutorial-reporting-for-data-insights#view-reports-in-the-resiliency-in-azure). |", "timestamp": "2026-06-26T00:28:43.493367+00:00"}, {"content": "[Support Matrices › Support Matrices]\n\n| Security | View security level | Only for the Azure Backup supported datasources given in [this section](#supported-solutions-and-datasources), and vaults. |  |\n| Governance | View resources not configured for protection. | **Azure Backup**:  \\- Azure VM backup  \\- Azure Storage account (with no file or blobs protected)  \\- Azure Managed Disks backup  \\- Azure Database for PostgreSQL Server backup  \\- Azure Kubernetes services  **Azure Site Recovery**:  \\- Azure to Azure disaster recovery |  |\n| Governance | View all protected items across solutions. | All solutions and datasource types as given in [this section](#supported-solutions-and-datasources). | Same as previous. |\n| Governance | View and assign built\\-in and custom Azure Policies under category Azure Backup and Azure Site Recovery. | N/A | N/A |\n\n| Manage | View all protection policies – backup policies for Azure Backup and Replication policies for Azure Site Recovery. | All solutions and datasource types given in [this section
  [...24412 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "Azure Backup", "target_entity_name": "PostgreSQL", "relation_type": "Supports", "fact": "Azure Backup supports PostgreSQL backups protected in Backup vaults.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "Cross Region Restore", "relation_type": "Provides", "fact": "Azure Backup provides Cross Region Restore to access and perform restores from secondary region recovery points even when no outage occurs in the primary region.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Cross Region Restore", "target_entity_name": "PostgreSQL", "relation_type": "AppliesTo", "fact": "Cross Region Restore applies to PostgreSQL backups protected in Backup vaults.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Cross Region Restore", "target_entity_name": "Geo-redundant Storage", "relation_type": "Requires", "fact": "Cross Region Restore requires Geo-redundant Storage (GRS) to replicate backups to an additional Azure paired region.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Cross Region Restore", "target_entity_name": "RA-GRS", "relation_type": "Requires", "fact": "Backup vaults enabled with Cross Region Restore are automatically charged at RA-GRS rates for the PostgreSQL backups stored in the vault.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Cross Region Restore", "target_entity_name": "Backup vaults", "relation_type": "AppliesTo", "fact": "Cross Region Restore is available for Backup vaults enabled with Geo-redundant Storage.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}]}
```

---

## `dedupe_edges.resolve_edge`
* provenance: **captured**
* loss_weight: `2.9471` (basis `full`)  |  response_schema: `EdgeDuplicate`
* article `dad69d7c-189b-4d53-8d70-40697eae685a` chunk `7` episode `b225d102-cf44-421f-8aff-7c34c0b737f1`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `new_fact_via_capture`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

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
[{'idx': 0, 'fact': 'AWS Backup Vault Lock is an optional feature of a backup vault in AWS Backup.'}, {'idx': 1, 'fact': 'AWS Backup Vault Lock is a feature of AWS Backup that is available at no additional charge, with standard AWS Backup storage charges applying to backups stored in a locked vault.'}]
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
[]
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
A vault locked in compliance mode cannot be deleted once the cooling-off period (grace time) expires if any recovery points are in the vault.
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

<EXAMP
  [...667 more chars...]
```

**assistant (target)**

```json
{"duplicate_facts": [], "contradicted_facts": []}
```

---

## `extract_nodes.extract_text`
* provenance: **captured**
* loss_weight: `0.5506` (basis `full`)  |  response_schema: `ExtractedEntities`
* article `4139d961-e9e8-4d05-9edb-658bb1d2f694` chunk `0` episode `19bcd9ec-85b7-42b4-8cc8-4c09af5de395`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

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
  [...11999 more chars...]
```

**assistant (target)**

```json
{"extracted_entities": [{"name": "Azure Copilot", "entity_type_id": 3, "episode_indices": [0]}, {"name": "Resiliency in Azure", "entity_type_id": 2, "episode_indices": [0]}, {"name": "Azure Business Continuity Center", "entity_type_id": 2, "episode_indices": [0]}, {"name": "Azure", "entity_type_id": 6, "episode_indices": [0]}, {"name": "Zone Resiliency", "entity_type_id": 5, "episode_indices": [0]}, {"name": "High Availability", "entity_type_id": 5, "episode_indices": [0]}, {"name": "Backup & Disaster Recovery", "entity_type_id": 5, "episode_indices": [0]}, {"name": "Ransomware Protection", "entity_type_id": 5, "episode_indices": [0]}, {"name": "Protection management", "entity_type_id": 5, "episode_indices": [0]}, {"name": "Ransomware protection", "entity_type_id": 5, "episode_indices": [0]}, {"name": "Monitoring and reporting", "entity_type_id": 5, "episode_indices": [0]}, {"name": "Azure Site Recovery", "entity_type_id": 2, "episode_indices": [0]}, {"name": "Recovery Points", "entity_type_id": 5, "episode_indices": [0]}]}
```

---

## `dedupe_edges.resolve_edge`
* provenance: **captured**
* loss_weight: `2.9471` (basis `full`)  |  response_schema: `EdgeDuplicate`
* article `a0e14280-837e-4e23-a930-9119e7692e94` chunk `9` episode `57e34cc9-0a9c-4bac-b60d-99d291c71e5a`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `new_fact_via_capture`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

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
[{'idx': 0, 'fact': 'Azure Backup backs up Azure VMs by installing an extension to the Azure VM agent that runs on the machine.'}, {'idx': 1, 'fact': 'Azure Backup supports enabling backup of Azure VMs.'}, {'idx': 2, 'fact': 'Azure Backup supports backup of Azure VMs without requiring internet connectivity.'}, {'idx': 3, 'fact': 'Azure Backup backs up Azure VMs with data in transit remaining on the Azure backbone network without needing to access the virtual network.'}, {'idx': 4, 'fact': 'Azure Backup supports backup and restore of managed Azure VMs that have their OS/data disks encrypted with ADE, with or without the Microsoft Entra app.'}, {'idx': 5, 'fact': 'Azure Backup supports backup and restore of managed Azure VMs that have OS/data disks encrypted with Azure Disk Encryption (ADE).'}, {'idx': 6, 'fact': 'Azure Backup supports backup and restore of managed Azure VMs encrypted with Azure Disk Encryption (ADE) and key encryption keys (KEKs).'}, {'idx': 7, 'fact': 'Azure Backup backs up Azure VMs, and backup or restore failures for a VM appear as datasource alerts.'}, {'idx': 8, 'fact': 'Azure Backup can restore SQL Server databases that are running on Azure VMs.'}, {'idx': 9, 'fact': 'Restoring a SQL Server database to an alternate location requires Contributor (write) access to the new target VM.'}]
<
  [...1735 more chars...]
```

**assistant (target)**

```json
{"duplicate_facts": [], "contradicted_facts": []}
```

---

## `extract_nodes.extract_text`
* provenance: **captured**
* loss_weight: `0.5506` (basis `full`)  |  response_schema: `ExtractedEntities`
* article `c57950e3-10c0-4815-88b7-f90a2de994c1` chunk `3` episode `224252db-2748-4157-93f8-d8cb75f33d40`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

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
  [...12564 more chars...]
```

**assistant (target)**

```json
{"extracted_entities": [{"name": "Azure Backup", "entity_type_id": 2, "episode_indices": [0]}, {"name": "Azure Import/Export service", "entity_type_id": 2, "episode_indices": [0]}, {"name": "Azure Monitor", "entity_type_id": 3, "episode_indices": [0]}, {"name": "Azure", "entity_type_id": 6, "episode_indices": [0]}, {"name": "Recovery Services vault", "entity_type_id": 2, "episode_indices": [0]}, {"name": "Azure IaaS VMs", "entity_type_id": 4, "episode_indices": [0]}, {"name": "application-consistent backups", "entity_type_id": 5, "episode_indices": [0]}, {"name": "immutability", "entity_type_id": 5, "episode_indices": [0]}, {"name": "cross-region copy", "entity_type_id": 5, "episode_indices": [0]}, {"name": "instant restore", "entity_type_id": 5, "episode_indices": [0]}, {"name": "synthetic full backup", "entity_type_id": 5, "episode_indices": [0]}, {"name": "changed block tracking", "entity_type_id": 5, "episode_indices": [0]}, {"name": "retention policy", "entity_type_id": 7, "episode_indices": [0]}, {"name": "recovery point", "entity_type_id": 7, "episode_indices": [0]}, {"name": "RPO", "entity_type_id": 7, "episode_indices": [0]}, {"name": "RTO", "entity_type_id": 7, "episode_indices": [0]}, {"name": "3-2-1 rule", "entity_type_id": 7, "episode_indices": [0]}, {"name": "storage classes/tiers", "entity_type_id": 7, "episode_indices": [0]}, {"name": "S3 Standard", "entity_type_id": 7, "episode_indices": [0]}, {"name": "S3 Glacier", "entity_type_id": 7, "episode_indices": [0]}, {"name": "Azure Archive tier", "entity_type_id": 7, "episode_indices": [0]}, {"name": "LRS", "entity_type_id": 7, "episode_indices": [0]}, {"name": "ZRS", "entity_type_id": 7, "episode_indices": [0]}, {"name": "SEC 17a-4", "entity_type_id": 7, "episode_indices": [0]}, {"name": "GDPR", "entity_typ
  [...35 more chars...]
```

---

## `dedupe_nodes.nodes`
* provenance: **captured**
* loss_weight: `0.3066` (basis `full`)  |  response_schema: `NodeResolutions`
* article `6d5aa2c9-7dff-430a-997e-86448604c7d2` chunk `19` episode `5443fd84-7dff-4600-9e84-672154e9bae8`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

**system**

```
You are an entity deduplication assistant. NEVER fabricate entity names or mark distinct entities as duplicates.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS MESSAGES>
[{"content": "[Configure Multi-user authorization using Resource Guard › Configure Multi-user authorization using Resource Guard]\n\n\n### Set up approvers for activating Backup MUA Operator role\n\nBy default, the above setup may not have an approver (and an approval flow requirement) configured in PIM. To ensure that approvers have the **Backup MUA Operator** role for request approval, the Security admin must follow these steps:\n\nNote\n\nIf this isn't configured, any requests will be automatically approved without going through the security admins or a designated approver’s review. More details on this can be found [here](../active-directory/privileged-identity-management/pim-resource-roles-configure-role-settings)\n\n1. In Microsoft Entra PIM, select **Azure Resources** on the left navigation bar and select your Resource Guard.\n2. Go to **Settings** and then go to the **Backup MUA Operator** role.\n\n![Screenshot showing how to add contributor.](/media/6d5aa2c9-7dff-430a-997e-86448604c7d2/ef44aeaa178c.png)\n\n> **Figure:** Screenshot of the 'Roles' page in a management interface, showing a list of available roles such as Contributor, Owner, and Reader, with columns for Role, Modified, and Last updated. The 'Settings' menu item is selected in the left navigation.\n3. Select **Edit** to add the reviewers who must review and approve the activation request for the **Backup MUA Operator** role in case you find that Approvers show None or display incorrect approver(s).\n4. On the **Activation** tab, select **Require approval to activate** and add the approver(s) who need to approve each request.\n5. Select security options, such as multifactor authentication (MFA), Mandating ticket to activate **Backup MUA Operator** role.\n6. Select the appropriate
  [...34498 more chars...]
```

**assistant (target)**

```json
{"entity_resolutions": [{"id": 0, "name": "Contributor role", "duplicate_candidate_id": 12}, {"id": 1, "name": "Backup admin", "duplicate_candidate_id": 15}]}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.3129` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `82ed2eb9-24a3-4322-bccb-958dd88c0f31` chunk `2` episode `0e0dbe61-b552-4921-9a05-716f15529870`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Continuous backups and PITR › Continuous backups and PITR]\n### Aurora\n\nTo enable continuous backup of your Aurora resources, see the steps in the first\n section of this page.\n\nThe procedure to restore an Aurora cluster to a point in time is a [variation of\n the steps to restore a snapshot of an aurora cluster](https://docs.aws.amazon.com/aws-backup/latest/devguide/restoring-aur.html).\n\nWhen you conduct a point in time restore, the console displays a **restore\n time** section. See  *Restoring a continuous backup* further\n down on this page in [Working with Continuous backups](https://docs.aws.amazon.com/aws-backup/latest/devguide/point-in-time-recovery.html#point-in-time-recovery-working-with).\n\n###### Important\n\nAurora continuous backups are supported in vaults protected by AWS Backup Vault Lock,\n and the vault's minimum and maximum retention settings are enforced on the recovery\n point. However, Aurora continuous backups do not support the logically air\\-gapped vault\n feature. To use a logically air\\-gapped vault with Aurora, use periodic snapshot backups\n instead.\n\nThe recovery point objective (RPO) for Aurora continuous backups is typically less\n than 5 minutes, as Aurora copies data to Amazon S3 continuously in the background. Use the\n `LatestRestorableTime` value to determine the most recent point to which you\n can restore.\n\n**Retention periods and backup windows:** When you enable\n or change continuous backup settings for an Aurora cluster, AWS Backup calls\n [`ModifyDBCluster`](https://docs.aws.amazon.com/AmazonRDS/latest/APIReference/API_ModifyDBCluster.html) to apply those changes. This can modify the\n cluster's `PreferredBackupWindow`. If you have other configuration updates\n pending the next m
  [...24650 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "AWS Backup", "target_entity_name": "Amazon EC2", "relation_type": "Supports", "fact": "AWS Backup supports restoring Amazon EC2 instances, during which all Amazon EC2 quotas and configuration restrictions apply.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "AWS Backup Vault Lock", "relation_type": "Provides", "fact": "AWS Backup provides AWS Backup Vault Lock, and if the vault containing Amazon EC2 recovery points has a vault lock, additional security considerations apply.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "Active Directory", "relation_type": "Limits", "fact": "AWS Backup does not automatically join restored Amazon EC2 instances to an Active Directory domain; if the original instance was domain-joined, you must rejoin the restored instance manually or by using automation.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "AWS Backup", "target_entity_name": "AWS Lambda", "relation_type": "IntegratesWith", "fact": "AWS Backup can trigger an AWS Lambda function in response to the restore job completed event to rejoin a restored Amazon EC2 instance to an Active Directory domain.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}]}
```

---

## `extract_nodes.extract_summaries_batch`
* provenance: **captured**
* loss_weight: `0.4523` (basis `full`)  |  response_schema: `SummarizedEntities`
* article `c57950e3-10c0-4815-88b7-f90a2de994c1` chunk `3` episode `224252db-2748-4157-93f8-d8cb75f33d40`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

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
[{"content": "[Azure Backup documentation › Azure Backup documentation]\n\nManage at scale\n-------
  [...21340 more chars...]
```

**assistant (target)**

```json
{"summaries": [{"name": "storage classes/tiers", "summary": "Azure Backup supports long-term retention using the Archive tier. AWS Backup supports tiering backups to cold storage for DynamoDB, with advanced features including cross-Region and cross-account copying and cost allocation tags available to customers onboarding after November 2021. AWS Backup cold storage tiering reduces costs while retaining backups. Azure Backup uses a pay-as-you-use model with automatic storage management and no cost for on-premises storage devices."}, {"name": "recovery point", "summary": "Azure Backup stores recovery points in Recovery Services vaults and Backup vaults, which are online-storage entities in Azure holding backup copies and recovery points. Recovery points are managed with built-in recovery point management. Azure Backup provides application-consistent backups, meaning a recovery point contains all required data to restore the backup copy without additional fixes, reducing restoration time. Vaults allow monitoring of backed-up items including Azure VMs and on-premises machines."}, {"name": "retention policy", "summary": "Azure Backup supports short-term and long-term data retention using Recovery Services vaults. Long-term retention using the Archive tier is available for Azure Virtual Machines. AWS Backup allows creating scheduled backups using backup plans and storing backups in backup vaults. For Amazon FSx file systems, AWS Backup provides unlimited retention options and the ability to create scheduled backups as frequently as every hour. AWS Backup retains backups even after the source file system is deleted, protecting against accidental or malicious deletion."}]}
```

---

## `dedupe_edges.resolve_edge`
* provenance: **captured**
* loss_weight: `2.9471` (basis `full`)  |  response_schema: `EdgeDuplicate`
* article `67a20d54-b82e-49a0-b6b8-2a9a8beb86e2` chunk `4` episode `a45c6d30-bcf0-4568-8c73-04cc4745ee3b`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `new_fact_via_capture`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

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
[{'idx': 0, 'fact': 'Azure Backup supports backing up Azure Files to a storage account.'}, {'idx': 1, 'fact': 'Azure Backup supports vaulted backup for Azure Files.'}, {'idx': 2, 'fact': 'Azure Backup supports Azure Files backup.'}, {'idx': 3, 'fact': 'Azure Backup supports reporting for Azure Files.'}, {'idx': 4, 'fact': 'Azure Backup supports file recovery for Azure Files.'}, {'idx': 5, 'fact': 'Azure Backup supports viewing backup and site recovery jobs and metrics for Azure Files workloads (monitoring and metrics).'}, {'idx': 6, 'fact': 'Azure Backup supports auditing and enforcing Azure Files backup with Azure Policy.'}, {'idx': 7, 'fact': 'Azure Backup supports Azure Storage account (with no file or blobs protected) for governance view of resources not configured for protection.'}]
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
[]
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
Azure Backup supports Azure files as a Mercury workload.
</NEW FACT>

You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - If no duplicates, return an empty 
  [...1077 more chars...]
```

**assistant (target)**

```json
{"duplicate_facts": [], "contradicted_facts": []}
```

---

## `extract_nodes.extract_text`
* provenance: **captured**
* loss_weight: `0.5506` (basis `full`)  |  response_schema: `ExtractedEntities`
* article `5d7d817e-7045-41c1-8b36-faa94a5b2c40` chunk `3` episode `2a193c0a-47fb-402e-b3dc-a02c7427e165`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

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
  [...12063 more chars...]
```

**assistant (target)**

```json
{"extracted_entities": [{"name": "AWS Backup", "entity_type_id": 2, "episode_indices": [0]}, {"name": "AWS Organizations", "entity_type_id": 1, "episode_indices": [0]}, {"name": "AWS Backup console", "entity_type_id": 3, "episode_indices": [0]}, {"name": "UpdateGlobalSettings API", "entity_type_id": 3, "episode_indices": [0]}, {"name": "Amazon EC2", "entity_type_id": 4, "episode_indices": [0]}, {"name": "EC2 Allowed AMIs", "entity_type_id": 7, "episode_indices": [0]}, {"name": "backup:CopyIntoBackupVault", "entity_type_id": 9, "episode_indices": [0]}]}
```

---

## `extract_edges.edge`
* provenance: **captured**
* loss_weight: `0.3129` (basis `full`)  |  response_schema: `ExtractedEdges`
* article `a6481417-ece0-481d-a73e-c4279e5eec5f` chunk `1` episode `c7a2e028-d018-45ac-a5da-8273a93d002b`
* captured from `upstage/solar-pro4` (cheap tier), finish_reason `stop`, attribution route `episode_content`, target_normalised `True`
* fidelity: provenance=captured: this is the string upstage/solar-pro4 actually emitted for this exact prompt, recovered by usage.capture_llm_call. No part of the prompt or the target is reconstructed.
* fidelity: the completion passed its RESPONSE_MODELS schema AND the semantic checks in validate_capture (prompt-relative index ranges, endpoint names drawn from the ENTITIES list, a terminal finish_reason). Completions that failed were dropped, never repaired.

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
[{"content": "[Azure Backup architecture › Azure Backup architecture]\n\nBackup features\n---------------\n\nThe following table summarizes the supported features for the different types of backup:\n\n| **Feature** | **Direct Backup of Files and Folders (using MARS Agent)** | **Azure VM Backup** | **Machines or apps with DPM/MABS** |\n| --- | --- | --- | --- |\n| Back up to vault | Yes | Yes | Yes |\n| Back up to DPM/MABS disk, then to Azure |  |  | Yes |\n| Compress data sent for backup | Yes | No compression is used when transferring data. Storage is inflated slightly, but restoration is faster. | Yes |\n| Run incremental backup | Yes | Yes | Yes |\n| Back up deduplicated disks |  |  | Partially For DPM/MABS servers deployed on\\-premises only. |\n\n![Table key](/media/e845095d-d22b-47c1-90e2-c60629331d66/610e32670998.png)\n\n", "timestamp": "2026-07-14T21:54:55.336092+00:00"}, {"content": "[Azure Backup architecture › Azure Backup architecture]\nBackup policy essentials\n------------------------\n\n* A backup policy is created per vault.\n* A backup policy can be created for the backup of following workloads: Azure VMs, SQL in Azure VMs, SAP HANA in Azure VMs and Azure file shares. The policy for files and folder backup using the MARS agent is specified in the MARS console.\n\t+ Azure File Share\n* A policy can be assigned to many resources. An Azure VM backup policy can be used to protect many Azure VMs.\n* A policy consists of two components\n\t+ Schedule: When to take the backup\n\t+ Retention: For how long each backup should be retained.\n* Schedule can be defined as \"daily\" or \"weekly\" with a specific point of time.\n* Retention can be defined for \"daily\", \"weekly\", \"monthly\", \"yearly\" backup points.\n\t+ \"weekly\" refers to a b
  [...31537 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "Azure Backup", "target_entity_name": "Multifactor authentication (MFA)", "relation_type": "Provides", "fact": "Azure Backup provides Multifactor authentication (MFA), which adds an extra layer of security by requiring more verification during sign-in to reduce the risk of unauthorized access.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "Azure role-based access control (Azure RBAC)", "relation_type": "Provides", "fact": "Azure Backup provides Azure role-based access control (Azure RBAC), which lets you define who can access Azure resources, what actions they can perform, and the scope of that access to enforce least-privilege permissions.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "Multiuser authorization (MUA)", "relation_type": "Provides", "fact": "Azure Backup provides Multiuser authorization (MUA), which adds an extra layer of protection for critical operations on Recovery Services vaults and Backup vaults by requiring explicit approval from a security administrator.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Multiuser authorization (MUA)", "target_entity_name": "Recovery Services vaults", "relation_type": "AppliesTo", "fact": "Multiuser authorization (MUA) adds an extra layer of protection for critical operations on Recovery Services vaults.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Multiuser authorization (MUA)", "target_entity_name": "Backup vaults", "relation_type": "AppliesTo", "fact": "Multiuser authorization (MUA) adds an extra layer of protection for critical operations o
  [...82 more chars...]
```

