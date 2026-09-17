# Sample records — readable digest
Generated from `docs/proposals/samples/sample.jsonl` (20 records). This file is a
REVIEW AID: the `<PREVIOUS_MESSAGES>` / `<MESSAGES>` blocks are elided and long
bodies are truncated. The JSONL is the actual deliverable and is verbatim.

Rebuild both with:

```
uv run --extra dev python scripts/build_extraction_dataset.py \
    --out docs/proposals/samples --limit 20 --sample-only
```

---

## `extract_nodes.extract_text`
* provenance: **graph_derived**
* loss_weight: `0.0739`  |  response_schema: `ExtractedEntities`
* article `85ca0827-4a02-4e0b-aca9-481508856cf5` chunk `12`
* fidelity: names are POST-dedup canonical names; where dedup merged an extracted name into an existing node, the string the model emitted is not recoverable
* fidelity: entities dropped by the noise filter or by graphiti's empty-name filter are absent from the target

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
- Bare generic objects that cannot be meaningfully qualifie
  [...13696 more chars...]
```

**assistant (target)**

```json
{"extracted_entities": [{"name": "AWS Backup", "entity_type_id": 2, "episode_indices": [0]}, {"name": "AWS Backup", "entity_type_id": 2, "episode_indices": [0]}, {"name": "AWS Command Line Interface", "entity_type_id": 3, "episode_indices": [0]}, {"name": "AWS services", "entity_type_id": 4, "episode_indices": [0]}, {"name": "AWS", "entity_type_id": 6, "episode_indices": [0]}, {"name": "cross-region backup", "entity_type_id": 5, "episode_indices": [0]}, {"name": "AWS Regions", "entity_type_id": 8, "episode_indices": [0]}, {"name": "AWS Organizations", "entity_type_id": 1, "episode_indices": [0]}, {"name": "Amazon EventBridge", "entity_type_id": 1, "episode_indices": [0]}, {"name": "Amazon EBS", "entity_type_id": 4, "episode_indices": [0]}, {"name": "Amazon Simple Storage Service (Amazon S3)", "entity_type_id": 1, "episode_indices": [0]}, {"name": "Amazon Aurora DSQL", "entity_type_id": 1
  [...2787 more chars...]
```

---

## `extract_edges.edge`
* provenance: **graph_derived**
* loss_weight: `0.0402`  |  response_schema: `ExtractedEdges`
* article `d386db8d-41e7-4d5e-bdcb-d933eb2c7c9d` chunk `9`
* fidelity: facts the model emitted that were then merged into an existing edge appear here with the SURVIVING edge's text, not the text the model produced
* fidelity: valid_at is set deterministically from the episode reference time by deterministic_valid_at when that patch is enabled, so it is not always a model-produced value

**system**

```
You are an expert fact extractor that extracts fact triples from text. 1. Extracted fact triples should also be extracted with relevant date information. 2. The CURRENT_MESSAGE may contain multiple episodes, each with its own timestamp. Use each episode's timestamp to resolve temporal references within that episode. REFERENCE_TIME is a fallback for when no per-episode timestamp is available.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS_MESSAGES>
  [...previous-episode context elided for readability...]
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
[Review security posture › Review security posture]


On the **Security level details** blade, the help provides guidance on the various security levels and the settings that are required to meet each level.

[

![Screenshot shows the security levels details selection.](/media/d386db8d-41e7-4d5e-bdcb-d933eb2c7c9d/a3ab37f25305.png)

> **Figure:** Screenshot of the 'Security posture' page in Azure Resiliency, showing the 'Security level details' dialog. The dialog explains security level values (Excellent, Good, Fair, Poor) based on Azure Backup settings, including immutability, soft-delete, and MUA requirements. The main page displays BCDR security coverage for Azure managed Active resources.

](media/tutorial-review-security-posture/select-security-level-details.png#lightbox)


Modify security level
---------------------

In Resiliency in Azure, you can change the security level for a protected item.

To modify the security level for an item, follow these steps:

</C
  [...11033 more chars...]
```

**assistant (target)**

```json
{"edges": [{"source_entity_name": "Azure Backup", "target_entity_name": "immutability", "relation_type": "Provides", "fact": "Azure Backup provides immutability to prevent data from being altered or deleted before its retention period.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "Resiliency", "relation_type": "IntegratesWith", "fact": "Azure Backup integrates with Resiliency in Azure, a unified platform for Zone Resiliency, High Availability, Backup & Disaster Recovery, and Ransomware Protection.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"source_entity_name": "Azure Backup", "target_entity_name": "soft delete", "relation_type": "Provides", "fact": "Azure Backup provides Soft Delete as a security setting for a vault.", "valid_at": null, "invalid_at": null, "episode_indices": [0]}, {"s
  [...237 more chars...]
```

---

## `dedupe_nodes.nodes`
* provenance: **graph_derived_negative**
* loss_weight: `0.4762`  |  response_schema: `NodeResolutions`
* article `a6481417-ece0-481d-a73e-c4279e5eec5f` chunk `5`
* fidelity: negatives only. These entities became their own nodes, so -1 is the label the pipeline acted on.
* fidelity: POSITIVES ARE ABSENT: when the LLM resolves a duplicate, the extracted name is discarded and only the surviving node remains, so no positive can be graph-derived. A model trained on this split alone will learn to always answer -1.
* fidelity: candidate ORDER is by stored name_embedding cosine; graphiti ranks by hybrid fulltext+vector RRF

**system**

```
You are an entity deduplication assistant. NEVER fabricate entity names or mark distinct entities as duplicates.
Do not escape unicode characters.


Any extracted information should be returned in the same language as it was written in. Only output non-English text when the user has written full sentences or phrases in that non-English language. Otherwise, output English.
```

**user**

```

<PREVIOUS MESSAGES>
  [...previous-episode context elided for readability...]
</PREVIOUS MESSAGES>

<CURRENT MESSAGE>
[Best practices for backup data protection › Best practices for backup data protection]
### Encryption of backup data

Encryption ensures that backup data is protected both at rest and in transit. With Azure Backup, all your backup data is encrypted by default with Microsoft\-managed keys. All the data in vaults is encrypted at rest with 256\-bit Advanced Encryption Standard (AES) encryption compliant with Federal Information Processing Standards (FIPS) 140\-2\. All communication and data transfer over HTTPS and Transport Layer Security (TLS) 1\.2\+ ensuring data in transit is encrypted and remains within Azure Network. Azure Backup also supports backup and restore of Azure Virtual Machines (VMs) that have their OS/data disks encrypted with Azure Disk Encryption (ADE).

To enable control over the encryption key by configuring key rotation and lifecycle or to bring their own keys for encryption, use Customer\-Managed Keys (CMK) for encryption of the backup data. [Lear
  [...14914 more chars...]
```

**assistant (target)**

```json
{"entity_resolutions": [{"id": 0, "name": "Infrastructure encryption", "duplicate_candidate_id": -1}, {"id": 1, "name": "Azure fabric", "duplicate_candidate_id": -1}, {"id": 2, "name": "Data redundancy", "duplicate_candidate_id": -1}]}
```

---

## `dedupe_edges.resolve_edge`
* provenance: **synthetic_alias_positive**
* loss_weight: `2.3017`  |  response_schema: `EdgeDuplicate`
* article `d4d25628-1f6e-42ea-a5c5-e66b2e1bf2f8` chunk `1`
* fidelity: NEW FACT was generated by the rule 'Amazon S3 -> S3 buckets' from ontology.py's canonical-name list. No model emitted it. Review before training.
* fidelity: positives cannot be graph-derived: graphiti discards the merged-away text and keeps the existing edge's fact

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
[{'idx': 0, 'fact': 'AWS Backup restore destination bucket must have versioning turned on; AWS Backup notifies you if the bucket you select does not meet this requirement.'}, {'idx': 1, 'fact': 'AWS Backup supports Amazon S3 continuous recovery points, where changing the retention period causes the recovery point status to become STOPPED and a new continuous recovery point with the altered retention settings is created.'}, {'idx': 2, 'fact': 'AWS Backup does not support using the original encryption method to restore a recovery point copied from another Region for Amazon S3 resources; cross-Region copy AWS KMS encryption
  [...2647 more chars...]
```

**assistant (target)**

```json
{"duplicate_facts": [4], "contradicted_facts": []}
```

---

## `extract_nodes.extract_summaries_batch`
* provenance: **graph_derived**
* loss_weight: `0.0889`  |  response_schema: `SummarizedEntities`
* article `f7fa671c-fad5-4557-aa52-cfd6264a56d7` chunk `3`
* fidelity: WEAKEST target in this set: Entity.summary is the FINAL summary after every later episode updated it, not the summary produced at this episode. Existing-summary state at call time is also unrecoverable, so the prompt shows an empty prior summary.
* fidelity: the >2*MAX_SUMMARY_CHARS gate that decides whether the LLM is called at all (node_operations.py:880-882) is replayed here against the FINAL summary, so a few of these calls may not have happened in production

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
        8. Never use filler verbs like "mentioned", "described", "stated", "reported", "noted", "discussed
  [...4177 more chars...]
```

**assistant (target)**

```json
{"summaries": [{"name": "Recovery Services vault", "summary": "Recovery Services vault stores backup data and recovery points and is the Recovery Services resource used by Azure Backup to manage backups and restores. Create, enable, modify, and delete backup policies and stop/disable protection require access to the vault. Soft delete retains backup data for 14 days. Vaults support CMK (customer-managed keys) and PMK (platform-managed keys) for encryption at rest; infrastructure-level encryption can provide double encryption. Cross Subscription Restore must be enabled on the vault to restore vault-tier recovery points to subscriptions other than the protected VM or vault subscription. Resource Guard enables Multi-User Authorization (MUA) to require authorization for critical operations such as disabling soft delete, stopping/deleting backups, or reducing retention. Resource Guard is used
  [...1098 more chars...]
```
