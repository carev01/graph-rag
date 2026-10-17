# Grounded extraction: no fact without its entities in the text — design

**Status:** approved in conversation 2026-10-17 (root cause, ingest fix, proof run). The
re-extraction of affected articles is on hold until the user approves it.

## 1. What went wrong (traced 2026-10-17)

"AWS Backup supports Azure Blob Storage" was cited in an AWS Backup list answer. Its two
supporting episodes show two extraction failures and no dedup failure:

- **Forced subject.** "Access verification for self-hosted connectors" (Veeam / Securiti.ai) is
  a 707-character list of connectors with no product named. Extraction produced 37 facts, all
  "AWS Backup supports <connector>". `EXTRACTION_INSTRUCTIONS` require a backup PRODUCT as the
  subject of EVERY chunk "even when … only implied", and their first example is "AWS Backup".
- **Example leakage.** "Controls and remediation" (AWS Backup) is a short Security Hub control.
  Extraction produced 40 facts, most absent from the text: "AWS Backup supports Hyper-V / Azure VM
  / VMware vSphere as a workload", "Azure Archive tier", "ZRS", "LRS", "East US", "backup
  frequency", "instant restore", "deduplication", "air gap". This is the instructions' own
  canonical-name and type-boundary vocabulary.
- **Dedup merged identical inventions.** Both chunks invented the same fact text, and the edge
  dedup correctly merged two identical statements.
- **Not the cause:** graphiti's previous-episode context. It held each article's own sibling
  chunks (chunks share the article's reference time).

Scale, measured on the live graph:

- 19,026 of 122,268 facts whose subject is a structural product name (16%) have at least one
  supporting episode whose article belongs to a different product and whose text does not name
  the subject.
- The largest groups are the instructions' example products: "AWS Backup" in Veeam B&R (2,022
  links), Securiti.ai (1,159) and Veeam Data Cloud (377) articles, and "Azure Backup" in VBR
  (775) and VDC (505).
- About 3,300 of 20,900 ingested articles are affected (Veeam 3,126; Microsoft 142; Cohesity 27;
  AWS 11).

## 2. Fix at ingest

1. **Document context per episode.** Ingest knows each article's vendor, product and document
   set from the structural chain (Vendor→Product→Source→Article). Every `add_episode` call
   appends a context block to the extraction instructions:
   `This chunk comes from {vendor} documentation for {product} (document set: {source}).`
   followed by the subject rule: use exactly "{product}" when a fact's subject is the documented
   product and the text only implies it; never make any other product a subject unless this
   chunk's text names it.
2. **Instructions that cannot leak.**
   - Remove every example product name.
   - Rewrite the canonical-name lists as naming rules ("if the text says K8s, name it
     Kubernetes").
   - Add: "Extract only entities and facts that this chunk's text states. The rules below say
     how to NAME something the text mentions; never add an entity because it appears in these
     instructions."
   - Keep the type boundaries and the limitation/availability guidance, reworded without
     example entity names where they could leak.
3. **Deterministic grounding gate.** The wrapper that already intercepts graphiti's LLM calls
   (`install_dedup_guard`) also filters the `extract_nodes.extract_text` reply, keeping an
   extracted entity only when one of these holds:
   - its name occurs in the chunk text, case-insensitive;
   - at least ceil(2n/3) of its n distinctive words occur in the text as word prefixes (a word
     or a simple inflection: s/es/ed/ing; distinctive = length >= 2 and not a stopword);
   - it is the document's product or vendor (the implied subject).

   The chunk text and document names come from a ContextVar that ingest sets around each
   `add_episode` call. With no context open (other callers), the gate is a no-op and counts the
   call as unscoped. Dropped entities are counted per article (`grounding_dropped`) and listed
   in a DEBUG log, and their names are logged at INFO with a cap. Graphiti then extracts edges
   only among the kept entities, so no fact can attach to an invented entity.
4. **Restore destinations as relations (user request 2026-10-17).** Restore statements had no
   relation type, so "VBR can restore physical machines from backups created by Veeam Agents to
   HPE Morpheus" became "VBR Supports Veeam Agents", with Morpheus only in the text. Restore
   facts also sat under invented subjects ("AWS Backup supports restore of EC2 instances … to
   Nutanix AHV"), and Morpheus is split over 14 nodes. A new edge type, **RestoresTo**, covers
   this: the backups of a product, capability or workload can be restored, recovered or
   migrated to a destination platform or workload. The destination is the target entity, and
   the backup source stays in the fact text. Allowed for (Product|Capability|Workload,
   Platform|Workload). The instructions say: for a restore, recovery or migration statement,
   the destination is the target and the relation is RestoresTo.
5. **Type descriptions without examples.** The entity and edge type docstrings graphiti puts
   into the prompt name concrete entities: "AWS Backup, Azure Backup, Veeam Backup &
   Replication, NetBackup", "Amazon S3, Azure VM", "LRS/ZRS", "Azure Archive tier",
   "SEC 17a-4", "East US". Several of the AWS article's invented facts are exactly these. The
   descriptions are rewritten to define each type in words, with no named entities.
6. **Report.** The per-article ingest result and the worker batch line carry
   `grounding_dropped` next to the dedup counters.

## 3. Proof (approved, ~$0.30)

A script re-extracts a fixed sample through the production ingest path into a throwaway Neo4j
testcontainer, with the structural chain copied read-only from production. The sample is:

- the two articles above;
- the HPE Morpheus VM Essentials page and three agent-restore pages (Scale Computing HyperCore,
  Proxmox VE, Sangfor aSV);
- 20 affected articles: 10 with an example-product subject, 10 other.

The script reports, per article, the facts written, `grounding_dropped`, and the count of facts
whose subject is a product not named in the text and not the article's product, compared with
production. It refuses to report success unless the gate dropped at least one entity on the
known-bad chunks.

Pass criteria:

- no fabricated foreign-product subject on the two known articles;
- foreign-unnamed-subject facts in the sample cut by at least 90%;
- facts per article not reduced by more than 30% on the articles production got right
  (spot-checked);
- each of the four restore pages yields at least one RestoresTo fact whose target is its
  destination (Morpheus, HyperCore, Proxmox, Sangfor).

## 4. Out of scope (held)

Re-extraction of the ~3,300 affected articles through the update path, which costs about $35
and is held for the user's approval. Read-time source filtering is not built.
