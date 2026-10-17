# Extraction grounding proof — 2026-10-17

The ingest fix (spec 2026-10-17-extraction-grounding-design.md) was re-run on a throwaway Neo4j
through the production ingest path: 26 articles, cheap and strong tiers, about $0.36 (plus about
$0.08 of smokes).

**Verdict: every §3 pass criterion met.**

| criterion | target | result |
|---|---|---|
| fabricated foreign-product subjects on the two known articles | none | 37 -> 0 (Securiti list), 0 -> 0 (AWS control, whose ~40 invented vocabulary facts are gone: 92 -> 60 facts) |
| foreign-unnamed-subject facts across the sample | cut >= 90% | 144 -> 5 (-96.5%); the 5 left are Veeam B&R capabilities described inside Service Provider Console docs |
| facts on articles production got right | not down more than 30% | 152 -> 134 (-11.8%) |
| RestoresTo to each restore page's destination | at least one each | HPE Morpheus (Morpheus page: 37 RestoresTo, incl. -> HPE Morpheus VM Essentials), Scale Computing HyperCore, Proxmox VE, Sangfor aSV |
| overall facts | -- | 869 -> 947 |

How the result came about (smokes 1-3):

1. The rewritten instructions and the per-chunk DOCUMENT CONTEXT alone stopped the inventions.
   The grounding gate saw every extraction reply (129 calls) and dropped only 5 entities.
2. A subjectless chunk (the Securiti connector list) lost its facts at first. The cheap model
   never emits the documented product as a node, so graphiti discards every edge whose subject
   is that product.
3. The fix is deterministic: the gate supplies the documented product entity when the reply
   omits it. It did so 67 times in the sample, and the Securiti article went 9 -> 94 facts.

Caveats:

- Content was fetched live from DocExtractor, so it may differ slightly from what production
  extracted.
- The proof graph starts empty, so entity dedup has less to merge than production.
- The cost estimate uses `haiku_extract_ab.RATES`.

The raw report follows.

# Extraction grounding proof (throwaway Neo4j)

- Run: 2026-10-17T16:55:47+00:00; commit `0ec790f`; test image `neo4j:2026.07.1-community`
- Articles: 26 (affected-example 10, affected-other 10, requested 6)
- Gate verdict: fired (dropped entities on 2 of 21 articles production got wrong)
- Failed articles: 0; excluded from totals: 0

## Totals (over 26 of 26 articles: the successfully ingested ones)

| | production | test | |
|---|---|---|---|
| facts | 869 | 947 | |
| foreign-unnamed-subject facts | 144 | 5 | reduction 96.5% |
| facts on articles production got right | 152 | 134 | change -11.8% |

included articles=26 grounding_calls=129 grounding_dropped=5 grounding_injected=67 episodes_added=129

Spend (every article run, included or not): prompt_tokens=2570250 cached_tokens=199872 completion_tokens=189396 estimated cost $0.357 (haiku_extract_ab.RATES at each article's routed tier; an estimate)

## Excluded from totals

- none

## Per article

| article | title | group | product | facts prod -> test | foreign prod -> test | grounding calls / dropped / injected | RestoresTo | tier | est. $ | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `00f16062-a6fb-481b-82d4-81fab26cd45d` | Access verification for self-hosted connectors | requested | Securiti.ai | 90 -> 94 | 37 -> 0 | 4 / 0 / 4 | 0 | cheap | 0.0079 | ok |
| `ec64eae8-3e67-4607-a635-614bfec1876b` | Controls and remediation | requested | AWS Backup | 92 -> 60 | 0 -> 0 | 13 / 0 / 0 | 0 | cheap | 0.0142 | ok |
| `a3a8692c-169f-4559-9dc2-5ccd637bd2df` | HPE Morpheus VM Essentials | requested | Veeam Backup & Replication | 35 -> 43 | 0 -> 0 | 1 / 0 / 0 | 37 | cheap | 0.0038 | ok |
| `aac21e13-57c2-48ae-88cc-a4518a9958f1` | Restoring Veeam Agent Backup to Scale Computing HyperCore | requested | Veeam Backup & Replication | 3 -> 4 | 0 -> 0 | 1 / 0 / 0 | 1 | cheap | 0.0020 | ok |
| `ce76e11c-c4bd-4bb0-82a3-18996c88eedf` | Restoring Veeam Agent Backup to Proxmox VM | requested | Veeam Backup & Replication | 5 -> 7 | 0 -> 0 | 1 / 0 / 0 | 1 | cheap | 0.0021 | ok |
| `03e7863b-338b-46f0-a196-cf203b63b0ee` | Restore to Sangfor aSV | requested | Veeam Backup & Replication | 17 -> 20 | 0 -> 0 | 2 / 0 / 0 | 17 | cheap | 0.0042 | ok |
| `0009d664-6b57-4dff-b2ca-b1a7c8eab75a` | IAM Permissions | affected-example | Veeam Backup & Replication | 11 -> 15 | 9 -> 0 | 2 / 0 / 2 | 0 | cheap | 0.0047 | ok |
| `002f473d-ce45-4fab-aeff-3909110736f1` | Alarms | affected-example | Veeam Service Provider Console | 86 -> 106 | 11 -> 1 | 10 / 0 / 3 | 0 | strong | 0.0893 | ok |
| `005239a6-c37f-4848-8fc9-8f4c0505177c` | Step 5. Specify VM Name and Resource Group | affected-example | Veeam Backup & Replication | 6 -> 5 | 5 -> 0 | 2 / 0 / 2 | 2 | cheap | 0.0033 | ok |
| `006a1dc3-240c-49aa-b005-1d2cfb1de15e` | Configuring Settings for DB Instances | affected-example | Veeam Backup & Replication | 9 -> 10 | 9 -> 0 | 2 / 1 / 1 | 0 | cheap | 0.0050 | ok |
| `00a7079a-8b01-4d1e-b84b-081a64a10747` | VPC Configuration Restore IAM Permissions | affected-example | Veeam Backup & Replication | 14 -> 49 | 12 -> 0 | 6 / 4 / 6 | 0 | cheap | 0.0125 | ok |
| `00af62cc-01a0-49bb-b17f-0eb5fc7dbb4a` | Step 3. Select Objects to Back Up | affected-example | Veeam Service Provider Console | 7 -> 9 | 4 -> 4 | 2 / 0 / 0 | 0 | cheap | 0.0042 | ok |
| `00c14394-727f-4873-aa0d-69e03d266ef4` | Job Configuration Change Tracking | affected-example | Veeam ONE | 16 -> 19 | 13 -> 0 | 2 / 0 / 2 | 0 | cheap | 0.0033 | ok |
| `00d4564d-1d6d-4a53-a0fa-c68c51da357d` | Step 7. Select Report Template | affected-example | Veeam Recovery Orchestrator | 2 -> 1 | 2 -> 0 | 1 / 0 / 0 | 0 | cheap | 0.0016 | ok |
| `00d6f34f-020f-475d-9298-880c987d5e78` | Step 1. Launch New Azure Virtual Machine Policy Wizard | affected-example | Veeam Data Cloud | 2 -> 2 | 2 -> 0 | 1 / 0 / 0 | 0 | cheap | 0.0017 | ok |
| `01033b22-6be1-460e-b3c4-b72586dd7b61` | Security Settings | affected-example | Veeam Backup for Microsoft 365 | 14 -> 6 | 1 -> 0 | 3 / 0 / 1 | 0 | strong | 0.0118 | ok |
| `0338b83a-7347-408e-b99b-daa78c5aee3f` | Restoring Document Libraries and Lists to Another Location | affected-other | Veeam Backup & Replication | 3 -> 3 | 3 -> 0 | 1 / 0 / 1 | 3 | cheap | 0.0025 | ok |
| `15ce38bb-97f1-429a-8845-eef950425d82` | Searching for Objects in Backup | affected-other | Veeam Data Cloud | 24 -> 40 | 4 -> 0 | 13 / 0 / 9 | 1 | cheap | 0.0210 | ok |
| `2e0e1e10-704c-4aba-bcda-21eb79865a24` | Viewing Backup Sessions | affected-other | Veeam Data Cloud | 27 -> 24 | 6 -> 0 | 5 / 0 / 1 | 0 | cheap | 0.0116 | ok |
| `3bcc4e5d-b834-4d16-b0f3-67673d2743e6` | Editing Flex Backup Policies | affected-other | Veeam Data Cloud | 49 -> 48 | 8 -> 0 | 7 / 0 / 3 | 0 | cheap | 0.0162 | ok |
| `429eda3e-f872-40c6-9c68-bc7dba25247d` | Creating Backup Jobs | affected-other | Veeam Service Provider Console | 1 -> 1 | 1 -> 0 | 2 / 0 / 1 | 0 | cheap | 0.0014 | ok |
| `6609e659-70a1-499b-9a94-a08a1bc3b925` | Restoring Outlook Mailboxes | affected-other | Veeam Data Cloud | 67 -> 60 | 1 -> 0 | 14 / 0 / 10 | 24 | cheap | 0.0272 | ok |
| `6eba97cf-6ffa-42ab-b0ca-11aca136f210` | Step 7. Specify Rates for Microsoft 365 Services | affected-other | Veeam Service Provider Console | 18 -> 13 | 7 -> 0 | 4 / 0 / 0 | 0 | cheap | 0.0064 | ok |
| `9bbd3e1a-34a8-44de-a179-b04b74661b63` | Considerations and Limitations | affected-other | Veeam Data Cloud | 225 -> 268 | 5 -> 0 | 16 / 0 / 13 | 12 | cheap | 0.0750 | ok |
| `a1443af5-354f-4e62-9963-5bc7386695fc` | Microsoft 365 Backup | affected-other | Microsoft 365 Backup | 2 -> 0 | 2 -> 0 | 1 / 0 / 0 | 0 | cheap | 0.0018 | ok |
| `c1b370f4-50b1-4a6c-aec5-bd891aab5779` | Searching for Objects in Backup | affected-other | Veeam Data Cloud | 44 -> 40 | 2 -> 0 | 13 / 0 / 8 | 3 | cheap | 0.0220 | ok |

## Details

### Access verification for self-hosted connectors (`00f16062-a6fb-481b-82d4-81fab26cd45d`)

- Veeam / Securiti.ai; requested; episodes_added=4; tokens 47428 in (11968 cached) / 12558 out; est. $0.0079
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: AWS Backup x37
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### Controls and remediation (`ec64eae8-3e67-4607-a635-614bfec1876b`)

- AWS / AWS Backup; requested; episodes_added=13; tokens 160237 in (46784 cached) / 8816 out; est. $0.0142
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: none
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### HPE Morpheus VM Essentials (`a3a8692c-169f-4559-9dc2-5ccd637bd2df`)

- Veeam / Veeam Backup & Replication; requested; episodes_added=1; tokens 23404 in (1088 cached) / 4909 out; est. $0.0038
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: none
- test foreign-unnamed facts (0):
- RestoresTo facts written (37):
  - Veeam Backup & Replication -> Sangfor aSV
  - Veeam Backup & Replication -> XCP-ng
  - Veeam Backup & Replication -> Amazon Web Services (AWS)
  - Veeam Backup & Replication -> Proxmox VE
  - Veeam Backup & Replication -> Nutanix AHV
  - Veeam Backup & Replication -> Scale Computing HyperCore
  - Veeam Backup & Replication -> Microsoft Azure
  - Veeam Backup & Replication -> Citrix XenServer
  - Veeam Backup & Replication -> Veeam Agents
  - Veeam Backup & Replication -> Microsoft SharePoint
  - Veeam Backup & Replication -> Nutanix AHV
  - Veeam Backup & Replication -> Microsoft Hyper-V
  - Veeam Backup & Replication -> Veeam Plug-in for Universal Hypervisor API
  - Veeam Backup & Replication -> Microsoft DNS
  - Veeam Backup & Replication -> Microsoft SQL Server
  - Veeam Backup & Replication -> Oracle Database
  - Veeam Backup & Replication -> Google Cloud
  - Veeam Backup & Replication -> Nutanix AHV
  - Veeam Backup & Replication -> Google Cloud
  - Veeam Backup & Replication -> Scale Computing HyperCore
  - Veeam Backup & Replication -> Microsoft Active Directory
  - Veeam Backup & Replication -> HPE Morpheus VM Essentials
  - Veeam Backup & Replication -> HPE Morpheus VM Essentials
  - Veeam Backup & Replication -> PostgreSQL
  - Veeam Backup & Replication -> Proxmox VE
  - Veeam Backup & Replication -> Sangfor aSV
  - Veeam Backup & Replication -> Amazon Web Services (AWS)
  - Veeam Backup & Replication -> VMware vSphere
  - Veeam Backup & Replication -> Microsoft Hyper-V
  - Veeam Backup & Replication -> oVirt KVM
  - Veeam Backup & Replication -> Microsoft Exchange
  - Veeam Backup & Replication -> Proxmox VE
  - Veeam Backup & Replication -> oVirt KVM
  - Veeam Backup & Replication -> XCP-ng
  - Veeam Backup & Replication -> Citrix XenServer
  - Veeam Backup & Replication -> VMware vSphere
  - Veeam Backup & Replication -> Microsoft Azure

### Restoring Veeam Agent Backup to Scale Computing HyperCore (`aac21e13-57c2-48ae-88cc-a4518a9958f1`)

- Veeam / Veeam Backup & Replication; requested; episodes_added=1; tokens 19878 in (1088 cached) / 754 out; est. $0.0020
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: none
- test foreign-unnamed facts (0):
- RestoresTo facts written (1):
  - Veeam Backup & Replication -> Scale Computing HyperCore

### Restoring Veeam Agent Backup to Proxmox VM (`ce76e11c-c4bd-4bb0-82a3-18996c88eedf`)

- Veeam / Veeam Backup & Replication; requested; episodes_added=1; tokens 21312 in (5440 cached) / 1457 out; est. $0.0021
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: none
- test foreign-unnamed facts (0):
- RestoresTo facts written (1):
  - Veeam Backup & Replication -> Proxmox VE

### Restore to Sangfor aSV (`03e7863b-338b-46f0-a196-cf203b63b0ee`)

- Veeam / Veeam Backup & Replication; requested; episodes_added=2; tokens 36648 in (2176 cached) / 3033 out; est. $0.0042
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: none
- test foreign-unnamed facts (0):
- RestoresTo facts written (17):
  - Veeam Plug-in for Xen -> Xen
  - Veeam Plug-In for Nutanix AHV -> Nutanix AHV
  - Veeam Backup & Replication -> Sangfor aSV
  - Veeam Backup for Google Cloud -> Google Compute Engine
  - Veeam Plug-In for Proxmox VE -> Proxmox VE
  - Veeam Backup for Microsoft Azure -> Microsoft Azure
  - Veeam Plug-in for Scale Computing HyperCore -> Scale Computing HyperCore
  - Veeam Backup & Replication -> VMware Cloud Director
  - Veeam Plug-In for oVirt KVM -> oVirt
  - Veeam Agent for Linux -> Sangfor aSV
  - Veeam Backup & Replication -> VMware vSphere
  - Veeam Plug-in for Sangfor aSV -> Sangfor aSV
  - Veeam Plug-in for HPE Morpheus VM Essentials -> HPE Morpheus VM Essentials
  - Veeam Agent for Microsoft Windows -> Sangfor aSV
  - Veeam Backup & Replication -> Microsoft Hyper-V
  - Veeam Backup for AWS -> Amazon EC2
  - Veeam Backup & Replication -> Sangfor aSV

### IAM Permissions (`0009d664-6b57-4dff-b2ca-b1a7c8eab75a`)

- Veeam / Veeam Backup & Replication; affected-example; episodes_added=2; tokens 39332 in (2176 cached) / 3562 out; est. $0.0047
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: AWS Backup x9
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### Alarms (`002f473d-ce45-4fab-aeff-3909110736f1`)

- Veeam / Veeam Service Provider Console; affected-example; episodes_added=10; tokens 220665 in (11136 cached) / 18334 out; est. $0.0893
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup & Replication x11
- test foreign-unnamed facts (1):
  - Veeam Backup & Replication: Veeam Service Provider Console checks whether Veeam Backup & Replication or Veeam Cloud Connect server version is no longer supported and the server requires upgrade (version is out-of-date alarm).
- RestoresTo facts written (0):

### Step 5. Specify VM Name and Resource Group (`005239a6-c37f-4848-8fc9-8f4c0505177c`)

- Veeam / Veeam Backup & Replication; affected-example; episodes_added=2; tokens 35379 in (2176 cached) / 712 out; est. $0.0033
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Azure Backup x5
- test foreign-unnamed facts (0):
- RestoresTo facts written (2):
  - Veeam Backup & Replication -> Azure VM
  - Veeam Backup & Replication -> Resource group

### Configuring Settings for DB Instances (`006a1dc3-240c-49aa-b005-1d2cfb1de15e`)

- Veeam / Veeam Backup & Replication; affected-example; episodes_added=2; tokens 47419 in (2176 cached) / 2331 out; est. $0.0050
- dropped (first 20 of 1): Amazon RDS
- production foreign-unnamed subjects: AWS Backup x9
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### VPC Configuration Restore IAM Permissions (`00a7079a-8b01-4d1e-b84b-081a64a10747`)

- Veeam / Veeam Backup & Replication; affected-example; episodes_added=6; tokens 111185 in (6528 cached) / 8327 out; est. $0.0125
- dropped (first 20 of 4): Amazon EC2, Amazon EC2, Amazon EC2, Elastic Load Balancing
- production foreign-unnamed subjects: AWS Backup x12
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### Step 3. Select Objects to Back Up (`00af62cc-01a0-49bb-b17f-0eb5fc7dbb4a`)

- Veeam / Veeam Service Provider Console; affected-example; episodes_added=2; tokens 43187 in (2176 cached) / 1443 out; est. $0.0042
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup & Replication x4
- test foreign-unnamed facts (4):
  - Veeam Backup & Replication: Veeam Backup & Replication can back up separate VMs.
  - Veeam Backup & Replication: Veeam Backup & Replication can back up separate vApps and automatically update backup job settings to include new VMs added to the vApp.
  - Veeam Backup & Replication: Veeam Backup & Replication can back up virtual datacenters and automatically update backup job settings to include new VMs added to the datacenter.
  - Veeam Backup & Replication: Veeam Backup & Replication can back up VMware Cloud Director organizations and automatically update backup job settings to include new VMs added to the organization.
- RestoresTo facts written (0):

### Job Configuration Change Tracking (`00c14394-727f-4873-aa0d-69e03d266ef4`)

- Veeam / Veeam ONE; affected-example; episodes_added=2; tokens 31328 in (4352 cached) / 2327 out; est. $0.0033
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup & Replication x13
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### Step 7. Select Report Template (`00d4564d-1d6d-4a53-a0fa-c68c51da357d`)

- Veeam / Veeam Recovery Orchestrator; affected-example; episodes_added=1; tokens 18177 in (1088 cached) / 233 out; est. $0.0016
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup & Replication x2
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### Step 1. Launch New Azure Virtual Machine Policy Wizard (`00d6f34f-020f-475d-9298-880c987d5e78`)

- Veeam / Veeam Data Cloud; affected-example; episodes_added=1; tokens 18237 in (1088 cached) / 313 out; est. $0.0017
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Azure Backup x2
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### Security Settings (`01033b22-6be1-460e-b3c4-b72586dd7b61`)

- Veeam / Veeam Backup for Microsoft 365; affected-example; episodes_added=3; tokens 40737 in (2688 cached) / 1094 out; est. $0.0118
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Azure Backup x1
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### Restoring Document Libraries and Lists to Another Location (`0338b83a-7347-408e-b99b-daa78c5aee3f`)

- Veeam / Veeam Backup & Replication; affected-other; episodes_added=1; tokens 23058 in (1088 cached) / 1353 out; est. $0.0025
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup for Microsoft 365 x3
- test foreign-unnamed facts (0):
- RestoresTo facts written (3):
  - Veeam Backup & Replication -> Microsoft 365 organizations
  - Veeam Backup & Replication -> Microsoft SharePoint
  - Veeam Backup & Replication -> on-premises Microsoft organizations

### Searching for Objects in Backup (`15ce38bb-97f1-429a-8845-eef950425d82`)

- Veeam / Veeam Data Cloud; affected-other; episodes_added=13; tokens 212321 in (14144 cached) / 7963 out; est. $0.0210
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup for Microsoft 365 x4
- test foreign-unnamed facts (0):
- RestoresTo facts written (1):
  - Veeam Data Cloud -> Microsoft 365

### Viewing Backup Sessions (`2e0e1e10-704c-4aba-bcda-21eb79865a24`)

- Veeam / Veeam Data Cloud; affected-other; episodes_added=5; tokens 115591 in (9792 cached) / 5145 out; est. $0.0116
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup for Microsoft 365 x6
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### Editing Flex Backup Policies (`3bcc4e5d-b834-4d16-b0f3-67673d2743e6`)

- Veeam / Veeam Data Cloud; affected-other; episodes_added=7; tokens 150738 in (7616 cached) / 8958 out; est. $0.0162
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup for Microsoft 365 x8
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### Creating Backup Jobs (`429eda3e-f872-40c6-9c68-bc7dba25247d`)

- Veeam / Veeam Service Provider Console; affected-other; episodes_added=2; tokens 16314 in (2176 cached) / 271 out; est. $0.0014
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup for Microsoft 365 x1
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### Restoring Outlook Mailboxes (`6609e659-70a1-499b-9a94-a08a1bc3b925`)

- Veeam / Veeam Data Cloud; affected-other; episodes_added=14; tokens 261018 in (25024 cached) / 15225 out; est. $0.0272
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup for Microsoft 365 x1
- test foreign-unnamed facts (0):
- RestoresTo facts written (24):
  - Flex Restore -> Microsoft Outlook
  - Veeam Data Cloud -> Microsoft 365
  - Veeam Data Cloud -> Outlook
  - Express Restore -> Microsoft 365
  - Veeam Data Cloud for Microsoft 365 -> Outlook
  - Express Restore -> Outlook
  - Express Full Restore -> Microsoft Outlook
  - Express Restore -> Outlook
  - Express Full Restore -> Maria Ramos
  - Restore to alternate location -> Outlook
  - Veeam Data Cloud for Microsoft 365 -> Microsoft 365
  - Express Full Restore -> Microsoft 365
  - Veeam Data Cloud -> Microsoft Outlook
  - Veeam Data Cloud -> Microsoft Outlook
  - Restore entire mailbox -> Outlook
  - Express Restore -> Microsoft Outlook
  - Express Restore -> Outlook
  - Veeam Data Cloud -> Johanna Lorenz
  - Flex Restore -> Outlook
  - Veeam Data Cloud -> Recovered Items YYYY-MM-DD, HH:MM
  - Veeam Data Cloud for Microsoft 365 -> Microsoft Outlook
  - Veeam Data Cloud -> Outlook
  - Restore to original location -> Outlook
  - Veeam Data Cloud -> Johanna Lorenz

### Step 7. Specify Rates for Microsoft 365 Services (`6eba97cf-6ffa-42ab-b0ca-11aca136f210`)

- Veeam / Veeam Service Provider Console; affected-other; episodes_added=4; tokens 61079 in (2176 cached) / 2925 out; est. $0.0064
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup for Microsoft 365 x7
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### Considerations and Limitations (`9bbd3e1a-34a8-44de-a179-b04b74661b63`)

- Veeam / Veeam Data Cloud; affected-other; episodes_added=16; tokens 578249 in (18496 cached) / 67514 out; est. $0.0750
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup for Microsoft 365 x5
- test foreign-unnamed facts (0):
- RestoresTo facts written (12):
  - Veeam Data Cloud for Microsoft 365 -> response
  - Veeam Data Cloud -> cloud root SharePoint sites
  - Veeam Data Cloud -> Exchange Online
  - Veeam Data Cloud for Microsoft 365 -> Exchange Online
  - Veeam Data Cloud for Microsoft 365 -> Archive Mailbox
  - Veeam Data Cloud for Microsoft 365 -> Exchange Online
  - Veeam Data Cloud for Microsoft 365 -> Microsoft Teams
  - Veeam Data Cloud -> OneDrive
  - Veeam Data Cloud for Microsoft 365 -> survey
  - Veeam Data Cloud for Microsoft 365 -> survey
  - Veeam Data Cloud -> Recovered Items YYYY-MM-DD, HH:MM
  - Veeam Data Cloud -> Inbox

### Microsoft 365 Backup (`a1443af5-354f-4e62-9963-5bc7386695fc`)

- Microsoft / Microsoft 365 Backup; affected-other; episodes_added=1; tokens 18156 in (1088 cached) / 557 out; est. $0.0018
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup for Microsoft 365 x2
- test foreign-unnamed facts (0):
- RestoresTo facts written (0):

### Searching for Objects in Backup (`c1b370f4-50b1-4a6c-aec5-bd891aab5779`)

- Veeam / Veeam Data Cloud; affected-other; episodes_added=13; tokens 219173 in (14144 cached) / 9282 out; est. $0.0220
- dropped (first 20 of 0): none
- production foreign-unnamed subjects: Veeam Backup for Microsoft 365 x2
- test foreign-unnamed facts (0):
- RestoresTo facts written (3):
  - Flex Restore -> SharePoint Sites
  - Veeam Data Cloud -> Outlook
  - Veeam Data Cloud -> Outlook


---

## Addendum: re-check after the final-review fixes (commit 43cf089, 6 named articles, ~$0.04)

The final review tightened the gate in three ways:

- a Product-typed name that is not the documented product must appear as a phrase (with
  "&"/punctuation normalised and never crossing a list or cell boundary) or through a
  single-target scope alias;
- abbreviations of the documented product are renamed to it;
- the cross-vendor naming rules (vSphere, Azure VM, EC2, SQL Server, Microsoft 365, Oracle RAC)
  are restored as conditional renames.

| article | production -> test facts | foreign | RestoresTo |
|---|---|---|---|
| Access verification for self-hosted connectors (Securiti) | 90 -> 93 | 37 -> 0 | 0 |
| Controls and remediation (AWS Backup) | 92 -> 51 | 0 -> 0 | 2 |
| HPE Morpheus VM Essentials | 35 -> 35 | 0 -> 0 | 28 |
| Restoring Veeam Agent Backup to Scale Computing HyperCore | 3 -> 4 | 0 -> 0 | 3 |
| Restoring Veeam Agent Backup to Proxmox VM | 5 -> 6 | 0 -> 0 | 2 |
| Restore to Sangfor aSV | 17 -> 21 | 0 -> 0 | 7 |

- **The headline "facts on articles production got right" change (-23%) is the AWS control
  article.** Its production facts included ~40 invented vocabulary facts that the
  foreign-subject measure does not count. Without it, the correct articles go 60 -> 66.
- **An intermediate run (commit b5788ab) caught a real invention.** The gate dropped seven
  "Veeam Plug-in for <hypervisor>" products on the Sangfor page, whose text never mentions a
  plug-in.
