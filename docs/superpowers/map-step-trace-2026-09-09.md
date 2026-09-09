# Map-step trace: where the global-mode drift actually starts

**Date:** 2026-09-09
**Backlog item:** 1 (P0)
**Method:** for two global questions, dumped each surviving community's `key_points`
beside the fact texts they cite. Read-only, live graph (385 episodes / 2,173 facts /
41 communities). Map model: `upstage/solar-pro4` (cheap tier), `relevance_min = 2`.

## Verdict

**The drift starts in the MAP step, not the reduce step.** Every hallucinated specific
in the traced reduce output was already present, verbatim, in the `key_points` the reduce
step was given. The reduce step copied its input faithfully; the input was unfaithful.

**This invalidates backlog item 3 as originally framed.** Constraining the reduce step
further would have constrained a step that is already doing its job — wasted work,
avoided by tracing first.

## Evidence 1 — "Compare AWS Backup and Azure Backup database restore workflows"

The community with relevance 3 produced these key points from these facts:

| Key point (what reduce sees) | Supporting fact | Verdict |
|---|---|---|
| "…PITR for RDS and Aurora **by replaying transaction logs**, but **not for RDS Multi-AZ clusters**" | "AWS Backup provides continuous backups for Amazon Aurora." | mechanism and Multi-AZ **invented** |
| "Aurora continuous backup retention **ranges from 1 to 35 days**" | "…only one recovery point is possible at a time…" | range **invented** |
| "SAP HANA on EC2 … **1-second precision** PITR **up to 35 days**" | "AWS Backup continuously backs up transaction logs for SAP HANA databases to enable point-in-time restore." | precision and duration **invented** |
| "Cross-Region copy … Aurora (**from full backups**) and RDS/S3 (**from incremental backups**)" | "AWS Backup provides cross-Region copy capability for some resource types." | full/incremental **invented** |
| "Cross-account backup **requires source and destination accounts to be in the same AWS Organization**" | "AWS Backup provides cross-account copy capability for some resource types." | requirement **invented** |

Every one of these reappeared in the final answer with a real marker attached. The
answer scored faithfulness 0-1, correctly.

## Evidence 2 — "Compare how AWS Backup and Azure Backup encrypt backup data"

The same map step, same model, same prompt, was **faithful** here:

- "automatically encrypts all backed-up data using Azure Storage encryption with AES-256"
  ← "encrypts data at rest using 256-bit AES encryption (AES 256)" ✓
- "Infrastructure encryption … configurable only after choosing CMK"
  ← "Infrastructure encryption can only be configured if you first choose to use your
  customer-managed keys" ✓
- "platform-managed keys by default, with support for CMK stored in Azure Key Vault" ✓
- "AWS Backup uses KMS-key-based encryption for backup vaults" ✓

This question scored faithfulness 4 — the best global score in the eval.

## The mechanism

**The map step hallucinates to fill gaps.** It does not fabricate at random: it fabricates
when the community's facts do not answer the question.

- Encryption question → facts are rich and specific (AES-256, DEK, infrastructure
  encryption ordering) → key points are accurate → answer scores 4.
- Database-restore question → facts are generic ("cross-Region copy capability for some
  resource types") → key points invent plausible specifics → answer scores 0-1.

This is why global faithfulness is bimodal rather than uniformly poor, and why no amount
of reduce-side prompting moved it.

## Second finding: meta-commentary is injected at the map step

Task 2 forbade the *reduce* step from commenting on absent evidence. But the map step is
generating exactly that and feeding it in as key points:

- "The report does not contain information about AWS Backup encryption, preventing a
  direct comparison."
- "No direct comparison between AWS Backup and Azure Backup encryption can be made from
  this report."
- "Report focuses on Azure Resiliency platform, not AWS Backup."
- "No information provided on AWS Backup restore workflows."

The reduce prompt is fighting its own input. That explains why banning meta-commentary in
reduce improved grounding but did not eliminate the underlying noise.

## Third finding: `relevance_min = 2` admits communities with nothing to contribute

In both traces, the third surviving community was junk:

- Database-restore question → "Resiliency in Azure: Unified BCDR Platform" at relevance 2,
  whose facts are about monitoring, alerts and vault filtering — nothing about database
  restore. Three of its four key points are meta-commentary.
- Encryption question → "AWS Backup Vault Lock" at relevance 2, whose own key point admits
  it "provides no details" on encryption.

These consume marker numbers and inject noise. A community whose map output is essentially
"no information provided" should not reach the reduce step at all.

## Recommended follow-ups (supersedes backlog item 3)

1. **Bind the map prompt to its facts** — the same treatment the reduce prompt received:
   every key point must be supported by the facts it cites, no invented specifics, no
   meta-commentary about what the report lacks. This is the primary fix and it is where
   the defect lives.
2. **Drop non-contributing communities** — either raise `relevance_min` above 2 or detect
   a map result that carries no supported key point, before it reaches reduce.
3. **Re-test the map tier afterwards.** The map model is `upstage/solar-pro4`, the same
   cheap-tier model implicated in the out-of-range dedup indices (backlog item 4). Fix our
   prompt first — on this project the fault has been in our own code or config every time
   — then compare cheap vs strong tier on the map step with the corrected prompt.
4. **Backlog item 3 (structural reduce constraint) — deprioritise.** The reduce step is
   faithful to its input. Revisit only if fixing the map step does not move faithfulness.

## Caveats

- n = 2 questions. The mechanism is established (every hallucinated specific traced 1:1 to
  a key point), but the gap-filling hypothesis would be firmer across more questions.
- Extraction is not deterministic (backlog item 7), so exact key points vary between runs;
  the pattern held across both traces.
