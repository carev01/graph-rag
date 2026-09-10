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

---

## Post-fix validation (2026-09-10)

Graph unchanged from the trace above: 385 episodes, 2,173 facts, 41 communities detected.

### Rebuild results across three runs

The first two runs were made under the ORIGINAL spec and exposed two design flaws in it.
Run 3 is the fixed system.

| | run 1 (original) | run 2 (original) | **run 3 (fixed)** |
|---|---|---|---|
| reports_written | 25 | 18 | **36** |
| reports_staged (recoverable) | — | — | **5** |
| **permanently lost** | **16** | **23** | **0** |
| findings_dropped | 6 | 0 | **13** |
| reports_reverified | 34 | 33 | 36 |
| reports_unverified | 7 | 8 | 5 |

`theme-build --verify-pending` then recovered the staged reports at a cost of five verify
calls rather than a full rebuild: **1 promoted, 2 rejected**, **2 still pending**
(verifier still flaky, recoverable whenever it clears).

No cause can be attributed to those 2 rejections. `reports_rejected` was a single
counter merging two very different outcomes — "the summary was judged unsupported" and
"every finding was dropped" — and at the time of this run the summary verdict was still a
rejection reason inside `--verify-pending` (the defect fixed as Critical 2 of the final
branch review). The earlier claim that these were "genuine content failures" was
unsupported by the data and has been removed; on the measured rates the likelier cause is
the summary rule.

Final state: **41 communities present, 37 retrievable**, 2 staged-and-recoverable,
2 rejected.

### The two design flaws the live runs exposed

1. **The summary rule was destroying reports for the wrong reason.** Runs 1 and 2 skipped
   15-16 non-transient reports while reporting `findings_dropped` of 6 and **0**. Because
   the "every finding dropped" path increments that counter before returning, a zero
   proves nothing was lost for bad findings — every non-transient loss was the summary.
   The rule was wrong in principle: a community summary is inherently synthetic, so
   judging it by "states nothing the facts do not state" rejects legitimate summarising.
   The summary is now regenerated from the findings that survive verification.
2. **A transient verifier error destroyed the report entirely.** `write_communities` keeps
   only communities with a new report and then `DETACH DELETE`s the layer, so a provider
   blip removed a community until a full rebuild — discarding generation already paid for
   because the *check* failed. Reports are now staged with no embedding (unreachable from
   every answering path by construction) and recovered by `--verify-pending`.

### Verification of the original defect

All six traced inventions are **gone** from every retrievable report:

| probe | reports carrying it |
|---|---|
| `1-second` | 0 |
| `35 day` | 0 |
| `1 to 35` | 0 |
| `Multi-AZ` | 0 |
| `same organization` | 0 |
| `from incremental backups` | 0 |

### The headline number

**36 of 41 reports failed first-pass verification** (`reports_reverified`). The report
writer was inventing on ~88% of communities, not merely the four cases originally traced.
`findings_dropped = 13` is what survived into a second attempt and still could not be
supported.

### Eval: RUN, and the gain is measured

At the time this trace was written the router golden-set eval could not be run:
`openrouter.ai` resolved only to IPv6 and this host had no default IPv6 route, so every
LLM call failed with `APIConnectionError` — a transient local network condition.

**It has since run.** Against the 1.6 pre-verification baseline (see
`router-eval-report.md`):

| metric | before | after |
|---|---|---|
| global faithfulness | 1.6 | **2.33** |
| global grounding | 1.00 | **0.86** |

Global faithfulness rose **1.6 → 2.33**, a real measured gain from stopping the report
writer's inventions. Global grounding dipped **1.00 → 0.86** on a 7-question base — one
question flipped. Global remains the weakest mode (2.33 against local's 4.79).

Note the eval harness swallowed 14 connection errors into per-question failures rather
than aborting, which would have produced a plausible-looking but meaningless report. That
is the same swallow-and-continue pattern this project has been bitten by repeatedly.
