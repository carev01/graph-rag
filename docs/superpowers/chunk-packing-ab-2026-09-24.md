# D6 A/B — chunk packing to 1,200 tokens

**Date:** 2026-09-23/24. **Spend:** $1.22 of the authorised $5 (smoke + both arms).
**Harness:** `scripts/chunk_ab.py` (commits `eb66bc2` packing, `3288ba6` harness).
**Decision:** **keep packing OFF** (`pack_target_tokens = 0`) for the bootstrap.

## Method

30 articles from the D6 corpus sample (≥3 episodes today, round-robin across 13 vendors),
each arm in its OWN throwaway Neo4j testcontainer — the live graph was never written.
Arm `today` = production settings; arm `pack1200` = `pack_target_tokens=1200`,
`cheap_max_chunk_tokens=1200`. Same-pair suppression (D4) on in both. Articles ran
sequentially through `IngestDriver.ingest_article`, bypassing `ingest_source`'s warm-up
barrier and sibling spreading (they do not interact with chunking). A 2-article smoke
gated the spend: it refused unless packing cut episodes and every row was metered — it
passed (8 vs 2 episodes).

## Result

| per article (paired, n = 30) | pack ÷ today, median | IQR | pack ≥ today |
|---|---:|---:|---:|
| facts | **0.73** | 0.55–0.88 | 4 of 29 |
| entities | 0.83 | 0.72–1.13 | 10 of 30 |
| cost | **0.57** | 0.45–0.77 | 2 of 30 |
| wall clock | **0.52** | 0.38–0.80 | 2 of 30 |

Totals: episodes 213 → 82 (−62%); facts 1,360 → 982 (**−28%**); entities 1,067 → 926;
cost $0.72 → $0.46 (**−36%**); wall 80 → 49 min (**−40%**). Cost per fact is roughly
unchanged ($0.53 vs $0.47 per 1,000).

## Why facts drop: per-call extraction saturates

The smoke run's LLM capture shows each extraction call returns a roughly constant number
of items regardless of input size — ~6–12 entities and ~4–10 facts per call. Five
~200-token episodes of one Commvault article yielded 46 entities / 32 facts; the same
article as one 1,086-token episode yielded 11 / 10. Bigger episodes get one call's worth
of extraction, not proportionally more.

**Confound:** the cheap tier's `CHEAP_TIER_SALIENCE` instruction (`ontology.py`) asks for
"BE SELECTIVE, NOT EXHAUSTIVE … roughly 10-25 facts" per chunk — a per-call yield cap
written for ~300-token chunks. Part of the saturation is therefore our own prompt, not the
model. (The strong tier saturated too, so it is not only the prompt.)

## What the lost facts are — hand-judged, two median-ratio articles

- **Nakivo** (19 → 14): today's surplus is largely redundant restatement — one idea
  ("Director generates a Certificate and Pre-shared Key") split into three facts, the PMA
  deployment fact twice, "supports backing up physical machines" beside "provides Physical
  Machine Backup". Packing lost real content (the Transporter's role; pre-shared-key
  auto-generation) but also *gained* content today lacked (incremental backup via
  proprietary change tracking; file and application-object recovery; P2V migration).
  Distinct ideas roughly equal (~13 each).
- **AvePoint** (12 → 9): today repeats the CAP Gateway fact three times; packing kept every
  version and authentication requirement, merged more cleanly, and lost two real facts
  (the app-profile requirement; the AvePoint Online Services tenant connection).

Reading (as first written): the 28% fact drop overstates the information loss — much of it
is redundancy — but real facts are lost, perhaps 10–20% of distinct content.

**CORRECTED 2026-09-24** (`coverage-prompt-ab-2026-09-24.md`): a distinct-idea judge over 27
articles found redundancy is only 10.6% of today's supported facts and distinct ideas fall
**27%** — as much as raw facts. Packing loses real content; the two hand-checked articles
over-stated redundancy. A coverage prompt did not recover it (0.74 vs 0.76).

## Decision and why

Packing off. The standing ruling is that answer quality outranks latency; the saving is
~$780 of a ~$2,176 bootstrap, and a measurable loss of real facts is the wrong trade for a
system whose answers are only as good as the facts it holds. The 40% wall-clock saving is
the stronger argument and is **not** dismissed: it is recorded as a follow-up (BACKLOG 42)
— retest packing with the per-chunk salience instruction scaled to episode size, which
would separate the prompt's share of the loss from the model's. The packing code stays
merged, default off, with this outcome recorded at the setting.
