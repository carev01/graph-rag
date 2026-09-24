# Coverage-prompt A/B — a prompt does not recover packing's fact loss

**Date:** 2026-09-24. **Spend:** $0.47 for the new arm (smoke included) plus the judge.
**Spec:** `specs/2026-09-24-coverage-prompt-ab-design.md`. **Harness:** `scripts/chunk_ab.py
--coverage`, `scripts/chunk_ab_judge.py`.
**Decision:** do **not** adopt. Packing stays off; production prompts unchanged.

## What was tested

The same 30 articles as the packing A/B. `today` and `pack1200` were reused from that run;
`pack1200_coverage` packed to 1,200 tokens with two instruction changes on both tiers:
the cheap tier's fixed "roughly 10-25 facts" sentence removed, and a coverage directive
("go through the message section by section … extract every … statement … a longer
message should yield proportionally more … never restate one fact in different words").
The smoke's LLM capture proved the directive reached graphiti's extraction prompts
(marker found in 4 captured `extract_nodes`/`extract_edges` prompts) before the full run.

Scoring is by **distinct ideas**, not raw facts: the eval judge (DeepSeek V4 Flash — a
different family from both extraction models) saw each article's source and the three
arms' facts under shuffled anonymous labels, built an inventory of distinct claims, and
labelled every fact with its idea and whether the source supports it. 27 of 30 articles
judged (1 over the 150-fact limit — the 461-fact Commvault page; 2 unscored after a retry).

## Result

| 27 articles | facts | supported | **distinct** | redundant | unsupported |
|---|---:|---:|---:|---:|---:|
| today | 867 | 814 | **728** | 86 | 53 (6.1%) |
| pack1200 | 613 | 580 | **534** | 46 | 33 (5.4%) |
| pack1200_coverage | 583 | 569 | **508** | 61 | **14 (2.4%)** |

Paired median distinct-idea ratio vs today: **pack1200 0.76**, **pack1200_coverage 0.74**
(IQR 0.60–0.90; ≥ today on 5 of 26). Cost ratio 0.53. The decision rule required ≥ 0.95 —
not met. Across all 30 articles, raw facts: today 1,360, pack 982, coverage 1,029.

## Reading

1. **The prompt does not move per-call yield.** Removing our fixed count and asking for
   coverage produced the same distinct content as plain packing (0.74 vs 0.76, within
   noise). The saturation is the models' — and graphiti's entity prompt ("could this have
   its own Wikipedia article?", facts only between extracted entities) sits underneath any
   instruction we append. Packing trades real content for cost, and no appended instruction
   buys it back.
2. **Correction to `chunk-packing-ab-2026-09-24.md`.** That doc said "much of [the fact
   drop] is redundancy". It is not: today's arm has 86 redundant of 814 supported facts
   (10.6%), and distinct ideas fall 27% — as much as raw facts. The two hand-checked
   articles over-stated redundancy; the judge, over 27 articles, does not bear it out.
3. **A side result worth following: the directive cut unsupported facts 6.1% → 2.4%**
   (53 → 14). The "never restate; each fact must add a claim or detail" clause is the
   likely cause. This is one run, one judge, at the packed chunk size — but an
   unsupported fact is a wrong answer waiting to be cited, so it is worth a targeted test
   at today's chunking (BACKLOG 44).

## Judge calibration

Against the two articles hand-checked earlier: Nakivo — judge 16 / 14 distinct (today /
pack), hand estimate ~13 / ~13; AvePoint — judge 11 / 9, hand count: pack loses two real
facts. Same direction, similar magnitude; the judge splits ideas slightly finer, as its
instruction ("a fact adding a concrete detail is a different idea") asks.
