# Coverage-prompt A/B — a prompt does not recover packing's fact loss

**Date:** 2026-09-24. **Spend:** $0.47 for the new arm (smoke included). The judge's own
spend was not tallied by `chunk_ab_judge.py` (no cost tracking in that script), so it is
not reported here.
**Spec:** `specs/2026-09-24-coverage-prompt-ab-design.md`. **Harness:** `scripts/chunk_ab.py
--coverage`, `scripts/chunk_ab_judge.py`.
**Decision:** do **not** adopt. Packing stays off; production prompts unchanged.

## What was tested

The same 30 articles as the packing A/B. `today` and `pack1200` were reused from that run;
`pack1200_coverage` packed to 1,200 tokens with two instruction changes on both tiers:
the cheap tier's fixed "roughly 10-25 facts" sentence removed, and a coverage directive
("go through the message section by section … extract every … statement … a longer
message should yield proportionally more … never restate one fact in different words").
The smoke's LLM capture proved the directive reached graphiti's extraction prompts on
**both** tiers (marker found in captured `extract_nodes`/`extract_edges` prompts for the
cheap tier's `messages`-shaped records and the strong tier's `input`-shaped
`responses.parse` records) before the full run.

Scoring is by **distinct ideas**, not raw facts: the eval judge (DeepSeek V4 Flash — a
different family from both extraction models) saw each article's source and the three
arms' facts under shuffled anonymous labels, built an inventory of distinct claims, and
labelled every fact with its idea and whether the source supports it. 27 of 30 articles
judged (1 over the 150-fact limit — the 461-fact Commvault page; 2 unscored after a retry).

Note on the "facts" column: it is `collect(DISTINCT f.fact)` in `chunk_ab.py`'s graph
query, i.e. distinct fact *strings* per article per arm — verbatim duplicates were already
removed before the judge ever saw them. The redundant/distinct split below is therefore
slightly understated (some duplication never reaches the judge to be counted).

## Result

| 27 articles | facts | supported | **distinct** | redundant | unsupported |
|---|---:|---:|---:|---:|---:|
| today | 867 | 814 | **728** | 86 | 53 (6.1%) |
| pack1200 | 613 | 580 | **534** | 46 | 33 (5.4%) |
| pack1200_coverage | 583 | 569 | **508** | 61 | **14 (2.4%)** |

Paired median distinct-idea ratio vs today: **pack1200 0.76** (IQR 0.54–0.97, ≥ today on
6 of 26), **pack1200_coverage 0.74** (IQR 0.60–0.90, ≥ today on 5 of 26). One article
(`e092bc6f-390e-4f9c-b2d9-b23f35047a70`) had 0 distinct ideas under `today` and is dropped
from all three ratio sets, leaving 26 of the 27 judged articles — hence "5 of 26" /
"6 of 26", not "of 27". Across all 30 articles (raw fact counts, unjudged): today 1,360,
pack 982, coverage 1,029.

`decide()` in `chunk_ab_judge.py` computes the median/IQR only for `pack1200_coverage`; the
`pack1200` figures above were computed ad hoc from `judged.jsonl`:

```python
ratios = [j["pack1200"]["distinct"] / j["today"]["distinct"]
          for j in judged if j["today"]["distinct"]]
statistics.median(ratios), statistics.quantiles(ratios, n=4)
# -> 0.7576887232059646, [0.539179104477612, 0.7576887232059646, 0.9741379310344828]
```

## Cost and time (aggregate, not the per-article median)

**0.53 is the median of per-article cost ratios**, not the aggregate spend ratio — a few
cheap-tier articles with tiny denominators pull the median down relative to what the run
actually cost. The aggregate ratios, both vs `today`, over all 30 articles and over the 27
judged:

| | cost, all 30 | cost, 27 judged | wall clock, all 30 | wall clock, 27 judged |
|---|---:|---:|---:|---:|
| pack1200 | 0.64 | 0.59 | 0.60 | 0.53 |
| pack1200_coverage | 0.63 | 0.57 | 0.72 | 0.65 |

(Sums from `results_coverage.json`'s per-article `cost`/`seconds`: `today` $0.722/4830s over
30, $0.568/3165s over the 27 judged; `pack1200` $0.463/2921s, $0.336/1666s;
`pack1200_coverage` $0.458/3472s, $0.326/2058s.)

**The coverage arm is not cheaper than plain packing** — its aggregate cost ratio
(0.57–0.63) sits essentially on top of `pack1200`'s (0.59–0.64), not below it — and it
**gives back part of the wall-clock saving**: `pack1200` cuts wall clock to ~53–60% of
`today`, `pack1200_coverage` only to ~65–72%. The "go section by section" directive keeps
both tiers working longer per call even though packing still cuts call count the same way
in both arms.

## Reading

1. **The prompt does not move per-call yield overall, and actively hurts the strong tier.**
   Removing our fixed count and asking for coverage produced about the same distinct content
   as plain packing in aggregate (0.74 vs 0.76, within noise) — the saturation is the
   models', and graphiti's entity prompt ("could this have its own Wikipedia article?",
   facts only between extracted entities) sits underneath any instruction we append. But the
   two tiers do not move together. Cheap-tier articles (20 of 27 judged) are barely
   affected: distinct ideas 431 (today) → 370 `pack1200` (0.86×) → 371 `pack1200_coverage`
   (0.86×) — a wash. Strong-tier articles (7 of 27) are where the directive actively makes
   things worse: 297 (today) → 164 `pack1200` (0.55×) → **137 `pack1200_coverage` (0.46×)**
   — the coverage prompt loses *more* distinct content on the strong tier than plain packing
   alone does, not less. This strengthens the do-not-adopt case: on the tier where per-call
   saturation already bites hardest, the added directive is not neutral, it is worse.
2. **Correction to `chunk-packing-ab-2026-09-24.md`.** That doc said "much of [the fact
   drop] is redundancy". It is not: today's arm has 86 redundant of 814 supported facts
   (10.6%), and distinct ideas fall 27% — as much as raw facts. The two hand-checked
   articles over-stated redundancy; the judge, over 27 articles, does not bear it out.
3. **A lead worth a targeted retest, cause unknown: unsupported facts fell, but the
   picture is not as clean as one number suggests.** Against the fair baseline —
   `pack1200`, same chunk size, only the instructions differ — unsupported facts fell
   5.4% (33/613) → 2.4% (14/583). Two strong-tier articles carry most of `today`'s
   unsupported count: Veeam `8ab86873…` (16 of 52 facts) and a different AvePoint article,
   `720039c5…` (12 of 104) — not the AvePoint article hand-checked in "Judge calibration"
   below — together 28 of `today`'s 53 unsupported facts (53%). On `720039c5…` the
   coverage arm's output also shrank sharply beyond what packing alone did — `today` 104
   facts → `pack1200` 59 → `pack1200_coverage` 20 — so a collapsing denominator, not
   necessarily the directive's wording, plausibly explains part of its drop to 0
   unsupported facts; volume confounds this article's contribution to the result.
   Excluding both articles from all three arms: `today` 3.5% (25/711), `pack1200` 4.1%
   (22/536), `pack1200_coverage` 1.8% (10/544) — `pack1200_coverage` is still the lowest
   of the three and the proportional drop from `pack1200` (4.1% → 1.8%) is similar to the
   full-set drop (5.4% → 2.4%), so the effect is not purely an artifact of these two
   articles, but they do account for a large, disproportionate share of it.
   Paired per-article sign counts on unsupported-fact count (n = 27): `pack1200_coverage`
   lower than `today` on 12, higher on 4, tied on 11; lower than `pack1200` on 9, higher on
   2, tied on 16 — a real majority-lower pattern, far from unanimous.
   Meanwhile **redundancy rose** under the directive: 46 (`pack1200`) → 61
   (`pack1200_coverage`). The "never restate one fact in different words" clause is
   therefore **not established as the cause** of the unsupported-rate drop — if the clause
   worked as intended, redundancy should have fallen, not risen — and the arm changed two
   things at once (the per-chunk fact cap removed *and* the coverage directive added), so
   no single clause can be credited from this run. This is a lead, cause unknown, from one
   judge run — see BACKLOG 44 (lowered to P3) for the targeted retest that isolates the
   restatement clause alone at today's chunking.

## Judge calibration

Against the two articles hand-checked earlier: Nakivo — judge 16 / 14 distinct (today /
pack), hand estimate ~13 / ~13; AvePoint — judge 11 / 9, hand count: pack loses two real
facts. Same direction, similar magnitude; the judge splits ideas slightly finer, as its
instruction ("a fact adding a concrete detail is a different idea") asks.

## Re-running

`scripts/chunk_ab.py --coverage` writes both the 2-article smoke run and the full run to
the same `out / "rows.jsonl"` (both `run_arm` calls receive the same `--out`), so that file
mixes smoke and full-run rows and the smoke articles appear in it twice. `judged.jsonl`
is opened in append mode by `chunk_ab_judge.py`, so a re-run against the same `--out`
appends duplicate lines rather than replacing them. Point `--out` at a fresh directory for
any re-run of either script.
