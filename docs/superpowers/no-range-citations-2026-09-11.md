# Range shorthand in the reduce step: prompt fix, detection, live measurement

**Date:** 2026-09-11
**Backlog item:** 0d (P0)
**Branch:** `no-range-citations` (from `main` at `4eaf05f`)
**Graph:** unchanged throughout — 385 episodes, 2,173 facts, level 1 12/12 retrievable.
`theme-build` was **not** run.
**Tiers:** as the previous slice — map `upstage/solar-pro4`, reduce `z-ai/glm-5.3-flash`,
rerank Voyage `rerank-3` (`top_n = 4`, floor 0.40), eval judge `deepseek-v4-flash`.

## 1. The defect

The reducer wrote citation ranges (`[1]–[26]`, `[14]–[25]`). `_finalize_answer` keeps
only literal markers — there is no range expander, by the citation-integrity spec's
decision that a 26-marker span is a guess, not a citation — so a range finalizes to its
two endpoints. The faithfulness judge scores an answer against its *cited* facts only, so
an answer resting on 26 facts was judged against 2.

Baseline, recomputed from the previous slice's saved answers (`live_recheck.json`,
2026-09-10, old reduce prompt) with the new detector — it reproduces the 4 / 2 / 6 figures
exactly, which is the detector's validation against real model output (the model writes
en-dashes, `[1]–[26]`):

| question | rep | facts available | citations kept | ranges written |
|---|---|---|---|---|
| database restore | 1 | 25 | 25 | 0 |
| database restore | 2 | 25 | **4** | **10** |
| database restore | 3 | 18 | 18 | 0 |
| compliance retention | 1 | 59 | **2** | **1** |
| compliance retention | 2 | 53 | **6** | **6** |
| compliance retention | 3 | 59 | 17 | 0 |
| encryption (control) | 1 | 17 | 17 | 0 |
| encryption (control) | 2 | 17 | 17 | 0 |
| encryption (control) | 3 | 17 | 6 | 0 |

3 of 9 runs wrote ranges (17 range instances); 112 of 290 available facts were cited
(39%). Note encryption rep 3 — 6 of 17 with **no** range: that loss is not this defect.

## 2. What changed

Expanding ranges stays rejected; stripping both endpoints was never on the table (they
resolve to real facts). The fix is that the model must not write a range at all.

**`_REDUCE_PROMPT`** (`src/answer_api/global_search.py`) gains one rule, placed directly
under the existing "cite every claim" rule and touching nothing else:

> `- Write each marker individually: [1] [2] [3]. Never write a range or span such as
> [1]-[3]; only the markers you write out are cited.`

The refusal trigger and the meta-commentary ban are byte-for-byte where they were
(a test now counts the rules: six tuned + one new).

**Detection** (`src/answer_api/synthesize.py`): `_range_markers(text)` finds every
range-shaped marker sequence (`[a]-[b]`, en/em dash, `…`, `...`, `to`, `through`; both
sides must be markers, so `[1]-recovery` and `[2] — and` do not match).
`_finalize_answer` calls it on the **final** text — the shared path for local, global
and DRIFT — and logs a WARNING with the range(s), markers cited, and facts available:

```
citation range shorthand survived into the answer: 1 range(s) ['[1]-[26]']; 3 marker(s)
cited of 26 facts available -- the markers inside each span are NOT cited (no expander, by design)
```

A range whose endpoint was unresolvable and removed is not reported: no span is left
for the reader to see.

**Eval harness** (`src/answer_api/eval_router.py`): each per-question record now
carries `cited` and `ranges`, printed on the progress line and as two new columns in
`router-eval-report.md`. The harness does not persist answer text, so until now a
range-shorthand answer judged against two facts and a well-cited one were
indistinguishable in the report.

**Tests** (all written failing first, then mutation-verified — removing the prompt rule
→ `2 failed`; silencing the warning → `1 failed`; dropping `ranges` from the record →
`1 failed`; each restored → green; `git diff --stat src/` empty):

- `tests/unit/test_reduce_prompt.py` — the rule is asserted inside the prompt's
  `Rules:` block only (`_MAP_PROMPT` shares "Absence of evidence is not a finding" with
  the reduce prompt; the module imports only `_REDUCE_PROMPT`, and the block scoping
  makes a cross-match impossible). None of "range", "span", "individually" or the
  `[1] [2] [3]` example existed anywhere in the prompt before, so the substring test
  could not pass by accident. A second test pins the refusal trigger, the
  meta-commentary ban and the rule count.
- `tests/unit/test_finalize_answer.py` — the endpoint-only behaviour is pinned (so
  nobody "fixes" it by expanding); every separator form is detected; prose dashes are
  not; the warning carries the counts; a removed-endpoint range and a range-free answer
  log nothing.
- `tests/unit/test_eval_router_harness.py` — the record and report columns.

CI gate on the final source: `ruff check src tests` clean, `mypy src` clean,
`pytest -m "not live"` **693 passed** (10m45s).

## 3. Live after: did the model comply?

**Yes.** Same method as the baseline (`global_search` end to end, fresh shortlist → map →
reduce → provenance, `_complete_or_none` wrapped to see the raw reduce output), 3 reps
per question, extended from the 3 baseline questions to all 5 global golden questions.
Ranges are counted in the **raw** model output as well as the final answer, so a range
cannot hide behind stripping.

| question | rep | facts available | citations kept | kept % | ranges (raw / final) |
|---|---|---|---|---|---|
| database restore | 1 | 54 | 5 | 9% | 0 / 0 |
| database restore | 2 | 32 | 7 | 21% | 0 / 0 |
| database restore | 3 | 44 | 11 | 25% | 0 / 0 |
| compliance retention | 1 | 53 | 8 | 15% | 0 / 0 |
| compliance retention | 2 | 53 | 18 | 33% | 0 / 0 |
| compliance retention | 3 | 59 | 23 | 38% | 0 / 0 |
| encryption (control) | 1 | 17 | 10 | 58% | 0 / 0 |
| encryption (control) | 2 | 17 | 17 | 100% | 0 / 0 |
| encryption (control) | 3 | 17 | 17 | 100% | 0 / 0 |
| cross-region restore | 1 | 29 | 8 | 27% | 0 / 0 |
| cross-region restore | 2 | 13 | 13 | 100% | 0 / 0 |
| cross-region restore | 3 | 47 | 23 | 48% | 0 / 0 |
| recovery-point deletion | 1 | 22 | 22 | 100% | 0 / 0 |
| recovery-point deletion | 2 | 47 | 17 | 36% | 0 / 0 |
| recovery-point deletion | 3 | 30 | 17 | 56% | 0 / 0 |

- **Ranges: 17 → 0**, across 15 runs, raw and final. No refusals, no None-path.
- **Citations kept did not recover: 216 of 534 available facts (40%) against 39% at
  baseline.** On the same three questions as the baseline, 116 of 346 (34%) against
  112 of 290 (39%).
- DRIFT probe (its `_SYNTH_PROMPT` is unchanged and forbids nothing): 2 questions × 1 rep,
  16/32 and 25/32 cited, **0 ranges**. Too small to clear DRIFT, but no range appeared.

### Why the citation count did not move

The model's alternative to `[14]–[25]` is not to write twelve markers; it is to write
**one marker per sentence, assigned by position**. Compliance rep 1 cites `[1] [2] [3]
[4] [5] [6] [7] [8]`, one per sentence, in order; cross-region reps 2–3 and
recovery-point reps 1–3 cite `[1]…[12]` strictly ascending. On 14 of 15 runs the cited
sequence is ascending within at most one block per community. That is BACKLOG 0b —
the reducer receives `Supporting facts: [1] [2] … [19]` as an unlabeled bag and cannot
know which fact backs which sentence — with nothing left to hide behind.

The four 100% runs are the other face of the same mechanism: there the model pasted the
whole bag after each sentence (`[1][2]…[12]` — up to 19 markers on one sentence). The
judge then sees every fact, which is why those answers can score well, but the markers
are no more claim-specific than the single ones.

So 0d is closed as specified — the model no longer writes ranges, and the loss is no
longer silent — but it was not the lever on citation retention or, by extension, on
the global faithfulness figure. Range shorthand was one symptom of 0b; removing it
exposes 0b rather than compensating for it.

## 4. Full eval

`uv run --extra dev python -m answer_api.eval_router`, 29/29 questions, 0 failed,
1h40m, started 2026-09-11 09:00. Report: `router-eval-report.md` (now with `cited` and
`ranges` columns).

| | 2026-09-10 re-baseline | 0c run (prev) | **this run** |
|---|---|---|---|
| Routing accuracy | 0.97 | 0.97 | 0.97 |
| Grounding precision | 0.85 | 0.88 | **0.88** |
| — global | 0.86 | 1.00 | **1.00** |
| Faithfulness mean | 3.92 | 4.11 | **3.75** |
| — global | 1.5 | 2.44 | **1.6** |
| **unscored** | 3/29 | 2/29 | **1/29** (Q10, local refusal — unchanged) |
| global-mode questions scored | 8/10 | 9/10 | **10/10** |
| ranges in any answer | (not measured) | (not measured) | **0/29** |
| comparative global / drift | — | 1.6 / 4.5 | 2.5 / 4.7 (`drift_wins` False) |

Per global-routed question (chosen = global), with the new columns:

| question | intent | prev | **now** | cited | ranges |
|---|---|---|---|---|---|
| Compare how AWS Backup and Azure Backup encrypt backup data (control) | global | 2 | **5** | 17 | 0 |
| Compare cross-region restore between AWS Backup and Azure Backup | global | 4 | **2** | 5 | 0 |
| How do AWS Backup and Azure Backup each protect recovery points from deletion? | global | 5 | **2** | 8 | 0 |
| Which backup capabilities do AWS and Azure share for compliance retention? | global | 1 | **0** | 15 | 0 |
| Compare AWS Backup and Azure Backup database restore workflows | global | 2 | 2 | 12 | 0 |
| What should I consider when planning long-term backup retention across cloud vendors? | drift | 1 | 1 | 28 | 0 |
| What should I consider for backup encryption across cloud providers? | drift | 2 | 1 | 12 | 0 |
| What are the key considerations for cross-region disaster recovery of cloud backups? | drift | 1 | 1 | 15 | 0 |
| How should I approach immutability and ransomware protection for cloud backups? | drift | 4 | 1 | 24 | 0 |
| What should I think about when restoring databases from cloud backups? | drift | refused | **1** | 16 | 0 |

### Reading it plainly

- **The model complied in the eval as well: 0 ranges in all 29 answers**, and the
  `_finalize_answer` warning never fired. The encryption control **recovered 2 → 5**,
  citing 17 of 17 — consistent with the reviewer's trace of its last drop to this
  defect, though a single run cannot prove that was the cause.
- **Global faithfulness did not improve: 1.6 over 10/10 scored, against 2.44 over 9/10.**
  Like-for-like over the nine questions scored last time it is 1.67. This is the fourth
  flat-or-down global result in a row. The comparative forced-global figure moved the
  other way (1.6 → 2.5), which says the per-question single-run numbers carry more
  variance than their differences.
- **The new columns say where the loss is.** Every low score sits on 5–28 citations with
  no range: compliance retention scored **0 on 15 citations**; long-term retention 1 on
  28; immutability 1 on 24. The judge is not starved of facts any more — it is handed
  facts that do not back the sentences they are attached to, which is exactly the
  positional pattern of §3 and the definition of BACKLOG 0b. Range shorthand was never
  the reason global scores are low; it was one way the same underlying defect showed up.
- **Unscored dropped 2 → 1.** The DRIFT-intent database-restore question, which refused
  last run, answered this time (scored 1). The only unscored entry is Q10 (local
  refusal), unchanged across three runs.
- **The None path did not fire**: 5 reduce first attempts exhausted 3,000 tokens and every
  retry recovered (0 "giving up"); 1 judge skip (Q10); 0 judge-unmeasurable.
- Not compared here: `theme-build` was not run, so the report layer is identical to the
  previous two evals.

### Caveats

- Ten global-mode questions, one run each, map-step fact counts that vary 3× between
  runs of the same question (§5.2): the 2.44 → 1.6 move is not a measured regression from
  the prompt change, and a 1.6 → 2.4 move next run would not be a recovery either.
- The prompt change *could* lower scores mechanically — a range-free answer that
  previously kept 2 endpoints and now keeps 5 positional markers gives the judge more
  wrong-fact pairs to find — but this run cannot separate that from the run-to-run
Reverting the rule would restore silent citation loss without evidence it
  would restore any score; it should stay.

## 5. Things we had not considered

1. **The range instruction trades "2 of 26" for "1 of 26 per sentence".** Citation
   retention is the wrong lever to pull from the reduce prompt; the count is set by
   whether the reducer can bind claims to facts, which it cannot from a marker bag. 0b is
   the fix, and the measurement in §3 is now clean enough to make 0b's effect visible:
   any run where citations jump above ~40% without bag-pasting would be a real signal.

2. **The map step hands the reducer far more facts than a slice ago.** 54 / 32 / 44
   available on database restore against 25 / 25 / 18 the day before, same graph, same
   prompt. Variance in the map step's fact_ids (already noted as 3× in the previous
   report) is now the largest single source of noise in "kept %", and it is why the
   three-question retention figure moved 39% → 34% while the five-question figure
   moved 39% → 40%: neither is a change.

3. **Bag-pasting is indistinguishable from good citation in every metric we have.** A
   100%-kept run and a claim-bound run look identical to `cited`, to grounding precision
   and to the judge. Nothing in the harness penalises 19 markers on one sentence. When 0b
   lands, "markers per sentence" is worth recording next to `cited` and `ranges`.

4. **Detection sits in `_finalize_answer`, so the local and DRIFT paths are covered too**
   even though only the reduce prompt was changed. Their prompts (`_PROMPT`,
   `_SYNTH_PROMPT`) do not forbid ranges; the DRIFT probe wrote none, but if the warning
   ever fires on those paths the same one-line rule is the response.

5. **`_range_markers` matches the plain-prose forms `to` / `through`** (`[3] to [9]`) as
   well as dashes and ellipses. No baseline answer used them; they are in the detector so
   a model that switches separator after the dash is banned cannot go quiet again.
