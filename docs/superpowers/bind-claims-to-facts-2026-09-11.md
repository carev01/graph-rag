# Bind claims to their facts in the reduce step: fix, per-claim audit, live measurement

**Date:** 2026-09-11
**Backlog item:** 0b (P0) — "the real fix"
**Branch:** `bind-claims-to-facts` (from `main` at `525ce32`)
**Graph:** unchanged throughout — 385 episodes, 2,173 facts, level 1 12/12 retrievable.
`theme-build` was **not** run.
**Tiers:** as the previous slice — map `upstage/solar-pro4`, reduce `z-ai/glm-5.3-flash`,
rerank Voyage `rerank-3` (`top_n = 4`, floor 0.40), eval judge `deepseek-v4-flash`.

## 0. What this slice did and did not improve

**Citation correctness rose; the evidence-faithfulness of the content did not.** Claims
*correctly cited to the fact they rest on* went **24% → 80%**. Claims *supported by the facts
the reducer was given* went **95% → 86%** — slightly DOWN on the finer-grained after-audit.

So the judge's 1.6 → 4.7 is it finally being handed the facts an answer actually rests on,
not the answer becoming more true. That is a real product win — exact provenance is this
system's core promise and design decision #2 — but "global faithfulness improved 3×" would
read as better answers, and that is not what happened.

## 1. The defect, and what changed

`map_report` returns `key_points[]` and `fact_ids[]` as two unrelated lists. The reduce
block rendered the key points as prose and the fact ids as a bag of markers
(`Supporting facts: [1] [2] … [19]`), and the reducer **never saw a fact**. The per-claim
audit in `faithfulness-investigation-2026-09-10.md` §3.2 found 6 of 6 claims fully
supported by the facts the reducer was given and 0 correctly cited: it numbered its
sentences by position.

**`src/answer_api/global_search.py`:**

- `_fact_texts(driver, group_id, fact_uuids)` — one batched read
  (`MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $u RETURN f.uuid, f.fact`)
  for the ordered-unique union of every surviving map result's `fact_ids`. Not one per
  community, not one per fact (test-pinned).
- `_render_blocks(results, fact_to_marker, texts)` — each community block is now
  `COMMUNITY "title":` followed by one `[N] <fact text>` line per selected fact, in the map
  step's order. **The numbering is the existing `fact_to_marker` assignment, untouched**;
  `marker_map`, `_build_citations` and provenance resolution are unchanged. A fact selected
  by two communities keeps one marker and appears under both.
- A fact whose text cannot be read is dropped from the block **and** from `marker_map`
  (so the reducer cannot cite what it never saw), with a WARNING naming the uuids. If no
  fact is readable, the reduce step refuses **without calling the model** and still
  reports `communities_used`. Nothing renders as an empty line under a real-looking marker.
- `_REDUCE_PROMPT`: the preamble now says *"Each finding is shown as a marker [N] followed
  by the fact it refers to"*, and the first rule binds the citation to it: *"Cite every
  claim with the [N] marker of the fact it rests on: a claim drawn from the fact shown
  after [N] must cite that [N]. A sentence with no marker is not allowed."* The refusal
  trigger, meta-commentary ban, URL ban and the 0d no-range rule are byte-for-byte where
  they were (seven rules, test-pinned).
- `key_points` are no longer rendered. `map_report` and `MapResult` are untouched: the map
  step still writes key points as its own scaffold for choosing `fact_ids` (its prompt says
  "ONLY fact_ids that support the points you listed"), so the facts carry the filtering
  the points carried. §6 checks whether dropping them from the reduce input cost anything.

**Folded in, as agreed:**

1. **Markers per sentence** (`router_eval.markers_per_sentence`): the count of `[N]`
   markers on every sentence that carries at least one, with markers written after the
   full stop folded into the sentence before them. The eval records `mps_mean` / `mps_max`
   per question, prints them on the progress line, adds an `mps` column to
   `router-eval-report.md` and a by-mode aggregate (`mps_by_mode`). This is what makes
   bag-pasting (19 markers on one sentence) distinguishable from binding.
2. **`_range_markers` detection gaps** (`synthesize.py`): the pattern now sits inside a
   lookahead group so `[1]-[2]-[3]` reports both ranges; `re.IGNORECASE` for `[1] To [3]`;
   `\s*` instead of `[ \t]*` for a range wrapped across a line. Detection only.
3. **The rule count** (`tests/unit/test_reduce_prompt.py`): counts lines starting with
   `- ` instead of the substring anywhere. Demonstrated: editing the URL rule to contain
   `--` leaves the new test green (line count 7) where the old expression counts 8.

## 2. Tests — written failing first, mutation-verified

All five new test groups failed for the intended reason before the change (old block
shape `- kp` / `Supporting facts:`, zero fact-text queries, missing prompt phrase, the
three `_range_markers` gaps, `KeyError: 'mps_mean'`), then passed.

Mutation run (`mutate.py`: apply one mutation, run the test that claims to pin it,
restore, byte-compare). **16 of 16 mutations made their test fail; `git diff --stat src/`
empty afterwards.**

| mutation | test that failed |
|---|---|
| block prints `[N]` with no fact text | `test_block_is_marker_bound_fact_lines_not_key_points_plus_marker_bag` |
| block renders key points again | same |
| fact text read once per community | `test_fact_text_is_read_in_one_batched_query_for_the_union` |
| missing text rendered as `[2] ` (empty), marker kept | `test_fact_without_text_is_dropped_and_logged_not_rendered_empty` |
| missing text dropped from block but not from `marker_map`, no warning | same |
| no-text guard removed (reducer called with empty findings) | `test_no_fact_text_at_all_refuses_without_calling_the_reducer` |
| prompt no longer explains the `[N]`-then-fact format | `test_explains_marker_bound_fact_lines_and_binds_the_citation_to_them` |
| citation rule no longer binds the marker to its fact | same |
| range detector back to non-overlapping `findall` | `test_range_markers_reports_both_ranges_of_a_chain` |
| `IGNORECASE` dropped | `test_range_markers_word_separators_are_case_insensitive` |
| whitespace back to `[ \t]*` | `test_range_markers_spans_a_line_break` |
| leading-marker fold removed | `test_markers_per_sentence_folds_a_trailing_marker_only_segment_into_its_sentence` |
| uncited sentences counted as 0 | `test_markers_per_sentence_skips_uncited_sentences_and_headings` |
| aggregate drops `mps_by_mode` | `test_aggregate_reports_markers_per_sentence_by_mode` |
| report omits the `mps` column | `test_per_question_records_markers_per_sentence` |
| harness records `mps_mean`/`mps_max` as None | same |

Scoping: the reduce-prompt assertions run on the text between `Rules:` and `QUESTION:`
of `_REDUCE_PROMPT` only, so the "Absence of evidence" phrase shared with `_MAP_PROMPT`
cannot be matched by accident. The two existing fakes that reach the reduce step
(`test_global_reduce_relevance.py`, `test_empty_answer_guard.py`) now serve fact text for
the new query; no existing assertion was weakened.

## 3. Per-claim audit, before and after — the measurement that matters

Method: `global_search` end to end on the live graph (fresh shortlist → map → reduce →
provenance), `_complete_or_none` wrapped to capture the reduce prompt and raw output, the
marker map rebuilt with fact text. Per-claim verdicts from the same `_PER_CLAIM_PROMPT`
and the same eval judge model as the investigation, then **every claim of the deletion
answers hand-checked against the fact text** (the hand check is the primary number; the
judge is the reproducible one). Question: *"How do AWS Backup and Azure Backup each
protect recovery points from deletion?"* — the one that audited 6/6 supported, 0 cited.

### 3.1 Hand-checked, recovery-point deletion

| run | claims | supported by the facts given | **cited to the fact(s) they rest on** | partial | wrong |
|---|---|---|---|---|---|
| before rep 2 (today, current prompt) | 8 | 8 | **0** | 2 | 6 |
| after rep 1 | 12 | 12 | **11** | 1 | 0 |
| after rep 2 | 8 | 8 | **8** | 0 | 0 |
| after rep 3 | 23 | 23 | **23** | 0 | 0 |
| **after, total** | **43** | **43** | **42 (98%)** | 1 | 0 |

Before rep 2 is the investigation's pattern exactly: `[1] [2] [3] [4] [5] [6]` one per
sentence in order, with `[3]` (a governance-mode IAM fact) on the "3-day cooling-off"
claim, `[4]` (compliance immutability) on the "lifecycle retention prevents early
deletion" claim, and so on. Two partials: `[1]` and `[23]` each cover half their
sentence. Before rep 1 pasted all 30 markers across 10 sentences (max 7 per sentence).

After: every AWS claim cites the Vault Lock fact it paraphrases (`[1] [3]` governance
IAM, `[7] [8]` lock removal, `[10]` the 72-hour grace time, `[11] [12]` ChangeableForDays,
`[15]` "Always" retention, `[18] [19]` the 1-day / 36,500-day bounds) and every Azure
claim cites the soft-delete facts `[23]–[26]`. The single partial is after rep 1's
*"including the account root user"*, a detail that lives in `[13]` while the sentence
cites `[4] [5] [6] [9]`.

### 3.2 Judge-scored per-claim audit

Same `_PER_CLAIM_PROMPT` and judge model as the investigation, `reasoning.effort=low`
(see §7.5), every run of the two "before" questions and all 15 "after" runs. The judge
splits claims finer than I do (13 vs 12 for after rep 1) and agrees with the hand check
on every deletion run: before rep 2 → 0 of 9, after rep 1 → 11 of 13, after rep 3 → 24
of 25. "J1" is the eval's own faithfulness prompt, byte-for-byte, over the cited facts.

**Before** (current prompt, today):

| question | rep | claims | supported by any fact given | **cited to the fact it rests on** | full or partial | wrong | J1 | kept |
|---|---|---|---|---|---|---|---|---|
| recovery-point deletion | 1 (bag: all 30 pasted) | 11 | 10 | **5** | 8 | 3 | 5 | 30/30 |
| recovery-point deletion | 2 | 9 | 8 | **0** | 3 | 6 | 1 | 8/30 |
| recovery-point deletion | 3 | 9 | 9 | **2** | 3 | 6 | 1 | 8/30 |
| encryption (control) | 1 | 9 | 9 | **0** | 2 | 7 | 1 | 6/17 |
| encryption (control) | 2 (bag: all 17 pasted) | 4 | 4 | **3** | 4 | 0 | 5 | 17/17 |

Before total: 42 claims, 40 supported (95%), **10 correctly cited (24%)** — treat this as a LOWER BOUND of unverified tightness: the pre-change prompt carried no fact text, so the before mapping cannot be re-derived (see §7.6), unlike the after runs. The after figures are safe — and **2 of 27
(7%)** on the three runs that did not paste the whole bag. J1 mean 2.60. Note the control:
it scores 5 only when it pastes every marker; the non-bag encryption run is 0 of 9,
J1 = 1, the same defect as deletion.

**After** (this slice):

| question | rep | claims | supported by any fact given | **cited to the fact it rests on** | full or partial | wrong | J1 | kept |
|---|---|---|---|---|---|---|---|---|
| recovery-point deletion | 1 | 13 | 12 | **11** | 12 | 1 | 5 | 26/26 |
| recovery-point deletion | 2 | 13 | 13 | **9** | 12 | 1 | 5 | 26/30 |
| recovery-point deletion | 3 | 25 | 25 | **24** | 24 | 1 | 5 | 30/30 |
| encryption (control) | 1 | 14 | 8 | **8** | 10 | 4 | 4 | 13/17 |
| encryption (control) | 2 | 24 | 23 | **22** | 22 | 2 | 5 | 15/17 |
| encryption (control) | 3 | 11 | 9 | **9** | 10 | 1 | 3 | 14/17 |
| cross-region restore | 1 | 8 | 7 | **7** | 8 | 0 | 5 | 13/29 |
| cross-region restore | 2 | 9 | 7 | **7** | 9 | 0 | 4 | 9/31 |
| cross-region restore | 3 | 10 | 7 | **6** | 9 | 1 | 5 | 8/26 |
| compliance retention | 1 | 15 | 13 | **13** | 13 | 2 | 5 | 21/53 |
| compliance retention | 2 | 15 | 10 | **8** | 10 | 5 | 5 | 18/53 |
| compliance retention | 3 | 12 | 10 | **10** | 10 | 2 | 5 | 14/49 |
| database restore | 1 | 22 | 19 | **19** | 21 | 1 | 4 | 23/24 |
| database restore | 2 | 16 | 16 | **13** | 15 | 1 | 4 | 16/25 |
| database restore | 3 | 17 | 14 | **13** | 17 | 0 | 5 | 29/62 |

After total: 224 claims, 193 supported (86%), **179 correctly cited (80%)**, 202 full or
partial (90%). J1 mean **4.60** over 15 runs (2.60 before; the investigation's low
scorers were 1–2). **On the deletion question: 7 of 29 before (2 of 18 outside the bag
run) → 44 of 51 after.**

Two things the after table says that a headline number would hide. First, "correctly
cited" now tracks "supported" almost one-to-one (179 vs 193): when a claim is wrong it is
wrong on both axes, i.e. the residual is synthesis error, not binding error (§7.2 lists
every one). Second, the "supported" fraction fell from 95% to 86% on the finer-grained
after answers — the reducer now writes cross-vendor summary sentences ("Both services…")
that the judge holds to every specific, plus an uncited topic sentence per section.
Those are real, small, and a different defect.

### 3.3 Markers per sentence, before and after

Before = the 15 answers captured by the 0d slice this morning (current code, key points +
marker bag); after = 15 fresh runs, 3 per global golden question.

| | runs | facts available | citations kept | cited sentences | markers/sentence mean | max | sentences with ≥ 8 markers |
|---|---|---|---|---|---|---|---|
| before (0d capture) | 15 | 534 | 216 (40%) | 106 | 3.23 | **19** | **8** |
| after | 15 | 489 | 275 (**56%**) | 119 | 2.77 | **7** | **0** |

Distribution (markers per cited sentence → count): before `1:27 2:36 3:16 4:8 5:3 6:5 7:3
9:1 10:1 11:3 13:2 19:1`; after `1:27 2:35 3:21 4:20 5:9 6:4 7:3`.

**CORRECTION (review, 2026-09-11): "the tail is gone" is withdrawn.** That capture covered
only the five global-*intent* questions. In the eval all five drift-intent questions also
route to `chosen=global` and pass through the same reducer, and of the ten global-mode
answers **five carry a sentence with ≥8 markers** (16, 9, 9, 10, 19) — every one of which
scored 4-5. Eval global `mps` is 3.82/19 against the pre-change capture's 3.23/19. So
bag-pasting affects **half the mode**, not two outliers. Bound on the damage: discounting
the two worst answers to 2 still leaves ~4.1, and the 42/43 hand audit is judge-independent,
so the result survives — but the claim below was measured on a subset and must not be read
as mode-wide.

The 3–5-marker sentences are the ones that genuinely
rest on several near-duplicate facts (the graph holds `[11]` and `[12]` as two identical
"lock removed by users with sufficient IAM permissions" facts, and five overlapping
Min/MaxRetentionDays facts). Citation retention rose because the reducer now cites what
it uses, not because it pastes.

Per question, after (kept / available; mean / max markers per sentence): deletion 26/26,
26/30, 30/30 (3.7/7, 3.7/7, 2.2/5); encryption 13/17, 15/17, 14/17 (2.6/4, 2.8/5, 2.2/4);
cross-region 13/29, 9/31, 8/26 (2.4/6, 2.1/6, 2.1/5); compliance 21/53, 18/53, 14/49
(3.0/5, 3.2/5, 2.8/4); database restore 23/24, 16/25, 29/62 (2.8/6, 2.7/5, 3.5/7).

## 4. Full eval

`uv run --extra dev python -m answer_api.eval_router`, 29/29 questions, 0 failed, 1h44m
(11:28 → 13:12). Report: `router-eval-report.md` (now with the `mps` column and a
by-mode markers-per-sentence line).

| | 09-10 re-baseline | 0c run | 0d run (prev) | **this run** |
|---|---|---|---|---|
| Routing accuracy | 0.97 | 0.97 | 0.97 | 0.97 |
| Grounding precision | 0.85 | 0.88 | 0.88 | **0.92** |
| — global | 0.86 | 1.00 | 1.00 | 1.00 |
| Faithfulness mean | 3.92 | 4.11 | 3.75 | **4.90** |
| — global | 1.5 (8/10) | 2.44 (9/10) | 1.6 (10/10) | **4.7 (10/10)** |
| — local / timeline | 4.9 / 5.0 | — | 4.92 / 5.0 | 5.0 / 5.0 |
| **unscored** | 3/29 | 2/29 | 1/29 | **0/29** |
| ranges in any answer | — | — | 0/29 | 0/29 |
| comparative local / global / drift | — | — / 1.6 / 4.5 | 5.0 / 2.5 / 4.7 | 4.89 / **4.9** / 4.8 (`drift_wins` False) |
| markers per sentence, global (mean / max) | — | — | — | 3.82 / 19 |

Per global-mode question (chosen = global), against the two previous runs:

| question | intent | 0c | 0d (prev) | **now** | cited | mps mean/max |
|---|---|---|---|---|---|---|
| Compare how AWS Backup and Azure Backup encrypt backup data (control) | global | 2 | 5 | **4** | 14 | 3.0/7 |
| Compare cross-region restore between AWS Backup and Azure Backup | global | 4 | 2 | **5** | 14 | 2.6/6 |
| How do AWS Backup and Azure Backup each protect recovery points from deletion? | global | 5 | 2 | **5** | 28 | 3.1/7 |
| Which backup capabilities do AWS and Azure share for compliance retention? | global | 1 | 0 | **5** | 23 | 7.7/**16** |
| Compare AWS Backup and Azure Backup database restore workflows | global | 2 | 2 | **4** | 18 | 3.0/9 |
| What should I consider when planning long-term backup retention across cloud vendors? | drift | 1 | 1 | **5** | 89 | 3.8/9 |
| What should I consider for backup encryption across cloud providers? | drift | 2 | 1 | **4** | 22 | 2.9/10 |
| What are the key considerations for cross-region disaster recovery of cloud backups? | drift | 1 | 1 | **5** | 62 | 5.2/**19** |
| How should I approach immutability and ransomware protection for cloud backups? | drift | 4 | 1 | **5** | 8 | 4.0/4 |
| What should I think about when restoring databases from cloud backups? | drift | refused | 1 | **5** | 48 | 2.9/5 |

### Reading it plainly

- **Global faithfulness moved for the first time in five runs: 1.6 → 4.7 over 10/10,
  against local's 5.0.** Every one of the ten global-mode questions is at 4 or 5; the
  five that scored 0–2 last run are 4, 5, 5, 5, 4. The comparative forced-global figure
  moved the same way (2.5 → 4.9) and forced-drift held (4.7 → 4.8), so the aggregate is
  not one lucky draw — though it is still one run per question, on a map step whose fact
  counts vary 3× between runs (§5.2 of the previous two reports).
- **The judge score moved together with the audit, so the two measures did not separate
  here.** BACKLOG 2 predicted that binding could raise citation-precision without moving
  the judge; on this run it raised both, which is what §3.2's "correctly cited tracks
  supported" says should happen once mis-citation stops being the dominant error. The
  case that would separate them — an answer with high evidence-faithfulness and low
  citation-precision — no longer occurs in global mode; the split judge is still worth
  building, but as a guard rather than a diagnosis.
- **Two answers scored 5 while carrying a bag sentence** (16 and 19 markers on one
  sentence, on 53- and 62-fact pools). The judge cannot see that; the `mps` column can.
  They are the "shared capabilities" / "key considerations" summary sentences at the end
  of a large-pool answer — the same shape as the cross-vendor "Both…" sentences in §7.2.
- **8 synthesis empty-content retries across ~40 reduce calls** (the comparative pass
  runs global on every broad question), all recovered at 9,000 tokens, 0 gave up, 0
  judge-unmeasurable. The previous run had 5 on the smaller prompt. See §7.3.
- **Nothing unscored.** Q10 (the local refusal of the last three runs) answered this
  time; that is unrelated to this slice (local mode is untouched) and is why the overall
  mean is over 29 rather than 28.
- Not compared here: `theme-build` was not run, so the report layer is identical to the
  previous three evals.

### Caveats

- One run per question; the map step is not deterministic. A 4.7 → 4.0 next run would
  be inside the variance this eval has shown before, and would not be a regression from
  this change. What is not inside that variance is the per-claim audit (§3), which is
  the measurement this slice is judged on.
- The eval's judge still scores against *cited* facts only. It is now handed the right
  facts, which is why it agrees with the evidence-faithfulness reading (the investigation's
  J3, 4–5) — but an answer that cited nothing at all would still be scored against
  "(none)" and read as unsupported. Uncited topic sentences (§7.2) are the residual
  exposure.

## 5. Encryption control

The control (5/5, 17/17 cited last run) read **4 this run, 14 of 17 cited, mps 3.0/7**.
It did not bag-paste (last run's 5 came with all 17 markers on 4 sentences, up to 11 per
sentence — bag-pasting, not binding). The judge's per-claim notes on the three captured
after-runs (4, 5, 3) name the cost: one *"Both use KMS-integrated encryption"* sentence
(no fact says Azure uses KMS — it uses Key Vault), one *"Both services encrypt backup
data at rest with AES 256"* (AES 256 is only stated for Azure), and *"Data in the
Recovery Services vault is protected by an AES 256-based DEK"* (the fact does not say
where). Those are cross-vendor synthesis over-reaches, cited to the right facts; before
this slice the same answer scored 5 only when it pasted every marker and 1 when it did
not (§3.2, encryption rep 1: 0 of 9 correctly cited). The control did not regress on the
axis this slice changes; on the judge it is 4 vs 5 on a single run each, on a question
whose captured reps scored 4, 5 and 3.

## 6. Did dropping `key_points` cost anything?

A/B on the live graph: the block renderer monkeypatched to print the key points **and**
the fact lines (`Key points: / - … / Facts: / [N] …`), 3 runs on deletion and 2 on
cross-region, against the fact-only runs above.

| question | variant | facts available | citations kept | answer chars | cited sentences | markers/sentence mean / max |
|---|---|---|---|---|---|---|
| deletion | facts only ×3 | 26 / 30 / 30 | 26 / 26 / 30 | 1376 / 1548 / 2746 | 7 / 7 / 14 | 3.7/7, 3.7/7, 2.2/5 |
| deletion | facts + key points ×3 | 23 / 23 / 23 | 19 / 21 / 20 | 1410 / 1409 / 1444 | 7 / 6 / 7 | 2.9/4, 3.5/4, 2.9/4 |
| cross-region | facts only ×3 | 29 / 31 / 26 | 13 / 9 / 8 | 1194 / 1123 / 1088 | 8 / 7 / 8 | 2.4/6, 2.1/6, 2.1/5 |
| cross-region | facts + key points ×2 | 22 / 29 | 22 / 29 | 2445 / 2809 | 11 / 12 | 3.5/11, 3.8/7 |

On deletion the two variants are the same answer (same claims, same binding — the
key-points rep 1 hand-checks at 11/11 correctly cited). On cross-region the key-points
variant is 2.3× longer and cites every fact, and the extra length is a **"Posture
visibility"** section reciting the Resiliency-portal status list (`Protected`, `Pending
protection`, `Protection paused`…, facts 14–22) — the same off-topic community the
investigation found being "pressed into service" for a cross-region *restore* question.
The fact-only reducer left those facts uncited. Dropping key points removed a prose hop
that was steering the reducer toward tangential facts; it did not remove on-topic
content. **Keep the fact-only rendering.**

Judge-scored, the two variants bind equally well — key points: 47 of 58 claims correctly
cited (81%), J1 mean 4.75 over 4; facts only: 179 of 224 (80%), J1 4.60 over 15 — so
binding is carried by the fact lines, not the points. What the key points add is
visible in their cross-region run's unsupported claims: *"Azure Backup is part of the
Resiliency in Azure unified platform"* (judge: "No fact states this") and *"Azure
Backup's cross-region capability is framed as restore"* — the map step's own framing
sentences, re-asserted by the reducer with no marker. That is the mechanism the
investigation called key-point drift (§3.3, the one over-reach it found came from a key
point verbatim), and the fact-only reducer has no channel for it.

## 7. Things we had not considered

1. **Communities with key points but no valid `fact_ids` used to feed the reducer
   uncitable prose.** On deletion and encryption the old prompt rendered 4 community
   blocks; the new one renders 2. The other two (AWS Core Capabilities, MABS VMware
   recovery) returned key points and an empty `Supporting facts:` line — prose with no
   marker to attach, which is exactly the unsupported material the judge penalises. It is
   gone by construction, not by instruction.

2. **What is left after binding is not mis-citation.** Every "not fully supported" verdict
   in the after audit is one of three things: an *uncited topic sentence* ("AWS Backup
   protects recovery points through Vault Lock with two modes." — no marker at all,
   despite the rule), a *cross-vendor summary sentence* ("Both services encrypt backup
   data at rest with AES 256" — one genuine over-reach, "Both use KMS-integrated
   encryption", in encryption rep 1), or *judge strictness on paraphrase* ("copy" vs
   "restore" for Cross Region Restore). Comparison sentences need two markers and tend to
   get none; that is the next binding target if one is wanted.

3. **The reduce prompt roughly doubled** (2.0–3.0k chars → 2.9–8.5k; 53-fact compliance
   runs are the largest) and the synthesis tier's empty-content retry fired **once in 15
   captures** (compliance rep 3, 7.5k chars, recovered at 9,000 tokens) and **once in the
   eval** (Q16, the encryption control). `_prefer_fast_provider` still does not bound
   reasoning for the synthesis tier — BACKLOG 5b, now with a larger input to trip on.

4. **The eval's compliance answer still has one 16-marker sentence** (`mps 7.7/16` on 23
   citations, 53 facts available). With a very large pool the reducer writes a bag
   sentence for the "shared capabilities" summary. Before this slice that answer would
   have been indistinguishable from a bound one; the column now shows it.

5. **The per-claim judge is unusable at full reasoning.** `deepseek-v4-flash` with
   `max_tokens=12000` ran over ten minutes per answer and returned unparseable JSON;
   `reasoning.effort=low` returns valid JSON in ~100 s but truncates on 20+-claim answers
   at 8,000 tokens (16,000 works). Anything that builds the split judge (BACKLOG 2) on a
   per-claim prompt needs this bounded up front.

6. **An instrument bug that looked like a product bug.** My capture script numbered the
   fact union in the order the map calls *completed*, while `global_search` numbers in
   shortlist order; the first after-run read as citing exactly one community's worth of
   facts off. The after prompt carries `[N] text`, so the truth was recoverable and the
   captures were renumbered and verified against every prompt line. Any future trace
   script should number from `communities_used` order or parse the prompt.

7. **Near-duplicate facts inflate markers per sentence.** The graph holds `[11]` and `[12]`
   as two byte-identical "lock removed by users with sufficient IAM permissions" facts and
   five overlapping Min/MaxRetentionDays facts; a bound sentence legitimately cites all of
   them. Dedup at map selection (or at extraction) would make 3–5-marker sentences rarer
   without changing what is claimed.

## 8. CI gate

On the final source (`39d0d73`, no source change after it): `uv run ruff check src tests`
clean; `uv run mypy src` clean (69 files); `uv run --extra dev pytest -m "not live"`
**711 passed, 15 deselected in 10m47s** (Neo4j/Postgres testcontainers).
