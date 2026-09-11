# Map-step meta-commentary: hypothesis test, fix, and re-baseline

**Date:** 2026-09-10
**Backlog item:** 0c (P0)
**Branch:** `map-meta-commentary`
**Graph:** unchanged throughout — 385 episodes, 2,173 facts, 40 communities, level 1
12/12 retrievable. `theme-build` was **not** run.
**Tiers:** map `upstage/solar-pro4`, reduce `z-ai/glm-5.3-flash`, rerank Voyage `rerank-3`
(`top_n = 4`, floor 0.40), eval judge `deepseek-v4-flash`.

## 1. The hypothesis, and how it was tested

BACKLOG 0c: the reduce step refuses two of five global golden questions because the map
step hands it key points about what a report *lacks* ("The provided report contains no
information about Azure Backup"), and `_REDUCE_PROMPT`'s refusal rule reads those as
"the findings do not support an answer".

Per `map-step-trace-2026-09-09.md`, the hypothesis was tested before anything was
changed. Method:

1. **Capture** the real shortlist + map output once for the two refusing questions and
   the encryption control (`scratchpad/capture_map.py`), so every reduce experiment
   below ran on the *same* map output and paid no further map calls.
2. **Drive the real `global_search`** with `shortlist_communities` / `map_report`
   monkeypatched to return the captured output, under three conditions:
   - **A** — as captured (meta-commentary present);
   - **B** — the meta-commentary key points removed, communities kept
     (some now with an empty key-point list);
   - **C** — B, plus communities left with no key points dropped entirely.
3. `_complete_or_none` was wrapped to record whether a refusal was **written by the
   model** or was the **None path** (`finish_reason=length` twice → `_REFUSAL`), which
   the answer string alone cannot distinguish.

### Captured map output (old prompt)

| question | survivors | key points | of which meta-commentary | fact_ids |
|---|---|---|---|---|
| database restore | 4 | 7 | 3 | 15 |
| compliance retention | 4 | 12 | 3 | 33 |
| encryption (control) | 4 | 6 | 1 | 17 |

Meta-commentary reproduced on every question, including the control ("The report does
not describe Azure Backup encryption."). On the compliance question two of the four
communities returned **only** a meta-commentary point with zero fact_ids.

### Reduce outcomes on identical map output

| question | A: as captured | B: meta removed | C: B + empty communities dropped |
|---|---|---|---|
| compliance retention (run 1, N=3) | **2/3 refused** | 0/3 | 0/3 |
| compliance retention (run 2, N=6) | **4/6 refused** | 0/6 | — |
| **compliance, combined** | **6/9 refused** | **0/9** | 0/3 |
| database restore (N=3) | 0/3 | 0/3 | 0/3 |

Every refusal was **model-written** — none came from the None path (0 "giving up"
lines across all runs; the reduce tier hit `finish_reason=length` on its first attempt
4 times in ~36 calls and recovered at 9,000 tokens each time).

### Verdict on the hypothesis

**Held for the compliance-retention question**, cleanly: the only thing that differed
between 6/9 refusals and 0/9 was three meta-commentary sentences; the 33 fact_ids and
nine substantive points were identical. B and C did not differ, so an empty key-point
list is *not* itself a refusal trigger.

**Not reproduced for the database-restore question.** On this run's map output it
answered 3/3 even with the meta-commentary in. Its eval-time refusal therefore rests on
a different map output (the map step is not deterministic — the eval trace saw 29
fact_ids, this capture 15) or on the None path, and this investigation cannot attribute
it. It is *consistent* with the hypothesis — this run's meta-commentary on that question
was milder ("does not contain any information about Azure Backup database restore
workflows" beside real AWS points) — but that is a reading, not evidence.

## 2. The first fix over-corrected, and what that revealed

The first wording of the ban (commit `d23650c`) told the map step: state only what the
report says, never what it lacks, and return an empty `key_points` list when nothing in
the report bears on the question. Unit test green, mutation-verified. Then the live
end-to-end re-check (3 questions × 3 reps, fresh map calls) came back **worse**:

| question | old prompt (eval) | ban v1 |
|---|---|---|
| database restore | refused | **2/3 refused** |
| compliance retention | refused | **3/3 refused** |

The refusals had changed cause. Under v1 `solar-pro4` returned an **empty** list from
communities that plainly hold relevant content: the AWS Vault Lock community
(Compliance mode, retention enforcement, Cohasset assessment) went from 5 points /
15 fact_ids to **0 / 0 in 3 of 3 runs** on the compliance question; on database restore,
rep 1 had all four communities return nothing. The model read "this report covers only
one vendor of a two-vendor comparison" as "nothing here bears on the question" and took
the empty-list exit. Forbidden from narrating the gap, it withheld the evidence instead.

One rep is instructive on its own: compliance rep 1 produced 8 Azure-only points,
32 fact_ids, zero meta-commentary — and reduce still refused, correctly, because a
"what do AWS and Azure *share*" question with one side of the evidence has nothing to
compare. **The reduce refusal rule is doing its job; it is the map step that must
deliver both sides.**

So the structure of the defect is: comparative questions are mapped per community, each
community covers one vendor, and the map model reacts to that mismatch by either
narrating the gap (old prompt) or withholding the evidence (v1). The prompt has to say
explicitly that a report covering one side of the question still returns that side.

## 3. What changed (v2, the shipped wording)

`_MAP_PROMPT` in `src/answer_api/global_search.py` now carries, in this order:

1. *The question may span several vendors or topics and this report may cover only one
   of them: return what the report says about the part it covers, as fully as the report
   supports, and leave the rest to other reports.*
2. *Every key point must state something the report SAYS. Do NOT write a point about what
   the report does not contain, does not mention, or cannot compare, and do not describe
   the report's scope. Absence of evidence is not a finding.*
3. *Return an empty key_points list only when the report says nothing about any subject
   of the question.*

`_REDUCE_PROMPT` is **unchanged**. The refusal trigger was not softened (step 3 of the
brief) because the refusals were gone after step 1 — see §4.

Map-only probe of v2 on the captured hits (2 reps per question, no reduce calls):

| question | rep | points | meta | fact_ids | single-vendor communities contributing |
|---|---|---|---|---|---|
| database restore | 1 | 5 | 0 | 18 | Azure 4 pts, AWS core 1 pt |
| database restore | 2 | 12 | 0 | 45 | Azure 7 pts, AWS core 5 pts |
| compliance retention | 1 | 12 | 0 | 53 | Vault Lock 6, Azure 5, Resiliency 1 |
| compliance retention | 2 | 17 | 0 | 67 | Vault Lock 8, Azure 8, Resiliency 1 |

Zero meta-commentary, both vendors present every time, and more cited facts than the
old prompt produced (15 / 33).

**Test:** `tests/unit/test_global_map_meta_commentary.py` asserts the rendered map
prompt carries the ban, the empty-list instruction, and the partial-coverage clause.
Mutation-verified: removing "Absence of evidence is not a finding." → `1 failed`;
removing the empty-list instruction → `1 failed`; the partial-coverage assertion was
added while v1 was committed and failed against it (`assert 'cover only one' in ...`)
before v2 was written. Restored → `1 passed`, `git diff --stat src/` empty.

## 4. Live before / after

End-to-end `global_search` (fresh shortlist → map → reduce → provenance), 3 reps per
question, same graph, same tiers. "urls" is the number of distinct resolved source URLs
across the answer's citations, split by vendor domain.

| question | old prompt (eval 2026-09-10) | ban v1 | **ban v2 (shipped)** |
|---|---|---|---|
| Compare AWS Backup and Azure Backup database restore workflows | refused | 2/3 refused | **0/3 refused** — 25 / 4 / 18 cites; 12 (aws 7, azure 5) / 4 (3, 1) / 9 (4, 5) urls |
| Which backup capabilities do AWS and Azure share for compliance retention? | refused | 3/3 refused | **0/3 refused** — 2 / 6 / 17 cites; 1 (1, 0) / 4 (1, 3) / 3 (1, 2) urls |
| Compare how AWS Backup and Azure Backup encrypt backup data (control, scored 5) | answered | 0/3 refused | **0/3 refused** — 17 / 17 / 6 cites; 8 (5, 3) / 8 (5, 3) / 3 (2, 1) urls |

Meta-commentary in the map output across the nine v2 runs: **0** (the old prompt had it
on every question). Every v2 answer cited both vendors' documentation except compliance
rep 1, which cited only `vault-lock.html` — the reducer wrote a two-sided answer but
attached markers only to the AWS half; the Azure facts were in its input.

Sample v2 answer, database restore (rep 1, 25 citations resolving to 12 URLs):

> AWS Backup supports database restore workflows across Aurora (with continuous backups and a
> single recovery point updated when retention changes), SAP HANA on EC2 via AWS Backint with
> transaction log backups for point-in-time restore, RDS and Aurora clusters, DynamoDB tables,
> and Redshift Serverless [14]…[25].
>
> Azure Backup supports SQL Server databases on Azure VMs as a core workload, including
> TDE-enabled databases [1]…[13]. Restoring a TDE-encrypted SQL database to another SQL
> Server requires restoring the certificate to the destination server [1]…[13]. Azure also
> offers Cross Region Restore for SQL in Azure VM, enabling restore to a secondary region
> [1]…[13].
>
> In summary, AWS Backup's database restore coverage spans multiple managed database
> services plus SAP HANA, while Azure Backup's documented database restore workflow centers
> on SQL Server in Azure VMs, with an added certificate prerequisite for TDE-encrypted
> restores and cross-region restore capability.

Citations `[14]`–`[19]` resolve to `docs.aws.amazon.com/aws-backup/latest/devguide/`
`plan-options-and-configuration.html`, `point-in-time-recovery.html`,
`point-in-time-recovery-retention-period.html`, `working-with-supported-services.html`,
`backup-feature-availability.html`; `[1]`–`[13]` to `learn.microsoft.com/en-us/azure/backup/`
pages. All graph-resolved; the LLM wrote no URL (design decision #2 unchanged).

The refusal rule in `_REDUCE_PROMPT` was **not** touched: step 1 removed the refusals,
so step 3 of the brief was never reached.

## 5. Full eval

`uv run --extra dev python -m answer_api.eval_router`, 29/29 questions, 0 failed,
2h05m, started 2026-09-10 22:59. Report: `router-eval-report.md`.

| | pre-repair (damaged corpus) | 2026-09-10 re-baseline | **this run** |
|---|---|---|---|
| Routing accuracy | 0.97 | 0.97 | 0.97 |
| Grounding precision | 0.88 | 0.85 | **0.88** |
| — global | — | 0.86 | **1.00** |
| Faithfulness mean | 4.07 | 3.92 | **4.11** |
| — global | 2.30 | 1.5 | **2.44** |
| **unscored** | **0/29** | **3/29** | **2/29** |
| global-mode questions scored | 10/10 | 8/10 | **9/10** |

Per global-routed question (chosen = global):

| question | intent | prev | **now** |
|---|---|---|---|
| Compare how AWS Backup and Azure Backup encrypt backup data | global | 5 | **2** |
| Compare cross-region restore between AWS Backup and Azure Backup | global | 1 | **4** |
| How do AWS Backup and Azure Backup each protect recovery points from deletion? | global | 1 | **5** |
| Which backup capabilities do AWS and Azure share for compliance retention? | global | **refused** | **1** |
| Compare AWS Backup and Azure Backup database restore workflows | global | **refused** | **2** |
| What should I consider when planning long-term backup retention across cloud vendors? | drift | 1 | 1 |
| What should I consider for backup encryption across cloud providers? | drift | 1 | 2 |
| What are the key considerations for cross-region disaster recovery of cloud backups? | drift | 1 | 1 |
| How should I approach immutability and ransomware protection for cloud backups? | drift | 1 | 4 |
| What should I think about when restoring databases from cloud backups? | drift | 1 | **refused** |

### Reading it plainly

- **The defect is fixed as specified.** The two refusing global questions answer, with
  citations, in the eval as well as in the nine live runs of §4. All five global-intent
  questions are scored; last run two were not.
- **Global faithfulness is roughly flat against the pre-repair 2.30, not improved.** 2.44
  over nine questions is within the noise of a five-question intent set whose map step
  varies 3× in fact count between runs (§6.5). Reporting "1.5 → 2.44" would be
  misleading: the 1.5 averaged eight questions with the two hardest missing.
- **The encryption control dropped 5 → 2.** In the nine live runs it never refused and
  cited both vendors every time; a 2 is consistent with the range-shorthand mechanism in
  §6.3 (an answer judged against its endpoint citations only), but the harness does not
  persist answer text, so that cannot be confirmed for this run. It did not regress in
  the sense the brief asked about — it answers, cites both vendors, and the three live
  runs scored it against 17 / 17 / 6 resolved citations — but its score is lower.
- **One new refusal**: "What should I think about when restoring databases from cloud
  backups?" (drift intent, routed to global) refused this run after scoring 1 last run.
  It is not one of the questions this fix targeted, and a drift-shaped question with no
  vendor named is the kind of thin evidence the reduce rule is meant to refuse on — but
  it was not traced, and it is the remaining global-mode unscored entry. The other
  unscored question is Q10 (local), unchanged from last run.
- **The None path did not fire**: 10 reduce first attempts exhausted 3,000 tokens and
  every retry recovered (0 "giving up"). 3 judge skips = Q10 + Q25 + one comparative
  sub-run refusal.
- Comparative global 1.6 (prev 1.56); drift 4.5 (prev 4.7); `drift_wins` still False.

`_REDUCE_PROMPT` was not softened; the global figure was not made to move by relaxing
the refusal rule.

## 6. Things we had not considered

1. **Banning the gap-narration makes the map step withhold evidence unless told
   otherwise.** §2. The partial-coverage clause is the load-bearing part of the fix, not
   the ban. Any future rewording of `_MAP_PROMPT` must keep it; the unit test guards the
   literal clause and its docstring says why.

2. **The reduce refusal rule is correct and should stay.** Compliance v1 rep 1 (8 Azure
   points, 32 fact_ids, no meta-commentary, refused) shows it refusing a one-sided answer
   to a "what do they share" question. That is the behaviour two slices built. Pressure
   belongs on the map step and the shortlist to get both vendors in front of it.

3. **Range shorthand silently discards citations.** The reducer sometimes writes
   `[1]–[26]` instead of listing markers; `_finalize_answer` keeps only markers that
   appear literally, so the citation list carries the two endpoints. In the nine v2 runs:

   | run | facts given to reduce | citations kept | ranges written |
   |---|---|---|---|
   | database restore rep 2 | 25 | **4** | 10 |
   | compliance rep 1 | 59 | **2** | 1 |
   | compliance rep 2 | 53 | **6** | 6 |
   | the other six | 17–59 | all, or 6–17 with no ranges | 0 |

   The eval judge scores an answer against its **cited** facts only, so a range-writing
   answer is judged against 2–6 of the facts it actually drew on and every other claim
   reads as unsupported. This is a mechanical, deterministic contributor to global
   faithfulness that sits underneath BACKLOG 0b (marker bag) and 2 (judge split), and it
   is cheap to fix — expand `[a]–[b]` / `[a]-[b]` in `_finalize_answer`, or forbid the
   shorthand in `_REDUCE_PROMPT` — but it is a different defect from 0c and was not
   touched here. It should be measured before 0b is attempted, because 0b's "feed the
   reducer [N] fact lines" design will make ranges *more* attractive to the model, not less.

4. **The refusal string hides two mechanisms.** `_REFUSAL` is returned both when the
   model refuses and when `_complete_or_none` gives up (`finish_reason=length` twice on
   `glm-5.3-flash` at 3,000 → 9,000 tokens). Across ~36 reduce calls here the first attempt
   exhausted its budget 4 times and the retry recovered every time, so the None path did
   not fire — but had it, the eval would have recorded an "honest refusal" that was an
   outage. Worth a distinct `degraded` reason (`_DEGRADED_DISCLAIMERS` already has the
   shape) so the eval can count them apart. Same family as BACKLOG 5 / 5b.

5. **The database-restore refusal was never reproduced** (§1). The map step is not
   deterministic at temperature 0 — fact_ids for the same question and communities
   ranged 15 → 45 between captures — so single-run eval numbers on five global questions
   carry more variance than the reports have been treating them with.

6. **Q10 ("How does Azure Backup encrypt backup data?", local) is unscored again**, as
   in the previous run. It is a local-mode refusal, not this defect, and is left alone.

7. **Stale testcontainers.** Nine `neo4j` containers from earlier test runs were still
   up on this host (some for days). Not caused here, but they cost memory and the
   integration suite ran slower than its usual ~11 min with the eval alongside it.
