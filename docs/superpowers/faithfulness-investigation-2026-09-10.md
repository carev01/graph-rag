# Global-mode faithfulness: what the 2.3 actually measures, and what would move it

**Date:** 2026-09-10
**Branch:** `rerank-selection`
**Graph:** live, unchanged from the eval runs — 385 episodes, 2,173 facts (2,118 current),
41 communities detected / 37 retrievable.
**Method:** read-only Neo4j queries; three global questions and one local question run
through the real code path with every intermediate captured (`shortlist → map →
reduce prompt → raw answer → marker_map → citations`); the configured eval judge
(`deepseek-v4-flash`) run in four variants over each captured answer; a per-claim audit
hand-checked against fact text. ~50 LLM calls in total, no rebuilds, no eval run.
Scripts and captured traces are in the session scratchpad
(`trace_global.py`, `trace_local.py`, `judge_experiments.py`, `shortlist_levels.py`,
`traces/*.json`).

---

## 1. Verdict in three sentences

1. **The metric is trustworthy for what it measures, and what it measures is not
   invention.** It measures whether each claim is supported by the facts the answer
   *cites*. In global mode the reduce step is handed no claim→fact binding at all, so
   it attaches markers by position; the answers it writes are supported by the evidence
   it was given (judge score 5/5/5 and 4/5/5 against that evidence) but score 0–2 against
   the markers it attached. **The residual 2.3 is a citation-binding defect, not
   invented content** — and none of the three prior slices touched binding.
2. **The community level global mode reads from is missing Azure Backup.** The level-1
   community holding Azure Backup (219 entities, 705 intra-community facts, 67 citable)
   was rejected by `--verify-pending` while the since-removed summary rule was still a
   rejection reason, and nothing has repaired it. At level 1, **0 of 10** golden
   global/DRIFT questions surface an Azure Backup community; at levels 0 and 2, **10 of
   10** do. Every AWS-vs-Azure comparison in the golden set is therefore half-unanswerable
   at level 1 by construction. This is a corpus-completeness (report-layer) defect caused
   by a bug that was fixed without repairing its damage.
3. **The report layer is a very lossy intermediate even when intact.** Global mode can
   cite 247 of 2,118 current facts at level 1 (11.7%), 350 across all levels (16.5%). The
   report writer for the largest AWS community sees 34% of that community's facts (context
   budget) and cites 6% of them.

The sequenced recommendation (§6): repair the lost level (or read level 0, the only
complete one) **first**, because it is the cheapest and it invalidates half the current
eval; then fix the binding so the reduce step sees fact text with per-fact markers; then
split the judge into two numbers so this confusion cannot recur.

---

## 2. What I measured

### 2.1 Traces

| key | question | eval score (last run) | communities | markers | cited |
|---|---|---|---|---|---|
| `xregion` | Compare cross-region restore between AWS Backup and Azure Backup. | 1 | 4 | 29 | 5 |
| `deletion` | How do AWS Backup and Azure Backup each protect recovery points from deletion? | 1 | 4 | 25 | 7 |
| `encrypt` | Compare how AWS Backup and Azure Backup encrypt backup data. | 5 | 4 | 13 | 13 |
| `local_pitr` | How does AWS Backup support continuous backups and point-in-time recovery? (local) | 5 | — | 15 retrieved | 10 |

Extraction is not deterministic (BACKLOG 7), so these are fresh generations, not the
eval's exact text. The patterns below held on every trace.

### 2.2 Judge variants

All use the eval judge model at temperature 0. `J1` is byte-for-byte the eval's prompt
and fact rendering (`_cited_fact_texts` order, unnumbered `- fact` lines).

| variant | facts shown to the judge | numbered? |
|---|---|---|
| **J1** (= the eval) | only the facts the answer cited | no |
| J2 | only the facts the answer cited | by marker |
| **J3** | **every fact the reduce step was given** (the full `marker_map`) | no |
| J3b | every fact the reduce step was given | by marker |
| per-claim | every fact, numbered; JSON verdict per claim: supported by *cited* marker / by *any* fact / best marker | — |

---

## 3. Finding 1 — the judge is consistent, and it is scoring citation binding

### 3.1 The numbers

| trace | J1 (eval, cited only) ×3 | J2 (cited, numbered) ×3 | **J3 (all evidence given to reduce)** ×3 | J3b |
|---|---|---|---|---|
| `xregion` | **1, 1, 2** | 2, 2, 2 | **4, 5, 5** | 3 |
| `deletion` | **2, 2, 0** | 2, 1, 1 | **5, 5, 5** | 5 |
| `encrypt` (global, eval 5) | 5, 5, 5 | 5, 5, 5 | 5, 5, 5 | 5 |
| `local_pitr` (local, eval 5) | 5, 5, 5 | 5, 5, 5 | 5, 5, 5 | 5 |

Two things are visible at once. The judge is reasonably stable (±1 across repeats, one
outlier 0). And the **same answer scores 1–2 against what it cited and 4–5 against what it
was given.** The three-point gap is the whole distance between global (2.3) and local
(5.0).

The two 5-scoring rows are the control. `encrypt` cited **13 of 13** markers it was given,
so "cited facts" and "given facts" are the same set and the binding defect cannot bite —
the map blocks were also one key point to one or two markers, so position happened to be
right. `local_pitr` is local mode: the synthesiser saw `[N] fact` lines and cited 10 of 15
correctly. The low scorers cited **5 of 29** and **7 of 25**. Whether a global answer
scores 5 or 1 under the current judge is, on these traces, a function of how much of its
marker pool it happened to cite — not of what it said.

### 3.2 Per-claim audit — `deletion` (every claim supported, zero markers right)

| claim (abridged) | cited | supported by cited? | supported by any fact? | fact that actually supports it |
|---|---|---|---|---|
| Vault Lock applies an immutable, WORM configuration | [1] | partial | **full** | [1] [2] |
| Compliance mode: after grace time nothing can be altered or deleted by any user or AWS | [2] | **none** | **full** | [7] [8] |
| Governance mode: users with sufficient IAM privileges can manage or remove the lock | [3] | partial | **full** | [3] [6] [12] |
| Enforces min/max retention on new backup and copy jobs, preventing early deletion | [4] | **none** | **full** | [14]–[17] |
| Retention "Always" → retained forever, cannot be altered or deleted after grace time | [5] | **none** | **full** | [19] |
| Access controlled through IAM policies and the Backup Vaults console page | [20][21] | partial | **full** | [20] [22] |

Six claims, **six fully supported by the evidence**, **zero correctly cited**. Look at the
cited markers: `[1] [2] [3] [4] [5]` — the reduce model numbered its sentences in order.
It was given `Supporting facts: [1] [2] … [19]` under the community block and no way to
know which fact backs which point, so it did the only thing it could. I checked each row
against the fact text myself; the judge's verdicts are right.

### 3.3 Per-claim audit — `xregion`

| claim (abridged) | cited | by cited | by any | actual support |
|---|---|---|---|---|
| AWS Backup provides cross-Region copy for some resource types | [1] | full | full | [1] |
| Available for RDS, EBS, Redshift Serverless in all Regions where they operate | [2] | partial | full | [2] [3] [4] |
| Cross-Region copy between China (Beijing) and China (Ningxia) in either direction | [23] | full | full | [23] |
| Azure Backup is integrated into the Resiliency platform with monitoring alongside ASR jobs | [24] | **none** | **none** | — (a stretch of "Resiliency allows viewing jobs across ASR" + "Reporting in Resiliency … for Azure Backup") |
| Retention details for protected items shown for primary and secondary regions | [27] | **none** | full | [28] [29] |

Five claims: four supported by the evidence, one mild over-reach that came from the *map*
key point verbatim (the only key-point drift I found in three traces — and note it is
"integration" language, not an invented number or limit). Two markers wrong, one right
only because the block had a single marker.

### 3.4 What this says about BACKLOG item 2

Item 2 worried that an unnumbered fact list lets a **misattributed-but-plausible marker
pass**. The direction is wrong. The reducer cites 5–7 of 25–29 markers, so when a marker
is misattributed the fact that *would* support the claim is usually **not in the cited
set**, the judge never sees it, and the claim **fails**. Numbering the cited facts (J2)
barely moves the score (2,2,2 vs 1,1,2; 2,1,1 vs 2,2,0), because the judge is already
finding the cited facts don't say what the sentence says. The metric was not "looser than
we believed"; it was **stricter along an axis nobody was working on**.

This also explains the slice history exactly:

| slice | why it moved / didn't |
|---|---|
| reduce-prompt binding (1.6, flat) | the reducer already used only its input; the input had no binding to obey |
| **report verification (2.33, moved)** | invented claims are unsupported under *both* readings (cited and any); removing them helps whichever axis you score |
| reranked selection (2.30, flat) | changes which communities arrive; the reducer still gets a bag of markers per community |

### 3.5 Other properties of the metric worth knowing

- **Refusals are unscored, partial refusals are scored.** `_faithfulness_judge` skips an
  answer only if the *whole* answer equals a refusal string. The `deletion` answer
  embedded *"I don't have enough thematic coverage…"* under an **Azure Backup** heading and
  was scored. Local mode's 5.0 is a mean over answers local chose to give: my
  `local_azenc` run ("How does Azure Backup encrypt backup data?", eval score 5) came back
  as a full refusal with 15 facts retrieved and would have been dropped from the mean.
- **Meta-commentary is free.** The `encrypt` answer ends with *"The community findings do
  not cover how Azure Backup encrypts backup data"* — forbidden by `_REDUCE_PROMPT`, but
  the judge scores claims, so an answer that covers only half the question and says so can
  score 5. Which is what happened in the eval.
- `(relevance 0.61328125)` is rendered into the reduce prompt as a raw float. Harmless
  noise, but it is text handed to the model that means nothing to it.

---

## 4. Finding 2 — the level global mode reads is missing Azure Backup

### 4.1 Detected versus retrievable, per level

| level | detected | retrievable | citable facts (retrievable) | of which AWS / Microsoft | **lost** citable facts | lost community |
|---|---|---|---|---|---|---|
| 0 | 19 | 17 | 234 | 121 / 113 | 74 (all Microsoft) | *Resiliency in Azure: Unified BCDR Platform* (36), *Azure Resiliency: Security Posture* (38) — both **staged**, recoverable |
| **1 (production)** | 12 | 11 | 247 | 120 / 127 | **67 (all Microsoft)** | ***Azure Backup Security, Encryption, and Cross-Region Resiliency*** — **rejected**, not recoverable without regeneration |
| 2 | 10 | 9 | 219 | 46 / 173 | 70 (all AWS) | *AWS Backup: Core Capabilities, Workload Coverage* — **rejected** |

Level 0 is the only level with nothing rejected. The vendor split of *retrievable* level-1
facts (120/127) hides the topic skew: the 127 Microsoft facts at level 1 are the
Resiliency **portal** community (100), MABS VMware recovery (17) and small UI fragments.
The **Azure Backup service** — vault, soft delete, encryption, Cross Region Restore, SQL
restore — lives in the one community that is gone. Its 705 intra-community facts are 701
Microsoft / 4 AWS.

### 4.2 The golden articles, by level

Where the facts of each golden `expected_article_id` are citable, and whether that
community is retrievable:

| golden article | L0 | **L1** | L2 |
|---|---|---|---|
| Azure Cross Region Restore | OK | **LOST** | OK |
| Azure Backup encryption | OK | **LOST** | OK |
| Azure soft delete (×2) | OK | **LOST** | OK |
| Azure SQL restore | OK (+ RecoveryPointTypes) | **LOST** (+ RecoveryPointTypes) | OK |
| Azure encrypted VM | OK | **LOST** | OK |
| AWS cross-Region copy | OK | OK | LOST |
| AWS Vault Lock | OK | OK | OK |
| AWS encryption (KMS) | OK | OK | partly LOST |
| AWS PITR | OK | **not citable at all** | LOST |
| AWS cross-account | OK | OK | LOST |

At level 1, **every Azure golden article is unreachable** and the AWS PITR article is not
cited by any level-1 report.

### 4.3 What the shortlist actually returns (embedding + rerank, no LLM)

For all 10 golden global/DRIFT questions, number of Azure Backup communities among the
survivors:

| level | questions with an Azure Backup community in the shortlist |
|---|---|
| 0 | **10 / 10** (ranked 1st or 2nd on 8) |
| **1 (production)** | **0 / 10** |
| 2 | **10 / 10** |

The reranker is not at fault: at levels 0 and 2 it puts *Azure Backup: Encryption, Soft
Delete, and Cross-Region Resiliency* first for the encryption, cross-region and database
questions (0.76, 0.71, 0.59). At level 1 there is nothing for it to find, so the slots go
to MABS VMware recovery and the Resiliency portal — which is exactly the "junk third
community" the map-step trace complained about. **The junk is not a threshold problem; it
is what is left when the right community does not exist at that level.**

### 4.4 Does it explain the low scores?

Partly — and the partition is clean. What the reducer does with the missing half:

| trace | Azure half | effect on faithfulness |
|---|---|---|
| `deletion` | writes the refusal string under an **Azure Backup** heading | none directly — but the AWS half then scores 0–2 on binding (§3.2) |
| `xregion` | presses Resiliency-portal facts into service ("retention details for primary and secondary regions") | one unsupported claim, one mis-cited |
| `encrypt` | meta-commentary sentence | none — judge ignores it; answer scored 5 |

So the missing level does **not** by itself explain a 0–2: `deletion` is pure AWS content
and scores 0–2 for binding alone. What the missing level does is (a) make every
comparison question a half-answer, (b) pull in off-topic communities whose markers dilute
the pool, and (c) invalidate the eval as a measurement of *comparison* quality — you
cannot measure how well the system compares AWS to Azure when Azure is absent from the
level it reads. Both defects are real; they stack.

### 4.5 How it was lost, and why nothing repaired it

`map-step-trace-2026-09-09.md` (post-fix section) records the sequence: full rebuild →
5 reports staged → `--verify-pending` → *1 promoted, 2 rejected, 2 still pending*, with the
note that at the time *"the summary verdict was still a rejection reason inside
--verify-pending"* and that on measured rates *"the likelier cause is the summary rule."*
The rejected pair are the two large, vendor-defining communities above (L1 Azure Backup,
L2 AWS Core). Rejection removes `pending_*` and leaves the node with `cited_fact_uuids`
but no `summary`, `full_report` or `embedding`. `write_communities_incremental` does not
regenerate a report-less community that is otherwise clean, and no counter reports "a
level has lost a community". The summary-rule bug was fixed (§4.4 of the verification
spec) but the graph state it produced was never rebuilt.

---

## 5. Finding 3 — the report layer is a narrow window even where intact

| community (level 1) | members | intra-community facts | facts that fit the report context (12k-token budget) | facts the report cites |
|---|---|---|---|---|
| AWS Backup: Cross-Account and Cross-Region Copy, Vaults, and Encryption | 260 | 929 | 315 (34%) | 59 (6%) |
| Azure Backup Security, Encryption, and Cross-Region Resiliency (lost) | 219 | 705 | 307 (44%) | 67 (10%) |
| Resiliency in Azure: Unified BCDR Posture Management Platform | 71 | 148 | 148 (100%) | 100 (68%) |

Two cuts, both silent: the context assembler drops two-thirds of the big communities'
facts (recency-first, so it is at least a *consistent* two-thirds), and the report writer
cites a tenth of what it sees. The result is that a question routed to global mode can, at
best, be answered from ~12% of the corpus, and the map step then selects 1–20 of those.
This is why local mode, which retrieves from all 2,118 facts and hands the synthesiser
numbered fact text, is both more faithful *and* more specific. The map-step trace's
"gap-filling" hypothesis (the model invents when facts are generic) is really this: the
facts that reach the reducer are generic because the specific ones were cut upstream.

This is the deepest of the three problems and the least urgent: fixing §3 and §4 first
will show how much of it still matters.

---

## 6. Where I disagree with the project's own conclusions

- **BACKLOG 2 (judge "looser than we believe").** Wrong direction — see §3.4. The judge
  is stricter, not looser, on the binding axis, and it is *right* to be: a reader who
  clicks `[4]` on the retention claim gets the compliance-mode fact. That is a product
  defect. It is just not the defect three slices were aimed at.
- **Map-step trace: "the drift starts in the map step" → later "the report writer."**
  Both were about invention, and invention is now largely gone (one soft over-reach in
  three traces). The trace's third finding — the "junk" third community admitted at
  `relevance_min = 2` — was misread as a threshold problem. It is a coverage problem
  (§4.3); no threshold produces an Azure Backup community at level 1.
- **Rerank measurement "the reranker fixes the ordering, selection must come from
  top_n."** True, and also beside the point: at level 1 the top-4 cannot contain the
  right community for 10/10 questions.
- **Verification spec §10: "if the graph's facts are too generic… honest reports will be
  thinner… that is the correct outcome."** Correct as far as it goes, but `findings_dropped
  = 13` was read as "the community layer was not largely embellishment". The number to
  have watched was *communities lost*, which the same run reported as 2 and attributed to
  a since-fixed rule without checking which two.

---

## 7. Sequenced strategy

### Step 0 — Repair the report layer, then re-baseline (hours, ~50 LLM calls)

1. Regenerate the two rejected communities (L1 Azure Backup, L2 AWS Core) and run
   `--verify-pending` for the two staged L0 communities. The rejection happened under a
   rule that no longer exists, so a regeneration under the current rules is the fix, not
   a workaround. This is a targeted rebuild, not `theme-build --full`; if the CLI cannot
   target a community, add that before anything else.
2. Add a `communities_lost_by_level` counter to `theme-build` and make an incremental run
   treat a report-less community as dirty. A level with a lost community is a degraded
   answering surface and should say so.
3. **Until (1) is done, consider `global_default_level = 0`.** It is the only level where
   nothing is lost, it is finer-grained (19 communities), and the shortlist puts the right
   Azure community first on 8/10 golden questions. Measured cost: same `rerank_top_n = 4`,
   so no extra map calls. Not a permanent answer — the point of level 1 was broader
   themes — but it is a one-line change that makes the eval meaningful tomorrow.
4. Re-run the eval. Expect global to move some, and expect *comparative* answers to become
   two-sided for the first time. This is the baseline everything after should be measured
   against; the current 2.3 is a measurement of a half-populated layer.

### Step 1 — Give the reducer fact text with per-fact markers (the binding fix)

The map step currently returns `key_points[]` and `fact_ids[]` as unrelated lists, and the
reducer sees prose plus a bag of markers. Two options, in order of preference:

- **(a) Drop key points from the reduce input.** Make the map step a *selector*: return
  only the fact ids that bear on the question (it already validates them against the
  report). Render the reduce prompt as `[N] fact text` lines grouped by community — the
  same shape local mode uses — and let the reducer synthesise from facts it can see.
  This removes an LLM prose hop, makes binding trivially checkable, and is what the
  measured J3 scores say the answers already deserve (4–5). Prompt size grows from ~25
  short key points to ~25 fact lines: negligible.
- **(b) Keep key points but bind them.** Return `[{point, fact_ids}]` and render each
  point with its own markers. Cheaper change, still leaves the reducer paraphrasing an
  LLM paraphrase.

Prediction to test: on `deletion` and `xregion`, J1 rises to within one point of J3.
If it does not, the per-claim audit (§7 step 2) will show exactly which claim drifted.

### Step 2 — Make the judge report two numbers

Replace the single 0–5 with a per-claim structured verdict (the `_PER_CLAIM_PROMPT` used
here is a working draft) and aggregate two metrics:

- **evidence faithfulness** — fraction of claims supported by *any* fact the mode handed
  its synthesiser (requires the envelope, or the eval, to keep the full `marker_map`;
  today only cited facts survive into `citations`);
- **citation precision** — fraction of claims supported by the fact(s) they cite.

Number facts by marker in both. Keep refusals unscored, but count embedded partial
refusals and meta-commentary sentences as a separate hygiene metric rather than letting
them vanish or score 5. Retain the old J1 for continuity until two runs agree.

### Step 3 — Widen the report layer's window (later, after re-measurement)

Only once Steps 0–2 are in: raise `report_token_budget` or select facts by degree rather
than recency alone for the big communities; consider whether `leiden` granularity should
keep communities under the budget. Decide from the new per-claim numbers whether
generic-fact answers are still the dominant failure.

### Do not do

- More prompt constraints on map or reduce. Both are already faithful to their inputs.
- Model-tier swaps for the map or reduce step. The failures traced here are structural.
- Reranker threshold tuning for the "junk community" symptom. See §4.3.

---

## 8. Uncertainty, stated

- **n is small**: three global traces and one local. The binding mechanism is not in
  doubt (it is visible in the code and in the literal `[1]…[5]` numbering), but the J1→J3
  gap size (≈3 points) comes from two questions. Step 0's re-run gives the real number.
- **One judge model.** All variants used `deepseek-v4-flash`. I hand-checked every
  per-claim verdict in §3.2 and §3.3 against the fact text and agree with all of them; I
  did not cross-check with a second model.
- **Why the L1 Azure community was rejected is inferred**, from the trace document's own
  note about the summary rule, not from a log. The remedy (regenerate under current
  rules) is the same either way; if it is rejected *again* under the current rules, that
  is a new and important fact.
- **Level 0 as a stopgap is measured on selection only** (which communities surface), not
  on answer quality. It could be worse in ways this investigation did not test.
- **The eval's local 5.0 is a selected mean** (refusals excluded). Its true quality on
  the questions it declines is unmeasured. This does not change any recommendation but
  it does mean "local is 5.0" overstates the contrast.
