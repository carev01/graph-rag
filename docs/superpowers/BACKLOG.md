# Project Backlog

Single tracking file for everything outstanding. Created 2026-09-09 after the
answer-citation-integrity slice, consolidating: the Fable whole-project review
(`project-review-2026-09-04.md` §3, §5, §6, §7), the deferred Slice B from
`specs/2026-09-08-answer-citation-integrity-design.md` §8, findings from the
temporal-coherence slice, and the follow-ups this slice produced.

**Conventions.** Each item states what it is, why it matters, and where the evidence
lives. Items marked **[verify first]** were recorded on 2026-09-04 and the code has
moved since — confirm before acting. Nothing here is scheduled; ordering within a
section is by my recommended priority.

---

## P0 — Blocks trusting the system's own numbers

### 0. ~~Restore level 1, then re-baseline~~ — **DONE 2026-09-10**
Level 1 is whole again: **19/19, 12/12, 9/9 retrievable, 0 staged, 0 dead.** The
"Azure Backup: Encryption, Soft Delete, and Cross-Region Resiliency" community (219
entities, 705 facts) is back at level 1, so global search sees both vendors again.

How it went, because the sequence matters:
1. `theme-build --full` under the fixed rules — 35 written, 5 staged, 0 destroyed. Under
   the old summary rule those 5 would have been deleted.
2. `--verify-pending` promoted 2, rejected 0.
3. The remaining 3 would not promote no matter how often retried, because the verifier was
   failing **deterministically** on large reports — see the merged fix below.
4. With reasoning bounded, one more `--verify-pending` cleared every remaining staged
   report: 1 promoted, 0 rejected, **0 still pending**.

`lost_by_level` now reports per-level losses so this cannot recur silently on the `--full`
path; since 2026-09-11 the incremental path reports it too (5d), and on both paths a failed
regeneration carries the persisted report over rather than deleting it (5c).

**Re-baselining is NOT done.** The eval has not been re-run against the repaired corpus,
so every faithfulness number on record — including global 2.30, and the 1.6 → 2.33 gain
attributed to report verification — was measured on a corpus missing its main Azure
community. Those numbers describe a damaged graph. Re-run before drawing conclusions, and
note `global_default_level = 0` is no longer needed as a workaround.

### 0c. ~~The reduce step over-refuses, and map meta-commentary is feeding it~~ — **DONE 2026-09-11**
Fixed on branch `map-meta-commentary`; full account in `map-meta-commentary-fix-2026-09-10.md`.
The hypothesis held for the compliance question (6/9 refusals with the meta-commentary,
0/9 without, on identical map output) and was not reproduced for database restore. The
first wording of the ban **over-corrected** — solar-pro4 returned nothing from
single-vendor communities on two-vendor questions — so the shipped `_MAP_PROMPT` also says
a report covering one side of the question returns that side. `_REDUCE_PROMPT` untouched.
Eval: all five global-intent questions now scored (was 3/5); global 2.44 over 9/10 vs
pre-repair 2.30 over 10/10 — **flat, not a gain**. New findings that are not this item:
**range shorthand `[1]–[26]` collapses citations to endpoints** (3 of 9 live runs: 25→4,
59→2, 53→6 facts; the judge then scores against the endpoints only — measure before 0b),
and `_REFUSAL` conflates model refusal with the `_complete_or_none` None path (10
length-exhaustions this eval, all recovered). Original trace kept below for the record.

**Traced live 2026-09-10 on the repaired corpus.** Two of five global golden questions now
**refuse** where they previously answered, which is what dragged the re-baseline to global
1.5 with 3/29 unscored.

It is not selection and not missing evidence. On *"Compare AWS Backup and Azure Backup
database restore workflows"*:
- the shortlist returns 4 survivors, with the restored Azure community ranked #1 (0.5781);
- the map step returns 4 results with **29 fact_ids** across substantive AWS *and* Azure
  database-restore content;
- and the reduce step refuses anyway.

**The likely trigger is map-step meta-commentary.** Among the key points handed to reduce:
*"The provided report contains no information about Azure Backup."* and *"The provided
report does not contain information about AWS Backup."* The reduce prompt was taught to
refuse when the findings do not support an answer (citation-integrity slice), so a finding
that literally says "no information" invites exactly that — while 29 cited facts sit
alongside it.

**I retired item 3a prematurely**, marking it superseded because the map step was shown
faithful. Faithful it is — it does not invent. But it still emits meta-commentary, which is
a different property, and the ban written into `_REDUCE_PROMPT` was never applied to
`_MAP_PROMPT`.

**Do, in order:** ban meta-commentary in `_MAP_PROMPT` (a key point must state something the
report says, never what it lacks); then re-check whether the refusals persist; only then
consider softening the reduce refusal trigger. Do not soften the trigger first — an honest
refusal on thin evidence is correct behaviour and worth keeping.

**Until this is fixed the re-baseline is not comparable**: 3 unscored refusals make the
global figure an average over 8 of 10 questions.

### 0d. ~~Range shorthand silently discards citations~~ — **DONE 2026-09-11**
Fixed on branch `no-range-citations`; full account in `no-range-citations-2026-09-11.md`.
One rule added to `_REDUCE_PROMPT` (cite markers individually, never as a range);
`_finalize_answer` now WARNs when a range survives, with markers-cited vs facts-available;
the eval report carries `cited` / `ranges` columns. **Live: ranges 17 → 0 over 15 runs
(model complied); citations kept did NOT recover — 40% of available facts vs 39% at
baseline.** The reducer's alternative to `[14]–[25]` is one marker per sentence, assigned
by position -- the problem moved from *quantity* of citations to *correctness* of them.

**Correction, per review:** "0d was a symptom of 0b" is *consistent with* the evidence but
not established by it. Ascending positional markers are equally explained by a benign
confound that was never ruled out: the reducer writes community-by-community and facts are
numbered in that same order, so a *correctly bound* answer would also read ascending within
a block. The real evidence for 0b remains its own per-claim audit (6/6 claims supported, 0
correctly cited), which is prior work, not this slice's.

**Kept despite the metric.** Reverting would restore a *user-facing* defect -- the envelope
silently dropping citations the prose rests on -- to protect a judge score that is an
instrument, not the deliverable. Note the change may itself depress the score: where a range
gave 2 endpoint citations, positional numbering gives ~5, handing the judge more mismatched
pairs. This run cannot separate that from variance, and the aggregate retention ratio cannot
detect it (it is a per-claim distribution effect, and map-step fact counts varied 3x between
the two measurements).

**Riding with 0b** (detection-only, no user impact): `_range_markers` misses `[1]-[2]-[3]`
as two ranges (non-overlapping `findall`), capitalised `"[1] To [3]"`, and line-wrapped
forms -- add `re.IGNORECASE` and `\s*`. And `rules.count("- ") == 7` counts the substring
anywhere, so editing a rule to contain `--` would fail a test whose docstring claims the
tuned rules were disturbed; count lines starting with `- ` instead.

**New measurement gap:** pasting 19 markers onto one sentence is indistinguishable from good
citation in every metric we have. Add a "markers per sentence" column when 0b lands.
**Eval: global faithfulness 1.6 over 10/10 scored (prev 2.44 over 9/10); 0 ranges in all
29 answers; the encryption control recovered 2 → 5 (17/17 cited).** Fourth flat-or-down
global result in a row; the `cited` / `ranges` columns now show low scores sitting on
12–28 citations with no range, which is 0b's signature. Original entry kept below.

**Found 2026-09-10 during the 0c fix; mechanism verified independently.** The reducer writes
citation ranges like `[1]-[26]`. `_finalize_answer` keeps only literal markers, so that
yields exactly **two** citations for prose resting on 26 facts:

```
_finalize_answer("AWS and Azure both encrypt at rest [1]-[26].", MM)
  -> cited == [1, 26]
```

Measured live: 3 of 9 runs kept **4, 2 and 6** citations out of **25, 59 and 53** available
facts. The faithfulness judge scores against *cited* facts only, so an answer well-supported
by 25 facts is judged against 2 and reads as unsupported. This is likely a material,
deterministic contributor to the low global scores that three slices failed to move.

**This revisits a decision I made and defended.** The citation-integrity spec (§3 rule 2)
rejected a range expander on the grounds that "a model emitting a 30-marker span is guessing,
not citing, so expanding it would legitimise the pattern and manufacture citations the model
never really made." That reasoning still holds on integrity grounds — and it has a
measurement cost nobody knew about.

**The fix is neither expanding nor keeping the status quo:** instruct the reducer to cite
markers individually and never as a range. That preserves the integrity guarantee (no
manufactured citations) and stops discarding genuine ones. Expanding remains rejected.

**The 0c fix increased exposure to this.** A reviewer traced the chain: the new map prompt
yields far more fact_ids per community (measured 15 → 45), longer marker lists make range
shorthand more attractive to the reducer, `_finalize_answer` collapses `[a]-[b]` to its two
endpoints, and the judge scores only those. That is the most plausible explanation for the
encryption control dropping 5 → 2 in the 0c eval, and it means global scores cannot be
compared meaningfully until 0d is fixed.

**Measure this before starting 0b.** 0b (binding claims to their facts) will make ranges
*more* attractive to the reducer, so leaving this unfixed would confound 0b's result — and
0b's whole purpose is to make citation correctness measurable.

### 0b. ~~Bind claims to their supporting facts in the reduce step~~ — **DONE 2026-09-11**
Fixed on branch `bind-claims-to-facts`; full account in `bind-claims-to-facts-2026-09-11.md`.
The reduce block is now `[N] <fact>` lines (one batched fact-text read, numbering
unchanged, missing text dropped loudly); key points are no longer rendered (A/B showed
they added only tangential prose). **Per-claim audit on the deletion question: 0–2 of 9
claims correctly cited before → 9–24 of 13–25 after (hand-checked 42/43); across all 15
after-runs 179/224 (80%) vs 10/42 (24%) before. Eval: global faithfulness 1.6 → 4.7 over
10/10, overall 3.75 → 4.90, unscored 0/29.** The diagnosis held. Folded in: the
`markers per sentence` eval column (two large-pool answers still carry a 16- and
19-marker bag sentence despite scoring 5), the `_range_markers` detection gaps, and the
line-based rule count. Left open: uncited topic sentences and cross-vendor "Both…"
summary sentences are now the residual error (synthesis, not binding); the reduce prompt
is 2–3× larger and tripped the synthesis tier's empty-content retry 8 times in ~40 calls
(all recovered) — BACKLOG 5b. Original entry kept below.

**Review corrections, 2026-09-11:**
- **What this did NOT improve.** Claims *supported by the facts given* went **95% → 86%**,
  slightly down, while *correctly cited* went 24% → 80%. The judge's 1.6 → 4.7 is it being
  handed the facts an answer actually rests on — exact provenance arriving (design decision
  #2) — not answers becoming more true. The tables read the other way; they should not.
- **"The tail is gone" is WITHDRAWN.** That capture covered only the five global-*intent*
  questions, but all five drift-intent questions also route to global. Of the ten
  global-mode answers **five carry a sentence with ≥8 markers** (16, 9, 9, 10, 19), all
  scoring 4-5. Bag-pasting affects **half the mode**, not two outliers. Discounting the two
  worst to 2 still leaves ~4.1 and the hand audit is judge-independent, so the result
  survives — but `mps` only exposes it, nothing thresholds it. Add a "share of citations in
  ≥8-marker sentences" measure.
- **The before-24% is a LOWER BOUND**, not a tight figure: the pre-change prompt carried no
  fact text, so that mapping cannot be re-derived. The after figures are safe.
- `communities_used` still lists communities that contributed no block.

`_MAP_PROMPT` returns `key_points[]` and `fact_ids[]` as **two unrelated lists**, and the
reduce block renders `Supporting facts: [1] [2] … [19]` as a marker bag. The reducer cannot
know which fact backs which point, so it numbers sentences **by position** — one traced
answer cites `[1]…[5]` in order.

Per-claim audit of a low-scoring answer: **6 of 6 claims fully supported by the facts the
reducer was given, 0 correctly cited.** Judge scores 0-2 against the *cited* facts versus
5/5/5 against the facts *given*. **That ~3-point gap is the global-vs-local gap.** The
control that scored 5 cited 13/13 markers, so cited = given and binding could not bite.

**Do:** feed the reducer `[N] fact` lines selected by the map step, instead of key points
plus a marker bag. Predicted to close the gap. Do NOT add prompt constraints, swap tiers, or
tune rerank thresholds — three slices have now shown those do not touch this.

### 0e. Bag-pasting is now measured — and the harness stops discarding its evidence — **MEASURE LANDED 2026-09-11, unthresholded**
Branch `eval-bag-share`. `bag_share` = share of an answer's citation MASS sitting in
sentences of **≥8 markers** (`BAG_MARKERS`), reported per question as a `bag` column and
aggregated `bag_share_by_mode`. It answers what `mps` cannot: `mps_max` is identical for an
answer with one 19-marker bag beside thirty clean claims and an answer that is *only* that
bag. Returns **None, never 0.0**, when an answer cites nothing — 0.0 is the *best* score on
this scale, so a refusal earning it would read as perfect citation hygiene (the same
coerce-a-failure-into-a-legitimate-value class as [[llm-empty-reply-coerced-to-value]]).

The threshold is 8 because the 0b distribution is bimodal — bags of 16, 9, 9, 10, 19 and
nothing cited between 7 and 9. That is five observations at one corpus size; re-check it
before treating 8 as anything but a working cut.

**Nothing thresholds it yet, and that is not deferral for its own sake** — no policy should
be set before the number is seen on a real run. What the number is for: if `bag_share` is
high, 0b's 80% correct-citation figure is inflated by bags that are trivially "correct"
because they cite everything.

**The structural half matters more than the metric.** Every scored column this harness has
gained — `cited`, `ranges` (0d), `mps` (0b), `bag` (now) — cost a **full paid eval run** to
observe, because `run_eval` scored each answer and then threw it away. The run now writes
`docs/superpowers/router-eval-raw.json` (git-ignored): answer text, cited fact texts and
citations per question, including the comparative passes. The next question asked of a run
is a re-read, not a re-purchase.

## P1 — Known-wrong behaviour in shipped code

### 2. Split the judge into evidence-faithfulness and citation-precision — **reframed**
**The original framing had the direction backwards, corrected 2026-09-10.** This item used
to argue the judge was too *loose* — that a misattributed-but-plausible marker would pass,
so we were optimising against a weaker metric than we believed. The investigation found the
opposite: misattributed markers **fail**, because the reducer cites only 5-7 of 25-29
markers and the true supporting fact is usually not in the judge's set at all. Numbering the
cited facts barely moves scores.

The judge is **stricter** than assumed, on an axis nobody was working on. That is exactly
why report verification moved the number (invented claims fail under any reading) while
reduce-prompt binding and reranking could not.

**What to do instead:** score two things separately — *is this claim supported by the
evidence given?* and *does its marker point at the right fact?* Today those are averaged
into one number that cannot distinguish "made it up" from "cited the wrong line", which is
why three slices chased the wrong causes. Sequence after items 0 and 0b, since 0b is
expected to move citation-precision sharply and the split is what will prove it.

### 4. ~~Slice B: out-of-range dedup indices during extraction~~ — **DONE + MEASURED 2026-09-11**
`slice-b-dedup-indices-2026-09-11.md`. `graph_extract.dedup_guard` wraps the dedup LLM
call: recovers N/M from the prompt, classifies every out-of-range index (inside the
invalidation range = the confusion signature, vs beyond it = hallucinated), counts per
article on `IngestArticleResult.dedup`, and re-issues the pristine prompt on the strong
tier before graphiti sees the reply (`dedup_retry_on_strong`, **default OFF** — see below).
Both impact claims confirmed from the code and pinned by tests, with one bound: a dropped
contradicted index only costs an invalidation when the candidate has an earlier
`valid_at`. Schema `maximum` was rejected as harmful (constrained decoding would turn
`[10,11]` into an in-range wrong merge).

**MEASURED LIVE** (17 articles, **1,145 dedup calls**, run stopped early because the
result was not in doubt): **85 calls (7.4%) carried an out-of-range index; 154 invalid
duplicate indices, 154 of them (100%) inside the invalidation range, 0 beyond it, 0
negative, 0 `contradicted_beyond_range`, 0 `parse_failures`.** The index-space-confusion
hypothesis is **CONFIRMED** — every invalid value was an invalidation-candidate index in
the `duplicate_facts` field, none hallucinated.

**And `gpt-5-mini` repeats the mistake on 20 of 84 retries (24%)**, with graphiti's own
drop warning still firing 21 times *after* the retry. A far stronger model reading the
identical prompt fails the same way, so this is a **prompt/interface defect, not model
quality** — and the retry is a ~76% partial recovery costing a slow strong-tier call on
7.4% of every dedup call in the corpus. The default was flipped True → False on that
evidence. Reinterpreting the indices stays rejected: an invalidation candidate does not
share endpoints, so promoting one to "duplicate" would manufacture a permanent wrong
merge — the same objection that killed schema `maximum`. Not fixed: duplicates already in the graph; the strong tier's own
events (counted, not retried); the search cap of 10 same-pair candidates.

Original entry, kept for the record. Deferred deliberately from the citation-integrity
slice — see `specs/2026-09-08-answer-citation-integrity-design.md` §8.

graphiti logs `LLM returned invalid duplicate_facts idx values [10,11,14,15] (valid
range: 0-9)` at `edge_operations.py:735` and **drops them**, so dedup silently misses
duplicates and duplicate facts survive in the graph. Our `maxItems` bound caps array
*length*, not element *values*, so it cannot prevent this. Seen with `solar-pro4` and
previously `mistral-nemo`.

Verified feasible: it is a real `logger.warning` on
`graphiti_core.utils.maintenance.edge_operations` with structured args, so a handler
attached during `ingest_article` can count occurrences per article; and `_tier_for`
(`ingest_driver.py:53`) already selects a tier per article, so "retry this article on the
strong tier" fits the existing shape. If in-flight proves infeasible, handle post-run.

Separate slice because it is a different subsystem (ingest, not answer), a different test
surface, and validating it requires a re-ingest cycle.

### 5. ~~`router.py:72` — the last `content or ""` site~~ — **DONE 2026-09-11**
Closed with 5b (branch `answer-path-client-hardening`): the classifier reply goes through
`usable_content`; an empty reply (no choices, or `content=None`) is logged as
`cheap classifier ... returned no usable content (finish_reason=...)` and defaults, and
an empty `choices` list no longer raises `IndexError` into the generic except. Still
nothing aggregates `routing.via == "default"`. Original entry kept below.

`max_tokens=8` against a possibly-reasoning classifier, then
`(resp.choices[0].message.content or "").strip()`. An empty reply falls through silently
to `default_mode`. Two problems: the unguarded `resp.choices[0]` (empty `choices` list
raises IndexError) and the silent fallthrough with no log.

Same bug class as the four already fixed — see the `llm-empty-reply-coerced-to-value`
note. Use `synthesize._usable_content`. Observable today as `routing.via = "default"`,
but nothing aggregates it, so a regression would be invisible. Routing accuracy 0.97 was
measured with the current classifier, so it is not biting *now*.

### 5b. ~~Answer-path clients: no timeout, and UNBOUNDED REASONING~~ — **DONE 2026-09-11**
Fixed on branch `answer-path-client-hardening`; full account in
`answer-path-hardening-2026-09-11.md`. `_prefer_fast_provider` moved to
`graph_extract.usage.prefer_fast_provider` (re-exported by `theme_builder.report`) with a
`bounded_llm_client` builder that all four answer-path tiers use: 180s timeout / 3 retries
(classifier 20s / 2), throughput routing, per-tier reasoning bound. **Eval: empty-content
retries 8 → 0 in ~40 reduce calls, wall-clock 1h44m → 26 min, slowest question 528s →
184s; routing 0.97, grounding 0.92, unscored 0/29 unchanged; faithfulness 4.90 → 5.00 is
synthesis-output variance, not judge leniency (judge probed: still 5/4/0/0 on a
faithful/one-invented/mostly-invented/unsupported set).**

**The named fix was wrong for two of the four tiers.** Measured: `upstage/solar-pro4`
(map + classifier) does not reason by default, and `reasoning: {effort: low}` SWITCHES
THINKING ON — every `max_tokens=8` classifier reply came back empty, and one of two
`max_tokens=2000` map calls burned the whole budget on reasoning. The bound is a property
of the model, so the settings are per tier (`synthesis_reasoning_effort="low"`,
`map_reasoning_effort=""`, `eval_judge_reasoning_effort="low"`; classifier hard-wired to
none). Left open: `theme_builder/report.py` still builds its two clients by hand (fold
into BACKLOG 20); nothing thresholds a per-request `/answer` SLA (BACKLOG 10). Original
entry kept below.

**Escalated 2026-09-11 by 0b.** The reduce prompt is now 2-3x larger and empty-content
first-calls rose to **8 of ~40 reduce calls** (previous run 5). All recovered via
`_complete_or_none`'s 3x retry, so nothing was lost — but a fifth of reduce calls now pay a
4x retry and the largest prompts are exactly the ones that trip it.

**The fix is already written and proven twice here.** `_prefer_fast_provider`
(`theme_builder/report.py:133`) bounds reasoning effort and routes by throughput. It is
applied to the report tier and the verifier tier and **still not to synthesis**. Apply it to
the synthesis client with `effort: low` exactly as the verifier fix did — and NEVER
`enabled: false`, which was measured returning a 13-token rubber stamp.

Timeouts (original entry) belong to the same client construction; do both together.
</details>
`report.py` was fixed on 2026-09-08 after a measured 380-second outlier: OpenRouter routes
the same model to different providers (29 tok/s vs 9.4 tok/s on the same prompt), so it
got `timeout=180.0, max_retries=3` plus `_prefer_fast_provider` (throughput sort). Its own
comment warns "unpinned, a 60-community build is anywhere from 30 minutes to 6 hours."

**That fix was never generalised.** Every answer-path client is still bare:

- `synthesize.py:176` — `AsyncOpenAI(api_key=..., base_url=...)`, no timeout, no routing
- `global_search.py:105` — same
- `eval_router.py:66` — same
- `router.py:54` — same

Two consequences. **In production**, an `/answer` request has no timeout, so a single bad
provider route can hang a user's query indefinitely with nothing to cut it off.
**In the eval**, observed 2026-09-10: a run that previously took ~20 minutes took over 100,
with only 2 retry warnings in the log — so the slowness is route latency, not retries.

Fix: extract the report-tier client construction (bounded timeout + throughput preference)
into one shared helper and use it for every tier. This is the third time in one session a
fix was applied at one call site instead of the layer that needed it — see
[[llm-empty-reply-coerced-to-value]] for the same pattern.

### 5c / 5d. ~~`rep is None` drops a community; incremental cannot report `lost_by_level`~~ — **DONE 2026-09-11**
Fixed on branch `preserve-verified-reports`. One rule, as the item proposed: **a failed NEW
attempt never removes what is already persisted.** On the incremental path, when
`generate_report` returns `None` with nothing staged (the `except Exception` route and the
`_generate_once` → `None` route — measured 3/29 empty-`choices` replies on a real run), the
community's persisted entry is now carried over instead of being omitted from `entries` and
deleted by the rebuild's `DETACH DELETE`.

- A **verified** persisted report is carried over **with its embedding** — it stays
  retrievable, because losing it from retrieval is the harm being avoided.
- A **staged** one is carried over in its staged shape, from `pending_summary` /
  `pending_full_report`, which `load_persisted` now reads. Writing it normally would have
  published an *empty* report: a staged row's `summary` reads back as `''`.
- Only a community with **nothing persisted** is a real loss. It is counted, logged with its
  level, and reported as `lost_by_level` (5d) — the caller accumulates it, since `entries`
  holds only survivors. Also new: `reports_preserved`.

**The non-obvious half — a preserved report must not look fresh.** It describes the member
set it was written against, and the carrying run stamps a new `corpus_cursor`, so the edits
that made it dirty fall *behind* the next run's watermark and it would read as clean
forever. A new `stale` flag (persisted, read by `load_persisted`, honoured by `classify`)
forces it dirty until it is actually regenerated. It keeps its embedding throughout — only
its dirtiness is forced, never its retrievability. `verified` was deliberately **not**
reused for this: that property means "its findings passed verification", which is still
true.

**The `--full` path now holds the same rule.** `_run_theme_build` loads the persisted layer
and matches it against the fresh detection, so a community whose regeneration fails is
carried through `write_communities`' new `preserved` argument instead of being deleted by
the `DETACH DELETE` rebuild. `--full` means "regenerate everything", not "destroy what we
have if the regeneration fails". Fixed alongside it: `PARENT_OF` edges were only written
between communities present in `reports`, so a staged **or** preserved community silently
lost its place in the hierarchy on the very run that saved it — the gate is now the full set
of communities this run wrote.

Original entries kept below for the record.

### 5d. ~~The incremental path cannot report `lost_by_level`~~ — **P1**
`write_communities` now reports per-level losses, which is what would have caught level 1
draining. `write_communities_incremental` **cannot**: it receives only the surviving
`entries`, so a genuinely-skipped community's level never reaches it — and incremental is
the DEFAULT path, so the blind spot persists exactly where routine runs happen.

The fix is in the caller: `_run_theme_build_incremental` already loops
`for i, c in enumerate(communities)` and knows `c.level` for every community it puts in
`skipped` rather than `staged`. It needs to accumulate its own skipped-by-level dict and
merge it into the result, the way it already adds `reports_skipped`.

### 5c. ~~`rep is None` still drops a community on the non-staged paths~~ — **P1**
Staging (2026-09-10) covers only the verifier-blip path. The other `rep is None` routes —
`except Exception` around report generation, and `_generate_once` returning `None` (measured
3/29 empty-`choices` replies on a real theme-build) — still omit the community from
`entries`, so `write_communities_incremental`'s `DETACH DELETE` removes it along with its
**previously verified** report.

Related, weaker: even on the staged path, a blip costs the community its retrievability
until `theme-build --verify-pending` runs, and a later genuine rejection loses the older
verified text permanently.

**One rule fixes both:** when `rep is None` for any reason, fall back to `matches[i]`'s
persisted verified entry rather than omitting it. Nothing verified should leave retrieval
because a *new* attempt failed.

### 5e. DRIFT-intent question misroutes to global and now refuses — **P1**
Surfaced 2026-09-10 while closing 0c, and recorded here so it stops living only inside
another item's closure note. The drift-intent question *"What should I think about when
restoring databases from cloud backups?"* routes to **global** and now returns a refusal
where it previously scored 1.

Not a regression in any strict sense — routing accuracy is unchanged at 0.97, the misroute
predates this work, and a worthless answer becoming an honest refusal is arguably an
improvement. But it is untraced, and the drift→global misroute pattern affects all five
drift-intent questions in the golden set, which is why global-mode figures average ten
questions rather than five.

Trace it alongside 0d, since range-shorthand citation loss is a candidate cause for the
underlying low scores.

### 6. graphiti's false invalidations (the "43 phantom invalidations")
Review §2.3, and reproduced live during the temporal-coherence slice: graphiti
invalidated a fact **at ingest time** (`invalid_at == the new fact's valid_at`,
`expired_by_sweep=false`) in a 2-chunk minimal reproduction. Previously seen only
statistically.

This corrupts the flagship temporal use case — "how did vendor X's treatment change over
time?" — by inventing change events that never happened. A live minimal repro exists,
which is the expensive half of the work.

### 7. Extraction is not deterministic at `temperature=0.0`
The dedup step's semantic search over existing facts can go either way between runs.
Matters for any live extraction test and for reproducing extraction bugs. Recorded during
the temporal-coherence live proof; no action decided.

---

## P2 — Scaling landmines (block broad ingestion, fine at pilot scale)

### 8. Vector scan — no vector index — **MEASURED 2026-09-12, returned to P2**
**My P1 promotion was wrong and the measurement refuted it** (`dedup-cost-profile-2026-09-11.md`,
second half). Three results:

1. **An index changes nothing for graphiti.** Neo4j uses a vector index only via
   `db.index.vector.query*` / `SEARCH`; graphiti emits a bare `vector.similarity.cosine`
   in a `WITH`. Measured: 254.6 ms before the index, 254.4 ms after — **0%**.
2. **Even a correct index-backed query is only 1.8x here** (255 ms → 144 ms). At 3,096
   facts the ~110-130 ms fixed overhead dominates. The crossover grows with corpus size, so
   this is "not the lever today", NOT "never the lever" — at millions of facts the
   brute-force scan projects to tens of seconds and the index becomes unavoidable.
3. **ANN is approximate and dedup needs exact candidates.** At `k=1` the index returned the
   rank-2 fact. graphiti asks for `limit=10` and feeds those to dedup, so a missed
   neighbour is a missed dedup — the very defect Slice B makes visible.

Also: `db.index.vector.queryRelationships` is **deprecated** in 2026.07.1 in favour of
`SEARCH`, so any override must target `SEARCH`. The experimental index was **dropped** after
measuring — keeping it would add write overhead to every fact insert on the path we are
trying to speed up, for a benefit no current query claims. Original entry below.

### 8-old. Vector scan — the confirmation and remediation route
**Verified on the live graph: 40 indexes, of which VECTOR = 0.** graphiti 0.30.1 scores with
`vector.similarity.cosine` per row and creates no vector index. This is no longer a
"[verify first]" item — it is measured, and item 28's profile shows it sitting on the
dominant cost: the unfiltered candidate search is 42% of the dedup cycle and 2.6x its
filtered twin.

**Cheapest next step, which costs nothing:** create a vector index on
`RELATES_TO.fact_embedding` and re-run `.superpowers/sdd/profile-dedup-search.py`. That
measures the real ceiling before any code is written. An index changes the scan, not the
two round trips, the RRF merge, or the 1.8 s LLM call — so it may not be sufficient, and
assuming it is would repeat this project's most expensive habit. Original entry below.

### 8-orig. Vector scan — the remediation route
graphiti 0.30.1 scores with `vector.similarity.cosine` (`graph_queries.py:163`) and
creates no vector index. Review §5 found a cheaper remediation than forking: the driver
exposes a pluggable `SearchInterface`
(`driver/search_interface/search_interface.py:22,59,111,232`; `driver.py:98`), so an
index-backed `edge_similarity_search`/`node_similarity_search` can be supplied without
patching library internals — same pattern as the existing wrappers in `graphiti_client.py`.

**This is the stated blocker to broad ingestion.** Review §6 notes fixing it first would
unblock the rest.

### 9. Two more Python-side scaling landmines
- `_vendor_episode_uuids` (`search.py:9-16`) collects *every* episode uuid of a vendor
  into a Python set per request.
- `shortlist_communities` (`global_search.py:62-75`) loads every community embedding at a
  level and does cosine in pure Python.

Fine at pilot scale, wrong at corpus scale.

### 10. No REQUEST-level deadline — **P3, DE-PRIORITISED by the user 2026-09-11**
User's call, recorded so it is not re-raised: *"I am not as concerned about bounding a
response to 180s. Quality of the response is ALWAYS the priority."* The per-call timeout
from 5b stays — it prevents an unbounded hang — but a request-level deadline that could
truncate or degrade an answer to meet a clock is explicitly not wanted. Do not propose
latency-for-quality trades. Original entry kept below.


The 5b hardening bounds each *call* at 180s, which stops an unbounded hang. It does **not**
bound a request: with `max_retries=3` on the client and `_complete_or_none`'s own 3x retry,
a single hung reduce can still consume roughly **24 minutes** of a user's `/answer`.
"Cannot hang indefinitely" is true; "safe for a user request" is not.

A deadline that spans the whole request — shared across shortlist, map, reduce and their
retries — is the real fix. Until it exists, the worst case is bounded but not acceptable.

### 10b. 90-second timeouts
Confirmed at `graphiti_client.py:150,154,204,256`. Add chonkie `timeout=120`
(`ingest_driver.py:80`) and docext `timeout=300` (`docext/client.py:12`) to the same fix.

---

## P3 — Operational and hygiene

### 11. The 593 pending semantic jobs, and the budget default
593 jobs sit `pending` in the `bootstrap` lane. `run_worker_once` claims bootstrap jobs
only while `today_token_total() < budget`, and `semantic_daily_token_budget` defaults to
5,000,000 (`graph_sync/config.py:19`). At ~200k tokens/article that is ~25 articles/day:
the 555 not-yet-ingested pilot-source articles take three weeks, the full corpus ~11
years. Conversely, anyone running `python -m graph_sync.cli worker` today starts a ~100M
token extraction with no further confirmation.

`--max-batches` was added (`b95d116`) which bounds a run, but **the question of what those
593 rows are for is still open** — decide, or clear them.

### 11b. Is the cheap tier good enough for map EXTRACTION? (was 3c)
`upstage/solar-pro4` no longer scores relevance — the reranker does — so the old question
("is it miscalibrated?") is moot. What remains is narrower: it still performs the map step's
**extraction** (key_points + fact_ids), and it is the same model implicated in the
out-of-range dedup indices (item 4). Worth measuring extraction quality specifically, but
only after items 0 and 0b, since the reduce step's input format is about to change.

### 12. The `judge` tier still does three jobs
The eval judge was separated this slice (`eval_judge_*`, with a guard that *refuses* to
fall back onto the synthesis tier). Still shared: synthesis (`synthesize.py:56-66`), map
(`global_search.py:97-103`) and report (`theme_builder/report.py:47-54`) all fall back to
`judge_*`. There is no `synth_llm_*` setting to separate them.

### 13. Extraction tier is not observable in the graph
`IngestArticleResult.tier` is set (`ingest_driver.py:75`) but never persisted, so the
live graph cannot say which tier extracted which episode. Blocks reproducing any
per-model quality attribution from the graph. Fix: persist `tier` on `HAS_EPISODE`.

### 14. Config no longer matches reality **[verify first]**
Review §3.5, recorded 2026-09-04 — several parts have since changed, so re-check each:
`.env` comments describe a parked cheap tier and `ling-3.0-flash` while setting
`EXTRACTION_ROUTING=true` and a different `CHEAP_LLM_MODEL`; `.env.example` (last touched
2026-07-14) lacks every `LLM_*`, `CHEAP_LLM_*`, `EMBED_*`, `CHONKIE_*`, `REPORT_LLM_*`,
`MAP_LLM_*` key and leaves `JUDGE_*` commented out — but `synthesize.py:59-63` raises at
answer-api startup without `JUDGE_BASE_URL`, **so a fresh clone following the example
cannot boot the API**. `docker-compose.yml:3` pins `neo4j:5.26` (we now run
2026.07.1 on `alpcirag01`). `CLAUDE.md:15` still says the `/answer` router is "not yet
built" and `:24` still says compose is 5.26 — both now false. `graph_sync/config.py`
re-declares `docext_*`/`neo4j_*` independently of `ExtractSettings`, so the worker loads
two settings objects.

### 15. Dead and duplicated code (all confirmed still present 2026-09-09)
`answer_api/eval_answer.py` and `answer_api/eval_golden.py` (superseded by
`eval_router.py`; no importer, no test); `graph_extract/probe.py` + the `probe` CLI
command (closed July investigation, `config.py:25-31`); `episode_builder.needs_presplit`
(tested, never called); `scripts/run_pilot.py` (builds a strong-only `IngestDriver`,
duplicates `cli ingest`); `scripts/reset_semantic_layer.py` (DETACH-DELETEs the semantic
layer); six `eval *`/`quality-*` CLI commands with no test and no runbook mention
(`graph_extract/cli.py:260-341`); `answer_api/app.py:191` re-exports the third-party
`Graphiti` class for no reason; `Neo4jRepo.delete_source_articles`
(`neo4j_repo.py:257-268`) is a `DETACH DELETE` test helper on the production repo class.

### 16. Tests that assert less than they appear to
Review §3.6. No test guards design invariant #1 or #4 explicitly (both hold by
construction only). `test_compat_checks.py:140-149` and `test_llm_penalties.py:103-114`
assert on `inspect.getsource()` substrings. `test_answer_api_app.py` monkeypatches every
retrieval function and asserts the stub's own value comes back (wiring, not retrieval).
`test_compat_harness.py:57` accepts any verdict.

**Most importantly:** `tests/integration/test_ingest_driver.py:28-31` runs `DETACH DELETE`
on episodes of the shared live graph as test setup — the one place in the repo that does
what design invariant #3 forbids.

GDS is in no automated lane (`conftest.py` uses the bare community image;
`detect_communities` is monkeypatched everywhere), so Leiden on the production GDS is
exercised only by the manual compat run.

This project has now shipped test-quality defects three times: two tests that did not
verify their own docstring claim, and five stale doubles that left a branch silently red.

### 16b. Integration tests are not hermetic against the developer's `.env`
**Corrected 2026-09-10 — my first write-up of this overstated it.** `tests/unit/conftest.py`
ALREADY has an autouse fixture stripping every `ExtractSettings` field from `os.environ`,
and its docstring documents this exact hazard plus a real past incident (changing
`CHEAP_LLM_MODEL` in `.env` broke three unrelated unit tests, order-dependently). **Unit
tests are protected.**

The real gap is `tests/integration/`, which has a `conftest.py` with no such fixture.

The hazard itself is real: `graphiti_core/helpers.py:33` calls `load_dotenv()` at **import
time**, so importing graphiti copies `.env` into `os.environ`, and
`ExtractSettings(_env_file=None, ...)` does **not** protect against it — pydantic-settings
reads `os.environ` regardless. Observed 2026-09-10: six integration tests picked up the real
`RERANK_*` credentials and began exercising the live Voyage endpoint — a **paid** API —
instead of their fakes. Patched by pinning `rerank_base_url=""` in those specific tests,
which fixes those tests and not the class.

**Fix:** extend the same autouse fixture to `tests/integration/conftest.py`. Note it must
not strip the variables the testcontainers fixtures legitimately set — check before
applying. Every future `*_BASE_URL` / `*_API_KEY` setting inherits the hazard until it is.

### 17. Label collision
The ontology's `Vendor`/`Product` entity types (`ontology.py:6,9`) collide with the
structural layer's labels, so `MATCH (v:Vendor)` returns 22 nodes mixing both layers.
Design decision #5 says link the two with `SAME_AS` rather than merging.

### 18. Type hygiene — ranked last, deliberately
61/1,300 (4.7%) of facts carry a name outside the eight declared types (23 casing
variants, 38 invented) — **not** the 12–28% originally claimed. The deeper number: 720 of
the 1,239 correctly-named facts (58%) sit on an endpoint-type pair `EDGE_TYPE_MAP` does
not sanction; 70/447 entities (16%) have no type label.

But **no query in `src/` filters by edge type**, so today's impact is nil. The casing
subset is a 20-line Cypher normaliser. Review explicitly re-ranked this from first to
last.

### 19. `write_communities` full-rebuild path
Not exercised. The incremental path is what runs today.

### 20. Report-tier config duplication
`generate_report(max_tokens=16000)` duplicates `report_max_tokens=16000` — two sources of
truth. `_prefer_fast_provider` gates on `"openrouter" in base`, so
`report_reasoning_effort` silently no-ops for any other provider.

### 21. Cosmetic marker-stripping artefacts (accepted, documented)
- `Point-in-time [9]-recovery` → `Point-in-time -recovery` leaves a stray hyphen. This is
  the conservative side of spec §3.3 (only remove a separator when a marker went from
  *each* side) and is preferred over the alternative, which deleted real sentence
  punctuation.
- `[9a]` is not treated as a marker form; the prompts never ask for it.

---

### 28. graphiti's per-fact dedup cycle is the dominant extraction cost — **P1, PROFILED 2026-09-11**
**Profiled: `dedup-cost-profile-2026-09-11.md`.** Per extracted fact graphiti runs **two**
`EDGE_HYBRID_SEARCH_RRF` searches plus one LLM call: invalidation search **1,852 ms**,
duplicate search 716 ms, LLM 1,837 ms, embed 35 ms — **~4.4 s per fact, of which SEARCH is
58%, not the LLM.** That inverts the assumption this item was written under.

**Correction to this item's original text:** the "~8 s per dedup call" was total elapsed
divided by dedup calls, which charged all extraction work to dedup. The cycle is ~4.4 s.
Dedup still dominates; the arithmetic did not.

The **unfiltered** invalidation search is **2.6x** the `edge_uuids`-filtered duplicate
search — constraining the candidate set is the whole difference, which is what an index
does structurally. Scan cost rises with fact count (253 ms clean over 3,096 facts) and the
corpus target is 2-3 orders of magnitude larger. This folds into item 8; do that first.
Do NOT quote a "dedup is X% of ingestion" figure without a concurrency-aware measurement —
`semaphore_gather` overlaps these searches, so components x facts overstates wall clock.

Original entry kept below.

### 28-orig. Per-fact dedup call volume — the raw measurement
Measured on the Slice B validation run, and not accounted for anywhere in the cost model:
graphiti issues **one dedup LLM call per extracted fact**. Seventeen articles produced
**1,145 dedup calls** — 13 to 207 per article, ~67 on average — at roughly **8 seconds
each**. Wall clock: **2h34m for 17 articles**, which is ~9 minutes per article on the
CHEAP tier.

Extrapolated naively that is years for 105k articles, so broad ingestion cannot proceed on
this shape regardless of token budget. The July cost model (item 25) counted extraction
tokens and never counted this call volume, so the Phase 1 go/no-go was taken without it.

Levers worth measuring before anything else: the candidate search is
`EDGE_HYBRID_SEARCH_RRF` with `limit=10` per fact; whether dedup can be batched, skipped
for facts with no same-endpoint candidates, or run concurrently is unknown. Do not assume
it is inherent — measure where the 8s goes first.

### 29. graphiti's dedup prompt shares one index space across two lists — **P2, upstream**
The root cause behind item 4, now confirmed at 100% of 154 observed invalid indices.
`resolve_extracted_edge` numbers `related_edges` (same-endpoint duplicate candidates)
`0..N-1` and `existing_edges` (any-endpoint invalidation candidates) `N..N+M-1` into ONE
prompt, then accepts only `0..N-1` in `duplicate_facts`. Models put the second list's
indices in that field; graphiti drops them.

**`gpt-5-mini` does it on 24% of retries**, so no tier swap fixes it. Our guard makes it
visible and counted; it deliberately does not reinterpret the indices, because an
invalidation candidate does not share endpoints and promoting one to "duplicate" would
manufacture a permanent, invisible wrong merge.

The fix is in the numbering: two independent 0-based lists, or an explicit per-list
prefix. That is a library-owned prompt (`prompts/dedupe_edges.py`), so the options are an
upstream report/PR or a local prompt override with a translation layer back to graphiti's
expected index space. The override is real surgery — graphiti validates against its own
numbering — and should not be attempted without tests that pin both directions.

## P4 — Roadmap and process

### 22. Rotate the leaked DocExtractor key and rewrite history
The read-only key at commit `a674116` is on `origin/main`. **User has deferred this three
times** — recorded here so it is not lost, not to re-litigate it.

Note the coupling: rewriting history before pushing requires the rotation. Until then,
282 local commits exist only on this host — and this host already lost a data volume
once. A `git bundle` to a second location costs nothing and is independent of the
rotation decision.

### 23. Never-pushed work
282 commits ahead of `origin/main`. Intentional (no pushes authorised), but see the
single-copy risk above.

### 24. Roadmap promises that quietly did not happen
Review §2.4: Phase 1 "structural layer complete (all vendors — it's free)"; Phase 1 exit
"incremental deltas flow within minutes of extraction" (never demonstrated); Phase 1 exit
"go/no-go on full-corpus budget" (taken on a wrong number — see item 25); Phase 3
"scheduled refresh / nightly" (built, never scheduled); §6.4 timeline "group into change
events … enrich with DocExtractor"; §10 golden set "60–100 questions curated with a domain
expert" (we have 29); §3.3 chunking "≤1,500–2,000 tokens, heading-boundary packing,
section path".

### 25. Cost model arithmetic error
The July cost report contains a 5× arithmetic error that made the plan's prior look
confirmed. `slice-2a-viability.md:29-31` reports 8.46M extraction tokens for a 39-article
pilot (~217k/article) and extrapolates to "~4B tokens" for 105k articles — but
105,000 × 217k is not 4B. See review §4 for the corrected arithmetic and the lever.
The Phase 1 budget go/no-go was taken on the wrong number.

### 26. Phase 5 — Copilot / MCP exposure
Not started. `answer-api` wrapped as an MCP server (streamable HTTP), exposed via
APIM/tunnel with OAuth, onboarded into Copilot Studio.

### 27. Incremental and update paths never exercised on this instance
Review §5 (#11): neither the incremental *nor* the update/supersede path has run on the
current Neo4j instance. The temporal-coherence slice added live proof for the update path
specifically, so **[verify first]** — but the webhook/queue/scheduler path as an
operating system remains untested in anger.

---

## Next steps, in order

**All four P0s are done** (0, 0b, 0c, 0d). Global faithfulness is 4.7 against local's 5.0;
the faithfulness gap that drove five slices is largely closed, and what remains is
synthesis over-reach rather than mis-citation.

**Also done since:** **5b** (answer-path clients bounded; eval 1h44m → 26 min), **0e** (the
bag-pasting measure, plus the eval finally persisting the answers it scores), and
**5c / 5d** — a failed new attempt no longer destroys a persisted report, on the
incremental path *and* on `--full`.

**Item 10 (request-level deadline) is de-prioritised by the user**: quality of the response
always outranks bounding its latency.

**Item 4 (Slice B) is DONE and MEASURED** — the confusion hypothesis is confirmed at
100% of 154 invalid indices, and the retry default was flipped off because `gpt-5-mini`
repeats the mistake 24% of the time.

1. **Ingestion throughput — and it needs a STRUCTURAL fix, not a storage one.** 9 min/article
   is the blocker to the project's goal. Profiling (item 28) put the dedup cycle at ~4.4 s
   per fact — two searches plus an LLM call, no single component a majority — and the
   vector-index experiment (item 8) showed indexing buys **0%** as graphiti's query is
   written and only 1.8x even rewritten. So the win has to come from doing **less work**:
   fewer dedup calls, batched candidate retrieval, or concurrency across articles. Next
   measurement: how much of the 9 minutes is already overlapped by `semaphore_gather`, since
   that decides whether concurrency is the lever or is already spent.
2. **Act on `bag_share`** — only after a run produces the number. Do not set a policy first.
3. **Item 6 — graphiti's false invalidations.** Corrupts the flagship temporal use case by
   inventing change events that never happened; a live minimal repro already exists, which
   is the expensive half of the work.
4. **Item 2 (judge split) — DE-PRIORITISED by evidence.** It existed because two
   interventions failed to move the number, implying the metric was suspect. When the real
   defect was fixed, judge score and per-claim audit moved *together* — the judge tracks
   reality. Still worth separating evidence-faithfulness from citation-precision eventually,
   but it is no longer diagnostic-critical.

Explicitly NOT next: more prompt constraints, LLM tier swaps, or rerank threshold tuning.
Four slices have shown those do not move faithfulness; the one that did was structural.

## Resolved (do not re-open)

**2026-09-10 — reranked selection + observable eval:**
- **1. Trace the map step** — done; the drift was traced through three hops and found in the
  report writer, not the map or reduce steps (`map-step-trace-2026-09-09.md`).
- **3a. Bind the map prompt to its facts** — superseded. `_MAP_PROMPT` no longer scores
  relevance at all, and the map step was shown faithful to its input. The real binding
  defect is item 0b, one hop downstream.
- **3a-bis. No relevance rubric** — resolved by removing the relevance field entirely rather
  than writing a rubric for it.
- **3b. Drop communities that contribute nothing** — resolved by `rerank_top_n` plus
  `rerank_score_floor`.
- **3d. Cross-encoder reranking** — implemented and merged. It corrected the observed
  inversion and cut map-step LLM calls ~60%, but did **not** improve faithfulness.
- **3. Structural constraint on the reduce step** — superseded by item 0b, which identifies
  the actual defect (marker binding) rather than constraining the prompt further.

**2026-09-09 — community report verification:**

- Global and DRIFT were silently dead (`Community` count 0) — `theme-build` has run,
  41 communities exist.
- The eval judge graded its own synthesis — separated via `eval_judge_*` with a guard
  that raises rather than falling back onto the synthesis tier.
- Five stale test doubles left the branch silently red — fixed (`d5a0007`).
- The faithfulness judge scored *unmeasurable* answers 0 — now returns unscored and
  reports the count (`3986ba5`).
- All three answer paths served an empty LLM reply as a blank answer with HTTP 200 —
  shared `_usable_content`/`_complete_or_none` guard (`1bf5f69`, `24f9fee`).
- Citation markers that resolve to nothing appeared in answer text — stripped, including
  comma lists and padded forms (`0452921`, `70ef007`, `24f9fee`).
- Worker had no bounded stop mode — `--max-batches` (`b95d116`).
