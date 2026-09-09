# Community Report Verification — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Stop the community-report writer inventing specifics that its own cited facts
do not support.
**Date:** 2026-09-09
**Status:** Approved design — ready for implementation planning
**Backlog:** retargets item 3a (P0)

---

## 1. The defect

`theme_builder/report.py` writes community reports as a list of
`{"finding": str, "fact_ids": [uuid, ...]}`. Its prompt (`_PROMPT`, line 21) **already**
says *"Using ONLY the numbered FACTS below"* and *"Do NOT use outside knowledge."*

The model ignores it. Traced live on 2026-09-09 (`map-step-trace-2026-09-09.md`), the
community *AWS Backup: Continuous Backups, Cross-Region/Cross-Account Copy…* produced:

| Report finding | The facts it cites actually say | |
|---|---|---|
| "…PITR **restores replay** to a specified time; however, **not supported for Amazon RDS Multi-AZ clusters**" | "AWS Backup provides continuous backups for Amazon Aurora." (+3 similar) | mechanism and Multi-AZ **invented** |
| "…retention for continuous backups **ranges from 1 to 35 days**" | "…only one recovery point is possible at a time…" (+1) | range **invented** |
| "…point-in-time restore with **1-second precision up to 35 days back**" | "AWS Backup continuously backs up transaction logs for SAP HANA databases to enable point-in-time restore." (+1) | precision and duration **invented** |
| "Cross-account backup is governed through AWS Organizations: **source and destination accounts must be in the same organization**" | "AWS Backup integrates with AWS Organizations to limit destination accounts using organizational units." (+1) | the requirement **invented** |

Note the character of the errors: 35-day retention, Multi-AZ exclusion and 1-second PITR
precision are all **genuinely true of AWS** — they are in the model's training data, just
not in our graph. This is outside-knowledge leakage, not random confabulation, which is
why it is fluent and plausible and survived this long.

## 2. Why this is the right place to fix it

Every downstream hop is faithful. Verified by tracing each one:

```
facts (accurate, generic)
  -> theme_builder/report.py   <-- INVENTS specifics
  -> map_report                    faithfully summarises the report
  -> reduce (_REDUCE_PROMPT)       faithfully copies the key points
  -> answer with real markers on invented claims
```

`map_report`'s prompt says *"Using ONLY the report"* — and it obeys. The reduce step
copies its key points verbatim. Two prior diagnoses were wrong and were corrected by
tracing one hop further each time:

- The citation-integrity slice assumed the reduce step outran its evidence. It does not.
- Backlog item 3a assumed the map step invented specifics. It does not.

Fixing the report writer means every downstream stage stays correct with no further
change. **Do not add constraints to the map or reduce prompts as part of this slice.**

## 3. Why not a deterministic gate

A token-matching gate (numbers and technical terms must appear in a cited fact) was
considered and **rejected**. Support is a semantic relation, and token matching fails on
an open-ended set of forms: abbreviations ("PITR" vs "point-in-time restore"), synonyms,
plurals, unit variants, and decomposed ranges ("1 to 35 days" vs "a 35-day maximum"). It
would be simultaneously leaky and false-positive prone, and tuning it would not converge.

Prompting alone is also rejected: the prompt already forbids exactly this behaviour.

## 4. The fix — an LLM verification pass

A second model checks each finding against the facts that finding cites, and the report
is only written once it passes.

### 4.1 Placement

In `generate_report` (`theme_builder/report.py`), after findings are parsed and their
`fact_ids` validated against the community's real fact UUIDs, and before the
`CommunityReport` is returned. `fact_ids` validation already drops hallucinated *uuids*;
this adds the missing check that the *prose* is supported by the uuids it kept.

### 4.2 Granularity and cost

**One verification call per report**, not per finding. The verifier receives every finding
beside the text of the facts that finding cites, and returns a per-finding verdict. A
report carries roughly six findings; a full rebuild is 41 communities, so this is ~41
extra calls at the measured ~$0.0006/report — cents.

### 4.3 The verifier must not be the report writer

Resolve the verifier independently and **raise** if it resolves to the report model, in
the same shape as `_eval_judge_client_and_model` (`eval_router.py`). A model grading its
own output inflates the result in a way the result cannot reveal.

Configured model: **`~deepseek/deepseek-v4-flash-latest`** (already set as
`eval_judge_model`). The report tier resolves to `z-ai/glm-5.3-flash` via `judge_model`,
so the two differ in both instance and family.

### 4.4 The summary is verified too

`summary` carries no `fact_ids`, so it is checked against the **union** of the facts cited
across the report's findings. This is not optional: `map_report` reads `hit.summary` as
well as `hit.full_report`, and the summary also feeds community shortlisting, so leaving
it ungated would relocate the leak rather than close it.

### 4.5 On violation — retry once, then drop

1. Verify. If everything is supported, return the report unchanged.
2. If any finding or the summary is unsupported, **regenerate the report once**, naming
   the offending findings and the specific unsupported content, and instructing the model
   to use only what the facts state.
3. Re-verify the regenerated report.
4. Findings still unsupported are **dropped**; the rest of the report is kept.
5. If the **summary** is still unsupported, the whole report is **skipped** — a
   one-paragraph summary cannot be partially salvaged, and an ungrounded summary poisons
   shortlisting.

### 4.6 An unusable verifier reply must never read as "supported"

If the verifier returns no usable content (empty `choices`, `None`/whitespace content, or
unparseable JSON) after its own retry, the report is **skipped**: the previous report
stays in place and the run counts it.

This is the exact defect fixed in the faithfulness judge one day earlier, where an
unmeasurable answer was recorded as a real score. Absence of a verdict is not a pass.
Reuse `synthesize._usable_content` rather than writing a fourth variant of this guard.

**Accepted trade-off:** a flaky verifier endpoint leaves communities running on their
previous reports rather than new ones. That is the safe direction — stale-but-verified
beats fresh-but-unverified — and the skip count makes it visible.

## 5. Components

| File | Change |
|---|---|
| `src/theme_builder/report.py` | `_verify_client_and_model` (raises if it resolves to the report model); a `verify_findings` call returning per-finding verdicts; the retry-then-drop flow inside `generate_report`. |
| `src/graph_extract/config.py` | `verify_llm_base_url` / `verify_llm_model` / `verify_llm_api_key`, falling back to `eval_judge_*`. |
| `src/theme_builder/cli.py` | Surface the new counters in the run summary. |
| `src/answer_api/global_search.py` | **none** — the map step is faithful; do not touch its prompt. |

## 6. Observability

The `theme-build` summary already reports `reports_written`, `reports_regenerated`,
`reports_reused`, `reports_skipped`, `communities_dissolved`. Add:

- `findings_dropped` — findings removed for being unsupported after the retry.
- `reports_reverified` — reports that needed the §4.5 regeneration pass. **Distinct from
  the existing `reports_regenerated`**, which counts reports rewritten because the
  community changed. A report can be counted in both; they answer different questions
  ("did the community change?" vs "did the model fail verification?").
- `reports_unverified` — reports skipped because verification could not be completed.

A per-item handler that hides its failure count has already cost this project weeks (see
`theme_builder/cli.py`'s per-community `except`), so these counts are part of the
deliverable, not a nicety.

## 7. Error handling and edge cases

| Case | Behaviour |
|---|---|
| All findings supported | Report returned unchanged; no retry, no extra cost beyond the one verify call. |
| Some findings unsupported | Regenerate once with the violations named; re-verify; drop those still failing. |
| Every finding dropped | Report is skipped rather than written empty; counted in `reports_skipped`. |
| Summary unsupported after retry | Whole report skipped. |
| Verifier returns nothing usable | Report skipped, counted in `reports_unverified`; previous report retained. |
| Verifier resolves to the report model | Raise at startup, do not run. |
| Report has no findings at all | Existing behaviour unchanged; nothing to verify. |
| Finding cites zero valid fact_ids | Unsupported by definition — no facts can support it. |
| **A cached report is reused** | **Not verified.** Verification runs only where a report is generated, so the 41 reports already in the graph stay contaminated until rebuilt. Closing this slice therefore *requires* a full regeneration (§8 live validation) — it is not optional cleanup. Verifying on reuse is deliberately out of scope: it would re-pay the cost on every incremental run to re-check content that has not changed. |

## 8. Testing

- **Unit (hermetic, fake clients), the core of this slice:** a supported report passes
  untouched with exactly one verify call; an unsupported finding triggers exactly one
  regeneration and is dropped if it fails again; a finding that passes on the retry is
  kept; an unsupported summary skips the whole report; an unusable verifier reply skips
  the report and never marks it supported; a verifier resolving to the report model
  raises; a finding citing zero valid fact_ids is unsupported.
- **Regression:** existing `theme_builder` tests must pass — `test_theme_cli.py`,
  `test_incremental_cli.py`. Their fakes will need the new call; update them honestly
  rather than weakening assertions.
- **Live validation:** regenerate all 41 communities, then re-trace the two questions in
  `map-step-trace-2026-09-09.md`. The four invented specifics (1–35 day range, Multi-AZ
  exclusion, 1-second precision, same-Organization requirement) **must be gone from the
  reports**. Report `findings_dropped` as a headline number.
- Full non-live suite, `uv run ruff check src tests`, `uv run mypy src` clean.

## 9. Success criteria

1. No finding survives that the verifier judged unsupported by its own cited facts.
2. The four traced inventions are absent from the regenerated reports.
3. An unusable verifier reply never results in a written report.
4. The verifier cannot be the report model.
5. `findings_dropped`, `reports_reverified` and `reports_unverified` are visible in the
   `theme-build` summary.
6. The eval is re-run and global faithfulness reported against 1.6, **whatever it shows**.

## 10. What this slice does not claim

Fixing the reports does not guarantee faithfulness improves. If the graph's facts are too
generic to answer comparison questions, honest reports will be *thinner* and the answers
may become less useful while becoming more truthful. **That is the correct outcome and
must be reported as such**, not tuned away.

`findings_dropped` is the number to watch: if reports lose most of their findings, the
community layer was largely embellishment, and the real problem is upstream in extraction
coverage — a different slice.

Backlog item 2 (the judge cannot check marker↔fact correspondence) still stands and is
**not** addressed here. Until it is, global faithfulness measures collective support, not
per-marker support.
