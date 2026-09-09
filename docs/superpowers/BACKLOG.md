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

### 1. Trace the map step: do community `key_points` drift from their facts?
**Status: DONE 2026-09-09. Answer: YES — the drift starts in the map step.**
Full evidence in `map-step-trace-2026-09-09.md`. Every hallucinated specific in the
traced reduce output was already present verbatim in the `key_points` the reduce step was
given. The map step fabricates *when the community's facts do not answer the question* —
rich facts produce faithful key points (that question scored 4), generic facts produce
invented specifics (scored 0-1). Meta-commentary is also injected here, so the reduce
prompt is fighting its own input. **This retargets item 3 — see items 3a/3b below.**

<details><summary>Original framing (kept for context)</summary>

The global reduce step never sees fact text — it sees `key_points`, LLM-written prose
produced by `map_report` (`global_search.py`). Faithfulness is scored against the
*facts*. So if the map step already drifts, the reduce step is faithfully summarising an
unfaithful intermediate and **no reduce-side fix can work**.

Decisive experiment: for one global question, dump each community's `key_points` beside
the fact texts they claim to rest on. Cheap, read-only.

**This gates item 3.** Do not design a structural reduce fix until this answers.

Evidence: global grounding is 1.00 (citations point at the right articles) while global
faithfulness is ~1.1–1.6 (prose not supported by them). The traced worst case asserts
"1-second PITR precision", "1–35 day retention", "RDS Multi-AZ", "Azure PostgreSQL" —
none present in any of its 21 cited facts. See `router-eval-report.md`.
</details>

### 2. The faithfulness judge cannot check marker→fact correspondence
`_cited_fact_texts` (`eval_router.py`) returns facts in arbitrary Neo4j order and the
judge prompt lists them unnumbered. The judge can therefore only ask "is this claim
supported by the fact set *collectively*?" — a misattributed-but-plausible marker passes.

We are optimising against a looser metric than the one we care about. Fix: number the
facts by marker and have the judge score marker-to-fact support.

**Do this before faithfulness gates any decision.** This slice is a case study in what
happens when a metric is trusted further than it deserves.

### 3a. Bind the map prompt to its facts — **the primary fix, do this first**
Give `map_report` (`global_search.py`) the same evidence-binding the reduce prompt
received in the citation-integrity slice: every key point must be supported by the facts
it cites, no invented specifics, and no meta-commentary about what the report lacks.

This is where the defect actually lives. Evidence in `map-step-trace-2026-09-09.md`:
five of five key points from one community contained invented specifics (transaction-log
replay mechanism, RDS Multi-AZ exclusion, 1–35 day range, 1-second precision, full-vs-
incremental copy, same-AWS-Organization requirement) — none in any cited fact, all of
them reproduced downstream with real markers attached.

### 3b. Drop communities that contribute nothing
`relevance_min = 2` admits communities whose own key points say "No information provided
on AWS Backup restore workflows" or "provides no details". They inject meta-commentary
and consume marker numbers. Either raise the threshold or detect a map result carrying no
supported key point before it reaches reduce.

### 3c. Re-test the map tier after 3a
The map model is `upstage/solar-pro4` — the same cheap-tier model implicated in item 4.
Fix our prompt first (on this project the fault has been in our own code or config every
time), then compare cheap vs strong tier on the map step with the corrected prompt.

### 3. ~~Structural constraint on the global reduce step~~ — **DEPRIORITISED 2026-09-09**
Superseded by 3a/3b/3c. The trace showed the reduce step is **faithful to its input** —
it copied the invented specifics it was handed. Constraining it further would constrain a
step that is already doing its job.

Revisit only if fixing the map step (3a) does not move faithfulness. If revisited, the
candidate approaches remain: feed verified fact text to the reduce step rather than
`key_points`, or post-verify each claim against its cited fact.

---

## P1 — Known-wrong behaviour in shipped code

### 4. Slice B: out-of-range dedup indices during extraction
Deferred deliberately from the citation-integrity slice — see
`specs/2026-09-08-answer-citation-integrity-design.md` §8.

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

### 5. `router.py:72` — the last `content or ""` site
`max_tokens=8` against a possibly-reasoning classifier, then
`(resp.choices[0].message.content or "").strip()`. An empty reply falls through silently
to `default_mode`. Two problems: the unguarded `resp.choices[0]` (empty `choices` list
raises IndexError) and the silent fallthrough with no log.

Same bug class as the four already fixed — see the `llm-empty-reply-coerced-to-value`
note. Use `synthesize._usable_content`. Observable today as `routing.via = "default"`,
but nothing aggregates it, so a regression would be invisible. Routing accuracy 0.97 was
measured with the current classifier, so it is not biting *now*.

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

### 8. Vector scan — no vector index **[verify first]**
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

### 10. 90-second timeouts
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

## Resolved this session (do not re-open)

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
