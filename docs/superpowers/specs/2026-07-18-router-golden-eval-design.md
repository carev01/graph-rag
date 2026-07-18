# Router Golden-Set Evaluation — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Phase 4 exit criterion — a labeled golden question set across all four
retrieval modes + a harness that scores the `/answer` router's **routing accuracy**,
per-mode **citation grounding**, and LLM-judge **faithfulness**, plus a comparative
check that **DRIFT beats pure-local/global on broad questions**.
**Date:** 2026-07-18
**Status:** Approved design — ready for implementation planning

The existing harness (`answer_api/golden.py` + `golden_questions.json` +
`eval_golden.py`) scores only *local* retrieval citation-precision on 15 factual
questions. This slice adds the multi-mode exit-criterion evaluation without
disturbing that narrower eval.

---

## 1. Scope

**In:** `answer_api/router_golden.json` (labeled 4-intent set); `answer_api/router_eval.py`
(pure scoring: routing hit, judge-score parse, aggregation — reuses
`golden.precision_at_k`); `answer_api/eval_router.py` (the `@live` harness CLI:
run the router + faithfulness judge + the comparative pass, aggregate, write the
report). A demonstration report.

**Out (deferred):** hard CI thresholds gating on scores (informational for now);
multi-sample / multi-judge averaging; expanding the set as more vendors ingest;
changes to `golden_questions.json`/`eval_golden.py` (the narrow local eval, left
untouched).

## 2. Grounding (verified)

- Existing reusable helper: `answer_api.golden.precision_at_k(results, expected_article_ids)
  -> bool` — true if any expected `article_id` appears among the `sources` of the
  input dicts. The `/answer` envelope's `citations` are `[{marker, fact_uuid,
  sources:[{url,title,article_id,…}]}]`, so `precision_at_k(env["citations"], …)`
  works directly across all modes.
- The `/answer` router returns `{mode, query, answer, citations, routing:{chosen,
  via,fallback_from}, freshness, …}`. `answer_router(...)` is the entry point;
  `?mode=` overrides classification (used by the comparative pass).
- Corpus: `backup-docs`, AWS + Azure(Microsoft) only, ~66 communities. All four
  intents are answerable (cross-vendor comparisons; broad "what should I consider";
  timeline over the bi-temporal facts, e.g. Azure soft-delete change events).
- Judge tier: `synthesize._synthesis_client_and_model(settings)` → GLM via `judge_*`.
- Cited fact TEXTS are not in the envelope (citations carry `fact_uuid` + sources,
  not the fact string); the harness resolves them by uuid for the judge.

## 3. The labeled set (`router_golden.json`)

A JSON list, ~25–30 entries across the four intents. Each entry:
```json
{
  "question": "Compare how AWS Backup and Azure Backup handle immutability",
  "intent": "local | global | drift | timeline",
  "expected_modes": ["global", "drift"],
  "expected_article_ids": ["<uuid>", "..."]
}
```
- `intent` — the question's category (also selects the comparative subset:
  `intent ∈ {global, drift}` = broad).
- `expected_modes` — the acceptable routing targets (a SET; ambiguous broad
  questions allow both `global` and `drift`). Routing is a hit iff
  `routing.chosen ∈ expected_modes`.
- `expected_article_ids` — grounding truth (≥1 must appear in the answer's
  citations). MAY be `[]` for broad questions hard to pin to specific articles →
  grounding is not scored for that question.
- Composition: the 15 existing local questions (labeled `intent:"local"`,
  `expected_modes:["local"]`, their existing `expected_article_ids`) + ~5 `global`
  (cross-vendor AWS-vs-Azure) + ~5 `drift` (broad, sourced) + ~4 `timeline`
  (change-over-time), each answerable from the AWS+Azure corpus.

## 4. Metrics

Per question (main pass, on the router's chosen mode):
1. **Routing hit** — `routing.chosen ∈ expected_modes` (bool).
2. **Grounding hit** — `precision_at_k(env["citations"], expected_article_ids)`
   when `expected_article_ids` non-empty; else `None` (not scored).
3. **Faithfulness** — GLM judge 0–5: how fully the ANSWER's claims are supported by
   the CITED FACT TEXTS (5 = every claim grounded, 0 = unsupported/hallucinated).

Aggregates: routing accuracy (overall + per `intent`); grounding precision (over
scored questions, overall + per chosen `mode`); mean faithfulness (overall + per
chosen `mode`).

**Comparative** (broad subset, `intent ∈ {global, drift}`): run each broad question
through `mode=local`, `mode=global`, `mode=drift` (via the `?mode` override);
compute grounding + faithfulness per forced mode; report the per-mode means and a
`drift_wins` verdict (drift's mean faithfulness ≥ both others AND drift's grounding
≥ both others on the broad subset).

## 5. Components

### 5.1 `router_eval.py` (pure, unit-tested)
- `routing_hit(chosen: str, expected_modes: list[str]) -> bool`.
- `_parse_judge_score(raw: str) -> int` — extract the first integer, clamp to
  `[0,5]`; unparseable → `0`.
- `aggregate(per_question: list[dict]) -> dict` — from per-question records
  `{intent, chosen, routing_hit, grounding_hit: bool|None, faithfulness: int,
  comparative?: {mode: {grounding_hit, faithfulness}}}` produce the summary
  (routing accuracy overall/per-intent; grounding & faithfulness overall/per-mode;
  the comparative block + `drift_wins`).

### 5.2 `_faithfulness_judge` (in `eval_router.py`)
`async _faithfulness_judge(client, model, q, answer, cited_facts: list[str]) -> int`:
one GLM call, prompt asks for ONLY an integer 0–5 scoring how fully `answer`'s
claims are supported by `cited_facts`; `_parse_judge_score` the reply.

### 5.3 `_cited_fact_texts` (in `eval_router.py`)
`async _cited_fact_texts(driver, group_id, fact_uuids) -> list[str]`:
`MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $uuids RETURN f.fact`.

### 5.4 `eval_router.py` harness (`@live`, CLI)
`python -m answer_api.eval_router`: build clients (graphiti, driver, embedder,
synth/map/cheap) exactly as the `answer_api` lifespan does; for each question run
`router_mod.answer_router(...)`, resolve cited fact texts, judge; run the
comparative pass on the broad subset; `aggregate`; echo the summary and write
`docs/superpowers/router-eval-report.md` + a JSON summary. `answer_router` and the
judge are referenced via module attrs so tests can substitute them.

## 6. Output

`docs/superpowers/router-eval-report.md`: a routing-accuracy table (overall + per
intent), grounding + faithfulness per mode, the comparative broad-question table
(local vs global vs drift with the `drift_wins` verdict), and per-question detail
(question, intent, chosen mode, routing/grounding hits, faithfulness). A companion
machine-readable summary dict is echoed by the CLI.

## 7. Error handling & edge cases

| Case | Behavior |
|---|---|
| `expected_article_ids == []` | Grounding not scored (`None`); excluded from the grounding-precision denominator. |
| Judge returns junk / non-integer | `_parse_judge_score` → `0` (never raises). |
| A mode returns a refusal (no citations) | Grounding miss + low faithfulness (correct — the harness records it, doesn't crash). |
| A question errors mid-run (a mode raises) | Logged; recorded as routing miss + grounding miss + faithfulness 0; the run continues (one bad question never aborts the eval). |
| Comparative only for `intent ∈ {global, drift}` | Local/timeline questions skip the comparative pass. |

## 8. Testing

- **Unit (`router_eval.py`, pure):** `routing_hit` (∈/∉); `_parse_judge_score`
  (plain int, fenced/chatty text, junk→0, clamp >5→5 and <0→0); `aggregate` over a
  canned `per_question` list → correct routing accuracy (overall + per intent),
  grounding precision (skipping `None`), mean faithfulness per mode, and the
  comparative `drift_wins` verdict (a case where drift wins and one where it
  doesn't).
- **Integration (fixture, no live infra):** the `eval_router` run loop with a FAKE
  `answer_router` (canned envelopes incl. one broad + one local + one `[]`-grounding
  question), a fake judge (canned scores), and a fake `_cited_fact_texts` → assert
  the produced summary matches expected aggregates and grounding is skipped for the
  empty-expected question.
- **`@live`:** run the real harness on `backup-docs` → writes the report; a loose
  sanity smoke (harness completes over the full set; routing accuracy > 0.5; no
  answer contains an LLM-authored URL).
- Full non-live suite + `ruff check src tests` + mypy clean.

## 9. Acceptance criteria

1. `router_golden.json` holds ~25–30 questions across all four intents, each with
   `intent`, `expected_modes`, and `expected_article_ids` (possibly `[]`).
2. `python -m answer_api.eval_router` runs the set through the real `/answer`
   router, scores routing accuracy + grounding + judge faithfulness per question,
   runs the comparative broad-subset pass, and writes
   `docs/superpowers/router-eval-report.md` with the aggregates + a `drift_wins`
   verdict.
3. Pure scoring functions are unit-tested; the harness loop is fixture-tested
   without live infra; the `@live` run completes and produces the report.
4. No LLM authors a URL in any evaluated answer (design #2 holds through the modes).
5. Full non-live suite + ruff/mypy clean.

## 10. Deferred

- Hard CI thresholds / regression gates on the scores (informational this slice).
- Multi-sample judging to reduce judge variance.
- Expanding the labeled set as more vendors are ingested (the "beats" signal
  strengthens with corpus breadth).
- Unifying `golden_questions.json` into `router_golden.json` (kept separate here).
