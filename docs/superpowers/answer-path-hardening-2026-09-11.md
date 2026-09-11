# Answer-path client hardening — 2026-09-11

**BACKLOG 5b (and 5), branch `answer-path-client-hardening`.** Every answer-path LLM
client now has a bounded timeout and a per-tier reasoning bound, through one shared
helper. Baseline for comparison: the 0b run (`router-eval-report.md`, 2026-09-11
11:28 → 13:12), 8 empty-content retries in ~40 reduce calls, per-question outliers of
478s and 528s, global faithfulness 4.7 (10/10), unscored 0/29.

## The defect

Four clients were constructed bare — `AsyncOpenAI(api_key, base_url)` — with the
library's default (no effective) timeout and no reasoning parameter:

| tier | site | model (live `.env`) | reasons by default? |
|---|---|---|---|
| synthesis / reduce | `synthesize._synthesis_client_and_model` | `z-ai/glm-5.3-flash` | yes |
| map | `global_search._map_client_and_model` | `upstage/solar-pro4` | **no** |
| eval judge | `eval_router._eval_judge_client_and_model` | `~deepseek/deepseek-v4-flash-latest` | yes |
| router classifier | `router._cheap_classify_client` | `upstage/solar-pro4` | **no** |

**No timeout:** a single slow OpenRouter route (the report tier measured 29 tok/s vs
9.4 tok/s for the same model on the same prompt, one call at 380s) hangs the caller for
as long as the route takes. In production that is a user's `/answer` request.

**No reasoning bound:** reasoning tokens are billed against `max_tokens`. On a model that
reasons by default, an unbounded effort can spend the whole budget before the answer
starts, and the reply comes back HTTP 200 with `content=None, finish_reason='length'`.
`_complete_or_none` recovers with one retry at 3x the budget — every one of the 8
baseline cases recovered — but a fifth of reduce calls paid ~4x, and the largest prompts
were exactly the ones that tripped it.

The remedy already existed as `theme_builder.report._prefer_fast_provider` (throughput
routing + `reasoning: {effort}`), applied to the report and verifier tiers on 2026-09-08
and never generalised. Same root cause, fourth defect.

## What changed

**`graph_extract/usage.py`** (the layer both services already import for `instrument`
and `usable_content`):

- `prefer_fast_provider(client, reasoning_effort)` — moved here verbatim from
  `theme_builder/report.py`, with the measurements that justify it in the docstring.
- `bounded_llm_client(base_url, api_key, *, reasoning_effort, timeout=180.0,
  max_retries=3)` — the one way to build a synthesis-side client: `instrument` +
  finite timeout + bounded retries, and on OpenRouter the throughput sort plus the
  tier's reasoning bound. Non-OpenRouter bases get a plain request (the report tier's
  existing guard).

**`theme_builder/report.py`** re-exports `prefer_fast_provider` as `_prefer_fast_provider`,
so the report and verifier tiers and their tests are untouched (`test_verify_client`
still monkeypatches `report_mod.AsyncOpenAI` and passes). `answer_api` does not import
`theme_builder`; the layering is preserved.

**Four call sites** now go through `bounded_llm_client`:

| tier | timeout | retries | reasoning |
|---|---|---|---|
| synthesis / reduce | 180s | 3 | `synthesis_reasoning_effort` = `"low"` |
| map | 180s | 3 | `map_reasoning_effort` = `""` (none) |
| eval judge | 180s | 3 | `eval_judge_reasoning_effort` = `"low"` |
| classifier | **20s** | 2 | none, hard-wired |

A timed-out request is retried by the openai client on a fresh route (`allow_fallbacks`),
so the bound is per attempt: the worst case for a synthesis call is 4 × 180s, and the
classifier's is 3 × 20s before `classify()` falls back to the default mode.

**`router.classify`** (BACKLOG 5, a natural consequence): the classifier's reply now goes
through `usable_content`. An empty reply — `content=None` or an empty `choices` list — is
logged as `cheap classifier ... returned no usable content (finish_reason=...)` and
defaults, instead of being coerced to `""` and silently defaulting (content case) or
raising `IndexError` into the generic `except` with a traceback (choices case). The
outcome (`routing.via = "default"`) is unchanged; it is now observable.

## Why per-tier settings, not one shared knob

The task left this to judgement, warning that four near-identical settings may be worse
than one. Before deciding I measured the two tiers on `upstage/solar-pro4` (12 calls,
scratchpad probes; usage fields only):

**Classifier, max_tokens=8** (3 questions each):

| config | content | finish | completion / reasoning tokens |
|---|---|---|---|
| no reasoning parameter | `'local'`, `'global'`, `'global'` | stop | 2 / 0 |
| `effort: low` | `None` ×3 | length | 8 / 8 |
| `effort: low`, max_tokens=64 | `None` ×3 | length | 64 / 64 |

**Map, max_tokens=2000** (the two longest level-1 reports):

| config | result |
|---|---|
| no reasoning parameter | both `finish=stop`, JSON parsed, 13 and 495 completion tokens, 0 reasoning |
| `effort: low` | report 1: `finish=length`, **2000/2000 tokens on reasoning, no JSON** (57.7s); report 2: stop, 1088 tokens (694 reasoning), JSON parsed |

So `solar-pro4` does not reason unless asked, and `reasoning: {effort: low}` is the ask.
The premise "bounding reasoning on the classifier is not optional" is inverted for the
deployed model: sending the bound is what empties every reply. On the map tier it would
have sent one of two reports to `map_report`'s identical retry and then dropped the
community from the answer.

The reasoning effort is therefore a property of the **model** a tier points at, and
models are configured per tier (`map_llm_*`, `eval_judge_*`, `judge_*`, `cheap_llm_*`),
so the effort has to be too. One shared `"low"` would have broken two of the four tiers
on the current `.env`. The three settings sit under one comment block in `config.py`
recording both measurements; the classifier has no setting because a reasoning cheap
model would need a far larger `max_tokens` as well, which is a code change either way.

`"low"` is the floor for the tiers that reason, never `enabled: false`: the verifier
measured `low` as a real verdict in 3028 tokens and `enabled: false` as a 13-token
rubber stamp that approved everything. That is recorded in `prefer_fast_provider`'s
docstring and in `config.py` so nobody "optimises" it later.

## Tests

`tests/unit/test_answer_path_clients.py` (12) and two additions to
`tests/unit/test_router_classify.py`. Written first; all failed for the right reasons
(no `AsyncOpenAI` in `graph_extract.usage` to patch, missing settings, missing re-export,
classifier not logging). Then eight mutations, each killed by exactly its tests, with
`git diff --stat src/` empty and 23 passed after restore:

| mutation | killed by |
|---|---|
| drop `timeout`/`max_retries` from the builder | `test_every_answer_path_client_has_a_bounded_timeout` ×4, `..._tighter_than_synthesis` |
| never send the reasoning bound | `test_reasoning_bound_and_throughput_routing_on_openrouter` ×3, `test_configured_effort_value_is_used_not_hardcoded` |
| classifier sends the synthesis effort | `test_classifier_sends_no_reasoning_parameter_but_still_routes` |
| classifier timeout 180s | `test_classifier_timeout_is_tighter_than_synthesis` |
| router back to the old coercion | `test_empty_reply_defaults_and_is_logged`, `test_no_choices_defaults_without_a_traceback` |
| synthesis default `""` | `test_synthesis_and_eval_judge_default_to_low_never_off` |
| map default `"low"` | `test_map_default_sends_no_reasoning_parameter_but_still_routes` |
| OpenRouter guard removed | `test_no_routing_added_for_a_non_openrouter_base` |

No existing assertion was weakened; `test_verify_client`'s three reasoning-bound tests
and `test_usable_content_shared`'s re-export test run unchanged against the moved code.

## Live validation

Full eval, `uv run --extra dev python -m answer_api.eval_router`, 2026-09-11 14:04:29 →
14:30:35, against the same corpus as the 0b baseline run (no `theme-build`). The full
test suite ran concurrently for the first 11 minutes; the eval is remote-latency-bound
and the local questions it overlapped took 9–15s each, so it did not distort the
timings.

### The defect's signature: empty-content retries

| | 0b baseline | this run |
|---|---|---|
| synthesis "returned no usable content; retrying" | **8** in ~40 reduce calls | **0** |
| synthesis "still returned no usable content; giving up" | 0 | 0 |
| eval judge "unmeasurable after retry" | 0 | 0 |
| any WARNING at all in the run log | — | **0** |

The reasoning bound removed the retry entirely on a prompt set that is unchanged from
the run that tripped it 8 times. Ten global-mode answers plus 20 comparative reduce
calls and 10 drift syntheses ran at their first `max_tokens=3000` attempt.

### Latency

| | 0b baseline | this run |
|---|---|---|
| wall-clock | 1h44m | **26 min** |
| slowest question | 528s (and a 478s) | **184s** |
| global-intent questions (4 modes + 4 judge calls each) | — | 100–143s |
| drift-intent questions | — | 121–184s |
| local questions | — | 9–15s |
| timeline questions | — | 12–16s |
| sum of per-question time | — | 1560s |

Two mechanisms, and the log cannot fully separate them: the throughput sort steers
each call to the fastest provider, and `effort: low` shortens every reasoning-model
call. No request reached the 180s timeout (no retry warnings), so the timeout did not
contribute to this run's speed; it is the ceiling for the next bad route.

### Quality — held, with one thing to read carefully

| | 0b baseline | this run |
|---|---|---|
| routing accuracy | 0.97 | 0.97 |
| grounding precision | 0.92 | 0.92 |
| faithfulness mean | 4.90 | **5.00** |
| — global (10/10) | 4.7 | 5.0 |
| unscored | 0/29 | 0/29 |
| ranges in any answer | 0/29 | 0/29 |
| comparative global / drift | 4.9 / 4.8 | 5.0 / 4.8 |
| global `mps` max | 19 | 18 |

Faithfulness moved up by 0.1: the three global-mode answers that scored 4 last run
scored 5. Since the eval judge's own client changed in this slice, that number was
checked rather than accepted.

**Judge-sensitivity probe** (16 calls, `deepseek-v4-flash`): the same four hand-written
answers to one question, judged twice each by a client built with `effort: low` (this
change) and by one sending no reasoning parameter (the old behaviour):

| answer | `effort: low` | unbounded |
|---|---|---|
| faithful restatement of the facts | 5, 5 | 5, 5 |
| faithful + one invented number ("maximum 90 days") | 4, 4 | 3, 4 |
| mostly invented (mode, minimum, auto-extension) | 0, 0 | 0, 0 |
| wholly unsupported (contradicts the facts) | 0, 0 | 0, 0 |

The judge still discriminates. It reasoned 70–290 tokens in both configurations on this
prompt, and the one-point difference on the single-invented-claim answer is inside the
unbounded judge's own rep-to-rep spread. This is not the verifier's `enabled: false`
rubber stamp, which would have scored everything 5.

What did change is the **synthesis output**. `cited` moved in 8 of 14 local answers
(8→1, 9→3, 5→3, 1→2, 10→8, 11→13, 14→13, 11→13) and in 8 of 10 global-mode answers
(89→64, 8→65, 22→37, 28→21, 23→18, 18→37, 62→64, 48→28). Under `temperature=0` the text still depends on
which provider served the call and on how much reasoning preceded the answer, and both
changed here. Three 4→5 moves on n=10 are within that variance; they are recorded, not
claimed. The residual problem the 0b review named — bag sentences with ≥8 markers — is
untouched (global `mps` max 18, two answers ≥11).

## Things we had not considered

1. **The reasoning bound is a switch as well as a bound.** On a model that does not
   reason by default, `reasoning: {effort: low}` turns thinking on. The BACKLOG entry and
   the task framing assumed a shared "apply `_prefer_fast_provider` everywhere" fix; on
   the live `.env` that would have blanked every classifier reply (routing accuracy would
   have fallen from 0.97 to the heuristic-only rate) and dropped map communities. Any
   future tier that changes model needs its effort setting revisited, and the
   `config.py` comment says so. A cheap guard worth considering: log
   `completion_tokens_details.reasoning_tokens` on empty replies, which is the field
   that made this diagnosable in one probe.
2. **The classifier's `max_tokens=8` is now a documented constraint** on which cheap
   models can hold that role (non-reasoning only). Moving the cheap tier to a reasoning
   model is a code change (bound + budget), not a `.env` change.
3. **`theme_builder/report.py` still builds its two clients by hand.** They already
   had the timeout and the wrapper; they were left untouched on purpose so the report
   and verifier tests keep monkeypatching `report_mod.AsyncOpenAI`. Migrating them to
   `bounded_llm_client` is a small follow-up (BACKLOG 20, report-tier config duplication).
4. **The eval judge is now `instrument`ed** (it was the one client that was not), so
   its tokens count in the usage tally. Harmless, but a cost report over an eval run
   will read higher than before for that reason alone.
5. **Timeouts stack with `_complete_or_none` and `map_report` retries.** A synthesis
   call's worst case is 4 attempts × 180s at the client, then one more `_complete_or_none`
   retry at 3× tokens — up to ~24 minutes for a single reduce if every route hangs. That
   is a bound where there was none, but it is not a user-facing SLA; if `/answer` needs
   one, it belongs at the request level (BACKLOG 10).

## Verification

- `uv run ruff check src tests` — clean.
- `uv run mypy src` — clean (69 files).
- `uv run --extra dev pytest -m "not live" -q` — **727 passed**, 15 deselected, 10:52.
- Eight mutation checks (table above), `git diff --stat src/` empty after restore.
- Live eval as above; 12 probe calls on the map/classifier model, 16 on the judge.


## Review correction (2026-09-11)

**The judge probe is weaker than this report first read it.** Three of its four fixtures sit
at the scale floor or ceiling (5/5, 0/0, 0/0), where leniency cannot register. The only
discriminating cell moved `3,4` → `4,4` — so the one signal the probe produced points
*toward* leniency, and n=2 cannot call that rep-to-rep spread. The decisive test was
available and skipped: re-judge, under both judge configurations, the three answers that
actually moved 4→5. The conclusion stands (4.90 → 5.00 is recorded as variance and not
claimed as improvement), but the probe does not establish it.

**180s bounds a CALL, not a REQUEST.** With `max_retries=3` on the client and
`_complete_or_none`'s own 3x retry, a single hung reduce can still consume roughly 24
minutes of a user's `/answer`. "Cannot hang indefinitely" is true; "safe for a user request"
is not. A request-level deadline (BACKLOG 10) is the real fix, not a nice-to-have.

**Classifier retries reduced 2 → 1** after review: on timeout the classifier falls back to
`via="default"`, which routes to DRIFT, the most expensive mode. 20s x 3 was a 60s worst
case against a measured 36s route on this model.
