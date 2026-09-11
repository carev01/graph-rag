# Slice B — out-of-range dedup indices during extraction

**Date:** 2026-09-11
**Backlog item:** 4 (P1)
**Branch:** `slice-b-dedup-indices` (from `main` at `5638ac5`)
**Graph:** untouched. **No live run of any kind** — no `ingest`, no `theme-build`, no
`eval_router`. Everything below is hermetic (graphiti 0.30.1 source + unit tests).
**Tiers (unchanged):** cheap extraction `upstage/solar-pro4` (OpenRouter), strong
`gpt-5-mini` (Azure). Tier selection `IngestDriver._tier_for`.

## 1. The defect, traced

`graphiti_core/utils/maintenance/edge_operations.py::resolve_extracted_edge` (0.30.1,
lines 623–846 in the installed copy). For each newly extracted fact graphiti builds ONE
prompt (`prompts/dedupe_edges.py`) containing two candidate lists with continuous
numbering:

| list | source | idx | validated field |
|---|---|---|---|
| `related_edges` — duplicate candidates, **same endpoints** as the new fact | `search(..., EDGE_HYBRID_SEARCH_RRF, edge_uuids=<edges between the two nodes>)` | `0 .. N-1` | `duplicate_facts` must be in `0..N-1` |
| `existing_edges` — invalidation candidates, any endpoints, minus those already in `related_edges` | `search(..., EDGE_HYBRID_SEARCH_RRF)` unfiltered | `N .. N+M-1` | `contradicted_facts` must be in `0..N+M-1` |

Both searches use `EDGE_HYBRID_SEARCH_RRF`, which does not override `limit`, so both are
capped at `DEFAULT_SEARCH_LIMIT = 10` (`search/search_config.py:29`). **A valid reply can
never contain an index above 19.**

Out-of-range entries are logged (`logger.warning`, lines 738 and 763) and dropped
(lines 744, 770–776). What is lost — read from the code and pinned by tests
(`tests/unit/test_dedup_guard.py`, section 1):

1. **`duplicate_facts`** — only the FIRST valid index is used (`for ...: resolved_edge =
   related_edges[id]; break`, 747–749). A dropped index therefore costs nothing while at
   least one valid index survives. When **every** index is invalid, the extracted edge is
   kept as new: a duplicate fact enters the graph with its own episode, permanently. The
   brief's claim 1 is **confirmed**.
2. **`contradicted_facts`** — every valid index is used (no `break`), so each dropped index
   is a missed contradiction. **Precondition the brief did not state:** graphiti only
   invalidates a candidate whose `valid_at` is strictly earlier than the new fact's
   `valid_at` (`resolve_edge_contradictions`, 563–571); a candidate with no `valid_at` is
   never invalidated at ingest, dropped index or not. So the temporal damage is bounded to
   candidates that carry an earlier `valid_at`. Claim 2 is **confirmed with that bound**.
3. **"Confusion in the opposite direction" does not exist** at the index level:
   `contradicted_facts` legally spans BOTH lists (prompt: "idx values from EITHER list"),
   and line 771 accepts duplicate-range indices there. The only invalid contradicted
   indices are `> N+M-1` or negative — hallucinated, not confused.

One more thing the warning tells us: "valid range: 0-9" means `related_edges` was at the
search cap — **at least 10 existing edges between one node pair**. That is either a hub
pair or accumulated dedup misses; the cap also means an 11th same-pair fact is invisible
to dedup. Not addressed here; noted.

## 2. The hypothesis — tested as far as hermetic evidence allows

**Hypothesis (brief):** the invalid indices are invalidation-candidate indices written into
`duplicate_facts` (index-space confusion), not hallucinated garbage. Decisive check: are
the invalid indices in `[N, N+M-1]`?

**What exists to test it against:** nothing recorded. A grep over the repository (excluding
`.venv`) for the warning text finds only `BACKLOG.md` and the citation-integrity spec —
no run logs survive. The single data point is the quoted line:
`[10,11,14,15] (valid range: 0-9)`.

**What that one line supports:**

- `N = 10`. The invalidation range is `10 .. 9+M` with `M ≤ 10`, so at most `10..19`.
- All four indices are in `10..15` — **4 of 4 inside the maximum possible invalidation
  range; 0 negative; 0 above 19.**
- For them to be in the *actual* range, that call needed `M ≥ 6`. `M` is not in the
  warning (it is in a DEBUG line, line 716, which was not captured), so unknown.

**Verdict: consistent with the hypothesis, not confirmed by it.** A model that hallucinates
would plausibly also land near the numbers it has just read, so one in-range event does
not separate the two explanations. The brief said to say so plainly if the hypothesis
fails; it has not failed, it is simply *untested*, and no hermetic work can test it. What
this slice does instead is make the check **automatic and per-call in the next live
ingest**: the guard classifies every out-of-range index as
`dup_in_invalidation_range` or `dup_beyond_range` and the counters print at the end of
`ingest`. The ratio between those two IS the decisive check.

## 3. What was built

### 3.1 `src/graph_extract/dedup_guard.py` (new)

Wraps `graphiti.llm_client.generate_response` (instance attribute, same pattern as the
`chat.completions.create` wrappers). Only `prompt_name == "dedupe_edges.resolve_edge"`
is touched. Per call:

1. **Recover N and M from the prompt** — `parse_candidate_counts`. graphiti renders each
   list with Python `repr`: `[{'idx': 0, 'fact': '…'}, …]` inside `<EXISTING FACTS>` /
   `<FACT INVALIDATION CANDIDATES>` tags. The parser counts `{'idx': k, 'fact': ` matches
   per block and **requires the runs to be exactly `0..N-1` and `N..N+M-1`**; anything
   else returns `None` (counted as `parse_failures`, call passed through unchecked). A
   fact that happens to contain an idx-shaped string therefore produces a *detectable*
   parse failure, never a wrong N.
2. **Classify the reply** against `(N, M)`: duplicate indices inside the invalidation
   range, duplicate indices beyond both ranges or negative, contradicted indices outside
   `0..N+M-1`.
3. **Record** into the `DedupIndexStats` published through the `CURRENT_DEDUP_STATS`
   `ContextVar` (or the guard's own `unscoped` sink when no article scope is open).
4. **Retry, optionally**: if any index is out of range and a `fallback` client is
   configured, re-issue the **pristine** prompt (graphiti's clients mutate `messages` in
   place — schema and language suffixes — so a copy is taken before the first call) on the
   fallback and hand graphiti *that* reply. Counted as `retried` and `retry_clean` /
   `retry_dirty`. The fallback's exceptions propagate — a dead strong tier fails the
   episode loudly, like any other LLM error.
5. **Log** one WARNING per event with structured args: invalid indices per field, N, M,
   the invalidation range, the in-range/beyond split, whether it retried and whether the
   retry was clean.

The retry happens **before graphiti sees the reply**, i.e. before anything is written to
Neo4j — that is why it is side-effect free and why the call level, not the article
level, is where a retry belongs.

`install_dedup_guard(...)` returns the guard, whose `.raw` is the wrapped client's
*unguarded* method: it is what the cheap tier uses as its fallback, so a retry is not
counted a second time by the strong tier's own guard.

### 3.2 `IngestDriver` (`ingest_driver.py`)

`ingest_article` opens a `CURRENT_DEDUP_STATS` scope per article (reset in `finally`),
attaches the counters as `IngestArticleResult.dedup`, logs a WARNING with the article id,
tier and full breakdown when `invalid_calls > 0`, and `ingest_source` merges into
`IngestResult.dedup`.

### 3.3 Wiring and config (`cli.py`, `config.py`)

`_build_ingest_driver` guards **both** tiers for detection (does gpt-5-mini ever do this?
The counters will say) and gives the cheap tier `strong_guard.raw` as fallback when
`dedup_retry_on_strong` (new setting, **default `True`**) is on. The `ingest` command
prints `dedup indices: calls=… invalid_calls=… dup_in_invalidation_range=…
dup_beyond_range=… contradicted_beyond_range=… parse_failures=… retried=…
retry_clean=… retry_dirty=…` after the existing summary line.

## 4. What was considered and rejected

- **A per-call `maximum` on the `duplicate_facts` items (schema bound).** Rejected as
  *harmful*, not merely fragile. Under constrained decoding the grammar masks tokens: a
  model committed to emitting `10` would be steered to `1` and produce an in-range wrong
  answer such as `[1, 1]` — a false dedup that merges two distinct facts and is invisible
  afterwards. A dropped index is detectable; a wrong merge is not. Enforcement is also
  provider-dependent (graphiti omits `strict: true`). The reachable-N question in the
  brief is therefore moot: even with N known, a value bound is the wrong tool.
- **A logging handler on `graphiti_core.utils.maintenance.edge_operations`.** The warning
  carries the invalid list and `N-1` but not `M`; `M` is only in a DEBUG line emitted
  before the LLM call, and `resolve_extracted_edge` runs under `semaphore_gather`, so
  DEBUG and WARNING records from different coroutines interleave and cannot be paired by
  order. The call-level wrapper pairs prompt and reply exactly, and it is the only layer
  where a retry is side-effect free. The handler is a strictly worse version of the same
  measurement.
- **Rewording the prompt.** It already states the constraint in capitals twice
  ("duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION
  CANDIDATES)"). Adding words for a cheap model is unmeasurable hermetically and would
  mean monkeypatching a library-owned prompt.
- **Article-level retry on the strong tier.** `add_episode` commits per episode; by the
  time an article's counters are known its cheap-tier episodes are already written.
  Re-running the article would either be skipped by the `already_ingested` gate or
  require deleting the just-written episodes — against design decision #3 — and would
  re-spend the whole article to fix one small call.
- **Reinterpreting the indices** (e.g. moving invalidation-range duplicates into
  `contradicted_facts`). The model said "duplicate", not "contradicts"; guessing what it
  meant is exactly the kind of plausible-mechanism assumption this project has been burned
  by.

## 5. Tests and the neutralisation proof

New files: `tests/unit/test_dedup_guard.py` (25 tests), `tests/unit/test_ingest_dedup_stats.py`
(5), `tests/unit/test_cli_dedup_wiring.py` (3) — 33 in total. All hermetic; the wiring test stubs
every constructor `_build_ingest_driver` touches.

Sixteen neutralisations were applied one at a time, each run against the three files and
restored with `git checkout` (`.superpowers/sdd/slice-b-mutations*.json` holds the raw
results). **Every mutant was killed**, and every fix-test is killed by at least one mutant:

| mutant | killed by |
|---|---|
| guard never installed | 15 tests |
| contiguity check removed | spoof test, parse-failure pass-through test |
| fallback gets the mutated messages | the UNMUTATED-messages test |
| fallback reply discarded | 4 (retry tests, real-pipeline e2e, CLI wiring) |
| in-range classification disabled | 11 |
| beyond-range classification disabled | 3 |
| prompt_name filter removed | ignores-other-prompts |
| context scope ignored | 4 |
| driver opens no scope | 4 |
| driver does not aggregate | aggregate test |
| driver does not log | log test |
| CLI wires no fallback | counted-once test |
| CLI wires the GUARDED strong client | counted-once test (double count) |
| driver scope not reset in `finally` | 2 (incl. the raises-path test) |
| every call treated as invalid | clean-call + no-retry-on-clean |
| `merge` overwrites instead of summing | merge test + aggregate test |

The six **library pins** (section 1 of `test_dedup_guard.py`) are *not* fix tests: they
run the real `resolve_extracted_edge` with a scripted client and pin the 0.30.1 semantics
the guard depends on (first-valid-only, all-invalid → new edge, contradicted drop, the
`valid_at` precondition, the both-ranges legality of `contradicted_facts`). They fail on a
graphiti upgrade that changes those semantics, which is their job. Likewise
`test_parse_counts_from_the_real_prompt_*` builds messages with the real prompt function:
if graphiti changes the prompt text, that test fails before the parser silently returns
`None` in production.

No `inspect.getsource()` assertions were added.

CI gate: `ruff check src tests`, `mypy src`, and the full non-live suite — see the report.

## 6. What this slice does NOT fix

- **Duplicates and missed invalidations already in the graph** from past events. No
  cleanup pass exists; a Cypher-only near-duplicate sweep would be a separate slice.
- **The strong tier's own out-of-range replies** — detected and counted, not retried
  (there is no stronger tier). Whether they occur at all is one of the numbers the next
  live run produces.
- **Calls whose prompt could not be parsed** (`parse_failures`) — passed through
  unchecked; graphiti's own validation still applies.
- **The search cap.** With `N = 10` an 11th same-pair fact is invisible to dedup. Untouched.
- **BACKLOG 13** (tier not persisted on `HAS_EPISODE`) — the per-article log line names
  the tier, but the graph still cannot attribute events per tier. **16b** untouched.
  **6** (false invalidations) untouched — the retry changes *which model* answers, not
  the temporal check that turns an answer into an `invalid_at`.
- **BACKLOG 16's `inspect.getsource()` test** in `test_llm_penalties.py` — not copied,
  not removed.

## 7. Live validation — required, and what it costs

Hermetic work cannot produce the rate, the in-range/beyond ratio, or the
`retry_clean` fraction. One cheap-tier ingest of a modest slice produces all three:

```
uv run --extra dev python -m graph_extract.cli ingest --source-id <src> --limit 50
```

Cost: the normal cheap-tier extraction cost of those articles plus **one `gpt-5-mini`
call (~1–2k tokens, small prompt) per detected event** — cents, not dollars, at any
plausible event rate. Read the `dedup indices:` line and the per-article WARNINGs:

- `dup_in_invalidation_range ≫ dup_beyond_range` → the confusion hypothesis holds.
- `dup_beyond_range` dominant, or negatives → hallucination; the retry is still the
  right lever, but the prompt is not the problem.
- `retry_dirty > 0` → the strong tier misreads the same prompt too: an interface
  problem, and the only remaining lever would be the (library-owned) prompt.
- `parse_failures > 0` → the parser's coupling to the prompt text has slipped; fix
  before trusting anything else.

BACKLOG 7 (extraction is not deterministic at `temperature=0`) means the *specific*
event will not reproduce; the *rate* will. Because the graph is written during that run,
it is a real ingest, not a probe — the user decides whether to spend it.
