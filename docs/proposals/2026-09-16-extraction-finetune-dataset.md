# Fine-tuning dataset for qwen-3.5-4b on the cheap extraction tier

**Date:** 2026-09-16
**Status:** Dataset built and verified. No training run, no money spent, no writes to
the live graph.
**Deliverables:** `scripts/build_extraction_dataset.py`,
`docs/proposals/samples/sample.jsonl` (20 records) + `sample-digest.md` (readable),
`docs/proposals/samples/schemas.json`.
**Supersedes one claim in** `docs/proposals/2026-09-15-extraction-model-self-hosting.md`
(see §1).

---

## 0. What was verified, and what is inference

Everything below is grounded in code at `file:line`, in a query against the live
`backup-docs` graph, or in a measurement printed by the builder. Where a number is
an estimate rather than a measurement it is labelled **[estimate]**. Four of the five
prompt reconstructions were proved byte-exact against graphiti's real call path and
the fifth against this project's own coupled parser (§3), rather than asserted.

---

## 1. The claim the proposal got wrong

The self-hosting proposal (2026-09-15, line 94-96) says distillation data is

> *"a capture job rather than a labelling job — `usage.py` already instruments every
> prompt/response pair"*

**That is false, and the correction matters more than the wording.**
`src/graph_extract/usage.py:42-47` is the whole of `record`:

```python
def record(kind: str, *, prompt: int, completion: int, cached: int = 0) -> None:
    _TALLY.add(kind, prompt=prompt, completion=completion, cached=cached)
```

`prompt` and `completion` are **integers**. `_tally_usage` (`usage.py:50-64`) reads
`resp.usage` — token counts — and nothing else. `instrument` (`usage.py:67-93`)
wraps `chat.completions.create` and `responses.create/parse`, passing `*args,
**kwargs` (which carry `messages`) straight through without reading them.

A full sweep of the repo confirms no prompt or completion text is persisted
anywhere:

| surface | what it holds |
|---|---|
| `usage.py` | integer token tallies only (`UsageTally`, `:8-24`) |
| `llm_timing.py` | call counts + latency lists keyed by `prompt_name` (`:44-54`) |
| `dedup_guard.py` | holds full prompt text in memory (`:114`, `:224`) but discards it on return; its stats dataclass (`:56-81`) is all ints, its log lines (`:218`, `:251-257`) emit indices only |
| Neo4j | model-authored *parsed fields* (`Episodic.content`, `Entity.summary`, `RELATES_TO.fact`) — never a raw completion |
| Postgres | `state_store.py:14-38` — cursors, webhooks, dead letters, a token ledger. No prompt column |
| `.superpowers/sdd/*.log` | index lists and counters. Longest line across all six logs: 674 chars; the largest model-output fragment is graphiti's own `"LLM returned summary for unknown entity (first 30 chars): AWS Backup"` |
| HTTP logging | none configured — `OPENAI_LOG` appears nowhere, no httpx logger, no `event_hooks` |

The only `(prompt, completion)` pair effectively on disk is in
`docs/superpowers/slice-2b-quality-baseline.json`: 15 `raw_answer` strings from the
**type-precision judge** (`eval.py:455-457`), whose prompt is reconstructible from
`_TYPE_JUDGE_PROMPT` (`eval.py:356-363`). That is an evaluation prompt, not one of
the five extraction prompts, so it is worth zero to this fine-tune.

**Consequence:** this is a *labelling* job. The proposal's step 4 ("harvesting a few
thousand examples per prompt type ... is mechanical") is not true today. It becomes
true after ~2 h of instrumentation work — which is option B below.

---

## 2. What data actually exists — the three options, priced

### Option A — reconstruct from the graph (**recommended**)

The live graph stores more than the proposal assumed. Measured read-only:

```
999 entities   655 episodes   3,590 RELATES_TO   4,479 MENTIONS   83 articles with episodes
Entity.name_embedding      999/999 populated, 768-d
RELATES_TO.fact_embedding  3,590/3,590 populated, 768-d
Episodic.content           655/655, 112–5,935 chars, mean 1,443
```

Three of those are decisive:

1. **`Episodic.content` is the exact prompt input.** `IngestDriver` calls
   `add_text_episode` once per chunk (`ingest_driver.py:191-194`), and graphiti's
   `concatenate_episodes` returns a single episode's content verbatim
   (`utils/text_utils.py:69-70`). So the `episode_content` slot needs no
   re-fetch. **DocExtractor is not needed at all** — contrary to the brief's
   framing, article text never has to be pulled back.
2. **`MENTIONS` and `RELATES_TO.episodes` give per-episode attribution**, so the
   targets for entity and relation extraction are recoverable per call.
3. **The stored embeddings reproduce graphiti's candidate ranking offline.** The
   same embedder wrote them during ingestion, so dedup candidate lists can be
   rebuilt with no embedder call and no cost.

**Cost: $0.00. Wall clock: 4 m 05 s measured** (`time` on the full build, one
command). Yield (printed by the builder):

| prompt | derivable | of which |
|---|---:|---|
| `dedupe_edges.resolve_edge` | 2,248 | 1,897 true negatives + 351 synthetic positives |
| `extract_nodes.extract_text` | 651 | all graph-derived |
| `extract_edges.edge` | 615 | all graph-derived |
| `extract_nodes.extract_summaries_batch` | 347 | all graph-derived |
| `dedupe_nodes.nodes` | 329 | negatives only |

**The honest loss.** The graph is the state *after* dedup and merging, and the loss
is not uniform — it is concentrated exactly where the brief warned:

- **`resolve_edge` and `dedupe_nodes` positives do not exist.** When graphiti
  resolves a duplicate it keeps the **existing** text and discards the newly
  extracted paraphrase. The string the model actually produced is gone. 1,232 merge
  events are visible in the graph (facts whose `episodes` list has more than one
  entry) but only as *events* — the text is unrecoverable. The negatives are fully
  faithful; the positives are structurally absent.
- **`extract_summaries_batch` has the weakest target in the set**, and is the
  smallest split for a reason. `Entity.summary` is the *final* summary after every
  later episode updated it, not the summary produced at the call being
  reconstructed, and the prior-summary state is unrecoverable, so the rebuilt
  prompt shows an empty existing summary. It is also the only prompt where the
  pipeline often makes **no LLM call at all**: `_extract_entity_summaries_batch`
  appends the node's edge facts to its summary and, when the result is
  `<= MAX_SUMMARY_CHARS * 2` (2,000 chars), assigns it directly and skips the LLM
  (`node_operations.py:880-882`). The builder replays that gate — which is why this
  split is 347 examples and not 655. Replaying it against the *final* summary
  over-admits slightly, and that is disclosed per record.
- **`extract_text` / `edge` names are post-dedup canonical names.** Where dedup
  merged an extracted name into an existing node, the target carries the survivor.
- **Candidate ordering is approximate.** Candidates are ranked by stored-embedding
  cosine and assembled in graphiti's own union order; graphiti's live ranking is
  hybrid fulltext+vector RRF. The candidate *set* largely overlaps, the *indices*
  can differ — which matters for an index-picking task.

Every one of these is recorded per record in `meta.fidelity`, so a reviewer sees the
caveat next to the example rather than only in this document.

### Option B — add capture instrumentation and re-ingest the 83 pilot articles

Truthful pairs for all five prompts, **including the positives Option A cannot
produce**. The instrumentation is a near-copy of `dedup_guard.install_dedup_guard`
(`dedup_guard.py:187-261`): wrap `graphiti.llm_client.generate_response`, write
`(prompt_name, messages, response)` to JSONL. It already keeps a `pristine` copy of
the messages at `:224`.

- **Cost: ~$10** — 83 articles × ~$0.12/article blended across both tiers
  (2026-09-15 proposal, line 101; the only measured per-article figure in the repo).
- **Wall clock: 2 h 10 m at N=4, 6 h 55 m sequential** (`ab-w8-validation.py:5-6`).
- **Dev: ~2 h** for the capture wrapper and a runner.
- **Risk:** it writes to the graph. It must use the established supersede-and-restore
  method (`ab-w8-validation.py:11-13`, `ab-revert.py`) and **never** reset the
  semantic layer — the 999-entity graph is the project's only reference copy.
- **Requires explicit authorisation** (money + a run of the ingest path the brief
  forbids by default).

### Option C — distil from the strong tier (`gpt-5-mini`)

Highest-quality targets, and the only option that can *exceed* the current teacher.

- **Cost: not derivable from anything recorded here.** The repo has no price table
  (`cost_report`, `eval.py:213-242`, reports tokens, not dollars) and the $0.12
  figure is blended across tiers, so it cannot be split. **[estimate]** a
  strong-tier-only pass over the same 83 articles is 2–4× the blended figure,
  i.e. **~$20–40** — but that is a guess and should be replaced by a priced probe
  on ~2 articles before anything larger is launched, per the standing cost rule in
  CLAUDE.md.
- **Requires explicit authorisation.**

### Recommendation

**Build from the graph now (A), and treat B as the targeted follow-up that buys the
one thing A structurally cannot: true index-picking positives.** A is free, is
already done, and covers ~90% of the prompt surface faithfully. B costs ~$10 and
fixes the specific hole. C is not worth its price until A+B have been measured,
because its advantage is quality headroom on a task where the binding constraint is
format, not comprehension (§5).

---

## 2b. Four defects found in review, and what they cost

The builder was reviewed against graphiti's source after it first ran clean. Four
real defects came out of that, all of which produced *plausible-looking* data rather
than errors. Recording them because each one is a way this kind of reconstruction
fails silently:

1. **`as_datetime` dropped the timezone.** A "keep the digit characters" filter ate
   the digits out of the UTC offset as well as the fractional seconds, yielding a
   *naive* datetime. graphiti's `EpisodicNode.valid_at` is tz-aware, and `str(dt)`
   on a naive datetime omits the `+00:00` that production renders into
   `<REFERENCE_TIME>`. **My own fidelity harness could not catch this**, because it
   built its EpisodicNode from the same converted value — a circular check. It is now
   asserted against the neo4j driver's native `to_native()` conversion instead:
   655/655 equal. On a `-05:00` input the old code also mis-parsed the fraction
   (120500 µs instead of 120000).
2. **`resolve_edge` grouped facts by an *undirected* node pair.** graphiti's
   duplicate candidates come from `EntityEdge.get_between_nodes`, whose Cypher is
   directional: `(n {uuid:$source})-[e:RELATES_TO]->(m {uuid:$target})`
   (`edges.py:421-423`). The undirected grouping showed the model B→A facts that were
   never in `EXISTING FACTS`, and mislabelled as "negatives" facts whose real
   candidate list was empty — where `edge_operations.py:653` issues **no LLM call at
   all**. Cost: 86 fabricated records (1,983 → 1,897).
3. **The summary-batch LLM gate was not replayed** (see §2). Cost: 308 records — 47%
   of that split — described calls that never happened.
4. **`dedupe_nodes` candidates were re-sorted by `created_at`**, destroying the
   similarity order `_merge_candidate_nodes` produces (`node_operations.py:387-404`)
   and renumbering every `candidate_id`. This also contradicted the builder's own
   `meta.fidelity` note, which claimed cosine ordering.

A fifth reported issue — the missing `**candidate.attributes` spread in
`existing_nodes` (`node_operations.py:519-528`) — was checked and is **not** a defect
here: no entity type in `ontology.py` declares attributes (deliberately, so graphiti
skips attribute extraction entirely), and the live graph confirms it — `Entity`
property keys are exactly `created_at, group_id, labels, name, name_embedding,
summary, uuid`. `candidate.attributes` is `{}`, so the spread is a no-op. It would
become a real gap the moment an attribute is added to any entity type.

---

## 3. Fidelity: proved, not asserted

The dataset must match graphiti 0.30.1's templates exactly — a fine-tune trained on
a paraphrase is trained on the wrong task. Two defences:

**The builder does not contain any prompt text.** It imports
`graphiti_core.prompts.prompt_library` and calls the real functions. This matters
more than it looks: `prompt_library` wraps every version in a `VersionWrapper` that
appends `DO_NOT_ESCAPE_UNICODE` to system messages (`prompts/lib.py:69-77`). Calling
the module-level functions directly would silently lose that line.

It then replays the mutation `OpenAIGenericClient.generate_response` performs before
the request goes out:

1. `messages[0].content += get_extraction_language_instruction(group_id)`
   (`openai_generic_client.py:203`)
2. `_clean_input` on every message (`:149`) — invoked through a real client instance
   so it is graphiti's code, not a copy.

The cheap tier runs `cheap_llm_client_mode='generic_json_schema'`
(`config.py:236-237`), so the JSON schema travels in `response_format` and is **not**
injected into the prompt text — the `json_object` branch at `:193-200` does that, and
it is not the branch we use.

**Verification.** A harness drove graphiti's own call path — `extract_nodes`,
`extract_edges`, `_resolve_with_llm`, `_process_summary_flight`
(`graphiti_core.utils.maintenance.*`) — against a fake LLM transport that captures
what *would* have been sent, then compared it to the builder's output for the same
episode:

```
extract_nodes.extract_text:            MATCH
extract_edges.edge:                    MATCH
dedupe_nodes.nodes:                    MATCH
extract_nodes.extract_summaries_batch: MATCH
```

Byte-for-byte, including the unicode note, the multilingual suffix and `_clean_input`.
Two bugs were caught this way before review and fixed rather than shipped: `edge` was
passing `nodes` as a list of bare name strings where graphiti passes
`[{'name': ..., 'entity_types': ...}]` (`edge_operations.py:189`), and `reference_time`
was being rendered from Neo4j's nanosecond `toString()` instead of `str(datetime)`.
Neither would have raised an error — both would have silently trained the model on a
prompt shape it will never see. Four further defects came out of code review; see §2b.

All four comparisons were re-run after those fixes and still MATCH, and the datetime
conversion is now cross-checked against the driver's native path rather than itself.

(`extract_summaries_batch` is compared with the candidate summaries blanked, because
the builder deliberately sends an empty prior summary — the call-time state is
unrecoverable. That is the §2 fidelity gap, not a rendering mismatch.)

**A second, independent tripwire** for `resolve_edge`: all 1,156 rendered dedup
prompts were fed to this project's own `dedup_guard.parse_candidate_counts`
(`dedup_guard.py:106-122`), which is deliberately coupled to graphiti's prompt text.
All 1,156 parsed, every `N` matched the record's candidate count, and every `M` was
0 — confirming the single-index-range shape the contradiction gate produces (below).

**And the targets are schema-valid**: all 2,890 assistant messages parse into their
pydantic `response_model` (`ExtractedEntities`, `ExtractedEdges`,
`SummarizedEntities`, `EdgeDuplicate`, `NodeResolutions`). 0 failures.

### One production-specific detail the dataset must carry

`contradiction_gate` returns an empty `SearchResults()` for the unfiltered
invalidation search (`contradiction_gate.py:98-101`), so in production
`edge_invalidation_candidates` is **always `[]`** and the `resolve_edge` prompt
carries **one** index range, not two. The dataset reflects that. This is
load-bearing: `dedup_guard.py:25-27` records that *all 267 observed out-of-range
indices landed in the second range*. Training on two-range prompts would teach a
failure mode the production pipeline no longer presents.

Also note graphiti issues **no LLM call at all** when both candidate lists are empty
(`edge_operations.py:653`), and only entities that survive the deterministic
exact-name/MinHash pass reach `dedupe_nodes.nodes`
(`node_operations.py:649-670`). The builder replays both filters using graphiti's own
`_resolve_with_similarity` and `_build_candidate_indexes`, so the difficulty
distribution matches production instead of being systematically easier.

---

## 4. Format

**Chat-completion style, JSONL, one record per line.**

Why chat and not completion-style: graphiti sends role-separated messages
(`openai_generic_client.py:150-155` builds `{'role': ..., 'content': ...}`), and
serving will be an OpenAI-compatible endpoint. A completion-style fine-tune would
have to invent a flattening that does not match what the serving stack sends, which
reintroduces exactly the train/serve mismatch this whole exercise is trying to avoid.

```jsonc
{
  "prompt_name": "dedupe_edges.resolve_edge",
  "response_schema": "EdgeDuplicate",          // key into schemas.json
  "messages": [
    {"role": "system",    "content": "..."},   // verbatim, post-mutation
    {"role": "user",      "content": "..."},
    {"role": "assistant", "content": "{\"duplicate_facts\": [], \"contradicted_facts\": []}"}
  ],
  "meta": {
    "provenance": "graph_derived_negative",    // see below
    "fidelity": ["true negative: this fact survived as its own edge, ...", "..."],
    "loss_weight": 2.21,                       // see §6
    "episode_uuid": "...", "article_id": "...", "chunk_index": 3,
    "fact_uuid": "...", "n_candidates": 7
  }
}
```

`provenance` is one of:

| value | meaning |
|---|---|
| `graph_derived` | the target is what the pipeline produced, with `fidelity` caveats |
| `graph_derived_negative` | a real negative — the call happened and "no duplicate" was correct |
| `synthetic_alias_positive` | **REVIEW BEFORE TRAINING.** See §7 |

`schemas.json` carries the exact `response_format` payload per prompt, including the
`maxItems: 25` that `graphiti_client._bound_index_arrays` injects (`:116-126`, bound
from `config.py:75`). Emitting it separately keeps the JSONL readable and gives the
training/serving stack the grammar directly.

> **Incidental finding — a docstring in `src/` undercounts.**
> `graphiti_client._bound_index_arrays:96-98` states *"graphiti emits three such
> fields"* and lists `ExtractedEdges.Edge.episode_indices`,
> `EdgeDuplicate.duplicate_facts`, `EdgeDuplicate.contradicted_facts`. Enumerating
> the five response models' JSON schemas finds **four**: it misses
> `ExtractedEntities.ExtractedEntity.episode_indices`
> (`prompts/extract_nodes.py:34-38`).
> **The code is correct** — `_bound` walks the schema generically and bounds all
> four, which is why `schemas.json` contains 4 `maxItems`. Only the comment is
> wrong. Not fixed here because this task must not modify `src/`; worth a one-line
> correction when someone is next in that file.

---

## 5. The schema problem

Every recorded cheap-tier failure in this project has been a **format** failure:

- *graphiti unbounded index arrays — the real cause of every cheap-model extraction
  failure; `maxItems` fixes it, repetition penalties make it worse*
- *ling is dead; require `structured_outputs` in `supported_parameters` and probe
  with a strict `json_schema` before trusting any cheap model*

`graphiti_client._bound_index_arrays:93-113` documents the mechanism precisely: an
ascending integer run is the strongest sequence prior there is, and under a JSON
grammar the only legal continuations are digits, `,` and `]`, so a weak model counts
until `max_tokens` runs out and truncates into invalid JSON. Observed with an
`episode_indices` run past 2,900.

### Does grammar-constrained decoding make the dataset moot?

**Partly — and knowing exactly which part is the useful answer.**

A hard grammar (vLLM + xgrammar/outlines) makes *syntactic* malformation
**unrepresentable**: the reply is always parseable JSON matching the schema, arrays
respect `maxItems`, enum fields respect their enums. That class of failure — the one
that killed ling and forced the frequency-penalty and `maxItems` patches — is solved
by the serving stack, not by the fine-tune. It is strictly stronger than any API's
best-effort structured-output mode, and graphiti deliberately omits `strict: true`
(`openai_generic_client.py:112-117`), so today enforcement is provider-dependent.

What a grammar **cannot** fix, and what this dataset therefore has to teach:

1. **Semantic validity inside a syntactically legal shape.** A grammar will happily
   emit `duplicate_facts: [17]` when only indices 0-6 exist. That is exactly the
   observed failure — 267 out-of-range indices — and `dedup_guard` refuses to clamp
   it on purpose (`:32-36`): under constrained decoding a `maximum` bound would turn
   `[10, 11]` into an in-range `[1, 1]`, "a wrong dedup that merges two distinct
   facts and is invisible afterwards. A detectable drop is strictly better than an
   undetectable merge." **The index range is a property of the prompt, not of the
   schema, so only training can teach it.**
2. **`source_entity_name` / `target_entity_name` must be drawn from the ENTITIES
   list** (`extract_edges.py:146-147`: "Using names not in the list will cause the
   edge to be rejected"). A grammar cannot express "one of the strings that appeared
   earlier in this prompt" — a dynamic per-request enum would, but graphiti builds
   the schema from a static pydantic model.
3. **Calibration** — how many facts a chunk should yield, when to answer `-1`, when
   an empty list is right. `CHEAP_TIER_SALIENCE` (`ontology.py`) exists purely
   because the cheap model over-enumerated dense tables; that is a learned
   behaviour, not a grammar.

**So: use grammar-constrained decoding *and* fine-tune, and understand the division
of labour.** The grammar buys syntax for free, which means the fine-tune's job is
narrower than the proposal assumed — it is a *pointing* task (pick the right index,
pick names that exist, decide how much to emit), not a JSON-formatting task. That is
good news for a 4B (§8).

**How the dataset teaches the schema anyway:** every target is emitted as the exact
compact JSON serialisation of a validated `response_model` instance, so the model
sees the canonical key order, the `null` convention for absent timestamps, and the
`[0]` convention for `episode_indices` in every single example. The 2,890/2,890
pydantic validation pass rate is the check that this holds.

---

## 6. Weighting — and the finding that changes it

The brief asked for weighting by the measured LLM-time share, noting that
index-picking (`resolve_edge` + `dedupe_nodes` = 33.8%) is where a 4B is most likely
to succeed. **Chosen target mix, by design, up-weighting index-picking to 50%:**

| prompt | LLM-time share (2026-09-14 A/B) | chosen share |
|---|---:|---:|
| `dedupe_edges.resolve_edge` | 22.9% | **40%** |
| `extract_edges.edge` | 33.6% | 20% |
| `extract_nodes.extract_text` | 12.7% | 18% |
| `extract_nodes.extract_summaries_batch` | 19.8% | 12% |
| `dedupe_nodes.nodes` | 10.9% | 10% |

Rationale: LLM-*time* share is driven by output length, not by call difficulty or
call count. `resolve_edge` is 22.9% of time while emitting a ~12-token answer — it is
a high-frequency, low-output, high-failure-rate call, and it is where every recorded
out-of-range failure landed. `extract_summaries_batch` is down-weighted because it
has the weakest targets in the dataset (§2).

### The finding: example share is not gradient share

Standard SFT masks the loss to completion tokens. A prompt's influence on the
gradient is therefore its share of **completion** tokens — and on this dataset those
three views diverge enormously. Measured on the 2,357-example train split:

| prompt | by example count | by full tokens | **by completion tokens** |
|---|---:|---:|---:|
| `extract_edges.edge` | 20.1% | 39.7% | **48.8%** |
| `extract_nodes.extract_summaries_batch` | 12.3% | 20.2% | **37.3%** |
| `extract_nodes.extract_text` | 18.1% | 16.2% | **10.1%** |
| `dedupe_edges.resolve_edge` | 39.8% | 8.3% | **1.9%** |
| `dedupe_nodes.nodes` | 9.8% | 15.6% | **1.9%** |
| **index-picking total** | **49.5%** | **24.0%** | **3.8%** |

A `resolve_edge` target is ~12 tokens; an `edge` target is ~460 and a
`summaries_batch` target ~499. So a dataset that is 49.5% index-picking *by example
count* delivers **3.8% of the gradient** to index-picking, while the two long-output
extraction prompts take 86% between them. Building the file to the nominal mix and
training normally would produce close to the opposite of the intended fine-tune.

**Correction:** the builder writes `meta.loss_weight` on every record, set so
TARGET_MIX is the **completion-token** share. Weights on the current build:

```
dedupe_edges.resolve_edge              2.24
dedupe_nodes.nodes                     0.55
extract_nodes.extract_text             0.20
extract_edges.edge                     0.04
extract_nodes.extract_summaries_batch  0.03
```

Mean weight is normalised to 1.0 so the effective learning rate is comparable to an
unweighted run. **Caveat:** a ~100× spread is numerically aggressive and can spike
gradients. A trainer that cannot apply per-example weights should **upsample by
repetition** instead (repeat each `resolve_edge` example ~20×), accepting the
overfitting risk on repeats. Either lever is fine; ignoring the issue is not.

### Size

| | |
|---|---|
| total examples | 2,890 (2,357 train / 533 val) |
| total tokens | ~11.3 M — **[estimate]**, chars÷4, not tokenised with Qwen's tokenizer |
| completion tokens | ~588 k (train) |
| median prompt | 779 tok (`resolve_edge`) → 7,509 tok (`edge`) |
| p95 / max prompt | 10.3 k / **17.3 k** tokens |

**The max matters:** training and serving both need ≥16 k context. The long tail comes
from `previous_episodes` — graphiti attaches the last 10 episodes
(`graphiti.py:1088-1093`, `RELEVANT_SCHEMA_LIMIT = 10`), which the builder replays.

---

## 7. The synthetic positives — read this before training

`resolve_edge` negatives are faithful. Positives are not derivable (§2). A
negatives-only split would teach the model to **always answer `[]`**, which is worse
than no fine-tune at all — the dedup call would become a no-op and duplicate facts
would enter the graph permanently (`dedup_guard.py:11-13`).

The builder therefore derives 351 `synthetic_alias_positive` records (175 survive
the mix into the current build — 15% of the `resolve_edge` split). The rule is
narrow and auditable: take a real fact, swap a canonical term for a non-canonical
alias that `ontology.py`'s `EXTRACTION_INSTRUCTIONS` explicitly tells the extractor
to fold into that canonical name (`Kubernetes`↔`K8s`, `Amazon S3`↔`S3 buckets`,
`immutability`↔`WORM`, …). The restated fact is a duplicate **by this project's own
declared semantics**, and it is not a verbatim string match, so graphiti's exact-match
fast path (`edge_operations.py`, `_normalize_string_exact`) would not have
intercepted it. The rule used is written into each record's `meta.fidelity`.

**They are still synthetic. No model emitted them.** Review the sample before
training, and prefer replacing them wholesale with Option B's captured positives.
`--no-synthetic-positives` omits them.

`dedupe_nodes.nodes` has **no** positives, synthetic or otherwise, and the builder
says so in every record. This is the largest remaining hole in the dataset.

---

## 8. Train/validation split, and the gate

**Split by article, not by episode.** Episodes of one article share entities and
near-identical prose; an episode-level split leaks the answer across the boundary.
`split_of` hashes `article_id` (SHA-256, first 8 hex) into a stable 80/20 split.
Verified on the current build: **66 train articles / 17 val articles, 0 overlap.**

**What the validation set is for — and what it is not.** Held-out cross-entropy tells
you the model learned this dataset's idiosyncrasies, including the ones §2 says are
wrong (post-dedup names, final-state summaries). It is a training-health signal only.
**It must not be the acceptance gate.**

**The acceptance gate is the measurement that already exists.** Do not invent a
metric. Run `.superpowers/sdd/ab-w8-validation.py` — the same 83 pilot articles, the
cheap tier swapped to the fine-tune, against the 999-entity sequential baseline. It
already reports every number that matters:

| signal | baseline | source |
|---|---|---|
| entity count after merge | **999**, 0 exact-name duplicates | `ab-w8-validation.py:5, 39` |
| near-duplicate groups | 6 | `:172` |
| out-of-range dedup indices | `DedupIndexStats.summary()` — `invalid_calls`, `dup_in_invalidation_range`, `dup_beyond_range` | `:177`, `dedup_guard.py:56-81` |
| fact yield | 3,590 `RELATES_TO` | live graph |
| provenance integrity | `eval.provenance_report` | `eval.py:183` |
| noise rate | `eval.noise_report` (pure Cypher, free) | `eval.py:65` |
| fact / type quality | `eval.fact_quality`, `eval.type_precision` (**LLM judge — costs money**) | `eval.py:297`, `:390` |

`DedupIndexStats` is the single most direct read on whether the fine-tune worked: it
counts precisely the failure this dataset is weighted to fix. A fine-tune that does
not reduce `invalid_calls` against the baseline has not earned its place, whatever
its validation loss says.

**Method constraint:** the A/B must use supersede-and-restore (`ab-revert.py`),
never a reset of the semantic layer. The 999-entity graph is the only reference copy
and destroying it destroys the baseline.

**Cost note:** running the A/B is not free even with a local model — the strong tier
still handles dense-matrix articles and the embedder still runs. Budget it as a
paid run and authorise it explicitly.

---

## 9. The coupling risk, stated plainly

**A fine-tune trained on graphiti 0.30.1's prompts goes stale the moment graphiti is
upgraded.** This project pins `graphiti-core==0.30.1` and never forks it, extending
instead by patching imported-by-name references (`dedup_guard`,
`contradiction_gate`, `deterministic_valid_at`, `lean_edge_search`). A fine-tune adds
a **second artefact that must move in lockstep with a library we do not control** —
and unlike the patches, it cannot be fixed by editing a file. It has to be retrained.

This is the 2026-09-15 proposal's central argument against fine-tuning, and building
the dataset does not refute it. What the dataset does change is the *cost* of the
coupling:

- The builder contains **no prompt text**. It imports `prompt_library` and renders
  whatever the installed graphiti says. On a graphiti upgrade, re-running it
  regenerates a correct dataset with no authoring work.
- The **targets survive an upgrade**; only the prompt rendering changes. Entity,
  fact and dedup-decision labels are properties of the corpus, not of the template.
- So the maintenance unit is **"re-render and retrain" (~free + GPU hours)**, not
  "re-label" (~$10 + a day).

**Concretely, what to commit to:**

1. Pin `graphiti-core==0.30.1` *harder* — it is now load-bearing for a model
   artefact, not just for code.
2. Add a **tripwire test** that fails CI when any of the five prompt templates
   changes. `tests/unit/test_contradiction_gate.py` already establishes the
   call-site-tripwire pattern, and `dedup_guard.parse_candidate_counts` is already
   coupled-and-tested this way. Hash the rendered output of each of the five prompt
   functions against a fixed context; a graphiti bump that changes one turns red.
   Without this the staleness is **silent** — the model keeps answering, just worse.
3. Treat the model version and the graphiti version as **one unit** in deployment.

**And the honest framing:** if the goal is genuinely "settle on one extraction model
and stop changing it", the proposal is right that self-hosted base weights behind a
hard grammar is the lower-coupling answer, and should be measured *first*. This
dataset's real value is that it is now **free to produce and free to regenerate**, so
the fine-tune can be tried as step 4 without a labelling project in front of it —
which is precisely the position the proposal wrongly believed it was already in.

---

## 10. 4B versus the 9B the proposal assessed

The proposal evaluated a 9B and explicitly declined to claim anything about its
quality ("No claim is made here about Qwen-3.5-9B's specific quality on this task").
A 4B is less than half the parameters. What changes:

**Gets easier / stays fine**

- **Hardware stops being the gate.** The proposal's blocker was "`nvidia-smi` finds
  no GPU" and "a 9B wants ~24 GB at bf16, ~16 GB quantized". A 4B is ~8-9 GB at
  bf16 and ~4-5 GB at 4-bit — it fits on a single consumer card, and LoRA fine-tuning
  fits comfortably on 24 GB. **[estimate]**, standard parameter arithmetic, not
  measured here. This may convert "collapses entirely" into "feasible on one
  workstation".
- **Format conformance is not a size problem** once grammar-constrained decoding is
  on (§5). The failures that killed ling were decoding-dynamics failures, and a
  grammar fixes those for a 4B exactly as well as for a 9B.
- **Index-picking is the best-suited 40% of the workload.** It is near-classification
  over a short prompt (median 791 tokens) with a ~12-token answer. This is the part
  most likely to survive the shrink, and it is what the weighting prioritises.

**Gets harder**

- **`extract_edges.edge` is the real risk.** 33.6% of LLM time, a ~7.5 k-token
  prompt, a ~460-token structured answer, and an instruction set that is genuinely
  long — `EXTRACTION_INSTRUCTIONS` + `CHEAP_TIER_SALIENCE` (`ontology.py`) run to
  several hundred words of type boundaries and canonical-name rules, on top of
  graphiti's own seven extraction rules. Following many simultaneous negative
  constraints ("never generalize", "never the same entity twice", "only names from
  the list") is where parameter count tells. **Fine-tuning helps here more than
  anywhere else**, because the instructions can be partly absorbed into weights —
  but a 4B is more likely to need the fine-tune to *reach* baseline rather than to
  beat it.
- **Long-context degradation.** p95 prompt is 10.3 k tokens and the max is 17.3 k.
  Small models degrade faster over long contexts, and the `previous_episodes` block
  is a large, low-value part of that (10 prior episodes, often unrelated). If the
  4B struggles, **dropping `previous_episodes` is the first lever** — it is a
  graphiti default, not a project requirement, and `add_episode` accepts
  `previous_episode_uuids` to override it (`graphiti.py:992`).
- **Over-enumeration risk returns.** `CHEAP_TIER_SALIENCE` exists because the
  previous cheap model produced one fact per table cell. A smaller model is *more*
  prone to that, and `maxItems` bounds the index arrays but not the `edges` list
  (deliberately — `_bound_index_arrays:106-108` leaves arrays of objects alone so
  fact yield is untouched).

**Practical consequence:** consider training **two adapters or a mixed adapter with
an explicit prompt-type signal**, and gate them separately. Index-picking is likely
to pass at 4B; `extract_edges.edge` may not. Measuring them separately is cheap —
`DedupIndexStats` and fact yield are already reported independently by the A/B
runner — and it avoids discarding a fine-tune that succeeded at 40% of the workload
because it failed at another 34%. It also preserves the option of keeping
`gpt-5-mini` on edge extraction while a local 4B takes the dedup calls.

---

## 11. Reproducing this

```bash
# full build (free, read-only, no LLM/embedder/DocExtractor calls)
uv run --extra dev python scripts/build_extraction_dataset.py --out data/ft

# the committed 20-record sample
uv run --extra dev python scripts/build_extraction_dataset.py \
    --out docs/proposals/samples --limit 20 --sample-only
```

Useful flags: `--prompts` (subset), `--tier strong` (render the strong tier's
instructions instead), `--no-synthetic-positives`, `--val-fraction`, `--seed`.

The builder opens its Neo4j session with `default_access_mode="READ"` and issues
only `MATCH`. It never writes, never calls an LLM, never calls the embedder, and
never contacts DocExtractor.

---

## 12. Open questions for the user

1. **Authorise Option B (~$10, 2-7 h wall clock, writes to the graph under
   supersede-and-restore)?** It is the only way to get true `resolve_edge` and
   `dedupe_nodes` positives, which is the dataset's biggest hole.
2. **Is the GPU story resolved?** The 2026-09-15 proposal gated everything on it. A
   4B changes the arithmetic enough that the answer may now be different.
3. **Is `MAP_LLM_MODEL` (also Solar Pro 4, the global-search map step) in scope?**
   It is not in this dataset — only the five extraction prompts are.
4. **Should `extract_edges.edge` be in scope at all for a 4B**, or should the
   fine-tune target index-picking only and leave edge extraction on an API model
   (§10)?
