# Opus Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** build a dataset of (production prompt, Claude Opus answer) pairs for every cheap-tier graphiti prompt under the new ontology, and the Colab notebook that trains qwen3.5-4b on it.

**Architecture:** graphiti's REAL `add_episode` pipeline runs on production chunks with every production wrapper installed. Two substitutions:

- **Transport.** The LLM transport is a fake `AsyncOpenAI` whose `chat.completions.create` returns a stored teacher answer, or raises `TeacherNeeded` with the exact request.
- **Driver.** The graph driver is read-only.

A chunk is re-run round after round; each round replays the answered calls and stops at the next unanswered prompt. Subagents (the teacher) answer the pending prompts in batches, and an intake step validates each answer before storing it. The production capture instrumentation records each answered call in the existing capture format, which the existing dataset builder turns into train/val files. The live graph is never written: the final save of each chunk fails on the read-only driver, and that failure is the completion signal.

**Tech Stack:** Python 3.12, graphiti-core 0.30.1, Neo4j (read-only), uv, Unsloth/TRL (Colab notebook).

**Spec:** docs/superpowers/specs/2026-10-17-opus-distillation-design.md

## Global Constraints

- The live graph is NEVER written. Every harness Neo4j session is READ access mode, and a write attempt raises. No `init_indices`, no graphiti index build.
- Prompts are byte-identical to what the cheap tier sends in production. The harness uses the production client construction (instrument, `_bound_index_arrays`, cache layout, `install_dedup_guard`, grounding context with `document_context`). Only the transport and the graph driver are swapped.
- Contradiction detection stays as production has it (`ingest_detect_contradictions=False`; suppression on).
- No API spend. The teacher is Claude Code subagents. Nothing in this repo rents a GPU.
- Teacher answers are validated before they are stored: schema, index ranges, and the grounding gate as a validator (an answer the gate would change is rejected). Each answer is re-asked once with the reason, then dropped and counted. Never silently repaired.
- Never write `content or ""`; hermetic unit tests (`ExtractSettings(_env_file=None, ...)` inside tests only); lint `uv run ruff check src tests scripts/distill`; `uv run mypy src`.
- The sample is about 100 articles: about 80 train and about 20 held out, split by article. Strata are vendor/product family across Veeam, Microsoft, AWS and Cohesity (at least 5 per vendor), with hard cases oversampled.

## File Structure

- `src/graph_extract/graphiti_client.py`: `build_cheap_graphiti` and `_llm_client` gain optional `transport=` and `graph_driver=` injection (defaults unchanged).
- `scripts/distill/__init__.py`
- `scripts/distill/readonly_driver.py`: `ReadOnlyNeo4jDriver`, a graphiti `Neo4jDriver` that never builds indices and refuses writes.
- `scripts/distill/teacher.py`: `AnswerStore`, `TeacherNeeded` and `TeacherTransport` (the fake AsyncOpenAI).
- `scripts/distill/sample.py`: deterministic stratified selection and the train/held-out split.
- `scripts/distill/harness.py`: CLI with `select | round | intake | status | review-sample | export`.
- `scripts/distill/validate.py`: per-prompt answer validation.
- `scripts/distill/TEACHER_BRIEF.md` and `scripts/distill/REVIEWER_BRIEF.md`.
- `scripts/build_extraction_dataset.py`: `load_graph` gains an `article_ids=` scope with embeddings skipped.
- `notebooks/distill_train.ipynb`, `scripts/distill/package_dataset.py` and `docs/superpowers/distill-runbook.md`.
- Tests: `tests/unit/test_distill_teacher.py`, `test_distill_validate.py`, `test_distill_sample.py`, `test_distill_harness.py`, `test_graphiti_client_transport.py`, and `tests/integration/test_distill_readonly.py`.

---

### Task 1: Injectable transport and read-only driver

**Files:** modify `src/graph_extract/graphiti_client.py`; create `scripts/distill/__init__.py` and `scripts/distill/readonly_driver.py`; tests `tests/unit/test_graphiti_client_transport.py` and `tests/integration/test_distill_readonly.py`.

**Interfaces produced:**
- `build_cheap_graphiti(settings, *, transport=None, graph_driver=None) -> Graphiti`. When `transport` is given, it replaces ONLY the raw `AsyncOpenAI(...)` construction inside `_llm_client`; every wrapper applied to the raw client today is applied to `transport` in the same order. When `graph_driver` is given, `Graphiti(graph_driver=graph_driver, ...)` is used instead of the URI. Both default to today's behaviour.
- `add_text_episode(..., previous_episode_uuids: list[str] | None = None)` is passed through to `graphiti.add_episode(previous_episode_uuids=...)`. The default None is unchanged (graphiti then looks up the last 10 episodes itself).
- `ReadOnlyNeo4jDriver(uri, user, password, database=None)`, a subclass of graphiti's `Neo4jDriver`:
  - it never schedules or runs graphiti's index/constraint build;
  - every session opens with `default_access_mode=READ_ACCESS`;
  - every `execute_query` passes `routing_=RoutingControl.READ`;
  - it raises `ReadOnlyViolation` (a new exception class) if a caller asks for write access.

  Server-side, a write sent through a READ session is rejected by Neo4j itself, the second line of defence.

**Steps:**
- [ ] Read graphiti 0.30.1's `Neo4jDriver` (`graphiti_core/driver/neo4j_driver.py`) to find where it builds indices and opens sessions, and override exactly those. Read `_llm_client`'s non-Azure, non-Anthropic branch (`graphiti_client.py:~255-290`).
- [ ] Unit test (`test_graphiti_client_transport.py`):
  - a fake transport object with `chat.completions.create` recording calls, passed to `build_cheap_graphiti(settings, transport=fake)`;
  - one `llm_client.generate_response([...], response_model=..., prompt_name="extract_nodes.extract_text")` call reaches `fake` with `response_format` carrying `maxItems` (from `_bound_index_arrays`) and the messages after `OpenAIGenericClient`'s mutation;
  - with `llm_capture_path` set to a tmp file, one capture record is written;
  - default construction (no transport) is unchanged: the existing graphiti_client tests stay green.
- [ ] Integration test (`tests/integration/test_distill_readonly.py`, Neo4j testcontainer as in tests/integration):
  - a `ReadOnlyNeo4jDriver` on the container runs a MATCH;
  - a CREATE through its session/execute_query raises (either `ReadOnlyViolation` or the server's read-only error), and the node count is unchanged;
  - constructing it runs no index build: `SHOW INDEXES` count unchanged.
- [ ] Implement; unit suite, the integration file, ruff, mypy; commit `feat(extract): injectable transport and graph driver for the cheap graphiti; read-only driver for the distillation harness` (trailers).

### Task 2: Teacher transport and answer store

**Files:** create `scripts/distill/teacher.py`; test `tests/unit/test_distill_teacher.py`.

**Interfaces produced:**

```python
class TeacherNeeded(Exception):
    def __init__(self, key: str, request: dict): ...  # request = {"prompt_name", "messages", "response_format", "model"}

class AnswerStore:
    def __init__(self, root: Path): ...              # root/answers.jsonl (append-only), root/pending/
    def key_for(self, messages: list[dict], response_format: dict | None) -> str: ...  # sha256 of canonical JSON (sort_keys, ensure_ascii=False)
    def get(self, key: str) -> str | None: ...       # the validated answer JSON text
    def put(self, key: str, answer_json: str, meta: dict) -> None: ...
    def add_pending(self, key: str, request: dict) -> None: ...  # idempotent; pending/<key>.json
    def pending(self) -> list[tuple[str, dict]]: ...
    def drop(self, key: str, reason: str) -> None: ...  # root/dropped.jsonl; a dropped key raises TeacherDropped on replay

class TeacherDropped(Exception): ...

class TeacherTransport:
    """Fake AsyncOpenAI: .chat.completions.create(**kwargs) returns a stored answer or raises TeacherNeeded."""
    def __init__(self, store: AnswerStore, prompt_name_getter: Callable[[], str]): ...
```

- `create(**kwargs)` computes `key_for(kwargs["messages"], kwargs.get("response_format"))`:
  - a stored answer returns a completion object shaped like the OpenAI SDK's: `choices=[SimpleNamespace(message=SimpleNamespace(content=answer, refusal=None), finish_reason="stop")]`, `usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0, prompt_tokens_details=None)` and `model="claude-opus-teacher"`. Check what `OpenAIGenericClient` and `usage.capture_llm_call` read from the response, and provide those attributes;
  - a dropped key raises `TeacherDropped`;
  - otherwise it calls `add_pending` and raises `TeacherNeeded`.
- `prompt_name_getter` returns `graph_extract.usage.CURRENT_PROMPT_NAME.get()`, which the dedup guard sets around every graphiti call, so pending requests carry the prompt type.

**Steps:**
- [ ] Tests:
  - the key is stable across dict key order and changes with any message byte;
  - a stored answer is returned with `finish_reason == "stop"`, and `OpenAIGenericClient.generate_response` parses it into the response model (build a real `OpenAIGenericClient` over the transport and call it);
  - a missing answer adds exactly one pending file even when called twice, and raises `TeacherNeeded` with the prompt name;
  - a dropped key raises `TeacherDropped`;
  - `put` after `add_pending` removes the pending file.
- [ ] Implement; tests, ruff; commit `feat(distill): teacher transport and answer store` (trailers).

### Task 3: Sample selection and the round runner

**Files:** create `scripts/distill/sample.py` and `scripts/distill/harness.py` (subcommands `select`, `round`, `status`); tests `tests/unit/test_distill_sample.py` and `tests/unit/test_distill_harness.py`.

**Interfaces:**
- Consumes: Task 1 (`build_cheap_graphiti(transport=, graph_driver=)`, `ReadOnlyNeo4jDriver`) and Task 2 (`AnswerStore`, `TeacherTransport`, `TeacherNeeded`, `TeacherDropped`).
- Produces:
  - `select_sample(driver, *, n_total=100, n_heldout=20, seed=20261017) -> list[SampleArticle]`, where `SampleArticle(article_id, vendor, product, source, split, stratum)`;
  - `harness round --sample sample.json --store DIR [--articles N]`;
  - `harness status --store DIR`.

**`select_sample`** uses read-only queries:
- strata are vendor × product family over articles that have episodes (`(:Article)-[:HAS_EPISODE]->()`), excluding removed and superseded episodes;
- the allocation is proportional with a floor of 5 per vendor;
- within each stratum, hard cases are oversampled 2:1:
  - an article in the 2026-10-17 affected set (the foreign-unnamed-subject predicate in `scripts/extraction_proof.py` `AFFECTED_ARTICLES_QUERY`; reuse it);
  - an article with a chunk whose text names no structural product;
  - an article whose title contains "Restore" or "Restoring";
  - an article with a markdown table chunk (`|---`).

Order is deterministic (`ORDER BY a.id` with a seeded shuffle per stratum). The held-out split is the last `n_heldout` by a seeded permutation, stratified. The sample, with its stratum counts, is written to `sample.json`.

**`round`**, for each sampled article in order, and each of its production episodes in chunk order:
- build the episode body from `Episodic.content` and the reference time from the article, the same way the ingest driver does (`ingest_driver._reference_time`);
- set `CURRENT_GROUNDING` with `GroundingContext(text=body, allowed=(product, vendor), product=product, stats=...)`;
- call `add_text_episode(cheap_graphiti, settings, previous_episode_uuids=<the uuids of this article's production episodes with a LOWER chunk_index, at most the last 10, oldest first>, name=..., body=body, source_description=article.source_url, reference_time=ref, instructions=EXTRACTION_INSTRUCTIONS + CHEAP_TIER_SALIENCE, context=document_context(vendor, product, source))`;
- outcomes: `TeacherNeeded` means a pending prompt, so record it and move to the next chunk; `TeacherDropped` marks the chunk "dropped"; `ReadOnlyViolation` (or the server read-only error) raised from graphiti's final save marks the chunk "complete"; any other exception is logged with the chunk id and marks it "error", never swallowed silently.

The harness builds its stack with production settings from `.env`, overriding:
- `llm_capture_path=<store>/capture.jsonl`;
- the cheap-tier model label `claude-opus-teacher`;
- `extraction_routing` forced to the cheap tier only.

It installs `install_dedup_guard(cheap_graphiti, fallback=None, ..., suppress_contradictions=True)`, as production does with retries off, and applies production's vector-search routing exactly as `build_cheap_graphiti`/the CLI do. It does NOT call `init_indices`.

**Why `previous_episode_uuids` is passed explicitly.** The live graph already holds this chunk's own production episode (same reference time). graphiti's default lookup would return it, putting the chunk text into its own prompt twice. Production at original ingest saw only the earlier sibling chunks (traced 2026-10-17). Test this: the prompt built for chunk k contains chunks 0..k-1 as previous episodes and never chunk k itself.

**Known caveat (accepted).** Dedup candidate searches run against the live graph, which already contains the entities and facts production extracted from this very chunk. That is the "updated article" situation, so the teacher may mark them as duplicates. Record the share of such self-matches in `status` (a candidate whose supporting episode is this chunk's production episode) so the pilot can judge it.

**`status`** prints counts per chunk state, pending prompts by prompt type, answered and dropped counts.

**Steps:**
- [ ] Tests:
  - `select_sample` with a fake driver returning a small fixed article table gives the same output twice, respects the per-vendor floor and the held-out size, and every held-out article is distinct from train;
  - `round` with a fake graphiti whose `add_episode` raises `TeacherNeeded` on the first call and `ReadOnlyViolation` after an answer is stored records one pending prompt in round 1 and "complete" in round 2;
  - the grounding ContextVar is set and reset around each episode;
  - an unexpected exception marks "error" and is logged.
- [ ] Implement; tests, ruff; commit `feat(distill): stratified sample and the round runner over the real graphiti pipeline` (trailers).

### Task 4: Answer intake, validation, briefs and reviewer sampling

**Files:** create `scripts/distill/validate.py`, `scripts/distill/TEACHER_BRIEF.md` and `scripts/distill/REVIEWER_BRIEF.md`; extend `scripts/distill/harness.py` (subcommands `batches`, `intake`, `review-sample`, `export`); tests `tests/unit/test_distill_validate.py` and `tests/unit/test_distill_harness.py`.

**Interfaces:**
- `harness batches --store DIR --size 10 --out DIR/batches/round-N/` writes `batch-XXX.jsonl` files. Each line is `{"key", "prompt_name", "messages", "response_schema"}`, grouped by prompt type. Re-asks carry `"previous_answer"` and `"rejection"`.
- A teacher writes `batch-XXX.answers.jsonl`, one line per key: `{"key", "answer": <JSON object>}`.
- `harness intake --store DIR --answers DIR/batches/round-N/` validates every answer:
  - accepted answers go to `store.put(key, json.dumps(answer), meta)`;
  - a rejected answer is re-queued with its reason if it has not been re-asked before, otherwise `store.drop`.
- `validate_answer(prompt_name, messages, answer, *, grounding_ctx) -> list[str]`: an empty list means valid; otherwise human-readable reasons. The checks:
  1. The answer parses with `scripts.build_extraction_dataset.RESPONSE_MODELS[prompt_name].model_validate`.
  2. `scripts.build_extraction_dataset.validate_capture(prompt_name, messages, obj, ...)` reports no defect (reuse; read its signature).
  3. For `extract_nodes.extract_text`: `graph_extract.grounding.filter_entities(answer, ctx)` returns an answer equal to the input (no drop, injection or rename). Otherwise the reasons name each changed entity. Rebuild the GroundingContext from the chunk text (the `<CURRENT MESSAGE>` block, extracted with the builder's `capture_episode_content`) and the article's product/vendor from the sample.
- `harness review-sample --store DIR --fraction 0.05 --seed 20261017 --out reviews/` writes about 5% of accepted answers, stratified by prompt type, with their prompts, for the reviewer subagent.
- `harness export --store DIR --out capture.jsonl` copies the capture records whose keys are accepted, dropping records for dropped keys. They are already in production capture format (Task 1 test), tier `cheap`.

**TEACHER_BRIEF.md** (verbatim core rules; the implementer may add formatting):
- You are producing training targets for a small extraction model. Each line of your batch file is one prompt that graphiti sends.
- Answer each prompt using ONLY the prompt's own text; no outside knowledge.
- Return for each key exactly one JSON object matching `response_schema`: no prose and no markdown fences.
- Follow every rule in the prompt: the naming rules, the documented-product rule (DOCUMENT CONTEXT), RestoresTo direction (the destination is the target), names as the text writes them, and the index ranges the prompt defines.
- Give the best answer the prompt allows. Extract every entity and fact the chunk states; invent none.
- If a line carries `rejection`, fix exactly what it names.
- Write your answers to `<batch>.answers.jsonl`, one `{"key", "answer"}` per line, and report only the counts.

**REVIEWER_BRIEF.md:** for each sampled answer, judge it against its prompt for:
- missed facts;
- invented facts or entities;
- wrong subject;
- wrong RestoresTo direction;
- a wrong duplicate or index choice.

Write one line per answer: `{"key", "verdict": "ok"|"issue", "issues": [...]}`.

**Steps:**
- [ ] Tests (`test_distill_validate.py`):
  - a valid `extract_text` answer passes;
  - an answer naming an entity absent from the chunk is rejected with that name in the reason;
  - an answer missing the documented product while facts would need it is rejected (gate injection);
  - "VBR" with documented product VBR is rejected (rename);
  - a `dedupe_nodes.nodes` answer with an out-of-range index is rejected;
  - a schema-invalid answer is rejected.
- [ ] Tests (`test_distill_harness.py`):
  - `batches` groups by prompt type with at most `--size` per file;
  - `intake` accepts and stores valid answers, re-queues a first rejection with its reason, and drops a second rejection;
  - `export` includes only accepted keys;
  - `review-sample` is deterministic.
- [ ] Implement; tests, ruff; commit `feat(distill): answer intake with deterministic validation; teacher and reviewer briefs` (trailers).

### Task 5: Dataset builder at production scale

**Files:** modify `scripts/build_extraction_dataset.py`; test `tests/unit/test_build_dataset_scope.py`.

`build_from_capture` attributes captured prompts to episodes through `load_graph`, which loads the whole graph with embeddings. That is infeasible at 240k edges (fetching vectors is bandwidth-bound, about 20 s per 1,000).

- Add `load_graph(..., article_ids: list[str] | None = None, with_embeddings: bool = True)`:
  - when `article_ids` is given, load only those articles' episodes, the entities they MENTION and the facts on those episodes;
  - when `with_embeddings=False`, skip `name_embedding` and `fact_embedding` (the capture path does not need them: check that `CaptureAttributor` and `build_from_capture` use no `SimIndex`; if they do, keep embeddings for the scoped set only).
- Add the CLI flag `--articles-from sample.json`, which scopes loading and implies `with_embeddings=False` for `--from-capture`.

**Steps:**
- [ ] Tests:
  - with a fake session, the scoped load issues queries parameterised by the article ids and returns only their episodes;
  - `with_embeddings=False` leaves embeddings empty;
  - `build_from_capture` on a tiny synthetic capture plus a scoped graph attributes records to the right episode (reuse the existing test patterns for the capture path, found with `grep -l build_from_capture tests`).
- [ ] Implement; tests, ruff; commit `feat(scripts): dataset builder loads a scoped graph without embeddings` (trailers).

### Task 6: Colab notebook, dataset packaging and runbook

**Files:** create `notebooks/distill_train.ipynb`, `scripts/distill/package_dataset.py` and `docs/superpowers/distill-runbook.md`; test `tests/unit/test_distill_package.py`.

- **`package_dataset.py --in data/ft-opus --out dist/ft-opus.zip [--hf-repo carev01/<name> --private]`** zips `train.jsonl`, `val.jsonl`, `sample.json` and a `MANIFEST.json`. The manifest records counts per prompt type, the loss-weight mix, the token estimates and the git commit. With `--hf-repo`, it pushes the files to a private HF dataset repo via `huggingface_hub`, using the token from the environment only and never printing it.
- **The notebook** (Unsloth, qwen3.5-4b, thinking off, `max_seq_length=20480`):
  - cells for mounting Drive or downloading the HF dataset;
  - loading train/val with `meta.loss_weight` applied as a per-example loss weight (follow the format the builder writes; read `assign_loss_weights`'s docstring);
  - a LoRA config from the previous run's defaults, documented in the runbook;
  - training with eval on val;
  - export of merged safetensors and a Q8 GGUF to a private HF model repo revision (`carev01/qwen35-4b-graphrag`, new revision `opus-distill-2026-10`).

  No secrets in the notebook: use `userdata.get` or env for HF_TOKEN.
- **The runbook** (`distill-runbook.md`):
  - the pilot and full-run commands: `harness select`, then the `round`/`batches`/teacher/`intake` loop, then `review-sample`, then `export`, then `build_extraction_dataset.py --from-capture <store>/capture.jsonl --articles-from sample.json`, then `package_dataset.py`;
  - the Colab steps;
  - evaluation: `scripts/eval_finetune.py` on val, and `scripts/extraction_proof.py` with `CHEAP_LLM_BASE_URL`/`CHEAP_LLM_MODEL` pointed at the student server, against solar-pro4 and the old fine-tune;
  - the pass criteria from the spec §8.

**Steps:**
- [ ] Test: on a tiny dataset directory, `package_dataset.py` writes the zip with the manifest counts. The HF push path is tested with a fake `HfApi`, so no network is used.
- [ ] Write the notebook (valid JSON; `nbformat` validation in the test if nbformat is available, otherwise a JSON parse check).
- [ ] Commit `feat(distill): Colab training notebook, dataset packaging, runbook` (trailers).

---

## Controller steps after the tasks (each paid or large step needs the user's approval)

1. **Pilot** (approved in the spec):
   - `harness select`, then restrict to 5 articles (stratified) with `--articles 5`;
   - run rounds until every chunk is complete or dropped, dispatching teacher subagents (Agent tool) on the batch files with TEACHER_BRIEF.md; then `intake` after each round;
   - run `review-sample` and one reviewer subagent;
   - report validity and re-ask rates per prompt type, reviewer findings, token sizes, and wall time and usage per batch.
2. **The full run** after the user approves the pilot numbers: a Workflow over rounds (stage-parallel batches, resume from the store).
3. **`export`, then the dataset build and packaging**; hand over the zip or HF dataset to the user for Colab.
4. **Evaluation** once the user returns the trained model (spec §8). Ask before any paid GPU.
