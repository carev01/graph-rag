# Concurrent Article Ingest — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Date:** 2026-09-14
**Status:** Approved design — ready for implementation planning
**Backlog:** the throughput lever identified by `pilot-reingest-2026-09-14.md` §4

---

## 1. The problem, measured

The 2026-09-14 pilot re-ingest (83 articles, 655 episodes, 6 h 55 m) produced two
numbers that together make this the next throughput lever:

**Sum of LLM in-call time was 25,211 s against 24,933 s wall — a ratio of 1.01.**
There is effectively no overlap across the run. Four phases dispatch strictly one at a
time and account for **55.9%** of all LLM time:

| phase | share | in-flight at dispatch |
|---|---|---|
| `extract_edges.edge` | 26.4% | always 1 |
| `extract_nodes.extract_summaries_batch` | 13.1% | 273 of 274 at 1 |
| `extract_nodes.extract_text` | 10.0% | always 1 |
| `dedupe_nodes.nodes` | 6.4% | always 1 |

These are per-episode calls, and `IngestDriver` processes episodes sequentially.

**The provider does not serialise us.** Latency by in-flight count at dispatch for
`dedupe_edges.resolve_edge`: 1231 / 1215 / 1287 / 1469 / **1082** ms across buckets 1,
2–4, 5–9, 10–19, 20+. Flat, with the most-concurrent bucket fastest. So the headroom is
real rather than assumed — this was the decisive question BACKLOG 31 was built to answer.

At the current rate the full corpus is roughly 17 months for one worker. That is the
thing concurrency has to move.

## 2. The hazard, and why the obvious mitigations fail

graphiti saves entities with `MERGE (n:Entity {uuid: $uuid})` where the uuid is generated
per extraction, and resolves duplicates by **semantic search against the graph as it
currently stands** (`resolve_extracted_nodes`: "semantic retrieval first, then
deterministic and LLM dedup"). So two episodes in flight that both extract an entity **not
yet present** each create it under a different uuid. Duplicate entity nodes fragment the
graph in a way duplicate facts do not.

**Measured: the risk does not decay.** Entities were created at a steady ~1.53 per episode
across the whole pilot run — the last decile of entities appeared 87% of the way through
(01:20 → 08:15 span, decile boundaries evenly distributed). So "run sequentially until the
entity backbone is dense, then go concurrent" has no point at which it triggers. This was
the first idea and the measurement killed it.

**Partitioning is ruled out by design invariant #4.** Scoping concurrency so entity spaces
cannot overlap — by vendor or product — would prevent "Amazon S3" from Veeam docs and AWS
docs resolving to the same node, which is precisely what the single corpus-wide `group_id`
exists to guarantee (`CLAUDE.md`, decision 4). Speed bought by fragmenting the entity graph
is not speed we want.

**Rejected alternatives**, with reasons:

- **Higher concurrency plus a post-hoc `dedupe_nodes_bulk` repair pass.** Available in
  graphiti, so not hypothetical. Rejected for now because the repair costs its own LLM
  calls and merging entities after the fact rewrites edges — a second destructive write
  path over the graph, added before we know whether there is anything to repair.
- **Serialising only the entity-resolution step behind a lock.** Preserves exact dedup, but
  that step is a large share of per-episode time, so it caps the achievable speedup —
  possibly to almost nothing. Worth revisiting only if measurement shows duplication is
  severe *and* the locked section is small.
- **Deferring concurrency entirely.** The 56% serial share is real and the new-entity rate
  was measured *not* to decay, so "revisit when the graph is denser" describes a state that
  does not arrive.

## 3. The decision

**Concurrency across articles, never within one, at a configurable limit, defaulting to
today's behaviour — then measured against an exact baseline before wider use.**

Episodes within an article are consecutive chunks of one document and share entities most
heavily, so they remain sequential. Articles run N-way concurrent.

This does not eliminate the duplicate risk; it reduces the highest-overlap case and makes
the remainder measurable. We have an unusually good instrument for that: the pilot produced
**999 entities over 655 episodes from a known set of 83 articles, sequentially**. Re-running
the identical set concurrently isolates duplication as the single variable.

## 4. Architecture

One shared helper so the two call sites cannot drift:

```
src/graph_extract/concurrent_ingest.py
    async def run_concurrently(items, worker, *, limit) -> list[object]
```

Bounded by an `asyncio.Semaphore`. Returns one result per item **in input order**, with a
failing item's exception returned in its slot rather than raised — one bad article must not
cancel its siblings.

### 4.1 `IngestDriver.ingest_source`

Fans out over article ids. Each task runs the existing `ingest_article` unchanged, so
episodes stay sequential within an article.

### 4.2 `run_worker_once`

**Groups claimed jobs by `article_id` first, then fans out over the groups.** Jobs within a
group run in their existing order.

This is load-bearing, not tidiness: `claim_semantic_jobs` has no `DISTINCT` on
`article_id` (`state_store.py:145-147`), so one batch can legitimately contain an upsert and
a later remove for the same article. Fanning out over raw jobs could apply them out of
order and tombstone episodes that the upsert then recreates, or the reverse.

### 4.3 What the existing instrumentation does under concurrency

`CURRENT_DEDUP_STATS` is a `ContextVar` and `asyncio.create_task` copies the context at
creation, so each article task gets its own scope and per-article attribution stays correct.
This was the reason for choosing a ContextVar over a driver attribute when `dedup_guard` was
written; it pays off here without change.

`PromptTimings` is a single shared object mutated from multiple tasks. asyncio is
single-threaded and its updates contain no awaits, so they are atomic in practice. Its
in-flight counter will now measure the real dispatch shape rather than an artificially
serial one — which is exactly what we want to read afterwards.

## 5. Configuration

```python
ingest_article_concurrency: int = 1
```

Default **1** — byte-for-byte today's behaviour. Concurrency is opt-in until the §7
measurement justifies raising it, so this is mergeable before anything is spent.

Values below 1 are rejected at config validation rather than silently coerced.

## 6. Error handling

| case | behaviour |
|---|---|
| One article raises | Its exception is returned in its result slot; siblings continue. `ingest_source` re-raises after the batch if any failed. **This is a deliberate behaviour change, not a preservation:** today a raising article propagates immediately out of the `for` loop (`ingest_driver.py`, `ingest_source`), so every article after it is silently never attempted. Under concurrency, articles already dispatched complete and the failure still surfaces at the end. Better — a mid-run failure no longer abandons the rest of the batch — but it IS different, and a test pins the new contract. |
| One worker job raises | Unchanged: that job fails and retries via the existing backoff. Other groups are unaffected. |
| `limit = 1` | Exactly the sequential path, including ordering. |
| A large batch | The semaphore bounds in-flight work, so connection and memory use stay bounded regardless of batch size. |

No retry logic changes. Nothing is swallowed: a failure that used to surface still surfaces.

## 7. Testing

**Hermetic:**
- Ordering within a group is preserved.
- Groups genuinely overlap (a slow first item does not block a later one).
- A raising item does not cancel its siblings, and its exception is returned in place.
- `limit=1` produces the same call order as the sequential path.
- `run_worker_once` fans out over **article groups**, not raw jobs: a batch with two jobs
  for one article applies them in order.
- Results are returned in input order regardless of completion order.

**Discrimination:** every test must fail with its fix neutralised, proven by mutation.

**Paid validation (§8), before concurrency is used for anything wider.**

## 8. Validation — the A/B that gates adoption

Re-ingest **the same 83 articles** at `ingest_article_concurrency = 4` and compare against
the sequential baseline already recorded:

| | sequential (measured) | concurrent (to measure) |
|---|---|---|
| entities | **999** | ? |
| episodes | 655 | 655 expected |
| wall clock | 6 h 55 m | ? |

Identical input, identical code, one variable. **Any excess over 999 entities is
duplication.** Cost roughly $10 and ~1.7 h at N=4.

Report the duplicate rate and the speedup together; a large speedup does not justify an
unbounded duplicate rate, and the decision to raise or lower N is the user's.

Note the graph must be reset to the same starting point first, as it was for the sequential
run — otherwise the second run dedups against the first's entities and measures nothing.

## 9. Success criteria

1. `limit=1` is indistinguishable from today in call order and results. The one intended
   difference is failure handling (§6): even at `limit=1` a raising article no longer
   abandons the articles after it. That is the change, stated rather than smuggled.
2. A batch containing two jobs for one article applies them in order.
3. One failing article does not prevent its siblings completing.
4. The A/B reports both entity count against 999 and wall clock against 6 h 55 m.
5. Per-article dedup attribution remains correct under concurrency, visible in the worker's
   batch log.

## 10. Out of scope

- **Concurrency within an article.** Highest entity overlap, and the per-article grouping is
  what makes the worker path safe.
- **A post-hoc `dedupe_nodes_bulk` repair pass.** Deliberately deferred until §8 says
  whether there is anything to repair.
- **Re-measuring the DB saturation ceiling.** The earlier ~1.9x figure was measured on the
  brute-force scan shape that ingest no longer runs. It needs re-taking, but it bounds how
  far N can usefully rise rather than whether this design is right.
- **The answer-path corpus scan.** Unrelated to ingest concurrency and still open.
