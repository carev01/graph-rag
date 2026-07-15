# Incremental Semantic Ingestion — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Pipeline hardening — Sub-slice A (incremental ingestion core). Finishes the Phase-1 exit criterion "incremental deltas flow into the graph within minutes."
**Date:** 2026-07-15
**Status:** Approved design — ready for implementation planning

Closes the deferred slice-2b pipeline-hardening items that are on the critical path: the temporal append/invalidate update policy, `provenance.link` atomicity + re-keying, and delta-driven incremental semantic ingestion wired to the slice-1 sync via a durable queue + standalone worker. Robustness/economics (dead-letter, retry/backoff, token budget + priority lanes) are **Sub-slice B**; maintenance jobs (staleness sweep, structural↔semantic `SAME_AS`) are **Sub-slice C**.

---

## 1. Problem & scope

Today the two halves are unwired: `graph_sync` (slice 1) drives the DocExtractor delta feed into the **structural** layer (`apply_structural`, `tombstone_article`, `content_hash` gating) but never touches the semantic layer; `graph_extract` (slice 2a/2b) does semantic (Graphiti) ingestion but only when invoked manually (`run_pilot`/CLI). So the structural graph tracks the live corpus while the semantic graph is a frozen pilot snapshot. And `ingest_article` only handles *first* extraction: on a changed article it adds new episodes without superseding the old ones, and `provenance.link` creates a duplicate `HAS_EPISODE` edge per chunk.

**In scope (A):** durable job queue (Postgres), sync→queue trigger, standalone worker, temporal update policy (`updated`/`removed`), `provenance.link` atomicity + re-keying.
**Out of scope:** rate/concurrency caps, token budget, priority lanes, retry/backoff, dead-letter (B); staleness sweep, `SAME_AS` reconciliation (C).

## 2. Architecture & data flow

```
DocExtractor delta ─▶ graph_sync.sync_core._apply_record
                        ├─ apply STRUCTURAL (unchanged)
                        └─ enqueue_semantic_job(...)         [Postgres semantic_jobs]
                                                                     │
        graph_sync semantic worker (standalone) ◀── claim (SKIP LOCKED)
                        ├─ op='upsert' ─▶ ingest_driver.ingest_article  (append + supersede + re-key prov)
                        └─ op='remove' ─▶ ingest_driver.tombstone_article_episodes  (mark removed)
```

The worker lives in `graph_sync` (which owns the queue/state store and "drives graph writes") and imports `graph_extract.ingest_driver`. This introduces a **`graph_sync → graph_extract`** dependency, consistent with the plan's A→B data flow. `graph_extract` must NOT import `graph_sync` (keep the layering one-directional). At-least-once throughout; the per-chunk `content_hash` gate makes re-processing idempotent.

## 3. Job queue — `semantic_jobs` (Postgres)

Lives in the graph_sync state store (Postgres, alongside the sync cursor). Schema:

| column | type | notes |
|---|---|---|
| `id` | bigserial PK | |
| `article_id` | text not null | |
| `op` | text not null | `'upsert'` \| `'remove'` |
| `content_hash` | text null | the hash that triggered upsert; null for remove |
| `status` | text not null default `'pending'` | `pending` \| `in_progress` \| `done` \| `failed` |
| `attempts` | int not null default 0 | |
| `last_error` | text null | |
| `enqueued_at` / `claimed_at` / `updated_at` | timestamptz | |

- **Idempotent enqueue:** partial unique index `ux_semantic_jobs_pending ON semantic_jobs(article_id) WHERE status='pending'`. `INSERT … ON CONFLICT (article_id) WHERE status='pending' DO UPDATE SET op=excluded.op, content_hash=excluded.content_hash, enqueued_at=now(), updated_at=now()`. A burst of changes to one article collapses to one pending job (latest wins). An already `in_progress` job does not block a new `pending` row (a change that lands mid-processing re-queues; the worker's hash-gate makes the double-run harmless).
- **Claim:** `UPDATE semantic_jobs SET status='in_progress', claimed_at=now(), updated_at=now() WHERE id IN (SELECT id FROM semantic_jobs WHERE status='pending' ORDER BY enqueued_at FOR UPDATE SKIP LOCKED LIMIT $batch) RETURNING *`. Concurrency-safe for multiple workers.
- **Complete:** `status='done'` on success; on exception `status='failed', attempts=attempts+1, last_error=$e`.

New `state_store` methods: `enqueue_semantic_job(article_id, op, content_hash)`, `claim_semantic_jobs(batch) -> list[Job]`, `complete_semantic_job(id)`, `fail_semantic_job(id, error)`. The table + partial index are added to the existing `CREATE TABLE IF NOT EXISTS …` DDL block in `state_store.py` (the slice-1 pattern; `StateStore` is asyncpg-backed). Note: the pre-existing `dead_letter` table is the sync's malformed-line dead-letter — unrelated to the semantic queue; semantic dead-lettering is Sub-slice B.

## 4. Trigger — `sync_core._apply_record`

`_apply_record` currently: tombstone → `tombstone_article`; else hash-gate (`existing == content_hash` → skip); else `apply_structural`/`apply_incomplete_article`.

Add enqueue calls:
- On a content change that will be applied (hash new or changed): `enqueue_semantic_job(rec.id, 'upsert', rec.content_hash)`.
- On a tombstone: `enqueue_semantic_job(rec.id, 'remove', None)`.

**At-least-once ordering:** enqueue the job *before* the `content_hash` is committed to the structural store, so a mid-batch failure (the sync advances its cursor only on the terminal `{"control":"cursor"}` line) re-applies the record and re-enqueues (idempotent). The pending-unique upsert means duplicate enqueues collapse. Consequence to verify in tests: an unchanged-hash replay does NOT enqueue (no needless re-ingest), but a changed hash always leaves a pending job.

Note: the structural writes go to Neo4j and the queue to Postgres — they cannot share one transaction. The ordering above (enqueue as part of the record's apply path, cursor-on-terminal retry, idempotent enqueue) is the at-least-once guarantee; a residual "structural applied, enqueue lost, hash now matches so no re-trigger" gap is closed by Sub-slice C's reconciliation and is explicitly out of scope here.

## 5. Standalone worker — `graph_sync` entrypoint

A new CLI entrypoint (`graph_sync … worker` or a `semantic_worker` module). Loop:
1. `claim_semantic_jobs(batch)`.
2. For each job: `op='upsert'` → `ingest_driver.ingest_article(article_id)`; `op='remove'` → `ingest_driver.tombstone_article_episodes(article_id)`.
3. Success → `complete_semantic_job(id)`. Exception → `fail_semantic_job(id, str(e))`, log, continue (SKIP LOCKED means a poison job never blocks the queue).
4. Empty claim → sleep a short poll interval; loop.

Builds one `IngestDriver` (Graphiti + Neo4j + DocExtractor client) reused across jobs; closes cleanly on shutdown (SIGINT). **No rate/concurrency cap, no retry/backoff, no dead-letter** — those are Sub-slice B; a `failed` job simply waits (a human/B re-queues). The worker is a separate process from the sync service.

## 6. Temporal update policy — `ingest_driver` (design-decision #3)

**`ingest_article` (upsert), extended:** per-chunk `already_ingested(article_id, chunk_index, content_hash)` still skips unchanged chunks. For a changed/new chunk:
1. `add_text_episode` → new `Episodic`.
2. `provenance.link(...)` — re-keyed (see §7): detaches the chunk's prior `HAS_EPISODE` edge and marks that **prior episode `superseded:true`**, then links the new episode.
3. Graphiti's `add_episode` invalidates contradicted facts (`invalid_at`) across the group automatically — no manual fact surgery.

Net: new episodes appended; superseded episodes kept (not deleted); contradicted facts invalidated by Graphiti. History is preserved (the whole point of the temporal graph).

**`tombstone_article_episodes` (remove), new:** Cypher-only — `MATCH (a:Article {id})-[:HAS_EPISODE]->(e:Episodic) SET e.removed=true`. Never call `remove_episode`. Facts whose only supporting episodes are all removed/superseded are expired by the weekly sweep (Sub-slice C) — A only marks. Handles mass tombstone batches (thousands, `run_id: null`) gracefully — it's a set-based property write.

## 7. Provenance atomicity + re-keying — `provenance.link`

Current bug: `MERGE (a)-[r:HAS_EPISODE {chunk_index:$i}]->(e)` binds `e` to the new episode uuid, so re-linking a chunk to a *different* episode creates a second edge with the same `chunk_index`.

Fix — one write transaction:
```cypher
MATCH (a:Article {id:$a})
OPTIONAL MATCH (a)-[old:HAS_EPISODE {chunk_index:$i}]->(oldE:Episodic)
  WHERE oldE.uuid <> $u
SET oldE.superseded = true
DELETE old
WITH a
MATCH (e:Episodic {uuid:$u})
MERGE (a)-[r:HAS_EPISODE {chunk_index:$i}]->(e)
SET r.heading_path=$hp, r.content_hash=$h
```
Guarantees exactly one `HAS_EPISODE` per `(article_id, chunk_index)`, marks the superseded episode, and is idempotent (re-running with the same episode uuid is a no-op on the MERGE and matches no `oldE`). `add_text_episode` + `link` should be effectively atomic per chunk: link immediately after the episode is created so a failure leaves no orphan un-linked episode on the next (idempotent) re-run.

## 8. Testing

Postgres + Neo4j via testcontainers (both already used in slice-1/2 tests):
- **Queue:** enqueue idempotency (two enqueues for one article → one pending row, latest op/hash); `in_progress` doesn't block a new `pending`; SKIP-LOCKED claim under two concurrent claimers returns disjoint sets; complete/fail transitions.
- **Trigger:** an `updated` record enqueues `upsert`; a tombstone enqueues `remove`; an unchanged-hash replay enqueues nothing; a mid-batch failure path re-enqueues (idempotent).
- **Temporal update:** re-ingesting a changed article marks the old chunk-episode `superseded=true`, adds the new one, and leaves exactly **one** `HAS_EPISODE` per chunk; a `remove` marks all the article's episodes `removed=true` without deleting them.
- **Provenance re-key:** linking chunk N to a new episode detaches the old edge (no duplicate `HAS_EPISODE`), marks the old episode superseded, and is idempotent on re-run.
- **End-to-end:** feed a delta stream containing an `updated` article through `sync_core`, run the worker once, assert the semantic graph reflects the update (new episode linked, old superseded).

## 9. Acceptance criteria

1. A created/changed article flowing through `sync_core` leaves a `pending` `semantic_jobs` row (idempotent; unchanged replays don't).
2. The standalone worker drains the queue: `upsert` re-ingests with append+supersede, `remove` marks episodes `removed:true`; failures mark `failed` without blocking the queue.
3. Re-ingesting a changed article yields exactly one `HAS_EPISODE` per chunk, the prior episode marked `superseded`, and Graphiti-invalidated contradicted facts — no deletion of history.
4. `graph_extract` does not import `graph_sync`; the worker (in `graph_sync`) imports `graph_extract`.
5. Unit + integration tests green (Postgres + Neo4j testcontainers); ruff/mypy clean.

## 10. Deferred (explicitly)

- **Sub-slice B:** rate/concurrency cap, token-budget metering, priority lanes (incremental preempts bootstrap), retry/backoff, dead-letter.
- **Sub-slice C:** weekly residual-staleness sweep (Cypher fact expiry for episodes all superseded/removed), structural↔semantic `SAME_AS` reconciliation.
- Backfill/bootstrap of the full corpus through the queue (a bootstrap enqueues upserts for all articles) rides on B's budget/lanes.
