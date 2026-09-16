# Vector-index crossover measurement — design

**Date:** 2026-09-16
**Status:** Design approved; not implemented.
**Scope:** The free, measurement-only first step of Phase B
(`production-readiness-review-2026-09-13.md` §3). It writes no production code and
spends no money.

---

## 1. Why this exists

Phase B proposes replacing graphiti's brute-force cosine scan with an index-backed
`SearchInterface`. The review calls it the fastest route out of its stated biggest risk.
Before building it, two things need to be true that currently are not: we need to know
what the index actually buys at corpus scale, and we need to know what it costs on write.

Neither is knowable from the graph we have. The live `backup-docs` graph holds 3,469
facts; the corpus projects to ~5.3M. Every claim about the index at scale today is a
linear extrapolation three orders of magnitude past the measured range — the
same reasoning that has been refuted repeatedly in this project.

## 2. What is already measured, and two corrections

`dedup-cost-profile-2026-09-11.md` §2, at 3,096 facts:

| query | median |
|---|---|
| brute force (what graphiti runs) | 255 ms |
| index-backed | 144 ms |

`production-readiness-review-2026-09-13.md` §1.2, on the live graph with the
`lean_edge_search` projection in place: 211 ms @ 1,000 facts, 265 @ 2,000, 343 @ 3,469 —
slope **0.053 ms/fact**, intercept ~160 ms. The entity scan behaves the same way:
0.050 ms/entity, ~159 ms intercept (`node_operations.py:439-445`, an unfiltered cosine
over every `:Entity` in the group — a second scan not previously in the backlog).

**Correction 1 — there is no crossover to find.** The index already wins at the smallest
scale measured. What is unknown is whether its curve stays flat while brute force climbs
linearly. This measurement needs *range*, not size.

**Correction 2 — the recall risk is overstated in BACKLOG 8-scale.** It records *"At k=1
the index returned the rank-2 fact."* Its own source retracts that in the next paragraph:
at k=3 the index returned all three in the right order, with top-3 uuids and scores
matching brute force exactly, and the k=1 miss is called *"an artifact of asking the index
for a single neighbour."* graphiti asks for `limit=10`. Recall at 1M facts remains
genuinely unknown, because HNSW recall degrades with N — that is a narrower worry than
the backlog implies, and §8 below addresses it as a trend.

**Correction 3 — `SEARCH` does not exist on this server.** BACKLOG 8-scale instructs that
*"any override must target `SEARCH`"*, because 2026.07.1 emits
`db.index.vector.queryRelationships is deprecated. It is replaced by SEARCH.` Probed
2026-09-16 on `neo4j:2026.07.1-community`: `SEARCH` is **not valid Cypher** in any tested
form, under either `CYPHER 5` or `CYPHER 25`; the parser's accepted-keyword list does not
contain it, and `SHOW PROCEDURES` offers only `db.index.vector.queryNodes` and
`db.index.vector.queryRelationships`. The server deprecates a procedure in favour of a
clause it does not ship. **The implementation must use the deprecated procedures**, which
work correctly under both language versions; migrating to `SEARCH` is a future concern
gated on a Neo4j version that has it.

## 3. Scope

**In scope**

1. Latency of brute force vs index, for edges and nodes, swept over N.
2. Whether the index's advantage grows as the linear/logarithmic model predicts.
3. Write-path overhead of maintaining the index (review item B4).
4. Recall@10 against brute force as a **function of N** (trend, not an absolute).

**Out of scope, deliberately**

- Any change to `src/`. No `SearchInterface` implementation — that is the next slice, and
  it should be designed against these numbers rather than in parallel with them.
- Absolute recall for real embeddings at corpus scale. Synthetic vectors cannot settle it;
  that is gated on a larger real graph and the infrastructure work preceding it.
- End-to-end ingest or retrieval timing. This measures one query, not a pipeline.

## 4. Where it runs

A **`neo4j:2026.07.1-community` testcontainer** — the exact production version, already
used by `tests/integration/conftest.py` and already cached on this host.

Not the live instance, for two reasons. Neo4j Community supports a single user database,
so a synthetic graph could only be isolated by `group_id`, while a vector index on
`:RELATES_TO(fact_embedding)` is instance-wide for that relationship type — it would add
write overhead to the reference graph's own insert path. That is precisely why the
2026-09-11 experimental index was dropped after measuring. Second, the reference graph is
read-only by standing rule.

Docker is local (`DOCKER_HOST` unset, root `/var/lib/docker` on `srv-openclaw`), with
**31 GB free plus 6.5 GB reclaimable**. The Neo4j host `alpcirag01` is a separate machine
with no Docker access from here; if a sweep beyond 1M is ever wanted, that is where it
would have to run.

## 5. Fixture construction

Per step N, insert into an isolated `group_id`:

- `(:Entity)-[:RELATES_TO {uuid, group_id, fact_embedding}]->(:Entity)` — N edges over a
  node pool of `max(1000, N/50)` entities, matching the corpus shape where facts greatly
  outnumber entities.
- `(:Entity {uuid, group_id, name_embedding})` — swept separately and to a lower ceiling,
  since entities dedup sublinearly.

**Vectors are resampled from the real 3,469 `fact_embedding` values** (read-only, one
query) with Gaussian noise added and renormalised, rather than drawn i.i.d. HNSW traversal
cost and recall both depend on cluster structure; i.i.d. vectors in 768 dimensions are
near-equidistant, which is the worst case for the index and would make it look worse than
reality. If the live graph is unreachable the harness falls back to i.i.d. vectors and
**says so loudly in its output**, because the two are not interchangeable.

Batched inserts with `UNWIND`, committed per batch, so a 1M-edge build is resumable and
observable.

## 6. What is measured

Three queries per N, median of ≥5 runs after a warm-up run, each with a fresh random query
vector drawn from the same distribution as the data:

**A — graphiti's actual brute-force shape** (from `profile-vector-index.py`, which matches
`graph_queries.py`):

```cypher
MATCH (n:Entity)-[e:RELATES_TO {group_id:$g}]->(m:Entity)
WITH DISTINCT e, n, m, vector.similarity.cosine(e.fact_embedding, $v) AS score
WHERE score > $min
RETURN e.uuid AS uuid, score ORDER BY score DESC LIMIT $k
```

**B — index-backed**, using the working (deprecated) procedure:

```cypher
CALL db.index.vector.queryRelationships($idx, $k, $v) YIELD relationship AS e, score
WHERE e.group_id = $g AND score > $min
RETURN e.uuid AS uuid, score ORDER BY score DESC
```

**C — control: A again, with the index present.** Neo4j consults a vector index only
through the procedures, so A must not change. This reproduces the 2026-09-11 "0%" finding
at every N, and is the guard against reporting an improvement that came from cache warmth
rather than the index.

The node sweep uses the `:Entity`/`name_embedding` equivalents and
`db.index.vector.queryNodes`.

Index DDL (confirmed working on 2026.07.1, relationship indexes included):

```cypher
CREATE VECTOR INDEX fact_embedding_vec_idx IF NOT EXISTS
FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}
```

Every measurement calls `db.awaitIndexes` before timing, so no run reports a partially
built index.

## 7. The sweep

Edges: **10k, 50k, 100k, 250k, 500k, 1M**. At 768 float64 per vector that is ~6 GB of
property store at the top step, comfortably inside the available 31 GB with the HNSW
index and transaction logs on top.

Nodes: **10k, 50k, 100k, 250k**.

The harness checks free disk before each step and stops cleanly with a partial curve
rather than filling the root filesystem. **Measured points and any projection beyond them
are reported separately and labelled**, following §1.2's own discipline of marking its
extrapolated rows `[inference]`.

## 8. Recall methodology

At each N, for 20 query vectors: take brute force's top-10 uuids as ground truth and
report `|index_top10 ∩ brute_top10| / 10`, plus rank-1 agreement.

This is a **trend measurement**. The absolute number belongs to resampled synthetic
vectors, not to real embeddings at corpus scale. What it can establish, and what nothing
else available can, is whether recall *degrades as N grows* — the open question that
matters for whether an ANN-backed dedup path is safe later. A flat curve is evidence; a
declining one is a reason to design the exact bounded candidate set instead (review D2).

## 9. Write overhead (B4)

For each N step, time the batched insert twice: once into a fixture with no vector index,
once with the index already created. The difference is the per-edge index maintenance
cost, expressed as ms per 1,000 inserts and as a percentage of the un-indexed insert.

This is the number that caused the 2026-09-11 index to be dropped — *"keeping it would add
write overhead to every fact insert on the path we are trying to speed up"* — and it has
never been measured.

## 10. Deliverables

- `scripts/vector_crossover.py` — the harness, self-contained, committed. Runs offline
  against a testcontainer, takes `--max-edges`, `--max-nodes`, `--steps`, `--seeds`.
- `docs/superpowers/vector-crossover-2026-09-16.md` — results: the two curves, the recall
  trend, the write overhead, the projection to corpus scale with measured and inferred
  rows distinguished, and a recommendation on whether Phase B's implementation half is
  justified.
- No changes under `src/`.

## 11. Testing

The harness is measurement code, so its correctness risk is "reports a number that is not
what it claims". Three unit tests against a small fixture, run in CI:

1. **The control holds:** query A returns identical uuids with and without the index.
   If it ever differs, the measurement is not isolating what it claims.
2. **Ground truth is ground truth:** brute-force top-10 on a fixture with known planted
   nearest neighbours returns them in the right order.
3. **The fallback is loud:** with the live graph unreachable, the harness still runs and
   its output records that vectors were i.i.d. rather than resampled.

Timing itself is not asserted — a test that pins a latency is a flaky test.

## 12. What would invalidate this

- **Neo4j's HNSW parameters are defaults.** `M` and `ef_construction` are not tuned, and
  recall/latency both depend on them. The result therefore describes the default index,
  which is what an implementation would ship first.
- **Resampled vectors are not real embeddings.** They preserve cluster structure
  approximately; they do not reproduce the true manifold. Latency conclusions are robust
  to this, recall conclusions are directional only (§8).
- **A container on a shared host is noisy.** Medians over ≥5 runs with a discarded warm-up,
  and the A/C control catching cache effects, are the mitigations.
- **One machine, one Neo4j configuration.** Page-cache size materially affects a scan; the
  container's default heap and page cache will be recorded in the output so the numbers
  can be reproduced or dismissed.

## 13. Open questions for the user

1. The sweep stops at 1M edges because of local disk. If the curves are still ambiguous
   there, extending to 5M needs Docker on the Neo4j host — worth arranging, or accept the
   projection?
2. Recall is measured as a trend only. If it declines with N, does that settle D2 toward
   the exact bounded candidate set, or would you still want ANN with a measured floor?
