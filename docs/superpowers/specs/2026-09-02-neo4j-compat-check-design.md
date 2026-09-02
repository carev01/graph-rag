# Neo4j / Graphiti Compatibility Check — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** A repeatable compatibility harness that proves `graphiti-core==0.29.2`
and our own retrieval/ingestion Cypher work against a target Neo4j **before** we
commit to a bootstrap ingestion there.
**Date:** 2026-09-02
**Status:** Approved design — ready for implementation planning

The stack was built and tested against Neo4j **5.22** (testcontainers) and **5.26**
(compose). The production target is **Neo4j 2026.07.1 Community** at
`alpcirag01.ai.area51.pp.ua`, which defaults to `db.query.default_language=CYPHER_25`.
GDS is already verified there (see §2). This slice closes the other half of the
version gate.

---

## 1. Scope

**In:** a `src/compat/` package (`checks.py`, `runner.py`, `report.py`, `cli.py`);
`compat_neo4j_*` settings; a `@live` pytest wrapper; a testcontainer-backed
integration test; unit tests for the pure logic; the resulting
`docs/superpowers/neo4j-compat-report.md`; the testcontainer image bump and the
Cypher-25 modernisation of our two bare `CALL {}` sites (§8).

**Out (deferred):** patching graphiti-core internals; fixing graphiti's
brute-force vector scan (§9); migrating any data to the new instance; the
bootstrap ingestion itself; a CI gate on the harness (it is run on demand,
against a named target).

## 2. Grounding (verified live, 2026-09-02)

Against `bolt://alpcirag01.ai.area51.pp.ua:7687`:

| Fact | Value |
|---|---|
| Kernel | **2026.07.1**, **community** edition |
| Cypher versions offered | `5`, `25` — **default `CYPHER_25`** |
| Databases | `neo4j`, `system` only (Community ⇒ no scratch database) |
| Graph state | 0 nodes, 0 constraints, 2 lookup indexes |
| GDS | **2026.07.0** — our `gds.graph.project` aggregation form and our seeded `gds.leiden.stream` options both verified working, unmodified |
| APOC | **not installed** (`apoc.create.node` → `ProcedureNotFound`) |

Probed and **passing** under `CYPHER_25`: `CREATE VECTOR INDEX` (node and
relationship), `CREATE FULLTEXT INDEX`, `CREATE INDEX`, bare `CALL { WITH n … }`,
bare `CALL { … UNION ALL … }`, `vector.similarity.cosine`,
`db.index.vector.queryNodes`, `db.index.fulltext.queryNodes`, dynamic label
`CREATE (n:$(lbl))`, and a bi-temporal edge write with `datetime()` plus a 768-d
relationship embedding.

Code inventory (`graphiti_core` 0.29.2 and our `src/`):

- `build_indices_and_constraints` (`driver/neo4j/operations/graph_ops.py:55-67`)
  issues **only** range/lookup and fulltext DDL (`graph_queries.py:54-140`) in
  modern `IF NOT EXISTS` syntax. It creates **no vector indexes** and never calls
  the removed `db.index.vector.create*` procedures.
- Graphiti scores similarity by **brute-force Cypher**:
  `vector.similarity.cosine(n.name_embedding, $search_vector)` with
  `WHERE score > $min_score` (`search_ops.py:148-160`; nine further sites in
  `search_utils.py`). No ANN index is consulted.
- Every graphiti subquery uses the Cypher-25-safe scoped form `CALL (n) { … }`.
- Graphiti's **bulk** node save uses native dynamic-label syntax
  `SET n:$(node.labels)` (`models/nodes/node_db_queries.py:260`) — this is the
  exact construct Neo4j 5.22 cannot parse and 5.26+ can. Its **single**-node save
  path interpolates labels as literal query text instead, and is version-agnostic.
- Neither graphiti-core nor our `src/` references `apoc.` anywhere, so the
  target's missing APOC is informational, not blocking.
- The only bare `CALL {}` subqueries in the repo are **ours**:
  `src/graph_extract/staleness_sweep.py:52` (importing `WITH eps`) and
  `src/theme_builder/cli.py:69` (non-importing, `UNION ALL`). Both pass on
  2026.07 today.

## 3. Architecture

A new top-level package `src/compat/`. It depends on `graph_extract`,
`answer_api`, and `theme_builder`; nothing depends on it. It is a diagnostic, not
a runtime service.

```
compat/
  checks.py   — the check registry (what to verify)
  runner.py   — execution, Cypher-5 retry, isolation, teardown
  report.py   — matrix + verdict rendering (pure)
  cli.py      — python -m compat.cli
```

Data flow: `cli` builds the target driver + graphiti client → `runner.run_all`
executes every check in registry order, accumulating `CheckResult`s → `report`
renders markdown → `cli` writes it and echoes the verdict. Teardown runs in a
`finally` regardless of outcome.

## 4. Targeting and isolation

Three optional settings on `ExtractSettings`, each defaulting to `""` and falling
back to the corresponding `neo4j_*` value when empty:

```python
compat_neo4j_uri: str = ""
compat_neo4j_user: str = ""
compat_neo4j_password: str = ""
```

A helper `compat.runner.compat_target(settings) -> tuple[str, str, str]` resolves
the triple. This lets the harness be pointed at a new instance without editing
the `neo4j_*` values the rest of the stack uses.

**Isolation contract** — the harness must be safe to run against a *populated*
instance:

- Every node, episode, entity, and fact the harness writes carries
  `group_id="compat-check"` (a module constant, not configurable). Teardown is
  `MATCH (n {group_id:'compat-check'}) DETACH DELETE n` plus an equivalent
  relationship sweep.
- Harness-private indexes are named with a `compat_` prefix and dropped in
  teardown.
- **Graphiti's own indexes are deliberately not torn down.**
  `build_indices_and_constraints()` is idempotent and creates exactly the schema a
  real bootstrap needs; leaving them is correct on a target instance and a no-op
  on an already-bootstrapped one. The report states this side effect explicitly.
- Because Community edition offers no scratch database, namespacing is the only
  isolation mechanism available. The harness never issues an unscoped `DELETE`.

## 5. Check model

```python
Status = Literal["pass", "fail", "skip"]

@dataclass(frozen=True)
class CheckResult:
    name: str
    group: str
    status: Status
    detail: str = ""                    # observed value, or the error message
    cypher5_retry: Status | None = None  # set only when status == "fail"
    informational: bool = False          # excluded from the verdict
```

Two check kinds:

- **`CypherCheck(name, group, cypher, params, expect)`** — a declarative
  statement. Because the runner holds the statement text, it can re-run a failure
  as `"CYPHER 5 " + cypher` and record `cypher5_retry`. `expect` is an optional
  predicate over the returned rows; when absent, "did not raise" is the pass
  condition.
- **`CallableCheck(name, group, fn)`** — an arbitrary `async fn(ctx) -> str`
  returning a short detail string, for calls into graphiti or our own modules.
  A language-version retry is not possible for these; `cypher5_retry` stays
  `None` and the report prints `n/a (procedural)`.

`ctx` is a small frozen dataclass carrying `driver`, `graphiti`, `settings`, and
the fabricated `embedding` vector, so checks never build their own clients.

## 6. The check registry (eight groups)

**1. `server` — environment facts.** Kernel version + edition
(`dbms.components()`), `db.query.default_language`, GDS version (skip if the
procedure is absent), APOC presence (**informational**), and availability of
`db.index.fulltext.queryNodes` / `queryRelationships` and
`vector.similarity.cosine`.

**2. `bootstrap` — schema.** `graphiti.build_indices_and_constraints()` completes
without raising; then `SHOW INDEXES` confirms the four fulltext indexes
graphiti declares (`episode_content`, `node_name_and_summary`, `community_name`,
`edge_name_and_fact`) and its range indexes exist. Also runs
`graph_sync.neo4j_repo.init_schema()` — our structural constraints/indexes.

**3. `vector` — similarity.** Narrow, because nothing in the stack uses a vector
index: `vector.similarity.cosine` over a 768-d **node** property and over a 768-d
**relationship** property (graphiti scores fact embeddings on `RELATES_TO`), plus
graphiti's exact `WHERE score > $min_score … ORDER BY score DESC LIMIT` shape.
One **informational** check that `CREATE VECTOR INDEX` DDL is accepted — it is the
obvious remediation for §9 and costs one statement.

**4. `fulltext` — BM25 and Lucene escaping.** Create a `compat_` fulltext index,
write text, query it. Then the escaping check: issue
`db.index.fulltext.queryNodes` with a query string containing every Lucene
metacharacter (`+ - && || ! ( ) { } [ ] ^ " ~ * ? : \ /`) passed through
graphiti's own sanitiser — `graphiti_core.helpers.lucene_sanitize`, as wrapped by
`graphiti_core.search.search_utils.fulltext_query(query, group_ids, driver)` — and
assert no `ParseException`. Cheap, and the kind of break that only shows up on a
real user question.

**5. `graphiti-write` — dynamic labels and bi-temporal edges, no LLM.**
Construct `EpisodicNode`, `EntityNode`, and `EntityEdge` objects directly with
**fabricated** 768-d embeddings and call graphiti's own `.save()` /
`add_nodes_and_edges_bulk` paths. This exercises `SET n:$(node.labels)` (the
5.22 fault line), the bi-temporal `valid_at`/`invalid_at` writes, and embedding
persistence — at zero LLM cost.

**6. `graphiti-search` — the recipes we actually use.** Run `graphiti._search`
against the synthetic graph from group 5 with a fabricated query embedding, under
each recipe the codebase invokes: `EDGE_HYBRID_SEARCH_RRF`,
`EDGE_HYBRID_SEARCH_NODE_DISTANCE` (supplying a `center_node_uuid`), and the node
and community recipes used by `global_search` and `drift`. The single
highest-value group: one call per recipe covers RRF/MMR/node-distance reranking
and all the vector + fulltext Cypher beneath them.

**7. `our-cypher` — every query we own.** Real calls against the synthetic graph:
`Provenance.resolve_citations`, `answer_api.search._vendor_episode_uuids`,
`answer_api.freshness.freshness`, the `timeline` query, `global_search`'s
community shortlist, `drift`'s follow-up query, `theme_builder.detect`
(GDS projection + seeded Leiden), `theme_builder.writeback`,
`theme_builder.incremental.touched_entities` / `load_persisted`, `cli._corpus_cursor`,
and `graph_extract.staleness_sweep` — the last two being the bare-`CALL {}` sites,
which are expected to pass and are reported as *pass (deprecated form)*.

**8. `e2e` — one real article.** A small fixed markdown body **inlined in the
harness** (not fetched from DocExtractor, so the check does not depend on that
service being up) through `add_text_episode`, then assert episodes, entities, and
`RELATES_TO` facts were created, then `search_local` returns at least one result
with resolved citations.

This check is **deliberately scoped to extraction + retrieval and never
synthesis**, so it does not depend on the GLM tier currently being replaced. If
the LLM or embedder endpoint is unreachable, the check reports **`skip`** with the
reason — an unreachable model is not a Neo4j incompatibility and must not fail the
verdict.

## 7. Verdict and report

`report.verdict(results) -> Literal["GO", "GO_WITH_CONFIG", "NO_GO"]`, over
non-informational, non-skipped results only:

- **`NO_GO`** — at least one check has `status == "fail"` and
  `cypher5_retry in (None, "fail")`. A genuine incompatibility.
- **`GO_WITH_CONFIG`** — every failure has `cypher5_retry == "pass"`. The
  remediation is `db.query.default_language=CYPHER_5` on the server, not a code
  change.
- **`GO`** — no failures.

Skips never move the verdict; they are listed in a *Not verified* section so the
report is honest about coverage. Informational results render in the matrix with
an `(info)` marker and are likewise excluded.

`docs/superpowers/neo4j-compat-report.md` contains: a target header (host, kernel
version, edition, default Cypher language, GDS version, APOC presence), the
verdict, a per-group table `| check | status | Cypher 5 | detail |`, the
*Not verified* list, a *Recommended actions* list derived from the failures, and a
note recording the schema side effect from §4.

## 8. Fix scope

Per the approved call: breaks in **our** Cypher are fixed in this slice; breaks
inside graphiti-core are reported with a recommendation and are not patched.

Two fixes are already known to be needed and are in scope:

1. **Bump the Neo4j testcontainer.** `tests/integration/conftest.py:18` and `:37`
   pin `neo4j:5.22`. That image cannot parse graphiti's bulk `SET n:$(node.labels)`
   save, which is why so much coverage had to be `@live`, and it also rejects the
   scoped `CALL (x) { … }` form that fix 2 requires. Bump both fixtures to
   **`neo4j:2026.07.1-community`** — the published image matching the production
   target exactly — so the non-live suite exercises what production runs,
   including its `CYPHER_25` default.
2. **Modernise the two bare `CALL {}` sites** to the scoped form —
   `staleness_sweep.py:52` → `CALL (eps) { … }`, `theme_builder/cli.py:69` →
   `CALL () { … }` — and delete the now-obsolete comment at
   `staleness_sweep.py:37-41` that documents the 5.22 workaround. Fix 1 must land
   first; these two changes are coupled.

**Acceptance is single-instance.** Verified 2026-09-02: the compose Neo4j
container and its Docker volume no longer exist, so the pilot `backup-docs` graph
(39 articles / 555 entities / 1,975 facts / 68 reports) is gone and 2026.07 is now
the only instance. Backward compatibility with 5.x is therefore **not** a
requirement of this slice — the scoped `CALL (x) { … }` form needs Neo4j 5.23+,
which is fine because nothing older is deployed. After the fixes the full non-live
suite passes and the harness reports `GO` against the 2026.07 target.

The lost graph is rebuildable by re-ingesting from DocExtractor (the community
layer was designed as derived-and-disposable); what was lost is the extraction
spend and the eval baseline behind `router-eval-report.md`. Re-ingestion is the
bootstrap this slice gates, not part of it.

## 9. Recorded follow-up (not this slice)

Graphiti 0.29.2 performs similarity by brute-force `vector.similarity.cosine`
scans with no ANN index (§2). At the pilot's 1,975 facts this is invisible; at the
projected **3–5 M facts** of the full corpus every hybrid search would scan every
fact embedding. This is a scaling blocker for broad ingestion, independent of the
version jump, and needs its own slice (options: a newer graphiti with index-backed
search, a pre-filter on `group_id`/structural scope, or vector indexes plus a
patched query path). The harness's informational `CREATE VECTOR INDEX` check
exists to confirm that remediation path is open. This finding also amends
`docs/infrastructure-sizing.md`, which assumed index-backed vector search.

## 10. Error handling and edge cases

| Case | Behaviour |
|---|---|
| A check raises | Caught; recorded `fail` with the exception type + message truncated to a readable length. The run continues — one bad check never aborts the harness. |
| A failed `CypherCheck` | Automatically re-run as `"CYPHER 5 " + cypher`; `cypher5_retry` set to `pass`/`fail`. |
| A failed `CallableCheck` | `cypher5_retry` stays `None`; rendered `n/a (procedural)`. |
| GDS absent on the target | The GDS checks report `skip`, not `fail` — GDS is a plugin, not a Neo4j capability. |
| LLM/embedder unreachable | Group 8 reports `skip` with the reason. Groups 1–7 use fabricated embeddings and are unaffected. |
| Teardown itself fails | Logged as a warning and appended to the report as an explicit **manual cleanup required** note naming the `compat-check` group_id. Never swallowed silently. |
| Target unreachable at startup | The CLI exits non-zero with a clear message before any check runs; no partial report is written. |
| Harness run against a populated instance | Safe by construction (§4): all writes namespaced, all deletes scoped to `group_id='compat-check'`. |

## 11. Testing

- **Unit (`report.py`, pure):** `verdict` over canned `CheckResult` lists —
  all-pass → `GO`; a failure with `cypher5_retry="pass"` → `GO_WITH_CONFIG`; a
  failure with `cypher5_retry="fail"` → `NO_GO`; a failure with
  `cypher5_retry=None` → `NO_GO`; skips and informational results excluded from
  every branch. Markdown rendering asserts the matrix rows, the *Not verified*
  section, and the `(info)` marker.
- **Unit (`runner.py`):** the Cypher-5 retry mechanism against a fake session that
  raises on the bare statement and succeeds on the `CYPHER 5 `-prefixed one;
  exception containment (a raising check yields `fail`, and later checks still
  run); teardown invoked even when a check raises.
- **Integration (testcontainer):** run the whole harness against the Neo4j
  testcontainer and assert it produces a complete result set and a rendered
  report. Group 8 will `skip` (no LLM in CI), which is itself asserted — it proves
  the skip path works.
- **`@live`:** run against the `COMPAT_*` target, write
  `docs/superpowers/neo4j-compat-report.md`, assert the verdict is not `NO_GO`.
- Full non-live suite, `uv run ruff check src tests`, and `uv run mypy src` clean.

## 12. Acceptance criteria

1. `python -m compat.cli` runs against the `COMPAT_*`-resolved target, executes
   all eight groups, and writes `docs/superpowers/neo4j-compat-report.md` with a
   verdict, a per-group matrix, and recommended actions.
2. The harness is safe against a populated instance: all writes namespaced to
   `group_id='compat-check'`, all harness indexes `compat_`-prefixed, teardown
   guaranteed and reported when it fails.
3. A failing Cypher check is automatically retried under `CYPHER 5` and the report
   distinguishes "needs a server config change" from "needs a code change".
4. The two bare `CALL {}` sites are converted to scoped form and the
   testcontainer image is bumped; the full non-live suite passes.
5. The harness reports `GO` against the 2026.07 target.
6. Pure logic is unit-tested; the harness loop is testcontainer-tested; the
   `@live` run produces the report.
7. `ruff check src tests` and `mypy src` clean.

## 13. Deferred

- Patching graphiti-core internals, or upgrading `graphiti-core` past 0.29.2.
- The brute-force vector-scan scaling problem (§9) — its own slice.
- Any data migration to the new instance, and the bootstrap ingestion itself.
- Running the harness in CI (it targets a named external instance on demand).
- Replacing the GLM synthesis tier — orthogonal, and §6 group 8 is scoped so this
  slice does not depend on it.
