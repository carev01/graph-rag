# Neo4j / Graphiti Compatibility Report

## Target

- **uri:** bolt://alpcirag01.ai.area51.pp.ua:7687
- **kernel:** 2026.07.1 (community)
- **default Cypher language:** CYPHER_25
- **gds:** 2026.07.0

## Verdict: GO

- No action required — the target is compatible.

## Scope of this verdict

**What this GO does establish:** the queries this harness exercises — the
retrieval-mode queries, the theme-builder subqueries and watermark, the staleness
sweep, and the schema-declaration DDL — together with
graphiti-core's own write and search paths, parse, plan, and execute successfully
against Neo4j 2026.07.1 under its default `CYPHER_25` language setting. Graphiti's
dynamic-label writes (`SET n:$(node.labels)`) and its bi-temporal fact properties
(`valid_at`/`invalid_at`) also write and read back correctly. That is a real,
load-bearing result: it means the target Neo4j version will not throw a syntax or
planning error partway through a ~1B-token bootstrap.

**What this GO does NOT establish** — read these before treating GO as "ready for
bootstrap, full stop":

- **The provenance join — the system's core citation guarantee — is unverified.**
  The synthetic fixture has no `(:Article)-[:HAS_EPISODE]->(:Episodic)` chain, so
  `resolve_citations` ran and returned its 2 expected fact entries, but every entry
  came back with an empty `sources` list. The query parses and plans, but nobody has
  proven it actually resolves a fact to a real source URL on this server.
- **Several `our-cypher` checks only proved parse-and-plan, not result
  correctness**, because they ran over empty result sets: `_vendor_scope` (0
  matching episode uuids — no vendor/product/source structural graph exists in the
  fixture), `_timeline_sweep_flags`, `_load_persisted` (0 communities, because
  nothing built any), and `_resolve_citations` (see above). A query that returns
  the right *shape* of empty answer looks identical, in this harness, to a query
  that is silently wrong.
- **The ingestion write path itself is uncovered.** `graph-sync`'s own write
  queries — `graph_extract.provenance.Provenance.link` and
  `graph_sync.neo4j_repo`'s structural-apply Cypher — are never exercised. Only
  `graph_sync.neo4j_repo.init_schema()` runs, which proves the constraints/indexes
  it declares are accepted, not that the structural writes it enables behave
  correctly.
- **The community layer is entirely unexercised.** No `Community` node exists in
  the fixture, so `global_search`'s community shortlist query, `drift`'s follow-up
  query, and `theme_builder.writeback` never run against this target at all.

None of this changes the verdict — nothing above failed, because nothing above
ran. It means the GO covers "the target speaks the Cypher this codebase writes"
and not yet "the target correctly answers a real user's question with a correct
citation." The synthetic fixture would need an Article/HAS_EPISODE chain, a
populated structural graph, and a Community layer to close that gap — tracked as
follow-up work, not a blocker for this verdict.

## Results

| group | check | status | Cypher 5 | detail |
|---|---|---|---|---|
| server | kernel version and edition | PASS | - | {'version': '2026.07.1', 'edition': 'community'} |
| server | default Cypher language | PASS | - | {'value': 'CYPHER_25'} |
| server | gds version | PASS | - | GDS 2026.07.0 |
| server | apoc installed (info) | FAIL | n/a (procedural) | expect predicate rejected rows: [{'n': 0}] |
| server | fulltext procedures available | PASS | - | {'n': 2} |
| server | vector similarity function available | PASS | - | {'n': 1} |
| bootstrap | graphiti build_indices_and_constraints | PASS | - | build_indices_and_constraints() completed |
| bootstrap | graphiti fulltext indexes exist | PASS | - | {'names': ['community_name', 'edge_name_and_fact', 'episode_content', 'node_name_and_summary']} |
| bootstrap | graphiti range indexes exist | PASS | - | {'n': 34} |
| bootstrap | graph_sync structural schema | PASS | - | graph_sync init_schema() completed |
| vector | seed vector probe nodes | PASS | - | {'created': 1} |
| vector | cosine similarity over a node property | PASS | - | {'score': 1.0} |
| vector | cosine similarity over a relationship property | PASS | - | {'score': 1.0} |
| vector | graphiti min_score query shape | PASS | - | {'uuid': 'compat-vec-b', 'score': 1.0} |
| vector | CREATE VECTOR INDEX accepted (info) | PASS | - |  |
| fulltext | CREATE FULLTEXT INDEX accepted | PASS | - |  |
| fulltext | await fulltext index online | PASS | - |  |
| fulltext | seed fulltext probe node | PASS | - | {'uuid': 'compat-ft-a'} |
| fulltext | await fulltext index refresh | PASS | - |  |
| fulltext | db.index.fulltext.queryNodes returns ranked hits | PASS | - | {'uuid': 'compat-ft-a', 'score': 0.13076457381248474} |
| fulltext | lucene metacharacter escaping | PASS | - | sanitised query accepted, 1 hits |
| graphiti-write | write synthetic graph via graphiti models | PASS | - | 1 episode, 3 entities, 2 bi-temporal facts written |
| graphiti-write | dynamic entity labels persisted | PASS | - | labels persisted: ['Entity', 'Product'] |
| graphiti-write | bulk dynamic-label expression SET n:$(node.labels) | PASS | - | {'labels': ['Entity', 'Product', 'Feature']} |
| graphiti-write | bi-temporal fact properties persisted | PASS | - | valid_at set, invalid_at null, embedding dim 768 |
| graphiti-search | EDGE_HYBRID_SEARCH_RRF recipe | PASS | - | RRF recipe returned 2 edges |
| graphiti-search | EDGE_HYBRID_SEARCH_NODE_DISTANCE recipe | PASS | - | node_distance recipe returned 2 edges |
| our-cypher | provenance resolve_citations | PASS | - | resolve_citations returned 2 entries |
| our-cypher | vendor episode scope | PASS | - | vendor episode scope query ran, 0 uuids |
| our-cypher | freshness stamps | PASS | - | freshness stamps resolved: {'graph_cursor_time': '2026-09-03T00:20:10.279019Z', 'reports_as_of': None} |
| our-cypher | timeline sweep flags | PASS | - | timeline sweep-flag query ran, 2 flags |
| our-cypher | theme-builder corpus cursor (CALL {} UNION ALL) | PASS | - | corpus cursor subquery ran: 2026-09-03T00:20:10.279019Z |
| our-cypher | staleness sweep (CALL {} importing WITH) | PASS | - | staleness sweep subquery ran: expired=2 |
| our-cypher | incremental touched_entities | PASS | - | touched_entities ran: all (None sentinel) |
| our-cypher | incremental load_persisted | PASS | - | load_persisted=0 communities, prev_cursor=None |
| our-cypher | GDS projection + seeded leiden detect | PASS | - | GDS projection + seeded Leiden ran: 1 communities |
| e2e | one article: extract then retrieve | PASS | - | ingested {'episodes': 2, 'entities': 9, 'facts': 7}; search_local returned 5 results with 0 resolved sources |

## Not verified

- Nothing skipped; every check ran.

## Side effects

Graphiti's own indexes (created by `build_indices_and_constraints()`) are left in place deliberately — the call is idempotent and those indexes are exactly what a real bootstrap needs. `graph_sync.neo4j_repo.init_schema()` similarly leaves its structural constraints and indexes (the Vendor/Product/Source/Chapter/Article schema) behind on the target, for the same reason: it is idempotent and is exactly the schema a real bootstrap needs in place. All harness *data* (`group_id='compat-check'`) and all `compat_`-prefixed indexes are removed in teardown.
