# Neo4j / Graphiti Compatibility Report

## Target

- **uri:** bolt://alpcirag01.ai.area51.pp.ua:7687
- **kernel:** 2026.07.1 (community)
- **default Cypher language:** CYPHER_25
- **gds:** 2026.07.0

## Verdict: GO

- No action required — the target is compatible.

## Scope of this verdict

**What this GO does establish, and now verifies results, not just parse-and-plan:**

- **The provenance join — the system's core citation guarantee — is verified end
  to end.** The fixture has a real `Vendor->Product->Source->Article` chain
  written through `graph_sync.neo4j_repo.apply_structural` (the actual ingestion
  write path, not hand-written Cypher), and `Provenance.link` attaches the
  `(:Article)-[:HAS_EPISODE]->(:Episodic)` edge the same way `graph-sync` does in
  production. `resolve_citations` is asserted against the exact expected
  `url`/`vendor`/`product`/`section` values, not merely checked for a non-empty
  shape.
- **The e2e check is a faithful mini-ingestion**, not a check that happened to
  pass on empty results. It runs structural write -> real extraction
  (`add_text_episode`) -> `Provenance.link` -> `search_local`, and asserts at
  least one resolved source whose URL matches the ingested article — it does
  **not** assert that every retrieved fact resolves to that article, since
  `FACT_AB`/`FACT_BC` from the group-5 synthetic fixture are still valid at e2e
  time and legitimately resolve to `ARTICLE_URL` instead. On this run,
  retrieval returned 5 results with 5 resolved sources, and at least one of
  those resolved to the e2e article's URL.
- **The community layer is verified through the real write and read paths.**
  `theme_builder.writeback.write_communities_incremental` writes a `Community`
  node, and it is read back correctly by both
  `theme_builder.incremental.load_persisted` and
  `answer_api.global_search.shortlist_communities` (asserted to return the
  written community, not merely a non-empty list).
- **Both directions of the staleness sweep are verified.** The fixture carries a
  fact supported only by an orphan (never-linked) episode alongside facts
  supported by a linked one; `sweep_stale_facts` is asserted to expire exactly
  the orphan's fact and leave the linked-episode facts untouched — the sweep's
  two failure modes (under- and over-expiring) both have a check that would
  catch them.
- **The two hybrid-search recipes behind `/search/local`, GDS Leiden detection,
  and the theme-builder incremental helpers now assert non-empty/non-null
  results, not just "did not raise."** `EDGE_HYBRID_SEARCH_RRF` and
  `EDGE_HYBRID_SEARCH_NODE_DISTANCE` raise if they return zero edges (this run:
  2 edges each — previously both recipes could pass on zero edges, the exact
  failure class this branch exists to catch), `detect_communities` raises on
  zero communities (this run: 1), `touched_entities` raises on an empty
  non-`None` result (this run: 3), and `freshness` now also asserts
  `reports_as_of` is not `None` (guaranteed since the community write precedes
  it).
- **Nothing skipped in this run.** The configured LLM/embedder endpoints were
  reachable, so `e2e` ran the real ingest path rather than taking its
  `SkipCheck` exit; every other group ran too. The only non-PASS row is the
  pre-existing, `informational=True` `apoc installed` check, which does not
  affect the verdict (nothing in this codebase calls `apoc.*`).

**What remains an honest, disclosed gap** — not exercised by this harness at
all:

- `drift`'s follow-up query (only `search_local`, reused by DRIFT's primer step,
  is covered).
- `apply_toc` (the chapter/TOC structural writer) — only `apply_structural` is
  exercised.
- The tombstone and `delete_source_articles` paths (out-of-band vendor/product/
  source deletion handling).
- `write_communities` (the full-rebuild path) — only the incremental
  `write_communities_incremental` writer is exercised.
- Synthesis — `answer_local` / `_finalize_answer` / `_synthesis_client_and_model`
  — the e2e check deliberately stops at retrieval and never calls the
  synthesis tier (`test_e2e_check_never_references_synthesis` enforces this).
- `timeline_local` — only its `_sweep_flags` helper is exercised, not the
  endpoint's full response assembly.
- `global_search`'s map-reduce reduce step — only `shortlist_communities` (the
  map/shortlist stage) is exercised.

None of this changes the verdict. It means the GO now covers "the target's
structural, extraction, provenance, community, and hybrid-search write/read
paths behave correctly and retrieval returns correctly cited results" — a
materially stronger claim than the earlier "the target speaks the Cypher this
codebase writes" — while the items above stay tracked as follow-up work rather
than silently assumed. **The harness retrieves and cites; it does not answer.**

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
| bootstrap | structural fixture chain | PASS | - | structural chain written: compat-check-vendor -> compat-check-article |
| vector | seed vector probe nodes | PASS | - | {'created': 1} |
| vector | cosine similarity over a node property | PASS | - | {'score': 1.0} |
| vector | cosine similarity over a relationship property | PASS | - | {'score': 1.0} |
| vector | graphiti min_score query shape | PASS | - | {'uuid': 'compat-vec-a', 'score': 1.0} |
| vector | CREATE VECTOR INDEX accepted (info) | PASS | - |  |
| fulltext | CREATE FULLTEXT INDEX accepted | PASS | - |  |
| fulltext | await fulltext index online | PASS | - |  |
| fulltext | seed fulltext probe node | PASS | - | {'uuid': 'compat-ft-a'} |
| fulltext | await fulltext index refresh | PASS | - |  |
| fulltext | db.index.fulltext.queryNodes returns ranked hits | PASS | - | {'uuid': 'compat-ft-a', 'score': 0.13076457381248474} |
| fulltext | lucene metacharacter escaping | PASS | - | sanitised query accepted, 1 hits |
| graphiti-write | write synthetic graph via graphiti models | PASS | - | 2 episodes (1 orphan), 3 entities, 3 bi-temporal facts written |
| graphiti-write | dynamic entity labels persisted | PASS | - | labels persisted: ['Entity', 'Product'] |
| graphiti-write | bulk dynamic-label expression SET n:$(node.labels) | PASS | - | {'labels': ['Entity', 'Product', 'Feature']} |
| graphiti-write | bi-temporal fact properties persisted | PASS | - | valid_at set, invalid_at null, embedding dim 768 |
| graphiti-write | provenance link (Article HAS_EPISODE) | PASS | - | article linked to episode, section='Retention' |
| graphiti-search | EDGE_HYBRID_SEARCH_RRF recipe | PASS | - | RRF recipe returned 2 edges |
| graphiti-search | EDGE_HYBRID_SEARCH_NODE_DISTANCE recipe | PASS | - | node_distance recipe returned 2 edges |
| our-cypher | write community layer | PASS | - | community layer written: {'reports_written': 1, 'by_level': {0: 1}} |
| our-cypher | provenance resolve_citations | PASS | - | provenance join resolved: AWS/AWS Backup https://example.invalid/compat-check/vault-lock |
| our-cypher | vendor episode scope | PASS | - | vendor scope resolved 1 episode uuid(s) for AWS |
| our-cypher | freshness stamps | PASS | - | freshness stamps resolved: {'graph_cursor_time': '2026-09-03T03:33:28.771976Z', 'reports_as_of': '2026-09-02T00:00:00Z'} |
| our-cypher | theme-builder corpus cursor (CALL {} UNION ALL) | PASS | - | corpus cursor subquery ran: 2026-09-03T03:33:28.771976Z |
| our-cypher | incremental touched_entities | PASS | - | touched_entities ran: 3 |
| our-cypher | incremental load_persisted | PASS | - | load_persisted=1 community(ies), prev_cursor=2026-09-02T00:00:00Z |
| our-cypher | global-search community shortlist | PASS | - | shortlist returned 1 hit(s), top=compat-check-community |
| our-cypher | staleness sweep (scoped CALL (eps)) | PASS | - | sweep expired exactly the unsupported fact: {'scanned': 3, 'expired': 1, 'expired_sample': ['compat-fact-orphan']} |
| our-cypher | timeline sweep flags | PASS | - | sweep flags correct: {'compat-fact-ab': False, 'compat-fact-orphan': True} |
| our-cypher | GDS projection + seeded leiden detect | PASS | - | GDS projection + seeded Leiden ran: 1 communities |
| e2e | one article: extract then retrieve | PASS | - | ingested + linked; search_local returned 5 result(s) with 5 resolved source(s) |

## Not verified

- Nothing skipped; every check ran.

## Side effects

Graphiti's own indexes (created by `build_indices_and_constraints()`) are left in place deliberately — the call is idempotent and those indexes are exactly what a real bootstrap needs. `graph_sync.neo4j_repo.init_schema()` (run by the `graph_sync structural schema` check) similarly leaves 5 global constraints (`vendor_id`, `product_id`, `source_id`, `article_id`, `chapter_id`) and 2 indexes (`article_source`, `chapter_source`) on the target — harmless and arguably desirable (a real bootstrap needs them too), but disclosed here since they are not `compat_`-prefixed and teardown does not drop them. All harness *data* (`group_id='compat-check'`) and all `compat_`-prefixed indexes are removed in teardown.
