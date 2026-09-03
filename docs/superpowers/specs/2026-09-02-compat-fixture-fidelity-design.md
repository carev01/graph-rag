# Compat-Harness Fixture Fidelity — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Give the Neo4j compatibility harness a realistic synthetic fixture, so its
checks verify **results** rather than merely that queries parse — above all the
provenance join, which is design decision #2 and the system's core citation guarantee.
**Date:** 2026-09-02
**Status:** Approved design — ready for implementation planning

Follows `2026-09-02-neo4j-compat-check-design.md`. That harness merged and returned
**GO** against Neo4j 2026.07.1, but its report's "Scope of this verdict" section
discloses that the verdict covers parse-and-plan compatibility, not result
correctness. This slice closes most of that gap.

---

## 1. The problem

The synthetic fixture has no structural layer — no
`(:Vendor)->(:Product)->(:Source)->(:Article)-[:HAS_EPISODE]->(:Episodic)` chain. Three
consequences, all recorded in the merged report:

1. **`resolve_citations` returned 0 sources.** It found its 2 fact entries but every
   `sources` list was empty. The provenance join parses; nobody has shown it resolves a
   fact to a real source URL on this server.
2. **Several `our-cypher` checks passed over empty result sets** — `_vendor_scope` (0
   uuids), `_timeline_sweep_flags`, `_load_persisted` (0 communities). A query returning
   the right *shape* of empty answer is indistinguishable, in this harness, from one
   that is silently wrong.
3. **The ingestion write path is untested.** `graph_sync.neo4j_repo.apply_structural`
   and `graph_extract.provenance.Provenance.link` are what a real ingestion runs; only
   `init_schema()` was exercised.

A fourth, subtler consequence: the staleness sweep **expired** the synthetic facts,
because with no `:Article` they had no live support. The check reported that without
asserting it, so it happened to exercise the fires-when-unsupported path by accident.

## 2. Approach

Build the fixture **through the real write path** rather than hand-written Cypher.
`_APPLY_STRUCTURAL` (`src/graph_sync/neo4j_repo.py:17-25`) is a chain of
`MERGE (v:Vendor {id: $vendor.id}) SET v += $vendor`, so any extra key in the dicts
becomes a node property — including `group_id`. That single fact makes the real path
strictly better than hand-rolled Cypher: it fixes the fixture *and* covers the
untested ingestion writes *and* keeps every created node inside teardown's existing
`group_id='compat-check'` sweep.

### 2.1 Safety requirement (non-negotiable)

Every structural id the harness writes MUST be prefixed `compat-check-`.
`MERGE (v:Vendor {id: $vendor.id})` against a **real** vendor id would stamp
`group_id="compat-check"` onto a production node, which teardown would then delete.
This is the one way this slice could destroy data. The implementation pins literal
`compat-check-`-prefixed ids as module constants; no id is derived from configuration
or from anything on the target.

## 3. The fixture

All nodes carry `group_id="compat-check"`.

```
(:Vendor  {id:'compat-check-vendor',  name:'AWS'})
  -[:HAS_PRODUCT]->(:Product {id:'compat-check-product', name:'AWS Backup'})
  -[:HAS_SOURCE ]->(:Source  {id:'compat-check-source'})
  -[:HAS_ARTICLE]->(:Article {id:'compat-check-article',
                              title:'Vault Lock',
                              source_url:'https://example.invalid/compat-check'})
  -[:HAS_EPISODE {chunk_index:0, heading_path:'Retention'}]->(:Episodic {uuid:'compat-ep-1'})
```

- **Vendor name is `AWS`** so `_vendor_episode_uuids(driver, "AWS")` returns non-empty.
- **`source_url` uses `.invalid`**, a reserved non-resolving TLD (RFC 2606), so no
  fabricated URL appearing in a report is ever clickable or mistakable for real
  documentation.
- The entity/fact layer is unchanged: 3 entities, facts `FACT_AB` and `FACT_BC`, both
  supported by `compat-ep-1`.

### 3.1 The orphan fact

One addition: an `Episodic` `compat-ep-orphan` that is deliberately **never** linked to
an `:Article`, and a fact `FACT_ORPHAN` whose `episodes` list contains only it. The
sweep must expire exactly this fact and leave `FACT_AB`/`FACT_BC` alone — verifying
that a query which silently invalidates data both fires and refrains.

### 3.2 The community

One `:Community` written through the real
`theme_builder.writeback.write_communities_incremental(driver, group_id, entries, *,
corpus_cursor)`. It needs **no embedder** — entries carry their own embedding — so this
stays inside the harness's LLM-free constraint. Its members are the fixture entities and
its embedding is the fabricated vector.

Two details that decide whether the read-back checks see anything:

- The entry's **`level` and `rating` must match what the reading check asks for.**
  `global_search.shortlist_communities(driver, embedder, q, *, level, k, group_id,
  min_rating)` filters on both, so a community written at a different level, or with a
  rating below `min_rating`, yields an empty shortlist and the new check would pass
  vacuously — the exact failure this slice exists to remove. The implementation pins the
  written `level` and `rating` and the queried `level`/`min_rating` to the same
  constants.
- `write_communities_incremental` **dissolves communities absent from `entries`**, but
  it is scoped by `group_id`, so it can only ever affect the `compat-check` namespace.
  It cannot touch a real community layer on the target.

## 4. Where the checks go

Registry order is execution order and is load-bearing. The order becomes:

| # | Group | Change |
|---|---|---|
| 2 | `bootstrap` | **New:** `apply_structural` writes Vendor→Product→Source→Article. |
| 5 | `graphiti-write` | **New:** the orphan episode + `FACT_ORPHAN`. **New:** `Provenance.link` connects the Article to `compat-ep-1`. |
| 7 | `our-cypher` | **New first check:** `write_communities_incremental` writes the community. Then the strengthened reading checks, then the sweep, then the timeline flags. |
| 8 | `e2e` | Rewritten as a faithful mini-ingestion (§5). |

The structural write must precede the episode write (an `:Article` must exist before
`Provenance.link` can attach to it), and the sweep must run **after** every check that
reads facts but **before** `_timeline_sweep_flags`, which then asserts the orphan carries
`expired_by_sweep`.

## 5. The e2e check becomes a faithful mini-ingestion

Today's e2e calls `add_text_episode` and asserts `search_local` returns results. Its
citations are necessarily empty, because `add_text_episode` creates episodes and
graphiti never links them to an `:Article` — in production that link is graph-sync's job.
Simply tightening the assertion would fail the check without improving anything.

So e2e mirrors the real pipeline:

1. `apply_structural` for a **second** article (`compat-check-article-e2e`, so the
   extracted content is attributable to its own source).
2. `add_text_episode` — real extraction.
3. `Provenance.link(article_id, episode_uuid=<from AddEpisodeResults>, chunk_index=0, …)`.
4. `search_local`.
5. Assert **≥1 result** AND **≥1 resolved source** carrying the expected `url`, `vendor`,
   and `product`.

The skip contract is unchanged: only `httpx.ConnectError`, `httpx.ConnectTimeout`, and
`openai.APIConnectionError` yield `skip`; everything else fails. Steps 1 and 3 are our
own Cypher — if they raise, that is a genuine `fail`, not a skip.

## 6. Strengthened assertions

Replacing today's did-not-raise passes:

| Check | New assertion |
|---|---|
| `_resolve_citations` | Both facts resolve; sources non-empty; the first source's `url`, `vendor`, `product`, and `section` equal the fixture's values. |
| `_vendor_scope` | ≥1 episode uuid for vendor `AWS`. |
| `_load_persisted` | ≥1 persisted community, and `prev_corpus_cursor` is not None. |
| `_staleness_sweep` | The expired set is **exactly** `{FACT_ORPHAN}`. |
| `_timeline_sweep_flags` | `FACT_ORPHAN` is flagged `expired_by_sweep`; `FACT_AB` is not. |
| **New** `shortlist_communities` | Returns ≥1 hit, via a small fake embedder returning the fabricated vector. |
| e2e | ≥1 result and ≥1 resolved source with the expected url/vendor/product. |

The fake embedder is a ~5-line object exposing the one method
`global_search.shortlist_communities` calls; it keeps group 7 free of network I/O.

## 7. Teardown

Unchanged in mechanism — every new node carries `group_id="compat-check"`, so the
existing scoped sweep removes it. Two additions to verification: the integration test
asserts zero residual `Vendor`/`Product`/`Source`/`Article`/`Community` nodes in the
namespace, and the live run's post-run check covers the same labels.

## 8. Error handling and edge cases

| Case | Behavior |
|---|---|
| A structural id collides with a real production node | Prevented by construction (§2.1): ids are literal `compat-check-`-prefixed constants. |
| `apply_structural` raises | A genuine `fail` in `bootstrap` — it is our Cypher, and this is exactly the coverage the slice adds. |
| The sweep expires more than the orphan | `fail` with the actual expired set in the detail — a real regression signal. |
| The community write raises | `fail` in `our-cypher`. Later community-reading checks then also fail, which is correct: they have nothing to read. |
| LLM/embedder unreachable | e2e reports `skip`, exactly as before. Groups 1-7 remain unaffected. |
| Teardown leaves residue | Already surfaced as "manual cleanup required" in the report; now also covers the structural and community labels. |

## 9. Testing

- **Unit (registry shape):** the existing tests extended — the new checks appear in the
  right groups and in the required relative order; every structural id is
  `compat-check-`prefixed (a direct test of §2.1); no new check calls `apoc.*`.
- **Integration (testcontainer):** the harness runs end to end; assert the structural and
  community checks pass rather than skip, and that teardown leaves zero residual nodes
  across all fixture labels.
- **`@live`:** re-run against the target; the report is regenerated.
- Full non-live suite, `uv run ruff check src tests`, `uv run mypy src` clean.

## 10. Deliverable

A regenerated `docs/superpowers/neo4j-compat-report.md` whose "Scope of this verdict"
section is materially shorter — the provenance join, the structural write path, the
community layer, and both sweep directions move from "not established" to verified. The
remaining disclosed gaps shrink to `drift`'s follow-up query, `apply_toc`, and the
tombstone paths.

## 11. Acceptance criteria

1. The fixture includes the full structural chain, written via `apply_structural` and
   `Provenance.link`, with every id `compat-check-`prefixed and every node carrying
   `group_id="compat-check"`.
2. `resolve_citations` is asserted to return the expected url/vendor/product/section —
   the provenance join is verified, not merely parsed.
3. The sweep is asserted to expire exactly `FACT_ORPHAN`, covering both directions.
4. The e2e check performs structural write → extract → link → retrieve, and asserts at
   least one resolved source.
5. The community layer is written via the real writeback and read back by
   `_load_persisted` and a new `shortlist_communities` check.
6. Teardown removes every fixture node; verified in both the integration test and the
   live run.
7. The regenerated report's scope section reflects the newly verified coverage.
8. Full non-live suite, ruff, and mypy clean.

## 12. Deferred

- `drift`'s follow-up query, `apply_toc`, and the tombstone/`delete_source_articles`
  paths — they need materially more fixture scaffolding than this slice's budget.
- Re-running the router golden-set eval (it needs a re-ingested corpus, not a fixture).
- The Minor items carried from the previous slice: the substring-scan test, the
  `_retry_cell` label for a failed `expect` predicate, GDS exception narrowing, and the
  `pyproject.toml` wheel-packages gap.
