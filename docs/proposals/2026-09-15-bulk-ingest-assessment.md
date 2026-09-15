# `add_episode_bulk` versus per-chunk `add_episode` — assessment

**Date:** 2026-09-15
**Status:** Assessment only. No decision taken, nothing implemented, no branch.
**Trigger:** graphiti ships `Graphiti.add_episode_bulk` and this project has never
evaluated it. The concurrency-duplicate branch just merged (warm-up barrier + exact-name
merge pass) exists to manage a race that `dedupe_nodes_bulk` claims to resolve in one
pass. If that claim holds, bulk might be the primary answer and the merged branch the
safety net. This document establishes whether it holds.

Everything below is read from graphiti-core **0.30.1** as installed in `.venv`
(`graphiti_core/…`), from our `src/`, and from one read-only session against the live
Neo4j (2026.07.1 Community) while the W=8 validation run was in flight. Nothing was
written, nothing was run that spends money. Where a statement is an inference rather
than a citation it says so.

---

## 0. Summary of the verdict

**Keep per-chunk `add_episode`.** Bulk does remove the duplicate race *inside one
batch* — deterministically, without an LLM — but only if bulk batches never overlap, and
in exchange it (a) re-opens the two-range dedup prompt and the in-batch same-pair
contradiction path that `contradiction_gate` was built to close, (b) turns one failing
chunk into a lost batch that leaves orphan `:Episodic` nodes in the graph, (c) widens
the provenance-link window from one chunk to one batch, (d) shrinks the extraction
context from 10 previous episodes to 3 and includes the episode itself, and (e) resolves
every node and every edge against the graph a second time. The duplicate problem already
has a measured deterministic repair (30 of 30). Bulk is not a better answer to it; it is
a different concurrency model with unmeasured throughput and several measured-in-code
regressions. §9 says what would change this verdict and what the cheapest experiment is.

---

## 1. The four claims in the brief — which held

| # | claim | verdict | evidence |
|---|---|---|---|
| 1 | the `add_episode_bulk` docstring says invalidation and `valid_at`/`invalid_at` extraction are performed in bulk | **held, and the code agrees** — with a caveat | docstring `graphiti.py:1290-1293`. Code: `_resolve_nodes_and_edges_bulk` (`graphiti.py:815`) calls `resolve_extracted_edges` (`:902-914`), which reaches `_extract_edge_timestamps` (`edge_operations.py:680`, `:813`) and `resolve_edge_contradictions` (`:842`); invalidated edges are persisted in the final save (`graphiti.py:1406`). Caveat: the *in-batch* stage `dedupe_edges_bulk` says "For now we won't track edge invalidation" (`bulk_utils.py:559`) but does so anyway by in-place mutation — §5.1. In our deployment "date extraction" means our `deterministic_valid_at` patch on both paths, not graphiti's LLM call. |
| 2 | `RawEpisode` carries a per-episode `reference_time` | **held** | `bulk_utils.py:101-107`; it becomes `EpisodicNode.valid_at` at `graphiti.py:1330`, exactly as `add_episode` does at `:1110`. Our `content_changed_at` grounding is expressible per chunk. |
| 3 | `dedupe_nodes_bulk` runs `resolve_extracted_nodes` per episode in parallel, then a cross-batch pass | **held, but incomplete** | first pass `bulk_utils.py:389-400`; cross-batch pass `:413-456`. What the brief did not say: `_resolve_nodes_and_edges_bulk` then runs `resolve_extracted_nodes` **a second time** against the graph for every node (`graphiti.py:842-853`). Node resolution against the live graph happens twice per batch — §6. |
| 4 | graphiti writes `(:Saga)-[:HAS_EPISODE]->(:Episodic)` itself when given `saga`; our queries are label-scoped on `(:Article)` | **held on both halves; the collision is already partly materialised** | save query `edge_db_queries.py:308-316` (`MATCH (saga:Saga {uuid}) … MERGE (saga)-[e:HAS_EPISODE {uuid}]->(episode)`), called from `edges.py:697-704`; the delete at `edges.py:719` is what the brief cited. Bulk writes it at `graphiti.py:1443-1449`, `add_episode` at `:767-773`. Every `HAS_EPISODE` pattern in `src/` anchors on `:Article` — listed in §4.3. Live graph (read-only, 2026-09-15): 0 `:Saga` nodes, 0 `NEXT_EPISODE`, **755 `HAS_EPISODE` edges, all starting at `:Article`**. However `init_indices` → `build_indices_and_constraints` has already created graphiti's `has_episode_uuid`, `saga_uuid` and `next_episode_uuid` RANGE indexes (`graph_queries.py:58,62,63`; confirmed present with `SHOW INDEXES`). Our edges carry no `uuid`, so the index is empty and harmless, but the relationship type is shared with the library today, not only in a hypothetical. |

Net: all four held as stated. Two of them (3 and 4) were true but missing the part that
matters for the decision.

---

## 2. What the two paths actually do

### 2.1 Today — `IngestDriver.ingest_article` (`src/graph_extract/ingest_driver.py:140-172`)

fetch → tier by `is_dense_matrix` (`:112-118`) → chunk → for each chunk, in order:
`already_ingested` gate (`provenance.py:8-16`, keyed on `chunk_index` + `content_hash`
on the `HAS_EPISODE` edge) → `graphiti.add_episode` (`graphiti_client.py:278-287`) →
`provenance.link` (`provenance.py:18-49`) → supersede trailing chunks (`:182-194`).

Inside `add_episode` (`graphiti.py:980`): previous episodes are fetched **before** the
episode exists, `last_n=RELEVANT_SCHEMA_LIMIT` = 10 (`:1087-1096`, `search_utils.py:64`),
filtered to `source=text`; the `EpisodicNode` is built in memory (`:1099-1112`) and is
**first written in the final single transaction** with its nodes and edges
(`_process_episode_data` → `add_nodes_and_edges_bulk`, `:726-733`, one `execute_write`
at `bulk_utils.py:136-148`). A chunk that fails mid-extraction leaves nothing in the
graph. Chunks of one article are sequential; only *articles* fan out
(`run_concurrently`, `:211-213`).

### 2.2 Bulk — `Graphiti.add_episode_bulk` (`graphiti.py:1230-1487`)

1. Build every `EpisodicNode` from the `RawEpisode`s, input order preserved (`:1319-1333`).
2. **Persist all episodes immediately**, before any extraction (`:1336-1343`).
3. Previous-episode context per episode: `retrieve_previous_episodes_bulk`
   (`bulk_utils.py:110-125`) → `retrieve_episodes(valid_at, last_n=EPISODE_WINDOW_LEN)`
   with **`EPISODE_WINDOW_LEN = 3`** (`graph_data_operations.py:29`), no `source`
   filter, predicate `e.valid_at <= $reference_time`. Because step 2 already saved the
   batch, **each episode's context is drawn from the batch itself and includes the
   episode** (equal `valid_at` satisfies `<=`). Our chunks of one article share one
   reference time, so an article's chunks see each other, and themselves, as "previous".
4. Extract nodes then edges per episode, all in parallel under `semaphore_gather`
   with no `max_coroutines` → `SEMAPHORE_LIMIT`, env-driven, default 20
   (`bulk_utils.py:340-369`, `helpers.py:38,127`).
5. `dedupe_nodes_bulk` (`bulk_utils.py:374-486`): per-episode `resolve_extracted_nodes`
   against the live graph in parallel (`:389-400`); then a serial cross-batch pass over
   the union — exact match on `_normalize_string_exact` (lower-case, whitespace
   collapsed; `:425-438`), else MinHash/LSH fuzzy via `_resolve_with_similarity`
   (`:440-456`); union-find into one canonical map (`:458-463`). **No LLM in the
   cross-batch pass.**
6. `dedupe_edges_bulk` (`:489-581`): for every extracted edge, every other in-batch edge
   with the same endpoints is a candidate if it shares a word or has cosine ≥ 0.6
   (`:520-539`); then `resolve_extracted_edge(llm, edge, candidates, candidates, …)` —
   **the same list is passed as both `related_edges` and `existing_edges`**
   (`:547-557`). This is an LLM call per edge that has any same-endpoint sibling in the
   batch.
7. `_resolve_nodes_and_edges_bulk` (`graphiti.py:815-924`): `resolve_extracted_nodes`
   **again** against the graph (`:842-853`), attributes/summaries (`:875-886`), then
   `resolve_extracted_edges` against the graph (`:902-914`) — this is the stage the
   contradiction gate and lean search reach.
8. One final `add_nodes_and_edges_bulk` transaction (`:1401-1408`) — nodes, `MENTIONS`,
   `RELATES_TO`, and the episodes re-saved with `entity_edges`.
9. Saga block only if `saga` is passed (`:1410-1459`).
10. Return `AddBulkEpisodeResults(episodes=…, nodes=…, edges=…)` (`:1475-1482`), the
    episode list in input order.

---

## 3. Question 1 — does bulk remove the intra-batch race?

**Inside one `add_episode_bulk` call: yes, by construction.** Two episodes in the batch
that both extract a new `AWS Backup` each fail to find it in the graph (step 5 first
pass — the same race as today, but now in memory), and the cross-batch pass then unifies
them on normalized name with no model involved (`bulk_utils.py:425-438`). The canonical
map is applied to edge endpoints (`resolve_edge_pointers`, `graphiti.py:1368-1370`) and
to `MENTIONS` (`:1398`), and the batch's unique nodes are distributed so that **no node
appears under two episodes** before the second graph resolution
(`nodes_by_episode_unique`, `:831-839`). The 30 measured duplicates were all exact-name
collisions (`ab-concurrency-2026-09-14.md`); every one of them would have been unified
had the colliding articles been in one batch.

Three limits, each decisive for whether this helps *us*:

1. **Chunks of one article do not race today.** `ingest_article` is a sequential loop
   (`ingest_driver.py:157-170`). The 30 duplicates came from *articles* running
   concurrently at N=4. So a batch of one article — the natural unit for our
   replay gate, tier routing and provenance — is **no different from today** with respect
   to the measured problem. For bulk to address it, a batch must contain the articles
   that would otherwise be in flight together.
2. **Across batches it cannot help, and it makes the window wider.** Nodes are written
   only in the final transaction (step 8). An entity created in batch A is invisible to a
   concurrently running batch B for A's entire duration, whereas today it becomes
   visible when its chunk commits (~38 s sequential, ~12 s at N=4). Running bulk calls
   concurrently therefore reproduces the race at a coarser and longer grain. Bulk
   removes the race only under **"one bulk call in flight at a time"** — parallelism
   inside the call, serialisation between calls.
3. **Its matching is normalized, ours is byte-exact.** The cross-batch pass merges
   `Access Control`/`Access control`; the sequential baseline holds six such pairs apart,
   and the merge pass deliberately preserves them (`…duplicate-mitigation-design.md`
   §4.1). Within a batch, bulk would behave differently from the reference graph. Not
   wrong, but not "the state sequential ingest would have produced", which is the
   criterion the merge pass was designed to.

So the honest restatement of the central question is not "does bulk fix duplicates" but
"is *sequential bulk batches over article groups* a better concurrency model than
*concurrent per-chunk articles with a warm-up barrier and a merge pass*". §6–§8 answer
that.

---

## 4. Question 3 — replay gate, provenance, and the `HAS_EPISODE` collision

### 4.1 Per-episode uuids are returned, in order

`AddBulkEpisodeResults.episodes` is the list built at `graphiti.py:1319-1333` — a list
comprehension over `bulk_episodes`, so position `i` is chunk `i`. Linking by position is
sound. Pre-assigning `RawEpisode.uuid` is **not** an option for new episodes: a non-`None`
uuid makes graphiti *load* the episode (`EpisodicNode.get_by_uuid`, `:1320`), which
raises `NodeNotFoundError` for a uuid that does not exist yet (`nodes.py`, the
`EpisodicNode.get_by_uuid` body, ~`:388`).

### 4.2 Pre-filtering to not-yet-ingested chunks is straightforward

The gate is per chunk and independent of graphiti: run `already_ingested` over the
article's chunks before building `RawEpisode`s, exactly as the loop does today, then
`_supersede_trailing_episodes` after. The gate stays structural and group-agnostic — the
A/B already learned that group isolation does not bypass it
(`ab-concurrency-2026-09-14.md`, method note).

### 4.3 The invariant "graph-sync is the only writer of `HAS_EPISODE`" survives iff `saga` is never passed

Every `HAS_EPISODE` pattern in `src/` anchors on the Article label (verified by grep):
`provenance.py:12,36,55,76`; `ingest_driver.py:177,189`; `warmup.py:29`;
`staleness_sweep.py:65,75,77`; `graph_cleanup.py:148,156`; `answer_api/search.py:13`;
`graph_extract/eval.py:103`; `compat/checks.py:426`. graphiti's own readers anchor on
`:Saga` (`edges.py:719,739,765,799`; `graph_data_operations.py:106`). The two can coexist
on one relationship type without either seeing the other's edges. The live graph shows
the collision is latent (0 `:Saga`, 755 edges all from `:Article`) while the shared
index already exists (§1, claim 4).

A bulk adoption must therefore **never pass `saga`** and should add a tripwire test:
`MATCH (n)-[:HAS_EPISODE]->() WHERE NOT n:Article RETURN count(*)` = 0. The saga feature
offers nothing we need (`NEXT_EPISODE` chains are ordering we already carry in
`chunk_index`).

### 4.4 The link window widens from a chunk to a batch, and failure leaves orphans

Today the gap between "episode + facts committed" and "`HAS_EPISODE` written" is one
chunk (`ingest_driver.py:161-167`). With bulk it is one batch: the final transaction
commits *every* episode's entities and facts, and only then can we link them, one
statement per chunk. A crash in that window leaves fully-extracted, cited-nowhere facts
(`resolve_citations` returns no sources for them), and the replay gate re-extracts the
chunks on retry.

Worse is step 2 of §2.2: episodes are persisted **before** extraction. A batch that
fails at any later stage leaves N `:Episodic` nodes with full `content`, no `MENTIONS`,
no `HAS_EPISODE`. Nothing in `src/` finds or removes such nodes (grep for orphan
handling: none; `graph_cleanup` walks from `:Article`). They are invisible to the
liveness model and the sweep — but **`retrieve_episodes` has no "is linked" filter**, so
they are served as previous-episode context to every later episode at a nearby
`valid_at`. On retry the gate re-extracts the chunks (no `HAS_EPISODE` exists), so the
orphans accumulate: one set per failed attempt. Today a failed chunk leaves nothing.

---

## 5. Question 4 — do our extensions reach the bulk path?

All four patch a module attribute that the target module imports by name, so the
question is which bulk stages resolve that name.

| extension | patches | reaches bulk? | detail |
|---|---|---|---|
| `contradiction_gate` | `edge_operations.search` (`contradiction_gate.py:106`) | **the graph-resolution stage only** | `resolve_extracted_edges` calls `search` twice (`edge_operations.py:394,409`) and is what `_resolve_nodes_and_edges_bulk` invokes (`graphiti.py:904`), so the O(corpus) invalidation scan stays skipped there. But **`dedupe_edges_bulk` never calls `search`** — it builds candidates in memory and hands the same list in as both duplicate and invalidation candidates (`bulk_utils.py:547-557`). See §5.1. |
| `deterministic_valid_at` | `edge_operations._extract_edge_timestamps` (`deterministic_valid_at.py:78`) | **yes, both stages** | both call sites (`edge_operations.py:680,813`) resolve the name through module globals at call time. The second stage finds `valid_at` already set and returns early (`:587-588`). |
| `lean_edge_search` | `search_utils.get_entity_edge_return_query` (`lean_edge_search.py:83`) | **yes** | via the same `search` calls. `EntityEdge.get_between_nodes` (`edge_operations.py:367`) uses `edges.py`'s own query on both paths, unchanged from today. |
| `dedup_guard` | `graphiti.llm_client.generate_response` on the instance (`dedup_guard.py:193,260`), keyed on `prompt_name == "dedupe_edges.resolve_edge"` (`:211`) | **yes, both stages; attribution coarsens** | every dedup call on either stage goes through the guard. In the in-batch stage the prompt has N `EXISTING FACTS` and the *same* N as `FACT INVALIDATION CANDIDATES`; `parse_candidate_counts` (`:106-122`) sees two contiguous runs and returns `(N, N)`, so counting works and `dup_in_invalidation_range` will light up on index-space confusion exactly as it did before the gate. What is lost is **per-article attribution**: `CURRENT_DEDUP_STATS` is scoped in `ingest_article` (`ingest_driver.py:131`); a multi-article batch has one scope, so `IngestArticleResult.dedup` becomes per batch, and the per-article WARNING at `:137` cannot name the article. |
| — `dedup_guard` and `dedupe_nodes_bulk` | | **never covered, on any path** | the guard is edge-only. Node dedup (`dedupe_nodes.nodes`) is guarded by graphiti's own range checks (`node_operations.py:589-619`) today and would be in bulk. The cross-batch node pass is LLM-free. No instrumentation is lost here because none existed. |
| client-level wrappers (`_bound_index_arrays`, `_inject_penalties`, `_inject_openrouter_provider`, `_batch_capped_embeddings`, `usage.instrument`) | the raw `AsyncOpenAI` client (`graphiti_client.py:37-201`) | **yes** | they wrap below graphiti; every path is covered. |
| `max_coroutines` (`config.py:53`) | `Graphiti(max_coroutines=…)` | **neither path** | graphiti uses `self.max_coroutines` only for community operations (`graphiti.py:1190,1512-1521,1636`). Extraction gathers on both paths use `SEMAPHORE_LIMIT` (env, default 20; `helpers.py:38`). Not a bulk regression — a pre-existing misunderstanding worth recording: `MAX_COROUTINES` in `.env` does not bound extraction concurrency. |

### 5.1 The in-batch edge stage reintroduces what the gate removed

`dedupe_edges_bulk` passes `candidates` as both `related_edges` and `existing_edges`
(`bulk_utils.py:547-557`). Inside `resolve_extracted_edge` that yields the two-range
prompt (`edge_operations.py:700-713`) — the shape in which **all 267 observed
out-of-range indices** landed in the second range (`contradiction_gate.py:24-27`), and
which the gate collapsed to one range by emptying `existing_edges`. Every candidate is
listed twice under different indices, which is a new way to confuse the model.

It also re-arms same-pair contradiction *within the batch*. The stage discards the
returned `invalidated_edges` (`bulk_utils.py:561-564` reads only index 2), but
`resolve_edge_contradictions` mutates the candidate objects in place
(`edge_operations.py:569-571`), and the new edge can be expired by a sibling with a later
`valid_at` (`:826-839`). Those candidate objects **are the batch's extracted edges** and
are saved as mutated in step 8. So a batch spanning articles with different
`content_changed_at` can stamp `invalid_at` on a fact because a sibling fact in the same
batch was judged contradictory — the exact mechanism BACKLOG 33 flags as producing bad
invalidations — and whether it happens depends on which coroutine's
`_extract_edge_timestamps` ran first (all resolutions are concurrent under one
`semaphore_gather`, `:543-557`). Nondeterministic by LLM latency.

A patch is possible in our usual style: `bulk_utils` imports `resolve_extracted_edge` by
name (`bulk_utils.py:51-54`), so a wrapper that forces `existing_edges=[]` would restore
the one-range prompt for the in-batch stage. That is a fifth extension to write, test
with a tripwire, and maintain — not a reason to adopt, but a cost of adopting.

---

## 6. Question 2 — per-article tier routing under bulk

`_tier_for` (`ingest_driver.py:112-118`) picks a `Graphiti` instance per article. One
`add_episode_bulk` call is one instance, so a batch is one tier. Batching-by-tier is:

- partition the batch's articles by `is_dense_matrix`; build one `RawEpisode` list per
  tier (chunk sizes already differ per tier — `ExtractionTier.max_chunk_tokens` — and
  are decided per article, so nothing new there); call `strong.add_episode_bulk` and
  `cheap.add_episode_bulk`;
- **sequentially**, or the two calls race against each other exactly as two articles do
  today (§3, limit 2). The strong batch is usually small (dense matrices only), so the
  serial cost is bounded, but the share is not recorded in the A/B documents and the
  pilot log does not tag tiers — it would need measuring.

Direct token cost of partitioning: none. Indirect cost of bulk itself, both tiers:

- **node resolution twice** (`bulk_utils.py:389` and `graphiti.py:842`) — every new
  node pays a second embedding + cosine candidate search, and any node that reaches the
  LLM in the first pass and is still new reaches it again in the second (candidates from
  the graph have not changed);
- **edge resolution twice** — the in-batch LLM call (§5.1) for every edge with a
  same-endpoint sibling in the batch, then the graph call. Fact dedup was 22.9% of LLM
  time before this (`…extraction-model-self-hosting.md`); the in-batch stage adds calls in
  proportion to same-endpoint co-occurrence, which is highest for hub-heavy batches —
  the batches bulk is supposed to help with.

Whether the parallelism outweighs the duplicated work is **not determinable without a
run**. I would not predict it; the A/B already measured the two slowest prompts degrading
1.5–1.8x at 10–19 in flight and max in-flight 36 at N=4, and bulk's outer gathers run at
`SEMAPHORE_LIMIT` = 20 per stage with inner gathers beneath.

---

## 7. Question 5 — failure granularity

`semaphore_gather` is `asyncio.gather` without `return_exceptions` (`helpers.py:123-134`):
the first exception propagates out of the stage, the sibling coroutines are **not**
cancelled and keep spending until they finish, and `add_episode_bulk` re-raises
(`graphiti.py:1484-1487`). Nothing after step 2 is persisted. Consequences of one bad
chunk (a malformed cheap-tier reply is enough — `NodeResolutions(**llm_response)` at
`node_operations.py:558` and `EdgeDuplicate(**llm_response)` at `edge_operations.py:732`
raise on it, and the guard's fallback path is designed to raise loudly,
`dedup_guard.py:241-244`):

| | today (`add_episode` per chunk) | bulk |
|---|---|---|
| LLM spend lost | that chunk's | the whole batch's, plus the doomed siblings' tail |
| persisted state | nothing for that chunk; earlier chunks linked and gated | N orphan `:Episodic` nodes with content (§4.4); nothing else |
| retry | gate skips the good chunks, redoes one | gate redoes every chunk; orphans accumulate |
| worker accounting | one job (`semantic_worker.py:50-69`, per-job complete/fail) | all jobs in the batch fail together; the per-job lease/backoff model needs a batch-level counterpart |

At 126,405 articles this is the dominant consideration, as the brief anticipated. The
cheap tier's known failure mode is format, not comprehension (memory:
*graphiti unbounded index arrays*, *OpenRouter extraction-tier gotchas*); each occurrence
would cost a batch.

---

## 8. Question 6 — interaction with the warm-up barrier and the merge pass

- **Sequential bulk batches** (the only regime in which §3 holds): the warm-up barrier is
  redundant — every batch is already a barrier, and hub entities that first appear in a
  cold source's opening articles are unified in memory if those articles share a batch.
  The merge pass stays a valid safety net: byte-exact, report-first, complementary to
  bulk's normalized in-batch matching, and still the only repair for anything that leaks
  across batch boundaries. Not wrong together.
- **Concurrent bulk batches**: the race is back at batch grain with a longer window
  (§3, limit 2). Warm-up would have to be re-applied at batch granularity (cold source →
  its batch runs alone), and the merge pass becomes *more* necessary, not less.
- **One actively-wrong combination, in both regimes:** `merge-duplicates --apply` during a
  bulk batch. The final save uses `MATCH (source:Entity {uuid}) … MERGE` for every
  `RELATES_TO` (`edge_db_queries.py:176-179`) and `MENTIONS` (`:42-51`); if
  the merge deleted a loser the batch resolved to, those edges are silently not written.
  This is the existing runbook rule (`…duplicate-mitigation-design.md` §4.9) with a much
  larger exposure window — a whole batch's resolution happens before its single commit.

So: bulk does not make the merged branch a safety net *instead of* the primary
mechanism; in the only regime where bulk works, warm-up is moot and the merge pass is
still the primary repair for the residual.

---

## 9. Recommendation

**Keep what we have.** Per-chunk `add_episode`, article-level fan-out, warm-up barrier,
exact-name merge pass. Do not adopt `add_episode_bulk` as the ingest path.

Why, in order of weight:

1. It addresses the measured problem only as "sequential batches of several articles" —
   a concurrency model whose throughput is unmeasured and structurally burdened by stage
   barriers and double resolution (§6), against a measured 3.20x we already have.
2. It reintroduces the two-range dedup prompt and unguarded in-batch same-pair
   contradiction stamping (§5.1) — the regressions this project spent two slices removing
   and measuring.
3. One failing chunk costs a batch and leaves orphan episodes that pollute future
   extraction context (§4.4, §7). At corpus scale that is the cost that matters.
4. The duplicate problem already has a deterministic repair that fixed 30 of 30, and the
   W=8 validation in flight is the measurement that decides whether the barrier reduces
   the residual to the predicted ~6. That result, not bulk, is the next decision point.
5. Our extensions mostly reach bulk (§5), so this is not "bulk would break the patches" —
   it is that bulk adds a stage the patches were never written for.

**What would change the verdict.** If the W=8 run shows warm-up does *not* reduce the
residual and operating the merge pass proves costly at scale (frequent quiet windows,
label conflicts on hubs), then sequential-bulk-over-article-groups is worth one paid
measurement — after the zero-cost checks below.

**Cheapest experiments, in order.**

1. *Zero cost, testcontainer, fake LLM:* pin (a) `results.episodes` order equals input
   order; (b) a raised exception after step 2 leaves N orphan `:Episodic` nodes and no
   `HAS_EPISODE`; (c) no `HAS_EPISODE` is written when `saga` is omitted; (d) the
   in-batch dedup prompt reaches the guard with `M == N`. These convert the §4–§7
   readings into tests and cost nothing.
2. *~$10–12, ~2.5 h:* the 83 pilot articles by the supersede-and-restore method
   (`.superpowers/sdd/ab-w8-validation.py`, `ab-revert.py`), batched 4 articles per bulk
   call by tier, batches sequential, `SEMAPHORE_LIMIT` pinned. Measure exact-name
   duplicates, wall clock, per-prompt call counts (the double-resolution cost is a count,
   not a guess), in-batch `invalid_at` stamps, and orphans after an injected failure.
   Expect the entity total to drift from 999 for reasons unrelated to dedup: the previous-
   episode context changes from 10 to 3-including-self (§2.2 step 3), and the A/B already
   saw ~28% of names differ run to run. Compare on duplicates and call counts, not on
   yield.

---

## 10. Migration sketch — only if §9's verdict is later reversed

Sequenced so each step is reversible until the last.

0. **Guard the invariant first.** Tripwire test: every `HAS_EPISODE` starts at `:Article`;
   `saga` is never passed (grep-level test on `graphiti_client`). Also assert
   `SEMAPHORE_LIMIT` is set explicitly in the environment the worker runs in.
1. **Fifth extension: one-range in-batch dedup.** Wrap `bulk_utils.resolve_extracted_edge`
   (imported by name, `bulk_utils.py:51-54`) to pass `existing_edges=[]`. Tripwire on the
   prompt shape via the guard (`M == 0`). Without this, step 3 writes in-batch
   `invalid_at` stamps that are **not reversible** — they are indistinguishable from
   legitimate supersession once in the graph.
2. **`IngestDriver.ingest_batch(article_ids)`**: fetch, navigation filter, chunk,
   `already_ingested` pre-filter per article; partition by tier; one `add_episode_bulk`
   per tier, tiers sequential; link by position; supersede trailing; one
   `CURRENT_DEDUP_STATS` scope per batch with the article list in the log line.
   Reversible: the graph shape produced is identical to today's (same node/edge types,
   our `HAS_EPISODE`), so the per-chunk path can resume at any batch boundary.
3. **Orphan handling.** On any exception from the bulk call, delete `:Episodic` nodes in
   the group with `created_at` ≥ batch start and no incoming `HAS_EPISODE` — their uuids
   are minted inside the call and are not returned on failure. Same justification as the
   merge pass's §4.7: these nodes record that a batch failed, not anything about the
   corpus. Without this step, orphans accumulate and are **not reversible** by any
   existing job.
4. **Worker semantics.** `run_worker_once` claims a batch (`semantic_worker.py:36`) and
   completes jobs individually (`:62`). Batch-level complete/fail with the existing
   per-job backoff applied to every job in a failed batch; `max_attempts` becomes a
   batch property in effect. Warm-up predicate at batch grain only if batches are ever
   run concurrently — otherwise leave `W` in place and inert.
5. **Merge-pass runbook.** Extend the "never `--apply` during ingest" rule to name the
   batch-wide window explicitly.

What breaks if the order is violated: step 2 before step 1 writes nondeterministic
`invalid_at`; step 2 before step 3 accumulates orphans on every failure; step 2 before
step 0 risks a future `saga` experiment writing `HAS_EPISODE` edges our sweep cannot see.

---

## 11. What I could not determine without running something

- Throughput of sequential bulk batches versus N=4 per-chunk (§6). Needs experiment 2.
- The strong/cheap article share in the pilot set (not recorded anywhere I can read).
- Whether OpenRouter's providers honour `maxItems` on the duplicated in-batch candidate
  lists as they did on the single-range prompt — the prompt shape differs.

---

## 12. Open questions for the user

1. Is the operational cost of the merge pass (quiet windows, `theme-build` refusing on
   non-zero duplicates) acceptable at corpus scale? If not, that — not throughput — is the
   argument that would reopen bulk.
2. Should `MAX_COROUTINES` in `.env` be reconciled with the fact that extraction
   concurrency is governed by `SEMAPHORE_LIMIT` (§5, last row)? Unrelated to bulk, found
   while checking it.
3. Is a fifth graphiti extension (one-range in-batch dedup, §10 step 1) within the
   "pin and patch by name, never fork" budget, or is that the point at which the patch
   surface is too large?
