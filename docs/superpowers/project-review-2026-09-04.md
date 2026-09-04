# Whole-project review — 2026-09-04

**Scope.** Read-only review of the temporal GraphRAG system against its own design
baseline (`graphrag-docextractor-plan.md`, 2026-07-12) and the five invariants in
`CLAUDE.md`. Evidence: the code on `main` at `01f75ab`, the docs tree, the git
history, the live Neo4j 2026.07.1 graph at `alpcirag01`, and the graph-sync
Postgres state store on `localhost:5433`. Nothing was modified; no LLM tokens were
spent. All line numbers are as of `01f75ab`.

**Live-graph snapshot used throughout** (read-only Cypher, 2026-09-04):
593 `Article` (2 sources: AWS 148, Microsoft 445) · 38 articles with episodes ·
247 `Episodic` · 447 `Entity` (70 with no type label) · 1,300 `RELATES_TO` ·
465 facts with `valid_at`, 43 with `invalid_at` · **0 `Community`** · **0 `SAME_AS`** ·
one `group_id` (`backup-docs`) everywhere · no vector index · no APOC · GDS 2026.07.0.

---

## 1. Verdict in one paragraph

The build is disciplined and the two invariants that make the system worth
building — deterministic structural writes and traversal-only citations — hold in
the code and in the live graph. What has drifted is the **temporal** half of the
design: the provenance re-key severs the citation chain for superseded content,
two supersession mechanisms contradict each other, "exclude superseded from
retrieval" was never implemented, and graphiti itself is producing phantom change
events on a corpus that has never been updated. Those are silent-wrong today and
would make `/timeline` — the mode the plan calls "the whole purpose" — the least
trustworthy answer the system gives. Separately, the operational path the plan
describes (webhook → cursor → queue → worker → nightly theme-build) exists as code
but has **never run as a system**: the queue is armed with 593 pending jobs, the
worker has never claimed one, no webhook has ever been delivered, and the
community layer on the new instance is empty, which quietly reduces three of the
four retrieval modes to local search. The cost model is off by roughly 15–25x for
reasons the plan could not have known and one arithmetic error in a July report
made it look fine.

---

## 2. Design-goal alignment

### 2.1 The five invariants, verified against code and the live graph

| # | Invariant | Code | Live graph | Verdict |
|---|---|---|---|---|
| 1 | Structural metadata written deterministically, never via LLM | `graph_sync/sync_core.py` has no LLM or graphiti collaborator; `neo4j_repo.py:17-25` is plain `MERGE`. Only `graph_sync/cli.py:81-105` (worker wiring) imports graphiti. | 593 articles, 728 chapters, all with `content_hash`; 0 episodes without an `Article` | **Holds** |
| 2 | Citations are a graph traversal; the LLM never writes a URL | `provenance.py:33-50` (`resolve_citations`) is pure Cypher; `synthesize.py:14,33` strips any URL; global/drift/router reuse `_finalize_answer` + `_build_citations`; `router.py:1-6` authors nothing | `facts_with_no_resolvable_article = 0`; every fact has ≥1 episode | **Holds today** — but see §2.2 for how it degrades after the first article update |
| 3 | Updates append; superseded/removed are flags; never delete history | No `remove_episode` anywhere in `src/`; tombstone = `SET e.removed=true` (`ingest_driver.py:102-108`); sweep only sets `invalid_at` (`staleness_sweep.py:61`) | 0 superseded, 0 removed episodes — the update path has never run on this instance | **Holds in letter, drifts in substance** — §2.2 |
| 4 | One corpus-wide `group_id` | `config.py:49` default; `graphiti_client.py:270` passes `s.group_id`; `build_cheap_graphiti` keeps it | 447/447 entities, 247/247 episodes, 1300/1300 facts in `backup-docs` | **Holds**. (The compat harness writes a second group `compat-check` to the *production* DB and tears it down — acceptable, but it is the only other writer.) |
| 5 | Graphiti schema library-owned; link via `SAME_AS`, never merge | `reconcile.py` MERGEs `SAME_AS` only; `graph_cleanup.retype_region_entities` relabels *our own* custom labels on graphiti nodes (defensible) | **0 `SAME_AS` edges** — `reconcile` has not been run on the new graph | **Holds, but the bridge the plan wanted (§4.3) is absent** |

Two schema-level hazards the plan did not anticipate, both latent rather than broken:

- **Label collision.** The ontology's `Vendor`/`Product` entity types (`ontology.py:6,9`)
  become node labels on graphiti `:Entity` nodes, colliding with the structural
  `:Vendor`/`:Product`. `MATCH (v:Vendor) RETURN count(*)` returns 22 on the live
  graph (2 structural + 20 entities: "AWS CloudFormation", "MABS", "Resiliency"…).
  Every production query I found is anchored through `HAS_PRODUCT` or `:Entity`,
  so nothing is wrong today, but the `vendor_id` uniqueness constraint now spans
  both populations and any future monitoring/count query will be silently wrong.
- **graphiti 0.30.x now writes `HAS_EPISODE` itself** (`Saga-[:HAS_EPISODE]->Episodic`;
  `graphiti_core/graphiti.py:766`, `edges.py:719`, plus the `has_episode_uuid`/
  `has_episode_group_id` indexes visible in `SHOW INDEXES`). The `CLAUDE.md`
  invariant "graph-sync is the only writer of `HAS_EPISODE` edges" is no longer
  true by construction. All of *our* patterns are `(:Article)-[:HAS_EPISODE]->`
  anchored (`provenance.py:12,21,37,58`, `staleness_sweep.py:53`, `search.py:13`),
  so this is safe as long as nobody writes an unanchored `()-[:HAS_EPISODE]->()`.
  The invariant should be restated as "the only writer of *Article*→Episodic
  `HAS_EPISODE`" and a guard test added.

### 2.2 Where the temporal design actually drifted (the important part)

The plan's §4.4/§4.5/§6.4 describe one coherent policy: new episodes are appended,
old ones are *flagged* `superseded`, superseded episodes are *excluded from default
retrieval but kept for `/timeline`*, and "the superseded article version's
`source_url` still resolves — cite both old and new" (plan line 234). The
implementation has four pieces that each made a locally reasonable choice and
together do not implement that policy:

1. **`Provenance.link` deletes the old provenance edge.**
   `provenance.py:23-25`: `OPTIONAL MATCH (a)-[old:HAS_EPISODE {chunk_index:$i}]->(oldE) … SET oldE.superseded = true DELETE old`.
   The episode node survives (`test_provenance_rekey.py:69-81` asserts exactly that)
   but its link to the article is gone. Consequence: `resolve_citations` for any
   fact whose only support is a superseded episode returns `sources: []`.
   `/timeline` will list the historical fact with `status: "superseded"` and **no
   citation** — the opposite of plan §6.4. This is the single largest drift from the
   design and it is invisible on the current graph only because no article has
   been updated yet.

2. **The sweep is built on top of that deletion.** `staleness_sweep.py:11-19` says
   in so many words that liveness is keyed on `HAS_EPISODE` linkage "NEVER on the
   `superseded` flag" *because* `link` deletes the edge. So the citation break is
   load-bearing for the weekly expiry job; fixing (1) requires touching the sweep.

3. **`_supersede_trailing_episodes` does the opposite.** `ingest_driver.py:110-114`
   flags trailing chunks `superseded=true` but *keeps* their `HAS_EPISODE` edge.
   For the sweep those episodes are alive, so when an article *shrinks* (5 chunks
   → 3), every fact solely supported by chunks 3–4 stays `invalid_at IS NULL`,
   is returned by `/search/local` as current, and resolves to the article URL as
   if the page still said it. That is precisely the "silent doc deletion" case the
   sweep exists to catch. `test_staleness_sweep.py:42` ("superseded-but-linked
   episode keeps the fact alive") locks this behaviour in as intended.

4. **Nothing excludes superseded episodes from retrieval.** `grep superseded src/`
   finds no filter in `search.py`, `timeline.py`, `drift.py` or `global_search.py`;
   `search_local` filters only `invalid_at` (`search.py:37-38`). Between an
   article update and the next weekly sweep, facts from the old version are served
   as current with empty `sources`.

Smaller temporal drifts: the sweep stamps `invalid_at = datetime()` (sweep time)
rather than the tombstone's `removed_at`/`last_updated_at` the plan specified
(§4.4, line 167); `freshness.graph_cursor_time` is `max(Episodic.created_at)` —
ingestion wall-clock — not the delta-feed cursor the plan wanted stamped
(`freshness.py:25`, plan §5.1 step 5); `HAS_EPISODE.heading_path` is the
single-level chapter title (`ingest_driver.py:134-139`), not the section path the
plan wanted for section-level citations (§3.3, line 122).

### 2.3 A temporal-correctness risk that is not a drift: graphiti's false invalidations

On a corpus that has **never been updated**, 43 of 1,300 facts (3.3%) carry
`invalid_at`, all set by graphiti's contradiction step during the fresh bootstrap.
35 of the 43 have a still-valid sibling fact between the same two entities.
Samples:

```
AWS Backup —Supports→ Amazon FSx for NetApp ONTAP   valid 2026-08-17  invalid 2026-08-17T23:53
AWS Backup —Provides→ cold storage tier (DynamoDB …) valid 2021-11-01  invalid 2026-08-17
AWS Backup —Limits→ continuous backups (PITR only …) valid 2026-08-17  invalid 2026-08-17T23:53
```

These are rewordings across chunks, not contradictions. Two consequences:
`/search/local` hides them (correct facts silently missing), and `/timeline`
presents them as change events that never happened. The router-eval report's
timeline grounding precision of **0.2** (`router-eval-report.md:11-14`) is
consistent with this. The plan's Phase-1 gate "temporal correctness on a handful of
known doc changes" (§10) was never run; this is what it would have caught.

### 2.4 Roadmap promises that quietly did not happen

- **Phase 1 "structural layer complete (all vendors — it's free)"** (plan line 347):
  2 of 280 sources, 593 of ~105k articles are in the structural layer. The
  structural bootstrap is genuinely free and would give vendor scoping and
  monitoring reconciliation across the whole corpus today.
- **Phase 1 exit "incremental deltas flow within minutes of extraction"**: never
  demonstrated. Postgres: `sync_cursor` set once (2026-09-03 04:22), 0
  `webhook_delivery` rows, 0 `source_debounce` rows, `token_ledger` empty,
  `semantic_jobs` = 593 `pending`/`bootstrap`, 0 done. The 38-article pilot was
  ingested through `graph_extract.cli ingest --source-id … --limit 19` (the
  ingested set is exactly the first 19 by `sort_order` per source), bypassing the
  queue entirely. graph-sync has never run as a service; nothing is deployed.
- **Phase 1 exit "go/no-go on full-corpus budget"** was taken on a wrong number — §4.
- **Phase 3 "scheduled refresh / nightly"**: the incremental refresh is built and
  demonstrated, but there is no scheduler anywhere and the community layer on the
  live instance is empty.
- **§6.4 timeline "group into change events … enrich with DocExtractor
  versions/diff"**: `/timeline` is a validity-sorted local search
  (`timeline.py:50-70`); no change-event grouping, no versions endpoint.
- **§10 golden set "60–100 questions curated with a domain expert"**: 29
  questions, self-authored. **§10 monitoring**: none. **§7.3 answer-api enforces
  auth (defense in depth)**: none (fine until Phase 5, but note it).
- **§3.3 chunking "≤1,500–2,000 tokens, heading-boundary packing, section path"**:
  replaced by chonkie neural chunking + `_merge_tiny` (`episode_builder.py:100-115`);
  live episodes average **302 tokens** (p50 235, max 1,451). This choice drives
  the cost finding in §4.

---

## 3. Loose ends not in the backlog

Ordered by how much they matter.

1. **The semantic queue is armed and the budget default makes it useless for a
   bootstrap.** 593 jobs are `pending` in the `bootstrap` lane. `run_worker_once`
   claims bootstrap jobs only while `today_token_total() < budget`
   (`semantic_worker.py:20`), and `semantic_daily_token_budget` defaults to
   5,000,000 (`graph_sync/config.py:19`). At the measured ~200k tokens/article that
   is ~25 articles/day: the 555 not-yet-ingested pilot-source articles take three
   weeks, the full corpus ~11 years. Conversely anyone who runs
   `python -m graph_sync.cli worker` today starts a ~100M-token extraction with
   no further confirmation. Decide what those 593 rows are for, or clear them.

2. **Global and DRIFT are silently dead; the router's default mode is DRIFT.**
   `Community` count is 0. `drift_search` degrades to `answer_local` with
   `degraded: "no-primer-communities"` (`drift.py:175-179`); `global_search`
   returns its refusal and the router falls back to local (`router.py:159-164`).
   With `router_default_mode = "drift"` (`config.py:105`) every un-heuristic
   question is therefore local search wearing a drift label. This is observable
   per response (`routing.degraded`) but nothing aggregates it. `theme-build --full`
   has not been run on the new instance and the router eval (backlog #3) is
   meaningless until it has.

3. **The July cost report contains a 5x arithmetic error** that made the plan's
   prior look confirmed — §4.

4. **Dead and duplicated code** (from the module audit): `answer_api/eval_answer.py`
   and `answer_api/eval_golden.py` (superseded by `eval_router.py`, no importer, no
   test); `graph_extract/probe.py` + the `probe` CLI command (a closed July
   gpt-oss investigation, `config.py:25-31`); `episode_builder.needs_presplit`
   (tested, never called); `scripts/run_pilot.py` builds a strong-only
   `IngestDriver` (`cheap_tier=None`) and duplicates `cli ingest`;
   `scripts/reset_semantic_layer.py` DETACH-DELETEs the semantic layer; six
   `eval *`/`quality-*` CLI commands with no test and no runbook mention
   (`graph_extract/cli.py:260-341`); `answer_api/app.py:191` re-exports the
   third-party `Graphiti` class for no reason; `Neo4jRepo.delete_source_articles`
   (`neo4j_repo.py:257-268`) is a `DETACH DELETE` test helper living on the
   production repo class.

5. **Config no longer matches reality.** `.env` says the cheap tier is "parked …
   strong-only" while setting `EXTRACTION_ROUTING=true` and a cheap key, and its
   comments describe `ling-3.0-flash` while `CHEAP_LLM_MODEL=upstage/solar-pro4`.
   `.env.example` (last touched 2026-07-14) lacks every `LLM_*`, `CHEAP_LLM_*`,
   `EMBED_*`, `CHONKIE_*`, `REPORT_LLM_*`, `MAP_LLM_*` key and leaves `JUDGE_*`
   commented out — but `synthesize.py:59-63` raises at answer-api startup without
   `JUDGE_BASE_URL`, so a fresh clone following the example cannot boot the API.
   `config.py:53-58` justifies `generic_json_schema` by gpt-oss/llama-server
   behaviour that no longer applies. `docker-compose.yml:3` pins `neo4j:5.26`;
   `CLAUDE.md:15` still says the `/answer` router is "not yet built" and `:24`
   says compose is 5.26; `graph_sync/config.py` re-declares `docext_*`/`neo4j_*`
   independently of `ExtractSettings` so the worker loads two settings objects.

6. **Tests that assert less than they appear to.** No test guards invariant #1 or
   #4 explicitly (both hold by construction/default only).
   `tests/unit/test_compat_checks.py:140-149` and `test_llm_penalties.py:103-114`
   assert on `inspect.getsource()` substrings. `tests/unit/test_answer_api_app.py`
   monkeypatches every retrieval function and asserts the stub's own value comes
   back (wiring, not retrieval). `test_compat_harness.py:57` accepts any verdict.
   Most importantly, **`tests/integration/test_ingest_driver.py:28-31` runs
   `DETACH DELETE` on episodes of the shared live graph as test setup** — the one
   place in the repo that does what invariant #3 forbids. GDS is present in no
   automated lane (`conftest.py` uses the bare `neo4j:2026.07.1-community`
   image; `detect_communities` is monkeypatched everywhere), so Leiden on the
   production GDS is exercised only by the manual compat run.

7. **Two more Python-side scaling landmines next to the vector-scan one.**
   `_vendor_episode_uuids` (`search.py:9-16`) collects *every* episode uuid of a
   vendor into a Python set per request; `shortlist_communities`
   (`global_search.py:62-75`) loads every community embedding at a level and does
   cosine in pure Python. Both are fine at pilot scale and wrong at corpus scale.

8. **The `judge` tier is the synthesis tier, the map tier, and the report tier.**
   `synthesize.py:56-66`, `global_search.py:97-103`, `theme_builder/report.py:47-54`
   all fall back to `judge_*`; `eval_router.py:69` passes the synthesis client
   `sc` to `_faithfulness_judge`. The faithfulness numbers in
   `router-eval-report.md` are GLM-5.2 grading GLM-5.2. There is no `synth_llm_*`
   setting to separate them.

9. **`is_dense_matrix` routing has no observability.** `IngestArticleResult.tier`
   is set (`ingest_driver.py:75`) but never persisted to the graph, so the live
   graph cannot say which tier extracted which episode; the type-hygiene
   attribution by model (backlog #1) cannot be reproduced from the graph.

---

## 4. The cost model: who is right

**The plan's reasoning** (§9, line 309): 260M content tokens → 150–200k episodes
(i.e. ~1.3–1.7k tokens each) × "3–6× extraction overhead" → 0.8–1.5B tokens.

**What the project measured, in its own docs:** `slice-2a-viability.md:29-31`
reports 8.46M extraction tokens for the 39-article pilot, "≈ per article ~217K
tokens", and then extrapolates to "~4B tokens" for 105k articles. **105,000 × 217k
= 22.8B, not 4B.** The July go/no-go ("within the plan's budget prior") rested on
that slip. Your 29–35k tokens/episode × the live 6.5 episodes/article ≈ 190–230k
tokens/article — the same number as July, measured independently. So your ~20B is
right and the plan is off by ~15–25x.

**Why the plan was wrong** (and it could not easily have known): graphiti's cost is
dominated by *per-call fixed overhead*, not content. Each episode makes ~5.4 LLM
calls (`prompt-caching-findings.md:36`), each carrying the system prompt, the
ontology + extraction instructions (~1,536 tokens, `prompt-caching-findings.md:22`),
the previous-episodes context, and — for dedup — the candidate lists. That
overhead is roughly constant per episode. The plan assumed a 3–6× multiplier on
content, which is only true when episodes are 1.5–2k tokens as §3.3 intended. Live
episodes are **302 tokens on average** (chonkie + `min_chunk_tokens=128`), so the
multiplier is ~100×.

**Implication, which is the actionable part:** cost is approximately linear in
*episode count*, not content. Raising episode size toward the plan's target
(`min_chunk_tokens` 128 → ~800–1,200 with `max_chunk_tokens` 1,800 unchanged)
should cut episode count ~3–4x and tokens nearly proportionally, at some cost in
per-chunk extraction recall that a 20-article A/B would measure in an afternoon.
No such experiment has been run; every quality iteration to date has kept the
small chunks. The dollar figure at gpt-5-mini/solar-pro4 input pricing is
uncomfortable but not fatal (order $5–10k); the **wall-clock** is the real blocker:
~680k episodes × ~27 s/episode (`slice-2a-viability.md:18`) is months for one
worker process, since `ingest_article` processes episodes sequentially.

---

## 5. Corrections to the backlog

- **#1 Type hygiene — overstated for this graph, and mis-framed.** On the live
  graph 61/1,300 (4.7%) facts carry a name outside the eight declared types:
  23 casing variants (`LIMITS` 12, `PROVIDES` 5, `REQUIRES` 3, `INTEGRATES_WITH` 2,
  `AVAILABLEIN` 1) and 38 invented (`HasColumn` 7, `INCLUDES_COLUMN` 5,
  `HAS_OPTION` 3 …). Not 12–28%. The deeper number is that **720 of the 1,239
  correctly-named facts (58%) sit on an endpoint-type pair that `EDGE_TYPE_MAP`
  does not sanction** (e.g. `Product —Supports→ Concept`) — graphiti offers types
  per pair but does not enforce the pair. And 70/447 entities (16%) have no type
  label at all. But: **no query in `src/` filters by edge type** (the only
  `SearchFilters`/name-filter is `graph_cleanup.py:52` on entity names). The
  impact today is nil; the casing subset is a 20-line Cypher normaliser. I would
  rank this last, not first.
- **#4 "The GLM tier is unset" — wrong.** `.env` sets `JUDGE_BASE_URL`,
  `JUDGE_MODEL=glm-5.2:cloud`, `JUDGE_API_KEY`; `report_llm_*` and `map_llm_*`
  fall back to it by design; answer-api would not start otherwise. What is true is
  the second half: there is one tier doing four jobs, and the eval judge grades its
  own synthesis.
- **#2 Vector scan — right, with a cheaper remediation than a fork.** graphiti
  0.30.1 still scores with `vector.similarity.cosine` (`graph_queries.py:163`) and
  creates no vector index, but its driver now exposes a pluggable
  `SearchInterface` (`driver/search_interface/search_interface.py:22,59,111,232`;
  `driver.py:98`). An index-backed `edge_similarity_search`/`node_similarity_search`
  can be supplied without patching library internals — the same pattern as the
  existing client wrappers in `graphiti_client.py`.
- **#5 `max_tokens=8`** — confirmed (`router.py:72`). Note the failure is already
  observable: an empty reply yields `routing.via = "default"` in the envelope. The
  0.93 routing accuracy was measured with ling; with `solar-pro4` it is unmeasured.
- **#6 90 s timeout** — confirmed at `graphiti_client.py:150,154,204,256`; add
  chonkie `timeout=120` (`ingest_driver.py:80`) and docext `timeout=300`
  (`docext/client.py:12`) to the same fix.
- **#7 Cost** — you are right; see §4 for the arithmetic error and the lever.
- **#11 Incremental never exercised** — confirmed and broader: neither the
  incremental *nor the update/supersede* path has ever run on this instance, and
  the update path is where the §2.2 defects live.
- **#13 Unpushed commits** — the only copy of 269 commits lives on the host that
  just lost a data volume. The leaked key at `a674116` is already on `origin/main`,
  so pushing does not leak it further, but rewriting history before pushing does
  require the rotation you deferred. Until then, a `git bundle` to a second
  location costs nothing.

---

## 6. Risk assessment — what is most likely already wrong or about to break

1. **Temporal answers are silently wrong** (§2.2, §2.3). Not a future risk: the 43
   phantom invalidations exist now, and the first real article update will produce
   uncited history and uncatchable stale facts. The flagship use case is the least
   protected one.
2. **The operational system has never existed.** Every service is a CLI that has
   been run by hand; the queue/budget/webhook/scheduler path is untested in anger,
   the budget default is wrong by orders of magnitude for a bootstrap, and the
   community layer is empty on the instance retrieval runs against.
3. **Broad ingestion is blocked twice**: by throughput/cost (§4) before it is
   blocked by retrieval scaling (backlog #2). Fixing the vector scan first would
   optimise a system that cannot be filled.
4. **The evaluation signal is untrustworthy**: judge = synthesis model; 29
   self-authored questions; baseline corpus gone; global/drift unexercisable.
5. **Single copy of the code, single copy of the expensive data.** The semantic
   layer cost real tokens to build and has no backup; the repo has no remote copy
   of 269 commits.
6. **Config/docs drift** means a new machine or a new person cannot reproduce the
   stack from the repo (`.env.example`, compose, `CLAUDE.md`).

---

## 7. Recommended course of action

Sequenced; each step is a precondition for the next.

**0. Today, before any code (≤1 h).** `git bundle create` the repo to a second
location. Rotate the DocExtractor read key (you already own the admin key), rewrite
`a674116` out of history, push. This is cheap insurance and it unblocks everything
that follows being pushed.

**1. First engineering work: make the temporal policy coherent (no LLM tokens).**
Change `Provenance.link` to *flag* the old edge/episode instead of deleting the
edge; make `_supersede_trailing_episodes`, the sweep, `search_local`, `timeline`
and `resolve_citations` all key on the same `superseded`/`removed` flags (sweep
counts an episode dead iff `removed OR superseded`; default retrieval excludes
both; `/timeline` includes both *with* citations; sweep stamps `invalid_at` from
`removed_at`/`last_updated_at` when known). Then exercise an update end-to-end on
one pilot article via the real queue (`sync-once` → `worker`, a few thousand
tokens) and add an integration test that asserts a superseded fact still resolves
to a URL. This is the smallest change that restores the plan's core promise, and
it also covers most of backlog #11. Do it before anything else because every other
step (theme-build, eval, ingestion) builds on facts whose validity this governs.
In the same pass, quantify the false-invalidation rate with the judge on the 43
facts and decide whether to post-process (e.g. un-invalidate a fact whose sibling
on the same endpoints is valid and whose text is a paraphrase) or accept it as a
known error bar on `/timeline`.

**2. Restore the community layer and separate the judge (small tokens).**
`theme-build --full` on the new instance (~50–70 communities × ~12k context ≈ 1M
tokens). Add `synth_llm_*` settings so synthesis and the judge can differ, point
the judge at a different model, then re-run the router golden set (backlog #3).
Run `reconcile` and `sweep` once so the runbook jobs have been seen to work on
2026.07.

**3. Run the chunk-size experiment before deciding anything about scale.** 20
articles, `min_chunk_tokens` 128 vs ~1,000, measure tokens/article, facts/article,
and the type-precision judge. If it cuts tokens 3x at acceptable recall, the
bootstrap budget conversation changes shape. Fix the daily-budget default and
decide the fate of the 593 pending jobs at the same time.

**4. Only then, the vector index** via graphiti's `SearchInterface`, plus the two
Python-side scans in §3.7. Gate the first multi-vendor bootstrap on it.

**5. One hygiene PR** (mechanical, an afternoon): `CLAUDE.md`, compose to
2026.07.1, regenerate `.env.example` from both settings classes, add
`theme_builder`/`compat` to the wheel, delete the dead modules in §3.4, move
`delete_source_articles` and the live `DETACH DELETE` out of production paths,
make the timeouts settings, raise `router.py:72`'s cap and log
`routing.via/fallback_from/degraded` counts (that is the cheapest possible start
on backlog #12), persist `tier` on `HAS_EPISODE`, add the casing normaliser and a
guard test for the `Article`-anchored `HAS_EPISODE` invariant.

**Deliberately not yet:** Phase 5 (MCP/Copilot) — there is nothing trustworthy to
expose until steps 1–2 land; a full-corpus or multi-vendor bootstrap — blocked by
§4 and step 4; monitoring dashboards beyond the counters in step 5 — premature
until something runs as a service; the `EDGE_TYPE_MAP` enforcement question and
ontology v2 — nothing consumes edge types yet; the compat-harness coverage gaps
(backlog #8) — the harness already proved what it needed to and the remaining
paths are better covered by the integration tests step 1 adds.

**One thing I would do that is not on your list:** bootstrap the *structural* layer
for all 280 sources now. It is free (no LLM), the plan called for it in Phase 1,
it makes vendor scoping and the dashboard-reconciliation check meaningful, and it
exercises the delta feed at realistic volume before the semantic path depends on
it.
