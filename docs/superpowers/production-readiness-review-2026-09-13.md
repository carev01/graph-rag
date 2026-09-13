# Production readiness review — 2026-09-13

**Scope.** End-to-end review of the graph-rag project against its original goals
(`graphrag-docextractor-plan.md`), and the plan from here to a production run. Brief:
`.superpowers/sdd/production-review-brief.md`. Analysis only — nothing was implemented,
no paid run was made, no test suite was run.

**How evidence is marked.** Every claim carries one of:
`[measured]` a number produced today or on record with a reproducible source;
`[code]` read directly from source at the stated path;
`[backlog]` the backlog's own evidence, spot-checked where the plan depends on it;
`[inference]` reasoning from the above — plausible, unmeasured, and this project's
history says such reasoning is wrong often enough to be labelled.

Probes run today (all read-only): a Cypher census of the live graph (`alpcirag01`,
Neo4j 2026.07.1), a Postgres census of the sync state, a `GET /api/dashboard/sources`
pull from DocExtractor, and two scan-cost timings on the live graph
(`.superpowers`-style scripts left in the session scratchpad, not the repo).

---

## 0. Corrections to the state the brief handed me

The brief said to distrust the documentation. Four things in the brief itself, or in the
documents it cites, do not survive a look at the live system.

1. **83 articles are ingested, not 593.** `[measured]` 593 is the count of `:Article`
   nodes (the structural layer); only **83** have a `HAS_EPISODE` edge (55 Microsoft,
   28 AWS). 655 episodes, 3,469 facts, 1,011 entities. The 593 figure is repeated in the
   brief, `scale-wall-2026-09-12.md` and BACKLOG 11.

   This matters because `scale-wall-2026-09-12.md` divides 3,469 facts by 593 to get
   "5.8 facts/article" and projects 610k facts at corpus scale. The real ratio is
   **41.8 facts per ingested article** and **7.9 episodes per article**. Every projection
   built on 5.8 is ~7x too optimistic: the ingest-scan crossover the document places at
   "~3,900 articles, 4% of the corpus" lands at ~550 ingested articles, and the full
   corpus is ~5.3M facts, not 610k (the July sizing doc's 3–5M was closer).

2. **The "282 never-pushed commits" (BACKLOG 22/23) are stale.** `[measured]`
   `git rev-list --count origin/main..HEAD` = 2; `origin/main` is dated 2026-09-13. The
   single-copy risk described there no longer exists. The *leaked key* half of item 22
   (commit `a674116` on `origin/main`) is still real.

3. **The temporal change is only partly committed.** `[code]` `git status` shows
   `ingest_driver.py`, `provenance.py`, `staleness_sweep.py` and
   `test_temporal_coherence.py` modified — the `superseded_at` half of "date expiries by
   when the source died". On the working tree: the five hermetic test files touching this
   path pass (56 tests), `ruff check src tests` and `mypy src` are clean `[measured]`.
   HEAD does not contain it; nothing in the graph reflects any of it (see §1.2).

4. **The corpus is 126,405 articles across 288 sources and 40 vendors, and it is
   concentrated.** `[measured]` from DocExtractor's dashboard today:

   | vendor | articles | share | | vendor | articles |
   |---|---:|---:|---|---|---:|
   | Commvault | 42,384 | 33.5% | | Rubrik | 4,456 |
   | Veeam | 20,174 | 16.0% | | Acronis | 2,026 |
   | Cohesity | 15,771 | 12.5% | | Zerto | 1,897 |
   | Dell | 10,389 | 8.2% | | Microsoft | 472 |
   | IBM | 8,982 | 7.1% | | AWS | 148 |
   | Arcserve | 7,793 | 6.2% | | (34 more) | 11,913 |

   The top six vendors are 83% of the corpus. The pilot vendors (AWS + Microsoft) are
   0.5%. "Vendor priority order" (plan §13.3) is therefore not a formality: choosing
   Veeam is a 20k-article commitment, choosing Commvault a 42k one.

---

## 1. Where we stand

### 1.1 Against the five objectives of plan §1

**1. Frequent incremental updates — built, never operated.** `[measured]` Postgres:
`sync_cursor` set once (2026-09-03 04:22), `bootstrap_progress` two shards complete,
`webhook_delivery` 0 rows, `token_ledger` empty, `semantic_jobs` = 593 `pending` in the
`bootstrap` lane, 0 `done`. The 83 ingested articles went through
`graph_extract.cli ingest --source-id`, bypassing the queue. `graph-sync` has never run
as a service on this instance, and the incremental/webhook path has never processed a
real delta. This was already in the September review (§2.4) and is unchanged.

Since the user re-extracts sources roughly quarterly, the plan's "deltas flow within
minutes" goal is not a production requirement. A scheduled `sync-once` + worker is.

**2. Temporal awareness — the flagship promise, and the one with no evidence.**

- `[measured]` 3,469 facts; 805 (23%) carry `valid_at`; 140 carry `invalid_at`, all
  140 with `expired_at` set — i.e. all contradiction-driven, none swept, none from an
  extracted end date. Of the dated facts, **450 share one crawl minute and 238 another**
  (796 of 805 are dated 2026; nine carry in-world dates from 2016–2023). The largest
  single `valid_at` value is shared by 172 facts. This is the crawl-order axis the
  DocExtractor exchange exists to replace.
- `[measured]` `HAS_EPISODE.superseded` = 0 on all 655 links; `Article.removed` = 0;
  `Episodic.removed` = 0. **The graph has never seen an update or a removal.** There is
  no temporal history in it at all — the only "change events" are the 140 invalidations,
  which hand inspection found to be refinements and near-duplicates
  (`invalidation-measurement-2026-09-12.md`, 6 of 6, n small) and which are unauditable
  (median 21 candidate invalidators per victim).
- `[backlog]` The plan's Phase 1 gate "temporal correctness on a handful of known doc
  changes" (§10) was never run. `/timeline` has never displayed a real change.
- **What the eval numbers do and do not say.** The router eval's timeline mode scores
  faithfulness 5.0 and grounding 0.8 (`router-eval-report.md`). Faithfulness measures
  whether the answer restates the cited facts; it cannot tell a genuine change from a
  crawl-order artefact, because the judge never sees the source. A 5.0 on four timeline
  questions is evidence the synthesis is honest about facts it was given, not that the
  facts describe history. `[inference]` from how `eval_router` is built, not measured.
- **What changed today, and what has not.** `[code]` `_reference_time`
  (`ingest_driver.py:56`) now orders by `content_changed_at` with a loud fallback;
  `map_content` persists the four new fields; the sweep dates expiries by `removed_at`
  and (uncommitted) `superseded_at`. `[measured]` Zero `:Article` nodes carry
  `content_changed_at` — the property key does not exist in the database. Nothing has
  been re-synced or re-ingested. **The entire semantic layer was built under the old
  axis.**
- **Same-pair invalidation is live and ungated.** `[code]` The contradiction gate
  (`contradiction_gate.py`) skips only the unfiltered search; `contradicted_facts`
  indices below `len(related_edges)` still reach `resolve_edge_contradictions`
  (BACKLOG 33, pinned by `test_same_pair_contradiction_stays_live_...`). The "Tier-1
  only" rule adopted in BACKLOG 30 (invalidate only when both articles are
  `content_changed_basis = exact`, and only on a `source_changed_at` change) is
  **adopted but not implemented**: `grep` finds no consumer of `content_changed_basis`
  or `source_changed_at` outside the models and the mapper. The counter that would size
  this path (`DedupIndexStats.contradicted_same_pair`) exists and is unmeasured.

**3. Cross-corpus thematic summarisation — built, measured on two vendors.**
`[measured]` 40 communities (19/12/9 by level), all verified, none stale or staged,
generated 2026-09-10. Global faithfulness 4.7–5.0 on 10 questions
(`router-eval-report.md`). `[backlog]` The gain came from a structural fix (binding
claims to `[N] fact` lines, hand-audited 80% correct citations), not prompt tuning.
What is not established: any of this at more than two vendors. Cross-vendor entity
dedup — the reason for the single `group_id` — has never been exercised across vendors
with divergent vocabularies. `[measured]` `SAME_AS` edges = 0: `reconcile` has never
been run on this instance, so the structural↔semantic bridge (plan §4.3) is absent.

**4. URL-level citation — the strongest part of the system.** `[code]`
`provenance.py` resolves fact → episode → article → `source_url` deterministically;
the synthesis tiers emit only markers. `[measured]` grounding precision 0.92, ranges
0/29, unscored 0/29. The citation-integrity work of 2026-09-08→11 is the best-evidenced
slice in the repo.

**5. Copilot exposure — not started.** `[code]` `grep -ri 'mcp\|copilot\|oauth\|apim'
src` returns nothing; `answer_api/app.py` has no authentication dependency. Plan §7 is
entirely open, including the Power Platform DLP conversation that §12 Phase 0 said to
open first.

### 1.2 Claimed as working on evidence that would not survive scrutiny

- **"The scale wall is solved."** The un-indexed cosine scan was removed from the
  *ingest* path by suspending contradiction detection. It was not removed from the
  system. `[code]` `answer_api/search.py:27` → `graphiti._search` →
  `graphiti_core/search/search.py:286-290` → `edge_similarity_search` with no
  `edge_uuids` filter: **every `/search/local`, `/timeline` and DRIFT follow-up runs the
  full-corpus scan per query.** `[measured today]` scan cost on the live graph, lean
  projection in place: 211 ms @ 1,000 facts, 265 @ 2,000, 343 @ 3,469 — slope
  **0.053 ms/fact**, intercept ~160 ms (the 09-12 curve measured 0.0699 with the heavier
  projection; both linear). At 41.8 facts/article:

  | ingested | facts | one retrieval scan | DRIFT (primer + 4 follow-ups) |
  |---|---:|---:|---:|
  | 83 (today) | 3.5k | 0.34 s `[measured]` | ~2 s |
  | 593 (pilot complete) | 25k | ~1.5 s | ~8 s |
  | +Veeam (~21k) | ~870k | ~46 s | ~4 min |
  | top-6 vendors (~105k) | ~4.4M | ~4 min | ~20 min |

  Rows below the first are `[inference]` — linear extrapolation 3 orders of magnitude
  past the measured range, which the 09-12 synthetic curve supports to 43k facts and
  nothing supports beyond. The plan's target is local 3–8 s end-to-end (§6.1). The
  user's "quality outranks latency" ruling bounds *answers*, not a retrieval floor that
  grows without limit; and it is per query, so concurrent users compound it (the DB
  saturates at ~2x, `throughput-profile-2026-09-12.md` §2).

- **A second ingest-side scan nobody has measured.** `[code]` Entity dedup calls
  `node_similarity_search(..., SearchFilters(), ...)`
  (`graphiti_core/utils/maintenance/node_operations.py:439-445`), an unfiltered cosine
  over every `:Entity` in the group. The contradiction gate patches
  `edge_operations.search` only; this path is untouched and not in the backlog.
  `[measured today]` 171 ms @ 250 entities, 185 @ 500, 209 @ 1,011 — slope
  **0.050 ms/entity**, ~159 ms intercept. At the sizing doc's 200–400k entities that is
  10–20 s per candidate search, roughly a dozen per episode `[inference]`. It grows
  sublinearly (entities dedup) so it arrives later than the fact scan did, but it is the
  same shape and the same fix.

- **"8.10 → 5.52 s per dedup call, 1.47x."** Retracted by its own author in the
  CORRECTIONS section of `throughput-profile-2026-09-12.md`: run 2 was a different mix
  (68 leftover episodes over 50 partially-ingested articles), and "s per dedup call"
  charges all work to dedup. The only end-to-end figure to plan on is **~45 s/episode**,
  and it has one run behind it.

- **The cost model.** `[backlog]` BACKLOG 25's 5x error is real (105,000 × 217k ≠ 4B).
  The corrected "$5–10k, months of wall clock" in the September review §4 rests on
  gpt-5-mini pricing that `ling-production-readiness.md` itself labels *estimated*, and
  on the ling cheap tier, which is dead (`openrouter-extraction-tier-gotchas` memory).
  The current cheap tier is `upstage/solar-pro4` (`.env`); **its price appears nowhere
  in the repo**, and no run since the tier change has produced a $/article. Every cost
  figure in this document is therefore a gpt-5-mini ceiling.

- **"The same-pair counter rides on the next paid ingest."** True only if that ingest
  goes through `graph_extract.cli ingest`. `[code]` `run_worker_once`
  (`semantic_worker.py`) calls `ingest.ingest_article(...)` and discards the
  `IngestArticleResult`; the per-article WARNING fires only when `invalid_calls > 0`, and
  a same-pair contradiction is an in-range index on an otherwise clean call. **Through
  the worker — the production path — the counter, the per-prompt timings and
  `reference_basis` are all invisible.**

- **"Structural layer complete (all vendors — it's free)."** `[measured]` 2 sources of
  288, 2 structural `:Vendor` nodes. The 42 returned by `MATCH (v:Vendor)` are 2
  structural + 40 `:Entity:Vendor` — the label collision of BACKLOG 17, still present.

- **The 29-question golden set.** Self-authored, AWS + Azure only, versus the plan's
  60–100 with a domain expert. Every quality number on record is measured on it.

### 1.3 Defects found in today's reading, not yet in the backlog

- **`superseded_at` is overwritten on every re-supersession.** `[code]`
  `Provenance.link` (`provenance.py:36-38`) sets `old.superseded_at = $sat` on *every*
  old edge at that `chunk_index`, including ones already superseded; and
  `_supersede_trailing_episodes` (`ingest_driver.py:178-182`) re-stamps every trailing
  edge on every ingest of the article. A chunk that died at T2 is re-dated T3 when the
  article changes again at T3 without touching it. The sweep takes the *max* death, so
  the expiry lands at T3. Bounded, but it is the same class of defect this change set
  exists to remove. One-line fix: only stamp when `coalesce(old.superseded, false) =
  false`. Uncommitted code — cheap to fix before it lands.
- **A bootstrap replay will never write the new fields onto existing articles.** `[code]`
  `_apply_record` (`sync_core.py:71-73`) returns on `existing == rec.content_hash`
  *before* `apply_structural`. The 593 `:Article` nodes will get `content_changed_at`
  only when their content changes upstream. Ordering still works (the ingest path reads
  the fields from `GET /api/articles/{id}` at ingest time, `content_fetch.py`), but the
  persisted-for-display promise of reply-3 does not hold for anything already synced.
- **No death timestamp for episodes tombstoned through the worker.** `[code]`
  `tombstone_article_episodes` sets `e.removed = true` only; the sweep reads
  `Article.removed_at`, which `map_tombstone` supplies — so this path works only if the
  structural tombstone landed first. It does (`_apply_record` writes both), but nothing
  pins the order.
- **`CLIENT-USAGE-GUIDE.md` does not document the five new fields** — only `removed_at`
  appears. `CLAUDE.md` says to read the guide before touching ingestion; today it would
  mislead.
- **Chunk size as a cost lever was recommended on 2026-09-04 (review §4) and never
  reached the backlog.** `grep min_chunk BACKLOG.md` finds nothing. Live episodes average
  ~300 tokens (`min_chunk_tokens = 128`); the plan intended 1,500–2,000. Cost is roughly
  per-episode, so this is potentially a 3–4x lever with no measurement either way.

### 1.4 What is solid and should not be re-litigated

The provenance chain and citation integrity (§1.1 item 4); the reduce-step binding
(hand-audited); the answer-path client hardening (eval 1h44m → 26 min, 0 empty-content
retries); report preservation on both theme-build paths; the dedup-index guard and its
267/267 confirmation; the lean edge projection (2.3–2.7x on the searches, measured
identical rows); the contradiction gate's *scale* claim (the ingest scan is gone,
asserted with a driver-level query counter); and the DocExtractor timestamp contract,
which was negotiated with unusual care and is the right one.

---

## 2. What must be true before a production run

A production run means: a chosen vendor set fully ingested under a clean time axis, the
community layer rebuilt over it, quality re-measured on a golden set that covers those
vendors, and answers reachable by real users through Copilot.

### Must be true

| # | condition | why it is a must, not a nice | status |
|---|---|---|---|
| M1 | **The vendor set is chosen** and its size known | 20k vs 42k articles is a 4-month vs 6-month difference at current throughput (§3) | open — decision D1 |
| M2 | **The semantic layer is built on `content_changed_at`**, and the crawl-minute clustering is shown to be gone | The current graph is measured to be crawl-ordered; layering new facts on it recreates the two-kinds-of-date incoherence the exchange was about | code landed (partly uncommitted); graph untouched |
| M3 | **Retrieval does not scan the corpus per query** (index-backed `SearchInterface` for edge *and* node similarity, recall measured against brute force) | §1.2: ~46 s per local search at one added vendor, growing linearly; also fixes node dedup at ingest | not started; `driver.search_interface` hook identified in BACKLOG 8-orig |
| M4 | **Throughput is measured on the chosen set's shape** with the worker as the production path, reporting timings, dedup counters and `reference_basis` | The only end-to-end number is one run's ~45 s/episode; the plan's elapsed time is 10x uncertain without it | needs the worker reporting fix + one bounded paid run |
| M5 | **A decision on same-pair invalidation, taken on the counter** — gate on `basis = exact` + `source_changed_at`, or suppress until measured | It is the only ingest-time writer of `invalid_at` left, it is ungated by basis, and it is known to fire on refinements. Invariant #3 requires that whatever is chosen appends rather than deletes | counter built, number pending |
| M6 | **Neo4j backups exist and are tested** before and during the run | Single Community instance; host lost a volume once; the semantic layer is the expensive part and the community layer is rebuildable | no dump tooling in the repo |
| M7 | **The budget and queue are configured deliberately**: `semantic_daily_token_budget` (5M ≈ 23 articles/day at 217k), worker batch size, `--max-batches` | The defaults make a 20k-article run take ~2.4 years of budget days | config only |
| M8 | **The golden set covers the chosen vendors** (local + global at minimum) and **at least one real document change is checked through `/timeline`** | Every quality number is AWS/Azure; the temporal gate was never run | 29 questions, 2 vendors |
| M9 | **Phase 5 minimum**: MCP wrapper (streamable HTTP), authentication on answer-api, public exposure, Copilot Studio agent, DLP sign-off | The definition of a production run | 0 code |
| M10 | **`cleanup` + `sweep` run after each ingest batch, `reconcile` weekly** — scheduled, not remembered | The sweep is now the primary invalidation mechanism (`staleness_sweep.py:11-17`); between an update and the next sweep, dead facts are served as current (`search.py:38` filters `invalid_at` only) | runbook exists, no scheduler |

### Would be nice

Structural bootstrap of all 288 sources (free, gives vendor scoping and reconciliation
across the corpus — do it during Phase D, it is an hour); the chunk-size A/B (a cost
lever, unmeasured); `SAME_AS` reconciliation (weekly maintenance already covers it);
`.env.example` / `docker-compose.yml` / `CLAUDE.md` config drift (BACKLOG 14 — a fresh
clone cannot boot answer-api); the judge split (BACKLOG 2); type hygiene (18); timeline
change-event grouping and DocExtractor `versions/diff` enrichment (plan §6.4);
`freshness.graph_cursor_time` from the real cursor.

---

## 3. The sequenced plan

Throughput and cost assumptions used throughout, so they can be replaced by one number
each when measured: **7.9 episodes/article** `[measured]`, **45 s/episode** `[measured,
n=1 run]`, **41.8 facts/article** `[measured]`, **~26k tokens/episode at gpt-5-mini's
estimated $0.25/$2.00 per 1M → ~$0.0097/episode → ~$0.077/article** `[backlog,
July, estimated pricing]`. The cheap tier's real price is unknown; treat every $ as a
ceiling.

### Phase A — Land the time axis and prove it (no paid run except A4; 2–3 days)

- **A1.** Commit the working tree after fixing the `superseded_at` overwrite (§1.3).
- **A2.** Make the worker report what the CLI reports: aggregate `IngestArticleResult`
  (`dedup.summary()`, `reference_basis` counts, `timings.report()`) per `run_worker_once`
  and log it. Without this, M4 and M5 cannot be measured on the production path.
- **A3.** Give `:Article` its new fields: either a metadata-refresh mode in graph-sync
  that applies `SET a += {content_changed_at, content_changed_basis, source_changed_at,
  last_updated_at, last_updated_source}` regardless of hash, or accept that existing
  articles get them lazily. Also update `CLIENT-USAGE-GUIDE.md`.
- **A4 — decision D1: reset and re-ingest the pilot.** `scripts/reset_semantic_layer.py`
  deletes `:Episodic`, `:Entity`, `:Community` and keeps the structural layer. Then
  `graph_extract.cli ingest --source-id` for the two pilot sources.
  - Elapsed: 655 episodes × 45 s ≈ **8 h**; tokens ~17M; cost **≲ $10** at the
    gpt-5-mini ceiling.
  - What it is for: (i) the graph's `valid_at` distribution under the new axis — the
    proof promised to DocExtractor in reply-3 (does the 450-in-one-minute cluster
    disappear?); (ii) the first `contradicted_same_pair` number; (iii) the first
    per-prompt timing breakdown (BACKLOG 31); (iv) the first $/article on solar-pro4.
  - Why a reset is defensible against invariant #3: the invariant protects *history*, and
    the graph contains none (0 superseded, 0 removed). What it would discard is 140
    phantom, unauditable invalidations and 805 crawl-ordered dates. Rebuilding is
    cheaper than any repair, and the community layer is disposable by design.
  - The user must choose this; it is irreversible for the pilot graph. Take a
    `neo4j-admin database dump` first (M6) — that also creates the backup tooling.
- **A5.** `theme-build --full` (30 min – 6 h on record, `report.py` comment), then
  `answer_api.eval_router` (~26 min). This is the new baseline; the 5.0 on the old graph
  is not comparable to it and should not be defended.

### Phase B — Take retrieval and node dedup off the corpus scan (no paid run; ~1 week)

- **B1.** Implement `driver.search_interface` for `edge_similarity_search` and
  `node_similarity_search` over Neo4j vector indexes (`SEARCH` — `queryRelationships` is
  deprecated on 2026.07.1, per BACKLOG 8-scale). All-or-nothing hook: every method must
  be implemented or raise clearly.
- **B2.** Measure recall against brute force on the live graph at `limit=10` (BACKLOG
  8-scale saw the rank-2 fact at k=1). **Decision D2:** the user chooses the recall floor
  they accept for dedup; a missed neighbour is a missed dedup, and the user has said
  quality outranks latency. If recall is not acceptable, the alternative is an exact
  pre-filter (endpoint-type or vendor-scoped candidate sets) — a design cycle, not a
  patch.
- **B3.** Re-measure DB saturation under concurrent searches with the index in place; the
  1.9x ceiling was measured on brute-force scans.
- **B4.** Add the vector index to the write path's cost accounting: BACKLOG 8-scale
  dropped it partly because of insert overhead; measure that too.

### Phase C — The throughput experiment (paid, bounded; ~1–2 days elapsed)

- Bootstrap the *first-wave vendor's* structural layer (free), enqueue a 200-article
  slice, and run the worker at 1, 2 and 4 processes (`--max-batches` bounded). Report
  episodes/hour, Slice-B dedup counters, `contradicted_same_pair`, `reference_basis`
  distribution, $/article.
- Cost: 200 × $0.077 ≈ **$15 ceiling**; elapsed ~20 h at one worker.
- **Decision D3:** concurrency vs dedup races. `throughput-profile` §5 records the
  hazard: concurrent episodes cannot see each other's writes. The counters from Slice B
  are the instrument; if duplicates rise with N, the user decides whether a slower run
  is the price of dedup quality.
- This experiment is what turns the table below from `[inference]` into a plan.

### Phase D — Wave 1 ingestion (paid; the long pole)

Elapsed at one sequential worker (45 s/episode × 7.9 episodes/article ≈ 5.9 min/article,
~243 articles/day), and the gpt-5-mini cost ceiling:

| candidate wave | articles | 1 worker | 4 workers if C shows ~3.5x | $ ceiling |
|---|---:|---:|---:|---:|
| complete the pilot (AWS + Microsoft) | 510 | 2 days | — | $40 |
| + Acronis + Zerto + Rubrik | +8.4k | +35 days | +10 days | +$650 |
| + Veeam | +20.2k | +83 days | +24 days | +$1,550 |
| + Dell | +10.4k | +43 days | +12 days | +$800 |
| top-6 vendors | 105k | 14 months | 4 months | $8.1k |
| full corpus | 126k | 17 months | 5 months | $9.7k |

`[inference]` throughout except the pilot row's inputs. **Decision D1 is this table.**
The honest reading: at today's measured rate, the corpus is not reachable in a quarter
at any concurrency the profile has shown; a wave of one large vendor plus the small ones
(~30k articles) is a 1–4 month run. The daily budget must be set to match
(`semantic_daily_token_budget` ≈ articles/day × 250k).

During D: structural bootstrap of all 288 sources (free); `cleanup` after each batch,
`maintenance` weekly, on a scheduler (M10); nightly `neo4j-admin` dump (M6); watch the
`dead` job count (`queue-status`).

### Phase E — Community layer and evaluation at wave-1 scale (paid; 1–2 weeks)

- `theme-build --full` over the wave (report cost unknown at scale — 40 communities took
  minutes-to-hours; expect 15–30k communities at full corpus per the sizing doc, so
  incremental refresh matters here).
- Extend the golden set to ~60 questions covering the wave's vendors, with at least ten
  cross-vendor questions and at least **one timeline question about a real document
  change** observed through the next quarterly re-extraction (M8). Without a real change
  in the graph, the temporal use case stays unproven — this is a calendar dependency,
  not a code one.
- **Decision D4:** the same-pair rule (M5), taken on the numbers from A4 and C.

### Phase F — Copilot exposure (~2 weeks; runs in parallel with D)

- MCP server over answer-api (streamable HTTP; tools per plan §7.1), authentication on
  answer-api itself, public exposure (tunnel for the pilot, APIM later — **decision D5**,
  and who owns the Entra app registration), Copilot Studio agent with the plan's
  instructions, published to Teams for a small user group.
- Open the Power Platform DLP conversation now; the plan put it in Phase 0 and it has not
  happened.
- Response caps (answers ~2–4k tokens, ≤15 citations) are already close to what
  `_finalize_answer` produces; verify against the connector payload limits in the pilot.

### Phase G — Steady state

Quarterly: `sync-once` → worker drains the incremental lane → `cleanup` → `sweep` →
incremental `theme-build` → eval. The webhook path can stay unregistered until someone
needs sub-day freshness.

### Decision points, collected

| | decision | when |
|---|---|---|
| D1 | Reset the pilot semantic layer and re-ingest; and the wave-1 vendor set | before A4 |
| D2 | Accept ANN recall for dedup/retrieval, or design an exact bounded candidate set | end of B2 |
| D3 | Worker concurrency vs dedup-race quality | end of C |
| D4 | Same-pair invalidation: gate on `exact` + `source_changed_at`, or suspend | after A4 + C numbers |
| D5 | Exposure route (tunnel vs APIM) and Entra ownership | start of F |
| D6 | Chunk-size A/B before wave 1 — a possible 3–4x cost lever, unmeasured | before D |

---

## 4. What to drop

Close unfixed, with the reason:

- **BACKLOG 23** (never-pushed work): stale — `origin/main` is two commits behind HEAD.
- **BACKLOG 27** (incremental/update paths never exercised): superseded by this plan's
  Phase A/C, which exercise them by construction.
- **BACKLOG 10b** (90-second timeouts on graphiti clients): the answer-path hardening
  covered what mattered; ingest timeouts become a worker-retry concern, already handled by
  `fail_semantic_job` backoff.
- **BACKLOG 7** (non-deterministic extraction at temperature 0): accept. Nothing in the
  plan depends on reproducing an extraction byte-for-byte.
- **BACKLOG 11b** (is solar-pro4 good enough for map extraction): the reranker took over
  relevance; map faithfulness is measured at 5.0. Reopen only if wave-1 eval drops it.
- **BACKLOG 12** (`judge` tier does three jobs) and **20** (report-tier duplication):
  hygiene with no measured effect; fold into any client refactor that happens anyway.
- **BACKLOG 18** (type hygiene): the review already ranked it last; no query filters by
  edge type.
- **BACKLOG 19** (`write_communities --full` unexercised): it was exercised on 2026-09-10
  (item 0's account) and fixed on 09-11 (5c). Close.
- **BACKLOG 21** (cosmetic marker artefacts): accepted by its own text.
- **BACKLOG 5e** (drift-intent questions route to global): routing is 0.97 and global
  answers those questions at 5.0; `drift_wins: False` says DRIFT is not beating global
  on this corpus. Keep the measurement, drop the fix.
- **BACKLOG 2** (judge split): de-prioritised by evidence on 2026-09-11; leave it there.
- **Plan §3.1 "deltas within minutes" and the webhook path**: quarterly re-extraction
  makes a scheduled `sync-once` sufficient. Do not spend time proving the webhook.
- **Plan §6.4 change-event grouping and `versions/diff` enrichment**: not until a real
  change exists in the graph to group.
- **Re-enabling cross-pair contradiction detection**: spec §7 says measure genuine
  contradictions first; the corpus may simply not contradict itself. Do not schedule it.
- **`dedup_retry_on_strong`** and any **zero-candidate short-circuit variant**: both
  measured and refuted; recorded so they are not re-proposed.
- **A vector index for the *ingest invalidation* scan specifically**: that scan is gated
  off. The index is needed for retrieval and node dedup (Phase B) — a different
  justification, so BACKLOG 8's closure stands and B1 is its own item.

Keep, and do cheaply: **BACKLOG 22** (rotate the leaked key — it is on `origin/main`;
rotation is independent of any history rewrite); **BACKLOG 16**'s one dangerous line
(`test_ingest_driver.py:30` DETACH-DELETEs episodes of the live graph under `@live` —
point it at a testcontainer or delete the test); **BACKLOG 14** (a fresh clone cannot
boot answer-api); **BACKLOG 15**'s safe deletions (`eval_answer.py`, `eval_golden.py`,
`probe`); **BACKLOG 17** (the label collision will bite the first vendor-scoped query
that forgets to traverse `HAS_PRODUCT`).

---

## 5. The biggest risk to the whole thing

**The per-episode pipeline is too slow for the corpus by one to two orders of
magnitude, and nothing measured so far says concurrency closes the gap.**

`[measured]` 45 s/episode, 7.9 episodes/article, 126,405 articles → ~17 months for one
sequential worker. `[measured]` the database saturated at ~1.9x under concurrent
searches on the old scan shape; `[backlog]` per-call latency is close to exhausted
(the lean projection was the last easy win; the dedup LLM call is ~70% of the residual
cycle and graphiti issues one per fact plus one timestamp call per new fact). The
system's purpose is answers over the corpus; at this rate it is a system over two or
three vendors, chosen once, and every quality number would have to be re-earned on the
one that is chosen.

Two things compound it. The retrieval scan (§1.2) grows linearly with whatever *is*
ingested, so success at ingestion degrades answers unless Phase B lands first. And the
temporal promise — the reason for choosing graphiti and append-only updates — has no
evidence yet and cannot get any until a real document change is ingested, which the
quarterly cadence puts at least one re-extraction away.

**What retires it soonest:** Phase B then Phase C, in that order, before any wave is
chosen — about a week of unpaid work and one ~$15 paid experiment. B removes the scan
that the concurrency measurement was taken on; C measures episodes/hour at 1/2/4
workers on the production path with the dedup counters watching for races. The result
is one number, and it decides whether the plan is "ingest the top six vendors over a
quarter" or "ingest one vendor and say so". Either is a plan. The current state — where
the number is unknown and the roadmap still says 105k articles — is not.

If C shows less than ~3x, the remaining levers are structural and each needs its own
measurement: larger chunks (D6, possibly 3–4x on cost and count), batching or
suppressing the per-fact timestamp call, and a bounded candidate set for dedup that
lets episodes run in parallel without racing each other. None of these should be
started on reasoning alone; this project's record on that is the reason the brief asked
for this review.
