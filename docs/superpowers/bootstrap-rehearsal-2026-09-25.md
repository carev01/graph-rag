# Bootstrap rehearsal on srv-k3s — 2026-09-25

**What:** the first semantic ingest on the production K3s deployment: 181 semantic jobs
(180 articles + 1 tombstone) from two deliberately scoped sources, run at 1, 2 and 4
workers, then the baseline restored (duplicate merge, full theme-build, router eval).
**Why:** measure throughput, cost and duplicate behaviour on production infrastructure
before sizing the full vendor-priority bootstrap, and prove the bootstrap-first lane fix
(BACKLOG 49) and the source-scoped claims hold in a real run.
**Image:** `ghcr.io/carev01/graph-rag:sha-7d8f7bb` (release revisions 3–8).
**Runbook:** `docs/deploy/k3s.md` §8–10, executed as written.

## 1. Setup

| step | result |
|---|---|
| `helm upgrade` to `sha-7d8f7bb` | revision 3; sync 0, workers 0, CronJobs suspended |
| queue before | 593 `bootstrap` + 5,646 `incremental` pending (all from the 2026-09-24 first pull) |
| `relane-jobs` report → `--apply` | identical counts: 4,533 articles' `source_id` backfilled, 1,706 missing in graph; **5,646 incremental → bootstrap, 0 kept**. Incremental lane now empty |
| structural `bootstrap --source-id` | Cohesity FortKnox User Guide `7baa5405…`: applied 31, skipped 62 (hash-gated, all already queued); Veeam Service Provider Console Deployment Guide `7931f8d1…`: applied 87 |
| scope | `--set-string 'config.SEMANTIC_CLAIM_SOURCE_IDS=7baa5405…\,7931f8d1…'` rendered locally first, ConfigMap confirmed byte-exact (revision 4) |
| one-article smoke | all four §9 checks passed: `Succeeded`; `semantic batch: jobs=1`; done 0 → 1; `semantic worker scope: 2 source(s): 7baa5405…, 7931f8d1…`. 64k tokens, ~100 s |

The 1,706 rows with no `source_id` are all `remove` tombstones for articles never written
structurally (1,802 removes − 96 resolved = 1,706). They have no Article node to read a
source from, so only an unscoped worker will claim them; a tombstone extracts nothing.

Before the structural bootstrap, FortKnox already had 63 Articles in the graph and exactly
63 pending jobs, all without episodes — so the hash gate could not strand any of them. It
would have, had any lacked a job: BACKLOG 50.

## 2. Throughput by worker count

Workers were scaled through Helm (`worker.replicas`, `--batch 1`, concurrency 1 per
worker). Phase windows are the Helm rollout times; episodes are counted from
`Episodic.created_at` in the graph, which does not depend on article size the way an
articles-per-hour figure does (FortKnox's later articles ran to 7+ episodes, VSPC's are
mostly 1–3).

| phase | workers | minutes | articles | episodes | episodes/h | vs 1 worker | extra duplicate copies |
|---|---|---|---|---|---|---|---|
| smoke | 1 | 3.1 | 1 | 2 | — | — | 0 |
| 1 | 1 | 49.9 | 41 | 118 | 142 | 1.0× | 0 |
| 2 | 2 | 45.9 | 44 | 231 | 302 | 2.1× | 2 |
| 3 | 4 | 24.5 | 94 | 189 | 463 | 3.3× | 4 |
| **total** | | 123.4 | 180 | 540 | | | **6** |

Phase 3 carries VSPC's whole warm-up window (see §3), which is serial by design; 3.3× at 4
workers includes that floor. No job failed, none was retried, no worker restarted. Worker
memory stayed ~150 Mi against a 384 Mi request.

Per-batch counters over the whole run: vector search routed 3,582 node searches to the
tuned index with **0 fallbacks**; dedup guard `parse_failures=0 invalid_calls=0
retried=0`; `contradictions_suppressed=0`.

## 3. The global warm-up lock engaged — proven from the graph

`SEMANTIC_GLOBAL_WARMUP_LOCK` (default on) logs nothing when it takes the lock, so its
engagement had to be reconstructed. VSPC started cold during phase 3 with **four** workers
running. Its articles ordered by first episode:

| # | article | first → last episode | overlaps with another of the first 14 |
|---|---|---|---|
| 1–8 | `043f3a40` … `0df23752` | 02:07:21 → 02:15:22 | **0** — strictly one at a time |
| 9–14 | `231025d7` … `09b147ab` | 02:15:45 → 02:17:30 | 1–3 each — parallel again |

The first `INGEST_WARMUP_ARTICLES` = 8 ran serially across four processes, and
parallelism resumed at article 9 — exactly the mechanism's contract. This run now adds an
INFO line per acquisition (`warm-up lock held for cold article <id> (waited <s>s)`), so
the next run can prove it from the log instead.

## 4. Duplicates

Six extra copies across five names, out of 1,948 entities:

| name | copies | created | source |
|---|---|---|---|
| Veeam Backup for Public Clouds | 3 | 02:24:11, :21, :24 | VSPC |
| Veeam Backup for Microsoft 365 | 2 | 02:24:24, :32 | VSPC |
| Veeam Cloud Connect Guide | 2 | 02:22:47, :51 | VSPC |
| Timeline mode | 2 | 01:25:35, 01:26:15 | FortKnox |
| MAC address | 2 | 01:32:34, :41 | FortKnox |

Every one is two concurrent articles of the **same** source creating the same entity
seconds apart, **after** that source's warm-up window closed — the residual race the
warm-up barrier deliberately does not cover (it protects the window where a source's own
novel entities are seeded; `docs/superpowers/specs/2026-09-14-concurrency-duplicate-mitigation-design.md`).
None was cross-source. Rate: 0 at 1 worker, 2 in 44 articles at 2, 4 in 94 at 4.

`merge-duplicates` (report, then `--apply`, as one-off pods with workers at 0): 5 groups,
6 merged, 0 label conflicts, oldest node survives; `RELATES_TO` count unchanged at 6,455;
entities 1,948 → 1,942; a re-run reports 0 groups. The 11 case-only near-duplicates
(`Tenant`/`tenant`, `Replication`/`replication`, …) are report-only by design.

## 5. Tokens and cost

| | value |
|---|---|
| LLM tokens (token_ledger, smoke + 3 phases) | **12.0M** |
| content tokens of the 180 articles (`Article.estimated_tokens`) | 256k (median article ~1.4k) |
| per article / per episode | ~66k / **~22k** |
| LLM tokens per content token | **~47×** |
| dollars | **~$1.6 (all cheap tier) to ~$6 (all strong)** — `token_ledger` has no tier or prompt/completion split (BACKLOG 51); extraction is mostly cheap-tier, so the low end is likelier |

The per-episode figure agrees with every earlier measurement (July's ~200k tokens/article
at ~6.5 episodes; September's 26–35k/episode) and is slightly *better*. What it
contradicts is CLAUDE.md's "~0.8–1.5B LLM tokens" for the corpus: Graphiti's cost is a
roughly fixed per-episode overhead (prompt scaffolding, entity and dedup context), not
proportional to content. At ~680k episodes (105k articles × ~6.5) the corpus is **~15–20B
tokens**. CLAUDE.md is corrected in this change.

## 6. Baseline restored

- **theme-build `--full`** (one-off pod, 59.5 min): 87 communities (L0 53, L1 20, L2 13
  reports), 86 reports written, 1,478 facts cited, 22 findings dropped by verification, 0
  lost. The verifier returned no usable JSON 13 times (each retried); 1 report stayed
  staged and `--verify-pending` promoted it (2 more findings dropped). 1 community skipped.
- **router eval:** see §7.

## 7. Router eval

`python -m answer_api.eval_router` from the dev checkout, against the rebuilt layer
(`router-eval-report.md` holds the per-question table). The previous run (2026-09-11)
was on the Neo4j instance that was later lost; the current graph was rebuilt from
2026-09-14 and never evaluated before this rehearsal, so the comparison is indicative,
not like-for-like.

| | 2026-09-11 (old graph) | this run |
|---|---|---|
| routing accuracy | 0.97 | **0.97** (1 miss: a local retention question routed to timeline) |
| grounding precision | 0.92 | **0.69** — local 0.86, timeline 0.80, **global 0.29** |
| faithfulness mean | 5.00 | **5.00** (1/29 unscored: global Q16 refused) |
| comparative global / drift | 5.0 / 4.8 | 4.8 / 4.6 |

**The global drop is cross-vendor crowding, and it will get worse with every vendor.**
Re-running only the retrieval step (no LLM) for the failing global/drift questions — all
of them explicitly about AWS and Azure — shows one or two of the four shortlisted
communities are Cohesity FortKnox communities (`Cloud Vault Storage Classes and
Security`, `Cloud Vaulting: Subscriptions, Quorum Approval…`), and a community titled
*Azure VM Recovery and Private Connectivity* cites 21 Cohesity facts to 4 Microsoft ones.
`shortlist_communities` ranks every community at the level by embedding similarity (+
rerank) with no vendor scoping, so a question that names its vendors competes with every
other vendor's reports on the same concepts (encryption, immutability, restore). With two
new sources that is 1–2 of 4 slots; with 40 vendors it is most of them. BACKLOG 52.

The local misses are not crowding: Q2 and Q10 returned 11–12 facts, all but one from
the AWS/Azure sources, just not from the single expected article — precision@k against
one expected id, on a graph extracted with different models than the golden set was
written against.

The eval crashed after writing the markdown report, serialising a `datetime` in the raw
answer dump, so this run's `router-eval-raw.json` was lost; fixed (`_raw_json`, tested).

## 8. What changed as a result

- **Warm-up lock logs each acquisition** (`semantic_worker._run_group`), test-pinned.
- **`neo4j.notifications` quieted to WARNING** (`logging_setup.configure_logging`): every
  pod log was dominated by `IF NOT EXISTS` schema no-ops and cartesian-product hints.
- **CLAUDE.md's token projection corrected** (§5).
- **BACKLOG 50** — a state reset silently strands every already-structural article.
- **BACKLOG 51** — no dollar accounting.
- **BACKLOG 52** — global/DRIFT community shortlist has no vendor scoping (§7).
- **`eval_router` no longer loses its raw answers** to a `datetime` in the dump.

## 9. Inputs for the full-bootstrap sizing decision

- Throughput scales near-linearly to 4 workers (3.3× including a warm-up floor); 463
  episodes/h at 4 means ~680k episodes ≈ 1,470 worker-hours ≈ **~2 months at 4 workers,
  ~2 weeks at 16** if scaling held (untested past 4; `scripts/dispatch_sim.py` projects
  saturation around 12–16 per process on the pilot's sources, and the global warm-up
  floor — 280 sources × 8 articles serial, ~8 days — dominates past ~32-way).
- Cost is ~22k tokens/episode, so a daily budget of B tokens buys B / 22k episodes/day:
  the default 840M ≈ 38k episodes/day ≈ 18 days for the corpus — above what 4 workers can
  consume (~11k episodes/day), so at 4 workers the budget is not the constraint.
- Duplicates rise with concurrency but stay small and repairable (6 per 181 articles at
  ≤4 workers, all same-source races); `merge-duplicates --apply` must run with workers at
  0 before each theme-build (theme-build refuses otherwise).
- The cluster had headroom: ~150 Mi per worker actual vs 384 Mi requested; memory
  requests, not usage, are what limit how many workers fit (82% of node memory was
  already requested before this deployment).
