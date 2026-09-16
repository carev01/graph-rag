# Spreading sibling articles apart in ingest dispatch order — assessment

**Date:** 2026-09-15
**Status:** Assessment only. No decision taken, nothing implemented, no branch.
**Trigger:** the W=8 validation (`docs/superpowers/ab-warmup-2026-09-15.md`) left 16
exact-name duplicates, all from two article pairs that were adjacent in dispatch order
(positions 32/33 and 71/72). The open question in that document — reorder dispatch so
siblings are not adjacent — is evaluated here: is it sound, is it safe (the resume/cursor
contract), and is it worth doing when the merge pass already reaches zero.

Everything below is read from `src/`, from graphiti-core **0.30.1** in `.venv`, from the
two A/B documents, and from read-only sessions against the live Neo4j (2026.07.1, group
`backup-docs` at its verified 999 / 655 / 655 / 0 baseline) and two read-only `GET`s
against DocExtractor. Nothing was written, nothing was run that spends money. The offline
simulation in §3 uses the baseline graph's own timestamps and entity spans; its scripts
lived in the session scratchpad and are described fully enough to re-create (§3.1).
Where a statement is an inference rather than a citation or a measurement it says so.

---

## 0. Summary of the verdict

**Adopt a small, deterministic version of it — in `ingest_source` only — and leave the
graph-sync stream, the cursor, and the worker's claim order alone.**

- **The cursor-gap risk is not real for semantic dispatch.** The bootstrap watermark
  (`bootstrap_after` / `last_id`) is computed and advanced only by `graph_sync.sync_core`,
  which consumes the delta stream *sequentially in stream order* and never learns what
  the semantic layer did. Semantic progress is a completed-set (`semantic_jobs.status` per
  article; the per-chunk `HAS_EPISODE` gate), not a high-water mark. Reordering semantic
  dispatch cannot strand an article. Reordering the *structural* stream would — and this
  proposal forbids it (§1).
- **The premise holds on two of three production paths, not all.** `ingest_source`
  dispatches by `sort_order` (siblings adjacent). The worker's *bootstrap* lane dispatches
  in the delta feed's order, which is **ascending article UUID** — measured on both pilot
  sources: 591/591 consecutive records ascending, 0 `sort_order`-adjacent — so it is
  already spread. The worker's *incremental* lane dispatches in `seq` order, and
  extraction order tracks `sort_order` tightly (147/147 and 328/444 adjacent pairs), so
  siblings arrive adjacent there (§2).
- **The benefit is real and measurable offline.** A replay of `run_concurrently`'s
  barrier semantics over the pilot's real chunk durations and entity spans reproduces both
  paid arms (W=0: 33–44 vs 30 measured; W=8: 17 vs 16 measured, 14 of the 17 from pair
  71/72). Keeping the warm-up prefix and spreading the remainder takes the pilot's
  residual from ~17 to ~1–2 at no throughput cost (slightly faster, in fact). The brief's
  `(c-1)/(N-1)` is half the true overlap probability — it is `≈ 2(c-1)/N`, ~7% at N=83 —
  and a slow article roughly doubles its own window (§3).
- **Deterministic beats random**, on the tail and on reproducibility: a uniform shuffle
  averages 3 but its p95 is 14 — one shuffle in twenty is as bad as `sort_order` (§4).
- **The warm-up set is insensitive to ordering** (cold-set risk share 77.5–81.2% for every
  ordering tried, 79.5% for `sort_order`), and the hybrid keeps it identical by
  construction (§5). The replay gate is order-blind (§6).
- **Worth it, narrowly.** It is not a correctness fix — the merge pass stays the
  correctness mechanism. It is (a) a ~10x cut in what the merge pass must rewire on the
  `ingest_source` path, and (b) more importantly, it makes the CLI/pilot path dispatch the
  way production bootstrap already does, so pilots stop measuring a duplicate class that
  production's bootstrap lane does not have. Cost: ~15 lines in one function plus tests
  (§7, §8).

---

## 1. The "Critical" question — can reordering create a resume gap?

**No, not for semantic dispatch. Traced, not inferred.**

### 1.1 Where the watermark is computed and advanced

`SyncCore.bootstrap` (`src/graph_sync/sync_core.py:113-152`) is the only bootstrap
consumer:

```
async for rec in stream.records():                      # :124
    touched = await self._apply_record(rec, res)        # :125  awaited inline
    ...
    if isinstance(rec, ContentRecord):
        last_id = rec.id                                # :129
...
await self._store.upsert_bootstrap(shard, watermark, last_id, ...)   # :134-136
...
bootstrap_after = last_id  # resume from where the stream dropped   # :139
```

The loop is strictly sequential: record *n+1* is not read until `_apply_record(n)` has
returned, and `_apply_record` (`:58-104`) writes the structural node
(`repo.apply_structural`, `:102`) **and enqueues the semantic job**
(`store.enqueue_semantic_job`, `:101` — and `:88` on the catalog-incomplete path) before
returning. So `last_id` always means "every record before this one in *stream order* has
been applied structurally and has its semantic job durably queued in Postgres". The
stream's order is ascending UUID (§2.2), which is what `bootstrap_after` keys on
(`CLIENT-USAGE-GUIDE.md:399`, `:603-604`). The invariant the contract assumes holds by
construction, and nothing in the semantic layer can weaken it.

The incremental cursor is advanced only at `sync_core.py:196-197`, only after a clean
terminal `cursor` line (`delta_client.py:52-55`), and again independently of semantic
work.

### 1.2 Who else writes the watermark or cursor

Nobody. `bootstrap_progress` and `sync_cursor` are written by `state_store.py:66-78`
and read/written from `sync_core.py` only. A grep of `src/` for `bootstrap_progress`,
`last_id`, `sync_cursor`, `set_cursor`, `upsert_bootstrap` finds no reference in
`graph_extract/`, `semantic_worker.py`, or `graph_sync/cli.py`.

### 1.3 How semantic progress is actually tracked — a completed-set, twice over

- **Worker path:** one `semantic_jobs` row per article with `status`
  (`state_store.py:20-31`); claimed `pending → in_progress` (`:138-149`), completed per
  job (`:151-155`), failed with backoff per job (`:157-168`), reaped by lease
  (`:170-178`). An article is done when *its* row is done; no row implies anything about
  any other row. Order of processing is irrelevant to resumability.
- **`ingest_source` path:** no persistence at all. Resume is re-running the command; the
  per-chunk gate `Provenance.already_ingested` (`provenance.py:8-16`) skips chunks whose
  `HAS_EPISODE` edge carries the same `chunk_index` + `content_hash` and is not
  superseded. Per article, per chunk — a completed-set.

### 1.4 The one place a gap *would* open

If someone reordered or parallelised the `async for rec in stream.records()` loop in
`sync_core.bootstrap`, `last_id` would stop meaning "prefix applied" and
`bootstrap_after=last_id` would skip whatever lower ids had not finished. That is not what
the brief proposes (it proposes reordering the semantic fan-out), but the proposal must
say it as a rule: **the structural stream is consumed in stream order, always.** There is
no fix to price because no fix is needed.

Related but distinct: the worker's warm-up predicate already tolerates the one ordering
race that does exist between the two layers — a job claimed before its Article node is
written — by answering "cold" (`warmup.py:95-100`).

---

## 2. Question 1 — where dispatch order comes from in production

Three call sites reach `run_concurrently`; the order each hands it is different.

| path | order handed to `run_concurrently` | siblings adjacent? | evidence |
|---|---|---|---|
| `graph_extract.cli ingest` → `IngestDriver.ingest_source` | `list_article_ids`: `MATCH (a:Article {source_id}) … ORDER BY a.sort_order` (`ingest_driver.py:120-125`), fanned out at `:211-213` | **yes** — `sort_order` is TOC order | the pilot list is exactly this order: JSON order == sequential-run `created_at` order, monotone in `sort_order` within each source (§3.1) |
| `graph_sync.cli worker`, **bootstrap lane** | `claim_semantic_jobs … ORDER BY (lane='bootstrap'), next_attempt_at` (`state_store.py:145`); `next_attempt_at` defaults to `now()` at enqueue (`:27`), enqueue happens per record in stream order (`sync_core.py:101,125`) → **feed order** | **no** — feed order is ascending UUID | read-only bootstrap `GET` of both pilot sources (§2.2): 147/147 and 444/444 consecutive pairs id-ascending, **0/147 and 0/444** `sort_order`-adjacent |
| `graph_sync.cli worker`, **incremental lane** | same claim query → feed order = `seq` order (`content_changes` BIGSERIAL, guide `:494`) | **mostly yes** (inference) | `seq` is assigned in the article-write transaction (guide `:501-502`); extraction order on the two sources tracks `sort_order`: **147/147** and **328/444** consecutive `extracted_at` pairs are `sort_order`-adjacent, whole-source spans of 20 s and 2 min (§2.3) |

Two structural facts about the worker matter later: it claims **10 jobs per batch** by
default (`graph_sync/cli.py:199`), drains that batch completely through
`run_concurrently` (`semantic_worker.py:145-146`), then claims again; and its
predicate is per article (`is_cold_article`, `graph_sync/cli.py:229-232`,
`warmup.py:91-101`).

### 2.1 So is the premise moot?

Partly. The premise ("siblings are adjacent in dispatch") is **true for `ingest_source`
and for the incremental lane, false for the bootstrap lane**. The bootstrap lane is the
path that will carry ~126k articles; it is already spread by accident of UUID ordering
(with the caveat that a 10-job batch at c=4 gives siblings that do land in one batch a
~40% collision chance — §3.4). The incremental lane carries updates (whose entities
mostly already exist, so the hazard needs *new* entities in both siblings) and, more
importantly, **every source onboarded after the initial bootstrap**, which arrives as
`added` records in `seq` order — i.e. in TOC order, siblings adjacent. `ingest_source` is
the pilot/A-B/vendor-by-vendor CLI path, and every measurement this project has made ran
through it or through a runner that copies its order.

### 2.2 The feed-order measurement

Read-only `GET /api/articles/delta?source_id=…` (bootstrap mode) for
`21632f3b…` (AWS Backup, 148 records) and `6da00d8b…` (445 records), via
`graph_sync.delta_client.DeltaStream`. Both terminated clean. Record ids were strictly
ascending in both (the guide's own sample at `:542-546` shows the same: `010d…`,
`011d…`, `014e…`), and no two consecutive records had consecutive `sort_order`. This is
consistent with `bootstrap_after=<id>` being a keyset cursor on the article primary key;
the guide does not state the order explicitly, so treat "ascending UUID" as *observed on
this deployment*, not contractual.

### 2.3 The extraction-order measurement (proxy for `seq`)

Read-only `GET /api/articles?source_id=…&limit=200` (paged). `extracted_at` is distinct
per article (148 and 445 distinct values). Sorted by `extracted_at`, consecutive articles
have consecutive `sort_order` 147/147 times on the AWS source and 328/444 on the Azure
source. The guide says the outbox row is appended "in the same transaction as the article
change" (`:501-502`), so `seq` order within a run is extraction order. That `seq` order
*is* extraction order is an inference from that sentence; the 147/147 figure is a
measurement.

---

## 3. Question 2 — quantifying the benefit, by offline replay

### 3.1 The simulation

Inputs, all read-only from the `backup-docs` baseline:

- the 83 pilot articles' 655 chunks with their **sequential-run start times**:
  graphiti sets `created_at = utc_now()` at the top of `add_episode`
  (`graphiti.py:1068`, used at `:1109`), so `Episodic.created_at` is the chunk's start.
  Consecutive stamps give per-chunk durations (mean 38.1 s; per-article mean 300 s, median
  215 s, max 1,019 s; sum 6.93 h against the logged 6.92 h);
- the 999 entity spans via `(:Article)-[:HAS_EPISODE]->(:Episodic)-[:MENTIONS]->(:Entity)`
  with, per (entity, article), the **first chunk** that mentions it;
- the recorded dispatch list `.superpowers/sdd/pilot-article-ids.json`, verified to equal
  the sequential run's actual order and to be the two sources merged by `sort_order`.

Model: chunk `(A,i)` occupies `[s, s+d]`; its node resolution happens at `s + α·d` and
its nodes commit at `s + d` (one transaction at the end of `add_episode`). For each
entity, order its first-mention chunks by resolution time; the first creates the node; a
later chunk creates *another copy* iff no earlier-resolving chunk has committed by its
resolution time. `excess = Σ(copies − 1)` — the same quantity `merge_duplicates._COUNT`
(`merge_duplicates.py:84-88`) reports. The scheduler replays `run_concurrently` exactly:
items in dispatch order; a warm item takes the next free slot FIFO; the predicate is
evaluated when the loop reaches the item (`concurrent_ingest.py:102`); a cold item drains
everything launched, runs alone, drains again (`:106-107`, `:116-117`); cold means fewer
than W articles of the source have a committed chunk at that moment (`warmup.py:28-39`).
Noise: each chunk duration × lognormal(σ=0.3), 200 seeds.

### 3.2 Calibration against the two paid arms (recorded order, c=4)

| arm | measured dupes | simulated, α = 0.1 / 0.3 / 0.5 | noisy mean ± sd (α=0.3) | wall: sim vs measured |
|---|---|---|---|---|
| W=0 | 30 | 44 / 38 / 33 | 33.8 ± 8.3 | 1.88 h vs 2.17 h |
| W=8 | **16** | **17 / 17 / 17** | 13.9 ± 6.4 | 2.93 h vs 3.09 h |

At W=8 the simulated residual is 17, of which **14 come from dispatch pair (71, 72)** —
the immutable-vault `Overview`/`Manage` pair whose Azure region table the A/B identified
(`ab-warmup-2026-09-15.md:55-62`); the names match (`Germany North`, `Japan East`,
`Korea Central`, …). The second measured pair (32/33) collides in a third of noisy
replays rather than deterministically. The wall clock runs 8–13% short because the model
does not include the provider's under-load degradation
(`ab-concurrency-2026-09-14.md:83-87`). The model is not exact; it is within the
run-to-run noise of the thing it models, and it reproduces the *population* of the
residual, which is what the brief asked to check.

### 3.3 The orderings, at c=4, W=8, α=0.3

"Hybrid" = the first W articles of each source by `sort_order` (the warm-up set, unchanged)
followed by the remaining 67 in the stated permutation.

| ordering | excess, no noise | noisy mean ± sd | wall |
|---|---|---|---|
| recorded (2 sources merged by `sort_order`) | 17 | 13.9 ± 6.4 | 2.93 h |
| source A then B, each by `sort_order` (the `ingest_source` shape) | 20 | 14.4 ± 7.0 | 2.77 h |
| ascending UUID (what the feed gives the worker), no warm-up prefix | 3 | 1.7 ± 1.1 | 2.92 h |
| **hybrid, remainder by ascending UUID** | **2** | **1.3 ± 1.0** | **2.66 h** |
| hybrid, remainder by sha256(id) | 2 | 2.0 ± 1.0 | 2.70 h |
| hybrid, remainder stride 4 | 3 | 2.9 ± 1.7 | 2.71 h |
| hybrid, remainder stride 17 (= ⌈67/4⌉) | 0 | 1.8 ± 1.3 | 2.67 h |
| hybrid, remainder bit-reversal | 0 | 1.0 ± 0.9 | 2.69 h |
| hybrid, remainder uniform random (200 permutations) | mean 3.0 | 3.0 ± 3.4, **p95 = 14, max = 18** | — |
| full random shuffle including the warm-up set (200 perms) | mean 2.9 | — | — |
| full random shuffle, **W=0** (200 perms) | mean 12.3 | — | — |

Readings:

1. **Spreading the remainder cuts the residual ~10x** (17 → 1–3) on every deterministic
   permutation. The differences *among* deterministic permutations (0–3) are inside the
   noise (sd ≈ 1) and should not be over-read.
2. **No throughput cost; a small gain.** The hybrid is 9% faster than the recorded order
   in the pilot shape because the recorded list interleaves two sources, and every cold
   item of one source drains the other source's in-flight warm work (`:106-107`). Running
   the 16 cold articles first removes those drains. For a real single-source
   `ingest_source` there is nothing to interleave, and the gain shrinks to the difference
   between rows 2 and 4 (~4%).
3. **Ordering matters more than the barrier on this sample** (simulation only, not
   measured): a full random shuffle with *no* warm-up (12.3) beats `sort_order` with W=8
   (17), because the sibling class is larger than the hub class here. The two mechanisms
   are complementary, as the A/B said: hybrid + W=8 gives ~1–2.

### 3.4 The brief's pair-overlap arithmetic

Equal durations, c slots, uniform shuffle of N items: items at dispatch distance ≤ c−1 are
in flight together, so P(overlap) = 2·Σ_{d=1}^{c−1}(N−d) / (N(N−1)) **≈ 2(c−1)/N**. The
brief's `(c−1)/(N−1)` counts one direction only and is half the true value.

| N | simulated | 2(c−1)/N | brief's (c−1)/(N−1) |
|---|---|---|---|
| 83 (pilot) | 6.9% | 7.2% | 3.7% |
| 67 (pilot's warm remainder) | 9.0% | 9.0% | 4.5% |
| 146 (AWS Backup source) | 4.0% | 4.1% | 2.1% |
| 445 (Azure Backup source) | 1.4% | 1.4% | 0.7% |

With the real durations and chunk-level windows, P(the pair's exclusive entities
actually collide | dispatch gap), 150 noisy replays each:

| pair | gap 1 | 2 | 3 | 4 | 5 | 6 | ≥8 |
|---|---|---|---|---|---|---|---|
| 32/33 (5 exclusive entities, short articles) | 33% | 15% | 9% | 5% | 0% | 1% | 0% |
| 71/72 (15 exclusive entities, slow) | 77% | 55% | 20% | 5% | 3% | 0% | 0% |

So adjacency is not a certainty either — the shared entities sit in particular chunks —
but a slow article does widen its own window: pair 71/72 still collides more than half
the time at gap 2. Folding the gap distribution under a uniform spread into these
collision rates gives an expected collision probability per sibling pair of roughly
**2.2/N**: 3.3% at N=67, 1.5% for a 146-article source, 0.5% for a 445-article source.
In production the fan-out unit is a whole source, so spreading works *better* than in the
83-article pilot.

Worker bootstrap lane, for completeness: siblings land in the same 10-job batch with
probability ≈ 2·9/N_source (4% at N=451) and, once there, collide ≈ 40% of the time
(mean of the gap-1..9 rows) → ~1.6% per pair. Same order of magnitude as spreading;
nothing to gain there.

### 3.5 Full-corpus extrapolation (126,405 articles, 280 sources, ~451 articles/source)

The at-risk population in the pilot: 46 span-2 entities at `sort_order` distance ≤ 4
(**30 at distance 1**, 8 at 2, 5 at 3, 3 at 4) out of 999. Under `sort_order` dispatch
at W=8 the simulation converts ~17 of those into duplicates (0.20 per article); under a
spread order, ~1–3 (0.02 per article).

| path | expected duplicate entities, corpus | merge-pass edge moves (pilot ratio 23/16) |
|---|---|---|
| `sort_order` dispatch everywhere (hypothetical) | ~25,000 | ~36,000 |
| spread / hybrid dispatch | ~1,000–3,000 | ~1,500–4,500 |
| worker bootstrap lane as it is (UUID order, batch 10) | ~2,000–3,000 | similar |

These are extrapolations from **two Azure/AWS sources**. The density of "sibling pages
sharing a table of otherwise-unseen names" is a property of how a vendor writes
documentation; Dell or Veeam may be 3x higher or lower. Treat the ratios as the finding,
not the absolute counts.

---

## 4. Question 3 — random versus deterministic

Deterministic wins, on three grounds:

1. **The tail.** A uniform shuffle of the warm remainder averages 3 excess but its p95 is
   14 and its max 18 — one run in twenty places a sibling pair adjacent and is as bad as
   `sort_order`. Every deterministic permutation tried is 0–3 with sd ≈ 1 (§3.3).
2. **Reproducibility.** Runs on the same source in the same graph state dispatch
   identically, so A/Bs stay comparable and a duplicate that appears once reappears.
   Random ordering would add a new source of variance on top of the ~28% name churn the
   A/B already measured (`ab-concurrency-2026-09-14.md:52-55`).
3. **Resumability** is a wash (§1, §6) — neither ordering can strand work — but a
   deterministic order makes "which articles are left" predictable from the graph alone.

Among deterministic choices:

- **Ascending UUID** (`sorted(ids)`): simplest possible code; **identical to the order
  the worker's bootstrap lane already uses**, so a CLI pilot then dispatches the way
  production does. Sim: 2 / 1.3 ± 1.0. No structural guarantee — siblings land near each
  other with probability ≈ 2(c−1)/N.
- **Bit-reversal of the `sort_order` rank**: sends `sort_order` distance 1 to dispatch
  distance n/2, distance 2 to n/4, distance 4 to n/8 — a *guarantee* for the class that
  is 30 of the 46 at-risk entities. Sim: 0 / 1.0 ± 0.9. About ten more lines and one more
  concept to explain.
- **Stride s**: moves the failure mode rather than removing it — `sort_order` distance
  *s* becomes dispatch distance 1 (stride 4: 3 / 2.9 ± 1.7, from the three distance-4
  entities). Rejected.
- **Hash of id**: equivalent to UUID order (UUIDv4 is already a hash-shaped key); no
  reason to prefer it.

**Recommendation: ascending UUID.** The bit-reversal guarantee is worth ~1 duplicate per
83 articles in the simulation — inside the noise — and UUID order buys something the
guarantee does not: one dispatch order across the CLI and the production bootstrap lane.

---

## 5. Question 4 — interaction with the warm-up barrier

Two mechanics decide this, both in `run_concurrently`:

- The predicate is evaluated **when the dispatch loop reaches the item**
  (`concurrent_ingest.py:102`), and the loop does not block on the semaphore
  (`_one` acquires it inside the task, `:79-81`). After a cold item's drain, every
  following warm item of that source is evaluated and launched immediately.
- Warmth is graph state (`warmup.py:28-39`, `:65-89`), so the cold set is "the first W
  articles *in dispatch order* of each source that reach the loop while fewer than W of
  that source's articles have a live episode".

So reordering changes which W articles form the cold set. Measured (risk-curve method of
`scripts/risk_curve.py:68-84`: share of Σ(k−1) carried by entities whose first appearance
in dispatch order lies in the cold set):

| ordering | cold-set risk share |
|---|---|
| recorded (`sort_order`, matches the spec's 79.5% at c=8 — `…mitigation-design.md:51`) | 79.5% |
| source A then B by `sort_order` | 76.1% |
| ascending UUID | 81.2% |
| full random shuffle (mean of 200) | 77.5% |
| any hybrid (cold set is the `sort_order` prefix by construction) | 79.5% |

**Neither helps nor hurts, materially.** The spread is 5 points across every ordering
tried, and random is only 2 points below `sort_order`. The spec's mechanism — "overview
pages carry the hubs" (`:61-65`) — is real but nearly redundant on this sample: the hubs
(`Azure Backup` spans 55 of the 83 articles, `AWS Backup` 30 — more than the AWS source's
28, since Azure pages mention it too) appear in most articles, so *any* eight articles of
a source seed them. What matters is that a sequential
prefix exists at all (W=0 shuffle: 12.3; W=8 hybrid: ~1–2), not which articles are in it.

The hybrid keeps the cold set byte-identical to today's, which is the conservative choice
and costs nothing. One consequence worth naming: in a multi-source list the hybrid runs
all cold articles first and so avoids the drains that a cold item of source B imposes on
source A's warm work — 9% faster in the pilot shape (§3.3). `ingest_source` is
single-source, so this is a pilot-runner effect, but the pilot runner is where the
measurements come from.

Resumed runs behave as today: the cold prefix's already-ingested articles are dispatched
cold, skip every chunk at the gate, and count as live; the remaining cold slots fill from
the prefix; the remainder fans out.

---

## 6. Question 5 — the replay gate is order-blind

`Provenance.already_ingested` (`provenance.py:8-16`) matches
`(:Article {id})-[r:HAS_EPISODE]->()` on `r.chunk_index`, `r.content_hash`, and
`coalesce(r.superseded,false) = false`. Every key is a property of *this article's own*
edge; no other article, no run, no position is consulted. `Provenance.link` (`:18-49`)
supersedes only the same article's same-`chunk_index` edge.
`_supersede_trailing_episodes` (`ingest_driver.py:182-194`) is per article. The
`content_hash` the driver keys on comes from the article's own node or its own markdown
(`:149`). There is no cross-article state anywhere in the gate path.

The incident the brief remembers — an A/B that skipped all 83 articles — was the gate
not being **group**-scoped (`ab-concurrency-2026-09-14.md:28-33`), which is a different
axis entirely and is unaffected by dispatch order.

Verified, not assumed. The one order-dependent input to extraction that *does* exist is
graphiti's `previous_episodes` context: `retrieve_episodes` takes the 10 most recent
episodes by `valid_at ≤ reference_time` **among those already in the graph**
(`graph_data_operations.py:107,117-118`). Under `sort_order` dispatch an article's
context tends to be its just-committed sibling; under UUID order it is whichever articles
happened to commit before it. This changes extraction context, not correctness, and its
effect on entity yield cannot be separated from the measured ~28% run-to-run name churn
without a paid run. Noted as unmeasured in §10.

---

## 7. Question 6 — is the benefit worth anything?

Honest framing first: **nothing here is a correctness fix.** The merge pass reaches 0 on
every arm (`ab-warmup-2026-09-15.md:84-85`; 30 of 30 on the earlier arm). Reordering
changes only how many groups it must rewire and what a pilot measures.

What it buys, in order of weight:

1. **Representative pilots.** Every measurement so far ran through `ingest_source`'s
   order or a copy of it. Production's bootstrap lane dispatches in UUID order and does
   not have the sibling class at this rate. A pilot at `sort_order` therefore
   over-reports duplicates by ~10x relative to what the bootstrap will produce, and the
   next W-value or concurrency decision would be made on that inflated number. Matching
   the CLI's order to the worker's removes the discrepancy.
2. **Blast radius on the CLI and incremental paths.** ~17 → ~2 groups per 83 articles;
   at corpus scale, tens of thousands of fewer edge moves on the paths that have the
   adjacency (§3.5). The merge pass is transactional per group and byte-exact, so this is
   a cost reduction, not a risk reduction — except for the `§1.2` escalation tax, which
   for span-2 names is small.
3. **Throughput**: neutral to slightly positive (§3.3).

What it costs:

- ~15 lines in `ingest_source` (§9), unit tests, and one more thing to explain in the
  docstring that already explains the barrier.
- A small, unmeasured shift in extraction context (§6).
- Nothing in persistence, cursors, the worker, or graphiti.

Verdict: worth it *at this size*. It would not be worth a claim-order change in the worker
today (§8, alternative e), and it would not be worth content-aware separation (§8, d).

---

## 8. Question 7 — alternatives compared

| | what | pilot residual (sim, W=8) | risk | verdict |
|---|---|---|---|---|
| (a) | do nothing; merge pass | ~17 on CLI/incremental paths, ~2–3 on bootstrap lane | none new | acceptable; leaves pilots unrepresentative |
| (b) | **deterministic hybrid in `ingest_source`**: `sort_order` prefix of W, remainder ascending UUID | ~1–2 | one function, no persistence; extraction-context shift unmeasured | **recommend** |
| (c) | random shuffle | mean 3, p95 14 | loses reproducibility; tail as bad as today | reject |
| (d) | content-aware separation | ≤ (b); could also catch distance-2..4 pairs (16 entities) | needs a pre-extraction similarity signal (shared tables / shingled markdown) that does not exist; the at-risk class is 65% distance-1, which (b) already handles | reject as disproportionate; revisit only if a vendor's siblings turn out not to be TOC-adjacent |
| (e) | worker claim order: `ORDER BY (lane='bootstrap'), date_trunc('minute', next_attempt_at), md5(article_id)` | incremental lane ~17 → ~2 per 83; bootstrap lane unchanged | changes a production query's FIFO guarantee to bucketed-FIFO; starvation bounded by bucket; needs a test | defer, gated on a measurement (§10 experiment 2) |

---

## 9. Implementation sketch — if (b) is adopted

Sequenced so each step is independently revertible; total well under a day.

0. **Commit the simulation** as `scripts/dispatch_sim.py` (read-only, ~200 lines:
   §3.1 describes it completely). It is the zero-cost check for any future ordering or
   W-value question and it turned this proposal's estimate into a measurement. Pin its
   calibration row (recorded order, W=8, α=0.3 → 17) the way `risk_curve.py:21-23` pins
   the spec's 79.5%.
1. **`IngestDriver.ingest_source`** (`ingest_driver.py:196-199`): after `limit` is
   applied to the `sort_order` list (so `--limit N` keeps meaning "the first N articles of
   the source"), reorder:
   `ids = ids[:W] + sorted(ids[W:])` where `W = settings.ingest_warmup_articles`
   (0 → no prefix, plain `sorted(ids)`). Keep `zip(ids, results)` at `:215` on the
   reordered list. Document in the docstring: *why* (sibling pairs, the two measured
   pairs, the ~10x), *why ascending UUID* (the worker's bootstrap order), and *why the
   prefix is untouched* (§5).
2. **Tests** in `tests/unit/test_ingest_source_concurrency.py` (which already stubs
   `list_article_ids`, `:36,:53,:241`): dispatch order observed by a recording worker is
   `[first W by sort_order] + sorted(rest)`; `--limit` truncates before reordering;
   `W=0` gives `sorted(ids)`; results map back to the right article ids. Keep the ruff
   rule in mind — no semicolons in fakes.
3. **Docs**: one paragraph in `concurrent_ingest.py`'s module docstring (it documents the
   hazard the order interacts with) and a line in `CLAUDE.md` § "Sync correctness rules":
   *the structural stream is consumed in stream order; only semantic dispatch may be
   reordered* (§1.4).
4. **Nothing else.** No change to `sync_core`, `state_store`, `semantic_worker`, graphiti
   patches, or the merge pass.

What breaks if the order is violated: step 1 without step 2 leaves the `--limit`
semantics unpinned; step 1 before step 3 leaves the one dangerous variant (reordering the
structural stream) undocumented as forbidden.

Reversibility: fully reversible at any time — the graph shape produced is identical, the
gate is order-blind, and no state records which order a run used.

---

## 10. What could not be determined without running something

1. **Whether the extraction-context shift (§6) moves entity yield.** Cheapest experiment:
   the next paid pilot run in any case — run it with the hybrid order and compare entity
   count and duplicate count against the W=8 arm (1022 / 16). Expect duplicates ~1–3 and
   an entity delta inside the known ±30 churn. ~$10, ~2.7 h. Not worth a run of its own.
2. **The incremental lane's real sibling density** (§2.3 is a proxy). Cheapest
   experiment, free: after the next source is onboarded through the incremental lane, run
   `merge-duplicates` (plan only) and, for each group, look up the `sort_order` distance
   between the first-mention articles of its members. If most groups are distance 1–2,
   alternative (e) is justified; if not, it is not.
3. **Whether the bootstrap feed's ascending-UUID order is contractual** (§2.2). One
   sentence from the DocExtractor maintainers. If it is not, the worker's bootstrap lane
   is spread by luck, and (e) becomes the way to make it spread by design.
4. **Vendor-specific at-risk density** (§3.5). Free once the structural layer holds more
   vendors: the span-2 / sort-distance histogram in §3.5 needs only the semantic graph,
   so it can be recomputed after each vendor's ingest.

---

## 11. Open questions for the user

1. Ascending UUID (one order across CLI and bootstrap lane) or bit-reversal (a structural
   guarantee for TOC-adjacent pairs, ~1 fewer duplicate per 83 in the simulation)? This
   proposal recommends UUID for uniformity; the choice is a one-line difference.
2. Should the worker's claim order (alternative e) be changed now, ahead of the
   measurement in §10.2, given that every source onboarded after bootstrap will arrive
   TOC-ordered through that lane? The case for waiting: it is a production query with a
   FIFO property nobody has yet needed to weaken.
3. Is the §1.4 rule — *never reorder or parallelise the structural stream* — worth
   promoting to `CLAUDE.md` now, independent of this proposal? It is the only way the
   brief's cursor gap can be created, and nothing currently documents it as a boundary.
