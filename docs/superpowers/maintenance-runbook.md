# Maintenance Runbook

The semantic graph needs periodic housekeeping. All jobs are deterministic
(pure Cypher, no LLM) and idempotent — safe to re-run. One of them,
`merge-duplicates --apply`, is destructive and is **not** safe alongside
ingestion; it has its own section below. Scheduling
(cron / systemd-timer / K8s CronJob) is a **deployment concern**; this runbook
documents *what* to run and *how often*, not the scheduler itself.

## The jobs

| Job | What it does | Command |
|---|---|---|
| **prune** | `DETACH DELETE`s `:Entity` nodes matching `noise_filter` (ARNs, error codes, IAM actions, API-field names, numeric/org IDs, regions-as-words). Returns an audit of pruned names. | (part of `cleanup`/`maintenance`) |
| **retype** | Enforces `:Region ⇔ region_names` gazetteer, both directions: **promotes** a mistyped region to `:Region` (removing the wrong custom label), and **demotes** an entity the model self-typed `:Region` that the gazetteer doesn't recognise (e.g. `Availability Zone`, `subscriptions`) to a bare `:Entity` + a `demoted_from_region` audit stamp. Demote is a reversible relabel (no delete) and self-heals: add the name to `region_names.py` and the next run re-promotes it. A guard skips all demotions in a run exceeding `max(2, 50% of :Region count)` — the signature of an `is_region` regression (fraction-dominant, so it protects small bootstrap/per-vendor groups too); if it trips, check the gazetteer. If the demotions are a *legitimate* large backlog, re-run `cleanup --force-demote` to bypass the guard. | (part of `cleanup`/`maintenance`) |
| **sweep** | Expires `RELATES_TO` facts whose supporting episodes are all dead (an episode is dead when its `HAS_EPISODE` edge is flagged `superseded`, or the episode or its article is `removed`) — sets `invalid_at` + `expired_by_sweep`. Never deletes; never touches Graphiti's own invalidations. | `sweep` |
| **reconcile** | Links structural `:Vendor`/`:Product` to the matching semantic `:Entity` via `SAME_AS` (alias-matched, link-not-merge). Reports unmatched structurals, classified in `unmatched_detail`: `no_candidate` (no alias-named entity exists — **expected** when the corpus never names the vendor as an actor, e.g. `AWS`; not an error) vs `wrong_type_candidate` (an alias-named entity exists but is the wrong kind, e.g. Vendor `Microsoft` vs semantic `Azure:Platform` — **not** auto-linked, since that would assert a false identity; a human decides). | `reconcile` |
| **merge-duplicates** | Finds `:Entity` nodes sharing a byte-identical `name` within one `group_id` — what concurrent ingest creates on hub entities (`AWS Backup`, `Azure Backup`, ...) — and reports them. **Only with `--apply`** does it merge: one survivor keeps its uuid, every `RELATES_TO`/`MENTIONS`/`IN_COMMUNITY`/`SAME_AS` edge is rewired onto it with its full property map, the losers are deleted, and the losers' communities are flagged `stale`. Report mode writes nothing. Case/whitespace variants are listed under `near_duplicates_not_merged` and never merged. | `merge-duplicates` (report) / `merge-duplicates --apply` — **the one job NOT safe alongside ingestion; see [Duplicate entities](#duplicate-entities-merge-duplicates)** |

## CLI

```bash
# post-ingest correctors only (prune + retype) -- cheap, run after each incremental batch:
uv run --extra dev python -m graph_extract.cli cleanup

# full housekeeping (prune + retype + sweep + reconcile) -- one entry point:
uv run --extra dev python -m graph_extract.cli maintenance

# individual jobs, if you want them on separate cadences:
uv run --extra dev python -m graph_extract.cli sweep
uv run --extra dev python -m graph_extract.cli reconcile

# exact-name duplicate entities: report only (writes nothing; safe any time).
# The merge itself (--apply) is NOT safe alongside ingestion -- read the
# "Duplicate entities" section before running it.
uv run --extra dev python -m graph_extract.cli merge-duplicates

# queue/dead-letter visibility (graph_sync):
uv run --extra dev python -m graph_sync.cli queue-status
```

## Recommended cadence

- **`cleanup` (prune + retype):** after every incremental ingestion batch. It's
  cheap and keeps noise/mistyped regions from accumulating between deeper passes.
- **`sweep`:** **weekly.** It expires facts whose docs were silently deleted or
  whose supporting chunks were detached — a slow-moving concern, so weekly is
  ample. (A fact only becomes sweep-eligible once *all* its supporting episodes
  are dead.)
- **`reconcile`:** **weekly**, or after a batch that added new vendors/products.
  It's idempotent, so extra runs are harmless.
- **`merge-duplicates`:** after every ingest that ran with
  `ingest_article_concurrency > 1`, and whenever `ingest`'s closing line or a
  `cleanup`/`maintenance` audit reports a non-zero `duplicates.totals.excess`.
  Report first; `--apply` only in a quiet window (below). `theme-build` refuses
  to run until the count is zero, so this is not optional once it is non-zero.
- **Simple deployment:** run **`maintenance` weekly** (it runs the jobs above in
  order and reports duplicates) and `cleanup` after each batch. Neither ever merges;
  `merge-duplicates --apply` is run by a person.

## Operational notes

- Every job prints a JSON audit (counts + samples). Watch the `sweep` `expired`
  count and the `reconcile` `unmatched_detail` — a **`wrong_type_candidate`** is
  the actionable signal (an alias or an upstream extraction-typing gap);
  `no_candidate` entries are expected and need no action. Watch `retype`'s
  `demoted_names` (junk `:Region` cleaned up) and `demote_guard_tripped` (should
  be `false`; `true` means investigate the gazetteer before trusting the run).
- `queue-status` surfaces `dead` semantic-ingestion jobs (articles that failed
  `max_attempts` times) — investigate and re-enqueue as needed.
- Every job **except `merge-duplicates --apply`** is safe to run concurrently
  with ingestion (they're idempotent and the sweep's expiry is a single atomic
  statement that never clobbers Graphiti's own `invalid_at`), but running them
  during a quiet window is preferable. `merge-duplicates --apply` is the one
  exception: it deletes `:Entity` nodes, and an ingest overlapping it loses
  edges silently — see [Duplicate entities](#duplicate-entities-merge-duplicates).

## Duplicate entities (`merge-duplicates`)

```bash
# 1. report -- writes nothing, safe at any time:
uv run --extra dev python -m graph_extract.cli merge-duplicates

# 2. merge -- ONLY in a quiet window: no `ingest`, no graph_sync semantic
#    worker, no other writer running. See the warning directly below.
uv run --extra dev python -m graph_extract.cli merge-duplicates --apply
```

> **`--apply` must never run while any ingest or semantic worker is running.**
> It is the one job in this runbook that is **not** safe alongside ingestion.
> graphiti saves every fact and mention with
> `MATCH (source:Entity {uuid: ...}) ... MERGE (...)`. If an in-flight episode
> has resolved an entity to a node that `--apply` then deletes, that `MATCH`
> yields zero rows and **the edge is silently not written — no error, no log, no
> retry.** The fact is extracted, paid for, and lost. The command cannot detect
> this condition (there is no lock; spec §11); it prints the constraint before
> it applies, and checking it is the operator's job: confirm no `ingest` process
> and no `graph_sync` semantic worker is running, and that the queue is idle
> (`queue-status`), before `--apply`.

**The default is report-only.** `merge-duplicates` without `--apply` opens a
read-only session and prints a JSON audit: each group's `name`, `members`
(uuid, `created_at`, labels, degree, summary length), the planned `survivor`,
the `name_embedding_cosine` between members, `label_conflicts`,
`near_duplicates_not_merged` (case/whitespace variants — listed, never merged),
and `totals` (`groups`, `excess` = nodes that would be removed). `--apply`
prints the same audit plus `totals.merged`, `edges_moved` by type,
`self_loops_created`, `summary_dropped`, `labels_promoted` and
`properties_dropped`.

**What `--apply` does.** Concurrent ingest (`ingest_article_concurrency > 1`)
can create two `:Entity` nodes with the same byte-identical `name` in the same
`group_id` — the A/B measured 30 at N=4, all on hub entities. These are not a
cosmetic blemish: graphiti escalates to an LLM dedup call on *every* later
mention of an ambiguous name, so each duplicate is a permanent tax on the
hottest names in the corpus, and community detection over the split graph
yields wrong communities, not stale ones. The merge is pure Cypher, one
transaction per name group: the survivor (earliest `created_at`, ties by lowest
uuid; a member with no `created_at` never survives) keeps its uuid; every
`RELATES_TO`, `MENTIONS`, `IN_COMMUNITY` and `SAME_AS` edge on a loser is
recreated on the survivor with
its full property map and uuid; the loser is removed with `DELETE` (not
`DETACH DELETE`), so a loser still carrying an edge type the pass does not know
aborts that group and rolls it back whole; the losers' communities are flagged
`stale` so the next incremental `theme-build` regenerates exactly them. It is
idempotent (a second run reports zero and changes nothing), a no-op on a clean
graph, and safe to interrupt (committed groups are complete; a re-run does the
rest). On the first failing group it stops and raises, after printing the
audit of the groups already committed.

**Recommended sequence** after a concurrent source ingest:

1. `merge-duplicates` — read the report. Check `label_conflicts` (members
   whose custom labels disagree: the merge promotes nothing and the survivor
   keeps its own labels — a bare survivor is only promoted when every typed
   loser agrees) and `near_duplicates_not_merged` (not concurrency's doing;
   left alone).
2. `merge-duplicates --apply` — in the quiet window described above.
3. `cleanup` — prune + retype over the merged graph; its audit's
   `duplicates.totals.excess` should now read `0`.
4. `theme-build` — regenerates the communities flagged `stale` by the merge.

**Where duplicates surface.** `ingest` (CLI) prints the count as its last line
— `N exact-name duplicate entities in group '...'`, followed by the remedy
(`run merge-duplicates to see them`) when N is non-zero. `cleanup` and
`maintenance` carry the full report-mode payload under the
`duplicates` key of their JSON audit — the report, never the merge: a job on a
timer must not delete nodes on its own. `theme-build` refuses to run while the
count is non-zero (exit 1, naming the count and the remedy);
`--allow-duplicates` overrides it for an operator who has read the report and
decided the duplicates are immaterial to the communities at hand.

**Prevention, and re-deriving the warm-up size.** The warm-up barrier
(`ingest_warmup_articles`, default 8) runs the first `W` articles of each
never-before-ingested source strictly sequentially, because the hub entities a
source keeps mentioning are named in its opening overview pages: on the
baseline, the first 8 articles of each source carry 79.5% of the duplicate-risk
weight (spec §1.1). If a concurrent run still leaves many more duplicates than
the model predicts, re-derive `W` from that run's own spans instead of paying
for another A/B:

```bash
uv run --extra dev python scripts/risk_curve.py            # ExtractSettings.group_id
uv run --extra dev python scripts/risk_curve.py --group G  # an isolated A/B group
```

It is read-only (the session is opened READ) and prints, for `c` in 1..30, the
cumulative share of `Σ(k-1)` risk weight carried by entities that first appear
within the first `c` articles of their source; pick `W` at the knee. Against
the baseline it prints `c=4 → 71.3%`, `c=8 → 79.5%`, `c=12 → 82.5%`.

## Extraction routing (ingestion)

`ingest` uses a **hybrid extraction router** (design:
`specs/2026-07-17-hybrid-extraction-router-design.md`): each article is sent to a
cheap model (`ling-2.6-flash`) or the strong model (`gpt-5-mini`) by table
density, since the cheap model over-generates on dense availability matrices (see
`ling-production-readiness.md`). Both tiers write the same group / embedding space.

- **Default ON.** Set `CHEAP_LLM_API_KEY` (OpenRouter) in `.env` to enable the
  cheap tier; without it (or with `EXTRACTION_ROUTING=false`) the pipeline runs
  **strong-only** — byte-for-byte the pre-router behaviour. `ingest` echoes
  `routing=hybrid` or `routing=strong-only`.
- **Routing rule:** an article goes to the strong tier when its table-line ratio
  ≥ `DENSE_TABLE_LINE_RATIO` (default 0.25) OR its `|` count ≥ `DENSE_PIPE_COUNT`
  (default 200); everything else goes to the cheap tier. The decision is
  deterministic (markdown only, no LLM). `IngestArticleResult.tier` reports which
  handled each article.
- The cheap tier uses smaller chunks (`CHEAP_MAX_CHUNK_TOKENS`, default 900) and a
  salience appendix to the extraction instructions; the strong tier uses the
  production instructions and `MAX_CHUNK_TOKENS` (1800). Tune the `DENSE_*`
  thresholds if the corpus widens past AWS/Azure.
