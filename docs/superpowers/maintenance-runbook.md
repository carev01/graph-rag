# Maintenance Runbook

The semantic graph needs periodic housekeeping. All jobs are deterministic
(pure Cypher, no LLM) and idempotent — safe to re-run. Scheduling
(cron / systemd-timer / K8s CronJob) is a **deployment concern**; this runbook
documents *what* to run and *how often*, not the scheduler itself.

## The jobs

| Job | What it does | Command |
|---|---|---|
| **prune** | `DETACH DELETE`s `:Entity` nodes matching `noise_filter` (ARNs, error codes, IAM actions, API-field names, numeric/org IDs, regions-as-words). Returns an audit of pruned names. | (part of `cleanup`/`maintenance`) |
| **retype** | Enforces `:Region ⇔ region_names` gazetteer, both directions: **promotes** a mistyped region to `:Region` (removing the wrong custom label), and **demotes** an entity the model self-typed `:Region` that the gazetteer doesn't recognise (e.g. `Availability Zone`, `subscriptions`) to a bare `:Entity` + a `demoted_from_region` audit stamp. Demote is a reversible relabel (no delete) and self-heals: add the name to `region_names.py` and the next run re-promotes it. A guard skips all demotions in a run exceeding `max(2, 50% of :Region count)` — the signature of an `is_region` regression (fraction-dominant, so it protects small bootstrap/per-vendor groups too); if it trips, check the gazetteer. If the demotions are a *legitimate* large backlog, re-run `cleanup --force-demote` to bypass the guard. | (part of `cleanup`/`maintenance`) |
| **sweep** | Expires `RELATES_TO` facts whose supporting episodes are all dead (an episode is dead when its `HAS_EPISODE` edge is flagged `superseded`, or the episode or its article is `removed`) — sets `invalid_at` + `expired_by_sweep`. Never deletes; never touches Graphiti's own invalidations. | `sweep` |
| **reconcile** | Links structural `:Vendor`/`:Product` to the matching semantic `:Entity` via `SAME_AS` (alias-matched, link-not-merge). Reports unmatched structurals, classified in `unmatched_detail`: `no_candidate` (no alias-named entity exists — **expected** when the corpus never names the vendor as an actor, e.g. `AWS`; not an error) vs `wrong_type_candidate` (an alias-named entity exists but is the wrong kind, e.g. Vendor `Microsoft` vs semantic `Azure:Platform` — **not** auto-linked, since that would assert a false identity; a human decides). | `reconcile` |

## CLI

```bash
# post-ingest correctors only (prune + retype) -- cheap, run after each incremental batch:
uv run --extra dev python -m graph_extract.cli cleanup

# full housekeeping (prune + retype + sweep + reconcile) -- one entry point:
uv run --extra dev python -m graph_extract.cli maintenance

# individual jobs, if you want them on separate cadences:
uv run --extra dev python -m graph_extract.cli sweep
uv run --extra dev python -m graph_extract.cli reconcile

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
- **Simple deployment:** run **`maintenance` weekly** (it does all four in order)
  and `cleanup` after each batch.

## Operational notes

- Every job prints a JSON audit (counts + samples). Watch the `sweep` `expired`
  count and the `reconcile` `unmatched_detail` — a **`wrong_type_candidate`** is
  the actionable signal (an alias or an upstream extraction-typing gap);
  `no_candidate` entries are expected and need no action. Watch `retype`'s
  `demoted_names` (junk `:Region` cleaned up) and `demote_guard_tripped` (should
  be `false`; `true` means investigate the gazetteer before trusting the run).
- `queue-status` surfaces `dead` semantic-ingestion jobs (articles that failed
  `max_attempts` times) — investigate and re-enqueue as needed.
- The jobs are safe to run concurrently with ingestion (they're idempotent and
  the sweep's expiry is a single atomic statement that never clobbers Graphiti's
  own `invalid_at`), but running them during a quiet window is preferable.

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
