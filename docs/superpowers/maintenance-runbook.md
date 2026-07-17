# Maintenance Runbook

The semantic graph needs periodic housekeeping. All jobs are deterministic
(pure Cypher, no LLM) and idempotent — safe to re-run. Scheduling
(cron / systemd-timer / K8s CronJob) is a **deployment concern**; this runbook
documents *what* to run and *how often*, not the scheduler itself.

## The jobs

| Job | What it does | Command |
|---|---|---|
| **prune** | `DETACH DELETE`s `:Entity` nodes matching `noise_filter` (ARNs, error codes, IAM actions, API-field names, numeric/org IDs, regions-as-words). Returns an audit of pruned names. | (part of `cleanup`/`maintenance`) |
| **retype** | Enforces `:Region ⇔ region_names` gazetteer, both directions: **promotes** a mistyped region to `:Region` (removing the wrong custom label), and **demotes** an entity the model self-typed `:Region` that the gazetteer doesn't recognise (e.g. `Availability Zone`, `subscriptions`) to a bare `:Entity` + a `demoted_from_region` audit stamp. Demote is a reversible relabel (no delete) and self-heals: add the name to `region_names.py` and the next run re-promotes it. A guard skips all demotions in a run exceeding `max(10, 30% of :Region count)` — the signature of an `is_region` regression; if it trips, check the gazetteer before re-running. | (part of `cleanup`/`maintenance`) |
| **sweep** | Expires `RELATES_TO` facts whose supporting episodes are all dead (`removed`, or no `HAS_EPISODE` from a non-removed `:Article`) — sets `invalid_at` + `expired_by_sweep`. Never deletes; never touches Graphiti's own invalidations. | `sweep` |
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
