# Maintenance Runbook

The semantic graph needs periodic housekeeping. All jobs are deterministic
(pure Cypher, no LLM) and idempotent — safe to re-run. Scheduling
(cron / systemd-timer / K8s CronJob) is a **deployment concern**; this runbook
documents *what* to run and *how often*, not the scheduler itself.

## The jobs

| Job | What it does | Command |
|---|---|---|
| **prune** | `DETACH DELETE`s `:Entity` nodes matching `noise_filter` (ARNs, error codes, IAM actions, API-field names, numeric/org IDs, regions-as-words). Returns an audit of pruned names. | (part of `cleanup`/`maintenance`) |
| **retype** | Relabels a mistyped region entity to `:Region` (removes the wrong custom type label), keyed on the `region_names` gazetteer. | (part of `cleanup`/`maintenance`) |
| **sweep** | Expires `RELATES_TO` facts whose supporting episodes are all dead (`removed`, or no `HAS_EPISODE` from a non-removed `:Article`) — sets `invalid_at` + `expired_by_sweep`. Never deletes; never touches Graphiti's own invalidations. | `sweep` |
| **reconcile** | Links structural `:Vendor`/`:Product` to the matching semantic `:Entity` via `SAME_AS` (alias-matched, link-not-merge). Reports unmatched structural nodes (add an alias in `vendor_aliases.py`). | `reconcile` |

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
  count and the `reconcile` `unmatched_structural` list — a growing unmatched
  list means `vendor_aliases.py` needs new entries.
- `queue-status` surfaces `dead` semantic-ingestion jobs (articles that failed
  `max_attempts` times) — investigate and re-enqueue as needed.
- The jobs are safe to run concurrently with ingestion (they're idempotent and
  the sweep's expiry is a single atomic statement that never clobbers Graphiti's
  own `invalid_at`), but running them during a quiet window is preferable.
