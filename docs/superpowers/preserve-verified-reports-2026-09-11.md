# Preserving verified reports through a failed regeneration (BACKLOG 5c / 5d)

**Date:** 2026-09-11
**Branch:** `preserve-verified-reports`
**Scope:** `theme_builder/{cli,incremental,writeback}.py`

## The defect

Staging (2026-09-10) fixed exactly one way a community could be destroyed: the verifier
blipping. Two others were left, and both run on the **default** `theme-build` path:

- the `except Exception` wrapped around report generation, and
- `_generate_once` returning `None` — **measured 3/29 empty-`choices` replies** on a real
  theme-build.

In both, `rep is None` with nothing staged, so the community was omitted from `entries`.
`write_communities_incremental` rebuilds the layer with `MATCH (c:Community) DETACH DELETE
c` and writes back only what it was given, so an omitted community was **deleted along with
its previously verified report** — one that had been generated, verified and paid for. The
loss was also invisible: `write_communities_incremental` receives only survivors, so it
could not report `lost_by_level`, the counter added after level 1 silently drained.

## The rule

**A failed NEW attempt never removes what is already persisted.**

| state on `rep is None`, nothing staged | outcome |
|---|---|
| persisted report exists, **verified** | carried over **with its embedding** — stays retrievable — and flagged `stale` |
| persisted report exists, **staged** | carried over in staged shape from `pending_summary` / `pending_full_report` |
| **nothing** persisted | genuine loss: counted, logged with its level, reported in `lost_by_level` |

New counters: `reports_preserved`, `lost_by_level`.

Carrying a staged report over required `load_persisted` to read `pending_summary` /
`pending_full_report`, which it previously did not. Without that it would have been written
as a *normal* report — and a staged row's `summary` reads back as `''` under the query's
`coalesce`, so the community would have been published **empty and retrievable**. That is
the same class of defect as the rest of this project's expensive ones: missing data coerced
into a legitimate-looking value.

## The non-obvious half: a preserved report must not look fresh

Carrying the report over is not enough, and getting only that half would have been worse
than a visible loss.

A preserved report describes the member set it was written against, and the run that
carries it stamps a **fresh `corpus_cursor`**. On the next run `touched_entities` measures
against that new watermark, so the very edits that made the community dirty now fall
*behind* it. The community would read as clean — permanently — and never regenerate.

So a new `stale` flag is persisted, read by `load_persisted`, and honoured by `classify`,
which already treats "no embedding" and "not verified" as always-dirty. The report keeps its
embedding throughout: **only its dirtiness is forced, never its retrievability.**

`verified` was deliberately not reused for this. It means "its findings passed
verification", which is still true of a preserved report; overloading it would have made the
graph say something false, and `--verify-pending` reads it.

## Verification

- `uv run ruff check src tests`, `uv run mypy src` clean.
- Eleven new tests: 3 integration on the incremental path, 2 integration on `--full`,
  6 unit (`classify` + `write_communities`' `preserved`).
- **Discrimination proven**, per the standing rule that a test must fail without its fix:
  each carry-over branch was neutralised in turn and the covering tests failed — three on
  the incremental path, one on `--full`. Restored, all pass, `git diff` of `src/` clean of
  the experiment.

## The `--full` path holds the same rule

`write_communities` rebuilds the whole layer with `DETACH DELETE` and writes back only what
it is handed, and `_run_theme_build` never loaded the persisted layer — so it could not have
carried anything over even in principle. One failed generation deleted the community
outright, report included. On a 41-community corpus at the measured blip rate that is
several communities gone per run. `--full` means "regenerate everything"; it does not mean
"destroy what we have if the regeneration fails".

It now loads the persisted layer, matches it against the fresh detection with the same
`match_communities` the incremental path uses, and passes survivors through a new
`preserved` argument. The shape rules are identical: verified keeps its embedding,
`generated_at` and `stale:true`; staged stays staged.

**Found while fixing it:** `PARENT_OF` edges were written only between communities present
in `reports`. A staged community already lost its place in the hierarchy, and a preserved
one would have too — on the very run that saved it from deletion. The gate is now the full
set of communities the run wrote.

## What this does NOT fix

**A husk from a genuine rejection is preserved too.** `--verify-pending` rejecting a report
leaves a node with no summary, no full report and no embedding. Carry-over keeps it rather
than deleting it. It is unreachable from every answering path and always-dirty, so it
regenerates on the next run; keeping it costs nothing and preserves the community's
membership edges.
