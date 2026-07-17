# Theme-Builder (Community Layer, Slice 1) — Demonstration Report

**Slice:** Phase 3, slice 1 — theme-builder (GDS hierarchical Leiden + GLM-5.2
community reports + `:Community` subgraph write-back).
**Date:** 2026-07-17
**Status:** Implemented, reviewed, merged locally. `@live` GDS detection smoke
green; demonstrated end-to-end on the live `backup-docs` graph.

## What it does

`theme-build` derives the thematic layer from the Graphiti entity graph: detect
hierarchical communities (GDS Leiden), generate an analytical MS-GraphRAG-style
report per community (GLM-5.2, per-finding fact-ID citations), and write a
`:Community` subgraph. Full rebuild (delete + rewrite, one atomic transaction).
No LLM at detection or write-back; citations are graph traversal (design
decision #2). The layer is the substrate for global (map-reduce) search and DRIFT
in the next slices.

## Live run on `backup-docs`

`uv run --extra dev python -m theme_builder.cli theme-build`:

```json
{ "communities_detected": 71, "reports_written": 54,
  "by_level": {"0": 32, "1": 12, "2": 10},
  "facts_cited": 393, "reports_skipped": 17 }
```

- **71 communities detected** across 3 Leiden levels; **54 reports written**
  (32 leaf / 12 mid / 10 top). Dust communities (`< min_community_size=3`) dropped
  at detection.
- **393 fact UUIDs cited** across all reports — every one validated against its
  community's real facts (hallucinated ids dropped) before write-back.
- `PARENT_OF` = 31 hierarchy edges; `IN_COMMUNITY` = 750 membership edges.
- **17 reports skipped** (GLM-5.2 returned unparseable JSON after one retry) —
  graceful degradation, counted, build continues. ~24% skip rate is the main
  quality follow-up (below).

## A real community + its resolvable citations

The highest-rated leaf community:

- **Title:** *Backup Plan Configuration and Retention Management*
- **Rating:** 8.0 · **Members:** 8 · **Tags:** AWS Backup, Azure Backup,
  Retention Policy, Continuous Backup, Backup Plan (cross-vendor — the QFS goal)
- **Summary:** "This report analyzes the configuration of backup plans, focusing
  on the relationship between backup frequency and retention periods, the
  management of continuous backups, and the impact of policy changes on recovery
  points across AWS and Azure environments."
- **A finding:** "A backup plan must contain at least one backup rule with a
  backup frequency of at least 1 day and a retention period of at least 35 days."
  — cites fact `ec48c290-…`.

**Citations resolve to source URLs** (design decision #2, `community →
cited_fact_uuid → episode → article → source_url`, via the existing `Provenance`
resolver — no LLM-authored URLs):

| cited fact uuid | resolves to |
|---|---|
| `51065546-…` | *Changing your retention period* — docs.aws.amazon.com/aws-backup/…/point-in-time-recovery |
| `ec48c290-…` | *Controls and remediation* — docs.aws.amazon.com/aws-backup/…/controls-and-remediation |
| `fd114ead-…` | *Changing your retention period* — docs.aws.amazon.com/aws-backup/…/point-in-time-recovery |

This is the whole point: a thematic answer built on community reports can cite
concrete, verifiable source URLs, not hand-wave at "the community."

## Design-decision alignment (verified)

- **#2 citations are traversal:** reports emit fact UUIDs; a deterministic
  validator keeps only real ones; the LLM never writes a URL (prompt + `_URL_RE`
  strip); global search will resolve `cited_fact_uuids` to URLs. Demonstrated above.
- **#4 one group / one embedding space:** `:Community` `title+summary` embedded
  with the SAME shared TEI/Jina embedder as entities/facts/queries — comparable
  for the next slice's embedding-similarity report shortlisting.
- **Derived & disposable:** a rerun deletes and rewrites the `:Community` subgraph
  in one transaction; the theme-builder is its only writer; Graphiti's schema is
  untouched (`:Community` linked via `IN_COMMUNITY`, never merged).

## Verification summary

- **`@live` GDS smoke:** `detect_communities` on `backup-docs` → multi-level
  communities, correct parent nesting (GDS 2.13; the projection uses the modern
  `gds.graph.project` aggregation form marked undirected, leiden with
  `includeIntermediateCommunities`).
- **Unit:** context assembly (ranking / current-first / budget / oversized-fact
  clip), report validation (drop hallucinated fact ids, strip URLs, robust JSON,
  non-dict findings), detect helpers (deterministic id, dust-drop, ragged-level
  parent derivation), config defaults.
- **Integration (testcontainer):** `:Community` write-back (atomic full rebuild,
  `IN_COMMUNITY`/`PARENT_OF`, embedding) and the `theme-build` orchestration
  (per-community fetch → assemble → report → write).
- Full non-live suite green; ruff/mypy clean.

## Follow-ups (next slices / hardening)

- **Global map-reduce `/search/global`** over this layer (the next slice) —
  shortlist reports by embedding + rating, map/reduce, cite fact UUIDs → URLs.
- **Report skip rate (~24%):** GLM-5.2 JSON adherence — add a stricter
  JSON/schema mode or a second retry, or route reports to a more JSON-reliable
  model tier (the report tier is already config-swappable).
- **Incremental refresh:** dirty-marking from graph-sync, Jaccard-stable community
  ids, regenerate only dirty/new reports, `corpus_cursor`-driven staleness (this
  slice is full-rebuild + content-hash ids).
- Roll-up reports for large parents; structural vendor attribution beyond
  model-inferred tags; DRIFT entry using community top-members as center nodes.
