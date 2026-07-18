# `/answer` Router Niceties — Demonstration Report

**Slice:** Phase 4, slice 3 — three deferred router enhancements: a **freshness
block**, **richer citations** (vendor/product/section + valid_at/invalid_at), and a
**global→local fallback** arc.
**Date:** 2026-07-18
**Status:** Implemented, reviewed, merged locally. Full non-live suite green
(431 passed); `@live` smoke green; demonstrated on `backup-docs`.

## What it adds

1. **Freshness** (`answer_api/freshness.py`): every `/answer` response carries
   `freshness: {graph_cursor_time, reports_as_of}`. `graph_cursor_time` = latest
   `Episodic.created_at` (the live-retrieval "graph as-of"); `reports_as_of` =
   latest `Community.generated_at`, non-null only for community-based modes
   (global/drift, by final mode). A freshness query failure degrades to `null` —
   it never fails an answer.
2. **Richer citations**: a single enrichment of `Provenance.resolve_citations` —
   each source now carries `vendor`/`product`/`section` (from the structural graph
   `Article→Source→Product→Vendor` + the `HAS_EPISODE.heading_path`), and each
   citation carries `valid_at`/`invalid_at` (from the fact edge). All four modes
   and the `/answer` envelope get the richer shape automatically.
3. **global→local fallback**: when a `global` classification shortlists no
   communities (its refusal, empty `communities_used`), the router escalates to
   `local` (`routing.fallback_from="global"`). Mutually exclusive with the existing
   local→drift arc — at most one escalation per dispatch.

Design decision #2 holds: every new citation field is graph-derived; no LLM
authors any of it, and the router still authors no prose/URL.

## Live run on `backup-docs`

**Global query** — *"Compare how AWS Backup and Azure Backup handle retention"*
(→ global via heuristic):

```
freshness = {
  "graph_cursor_time": "2026-07-16T13:12:53.754654Z",   # latest episode
  "reports_as_of":     "2026-07-17T23:32:42.615Z"        # thematic layer build (global is community-based)
}
```

A **richer citation** from that answer:

```
marker 1  valid_at=2026-07-12T16:00:22Z  invalid_at=2026-07-12T18:07:06Z
  source:
    vendor  = AWS
    product = AWS Backup
    section = Changing your retention period
    url     = https://docs.aws.amazon.com/aws-backup/latest/devguide/point-in-time-recovery-retention-period.html
```

The citation is now human-readable ("AWS › AWS Backup › Changing your retention
period") and temporally scoped (valid 16:00–18:07 on 07-12) — versus the bare
`{url, title, article_id}` before this slice. All fields come from graph traversal.

**Local query** — *"Does AWS Backup support Vault Lock in compliance mode?"*
(`mode=local`):

```
freshness = {"graph_cursor_time": "2026-07-16T13:12:53Z", "reports_as_of": null}
```

`reports_as_of` is correctly `null` — local retrieval doesn't touch the community
report layer.

**global→local fallback** — this arc fires only when the community shortlist is
genuinely empty. With the live embedder, even an off-topic query
("xyzzy quux frobnicate compare vendors") still returns its nearest 3 communities
(`communities_used=3`, `fallback_from=None`), so the arc doesn't trigger against a
populated layer. It is covered by the unit test
`test_global_empty_escalates_to_local` (a global that returns empty
`communities_used` → local dispatched, `routing.fallback_from="global"`, final
`freshness.reports_as_of=None`). Live it would fire only on an empty/level-missing
`:Community` layer.

## Verification summary

- **Unit — freshness** (`test_freshness.py`): both queries; community query skipped
  when `reports=False`; a raising driver → both fields `None` (no exception).
- **Integration — enriched `resolve_citations`** (`test_resolve_citations.py`):
  batch case (absent uuid stays absent, dangling fact → `sources:[]`, edge-derived
  valid_at/invalid_at) + enrichment case (full `Vendor→Product→Source→Article-[HAS_EPISODE
  {heading_path}]->Episodic` → source carries vendor/product/section, fact carries
  valid_at). The shape change was applied atomically across all 7 files
  (`provenance`, `search`, `timeline`, `global_search`, `drift`, `synthesize`,
  `router._render_timeline`); 34-test regression set green.
- **Unit — router** (`test_router_dispatch.py`): envelope carries `freshness`;
  `reports_as_of` set for global/drift, `None` for local (by final mode);
  global-empty→local with `fallback_from="global"`; global-with-communities stays
  global; mutually-exclusive arcs.
- **`@live` smoke** (`test_router_niceties_live.py`): a global query returns a
  `freshness` block with both fields, no URL in the answer, and ≥1 citation whose
  source carries vendor/product.
- Full non-live suite: **431 passed, 10 `@live` deselected**; `ruff check src tests`
  + `mypy` clean.

## Resulting citation shape (all modes, uniform)

`{marker, fact_uuid, valid_at, invalid_at, sources:[{url, title, article_id,
vendor, product, section}]}` (`answer_local` also keeps `fact`). Additive for
consumers — existing `sources[0]["url"]` access still works.

## Follow-ups (deferred, per spec §9)

- Single-product→local heuristic; classifier confidence score (skipped — the cheap
  LLM already routes single-product queries to local).
- Caching the `graph_cursor_time` aggregation (`max(Episodic.created_at)` scan) if
  it proves slow at scale.
- The superseded article version's URL for temporal citations.
- Minor (final-review triage): the identical 5-line citation-rebuild block in
  `global_search`/`drift` could share a helper; a stale source-keys comment at
  `test_global_search.py:63`.
- MCP exposure (Phase 5).
