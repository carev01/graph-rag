# `/timeline` — Demonstration Report

**Slice:** Phase 2 retrieval — bi-temporal `/timeline` endpoint
**Date:** 2026-07-16
**Status:** Implemented, merged locally. `@live` smoke green; demonstrated against the live graph.

## What it does

`GET /timeline?q=<topic>&vendor=<optional>&limit=30` hybrid-retrieves the fact
edges relevant to a topic — **including invalidated ones** — annotates each with
its validity interval and supersession status, orders them ascending by
`valid_at`, and resolves citations deterministically. It is the endpoint that
surfaces the append-not-overwrite temporal model (design-decision #3). No LLM at
query time; citations are graph traversal (design-decision #2).

Status vocabulary (from `_fact_status`):

- **current** — `invalid_at IS NULL` (still holds).
- **superseded** — `invalid_at` set by graphiti (a later/contradicting fact won).
- **expired** — `invalid_at` set by our staleness sweep (`expired_by_sweep=true`;
  the supporting docs went away). Read via a batched Cypher lookup on the
  returned uuids, because `expired_by_sweep` is our custom `RELATES_TO` property,
  not a graphiti `EntityEdge` field.

## Live demonstration

### A real supersession chain — "AWS Backup CloudTrail events and API calls"

`timeline_local(q="AWS Backup CloudTrail events and API calls", limit=12)` — 12
facts, ordered ascending by `valid_at`, all three statuses present
(`{superseded: 8, expired: 2, current: 2}`):

```
[superseded] 2019-01-10 08:24:39  -> invalid 2026-07-12 16:00:35
             AWS Backup is integrated with CloudTrail which captures AWS Backup API c...
             <Logging AWS Backup API calls with CloudTrail>
[superseded] 2019-01-10 13:45:24  -> invalid 2026-07-12 16:00:28
             StartBackupJob is an AWS Backup API action represented as a CloudTrail e...
             <Logging AWS Backup API calls with CloudTrail>
[superseded] 2020-06-02 00:34:00  -> invalid 2022-06-11 13:29:23
             backup.amazonaws.com produces service events recorded as CloudTrail...
             <Logging AWS Backup API calls with CloudTrail>
[expired   ] 2021-05-01 00:00:00  -> invalid 2026-07-16 22:01:19
             Blog 'Manage Amazon EFS backup costs...' ...
             <Blogs, videos, tutorials, and other resources>
[expired   ] 2022-01-01 00:00:00  -> invalid 2026-07-16 22:01:19
             Blog 'Optimizing SAS Grid on AWS with FSx for Lustre' ...
             <Blogs, videos, tutorials, and other resources>
[superseded] 2022-06-01 00:00:00  -> invalid 2026-07-12 16:00:28  ...
[superseded] 2026-07-12 16:00:17  -> invalid 2026-07-12 16:00:26  ...
[superseded] 2026-07-12 16:00:17  -> invalid 2026-07-12 16:00:22  ...
[superseded] 2026-07-12 16:00:17  -> invalid 2026-07-12 16:00:18  ...
[superseded] 2026-07-12 16:00:17  -> invalid 2026-07-12 16:00:22  ...
[current   ] 2026-07-12 16:00:21
             If the permission is missing, CreateRestoreAccessBackupVault failures...
             <Requester tasks>
[current   ] 2026-07-12 16:00:27
             By default, AWS Backup restores Namespace-Scoped Kubernetes resources...
             <EKS restore>
```

What this shows, concretely:

- **Chronological ordering works** — facts run 2019 → 2026 ascending by `valid_at`.
- **The bi-temporal interval is exposed** — e.g. the `backup.amazonaws.com`
  service-event fact was valid `2020-06-02 → invalid 2022-06-11`, a genuine
  event-time supersession window read straight off the edge.
- **All three statuses resolve correctly** — graphiti-superseded, sweep-expired
  (the two blog facts our weekly sweep expired on 2026-07-16 because their source
  docs disappeared), and still-current facts co-exist in one view.
- **Citations are resolved by traversal** — every row carries its source article
  title (`Logging AWS Backup API calls with CloudTrail`, `EKS restore`, …), never
  authored by an LLM.

### A common-case topic — "AWS Backup Vault Lock immutability compliance mode"

`limit=8` → `{superseded: 6, expired: 2}`, valid_at ascending
`['2021-10-01', '2021-10-01', '2026-07-12' ×6]`. Confirms ordering and status
classification hold on an unrelated topic.

## Honest caveat (unchanged from the design spec)

Today's invalidations are largely **event-time supersession inside document
content** (the CloudTrail event pair hours apart) plus article `reference_time`,
**not** documentation evolving over months. That kind of change accrues as the
incremental pipeline runs over time. So `/timeline` **works today** on the
event-time/supersession data, and its "how did vendor X's treatment change over
time?" value **grows with accumulated history** — the mechanism is the
deliverable now. The live run above proves the mechanism end to end.

## Verification summary

- **`@live` smoke:** `pytest -m live -k timeline` → 1 passed.
- **Non-live suite:** 297 passed, 6 deselected; ruff + mypy clean.
- **Unit:** `_fact_status` (current / superseded / expired) and `timeline_local`
  ordering + status + citations + `limit` truncation, with a stubbed
  `_retrieve_edges` and a real Neo4j driver for citations.
- **Refactor safety:** `_retrieve_edges` extracted from `search_local` with no
  behaviour change — existing `search_local` tests stayed green.
- **`GET /timeline` TestClient:** 200 `{query, count, timeline}`; missing `q` →
  422; `/search/local` + `/answer` + `/health` still 200; DB/GLM-free.

## Deferred (per spec §8)

- LLM narrative timeline (synthesized "here's how X evolved", reusing the GLM
  synthesis layer + design-decision #2 markers) — the natural next add.
- Time-window filtering (`?since=&until=`), `superseded_by` fact linkage,
  richer grouping.
