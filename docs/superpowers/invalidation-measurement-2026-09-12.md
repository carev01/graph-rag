# BACKLOG 6 — measuring the ingest-time invalidations

**Read-only. No LLM calls.** Reproduce: `.superpowers/sdd/measure-invalidations.py`.

Run to decide BACKLOG 30 (`valid_at` semantics), because that change would take
invalidation coverage from 23% of the graph to 100% — multiplying whatever error rate
this mechanism has.

## What is solid

**All 140 invalidations are contradiction-driven.** Every edge with `invalid_at` also has
`expired_at`, which only `resolve_edge_contradictions` sets; `_extract_edge_timestamps`
sets `invalid_at` alone. So none of them came from the extractor reading an end-date out of
the fact text — all came from the LLM asserting a contradiction.

## What could NOT be measured, and why that is itself the finding

graphiti records **no link from an invalidated edge to whatever invalidated it**. The only
recoverable join is the assignment `edge.invalid_at = resolved_edge.valid_at`
(`edge_operations.py:566`), so the invalidator is any edge whose `valid_at` equals the
victim's `invalid_at`.

That join is unusable:

| candidate invalidators per victim | |
|---|---|
| min | 0 |
| **median** | **21** |
| mean | 22.6 |
| max | **172** |
| victims with exactly one candidate | **1 of 140** |

**An invalidation in this graph cannot be audited after the fact.** That is a real defect
independent of any error rate: the system can delete a fact's currency and leave no record
of why.

**My first classification pass is therefore withdrawn.** It reported "85.7% different
endpoints" and a date-kind mismatch rate; both were counting join fan-out, and the query
truncated candidates before filtering for the same node pair. Do not cite those numbers.

## The mechanism, and a correction to what `reference_time` is

I previously wrote that reference time is "the article's `last_updated_at`". **Wrong.**
Asked directly, DocExtractor returns:

```
last_updated_at = None          <- for every article checked
created_at      = 2026-06-25T23:20:42.959555Z
extracted_at    = 2026-08-17T23:53:20.360956Z
```

`_parse_ts` is `last_updated_at or extracted_at`, so **reference time is DocExtractor's
scrape time**, not the vendor's document revision date. The vendor revision date is not
available at all. (The sub-second precision is `extracted_at`'s, not a `datetime.now()`
fallback — an earlier reading of mine that was also wrong.)

This explains the invalidations without needing a date-kind theory:

1. The timestamp prompt says *"if the fact is ongoing (present tense), set `valid_at` to
   REFERENCE TIME"* — and vendor documentation is overwhelmingly present tense. That is why
   415 of 805 dated edges carry exactly the scrape time.
2. Within one bulk scrape, articles are scraped **seconds apart**, so those facts get
   `valid_at` values that differ by seconds in **scrape order**.
3. `resolve_edge_contradictions` invalidates when `edge.valid_at < resolved_edge.valid_at`.

So whenever the LLM asserts a contradiction between two present-tense facts from the same
scrape, **which one survives is decided by the order DocExtractor happened to crawl the
pages.** Not by content, not by document dates.

The most common `valid_at` value is shared by **172 edges**; the top five values cover
~320 of 805.

## Hand inspection

In the small subset where the join is most likely right (same endpoints **and** same
relation name, n=9), none of the inspected pairs is a contradiction:

- *"AWS Backup provides cold storage tiering for DynamoDB backups"* invalidated by
  *"AWS Backup provides a low-cost cold storage tier for storing backups to meet compliance
  requirements"* — both true.
- *"Azure Backup supports Azure database for PostgreSQL"* invalidated by *"...supports Azure
  Database for PostgreSQL **Flexible Server** backup"* — a refinement.
- *"Azure Backup supports viewing backup and site recovery jobs for Azure Disks
  (monitoring)"* invalidated by *"Azure Backup (via Backup center) supports monitoring backup
  jobs for Azure Managed Disks"* — a near-duplicate that **dedup should have merged**.

Six of six inspected are refinements or duplicates, not change events. n is small and
selected from the most-joinable subset, so treat it as directional.

## Consequence for BACKLOG 30

**Do not set `valid_at = reference_time` deterministically.** It would give 100% of facts a
timestamp that means "when DocExtractor crawled this page", make `valid_at` ordering within
a scrape pure crawl order, and extend a contradiction mechanism that currently fires on
refinements and duplicates from 23% of the graph to all of it.

The prerequisite is upstream: **there is no document revision date in the corpus.**
`content_hash` already detects *that* content changed between scrapes, which is the honest
temporal signal available today. Whether DocExtractor can supply a real `last_updated_at`
is a question for its owners.
