# Proposal: article change timestamps in DocExtractor

**For:** DocExtractor maintainer
**From:** the graph-rag consumer (temporal GraphRAG over the DocExtractor corpus)
**Date:** 2026-09-12
**Status:** proposal for review — nothing implemented on either side

---

## 1. The ask, in one paragraph

The downstream knowledge graph is *temporal*: its headline use case is "how did vendor X's
treatment of Y change over time?", and it decides which of two conflicting facts is current
by comparing timestamps. Today the only per-article timestamps available are
`created_at` and `extracted_at`, both of which describe **when DocExtractor crawled the
page**, not when the vendor changed it. `last_updated_at` exists in the API contract but is
`null` for every article sampled. The consumer is therefore ordering facts by crawl order,
which produces wrong answers. **The most valuable thing DocExtractor could add is not a
guess at the vendor's date, but a precise record of when the served content itself changed
— something DocExtractor already has the data to compute exactly.**

## 2. Why it matters — the concrete failure

The graph stores facts as edges with `valid_at` / `invalid_at`. When two facts conflict, the
one with the earlier `valid_at` is marked invalid — that is how "this used to be true, now
it isn't" is represented.

`valid_at` is derived from the article's timestamp. Because `last_updated_at` is null, it
falls back to `extracted_at`. Within one bulk crawl, articles are fetched **seconds apart**,
so:

> When two facts conflict, **which one survives is decided by the order the crawler happened
> to visit the pages.**

Measured on the live graph (2026-09-12): 140 facts have been invalidated this way; 415 of
805 dated edges carry exactly a crawl timestamp, **172 of them sharing a single value**.
Inspected examples are not change events at all —

- *"Azure Backup supports Azure database for PostgreSQL"* was invalidated by *"...supports
  Azure Database for PostgreSQL **Flexible Server** backup"* — a refinement.
- *"AWS Backup provides cold storage tiering for DynamoDB"* was invalidated by *"AWS Backup
  provides a low-cost cold storage tier for compliance"* — both true.

This is a downstream bug too (the consumer should not invalidate on evidence this weak, and
that is being fixed separately). But no downstream fix can manufacture a time axis that
isn't in the data.

## 3. Evidence about what's available today

Sampled live via `GET /api/articles/{id}`:

| | result |
|---|---|
| `last_updated_at` populated | **0 of 40** articles, across both sampled sources (AWS, Microsoft) |
| `created_at` | always present — first time DocExtractor saw the article |
| `extracted_at` | always present — most recent extraction, microsecond precision |
| `content_hash` | always present — SHA-256 of served markdown |

The documented examples in `CLIENT-USAGE-GUIDE.md` (§ delta feed, lines ~240, ~289, ~442)
also show `"last_updated_at": null`, including for Dell — so this looks corpus-wide rather
than specific to one scraper.

**The vendor date is mostly not in the served markdown either.** Scanning content for update
markers across 24 articles: no `Last updated:` marker in any; an ISO date appeared somewhere
in 2 of 12 AWS articles and a `Month D, YYYY` string in 1 of 12 Microsoft articles — and
those are almost certainly content dates (release notes, log examples), not page-revision
dates. So text extraction is not a reliable route.

## 4. What the consumer actually needs

Ranked. **(a) is worth more than (b)**, which is the main point of this document.

**(a) A trustworthy "the content changed at T" signal.** To order facts and detect genuine
change, the consumer needs to know *when the text it is reading became the current text*. It
does **not** need to know the vendor's editorial date to do this.

**(b) The vendor's own revision date, where obtainable.** Strictly better when available —
it distinguishes "the vendor rewrote this page" from "we re-crawled it" — but it is a
best-effort signal and must be labelled as such.

**(c) Explicit provenance for whichever timestamp is supplied.** A consumer must be able to
tell a vendor-declared date from an inferred one, because they justify different actions.
Silently mixing them is worse than having neither: it produces confident wrong answers, which
is the situation today.

## 5. Proposal A — `content_changed_at` (recommended)

**Definition:** the timestamp of the earliest extraction run in which the article's *current*
`content_hash` was observed.

DocExtractor already stores `content_hash` per article and runs extractions with a `run_id`
and timestamp, so this is derivable **exactly, with no vendor cooperation and no inference**.

```
content_changed_at   # first observation of the CURRENT content_hash
```

Semantics:

- A re-crawl that returns identical bytes **does not** move it — that is what makes it
  different from `extracted_at`.
- A re-crawl with different bytes sets it to that run's time.
- For an article seen only once, it equals the first extraction time.

**Why this is the high-value item:** it converts the corpus from "we crawled these pages in
some order" to "this text has been current since T". That is a real, monotonic, verifiable
time axis. For the temporal use case it is *nearly as good as* a vendor revision date, and
unlike a vendor date it is available for **100% of articles from day one**.

**Caveat to state in the docs:** it is bounded below by crawl frequency. If a page changed in
March and was first re-crawled in August, `content_changed_at` says August. It is "changed no
later than T", not "changed at T" — and that is exactly the kind of thing that should be
documented rather than left for a consumer to discover.

### A1 (optional, later): change history

If cheap, a per-article list of `(content_hash, first_seen_at)` would let a consumer
reconstruct the actual revision timeline rather than only the latest change. Not required for
the current use case; worth noting while the schema is being touched.

## 6. Proposal B — populate `last_updated_at`, best effort, with provenance

Fill the existing field where a vendor signal genuinely exists, and **always** say where it
came from:

```
last_updated_at          # ISO-8601, or null
last_updated_source      # enum, see below — required whenever last_updated_at is non-null
```

Suggested `last_updated_source` values, in descending trust:

| value | meaning |
|---|---|
| `vendor_meta` | HTML metadata on the page (e.g. `<meta name="last-modified">`, OpenGraph `article:modified_time`, JSON-LD `dateModified`) |
| `sitemap_lastmod` | `<lastmod>` from the vendor's sitemap for that URL |
| `http_last_modified` | the HTTP `Last-Modified` response header |
| `page_text` | a date parsed out of visible page text (e.g. a footer "Last updated" line) |
| `vendor_api` | a vendor-provided API or docs-repo commit date, where one exists |

Notes:

- AWS and Microsoft documentation both generally expose a modification date in page metadata
  or sitemaps even though it is absent from the extracted markdown — worth checking before
  assuming it is unavailable. I could not verify this from the consumer side; I only know it
  is absent from the served markdown and from the API.
- `http_last_modified` is weak for CDN-served docs (it often reflects cache behaviour) — hence
  ranking it below sitemap data, and hence the need for the provenance field rather than a
  single opaque timestamp.

## 7. Explicitly NOT proposed

**Do not backfill `last_updated_at` with `extracted_at` or `created_at`.** That would make the
field non-null everywhere and destroy the consumer's ability to tell a real vendor date from a
crawl artefact — converting a visible gap into an invisible wrong answer. The current
`null` is *correct behaviour* and should stay until a real signal exists.

**Do not synthesise a plausible date when none is found.** Null is a useful, actionable value;
a fabricated timestamp is not. (This is the same principle the consumer applies internally: a
measurement that could not be taken is recorded as "unmeasured", never as a legitimate value.)

## 8. Contract impact

Both fields are additive, so existing consumers are unaffected:

- `GET /api/articles/{id}` — add `content_changed_at`, `last_updated_source`.
- `GET /api/articles/delta` — same fields on each NDJSON record. The consumer reads the delta
  feed as its primary path, so this is the one that matters most.
- `CLIENT-USAGE-GUIDE.md` — document the semantics above, especially the "changed no later
  than T" caveat and the provenance enum.

Backfill for `content_changed_at` is possible only as far back as retained run history; for
articles where that is unknown, returning `null` is better than guessing. If history was not
retained, the field can start accruing meaning from the next run onward — still an
improvement over today.

## 9. How the consumer would use it

- `content_changed_at` becomes the article timestamp that facts inherit, replacing the crawl
  time. Facts from an unchanged page keep a stable date across re-crawls, so re-ingestion
  stops reordering the graph.
- `last_updated_at` with `last_updated_source` in (`vendor_meta`, `sitemap_lastmod`,
  `vendor_api`) is preferred over `content_changed_at` when present.
- `page_text` / `http_last_modified` would be ingested but flagged lower-confidence, and not
  used to invalidate an existing fact on their own.

## 10. Questions for you

1. Does DocExtractor retain per-run `content_hash` history, or only the current hash? That
   determines whether `content_changed_at` can be backfilled or only accrues going forward.
2. Do the scrapers currently capture page metadata / sitemap `lastmod` and discard it, or is
   it never fetched? If it is fetched, Proposal B may be mostly a plumbing change.
3. Is `last_updated_at` null because no scraper populates it, or because a populate step
   exists and is failing silently? The answer changes whether this is a feature or a bug.
4. Is there a vendor in the corpus that *does* supply revision dates? Even one would let both
   sides validate the end-to-end behaviour against a known-good case.
