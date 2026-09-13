# Reply: article change timestamps

**To:** DocExtractor maintainers
**From:** graph-rag (the downstream consumer)
**Date:** 2026-09-12
**Re:** your response of 2026-09-12 to our proposal of the same day

---

## Answers first

**§5 semantics — define `content_changed_at` over the SERVED markdown.** Your lean is
right, and for a reason stronger than verifiability: our re-ingestion gate is already keyed
to it. One question below could change this answer, and it is about your backfill, not our
preference.

**§6 `page_text` — yes, trustworthy enough to ingest and to order facts by.** With a caveat
that is about our state, not your data quality.

**Removed articles — yes, we want `content_changed_at` on them.** A retracted page is
signal, and we never delete history by design.

---

## 1. Corrections we owe you

**"`last_updated_at` is null corpus-wide" was wrong.** It is populated for 1,732 articles
across five vendors, with genuine editorial dates back to 2020. We sampled 40 articles from
two sources and generalised from them — the same error class we have been catching in our
own analysis all week, committed in a document asking you to trust our measurements. The
proposal's §3 should be read as "null for AWS and Microsoft", which is all we actually
established.

That correction improves our position, not just our manners: it means the field's contract
is exercised end to end today, and there is real data to validate against before anything
ships.

**Your crawl-clustering figure supersedes ours.** We reported 172 edges sharing one
timestamp. You measured 2,470 articles in a single crawl minute. Use yours — it is the
corpus-wide number and ours was a sample artefact.

---

## 2. §5 — why served markdown, and the one thing that could change it

### The decisive reason is our re-ingestion gate

`CLIENT-USAGE-GUIDE.md` §"content_hash — the served-content fingerprint" defines the delta
record's `content_hash` as the SHA-256 of the **served** markdown, and our ingestion gates
on exactly that (`ingest_driver.py:101`, `provenance.already_ingested`). So:

> If `content_changed_at` were defined over the raw scrape, there would be a class of
> articles — your 28,757, 22.7% — where **we re-ingest because the bytes changed, while the
> timestamp says nothing changed.**

The new facts from that re-ingestion would carry a `valid_at` identical to the facts they
supersede. Two generations of fact, indistinguishable in time. That is precisely the failure
mode we are trying to escape, reintroduced through a different door — and unlike crawl-order
clustering it would be invisible, because the timestamp would look stable and correct.

Served-markdown semantics make the timestamp agree with the bytes we hash, so a change we
act on is always a change we can date.

### The cost, stated plainly

A caption injection moves the timestamp though the vendor edited nothing. That is a real
cost and we accept it, for two reasons.

First, from our side it is not entirely spurious: the caption enters the markdown we extract
facts from, so the text that produced our graph genuinely changed. "When did the content we
read become current" is the question our graph needs answered — not "when did the vendor
edit", which we cannot obtain for most of the corpus anyway.

Second, the blast radius is bounded by our dedup: if a caption adds no new facts, the
re-extraction resolves to the existing edges and nothing is superseded. If it does add facts
(your image descriptions are substantive), those are genuinely new facts and deserve a new
timestamp.

We would rather have a timestamp that is occasionally too eager and always consistent with
the bytes, than one that is semantically pure and silently disagrees with them on a quarter
of the corpus.

### The question that could change this answer

**Is the served-markdown definition still exactly backfillable?**

Your Q1 answer says the backfill comes from `article_versions.content_hash`, and your §5
says that hash fingerprints the **raw** scrape. If the archived version rows store raw
content and raw hashes, then deriving a served-markdown `content_changed_at` historically
requires re-running caption injection and URL rewriting over archived content — which is
only exact if both steps are deterministic and the image data they consume is retained at
the same version.

If that holds: build served-markdown semantics, and we are done.

If it does not, we would rather you tell us than approximate. The options we can see, in our
order of preference:

1. **Served semantics going forward, raw-derived for the backfill**, with a field or a
   documented cutover date saying which regime an article's value came from. We can handle a
   labelled discontinuity; we cannot handle an unlabelled one.
2. **Both fields** — `content_changed_at` (served) and something like `source_changed_at`
   (raw) — if you are storing enough to compute both anyway. We would order by the first and
   use the second to answer "did the vendor actually change anything", which is a question
   we would like to be able to ask later even though we do not need it now.
3. **Raw semantics with the divergence documented**, and we carry the 22.7% discrepancy
   ourselves by treating our own served-hash change as the trigger and your timestamp as a
   lower bound.

Do not build (2) speculatively. It is only worth it if both values already fall out of what
you store.

### An offer: we can test the semantics before you build

`GET /api/articles/{id}/versions` already returns `(content_hash, extracted_at,
extraction_run_id)` per snapshot. That is enough for us to derive a candidate
`content_changed_at` client-side for a sample, compare it against our own served-markdown
hashes, and measure how often the two definitions actually disagree in practice — not just
how many articles are theoretically affected.

If that would be useful before you commit to a definition, say so and we will run it against
a few hundred articles and send you the numbers. It costs us nothing.

---

## 3. §6 — `page_text`, and a caveat about us rather than you

**Yes, ingest-worthy.** A `<time datetime>` element is structured markup, not a date
inferred from prose. For the purpose we need — ordering facts in time — we would treat
`vendor_meta` and `page_text` as equally trustworthy, because both are the vendor asserting
a date in a machine-readable field. The tier distinction matters for provenance display, not
for whether we believe the value.

**The caveat is our own state.** As of today we have **suspended ingest-time contradiction
detection entirely** — precisely because of the crawl-order problem you confirmed. Until we
have measured a residual invalidation path in our own code, we will not invalidate a fact on
*any* timestamp signal, `vendor_meta` included. So "we would ingest `page_text` but not
invalidate on it alone" is still accurate, but it currently understates our caution: we are
not invalidating on anything.

That is not a reason for you to weaken the provenance enum. Keep it — it is exactly what we
will need when we re-enable, and the distinction should be recorded at ingest time, not
reconstructed later.

**Both declines endorsed.** On `http_last_modified` your reasoning is better than ours: we
ranked it low, you measured it returning today's date for AWS and correctly concluded that a
fabricated timestamp wearing a provenance label is worse than an honest null. Drop the enum
value rather than rank it. The sitemap deferral is right for the same reason — check whether
the `lastmod` is a revision date or a generation date before building the fetcher.

---

## 4. Removed articles — yes, and here is the use

We never delete history: an article's tombstone marks its episodes `removed: true` rather
than removing them (`ingest_driver.py:129`), because deleting history defeats the
"how did vendor X's treatment change over time" question the whole system exists to answer.
A retracted page is signal — "this guidance was withdrawn" is a change event, and often a
more interesting one than an edit.

There is also a concrete mechanism that needs it. We run a weekly sweep that expires facts
whose only supporting episodes are all removed. Today it stamps those expirations with the
*sweep's* run time, which is another crawl-schedule artefact of exactly the kind this whole
exchange is about. A real removal timestamp would let us date the expiry properly.

So: `content_changed_at` on removed articles, please, and if the removal itself carries a
distinct timestamp we would take that too.

---

## 5. A joint validation case

Your five dated vendors are not in our graph — we have ingested AWS and Microsoft only. That
means neither side can currently test the full path on an article with a real editorial
date.

**Trilio looks like the right validation case**: 163 articles, dates from 2026-08-10 to
2026-08-18, `page_text` tier. Small enough for us to ingest cheaply, recent enough that the
dates are live, and narrow enough that a wrong result is obvious rather than statistical.

If you are willing, we would ingest Trilio once `content_changed_at` ships and report back
what our graph's time axis looks like with genuinely dated content — which is the only way
either of us finds out whether this works end to end.

---

## 6. What this unblocks on our side, and what it does not

Your work removes the blocker on our temporal layer. It does **not**, on its own, let us
turn contradiction detection back on, and we want to be explicit about that so no one is
surprised later.

A correct timestamp fixes the *ordering* of two facts. It does not stop a model asserting
that a refinement contradicts the thing it refines — we have measured that happening
("supports Azure Database for PostgreSQL" invalidated by "supports Azure Database for
PostgreSQL **Flexible Server**"). With real dates those pairs still fire whenever the two
facts come from different revisions; they simply arrive wearing a credible date, which makes
them harder to spot, not easier.

Re-enabling needs the timestamp **and** a measurement of how many genuine contradictions
this corpus contains at all. We are instrumenting for the second now. Your half is the half
we could not do ourselves.

---

## Summary of what we are asking for

| | |
|---|---|
| `content_changed_at` | **Served markdown**, pending your backfill answer in §2 |
| Backfill regime | Tell us if served-semantics backfill is inexact; a labelled discontinuity is fine, an unlabelled one is not |
| `last_updated_source` | Build as proposed, minus `http_last_modified` — your call, better reasoned than our ranking |
| Microsoft head metadata | Yes, please |
| Sitemap `lastmod` | Agreed deferral |
| Removed articles | Yes, and a removal timestamp too if one exists |
| Versions endpoint | Noted and useful — we may use it to test the §5 semantics before you build |
