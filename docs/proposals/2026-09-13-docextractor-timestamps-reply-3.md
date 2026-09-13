# Reply 3: the precision-vs-provenance decision

**To:** DocExtractor maintainers
**From:** graph-rag
**Date:** 2026-09-13
**Re:** your ship notice of 2026-09-13

---

## Your decision point — our §9 rule was wrong, and we are withdrawing it

Our §9 said we would prefer `last_updated_at` over `content_changed_at` whenever the source
is `vendor_meta`. **Do not build anything around that. It is withdrawn.**

You are right that it trades exact ordering for provenance, but the deeper problem is one we
diagnosed in our own graph last week and were about to walk straight back into.

Our `valid_at` field currently holds two incompatible kinds of date: an in-world date the
extractor read out of the fact text ("supported since 2019") for 48% of dated edges, and a
crawl timestamp for the other 52%. Comparing them is meaningless, and doing so is what
produced the phantom invalidations that started this whole exchange. A field holding two
kinds of date is worse than a field holding the less precise one.

The §9 rule would have recreated exactly that: `valid_at` meaning "vendor's declared day" on
some articles and "served bytes became current, to the microsecond" on others, with the
switch depending on which vendor a page came from.

**So we are taking all three fields and giving each one job:**

| field | our use | why |
|---|---|---|
| `content_changed_at` | the fact's `valid_at` — the **ordering axis** | Exact, always present, monotonic, and agrees with the `content_hash` we gate re-ingestion on |
| `source_changed_at` | **gate on invalidation** | Only a vendor change may ever expire a fact. Enrichment must not. |
| `last_updated_at` | stored alongside, **never used for ordering** | The vendor's own claim, surfaced in answers and available for filtering |

`last_updated_at` is not demoted — it is the only field that can answer "when did the vendor
say this changed", and no derived timestamp substitutes for it. It simply is not the thing
we sort by.

Your day-granularity point is what makes this concrete rather than theoretical. Ties in
`valid_at` mean "no invalidation" in graphiti's comparison, so a day-granular ordering axis
would silently stop detecting same-day changes — degradation that looks exactly like working
correctly. We have had enough of those.

---

## The Veeam finding matters more here than anywhere else

34 sources publishing a real per-page revision date in JSON-LD, invisible to every check
either of us ran. For a backup-documentation corpus, Veeam is not one vendor among 194 — it
is one of the largest, and among the most likely to be asked about.

It also means our eventual Tier-1-only invalidation rule will have genuine vendor dates to
cross-check against on a substantial slice of the corpus, which we did not expect to have.
That is a better position than we were planning for.

---

## On the three you refused

The Commvault case is the one worth naming. A visible "Updated" date, real, on every page —
and it is site-deploy mtime, twelve pages landing within six seconds. All 42,384 articles
would have claimed the same update day and moved wholesale on every rebuild. It would have
looked like the single biggest win in the audit.

And Arcserve: present on one sampled page, absent on eight when resampled. "It would have
shipped had we trusted the single observation" is the same failure we committed in our
original proposal, when we sampled 40 articles from two vendors and told you the field was
null corpus-wide. You resampled; we did not.

The test you settled on — **a revision date must vary per page** — is a better rule than
anything in our proposal, and it is the one we will apply to any future signal.

---

## What we are doing next

Consuming the fields is a change to our ingestion, not a switch: `content_changed_at`
becomes the episode reference time, `last_updated_at` and `source_changed_at` get persisted
on the article, and our sweep starts using `removed_at` instead of its own run time. We will
do that work before ingesting anything new, so the first run under the new contract produces
a clean time axis rather than a mixed one.

Then Trilio, on your signal.

One thing we will report back that you cannot see from your side: what the graph's `valid_at`
distribution looks like once it is built on `content_changed_at` — specifically whether the
crawl-minute clustering you measured (2,470 articles in one minute) disappears from our fact
timestamps, which is the actual proof that this worked.
