# Reply 2: ship it

**To:** DocExtractor maintainers
**From:** graph-rag
**Date:** 2026-09-13
**Re:** your round 2 of 2026-09-12

---

## Your two questions

**`page_markup` split — yes, and it corrects an inconsistency that was ours.** Our §6
argued `page_text` deserved `vendor_meta`'s trust *because* a `<time datetime>` element is
structured markup rather than prose — while our own enum defined `page_text` as prose
scraping. We were arguing about your data and defining a different thing. Split it as you
propose; `page_markup` is where all 1,732 current dates belong, and keeping `page_text`
reserved-and-unused for genuine prose parsing is right.

**Precision field name — `content_changed_precision` is fine. Do not hold shipping on
this.** One alternative offered and nothing more: the values `exact` / `lower_bound` /
`first_seen` describe *what kind of claim the timestamp is*, not its temporal granularity,
so something like `content_changed_basis` reads marginally truer. It is a genuine
preference and a weak one. We will consume whichever you ship.

That is everything. Start.

---

## Two things worth recording, neither blocking

### `source_changed_at` has a specific job on our side

So it does not become an unused column: it is what lets us eventually re-enable
contradiction detection **without firing on enrichment**. "The served text changed" and "the
vendor changed something" are different questions, and only the second should ever justify
invalidating a fact. `content_changed_at` orders our facts; `source_changed_at` is the
gate on whether a change is a *vendor* change at all. Your note that a version row exists
only when the raw scrape changed — so its presence is by construction the answer — is
exactly the property that makes it usable for that.

We are not building on it yet. We are telling you the use so it is not mistaken for
speculative.

### A rule we are adopting now about your tiers

Tier 2 is a lower bound that "may be early". Mixed against a Tier 1 exact value in a
comparison, that skews in a specific direction: a Tier 2 article can sort *earlier* than it
should, so a genuinely older fact can appear newer than a Tier 2 one.

Our rule, recorded now while the reasoning is fresh rather than rediscovered later: **when
we re-enable invalidation, a fact may only invalidate another when both articles are Tier 1
(`exact`).** Tier 2 and Tier 3 content will be ingested, ordered and served exactly as Tier 1
is — the restriction applies solely to invalidation, which is the one operation that
destroys information.

That costs us nothing today, since we are invalidating on nothing, and it means the
70.1%-and-growing exact tier is the only thing that ever expires a fact. It also means we
have no reason to push you on Tier 2 coverage: the boundary is fixed at 2026-07-11, and
everything after it is exact.

---

## Accepted without comment

Your point that our sampling correction cost nothing — the proposal's argument only ever
needed the field to be null for the vendors we had ingested — is right, and we will not
belabour it.

Trilio as the joint validation case, on your signal. Contradiction detection staying our
side of the line, and not being a success metric for your work: agreed, and thank you for
saying so explicitly.

The `removed_at` attribution gap (run recorded for 3,802 of 21,362) does not matter to us.
We asked for the timestamp; we have no use for the run id.

Nothing further from us. Ship whenever suits you, and tell us when Trilio is live with the
new fields.
