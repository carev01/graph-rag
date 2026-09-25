# Vendor and product, end to end — design

**Date:** 2026-09-25. **Status:** draft for review.
**Covers:** BACKLOG 52 (global/DRIFT shortlist not vendor-aware) and the user requirement
of 2026-09-25: *"We should be able to track vendor and product end-to-end. When producing
responses it is important to identify which vendors and specific products apply."*

## 1. Goal and principle

Every answer says which vendors and products each claim applies to, and a question that
names vendors or products is answered from their documentation only. Both come from the
structural layer by graph traversal, never from the LLM — the same rule as citations
(design invariant #2). The LLM receives vendor/product labels as input and must use them;
it never invents one.

**What "applies to" means.** A fact's applicability is the set of (vendor, product) pairs
over the Articles of its supporting episodes (`fact.episodes → Episodic ← HAS_EPISODE ←
Article ← HAS_ARTICLE ← Source ← HAS_SOURCE ← Product ← HAS_PRODUCT ← Vendor`),
excluding superseded/removed episodes. A fact stated by two vendors' docs legitimately
applies to both (invariant #4 unifies the entity; provenance keeps the vendors apart).

## 2. What exists today

| piece | state |
|---|---|
| `Provenance.resolve_citations` | already returns `vendor` and `product` per source |
| local synthesis prompt | fact lines are `[N] <fact>` — **no label**; the model cannot tell a Commvault fact from a Veeam one |
| global map/reduce | `_render_blocks` fact lines have no label either |
| `/answer` `vendor=` | honoured by local and timeline; **silently dropped for global and DRIFT** (`router._dispatch`) |
| `search_local(vendor=)` | one vendor name, via an episode-uuid set; no product scoping |
| global shortlist | cosine (+ rerank) over every community at the level — no scoping (BACKLOG 52) |
| `SAME_AS` structural ↔ entity links (invariant #5) | never built (0 in the graph) |
| `Product.version` | null everywhere (the catalog serves none) |

**Label collision to respect:** graphiti's extracted entities also carry `:Product` and
`:Vendor` labels (56 `:Vendor` nodes against 40 vendors). Everything here reads only
structural nodes — identified by `id` and reached through `HAS_PRODUCT` / `HAS_SOURCE` —
never by label alone.

## 3. Answer side: attribution

**3.1 Labelled fact lines, every mode.** Each fact line handed to an LLM — local
synthesis, global map and reduce, DRIFT primer/follow-ups/synthesis, timeline — carries
its applicability:

```
[3] (Veeam · Veeam Backup & Replication) Hardened repositories keep backups immutable for …
[7] (Veeam · VBR; Commvault · Commvault Cloud) Object lock requires versioning …
```

Resolved in one batched Cypher per call (the chain in §1), cached per request. Always
`Vendor · Product`, even when the product name repeats the vendor, so the format is
uniform and parseable.

Prompt rules added to each synthesis prompt: attribute every claim to the labelled
vendor/product; never generalise a fact labelled with one vendor to others; when a
question compares vendors, keep their claims apart.

**3.2 Response envelope.** Deterministic, from the cited facts only:

```json
"applies_to": [
  {"vendor": "Veeam", "products": ["Veeam Backup & Replication"], "facts": 9},
  {"vendor": "Commvault", "products": ["Commvault Cloud"], "facts": 4}
],
"scope": {"vendors": ["Veeam", "Commvault"], "products": [], "source": "detected"}
```

Each citation's `sources[]` already has `vendor`/`product`; unchanged. `scope.source` is
`explicit` (API params), `detected`, or `none`.

## 4. Query side: scoping

**4.1 Resolve the scope.** A `ScopeResolver` loaded at startup from the structural layer
(40 vendors, 79 products — reloaded on the freshness interval) plus a small alias file
(`config/scope_aliases.json`: `VBR` → Veeam Backup & Replication, `M365`, `AWS`,
`Azure` → Microsoft, …). It finds names in the question with whole-word,
case-insensitive, longest-match-first matching; a product implies its vendor. Explicit
API parameters — `vendor` and a new `product`, both repeatable — override detection;
`scope=none` disables it. Phrasing that is explicitly cross-vendor ("across vendors", "all
vendors", "which vendors") and names none stays unscoped.

**4.2 Local and timeline.** Generalise the existing single-vendor episode filter to the
resolved vendor/product sets (episode uuids via `Product → Source → Article → Episodic`).

**4.3 Global and DRIFT (BACKLOG 52).**
1. Widen the candidate pool (`global_scope_candidates`, default 24) before the cut.
2. One Cypher computes each candidate's **in-scope share**: the fraction of its
   `cited_fact_uuids` whose applicability intersects the scope (the §1 chain).
3. Keep communities with share ≥ `global_scope_min_share` (default 0.5), then rerank
   and cut to k as today. The observed offender — *Azure VM Recovery…* with 4 of 25
   facts Microsoft — has share 0.16 and drops out; *AWS Backup: Capabilities…* at 49/57
   stays.
4. In the map input, drop fact lines outside the scope, so a kept community cannot carry
   another vendor's facts into the answer.
5. Fewer than k survivors is fine; zero survivors makes global return no communities, and
   the router's existing fallback runs local — now scoped too.
6. DRIFT: the primer uses the same shortlist; follow-ups use scoped local search.

The router passes the scope to every mode (fixes the dropped `vendor=`).

Query-time computation, not a stored per-community vendor mix: it needs no schema
addition, no backfill and no rebuild, is always consistent with the current graph, and
costs one Cypher over ~24 communities × ~20 facts (tens of ms).

**4.4 Detected scope is soft (added after the final review).** Cross-vendor wording
("vendors", "across clouds/providers", "third-party tools", "all/other products") keeps
a question unscoped even when it names a platform ("Which backup vendors can protect
Azure VMs?"). A *detected* scope whose answer grounds nothing is re-run unscoped and
reported as `scope.source = "detected-relaxed"`; an explicit scope never is. This also
fires when the scoped path retrieved facts but synthesis correctly refused — the answer
then comes from other vendors, which the `(Vendor · Product)` labels and `scope.source`
make visible. Accepted: a question naming a product is better answered with visibly
labelled neighbouring evidence than refused.

## 5. Out of scope, deliberately

- **`SAME_AS` links (invariant #5).** Provenance already carries vendor/product for every
  fact; entity-level links add value for entity-centric questions ("what is Veeam
  Backup for Public Clouds?") and belong in their own slice.
- **Product versions.** The catalog serves none; nothing to track until it does.
- **Per-community vendor mix stored by theme-builder.** Revisit only if §4.3's query-time
  cost shows up in latency.

## 6. Evaluation

- AWS/Azure golden set: global grounding back from 0.29 to ≥ 0.8, routing and
  faithfulness unchanged.
- New Tier 1 golden questions once Veeam and Commvault are ingested: 6 scoped (one vendor
  or product), 4 comparisons (two named vendors), 2 explicitly cross-vendor (must stay
  unscoped).
- **Attribution check** added to the eval judge: it receives the labelled facts and
  scores whether any claim is attributed to a vendor/product its facts are not labelled
  with; reported as `misattributed` count per answer.
- Unit tests for the resolver (aliases, longest match, cross-vendor phrasing, product
  implies vendor), the share filter, label rendering; integration test over a
  two-vendor fixture graph.

## 7. Cost

No new LLM calls. Prompt tokens grow by the label, ~6–12 tokens per fact line. One or two
extra Cypher reads per request.
