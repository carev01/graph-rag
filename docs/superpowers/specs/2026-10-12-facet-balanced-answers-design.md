# Facet-balanced answers — design

**Status:** approved in conversation 2026-10-12 (sections 1–3); this document records it.
**Goal:** perfect answers on cross-vendor questions — every named side of a comparison
answered, every vendor that matches an enumeration listed — without regressing
single-vendor questions, and without spending money to find out.

## 1. Why answers miss today (measured, 2026-10-12)

A free retrieval trace over the Tier 1 golden set (no LLM calls;
`scratchpad/retrieval_trace.py`, to become `eval_retrieval`) after Tier 1 run 5
(scoped grounding 0.70, target 0.8) shows three independent causes:

1. **Global keeps 4 communities ranked by whole-question similarity, unbalanced.**
   `rerank_top_n = 4`, and a multi-vendor scope admits a community whose cited facts
   are >= 50% in *any* scoped vendor, so all 4 can come from one side:
   - "Veeam Backup for Microsoft 365 immutability vs Azure Backup soft delete":
     4/4 Microsoft communities; Veeam never reaches the answer.
   - "Veeam backup copy vs AWS Backup cross-Region copy": 4/4 Veeam.
   - "Which backup vendors support immutable backups?": Cohesity FortKnox + 3 Veeam;
     the Azure WORM and AWS vault-protection communities exist but rank below 4th.
   - "What approaches do backup vendors take to verify restores?": 4 Veeam.
2. **Narrow scopes starve local search.** `search_local` fetches `k*3 = 45` edges and
   filters by scope afterwards. Scoped to the six Veeam Agent guides, 7 of 45 survive,
   none from the expected articles — the run 4 regression on "Veeam Agent
   immutability", caused by the AWS/Azure ingest crowding the 45, not by reports.
3. **Local on comparisons splits by chance** (ransomware question: 8 Microsoft / 7 Veeam
   edges; cross-Region copy: 10 AWS / 5 Veeam, 0/3 expected articles).

Vendors with extracted content today: Veeam (20,192 articles), Microsoft (468),
AWS (147), Cohesity (92).

## 2. Question shapes

Before retrieval, every question gets a **shape**, decided deterministically (no LLM):

| Shape | Rule | Example |
|---|---|---|
| **enumeration** | no explicit scope, and the cross-vendor wording `scope._CROSS_VENDOR` matches ("vendors", "which/what/any vendor", "across vendors/clouds/…", "competing products", …) | "Which backup vendors support immutable backups?" |
| **comparison** | two or more **facets** (below) | "Compare Veeam hardened repositories and AWS Backup Vault Lock" |
| **single** | zero or one facet | "How does Veeam SureBackup verify backups?" |

An explicit scope (`vendor=`/`product=`/`source=`) is the caller's instruction: it is
one facet and the shape is **single**, whatever the wording. `scope=none` gives
**single** with an empty facet (today's unscoped behaviour).

### Facets

A **facet** is a label plus its own `Scope`. Facets come from scope detection's
*mentions*, not its flattened result:

- Each regex match in `ScopeResolver.detect` is one mention; the terms it maps to form
  one facet. An alias with several targets stays one facet ("Veeam Agent" -> six
  Agent guides, label "Veeam Agent").
- Identical facets merge. A **vendor** facet is dropped when another facet is a product
  or source *of that vendor* (it is subsumed: "Veeam's Veeam ONE" is one facet).
- Facet order follows first mention in the question.
- `ScopeResolver.detect` keeps returning the union `Scope` (every existing caller is
  unchanged); a new `detect_facets(q) -> list[Facet]` returns the grouping.

For an **enumeration**, the facets are one per **vendor with extracted content**
(structural `Vendor` reached through `HAS_PRODUCT/HAS_SOURCE/…/HAS_ARTICLE/HAS_EPISODE`),
loaded with the scope catalog at startup.

`answer_api/shape.py` holds `Facet`, `QuestionShape(kind, facets)` and the pure
`classify_shape(q, resolver, *, explicit_scope, scope_disabled, content_vendors)`.

## 3. Retrieval per mode

The router still picks local / global / drift / timeline. Shape is a second input that
every mode honours, so a comparison stays balanced whichever mode answers it.
Enumeration has its own path (§3.4) regardless of the routed mode.

### 3.1 Local
- **Per-facet retrieval.** Each facet runs `search_local` with its own scope and
  `k_facet = max(local_min_per_facet, ceil(k / n_facets))` (`local_min_per_facet = 6`;
  k stays 15). Results merge in facet order, deduplicated by fact uuid.
- **Adaptive over-fetch for scoped searches.** When the scope filter leaves fewer than
  `k` edges, re-fetch at the next limit in `(k*3, k*10, k*30)` capped at
  `local_scoped_fetch_cap = 450`, stopping as soon as `k` survive. Unscoped searches
  are unchanged (one fetch).
- **Facet-grouped synthesis.** The answer prompt receives fact lines under facet
  headings (`=== Veeam Backup & Replication ===`) and the rule: address every facet; if
  a facet has no facts, say plainly that the documentation retrieved does not cover it.
  Single-facet prompts are byte-identical to today's.

### 3.2 Global
- **Per-facet shortlist.** Each facet runs `shortlist_communities` with its own scope
  and `top_n = global_facet_top_n` (3) for comparisons; single keeps `rerank_top_n` (4).
- Maps run per community as today. The reduce prompt gets blocks grouped by facet and
  the same address-every-facet rule.
- The scope filter on map output uses the community's facet scope.

### 3.3 DRIFT
- The primer shortlists per facet (as §3.2), and each follow-up query carries the
  scope of the facet whose community produced it, so one side cannot consume every
  follow-up. Synthesis gets facet-grouped facts as in §3.1.

### 3.4 Enumeration (own path)
1. **Gather per vendor** (concurrently): the vendor's top
   `enum_communities_per_vendor = 2` communities and top `enum_facts_per_vendor = 8`
   facts for the question, both scoped to that vendor.
2. **Check per vendor**: one cheap-tier call (`map_llm_*`) per vendor with that
   evidence, returning either `{"documented": true, "points": [...], "fact_ids": [...]}`
   or `{"documented": false}`. It reuses the map prompt's guards: only what the
   evidence states, fact ids must come from the evidence, absence is not a finding.
3. **Answer**: the synthesis tier writes one cited entry per vendor with
   `documented: true`, in the order given, then a closing line generated
   deterministically (not by the LLM):
   `Checked: Veeam, Microsoft, AWS, Cohesity. Documented for: Veeam, AWS, Cohesity.`
   No ceiling: every content vendor is checked.

## 4. Unchanged invariants

Citations resolve deterministically from fact ids; the LLM never writes a URL. A
*detected* scope that grounds nothing is relaxed and re-run unscoped; an explicit scope
never is. Attribution labels, `applies_to`, freshness and the global->local /
local->drift fallbacks keep their behaviour. The envelope gains `shape`,
`facets` (labels), and for enumerations `checked_vendors`, `documented_vendors`.

## 5. Failure handling

A failure must never read as absence.
- A facet whose retrieval raises, or a vendor whose check returns unusable output
  after one retry, is reported in the answer ("Could not check Cohesity."), listed in
  `degraded_facets`, and logged with the cause. It is never folded into "not
  documented".
- An enumeration where no vendor is documented returns a statement naming the checked
  vendors ("None of Veeam, Microsoft, AWS, Cohesity document …") — not the generic
  refusal.
- `content or ""`-style coercion is forbidden; empty replies go through
  `usable_content` / `_complete_or_none`.

## 6. Testing and validation

Cheapest first; no paid run until the free gates pass.

1. **Unit tests (hermetic):** `classify_shape` on all golden questions plus edges
   ("Veeam Agents" is one facet; "across clouds" with no name is enumeration; an
   explicit scope forces single; vendor subsumed by its product); facet slot
   allocation; adaptive over-fetch (fills, stops early, respects the cap, unscoped
   fetches once); enumeration assembly with fakes, including the "could not check"
   and "none documented" paths; facet-grouped prompt rendering.
2. **`eval_retrieval` (free, committed):** the trace as pass/fail per golden question —
   expected-article recall, every facet has evidence, enumeration vendors found ==
   ground truth. This is the development loop.
3. **Paid Tier 1 eval (~$0.20), once at the end**, with two new citation-derived
   metrics: *facet coverage* (every facet cited) and *enumeration completeness*
   (every ground-truth vendor listed, none extra). At most one more paid run after a
   diagnosed miss before reporting back.

### Ground truth and golden set
- Vendor-level ground truth for each enumeration question is built from the corpus
  with free Cypher searches and confirmed by the user before it is scored.
- About four new questions — two enumerations, two comparisons in the other direction
  (AWS vs Veeam, Cohesity vs Veeam) — are proposed to the user before being added.
- An expected-article list that looks too narrow is shown to the user with the
  evidence; the user decides. Targets are not moved silently.

### Targets
Routing 1.00; faithfulness 5.0; misattributed claims 0; scoped grounding 1.00; facet
coverage 1.00; enumeration completeness 1.00; no regression on Veeam-only questions.

## 7. Cost

Retrieval is free (local embedder, Neo4j; the Voyage reranker is fractions of a
cent). Comparisons: one extra shortlist per facet, map calls ~4 -> ~6 per global
question. Enumerations: one cheap-tier check per content vendor (4 today, <= 40 at full
corpus). No theme rebuild.

## 8. Settings

`local_min_per_facet = 6`, `local_scoped_fetch_cap = 450`, `global_facet_top_n = 3`,
`enum_communities_per_vendor = 2`, `enum_facts_per_vendor = 8` (all in
`ExtractSettings`).
