# DRIFT Search `/search/drift` — Demonstration Report

**Slice:** Phase 4, slice 1 — DRIFT search (primer → graph-biased follow-up loop →
cited synthesis) for broad questions that expect concrete, sourced detail.
**Date:** 2026-07-18
**Status:** Implemented, reviewed, merged locally. Full non-live suite green
(403 passed); `@live` smoke green; demonstrated end-to-end on `backup-docs`.

## What it does

`GET /search/drift?q=&level=&iterations=` runs the broad→deep DRIFT motion,
reusing the Phase-3 pieces:

1. **Primer** (1 strong LLM call): `shortlist_communities(q, level)` picks the most
   relevant community reports; the strong model drafts a preliminary answer + 3–6
   targeted follow-up queries, each tagged with the `community_id` that inspired it.
2. **Follow-up loop** (no LLM in retrieval; 1–2 iterations): each follow-up runs a
   local fact search. When tagged with a community, retrieval is **biased by that
   community's top-degree member entity as the Graphiti center node**
   (node-distance reranking) — "enter through the theme, drill into the facts."
   Iteration 2 (optional) drafts refined follow-ups from round-1 evidence.
3. **Synthesis** (1 strong LLM call): merges the primer draft + the deduped fact
   union, citing `[N]` markers only; `_finalize_answer` strips any URL; `Provenance`
   resolves fact UUIDs → source URLs.

LLM cost: 2 strong calls (iterations=1) or 3 (iterations=2), all on the synthesis
tier (GLM). Retrieval has no LLM. No LLM authors a URL (design decision #2).

## Live run on `backup-docs`

**Query:** *"What should I consider when planning long-term backup retention across
cloud vendors?"* (`iterations=1`, `level=1`)

**Primer-drafted follow-ups** (each tagged with its source community → center-node
bias):

| community | follow-up query |
|---|---|
| `34015922…` Backup Vault Security, Governance & Recovery | AWS Backup Vault Lock (Governance/Compliance) vs Azure immutable vaults — WORM behavior |
| `42bd4c69…` Azure Backup: Cross Region Restore, Immutability | storage replication types (GRS/LRS/ZRS) + CRR constraints for LTR |
| `34015922…` (same community) | Multi-party approval in logically air-gapped vaults |
| `b389bf55…` Cross-Subscription Database Restoration | moving `.bak` files across subscriptions, restore via SSMS/TSQL |

**Communities shortlisted** (`communities_used`): 5 — Cross-Subscription Database
Restoration, Backup Vault Security/Governance, Azure Cross Region Restore/
Immutability, Azure Backup Server Recovery, KeyVault Permissions.

**Synthesized answer** (excerpt) — organized by theme, spanning both vendors:

> When planning long-term backup retention across cloud vendors, consider
> cross-region and cross-subscription restore (restoring to a secondary region when
> CRR is enabled [10], SQL databases across subscriptions via CSR [27][29]);
> security features like immutability (vault locks removable during a
> compliance-mode grace time before the vault becomes immutable [1], immutable
> Backup vaults [2]); soft-delete [24]; and multi-party approval for air-gapped
> vaults [17][20]. For long-term retention specifically, Azure Backup stores LTR
> data [11] and supports an archive tier [12][13] … Azure frees only overwritten
> data blocks between incremental recovery points, carrying forward unoverwritten
> blocks to maintain the incremental chain [14] … Also consider access controls:
> restricting manual deletion of recovery points to specified IAM roles [3], AWS
> CloudTrail for API capture [8] …

**Stats:** 21 citations, **every one resolving to ≥1 source URL**, split across
`docs.aws.amazon.com` **and** `learn.microsoft.com` (genuinely cross-vendor); no
URL authored by the LLM (`"http" not in answer`).

**Sample citation resolution** (design decision #2, `fact_uuid → episode → article
→ source_url`):

| marker → fact_uuid | resolves to |
|---|---|
| `[1]` → `69e49bee…` | *Vault Lock* — docs.aws.amazon.com/aws-backup/…/vault-lock.html |
| `[10]` → `b6d44c37…` | *Restore SQL Server database from the vault* — learn.microsoft.com/azure/backup/restore-sql-database-azure-vm |
| `[27]` → `92665a6d…` | *Restore SQL Server database from the vault* — learn.microsoft.com/azure/backup/… |

## Design-decision alignment (verified)

- **#2 citations are traversal, never LLM:** only the synthesis answer is cited; its
  `[N]` markers are validated against the marker map; `_finalize_answer` strips
  URLs; `Provenance` expands fact UUIDs → URLs. The primer draft and follow-up
  *queries* never carry citations and are never surfaced as sourced answers.
  Demonstrated: `"http" not in answer`, all 21 citations resolve.
- **#4 one embedding space:** the primer embeds `q` with the SAME shared embedder as
  `:Community.embedding`; follow-up retrieval reuses the one-group Graphiti hybrid
  search.
- Reuses the `/answer` + `/search/global` citation machinery (`_finalize_answer`,
  `Provenance`) — DRIFT answers are cited the same verifiable way.

## Verification summary

- **Unit (`test_drift_parse.py`):** `_parse_followups` (relevance budget, order,
  invalid-community→None, blank-query drop, garbage), `_refine_followups` (parse +
  iteration-2 tag, bad-JSON→[]), `_dedup_facts` (first-seen order).
- **Integration (Neo4j testcontainer, `test_drift.py`):** primer shortlist+budget,
  empty-shortlist→degrade signal, primer-bad-JSON→single-`q` fallback,
  `_top_member_entity` (top-degree member), `_run_followup` center-node gating
  (tagged→member, untagged→None), full `drift_search` end-to-end (cited answer,
  URL stripped, invalid marker dropped, sources resolve to `url`),
  empty-shortlist→`answer_local` degrade, zero-facts→refusal with **no synthesis
  call**, iterations=2 runs refinement (both iteration tags present), a raising
  follow-up does not abort the request.
- **`search.py` regression (`test_search_center_node.py`):** `center_node_uuid`
  switches to the node-distance recipe + passes through; the RRF path and existing
  `search_local` callers are unchanged.
- **App (`test_answer_api_app.py`):** `GET /search/drift` returns
  `{query,answer,citations,follow_ups,communities_used}`; missing `q` → 422;
  `iterations=3` → 422; all pre-existing endpoints unchanged (20 app tests).
- **`@live` smoke (`test_drift_live.py`):** `/search/drift` on `backup-docs` →
  non-empty answer, no URL in answer, every citation resolves.
- Full non-live suite: **403 passed, 8 `@live` deselected**; `ruff check src tests`
  + `mypy` clean.

## Review findings addressed

- **Plan bug caught at Task 3 (fixed):** `_PRIMER_PROMPT`'s literal JSON braces
  collided with `str.format()` (`KeyError` on every call) — doubled the braces
  (`{{…}}`) so only `{q}`/`{blocks}` substitute. The `_REFINE_PROMPT` was already
  correctly escaped; `_SYNTH_PROMPT` has no literal braces.
- **Latent F401 (fixed):** `tests/unit/test_drift_parse.py` imported an unused
  `FollowUp` — dropped from the top import so `ruff check src tests` stays clean
  (the CI gate lints test files; the E702/E402 lesson from the global-search slice
  was applied throughout — no lint debt this time).

## Follow-ups (deferred, per spec §10)

- The `/answer` **router** + uniform `mode` contract (Phase 4 slice 2) — which also
  **normalizes the empty-shortlist degrade response shape** (v1's degrade returns
  `answer_local`'s shape + a `degraded` flag, a documented divergence).
- Iterations > 2 / adaptive depth by query breadth; cross-encoder reranking for
  follow-ups; caching primer shortlists across requests; per-follow-up (vs merged)
  evidence attribution.
- Minor (final-review triage): `_dedup_facts` is recomputed a few times over the
  (small) accumulated-facts list in `drift_search` — dedup-once cleanup.
