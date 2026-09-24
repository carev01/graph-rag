# Pre-bootstrap decisions D4 and D6 — evidence and recommendations

**Date:** 2026-09-23. **Cost:** $0 — read-only Cypher on the live graph, read-only
DocExtractor fetches, the local neural chunker, and the existing capture dataset.
**Decides:** D4 (same-pair invalidation, BACKLOG 33) and D6 (chunk size) of
`production-readiness-review-2026-09-13.md`, both of which change what the bootstrap
writes.

---

## D4 — same-pair invalidation

### What the live graph holds

19 facts carry `invalid_at` (pilot re-ingest 2026-09-14). Two different mechanisms
produced them:

| mechanism | n | what it is |
|---|---:|---|
| end date **stated in the text** | 5 | "will retire classic alerts on March 31, 2026", "ADE … retirement in September 2028" — extraction dating a fact the document itself bounds |
| **same-pair contradiction** | 14 | a newer fact between the same two entities judged to contradict an older one; the older one expired |

The invalidator of each contradiction was identified as the same-pair fact(s) created at
the victim's `expired_at` (graphiti writes both in one resolution step).

### The 14 contradictions: none is a document change

The graph has never ingested an update (0 superseded, 0 removed), so every contradiction
fired between two *different* articles of the *same* crawl. Hand-classified:

| # | victim (invalidated) | invalidator | verdict |
|---|---|---|---|
| 1 | Azure Backup supports auditing/enforcing Azure Files backup with Azure Policy | supports reporting / file recovery for Azure Files | unrelated |
| 2 | Azure Site Recovery supports protection for Azure VMs | Azure Backup reporting / metrics for VMs | unrelated |
| 3 | **Azure Backup supports backing up Azure Virtual Machines** (cited by 11 articles) | "Undelete action isn't available for … Azure Virtual machine" | limitation read as contradiction |
| 4 | **Azure Backup supports Azure Disks backup** | "Backup Now … aren't available for … Azure Disks" | limitation read as contradiction |
| 5 | **Azure Backup supports Azure Files backup** | undelete not available / reporting | limitation read as contradiction |
| 6 | supports Storage account for governance views | requires Owner privileges on the Storage account | unrelated |
| 7 | requires Owner privileges on the Storage account | (not identifiable) | — |
| 8 | Resiliency provides a retention details view | Resiliency provides a security posture view | unrelated |
| 9 | Undelete isn't available for Files | supports reporting / file recovery for Files | unrelated |
| 10 | Backup Now … not available for Blobs | (not identifiable) | — |
| 11 | provides snapshot backup for all resources in the selection | provides snapshot backups for some resources | refinement |
| 12 | integrates with AWS Organizations for cross-account backups | opt-in Regions: delegated admin lacks cross-account monitoring | refinement |
| 13 | **AWS Backup supports FSx for ONTAP** | does not support cross-Region copy of FSx for ONTAP | limitation read as contradiction |
| 14 | no continuous backup / PITR for RDS **Multi-AZ clusters** | continuous backup / PITR for RDS **instances** | refinement (both true) |

**0 of 14 genuine** (95% Wilson upper bound 21.5%). The damage is concentrated on the
most-cited core facts: "supports backing up Azure VMs / Disks / Files / FSx for ONTAP"
are the facts a user asks about first, and `/search/local` hides every invalidated fact
(`search.py:38`). This matches item 6's hand inspection (6 of 6 bad) and the pilot's
**11.8%** of dedup calls asserting a same-pair contradiction. At bootstrap scale that is
thousands of core facts silently removed from answers.

### Recommendation

**Suppress same-pair contradiction invalidation at ingest.** Drop `contradicted_facts`
before graphiti acts on them (an extension of `contradiction_gate`, which already skips
the cross-pair search), keeping the `contradicted_same_pair` counter so the rate stays
visible. Invariant #3 is unaffected: genuine change is carried by *updates* — superseded
episodes and the weekly sweep — which is the mechanism the invariant describes; graphiti's
same-pair judgement has produced no genuine change in 20 observed cases (item 6 + this).

Not recommended: BACKLOG 30's Tier-1 rule (invalidate only when both articles are
`content_changed_basis = exact` and on a `source_changed_at` change). It would have
blocked none of these 14 for the right reason — all are same-crawl — and there is no
observed genuine case for it to preserve.

This is a second semantic patch into graphiti; BACKLOG 33 requires its own design cycle,
which this document is the input to. Separately, the 14 existing bad invalidations on the
live graph could be reverted (clear `invalid_at`/`expired_at`) — a write to the baseline,
so only with explicit approval.

### A separate defect found on the way: future end dates hide current facts

Three facts carry `invalid_at = 2028-09-01` (ADE retirement). They are **still true
today**, yet `search_local` drops any fact with a non-null `invalid_at`
(`src/answer_api/search.py:38`) and `/timeline` labels them "superseded"
(`timeline.py:29-34`). Both should compare `invalid_at` with *now* (a fact is current
while `invalid_at` is null **or in the future**). Small, independent fix.

---

## D6 — chunk size

### Why episodes are small

Live episodes average ~300 tokens against the plan's 1,500–2,000. The cause is not the
settings' ceiling: the neural chunker (`chonkie`, `mirth/chonky_modernbert_base_1`)
emits semantic-boundary chunks, `min_chunk_tokens=128` merges only fragments below 128,
and `max_chunk_tokens` (1,800 strong / **900 cheap** — the tier most articles use) is only a
cap. **Nothing packs chunks up toward a target.**

### How much of the cost is per episode — measured from the capture

Grouping the 13,176 captured solar-pro4 calls by episode (602 episodes with full
extraction) and regressing each episode's total volume on its content length:

| | per episode (fixed) | per content character | at the median episode (1,167 chars) |
|---|---:|---:|---|
| prompt characters | **54,751** | 19.8 | 70% fixed |
| completion characters | 1,263 | 2.23 | 33% fixed |

The fixed part is the instructions, ontology, previous-episode context and entity lists
every episode's five prompt types repeat. Per type: `extract_edges` 25.0k fixed,
`dedupe_nodes` 13.3k, `extract_text` 11.8k; `resolve_edge` calls scale with content
(0.7 + 1.88 per 1,000 chars), so packing does not reduce them.

### Projection on the corpus — 197 articles, 25 sources, 13 vendors

Sampled proportionally to source size from DocExtractor, chunked by the production
chunker, episodes built by the production `build_episodes` with the production tier rule
(22 dense → strong tier, rest cheap), then greedily packed to a target:

| variant | episodes | per article | median tokens | prompt vol | completion vol | **bill** |
|---|---:|---:|---:|---:|---:|---:|
| today | 531 | 2.70 | 265 | 100% | 100% | 100% |
| pack 600 | 408 | 2.07 | 412 | 85% | 93% | 86% |
| pack 900 | 326 | 1.65 | 536 | 74% | 89% | 76% |
| **pack 1200** | 277 | 1.41 | 585 | 68% | 86% | **71%** |
| pack 1800 | 237 | 1.20 | 570 | 63% | 84% | 66% |

Bill = 83.8% input + 16.2% output at solar-pro4's $0.09/$0.36 (mixed-tier proposal §2).
Against the $2,176 all-API bootstrap estimate, pack 1200 saves **~$630**, pack 1800
~$740. Episodes fall **48%** at 1200: the three per-episode phases that run strictly
serially (`extract_text`, `extract_edges`, `dedupe_nodes` — 42.8% of LLM time, pilot
re-ingest §4) run half as often, so wall clock should fall by a similar order —
unmeasured. The corpus mean is 2.7 episodes/article (the pilot's AWS/Azure articles were
7.9), so the curve flattens early: most articles become one episode by 1,200.

### What this does not establish — quality

Bigger chunks may lower extraction recall (more content per prompt, more entities per
dedup list) — and the cheap tier's 900 cap was set because the verbose cheap model struggled
with larger inputs (`config.py:262`). The slope
fitted above comes from episodes up to ~2,700 chars; extrapolating to 1,200-token episodes
is inference. **Cost is measured; quality is not.**

Interaction with prompt caching: the fixed 54.7k characters per episode is largely the
static part a provider cache would serve at 20% of the input price. If caching is adopted,
packing's *cost* benefit shrinks; its *episode-count* (wall-clock) benefit does not.

### Recommendation

1. **Add a packing step** to `episode_builder` (greedy merge of consecutive chunks up to
   `pack_target_tokens`, never across the tier cap), default off until measured.
2. **One small paid A/B to decide the target** (needs authorisation): ~30 articles biased
   toward multi-episode ones, into a scratch `group_id`, today vs pack 1200 (cheap cap
   raised to 1200). Measure facts and entities per article, structured-output defects,
   dedup counters, wall clock, $/article, and a hand-judged sample of facts present in one
   arm and missing from the other. Estimated cost: well under $5.
3. If recall holds within noise, adopt 1200 for the bootstrap; if it drops, try 900 (a
   pure packing change, no cap change) before giving up the lever.
