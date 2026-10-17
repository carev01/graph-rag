# Exhaustive answers to single-vendor list questions — design

**Status:** approved in conversation 2026-10-15 (sections 1–3); this document records it.
**Goal:** a single-vendor "list" question ("Which workloads does Veeam support immutability
for?", "Where can I restore a Windows agent backup?") is answered with **every** item the
documentation states — grouped, each cited, limitations in their own short section — and
every other single question picks better facts at no extra cost.

## 1. Why answers miss today (measured 2026-10-15, free traces)

- Single questions take raw hybrid-search order (no cross-encoder rerank; only comparison
  sides rerank) and keep **k = 15**. Veeam's scope is over the scan cap, so the 15 are
  whatever lexically matches: for Q1 mostly limitations ("Windows host ... does not support
  immutability") and off-topic LVM/RAID limits; for Q2 backup-I/O throttling facts.
- **Near-duplicates** fill the slots: the same fact in 3–4 wordings from different
  articles (four variants of "Instant Recovery ... as a Hyper-V VM").
- **First-stage recall**, not ranking, is the remaining gap: Hyper-V/VMware instant
  recovery, Amazon EC2 and disk export are retrievable with a query naming them, and the
  reranker places them #26–#51 of 52 against the original question — but the bare
  question never pulls them into the pool. A paraphrase expansion did not help; an
  expansion must name *kinds of item*.
- Measured coverage (tracked items present in retrieved facts): Q2 restore targets —
  today 3/13, reranked pool 80 + k = 40: 8/13; Q1 workload kinds — today 5/9, reranked
  pool + k = 40: 8/9, k = 60: 9/9. Wider pool (200) adds nothing; entity-name anchoring
  picked poor anchors.

## 2. Scope of the change

| question | path |
|---|---|
| single, list-shaped (§3) | **list mode** (§4–§5) |
| single, other | today's path + reranked pool and near-duplicate collapse (§6) |
| comparison | unchanged |
| enumeration ("which vendors ...") | unchanged |
| explicit `mode=` | unchanged — the override skips list mode |

## 3. List detection (deterministic, no LLM)

A question is list-shaped when its shape is **single** and it matches one of:
"which/what" + a plural noun ("which workloads", "what destinations", "which
hypervisors"); "where can I ..."; "what are the (options|ways|methods|...)"; a leading
"list ..."; "supported <noun>s" wording. Matches and non-matches are pinned by unit tests
(e.g. "What is a hardened repository?" and "How does SureBackup work?" are not lists).

## 4. List-mode retrieval

1. **Expansion.** One cheap-tier call (the map tier, as for side-question rewrites) returns
   6–10 retrieval queries that name likely *kinds of item* (for restores: hypervisors,
   clouds, physical machines, file and application level), keeping the question's
   vendor/product names. The model's domain knowledge only steers *which facts are
   searched for*; the answer states only what retrieved facts say. A failed or unusable
   expansion falls back to the question alone, with a warning logged — never silently
   empty.
2. **Per-query retrieval.** The original question and each expansion query run the
   existing scoped local search (`search_local`) with a reranked pool of 40, top 20 each,
   concurrently.
3. **Merge and collapse near-duplicates.** Facts with the same normalized text, or whose
   fact embeddings have cosine ≥ 0.92, collapse into one fact that keeps **every** source
   article (all of them become citations).
4. **Final rerank** of the merged facts against the original question (cross-encoder).
5. **Supported first.** Facts phrased as limitations ("not supported", "does not",
   "cannot", "only ...") are budgeted separately: up to **50 supported** facts plus up to
   **12 limitations** reach synthesis, so limitations cannot crowd out the list.

A failed per-query retrieval is dropped with the answer marked degraded; if every
retrieval fails, the error surfaces as an error (failure never reads as absence).

## 5. List-mode answer

- A list-answer prompt: answer with **every** item the facts state (no ceiling), grouped
  under a few headings the facts suggest; one short line per item with its citation;
  items repeated across facts merged into one line with their citations; then a
  **"Not supported / limitations"** section from the limitation facts, kept short and
  omitted when there are none. No padding; never claim something is undocumented beyond
  what the facts show; if the facts are thin, say what was found.
- The existing claim and attribution rules apply (qualifiers kept, never apply one
  product's facts to another, ≤ 3 markers per sentence); the LLM never writes a URL —
  citations resolve deterministically as everywhere else.
- Envelope: `list_mode: true`, `expansion_queries`, and `degraded` when a retrieval
  failed. List mode runs inside `/answer`, so the MCP `answer_question` tool gets it.
- If list mode grounds nothing, the router falls back to its normal route (local, then
  DRIFT) as today.

## 6. Other single questions

Local search uses a reranked pool of 80 and the same near-duplicate collapse; k stays 15;
prompts and answer length are unchanged.

## 7. Evaluation and testing

- **Golden list questions** (added to the Tier 1 set): (1) Which workloads does Veeam
  support immutability for? (2) Where can I restore a Windows agent backup? (3) Which
  hypervisors does Veeam Backup & Replication support? (4) Which resource types can AWS
  Backup protect? (5) What recovery options does Azure Backup offer for Azure VMs?
  Each gets an **expected-item list** built from the graph's own facts and **confirmed by
  the user** before it is scored.
- **Item coverage metric:** each expected item matched deterministically (aliases allowed)
  in the answer text; plus no listed item unsupported by the cited facts.
- **Free gate:** `eval_retrieval` checks every expected item appears among the retrieved
  facts for each list question — the development loop.
- **Paid Tier 1 eval (~$0.25), once the free gate passes:** item coverage 1.00 on the list
  questions; faithfulness and misattribution as before; no regression on the existing 16.
- **Unit tests (hermetic):** list detection; expansion fallback; near-duplicate collapse
  keeping all sources; supported-before-limitations budgets; the list prompt's rules;
  envelope fields; reranked pool + collapse on the single path; degraded / error paths.

## 8. Settings

`list_expansion_queries_max = 10`, `list_per_query_pool = 40`, `list_per_query_k = 20`,
`list_supported_k = 50`, `list_limitations_k = 12`, `list_dedup_cosine = 0.92`,
`local_single_rerank_pool = 80` (all in `ExtractSettings`).

## 9. Out of scope

Cross-vendor enumerations, global/DRIFT internals, theme reports.

## 10. Amendment (2026-10-15, approved in conversation): grounded recall and diverse selection

The free gate after the first build found 52/67 expected items. Per missing item: 10 were
never retrieved (the expansion's domain knowledge misses long-tail items — Sangfor,
Morpheus, Scale Computing, DocumentDB, Neptune — and with no vendor named it wrote
placeholders like "Vendor backup software"), 4 were cut by the 50-fact budget (filled by
repeats about popular items), 1 was merged away at cosine 0.92 ("VMware Cloud Director"
into a vSphere fact). A free prototype showed following the graph recovers 12 of the 15.

1. **Relation expansion (grounded, no LLM).** After round one (§4 steps 1–4), take the
   subject entity and relation name of the top `list_relation_seed_facts` (40) facts,
   group them, and for the `list_relation_groups` (6) most frequent (subject, relation)
   pairs pull every current sibling edge (same subject entity, same relation name), at
   most `list_relation_group_cap` (300) per group. Keep only siblings with an episode in
   the question's scope (when scoped), resolve their sources deterministically, and add
   them to the candidates. A failure records "relation expansion unavailable" in
   `degraded` and the answer proceeds from round one.
2. **Batched final rerank** of every candidate against the original question, at most
   `list_rerank_batch` (900) documents per call, merged by score.
3. **Diversity-aware selection.** Supported facts are chosen greedily by maximal marginal
   relevance — rerank score minus `list_mmr_lambda` (0.5) times the highest raw cosine to
   an already chosen fact — up to `list_supported_k`, now **80**. Limitations keep rerank
   order. Without embeddings, rerank order.
4. **Collapse threshold** `list_dedup_cosine` raised to **0.96**; a fact with no vendor
   merges only with another fact with no vendor (no vendorless bridge between vendors).
5. **Expansion prompt:** never write placeholder words; when the question names no
   vendor or product, keep the question's own terms.
6. **Degradation note** uses fixed wording ("N searches could not be completed";
   "results ranked by retrieval order") — no model-written text in the note.

## 11. Synthesis budget (2026-10-16, approved in conversation)

`list_supported_k` raised 80 -> **250** and the list synthesis `max_tokens` 6000 -> 10000.
The free gate went 59/67 -> 63/67: most missing items were retrieved but ranked past #80.
Still missing: Entra ID (Q1); Scale Computing HyperCore, HPE Morpheus VM Essentials and
"publish disks" (Q2) — the hypervisor ones rest only on generic "different workloads"
facts, which grounding deliberately does not stretch to.

## 12. Keyword sweep and per-item evidence grades (2026-10-16, approved in conversation)

The user's bar is a complete answer, with any claim whose support is weaker kept but
flagged. A free trace (2026-10-16) found that every item still missing at 250 facts has
a fact naming it, e.g. "VBR can restore physical machines from backups created by Veeam
Agents to the HPE Morpheus VM Essentials environment" and "Veeam Data Cloud for
Microsoft Entra ID offers service-level immutability". These facts hang off subjects the
relation expansion never visits, and the expansion queries never name the items. Entity
name embeddings cannot find peers (the nearest names to Proxmox VE / Nutanix AHV are
noise). A full-text OR query over the question's own words, ranked by BM25, returns
facts for all of them within the top 2000.

1. **Keyword sweep (recall, no LLM).** `question_terms(q, scope)` keeps the question's
   distinctive words: lower-cased, without stopwords, generic verbs ("support",
   "offer"), the list's item noun ("workloads", "options") or words of the scope's
   vendor/product/source names. A Lucene OR query over those words, each also as a
   suffix-stripped prefix ("restore restor*"), runs on graphiti's `edge_name_and_fact`
   full-text index: current edges of the question's group, BM25 order, at most
   `list_sweep_cap` (2000) facts kept. When scoped, `list_sweep_scoped_overfetch` (4) x
   the cap are fetched and filtered by scope before the cap. No words → no sweep. The
   sweep runs concurrently with the per-query searches and joins round one; a failure
   records "keyword search unavailable" in `degraded` and the answer proceeds.
2. **Per-item grades (deterministic, list mode only).** The list prompt writes each
   item as `- **<item>** — <short description> [N]`, and includes items whose fact
   states them more generally than the question asks. After synthesis, each item line
   before the "Not supported / limitations" section is graded from its cited facts
   (fact text plus the source article titles):
   - **confirmed**: a cited fact names the item (at least half of the item's
     distinctive words), its rerank score is at least `list_item_floor` (or unscored
     after a degraded rerank), and the fact contains every question term (stem prefix
     match, small synonym groups such as restore ~ recover);
   - **needs validation**: as confirmed, but some question terms are missing; the line
     is kept with the fixed note *(needs validation: the cited source does not mention
     <terms>)*, the terms spelled as in the question;
   - **dropped**: no cited fact names the item at or above the floor; the line is
     removed (headings left empty are removed too) and logged.
   No grade text is model-written. The envelope gains `item_grades`
   (`[{item, grade, missing_terms, markers, fact_uuids}]`), also in the MCP structured
   output. The answer-level confidence proposed earlier is not built.
3. **Calibration.** The free gate reports, per expected item, the best rerank score of
   a retrieved fact naming it; `list_item_floor` is set below every expected item's
   score. The paid eval reports flagged and dropped item counts per question.
4. **Calibration result (2026-10-16 free gate) and adjustments.** Retrieval found 65/67
   expected items. Every expected item's best rerank score was >= 0.689. Facts that are
   clearly off-topic but name an item scored 0.33–0.48. Three changes followed:
   - **Inflected exact forms in the sweep.** Lucene wildcards are constant-score, so
     "agent*" barely ranked "Veeam Agents". Each term also adds its plural/singular and,
     for verbs, -ed/-ing forms as exact words. The Morpheus agent fact moved from #2774
     to #325.
   - **Unnamed but relevant is flagged, not dropped** (the user's original intent: keep
     a weakly grounded claim, flagged). A cited fact that does not name the item but
     scores >= `list_item_unnamed_floor` (0.65) makes the item *needs validation* with
     the fixed note *(needs validation: the cited source does not name this item)*.
     Named support needs >= `list_item_floor` (0.4). Anything else is dropped.
   - The list prompt names each item as the facts name it.

## 13. After paid eval run 12 (2026-10-16, approved in conversation)

Run 12 results:
- **Scored lines:** list item coverage in the answer text was 0.94, with 67 items flagged and 5 dropped. Routing, facet coverage and misattribution were unchanged.
- **Synthesis omissions:** the model left out retrieved items (Sangfor, publish disks, OLVM, RHV).
- **Kind words flagged:** words naming the item kind ("resource" in "resource types", "recovery" in "recovery options") were treated as qualifiers, which flagged all 11 AWS items.
- **Noisy flags:** 41 of the Azure answer's items carried the same "does not mention VMs" note.
- **Unscored answers:** the judge returned nothing on the three answers citing 110–254 facts.
- **Not changed:** the SureBackup grounding miss was traced. The answer cites on-topic SureBackup pages other than the golden set's five, and an expected article is still in the top 15. The verify-enumeration miss is Cohesity documented through anomaly-detection facts, an enumeration-precision question outside this slice. Neither changes here.

1. **Item-kind words are not qualifiers.** The words between "which/what" and the item noun
   ("resource" in "Which resource types", "recovery" in "What recovery options") are excluded from
   the grading terms. The sweep keeps them, since they help recall.
2. **Grouped flag note.** When one missing term is shared by at least `list_flag_group_min` (5)
   items and by at least half of the kept items, one fixed line goes above the list:
   *"Items marked † come from documentation that does not mention <Term> specifically;
   validate them for your case."* Those items' lines get " †" instead of the full note. Other
   notes are unchanged.
3. **Completeness pass.** After synthesis and before grading, supported facts the answer did not cite go to the map
   tier with the answer's item names. The tier returns `{item, description, markers}` only for
   items those facts state and the answer lacks, naming each item as the facts do, with at most
   3 markers each. These are appended as an "Also documented" section before the limitations
   section and graded like every other line. A failed pass records "completeness check
   unavailable" and the note reads "some documented items may be missing". Controlled by
   `list_completeness_pass` (default true).
4. **Judge on long answers (eval).** An answer citing more than 60 facts or longer than 6000
   characters is judged in chunks of consecutive paragraphs/sections, each with only the facts
   it cites. Faithfulness is the minimum chunk score, so "5" still means every claim is supported.

## 14. After paid eval run 13 (2026-10-17, option 1 approved in conversation)

Run 13: item coverage 0.98 and no unscored answers. An itemised audit of the lowest answer
(Azure, faithfulness 2) found three answer defects and three judge artifacts:

- Answer defects:
  - facts about *backing up* to a vault or MABS were written as "Restore from …" items;
  - a completeness-pass description claimed more than its fact;
  - the answer drifted to other workloads (Azure PostgreSQL, AKS, Hyper-V) under a
    question about Azure VMs.
- Judge artifacts:
  - the faithfulness judge sees unlabelled fact text, so "Veeam Data Cloud states …"
    lines citing Veeam-documentation facts read as unsupported;
  - the judge counts lines carrying a "needs validation" note or † as unsupported,
    although the note discloses the gap;
  - the AWS "misattributions" are facts merged from AWS and Veeam documents, each with
    an AWS Backup source (invariant #4), not a scope leak.

1. **List prompt.** Two new rules:
   - Describe each item only as its cited facts state it. Never change what a fact
     says it does: a fact about backing up is not a restore option, and supported-for
     is not recommended.
   - List only items that answer the question for the subject it names. Skip items
     about a different workload or product. Keep items a fact states for a broader
     category that contains the subject, for example all agents when the question
     asks about a Windows agent.
2. **Completeness pass.** The description restates only what the cited facts say, in at
   most 15 words, and the pass applies the same subject rule. `max_tokens` rises from
   4000 to 8000. A reply cut short (`finish_reason == "length"`) is treated as a failed
   pass, as today, but logged with its length.
3. **Faithfulness judge (eval).** Fact lines go to the judge with their (Vendor ·
   Product) labels, as the attribution judge already receives them. The judge prompt
   says that a line carrying a "needs validation" note or † discloses that its fact
   does not state the noted qualifier. Such a line is judged on whether its cited
   facts support the item apart from that disclosed qualifier.
