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
