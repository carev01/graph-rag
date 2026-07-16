# Batch API for Bootstrap Cost Reduction — Evaluation

**Date:** 2026-07-16
**Question:** Can we use an LLM **Batch API** (async, ~50% discount) to cut the cost of the first full-corpus bootstrap extraction?
**Verdict: NO-GO for now.** The saving is real but partial and would require bypassing Graphiti's extraction orchestration *and* leaving the Azure enterprise posture — poor trade for a one-time job when cheaper, zero-risk levers already exist. Revisit only if the Phase-1 cost benchmark proves prohibitive.

---

## 1. What a Batch API offers

OpenAI's Batch API (and equivalents): submit a JSONL of independent chat-completion requests, the provider processes them asynchronously and returns results within **24h**, at roughly **50% off** input+output token pricing. It is **fire-and-forget** — you get no result inline; you poll for the completed output file.

**Scale at stake:** a full bootstrap is ~260M content tokens → **~0.8–1.5B LLM tokens** through Graphiti's multi-call pipeline (CLAUDE.md §Cost awareness). A 50% discount on that is a meaningful absolute number.

## 2. Why Graphiti's extraction resists batching

`add_episode` is a **synchronous, stateful dependency chain** per episode:

```
extract_nodes(episode text)                      LLM call — depends only on episode text
  → resolve_extracted_nodes(vs existing graph)   LLM call — needs extraction result + reads live graph
    → extract_edges(resolved nodes)              LLM call — needs resolved nodes
      → resolve_extracted_edges(vs graph)        LLM call — invalidates contradicted facts, reads/writes graph
        → extract_attributes(nodes)              LLM call
```

Each step's **prompt does not exist until the previous step's LLM result returns** (later prompts embed earlier outputs), and the `resolve_*` steps read and write live Neo4j state, so they are inherently **order-dependent**. This is the opposite of the Batch API's model, which needs all requests known up-front and independent. Confirmed in code: Graphiti's `LLMClient.generate_response` is a plain inline `await` — there is **no batch/async-submit hook** in the library to plug into.

## 3. The one batchable seam — and its ceiling

Only the **first extraction call(s)** of each episode (`extract_nodes`, arguably `extract_edges`) depend solely on the episode *text* — those are independent across all episodes and could, in principle, be batched. But:

- The **resolution/dedup calls are not batchable** (graph-state-dependent, order-sensitive). A measured probe showed **~5.4 LLM calls/episode**; extraction is only ~2 of them. So even a perfect batch of the extraction stage leaves ~half the calls online.
- **Realizable saving ≈ 20–30% of bootstrap LLM spend**, not 50%: only the batchable extraction fraction, at 50% off. (Extraction prompts are the *large* ones — full ontology + episode — so their token share is above their call share, which is the one point in batch's favour.)
- **Capturing it requires bypassing Graphiti:** you would re-implement `extract_nodes` prompt-building + response-parsing outside `generate_response`, submit those as a batch, then hand-feed the results back into Graphiti's resolution steps. That is a real build **and** an ongoing maintenance liability against a fast-moving vendored library (`graphiti-core`), for a **one-time** bootstrap.

## 4. The Azure constraint (decisive)

**Azure OpenAI does not expose a Batch API for `gpt-5-mini`.** Using batch would mean going to **OpenAI-direct** (`api.openai.com`). The pipeline *can* structurally target that endpoint (the non-Azure `AsyncOpenAI` branch exists in `graphiti_client.py`), but doing so:

- **Abandons the Azure enterprise / internal data posture** that was a deliberate architecture choice — corpus content would leave the Azure tenancy.
- Changes model availability, pricing, and data-handling terms.

So batch isn't a config flip on the current deployment; it's a vendor change on top of the rearchitecture.

## 5. Cheaper, lower-risk levers (already in place or free)

The dominant cost levers are already pulled or available at near-zero effort:

1. **Cheap model tier** — `gpt-5-mini` is already the extraction tier; this is the biggest per-token lever and it's done.
2. **Prompt caching** — already active at ~13% with **zero work** (our static instructions/ontology are in the cached prefix; see `prompt-caching-findings.md`); more is available later via a Graphiti prompt-layout patch.
3. **Token-budget metering + phased vendor-by-vendor bootstrap** (Sub-slice B) — already controls *spend rate* and spreads the bootstrap, so the 24h batch latency advantage is largely moot (the phased rollout already tolerates a slow bootstrap).
4. **Deterministic structural layer + noise filtering** — keeps LLM work to article *content* only, and prunes junk before it costs downstream work.

Stacked, these already address the bootstrap cost concern with no rearchitecture and no data-posture change.

## 6. Recommendation

**NO-GO.** The batch discount is real but:
- Azure can't batch `gpt-5-mini` → forces an OpenAI-direct vendor change that **breaks the Azure data posture**;
- only ~20–30% of the spend is actually batchable (resolution can't batch);
- capturing it needs **bypassing Graphiti's extraction orchestration** — significant build + maintenance for a one-time job;
- cheaper, zero-risk levers (cheap tier + prompt caching + budget metering + phased rollout) already exist.

**Revisit IF, at the Phase-1 cost/quality benchmark (~200 articles), bootstrap cost proves prohibitive AND** (a) a batch-capable equivalent model is available on an acceptable-posture endpoint, **AND** (b) Graphiti gains a batch hook or the "extract-in-batch, resolve-online" rearchitecture is independently justified. Until then, run the bootstrap on the cheap tier with caching + budget metering.

## 7. Deferred / follow-on (if ever revisited)

- Prototype the **extract-in-batch / resolve-online** split (batch `extract_nodes` across many episodes, feed results into online resolution) — the only architecturally-sound way to use batch here.
- Re-price the exact batch vs on-demand delta once the Phase-1 benchmark gives a measured tokens-per-article and a settled model/endpoint.
- Reassess if `graphiti-core` adds native batch support upstream.
