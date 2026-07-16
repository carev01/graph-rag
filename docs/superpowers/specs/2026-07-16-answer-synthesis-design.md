# Answer Synthesis (`/answer`) — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Phase 2 retrieval — synthesis layer: an `/answer` endpoint that writes a cited prose answer over retrieved facts, with citations resolved deterministically (the LLM never writes a URL).
**Date:** 2026-07-16
**Status:** Approved design — ready for implementation planning

Builds directly on `/search/local` (retrieval-first, ranked cited facts). Adds an LLM synthesis layer on top, enforcing **design-decision #2**: the synthesis LLM emits only opaque fact markers; a deterministic resolver expands the markers it used into source URLs. Synthesis model = **GLM-5.2 via the existing Ollama judge config** (swappable later). This is the trust-critical piece of GraphRAG.

---

## 1. Scope

**In:** `answer_api.answer_local` (synthesis core); a `_synthesis_client_and_model` seam reusing the GLM-5.2 config; a `GET /answer` FastAPI endpoint; a marker-based no-URL citation mechanism; refuse-when-unsupported grounding; a groundedness extension to the golden harness. **Out:** streaming responses; `/timeline` (next slice); verbose-query rewrite (the slice after); multi-turn/session; re-ranking beyond RRF.

## 2. Grounding (verified)

- `search_local(graphiti, driver, *, q, k, vendor, include_invalid, group_id)` returns `{query, count, results: [{fact, fact_uuid, valid_at, sources}]}` — reused as-is for retrieval.
- `resolve_citations(fact_uuids)` (batch) exists — used to expand cited markers → sources.
- GLM-5.2 is one `AsyncOpenAI` built from `settings.judge_base_url/judge_model/judge_api_key` (`.env`: `https://ollama.com/v1`, `glm-5.2:cloud`) — a reasoning model (needs an adequate `max_tokens`). `instrument()` captures its token usage.

## 3. Architecture & flow

```
GET /answer?q=<query>&k=15&vendor=<optional>
  1. res = await search_local(..., k=k)                       ranked cited facts
  2. label facts [1..N] by rank; build marker -> fact_uuid map (and marker -> sources)
  3. prompt GLM-5.2 with ONLY "[N] <fact text>" lines (NO urls, NO titles) + the question:
       - answer using ONLY these facts; cite inline by [N];
       - if the facts don't cover it, say "I don't have enough information ..."
       temperature=0, max_tokens adequate for a reasoning model.
  4. GLM returns prose containing [N] markers.
  5. DETERMINISTIC post-processing:
       - strip any URL the model emitted (it must never write one -> hallucination guard);
       - parse the [N] markers actually used; keep only markers that map to a retrieved fact;
       - build citations from those markers via the marker->sources map (already resolved in step 1).
  6. return { query, answer, citations, retrieved, cited }.
```

No URL is ever authored by the LLM; citations are produced only by the deterministic marker→fact→source map.

## 4. Components

### 4.1 `answer_api/synthesize.py`
`async def answer_local(graphiti, driver, synth_client, synth_model, *, q, k=15, vendor=None, group_id) -> dict`:
1. `res = await search_local(graphiti, driver, q=q, k=k, vendor=vendor, group_id=group_id)`.
2. Build the numbered fact block + `marker(int) -> {fact_uuid, fact, sources}` map from `res["results"]` (rank order → `[1], [2], …`).
3. Call `synth_client.chat.completions.create(model=synth_model, temperature=0, max_tokens=..., messages=[{role:user, content: PROMPT}])`.
4. `answer_raw = resp.choices[0].message.content or ""`.
5. Post-process (§4.3) → `answer, cited_markers`.
6. Return `{"query": q, "answer": answer, "citations": [ {"marker": m, "fact": ..., "fact_uuid": ..., "sources": [...]} for m in cited_markers ], "retrieved": len(res["results"]), "cited": len(cited_markers)}`.

If retrieval returns zero facts, skip the LLM call and return a fixed *"I don't have enough information"* answer with empty citations (no tokens spent, no hallucination surface).

### 4.2 Synthesis client seam
`answer_api.synthesize._synthesis_client_and_model(settings) -> (AsyncOpenAI, str)` — builds `instrument(AsyncOpenAI(base_url=settings.judge_base_url, api_key=settings.judge_api_key or "not-needed"))` and returns `(client, settings.judge_model)`. Named for *synthesis* (not judging) so a later change can point it at a stronger model via new `synthesis_*` config without touching the judge. A clear comment records that synthesis currently shares the GLM-5.2 judge endpoint. Raise a clear error if `judge_base_url` is unset (synthesis has no configured model).

### 4.3 Post-processing (design-decision #2 enforcement — deterministic, pure)
`_finalize_answer(raw: str, marker_map: dict[int, dict]) -> tuple[str, list[int]]`:
- **URL strip:** remove any `http://`/`https://` token from `raw` (the LLM must never write a URL; any is a hallucination). Replace with nothing / a redaction marker.
- **Marker parse:** find `[N]` integer markers in the (URL-stripped) text; keep the ordered unique set that exists in `marker_map` (drop invented markers).
- Return the cleaned answer text + the list of valid cited markers.
Pure and unit-testable without any LLM.

### 4.4 Prompt
A tight instruction: "You are answering a question about backup products using ONLY the numbered facts below. Cite every claim inline with its `[N]` marker. Do NOT use outside knowledge. Do NOT write any URL or link. If the facts do not answer the question, reply exactly: `I don't have enough information to answer that from the available sources.`" followed by the numbered facts and the question. Kept in `synthesize.py`.

### 4.5 `GET /answer` endpoint (`answer_api/app.py`)
Query params `q` (required), `k: int = 15`, `vendor: str | None = None`. On startup the lifespan additionally builds the synthesis client+model (from `app.state.settings`) and stores on `app.state.synth_client`/`app.state.synth_model`; closes the client on shutdown alongside graphiti/driver. The handler calls the module-level `synthesize.answer_local(...)` (injectable seam for tests, like `search_local`). `/search/local` and `/health` unchanged.

## 5. Testing

- **Unit — post-processing (`_finalize_answer`), no LLM:** a URL in the raw answer is stripped; valid `[N]` markers kept, invented markers dropped; ordered-unique; empty/no-marker answer → empty cited list.
- **Unit — `answer_local` with a stubbed synth client** (returns canned prose with markers) + a stubbed `search_local` (canned facts): asserts the response shape, that citations come only from cited+valid markers with their resolved sources, and the zero-facts short-circuit (fixed refusal, no client call).
- **FastAPI `/answer` TestClient** (stub `answer_local`): 200 with `{query,answer,citations,retrieved,cited}`; missing `q` → 422; `/search/local`+`/health` still green.
- **`@live` GLM smoke:** `answer_local` against the real graph + GLM-5.2 for a known question returns a non-empty answer citing ≥1 fact with a resolved source; and a nonsense/off-corpus question returns the refusal.
- **Groundedness (golden harness extension, controller-run):** for each golden question, run `/answer` and check the answer cites ≥1 fact whose source is an expected article-id (answer-groundedness@k) + a spot-read that the prose is faithful and URL-free. Report the number.
- Full non-live suite + ruff/mypy clean.

## 6. Acceptance criteria

1. `GET /answer?q=...` returns a GLM-5.2-synthesized prose answer with `[N]` markers and a `citations` list whose URLs come *only* from the deterministic marker→fact→source resolver.
2. The LLM never authors a URL: any URL in the raw output is stripped; invented markers are dropped; citations are resolver-produced only.
3. Unsupported/off-corpus questions return the fixed refusal (no hallucinated answer); zero-retrieval short-circuits without an LLM call.
4. Synthesis uses GLM-5.2 via the judge config through a synthesis-named, swappable seam; token usage captured.
5. The groundedness harness reports answer-groundedness over the golden set; a spot-read confirms faithful, URL-free answers.
6. Unit + FastAPI tests green; `@live` GLM smoke passes; ruff/mypy clean.

## 7. Deferred

- Streaming responses; multi-turn context.
- A dedicated stronger synthesis model (via `synthesis_*` config) — the seam is built; the swap is later.
- `/timeline` (bi-temporal) and verbose-query rewrite — the next two retrieval slices.
- Answer-quality evaluation beyond groundedness (faithfulness scoring by an independent model) — a later eval slice.
- The `/answer` router that picks local/global/drift — Phase 4.
