# `/answer` Router — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Phase 4, slice 2 — the `/answer` router: one endpoint that classifies a
query to a retrieval mode (local / global / drift / timeline), dispatches to the
existing mode function, and returns a **uniform response contract** with a `mode`
field. Closes the DRIFT empty-shortlist degrade-shape divergence.
**Date:** 2026-07-18
**Status:** Approved design — ready for implementation planning

`/answer` currently *is* the local-synthesis endpoint. This slice turns it into the
router; its former behavior becomes `mode=local`. Nothing external consumes
`/answer` yet (Copilot/MCP is Phase 5), so changing its response shape is safe.

---

## 1. Scope

**In:** `answer_api/router.py` (`classify`, `_render_timeline`, `answer_router`,
the uniform-envelope normalizer); rewiring the `/answer` endpoint to the router +
a cheap classifier client on `app.state`; one config field
(`router_default_mode`). Reuses the existing mode functions unchanged
(`synthesize.answer_local`, `global_search.global_search`, `drift.drift_search`,
`timeline.timeline_local`) and the cheap tier (`cheap_llm_*`, ling-2.6-flash) built
for the extraction router.

**Out (later slices / deferred):** the `freshness` block
(`reports_as_of`/`graph_cursor_time`); richer citation fields
(vendor/product/section/valid_at); a reliable single-product→local heuristic (left
to the LLM for now); fallback arcs beyond local-empty→drift; MCP exposure
(Phase 5).

## 2. Grounding (verified)

- Current mode return shapes: `answer_local` → `{query, answer, citations:
  [{marker,fact,fact_uuid,sources}], retrieved, cited}`; `global_search` →
  `{query, answer, citations:[{marker,fact_uuid,sources}], communities_used}`;
  `drift_search` → `{query, answer, citations:[{marker,fact_uuid,sources}],
  follow_ups, communities_used}` (empty-shortlist degrade → `answer_local`'s shape
  + `degraded:"no-primer-communities"`); `timeline_local` → `{query, count,
  timeline:[{fact,fact_uuid,valid_at,invalid_at,status,sources}]}` (**no `answer`
  field** — the outlier).
- Sources everywhere are keyed `{url, title, article_id}` (from
  `Provenance.resolve_citations`).
- Cheap tier exists: `cheap_llm_base_url` (default `https://openrouter.ai/api/v1`),
  `cheap_llm_model` (default `inclusionai/ling-2.6-flash`), `cheap_llm_api_key`
  (default `""` → tier unavailable). No routing/classify code exists yet.
- `answer_api` app lifespan sets `app.state.{settings,graphiti,driver,synth_client,
  synth_model,embedder,map_client,map_model}`; endpoints call module-attr seams
  (`search_mod`/`synth_mod`/`timeline_mod`/`global_mod`/`drift_mod`).
- Mode-function signatures the router dispatches to:
  - `answer_local(graphiti, driver, synth_client, synth_model, *, q, k=15, vendor, group_id)`
  - `global_search(driver, embedder, map_client, map_model, synth_client, synth_model, *, q, level, k, group_id, relevance_min)`
  - `drift_search(graphiti, driver, embedder, synth_client, synth_model, *, q, level, iterations, primer_k, max_followups, followup_k, group_id)`
  - `timeline_local(graphiti, driver, *, q, limit=30, vendor, group_id)`
  - Per-mode knobs come from `settings` (`global_default_level`,
    `global_shortlist_k`, `global_map_relevance_min`, `drift_primer_level`,
    `drift_iterations`, `drift_primer_k`, `drift_max_followups`,
    `drift_followup_k`).

## 3. Architecture & flow

```
GET /answer?q=&mode=&vendor=
  1. CLASSIFY -> (mode, via)
     a. mode_override (?mode=) present -> (override_mode, "override")
     b. heuristic guardrails (deterministic, first match wins):
        - temporal   -> ("timeline", "heuristic")
        - cross-vendor comparison -> ("global", "heuristic")
     c. cheap_client set -> cheap LLM classifies -> (label, "llm"); unknown label -> default
     d. no cheap_client OR unparseable -> (settings.router_default_mode, "default")   # "drift"
  2. DISPATCH to the chosen mode function (knobs from settings; `vendor` forwarded to local/timeline)
  3. NORMALIZE the mode's dict into the uniform envelope (below); timeline gets a
     deterministic markdown `answer` + flattened numbered top-level citations
  4. FALLBACK: if chosen=="local" AND result["retrieved"]==0 -> re-dispatch as drift;
     envelope mode="drift", routing.fallback_from="local" (one escalation, no loops)
  return uniform envelope
```

**Classifier tier:** cheap (ling), same key/endpoint as the extraction router. One
cheap call only for queries the heuristics miss; heuristics-only when no cheap key.

## 4. Components (`answer_api/router.py`)

### 4.1 `classify`
`async def classify(q, *, cheap_client, cheap_model, mode_override=None,
default_mode="drift") -> tuple[Mode, str]`
- `Mode = Literal["local","global","drift","timeline"]`.
- If `mode_override` is a valid `Mode` → `(mode_override, "override")`.
- Heuristic guardrails (case-insensitive regex, first match wins):
  - **temporal → timeline:** `\b(chang(e|ed|es|ing)|history|used to|since \d|over time|evolv|deprecat|no longer|when did|previously)\b`
  - **cross-vendor → global:** `\b(compare|comparison|across (all )?vendors|all vendors|which vendors|every vendor)\b`
- Else if `cheap_client`: one cheap LLM call (few-shot prompt, returns a single
  bare label). Parse tolerantly (lowercase, strip, match against the four labels);
  a recognized label → `(label, "llm")`; anything else → `(default_mode, "default")`.
- Else → `(default_mode, "default")`.

### 4.2 `_render_timeline`
`def _render_timeline(timeline_result) -> tuple[str, list[dict]]`
- Deterministic (no LLM). Number the `timeline` events `[1..N]`; build a markdown
  narrative, one line per event, e.g.
  `- **{fact}** — valid_at {valid_at}, {status}{ " (invalid_at " + invalid_at + ")" } [n]`.
  Empty timeline → `("No recorded changes for that query.", [])`.
- Return `(answer_markdown, [{marker:n, fact_uuid, sources} for each event])`.
- Never writes a URL — `sources` come straight from the event's resolved sources.

### 4.3 `answer_router`
`async def answer_router(graphiti, driver, embedder, synth_client, synth_model,
map_client, map_model, cheap_client, cheap_model, *, q, mode_override, vendor,
settings) -> dict`
1. `mode, via = await classify(q, cheap_client=cheap_client, cheap_model=cheap_model,
   mode_override=mode_override, default_mode=settings.router_default_mode)`.
2. Dispatch (knobs from `settings`):
   - `local` → `answer_local(graphiti, driver, synth_client, synth_model, q=q, vendor=vendor, group_id=…)`
   - `global` → `global_search(driver, embedder, map_client, map_model, synth_client, synth_model, q=q, level=settings.global_default_level, k=settings.global_shortlist_k, group_id=…, relevance_min=settings.global_map_relevance_min)`
   - `drift` → `drift_search(graphiti, driver, embedder, synth_client, synth_model, q=q, level=settings.drift_primer_level, iterations=settings.drift_iterations, primer_k=settings.drift_primer_k, max_followups=settings.drift_max_followups, followup_k=settings.drift_followup_k, group_id=…)`
   - `timeline` → `timeline_local(graphiti, driver, q=q, vendor=vendor, group_id=…)`
3. Fallback: `if mode=="local" and raw.get("retrieved")==0:` re-dispatch as
   `drift`; set `fallback_from="local"`, `mode="drift"`.
4. Normalize `raw` → envelope via `_normalize(mode, via, fallback_from, raw, q)`.

### 4.4 `_normalize`
`def _normalize(mode, via, fallback_from, raw, q) -> dict`
- Always: `{mode, query: q, answer, citations, routing:{chosen:mode, via, fallback_from}}`.
- `timeline` mode: `answer, citations = _render_timeline(raw)`; carry
  `"timeline": raw["timeline"]`.
- Other modes: `answer = raw.get("answer","")`, `citations = raw.get("citations",[])`.
- Carry structured extras when present in `raw`: `communities_used` (global/drift),
  `follow_ups` (drift), `timeline` (timeline).
- DRIFT degrade: `raw` has a `degraded` key → surface it as
  `routing["degraded"] = raw["degraded"]` (no divergent top-level shape).

### 4.5 Endpoint — `GET /answer` (`answer_api/app.py`)
`q: str` (required → 422), `mode: Mode | None = None` (invalid value → 422 via the
`Literal`), `vendor: str | None = None`. Calls module-attr
`router_mod.answer_router(...)` using `app.state.*`. The lifespan builds a cheap
classifier client from `cheap_llm_*` onto `app.state.cheap_client`/`cheap_model`
(or `None`/`""` when `cheap_llm_api_key` is empty — heuristics-only), closed on
shutdown alongside `map_client`. `/search/local`, `/search/global`,
`/search/drift`, `/timeline`, `/health` unchanged. The former `answer_local`
behavior is now reachable via `/answer?mode=local`.

## 5. Config additions (`ExtractSettings`)

```
router_default_mode: str = "drift"   # fallback mode when no heuristic fires and the cheap classifier is absent/uncertain
```
(The classifier client reuses the existing `cheap_llm_base_url/model/api_key`.)

## 6. Uniform response contract

```json
{
  "mode": "local|global|drift|timeline",
  "query": "...",
  "answer": "<markdown>",
  "citations": [{"marker": 1, "fact_uuid": "…", "sources": [{"url": "…", "title": "…", "article_id": "…"}]}],
  "routing": {"chosen": "drift", "via": "heuristic|llm|override|default", "fallback_from": "local"|null},
  "communities_used": [ ... ],
  "follow_ups": [ ... ],
  "timeline": [ ... ]
}
```
`mode`, `query`, `answer`, `citations`, `routing` always present; the three
structured extras appear only for their modes. Citation shape is the existing
`{marker, fact_uuid, sources:[{url,title,article_id}]}` (richer fields deferred).

## 7. Design-decision alignment

- **#2 citations are traversal, never LLM:** the router only classifies (emits a
  mode label) and normalizes; it never synthesizes prose or writes a URL. Every
  `answer`/`citations` pair comes from a mode function or the deterministic
  `_render_timeline`, all of which resolve URLs via `Provenance`. The cheap
  classifier's output is a mode label, never user-facing text.
- **#4 one embedding space:** unchanged — global/drift modes embed the query with
  the shared embedder as before.

## 8. Error handling & degradation

| Condition | Behavior |
|---|---|
| No cheap key (`cheap_client is None`) | Heuristics-only; uncaught → `router_default_mode` (drift). No error. |
| Cheap LLM junk / unparseable / unknown label | `(default_mode, "default")`. |
| `?mode=` invalid value | 422 (FastAPI validates against the `Mode` literal). |
| Missing `q` | 422. |
| Chosen `local` retrieves 0 facts | Escalate to `drift`; `routing.fallback_from="local"`. One escalation, no loops. |
| DRIFT empty-shortlist degrade shape | Normalized into the envelope; `routing.degraded="no-primer-communities"`. |
| A mode function raises | Propagates (a 500) — the router adds no new swallowing; per-mode internal resilience already exists. |

## 9. Testing

- **Unit — `classify`** (fake cheap client): temporal query → timeline;
  cross-vendor query → global; `mode_override` wins over heuristics; an uncaught
  query → cheap-LLM label; `cheap_client=None` → default drift; junk/unknown LLM
  output → default drift.
- **Unit — `_render_timeline`:** events → deterministic markdown + numbered
  top-level citations whose `sources` match the events; empty timeline → empty
  citations + "no changes" answer; no URL in the markdown beyond `sources`.
- **Unit — `answer_router` / `_normalize`** (monkeypatched mode functions returning
  canned dicts, fake cheap client): each mode → correct envelope (extras present
  only for their mode); local-empty (`retrieved==0`) → drift fallback with
  `routing.fallback_from="local"`; DRIFT degrade dict → envelope with
  `routing.degraded`; `routing.via` set correctly per path.
- **App (`test_answer_api_app.py`):** `/answer` returns the uniform envelope with
  `mode`; `?mode=timeline` forces timeline; `?mode=bogus` → 422; missing `q` → 422;
  lifespan builds/omits the cheap client; pre-existing endpoints unchanged. Router
  monkeypatched (seam) — no real infra.
- **`@live` smoke + demonstration:** run several intent-varied queries through
  `/answer` on `backup-docs`; show each routed to the expected mode with a cited
  answer and the `routing` block → `docs/superpowers/answer-router-report.md`.
- Full non-live suite + `ruff check src tests` + mypy clean.

## 10. Acceptance criteria

1. `GET /answer?q=…` classifies to one of local/global/drift/timeline (heuristics
   first, cheap LLM otherwise, default drift) and returns the uniform envelope
   with `mode`, `answer`, `citations`, and a `routing` block explaining the choice.
2. `?mode=` overrides classification; an invalid value → 422.
3. Timeline is normalized to the same envelope via a deterministic markdown render
   + flattened numbered citations (no LLM, no LLM-authored URL).
4. A `local` classification that retrieves nothing escalates to `drift`, recorded
   in `routing.fallback_from`.
5. The DRIFT empty-shortlist degrade shape is folded into the uniform envelope
   (`routing.degraded`), not surfaced as a divergent top-level shape.
6. No cheap key → heuristics-only, default drift; no crash. Design #2 preserved
   (the router authors no prose and no URL).
7. Unit + app tests green; `@live` smoke passes; a demonstration report is written;
   full non-live suite + ruff/mypy clean.

## 11. Deferred

- `freshness` block (`reports_as_of` from the community layer's `generated_at`/
  `corpus_cursor`; `graph_cursor_time` from the latest episode) — part of the
  plan's §6 contract, additive later.
- Richer citation fields (vendor/product/section/valid_at) via extended provenance
  traversal.
- A reliable single-product→local heuristic (currently left to the cheap LLM).
- Additional fallback arcs (e.g. global/drift-no-communities → local); a
  confidence score from the classifier.
- MCP exposure of the modes as separate Copilot tools (Phase 5).
