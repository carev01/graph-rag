# Hybrid Extraction Router — Design Spec

**Project:** Temporal GraphRAG over DocExtractor
**Slice:** Extraction-layer routing — send each article to a cheap model
(`ling-2.6-flash`) or a strong model (`gpt-5-mini`) based on table density, so
dense availability matrices (which the cheap model can't extract reliably) go to
the strong model and everything else gets the ~23× cost saving.
**Date:** 2026-07-17
**Status:** Approved design — ready for implementation planning

Grounded in `docs/superpowers/ling-production-readiness.md`: `ling-2.6-flash` is
~23× cheaper with comparable quality on prose, but over-generates unboundedly on
dense availability/support-matrix tables → 16k-token truncation → retry storms →
lost chunks + 20–60× slowdown, unfixable by any lever. The hybrid routes the
dense minority to `gpt-5-mini` and keeps the cheap tier for the majority.

---

## 1. Scope

**In:** a pure `is_dense_matrix` classifier; an `ExtractionTier` bundling the
per-tier differences (graphiti client, extraction instructions, chunk size); two
tiers wired into `IngestDriver` with per-article routing; config for the cheap
tier + thresholds + a routing toggle (default ON, graceful fallback to strong-only
when the cheap key is absent); `add_text_episode` parametrised for per-tier
instructions + chunk size; a `tier` field on the ingest result for observability.

**Out:** per-chunk fallback (rejected — ling's retry storms happen *before* it
fails, so per-chunk fallback wouldn't avoid the slowness; per-article routing dodges
the storms entirely); changing the semantic layer, retrieval, or the deterministic
correctors; committing the v4 salience globally (it is cheap-tier-only); tuning the
strong tier.

## 2. Grounding (calibrated thresholds)

Table density measured on the known dense (fail) vs clean (ok) articles:

| article | table-line % | pipe count |
|---|---|---|
| DENSE AWS feature availability | 35.4% | 1415 |
| DENSE Azure support matrix | 33.3% | 264 |
| clean Cross-Region backup | 0% | 0 |
| clean Encryption AWS | 10.6% | 80 |
| clean Azure VM restore | 17.1% | 28 |
| clean CloudTrail logging | 0% | 0 |

Table-line ratio and total pipe-count both cleanly separate dense (≥33% / ≥264)
from clean (≤17% / ≤80). Largest-table-block does NOT (clean Encryption has 20
rows > dense Azure's 11), so it is not used. Chosen thresholds (configurable):
route to **strong** if `table_line_ratio ≥ 0.25` OR `pipe_count ≥ 200`.

## 3. Architecture & flow

```
ingest_article(article_id):
  fetch article markdown
  tier = strong_tier if is_dense_matrix(markdown) else (cheap_tier or strong_tier)
  chunks   = neural_chunk(markdown)
  episodes = build_episodes(chunks, max_chunk_tokens = tier.max_chunk_tokens)
  for each new episode:
      add_text_episode(tier.graphiti, ..., instructions=tier.instructions,
                       max_chunk_tokens=tier.max_chunk_tokens)  # writes to shared group
      provenance.link(...)                                       # identical for both tiers
  result.tier = tier.name
```

Both tiers write the same Neo4j `group_id`, so Graphiti entity resolution is
shared across tiers (a cheap-tier `Amazon S3` and a strong-tier `Amazon S3`
dedup to one node). Routing is per-article and deterministic; no LLM in the
routing decision.

## 4. Components

### 4.1 `article_router.py` (new) — the classifier
```python
def is_dense_matrix(markdown: str, *, ratio_threshold: float = 0.25,
                    pipe_threshold: int = 200) -> bool:
    """True if the article is a dense availability/support matrix the cheap
    model can't extract reliably. Deterministic, no I/O. A line is a table row
    if its lstrip()'d form starts with '|'. Routes to the strong tier when the
    table-line ratio >= ratio_threshold OR total '|' count >= pipe_threshold."""
```
Empty/whitespace markdown → `False` (prose). Pure function; the single source of
truth for "dense".

### 4.2 `ExtractionTier` (dataclass) — `graphiti_client.py`
```python
@dataclass
class ExtractionTier:
    name: str            # "cheap" | "strong"  (goes into IngestArticleResult.tier)
    graphiti: Graphiti
    instructions: str    # EXTRACTION_INSTRUCTIONS (+ SALIENCE for cheap)
    max_chunk_tokens: int
```
Bundles exactly what differs per tier so `IngestDriver` stays tier-agnostic.

### 4.3 Cheap-tier salience constant — `ontology.py`
The v4 salience text (from the production-readiness tuning) becomes a named
constant `CHEAP_TIER_SALIENCE` in `ontology.py`, appended to
`EXTRACTION_INSTRUCTIONS` for the cheap tier only. NOT added to the global
`EXTRACTION_INSTRUCTIONS` (the strong tier does not need it and it slightly
reduces gpt-5-mini extraction). Verbatim text is in the plan.

### 4.4 Config additions — `config.py`
```
extraction_routing: bool = True          # ON by default
cheap_llm_base_url: str = "https://openrouter.ai/api/v1"
cheap_llm_model: str = "inclusionai/ling-2.6-flash"
cheap_llm_api_key: str = ""              # OpenRouter key, from .env (never committed)
cheap_llm_client_mode: Literal["structured","generic_json_schema","generic_json_object"] = "generic_json_schema"
cheap_max_chunk_tokens: int = 900        # smaller chunks for the verbose cheap model
dense_table_line_ratio: float = 0.25
dense_pipe_count: int = 200
```
The strong tier reuses the existing `llm_*` fields (Azure gpt-5-mini) and
`max_chunk_tokens` (1800). **Graceful fallback:** if `extraction_routing` is False
OR `cheap_llm_api_key` is empty, no cheap tier is built and every article uses the
strong tier — behaviour identical to today's single-model path.

### 4.5 Cheap-graphiti builder — `graphiti_client.py`
`build_cheap_graphiti(s: ExtractSettings) -> Graphiti` builds a Graphiti whose LLM
client points at the cheap model, by constructing a cheap-tier settings view
(`s.model_copy(update={llm_base_url: cheap_llm_base_url, llm_model: cheap_llm_model,
llm_api_key: cheap_llm_api_key, llm_client_mode: cheap_llm_client_mode})`) and
reusing the existing `build_graphiti`. The embedder/reranker/Neo4j are unchanged
(one embedding space across tiers — plan §4 invariant #4-adjacent). The OpenRouter
provider-preference injection already triggers on `"openrouter" in base_url`.

### 4.6 `IngestDriver` routing — `ingest_driver.py`
`IngestDriver.__init__` gains `strong_tier: ExtractionTier` and
`cheap_tier: ExtractionTier | None` (replacing the single `graphiti`/`settings`
usage in `ingest_article`; the driver keeps `settings` for chonkie/model-independent
config). `ingest_article`:
- picks `tier = strong_tier if is_dense_matrix(markdown, ...) else (cheap_tier or strong_tier)`;
- builds episodes at `tier.max_chunk_tokens` (in the `build_episodes` call — this
  is where chunk size applies, NOT in `add_text_episode`);
- calls `add_text_episode(tier.graphiti, ..., instructions=tier.instructions)` per
  episode;
- sets `res.tier = tier.name`.
Thresholds read from settings. `already_ingested` / `provenance.link` /
`_supersede_trailing_episodes` unchanged. `IngestArticleResult` gains
`tier: str = "strong"`.

### 4.7 `add_text_episode` parametrisation — `graphiti_client.py`
Currently hardcodes `custom_extraction_instructions=EXTRACTION_INSTRUCTIONS`. Add
`instructions: str = EXTRACTION_INSTRUCTIONS` (and, if build_episodes is called
inside it — it is not; chunking is in the driver — no chunk-size param needed
there). The driver passes `tier.instructions`. Existing single-arg callers keep the
default (backward compatible).

### 4.8 CLI wiring — `cli.py`
`_build_ingest_driver` builds the strong tier always and the cheap tier when
`settings.extraction_routing and settings.cheap_llm_api_key`. On failure partway
through, every already-built graphiti/driver/docext is closed (extend the existing
build-or-cleanup guard). The `ingest` command's cost report already reads the
process-wide token tally, which now spans both tiers; add a per-tier article count
line to the echoed summary.

## 5. Design-decision alignment

- **#1 (structural is deterministic):** routing reads only article markdown; no LLM.
- **#4 (one group / one embedding space):** both tiers share the group_id and the
  single embedder/reranker — cross-tier dedup works, embeddings stay comparable.
- **Provenance (#2):** unchanged — `provenance.link` runs identically per tier;
  citations still resolve fact→episode→article→url.
- The strong tier remains exactly today's config (no regression when routing off).

## 6. Error handling & fallback

- **No cheap key / routing off:** strong-only, identical to today.
- **A cheap-tier episode still truncates** (rare prose case): unchanged Graphiti
  behaviour — it retries and may drop the chunk; not the router's concern (the
  router's job is only to keep dense articles off the cheap tier).
- **Cheap endpoint down mid-run:** the per-episode call raises; surfaced by the
  existing ingest error path (no new swallowing).

## 7. Testing

- **Unit — `is_dense_matrix`:** the 6 calibrated articles' shapes as fixtures
  (dense→True, clean→False); empty/whitespace→False; threshold edges (a table at
  exactly 25% ratio; pipe count at 200); a prose article with one small table stays
  False.
- **Unit — tier selection:** given a stub classifier, `ingest_article` picks the
  strong tier for dense and cheap for prose, and falls back to strong when
  `cheap_tier is None`.
- **Integration — routing:** two monkeypatched graphiti stubs (record which
  received the episode); a dense article routes to strong, a prose article to
  cheap; `res.tier` correct; both link provenance.
- **Integration — graceful fallback:** `extraction_routing` off (or empty cheap
  key) → only the strong graphiti is built/used; existing ingest tests stay green.
- **Backward-compat:** `add_text_episode` default-instructions path unchanged;
  existing `IngestDriver` tests updated for the tier constructor.
- Full non-live suite + ruff/mypy clean.

## 8. Acceptance criteria

1. With routing ON + a cheap key, dense-matrix articles are extracted by
   `gpt-5-mini` and all others by `ling-2.6-flash`; `IngestArticleResult.tier`
   reports which.
2. `is_dense_matrix` classifies the 6 calibrated articles correctly and is
   configurable via `dense_table_line_ratio` / `dense_pipe_count`.
3. Cheap tier applies `EXTRACTION_INSTRUCTIONS + CHEAP_TIER_SALIENCE` and
   `cheap_max_chunk_tokens`; strong tier uses production instructions + 1800.
4. With routing OFF or no cheap key, the pipeline is byte-for-byte today's
   single-model path (graceful fallback; existing tests green).
5. Both tiers write one group / one embedding space; provenance resolves for both.
6. Unit + integration tests green; ruff/mypy clean.

## 9. Deferred

- Auto-detecting cheap-tier failures and re-routing that article to strong on the
  next run (a learned deny-list); a cost dashboard; per-source tier overrides.
- Re-tuning `dense_*` thresholds as the corpus widens past AWS/Azure (they are
  config, so no code change needed).
- Committing v4 salience globally / a strong-tier salience variant.
