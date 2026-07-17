# Hybrid Extraction Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route each article to a cheap model (`ling-2.6-flash`) or a strong model (`gpt-5-mini`) by table density, so dense availability matrices go to the reliable strong model and everything else gets the ~23× cost saving.

**Architecture:** A pure `is_dense_matrix(markdown)` classifier decides the tier per article. The ingest driver holds two `ExtractionTier`s (cheap = ling + salience + 900-token chunks; strong = gpt-5-mini + production instructions + 1800) and routes each article to one; both write the same Neo4j group so entity resolution/embeddings are shared. Routing is ON by default and degrades to strong-only when no cheap key is configured.

**Tech Stack:** Python 3.12, graphiti-core on Neo4j, pydantic-settings, pytest (+ Neo4j testcontainers for integration).

## Global Constraints

- Routing is ON by default (`extraction_routing: bool = True`); it degrades to **strong-only** (today's exact behaviour) when `extraction_routing` is False OR `cheap_llm_api_key` is empty. Graceful fallback must never hard-fail.
- The routing decision uses ONLY article markdown — no LLM (design decision #1).
- Both tiers write the same `group_id` and share the single embedder/reranker — one embedding space (design decision #4).
- Cheap tier instructions = `EXTRACTION_INSTRUCTIONS + CHEAP_TIER_SALIENCE`; strong tier = `EXTRACTION_INSTRUCTIONS` only. **Do NOT** add the salience text to the global `EXTRACTION_INSTRUCTIONS`.
- Cheap tier chunk size = `cheap_max_chunk_tokens` (900); strong tier = `max_chunk_tokens` (1800).
- Classifier thresholds are config: `dense_table_line_ratio = 0.25`, `dense_pipe_count = 200`. Route to strong if `ratio ≥ dense_table_line_ratio` OR `pipes ≥ dense_pipe_count`.
- `add_text_episode`'s new `instructions` parameter defaults to `EXTRACTION_INSTRUCTIONS` (backward compatible — existing callers unchanged).

---

## File Structure

- **Create** `src/graph_extract/article_router.py` — the pure `is_dense_matrix` classifier.
- **Modify** `src/graph_extract/ontology.py` — add the `CHEAP_TIER_SALIENCE` constant.
- **Modify** `src/graph_extract/config.py` — cheap-tier fields, thresholds, routing toggle.
- **Modify** `src/graph_extract/graphiti_client.py` — `instructions` param on `add_text_episode`; `ExtractionTier` dataclass; `build_cheap_graphiti`.
- **Modify** `src/graph_extract/ingest_driver.py` — tier-based constructor + per-article routing + `IngestArticleResult.tier`.
- **Modify** `src/graph_extract/cli.py` — `_build_ingest_driver` builds both tiers with graceful fallback; per-tier count in the `ingest` summary.
- **Modify** callers of `IngestDriver(...)`: `tests/unit/test_ingest_skip.py`, `tests/integration/conftest.py`, `tests/integration/test_temporal_update.py` (Task 5).
- **Tests** `tests/unit/test_article_router.py`, additions to `tests/unit/test_ontology.py`, `tests/unit/test_config.py` (create if absent), `tests/integration/test_graphiti_client.py`, `tests/integration/test_ingest_routing.py`.

---

### Task 1: `is_dense_matrix` classifier

**Files:**
- Create: `src/graph_extract/article_router.py`
- Test: `tests/unit/test_article_router.py`

**Interfaces:**
- Produces: `is_dense_matrix(markdown: str, *, ratio_threshold: float = 0.25, pipe_threshold: int = 200) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_article_router.py
from graph_extract.article_router import is_dense_matrix

_DENSE = "intro line\n" + "\n".join("| feat%d | us-east-%d | yes |" % (i, i) for i in range(40))
_PROSE = "AWS Backup centralizes data protection.\n" * 30  # no tables


def test_dense_matrix_true():
    assert is_dense_matrix(_DENSE) is True          # ~40/41 lines are table rows


def test_prose_false():
    assert is_dense_matrix(_PROSE) is False


def test_empty_false():
    assert is_dense_matrix("") is False
    assert is_dense_matrix("   \n  \n") is False


def test_small_table_in_prose_false():
    md = ("AWS Backup notes.\n" * 30) + "\n".join("| a | b |" for _ in range(5))
    assert is_dense_matrix(md) is False              # 5 rows, low ratio, few pipes


def test_pipe_count_triggers_even_if_ratio_low():
    # one very wide table row buried in lots of prose: high pipe count -> dense
    wide = "| " + " | ".join(str(i) for i in range(250)) + " |"
    md = ("prose line\n" * 300) + wide
    assert is_dense_matrix(md) is True               # >=200 pipes


def test_ratio_triggers_even_if_pipes_low():
    # many short table rows: high ratio, modest pipes -> dense
    md = "\n".join("| x |" for _ in range(30)) + "\nprose\nprose"
    assert is_dense_matrix(md) is True               # ratio ~0.9


def test_thresholds_are_configurable():
    md = "\n".join("| x |" for _ in range(3)) + "\nprose\nprose\nprose\nprose\nprose\nprose\nprose"
    # 3/10 = 0.3 ratio; default 0.25 -> dense; raise threshold -> not dense
    assert is_dense_matrix(md) is True
    assert is_dense_matrix(md, ratio_threshold=0.5, pipe_threshold=10_000) is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_article_router.py -q`
Expected: FAIL (module `article_router` does not exist).

- [ ] **Step 3: Implement**

```python
# src/graph_extract/article_router.py
"""Deterministic per-article tier routing for hybrid extraction.

`is_dense_matrix` decides whether an article is a dense availability/support
matrix that the cheap model (ling-2.6-flash) cannot extract reliably (it
over-generates on such tables -> 16k truncation -> lost chunks). Dense articles
route to the strong model; everything else to the cheap one. No I/O, no LLM.
"""
from __future__ import annotations


def is_dense_matrix(markdown: str, *, ratio_threshold: float = 0.25,
                    pipe_threshold: int = 200) -> bool:
    """True if `markdown` is a dense table the cheap tier can't handle.

    A line is a table row when its left-stripped form starts with '|'. Routes
    to the strong tier when the table-line ratio >= ratio_threshold OR the total
    '|' count >= pipe_threshold. Calibration (2026-07): dense articles measured
    33-35% table-line ratio / 264-1415 pipes; clean articles <=17% / <=80.
    """
    lines = markdown.splitlines()
    if not lines:
        return False
    table_lines = sum(1 for ln in lines if ln.lstrip().startswith("|"))
    ratio = table_lines / len(lines)
    pipes = markdown.count("|")
    return ratio >= ratio_threshold or pipes >= pipe_threshold
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_article_router.py -q`
Expected: PASS (7 tests). Then `uv run ruff check src/graph_extract/article_router.py` and `uv run mypy src/graph_extract/article_router.py` — both clean.

- [ ] **Step 5: Commit**

```bash
git add src/graph_extract/article_router.py tests/unit/test_article_router.py
git commit -m "feat(router): is_dense_matrix table-density classifier"
```

---

### Task 2: Config fields + cheap-tier salience constant

**Files:**
- Modify: `src/graph_extract/config.py`
- Modify: `src/graph_extract/ontology.py`
- Test: `tests/unit/test_config.py` (create), additions to `tests/unit/test_ontology.py`

**Interfaces:**
- Produces on `ExtractSettings`: `extraction_routing: bool`, `cheap_llm_base_url: str`, `cheap_llm_model: str`, `cheap_llm_api_key: str`, `cheap_llm_client_mode: Literal[...]`, `cheap_max_chunk_tokens: int`, `dense_table_line_ratio: float`, `dense_pipe_count: int`.
- Produces in `ontology`: `CHEAP_TIER_SALIENCE: str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_config.py
from graph_extract.config import ExtractSettings

_MIN = dict(docext_base_url="http://x", docext_read_key="k", neo4j_uri="bolt://x",
            neo4j_user="u", neo4j_password="p")


def test_routing_defaults_on_with_cheap_ling():
    s = ExtractSettings(_env_file=None, **_MIN)
    assert s.extraction_routing is True
    assert s.cheap_llm_model == "inclusionai/ling-2.6-flash"
    assert s.cheap_llm_base_url == "https://openrouter.ai/api/v1"
    assert s.cheap_llm_client_mode == "generic_json_schema"
    assert s.cheap_llm_api_key == ""            # must be supplied via .env
    assert s.cheap_max_chunk_tokens == 900


def test_dense_thresholds_default():
    s = ExtractSettings(_env_file=None, **_MIN)
    assert s.dense_table_line_ratio == 0.25
    assert s.dense_pipe_count == 200
```

```python
# add to tests/unit/test_ontology.py
def test_cheap_tier_salience_present_and_not_global():
    from graph_extract.ontology import CHEAP_TIER_SALIENCE, EXTRACTION_INSTRUCTIONS
    assert "BE SELECTIVE" in CHEAP_TIER_SALIENCE
    assert "AvailableIn" in CHEAP_TIER_SALIENCE and "Limits" in CHEAP_TIER_SALIENCE
    # cheap-tier-only: the salience text must NOT be in the global instructions
    assert "BE SELECTIVE" not in EXTRACTION_INSTRUCTIONS
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_config.py tests/unit/test_ontology.py::test_cheap_tier_salience_present_and_not_global -q`
Expected: FAIL (`extraction_routing`/`CHEAP_TIER_SALIENCE` do not exist).

- [ ] **Step 3: Implement config fields**

In `src/graph_extract/config.py`, inside `class ExtractSettings`, after the `judge_api_key` field, add:

```python
    # --- hybrid extraction routing (design: hybrid-extraction-router) ---
    # ON by default; degrades to strong-only when cheap_llm_api_key is empty.
    extraction_routing: bool = True
    # Cheap tier = ling-2.6-flash via OpenRouter. api_key from .env (never committed).
    cheap_llm_base_url: str = "https://openrouter.ai/api/v1"
    cheap_llm_model: str = "inclusionai/ling-2.6-flash"
    cheap_llm_api_key: str = ""
    cheap_llm_client_mode: Literal[
        "structured", "generic_json_schema", "generic_json_object"] = "generic_json_schema"
    cheap_max_chunk_tokens: int = 900   # smaller chunks for the verbose cheap model
    # Route an article to the STRONG tier when it is a dense table:
    dense_table_line_ratio: float = 0.25
    dense_pipe_count: int = 200
```

(`Literal` is already imported at the top of `config.py`.)

- [ ] **Step 4: Implement the salience constant**

In `src/graph_extract/ontology.py`, after the `EXTRACTION_INSTRUCTIONS = """..."""` block, add:

```python
# Cheap-tier-only salience appendix (ling-2.6-flash over-enumerates dense tables).
# Appended to EXTRACTION_INSTRUCTIONS for the cheap tier ONLY -- do NOT add to the
# global instructions (it slightly reduces the strong model's extraction).
CHEAP_TIER_SALIENCE = (
    "\n\nBE SELECTIVE, NOT EXHAUSTIVE — never emit one fact per table cell, row, "
    "or feature-x-region combination.\n"
    "For availability/support MATRICES: extract each feature or capability ONCE "
    "(the product provides/supports it), and express availability by EXCEPTION — "
    "state where something is NOT available or is limited (these gaps are the "
    "valuable signal; a wall of checkmarks is not). Type those negative/partial "
    "statements as Limits facts ('not supported', 'only', 'except') and positive "
    "residency statements as AvailableIn facts. Availability and limitation "
    "statements are the HIGHEST-VALUE facts: keep them, but state each ONCE at "
    "the most general level that is TRUE (e.g. '<feature> is not available in "
    "China regions') — never over-generalize an exception into a global claim.\n"
    "Use short canonical entity names ('Amazon S3', 'cross-Region copy'), never "
    "descriptive phrases. A dense table chunk should yield roughly 10-25 facts "
    "total, not dozens."
)
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run --extra dev pytest tests/unit/test_config.py tests/unit/test_ontology.py -q`
Expected: PASS. Then `uv run ruff check src/graph_extract/config.py src/graph_extract/ontology.py` and `uv run mypy src/graph_extract/config.py src/graph_extract/ontology.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add src/graph_extract/config.py src/graph_extract/ontology.py tests/unit/test_config.py tests/unit/test_ontology.py
git commit -m "feat(config): cheap-tier fields + dense thresholds + CHEAP_TIER_SALIENCE"
```

---

### Task 3: Parametrise `add_text_episode` with `instructions`

**Files:**
- Modify: `src/graph_extract/graphiti_client.py:149-157`
- Test: `tests/integration/test_graphiti_client.py`

**Interfaces:**
- Consumes: `ENTITY_TYPES`, `EDGE_TYPES`, `EDGE_TYPE_MAP`, `EXCLUDED_ENTITY_TYPES`, `EXTRACTION_INSTRUCTIONS` (already imported in `graphiti_client.py`).
- Produces: `add_text_episode(graphiti, s, *, name, body, source_description, reference_time, instructions: str = EXTRACTION_INSTRUCTIONS) -> AddEpisodeResults`.

- [ ] **Step 1: Write the failing test** (a fast unit-style test with a fake graphiti — no live LLM)

```python
# add to tests/integration/test_graphiti_client.py
import pytest


@pytest.mark.asyncio
async def test_add_text_episode_passes_custom_instructions():
    from datetime import datetime, timezone
    from graph_extract.graphiti_client import add_text_episode
    from graph_extract.config import ExtractSettings

    captured = {}

    class _FakeG:
        async def add_episode(self, **kw):
            captured.update(kw)
            return "ok"

    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    await add_text_episode(_FakeG(), s, name="n", body="b", source_description="u",
                           reference_time=datetime.now(timezone.utc),
                           instructions="CUSTOM-INSTR")
    assert captured["custom_extraction_instructions"] == "CUSTOM-INSTR"
    assert captured["group_id"] == s.group_id


@pytest.mark.asyncio
async def test_add_text_episode_defaults_to_global_instructions():
    from datetime import datetime, timezone
    from graph_extract.graphiti_client import add_text_episode
    from graph_extract.ontology import EXTRACTION_INSTRUCTIONS
    from graph_extract.config import ExtractSettings

    captured = {}

    class _FakeG:
        async def add_episode(self, **kw):
            captured.update(kw)
            return "ok"

    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    await add_text_episode(_FakeG(), s, name="n", body="b", source_description="u",
                           reference_time=datetime.now(timezone.utc))
    assert captured["custom_extraction_instructions"] == EXTRACTION_INSTRUCTIONS
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/integration/test_graphiti_client.py -k "custom_instructions or defaults_to_global" -q`
Expected: FAIL (`add_text_episode` has no `instructions` param).

- [ ] **Step 3: Implement**

Replace `add_text_episode` in `src/graph_extract/graphiti_client.py` with:

```python
async def add_text_episode(graphiti: Graphiti, s: ExtractSettings, *, name: str,
                           body: str, source_description: str,
                           reference_time: datetime,
                           instructions: str = EXTRACTION_INSTRUCTIONS) -> AddEpisodeResults:
    return await graphiti.add_episode(
        name=name, episode_body=body, source_description=source_description,
        reference_time=reference_time, source=EpisodeType.text, group_id=s.group_id,
        entity_types=ENTITY_TYPES, excluded_entity_types=EXCLUDED_ENTITY_TYPES,
        edge_types=EDGE_TYPES, edge_type_map=EDGE_TYPE_MAP,
        custom_extraction_instructions=instructions)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --extra dev pytest tests/integration/test_graphiti_client.py -k "custom_instructions or defaults_to_global" -q`
Expected: PASS. `uv run ruff check` + `uv run mypy src/graph_extract/graphiti_client.py` — clean.

- [ ] **Step 5: Commit**

```bash
git add src/graph_extract/graphiti_client.py tests/integration/test_graphiti_client.py
git commit -m "feat(graphiti): per-tier instructions param on add_text_episode"
```

---

### Task 4: `ExtractionTier` dataclass + `build_cheap_graphiti`

**Files:**
- Modify: `src/graph_extract/graphiti_client.py`
- Test: `tests/integration/test_graphiti_client.py`

**Interfaces:**
- Consumes: `build_graphiti(s) -> Graphiti` (existing), `ExtractSettings.model_copy`.
- Produces: `@dataclass class ExtractionTier: name: str; graphiti: Graphiti; instructions: str; max_chunk_tokens: int` and `build_cheap_graphiti(s: ExtractSettings) -> Graphiti`.

- [ ] **Step 1: Write the failing test** (monkeypatch `build_graphiti` so no real client is built)

```python
# add to tests/integration/test_graphiti_client.py
def test_build_cheap_graphiti_uses_cheap_model(monkeypatch):
    import graph_extract.graphiti_client as gc
    from graph_extract.config import ExtractSettings

    seen = {}
    monkeypatch.setattr(gc, "build_graphiti", lambda cfg: seen.update(
        model=cfg.llm_model, base=cfg.llm_base_url, key=cfg.llm_api_key,
        mode=cfg.llm_client_mode) or "G")

    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                        cheap_llm_api_key="or-key")
    out = gc.build_cheap_graphiti(s)
    assert out == "G"
    assert seen == {"model": "inclusionai/ling-2.6-flash",
                    "base": "https://openrouter.ai/api/v1",
                    "key": "or-key", "mode": "generic_json_schema"}


def test_extraction_tier_holds_fields():
    from graph_extract.graphiti_client import ExtractionTier
    t = ExtractionTier(name="cheap", graphiti="G", instructions="I", max_chunk_tokens=900)
    assert (t.name, t.graphiti, t.instructions, t.max_chunk_tokens) == ("cheap", "G", "I", 900)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/integration/test_graphiti_client.py -k "cheap_graphiti or extraction_tier" -q`
Expected: FAIL (`build_cheap_graphiti`/`ExtractionTier` do not exist).

- [ ] **Step 3: Implement**

Add `from dataclasses import dataclass` to the imports of `src/graph_extract/graphiti_client.py`, and add near the top (after imports):

```python
@dataclass
class ExtractionTier:
    """One extraction tier: which graphiti client, which instructions, which chunk
    size. Two are used by the hybrid router (cheap=ling, strong=gpt-5-mini)."""
    name: str            # "cheap" | "strong" -> IngestArticleResult.tier
    graphiti: Graphiti
    instructions: str
    max_chunk_tokens: int
```

And add this function (place it right after `build_graphiti`):

```python
def build_cheap_graphiti(s: ExtractSettings) -> Graphiti:
    """Build a Graphiti whose LLM points at the cheap model (ling). Reuses
    build_graphiti with a cheap-tier settings view; embedder/reranker/Neo4j and
    the group_id are unchanged, so both tiers share one embedding space."""
    cheap = s.model_copy(update=dict(
        llm_base_url=s.cheap_llm_base_url, llm_model=s.cheap_llm_model,
        llm_api_key=s.cheap_llm_api_key, llm_client_mode=s.cheap_llm_client_mode))
    return build_graphiti(cheap)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run --extra dev pytest tests/integration/test_graphiti_client.py -k "cheap_graphiti or extraction_tier" -q`
Expected: PASS. `ruff` + `mypy src/graph_extract/graphiti_client.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/graph_extract/graphiti_client.py tests/integration/test_graphiti_client.py
git commit -m "feat(graphiti): ExtractionTier + build_cheap_graphiti"
```

---

### Task 5: `IngestDriver` per-article routing

**Files:**
- Modify: `src/graph_extract/ingest_driver.py`
- Modify (constructor callers): `src/graph_extract/cli.py:88`, `tests/unit/test_ingest_skip.py:43`, `tests/integration/conftest.py:80`, `tests/integration/test_temporal_update.py:10`
- Test: `tests/integration/test_ingest_routing.py` (create)

**Interfaces:**
- Consumes: `is_dense_matrix` (Task 1), `ExtractionTier` (Task 4), `add_text_episode(..., instructions=)` (Task 3), settings `dense_table_line_ratio`/`dense_pipe_count` (Task 2).
- Produces: `IngestDriver(settings, strong_tier: ExtractionTier, cheap_tier: ExtractionTier | None, docext, provenance, driver)`; `IngestArticleResult.tier: str`.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/integration/test_ingest_routing.py
import pytest
from types import SimpleNamespace

pytestmark = pytest.mark.asyncio(loop_scope="module")


class _RecordingTier:
    def __init__(self, name):
        self.name, self.calls, self.instructions, self.max_chunk_tokens = name, [], f"instr-{name}", 1234
        self.graphiti = self

    async def add_episode(self, **kw):        # graphiti stand-in
        self.calls.append(kw["episode_body"])
        return SimpleNamespace(episode=SimpleNamespace(uuid=f"ep-{len(self.calls)}"),
                               nodes=[], edges=[])


async def _driver(extract_driver, monkeypatch, strong, cheap, dense):
    import graph_extract.ingest_driver as idmod
    from graph_extract.ingest_driver import IngestDriver
    from graph_extract.config import get_extract_settings

    # stub fetch + chunking + provenance so the test is deterministic and offline
    monkeypatch.setattr(idmod.content_fetch, "fetch_article", _fake_fetch)
    monkeypatch.setattr(idmod, "is_dense_matrix", lambda md, **k: dense)
    async def _chunk(ch, text, model): return [idmod.chonkie_client.Chunk(text="body", start_index=0, end_index=4, token_count=10)]
    monkeypatch.setattr(idmod.chonkie_client, "neural_chunk", _chunk)
    s = get_extract_settings.__wrapped__()
    prov = idmod.Provenance(extract_driver)
    return IngestDriver(s, strong, cheap, None, prov, extract_driver)


async def _fake_fetch(docext, aid):
    return SimpleNamespace(id=aid, title="T", content_markdown="md", source_url="u",
                           last_updated_at=None, extracted_at=None)


async def test_prose_routes_to_cheap(extract_driver, monkeypatch):
    strong, cheap = _RecordingTier("strong"), _RecordingTier("cheap")
    d = await _driver(extract_driver, monkeypatch, strong, cheap, dense=False)
    res = await d.ingest_article("a-prose")
    assert res.tier == "cheap"
    assert len(cheap.calls) == 1 and len(strong.calls) == 0


async def test_dense_routes_to_strong(extract_driver, monkeypatch):
    strong, cheap = _RecordingTier("strong"), _RecordingTier("cheap")
    d = await _driver(extract_driver, monkeypatch, strong, cheap, dense=True)
    res = await d.ingest_article("a-dense")
    assert res.tier == "strong"
    assert len(strong.calls) == 1 and len(cheap.calls) == 0


async def test_no_cheap_tier_falls_back_to_strong(extract_driver, monkeypatch):
    strong = _RecordingTier("strong")
    d = await _driver(extract_driver, monkeypatch, strong, None, dense=False)
    res = await d.ingest_article("a-prose")
    assert res.tier == "strong" and len(strong.calls) == 1
```

*(Note: `add_text_episode` calls `tier.graphiti.add_episode`; `_RecordingTier` is its own graphiti. The driver must pass `instructions=tier.instructions` and build episodes with `tier.max_chunk_tokens` — the recording tier ignores those but the driver must reference them.)*

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/integration/test_ingest_routing.py -q`
Expected: FAIL (`IngestDriver` still takes the old `graphiti` arg; no `tier`).

- [ ] **Step 3: Implement the driver changes**

In `src/graph_extract/ingest_driver.py`:

1. Add imports near the top:
```python
from graph_extract.article_router import is_dense_matrix
from graph_extract.graphiti_client import add_text_episode, ExtractionTier
```
(remove the old `from graph_extract.graphiti_client import add_text_episode` if it duplicates.)

2. Add `tier` to the result dataclass:
```python
@dataclass
class IngestArticleResult:
    article_id: str
    episodes_added: int = 0
    episodes_skipped: int = 0
    entities: int = 0
    edges: int = 0
    skipped_navigation: bool = False
    tier: str = "strong"
```

3. Replace `__init__` and `ingest_article`:
```python
    def __init__(self, settings: ExtractSettings, strong_tier: ExtractionTier,
                 cheap_tier: ExtractionTier | None, docext: httpx.AsyncClient,
                 provenance: Provenance, driver: AsyncDriver):
        self._s = settings
        self._strong = strong_tier
        self._cheap = cheap_tier
        self._docext = docext
        self._prov = provenance
        self._driver = driver

    def _tier_for(self, markdown: str) -> ExtractionTier:
        if self._cheap is None:
            return self._strong
        dense = is_dense_matrix(
            markdown, ratio_threshold=self._s.dense_table_line_ratio,
            pipe_threshold=self._s.dense_pipe_count)
        return self._strong if dense else self._cheap

    async def ingest_article(self, article_id: str) -> IngestArticleResult:
        res = IngestArticleResult(article_id=article_id)
        art = await content_fetch.fetch_article(self._docext, article_id)
        if is_navigation_article(art.title or ""):
            res.skipped_navigation = True
            return res
        tier = self._tier_for(art.content_markdown)
        res.tier = tier.name
        ref = _parse_ts(art)
        content_hash = await self._content_hash(article_id) or _sha(art.content_markdown)
        chapter_path = await self._chapter_path(article_id)
        async with httpx.AsyncClient(base_url=self._s.chonkie_base_url, timeout=120) as ch:
            chunks = await chonkie_client.neural_chunk(ch, art.content_markdown, self._s.chonkie_model)
        episodes = episode_builder.build_episodes(
            article_id=art.id, title=art.title, chapter_path=chapter_path,
            content_hash=content_hash, chunks=chunks,
            max_chunk_tokens=tier.max_chunk_tokens, min_chunk_tokens=self._s.min_chunk_tokens)
        for e in episodes:
            if await self._prov.already_ingested(art.id, e.chunk_index, e.content_hash):
                res.episodes_skipped += 1
                continue
            r = await add_text_episode(tier.graphiti, self._s, name=e.name, body=e.body,
                                       source_description=art.source_url, reference_time=ref,
                                       instructions=tier.instructions)
            await self._prov.link(art.id, r.episode.uuid, chunk_index=e.chunk_index,
                                  heading_path=e.heading_path, token_count=e.token_count,
                                  content_hash=e.content_hash)
            res.episodes_added += 1
            res.entities += len(r.nodes)
            res.edges += len(r.edges)
        await self._supersede_trailing_episodes(art.id, len(episodes))
        return res
```
(Delete the old `self._g` field and its use.)

- [ ] **Step 4: Update the other `IngestDriver(...)` callers so the suite compiles**

`src/graph_extract/cli.py:88` — wrap the built graphiti in a strong tier, cheap tier None (Task 6 replaces this with the real builder):
```python
        strong_tier = ExtractionTier("strong", graphiti, EXTRACTION_INSTRUCTIONS,
                                     settings.max_chunk_tokens)
        ingest = IngestDriver(settings, strong_tier, None, docext, provenance, driver)
```
Add imports to `cli.py`: `from graph_extract.graphiti_client import build_graphiti, init_indices, build_cheap_graphiti, ExtractionTier` and `from graph_extract.ontology import EXTRACTION_INSTRUCTIONS, CHEAP_TIER_SALIENCE`.

`tests/unit/test_ingest_skip.py:43`, `tests/integration/conftest.py:80` — wrap their `graphiti` the same way:
```python
    from graph_extract.graphiti_client import ExtractionTier
    from graph_extract.ontology import EXTRACTION_INSTRUCTIONS
    strong = ExtractionTier("strong", graphiti, EXTRACTION_INSTRUCTIONS, s.max_chunk_tokens)
    driver = IngestDriver(s, strong, None, <docext>, provenance, <driver>)
```
(use each call site's existing variable names for `<docext>`/`<driver>`; in `test_temporal_update.py:10` the call is `IngestDriver(None, None, None, None, extract_driver)` → make it `IngestDriver(None, None, None, None, None, extract_driver)` — it never touches a tier in that test.)

- [ ] **Step 5: Run to verify pass + no regressions**

Run: `uv run --extra dev pytest tests/integration/test_ingest_routing.py tests/unit/test_ingest_skip.py tests/integration/test_temporal_update.py -q`
Expected: PASS. `ruff` + `mypy src/graph_extract/ingest_driver.py` clean.

- [ ] **Step 6: Commit**

```bash
git add src/graph_extract/ingest_driver.py src/graph_extract/cli.py tests/unit/test_ingest_skip.py tests/integration/conftest.py tests/integration/test_temporal_update.py tests/integration/test_ingest_routing.py
git commit -m "feat(ingest): per-article tier routing via is_dense_matrix"
```

---

### Task 6: CLI builds both tiers with graceful fallback

**Files:**
- Modify: `src/graph_extract/cli.py` (`_build_ingest_driver`, `ingest` summary)
- Test: `tests/unit/test_extract_cli.py`

**Interfaces:**
- Consumes: `build_graphiti`, `build_cheap_graphiti`, `ExtractionTier`, `EXTRACTION_INSTRUCTIONS`, `CHEAP_TIER_SALIENCE`, settings routing fields.
- Produces: `_build_ingest_driver` returns an `IngestDriver` with a real cheap tier when `settings.extraction_routing and settings.cheap_llm_api_key`, else `cheap_tier=None`.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/unit/test_extract_cli.py
import pytest


@pytest.mark.asyncio
async def test_build_ingest_driver_builds_cheap_tier_when_configured(monkeypatch):
    import graph_extract.cli as cli
    from graph_extract.config import ExtractSettings

    built = []
    monkeypatch.setattr(cli, "build_graphiti", lambda s: built.append("strong") or "SG")
    monkeypatch.setattr(cli, "build_cheap_graphiti", lambda s: built.append("cheap") or "CG")
    monkeypatch.setattr(cli, "make_docext_client", lambda **k: _FakeAsync())
    monkeypatch.setattr(cli, "AsyncGraphDatabase", _FakeNeo())
    async def _noop(g): return None
    monkeypatch.setattr(cli, "init_indices", _noop)

    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                        cheap_llm_api_key="or-key")   # routing on by default
    ingest, g, dx, drv = await cli._build_ingest_driver(s)
    assert "cheap" in built and ingest._cheap is not None
    assert ingest._cheap.instructions.endswith(cli.CHEAP_TIER_SALIENCE[-40:])
    assert ingest._cheap.max_chunk_tokens == s.cheap_max_chunk_tokens


@pytest.mark.asyncio
async def test_build_ingest_driver_strong_only_without_cheap_key(monkeypatch):
    import graph_extract.cli as cli
    from graph_extract.config import ExtractSettings

    monkeypatch.setattr(cli, "build_graphiti", lambda s: "SG")
    monkeypatch.setattr(cli, "build_cheap_graphiti", lambda s: (_ for _ in ()).throw(AssertionError("must not build cheap")))
    monkeypatch.setattr(cli, "make_docext_client", lambda **k: _FakeAsync())
    monkeypatch.setattr(cli, "AsyncGraphDatabase", _FakeNeo())
    async def _noop(g): return None
    monkeypatch.setattr(cli, "init_indices", _noop)

    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")  # no cheap key
    ingest, g, dx, drv = await cli._build_ingest_driver(s)
    assert ingest._cheap is None
```

Add these helpers at the top of `tests/unit/test_extract_cli.py` if not present:
```python
class _FakeAsync:
    async def aclose(self): pass
    async def close(self): pass

class _FakeNeo:
    def driver(self, *a, **k): return _FakeAsync()
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_extract_cli.py -k build_ingest_driver -q`
Expected: FAIL (no cheap tier built; `_cheap` is None / attribute error).

- [ ] **Step 3: Implement `_build_ingest_driver`**

Rewrite `_build_ingest_driver` in `src/graph_extract/cli.py` so `cheap_graphiti` is initialised to `None` before the `try` (so the `except` can always close it), the cheap tier is built inside the `try`, and the `except` closes `cheap_graphiti` first. Full function:

```python
async def _build_ingest_driver(
    settings: ExtractSettings,
) -> tuple[IngestDriver, Graphiti, httpx.AsyncClient, AsyncDriver]:
    graphiti = build_graphiti(settings)
    cheap_graphiti: Graphiti | None = None
    docext: httpx.AsyncClient | None = None
    driver: AsyncDriver | None = None
    try:
        docext = make_docext_client(
            base_url=settings.docext_base_url, read_key=settings.docext_read_key,
            admin_key=settings.docext_admin_key, verify_tls=settings.docext_verify_tls,
            admin=False,
        )
        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        await init_indices(graphiti)
        provenance = Provenance(driver)
        strong_tier = ExtractionTier("strong", graphiti, EXTRACTION_INSTRUCTIONS,
                                     settings.max_chunk_tokens)
        cheap_tier: ExtractionTier | None = None
        if settings.extraction_routing and settings.cheap_llm_api_key:
            cheap_graphiti = build_cheap_graphiti(settings)
            await init_indices(cheap_graphiti)
            cheap_tier = ExtractionTier(
                "cheap", cheap_graphiti,
                EXTRACTION_INSTRUCTIONS + CHEAP_TIER_SALIENCE,
                settings.cheap_max_chunk_tokens)
        ingest = IngestDriver(settings, strong_tier, cheap_tier, docext, provenance, driver)
    except Exception:
        for closer in (
            cheap_graphiti.close if cheap_graphiti is not None else None,
            driver.close if driver is not None else None,
            docext.aclose if docext is not None else None,
            graphiti.close,
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception:
                logger.exception("error closing a resource during _build_ingest_driver cleanup")
        raise
    return ingest, graphiti, docext, driver
```

The cheap graphiti is reachable at `ingest._cheap.graphiti` for the shutdown path (next step). `_build_ingest_driver`'s return tuple is unchanged (Task 5 already relies on it).

- [ ] **Step 4: Close the cheap tier on shutdown + report routing mode**

In the `ingest` command's `finally` block, also close the cheap graphiti; and update the summary echo. Replace the `try/finally` body of the `ingest` command with:

```python
        try:
            res = await ingest_driver.ingest_source(source_id, limit)
            typer.echo(
                f"ingest complete: articles={res.articles} "
                f"episodes_added={res.episodes_added} "
                f"episodes_skipped={res.episodes_skipped} "
                f"routing={'hybrid' if ingest_driver._cheap is not None else 'strong-only'}")
            typer.echo("--- cost (this run) ---")
            _dump(await cost_report(res.episodes_added))
        finally:
            await driver.close()
            await docext.aclose()
            await graphiti.close()
            if ingest_driver._cheap is not None:
                await ingest_driver._cheap.graphiti.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run --extra dev pytest tests/unit/test_extract_cli.py -q`
Expected: PASS. `ruff` + `mypy src/graph_extract/cli.py` clean.

- [ ] **Step 5: Commit**

```bash
git add src/graph_extract/cli.py tests/unit/test_extract_cli.py
git commit -m "feat(cli): build cheap+strong tiers with graceful strong-only fallback"
```

---

### Task 7: Full-suite gate + runbook note

**Files:**
- Modify: `docs/superpowers/maintenance-runbook.md` (or the ingest section) — one paragraph on routing.

- [ ] **Step 1: Run the full non-live suite**

Run: `uv run --extra dev pytest -m "not live" -q`
Expected: PASS (all green, no regressions). Fix any fallout before continuing.

- [ ] **Step 2: Document routing**

Add a short paragraph wherever ingestion is documented (e.g. `docs/superpowers/maintenance-runbook.md`): hybrid routing is ON by default; set `CHEAP_LLM_API_KEY` (OpenRouter) in `.env` to enable the cheap tier, else it runs strong-only; dense-matrix articles (table-line ratio ≥ `DENSE_TABLE_LINE_RATIO` or pipes ≥ `DENSE_PIPE_COUNT`) go to gpt-5-mini, the rest to ling; `IngestArticleResult.tier` reports which handled each article.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/maintenance-runbook.md
git commit -m "docs: hybrid routing runbook note"
```

---

## Notes for the implementer

- Run all Python via `uv run` (e.g. `uv run --extra dev pytest ...`, `uv run ruff check ...`, `uv run mypy ...`).
- `ExtractSettings(_env_file=None, ...)` in tests bypasses `.env` so tests are hermetic; always pass the 5 required fields (`docext_base_url`, `docext_read_key`, `neo4j_uri`, `neo4j_user`, `neo4j_password`).
- Integration tests use the `extract_driver` Neo4j-testcontainer fixture (see `tests/integration/conftest.py`); they are NOT `@pytest.mark.live`.
- Do not add the salience text to the global `EXTRACTION_INSTRUCTIONS` (Global Constraints).
