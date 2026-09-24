# Pre-bootstrap Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Suppress same-pair contradiction invalidation at ingest, treat a future `invalid_at` as current, add optional chunk packing, and build the paid D6 A/B harness that decides the packing target.

**Architecture:** Suppression lives in `dedup_guard`'s existing wrapper of graphiti's edge-dedup LLM call (the only point that neutralises both of graphiti's uses of `contradicted_facts`). The current-fact rule is one helper used by `search_local` and `/timeline`. Packing is a post-step in `episode_builder`, default off. The A/B harness drives `IngestDriver` against two throwaway Neo4j testcontainers.

**Tech Stack:** Python 3.12, graphiti-core 0.30.1 (pinned, never modified), Neo4j 2026.07.1 Community, pytest + testcontainers, typer.

**Spec:** `docs/superpowers/specs/2026-09-23-pre-bootstrap-fixes-design.md`. Evidence: `docs/superpowers/pre-bootstrap-decisions-2026-09-23.md`.

## Global Constraints

- graphiti-core stays pinned at 0.30.1 and unmodified.
- `dedup_guard` never rewrites `duplicate_facts`. Suppression clears ONLY `contradicted_facts`, on EVERY return path of the dedup branch (clean, out-of-range, retried, parse-failure), AFTER all counting.
- New setting `ingest_same_pair_contradictions: bool = False` (False = suppressed). New counter `DedupIndexStats.contradictions_suppressed`.
- A fact is current iff `invalid_at is None or invalid_at > now` (UTC; naive datetimes treated as UTC).
- New setting `pack_target_tokens: int = 0` (0 = off, today's behaviour exactly). Packing merges CONSECUTIVE chunks while the total stays `<= min(pack_target_tokens, max_chunk_tokens)`, joining text with `"\n"`.
- The A/B harness never writes the `.env` Neo4j; each arm uses its own testcontainer. Arms: `today` = production settings; `pack1200` = `pack_target_tokens=1200, cheap_max_chunk_tokens=1200`. Spend cap $5 total. 2-article smoke first; refuse to continue unless arm `pack1200` added FEWER episodes than `today` on the smoke articles AND the usage tally moved.
- Prices ($/1M in, out): cheap tier (solar-pro4) 0.09 / 0.36; strong tier (gpt-5-mini) 0.25 / 2.00.
- Tests hermetic: no LLM calls, never the `.env` Neo4j. Lint gate lints tests too: `uv run ruff check src tests`; types `uv run mypy src`.
- CI gate: `uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest -m "not live"`.
- Mutation patches assert `old in s` before replacing.
- Subagents never run the A/B harness and never launch long background jobs.

---

### Task 1: Suppress same-pair contradictions in `dedup_guard`

**Files:**
- Modify: `src/graph_extract/dedup_guard.py` (module docstring; `DedupIndexStats` ~line 58; `install_dedup_guard` ~line 188-300)
- Modify: `src/graph_extract/config.py` (next to `ingest_detect_contradictions`, ~line 105)
- Modify: `src/graph_extract/cli.py` (both `install_dedup_guard(...)` calls in `_build_ingest_driver`)
- Modify: `CLAUDE.md` (the `dedup_guard.py` bullet in "Module layout"; the "Current status" note under design decision 3)
- Test: `tests/unit/test_dedup_guard.py`, `tests/unit/test_cli_dedup_wiring.py`

**Interfaces:**
- Produces: `install_dedup_guard(graphiti, *, fallback, unscoped, timings=None, suppress_contradictions: bool = False)`; `DedupIndexStats.contradictions_suppressed: int`; `ExtractSettings.ingest_same_pair_contradictions: bool = False`.

- [ ] **Step 1: Write the failing tests** — append to `tests/unit/test_dedup_guard.py`, reusing its existing helpers (`_ScriptedLLM`, `_dedup_response`, `_call`, `_edge`, `_episode`, `_T0`, and whatever the file already uses to build a related/existing edge list — read the file first and use its real helper names and signatures):

```python
async def test_suppression_clears_contradictions_and_keeps_duplicates():
    primary = _ScriptedLLM(_dedup_response([1], [0, 2]))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=None, unscoped=stats,
                        suppress_contradictions=True)
    reply = await _call(primary, 3, 0)
    assert reply["contradicted_facts"] == []
    assert reply["duplicate_facts"] == [1]
    assert stats.contradicted_same_pair == 2      # counted BEFORE suppression
    assert stats.contradictions_suppressed == 2


async def test_no_suppression_by_default():
    primary = _ScriptedLLM(_dedup_response([], [0]))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=None, unscoped=stats)
    reply = await _call(primary, 3, 0)
    assert reply["contradicted_facts"] == [0]
    assert stats.contradictions_suppressed == 0


async def test_suppression_applies_on_the_parse_failure_path():
    primary = _ScriptedLLM(_dedup_response([], [0]))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=None, unscoped=stats,
                        suppress_contradictions=True)
    reply = await primary.generate_response(
        [Message(role="user", content="not a parseable dedup prompt")],
        prompt_name=DEDUP_PROMPT_NAME)
    assert reply["contradicted_facts"] == []
    assert stats.parse_failures == 1 and stats.contradictions_suppressed == 1


async def test_suppression_applies_to_the_fallback_reply():
    primary = _ScriptedLLM(_dedup_response([15], [0]))       # out of range -> retry
    fallback = _ScriptedLLM(_dedup_response([], [1]))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=fallback,
                        unscoped=stats, suppress_contradictions=True)
    reply = await _call(primary, 3, 0)
    assert reply["contradicted_facts"] == []
    assert stats.retried == 1 and stats.contradictions_suppressed == 1


async def test_suppressed_same_pair_contradiction_invalidates_nothing_end_to_end():
    """Through graphiti's real resolve_extracted_edge: the related edge is NOT
    invalidated and the new edge is NOT expired, although the model said 'contradicts'."""
    # Build `llm`, `related`, `existing` exactly as the existing library-pin test
    # that asserts `invalidated == [related[0]]` does, but install the guard with
    # suppress_contradictions=True on that llm before calling.
    ...
    resolved, invalidated, _ = await resolve_extracted_edge(
        llm, _edge("new", valid_at=_T0), related, existing, _episode())
    assert invalidated == []
    assert resolved.expired_at is None
```

The last test's `...` is deliberate: copy the setup lines from the existing pin test (the one ending `assert invalidated == [related[0]]`, ~line 140-152) verbatim and add the guard install with `suppress_contradictions=True`. If `_call`'s signature differs from `(llm, n, m)`, use its real signature. If a reply is not a dict in these helpers, adapt the assertions to the helper's real return type — do not change the guard's behaviour to fit the test.

In `tests/unit/test_cli_dedup_wiring.py`, following that file's existing pattern for capturing `install_dedup_guard` calls, add a test asserting every captured call received `suppress_contradictions=True` under default settings, and `False` when `ingest_same_pair_contradictions=True`.

- [ ] **Step 2: Run to verify failure**

Run: `uv run --extra dev pytest tests/unit/test_dedup_guard.py tests/unit/test_cli_dedup_wiring.py -q`
Expected: the new tests FAIL (`TypeError: ... unexpected keyword argument 'suppress_contradictions'`).

- [ ] **Step 3: Implement**

`DedupIndexStats`, directly after `contradicted_same_pair: int = 0`:

```python
    # contradicted_facts entries cleared before graphiti saw them (D4 suppression,
    # ingest_same_pair_contradictions=False). Equals every contradiction the model
    # asserted while suppression was on.
    contradictions_suppressed: int = 0
```

`install_dedup_guard`: add the keyword parameter `suppress_contradictions: bool = False` and this closure helper before `generate_response`:

```python
    def _suppressed(response: Any, stats: DedupIndexStats) -> Any:
        """Clear contradicted_facts so graphiti invalidates nothing on this fact --
        neither the old edge (resolve_edge_contradictions) nor the new one (the
        inline later-valid_at block). duplicate_facts is never touched."""
        if not suppress_contradictions:
            return response
        if not isinstance(response, dict):
            raise TypeError(
                f"dedup reply is {type(response).__name__}, not dict; cannot suppress "
                "contradictions (graphiti's generate_response contract changed?)")
        contra = response.get("contradicted_facts") or []
        if not contra:
            return response
        stats.contradictions_suppressed += len(contra)
        return {**response, "contradicted_facts": []}
```

Then route every return of the dedup branch through it:
- parse-failure: `return _suppressed(await _timed(orig, DEDUP_PROMPT_NAME, messages, *args, **kwargs), stats)`
- clean path: `return _suppressed(response, stats)` (the `if verdict is None or verdict.clean:` return)
- final return after the out-of-range handling: `return _suppressed(response, stats)`

The non-dedup return (`if name != DEDUP_PROMPT_NAME`) is unchanged.

Module docstring: append step `5. when suppress_contradictions is set (D4), clear contradicted_facts after counting -- see pre-bootstrap-decisions-2026-09-23.md.` and change the "What it deliberately does NOT do" paragraph's first sentence to: "What it deliberately does NOT do: rewrite, filter or reinterpret the model's DUPLICATE indices, and it never adds a per-call `maximum` to the schema." (keep the rest of that paragraph).

`config.py`, after `ingest_detect_contradictions: bool = False`:

```python
    # Same-pair contradiction (BACKLOG 33): when the dedup model says a new fact
    # contradicts an existing fact between the SAME two entities, graphiti expires
    # one of them. Measured 0 of 14 genuine on the live graph -- limitations and
    # refinements read as contradictions, removing core facts from answers
    # (pre-bootstrap-decisions-2026-09-23.md D4). False clears contradicted_facts in
    # dedup_guard; genuine change is carried by updates and the weekly sweep.
    ingest_same_pair_contradictions: bool = False
```

`cli.py` `_build_ingest_driver`: add `suppress_contradictions=not settings.ingest_same_pair_contradictions` to BOTH `install_dedup_guard(...)` calls.

`CLAUDE.md`: in the `dedup_guard.py` bullet replace "never rewrites indices" with "never rewrites duplicate indices; clears `contradicted_facts` when same-pair suppression is on (`INGEST_SAME_PAIR_CONTRADICTIONS=false`, the default)". In design decision 3's "Current status" note, replace the sentence beginning "Only *same-pair* contradictions can still set `invalid_at` at ingest" with: "Same-pair contradictions are also suppressed by default since 2026-09-23 (`ingest_same_pair_contradictions=False`; 0 of 14 live cases were genuine — `docs/superpowers/pre-bootstrap-decisions-2026-09-23.md`), so ingest writes no contradiction invalidations; an end date stated in the text still sets `invalid_at`, and the weekly sweep is the invalidation mechanism."

- [ ] **Step 4: Run to verify pass**

Run: `uv run --extra dev pytest tests/unit/test_dedup_guard.py tests/unit/test_cli_dedup_wiring.py -q`
Expected: PASS.

- [ ] **Step 5: Mutation-test** (assert `old in s`; restore; never commit): (a) make `_suppressed` return `response` unconditionally; (b) remove `_suppressed(...)` from the parse-failure return; (c) remove it from the final (out-of-range) return; (d) clear `duplicate_facts` as well. Each must be killed.

- [ ] **Step 6: Gate and commit**

Run the full CI gate. Commit: `feat(dedup-guard): suppress same-pair contradictions at ingest (D4)`.

---

### Task 2: A future `invalid_at` is current

**Files:**
- Create: `src/answer_api/temporal.py`
- Modify: `src/answer_api/search.py:38`, `src/answer_api/timeline.py` (`_fact_status`, ~line 29)
- Test: `tests/unit/test_temporal.py` (new); extend the existing unit tests of `search_local` and `timeline` (find them with `grep -rln "search_local\|_fact_status" tests/unit`)

**Interfaces:**
- Produces: `answer_api.temporal.is_current(invalid_at: datetime | None, now: datetime | None = None) -> bool`.

- [ ] **Step 1: Failing tests** — `tests/unit/test_temporal.py`:

```python
from datetime import datetime, timedelta, timezone

from answer_api.temporal import is_current

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


def test_no_end_date_is_current():
    assert is_current(None, NOW) is True


def test_a_past_end_date_is_not_current():
    assert is_current(NOW - timedelta(days=1), NOW) is False


def test_a_future_end_date_is_current():
    assert is_current(datetime(2028, 9, 1, tzinfo=timezone.utc), NOW) is True


def test_naive_datetimes_are_read_as_utc():
    assert is_current(datetime(2028, 9, 1), NOW) is True
    assert is_current(datetime(2026, 3, 31), NOW) is False


def test_now_defaults_to_the_clock():
    assert is_current(datetime(2999, 1, 1, tzinfo=timezone.utc)) is True
```

In the existing `search_local` unit tests add: a fake edge with `invalid_at` in the future is RETURNED with default `include_invalid=False`, and one in the past is filtered out. In the existing timeline tests add: `_fact_status(<future datetime>, False) == "current"`, `_fact_status(<past datetime>, False) == "superseded"`, `_fact_status(<past datetime>, True) == "expired"`.

- [ ] **Step 2: Run to verify failure** — `uv run --extra dev pytest tests/unit/test_temporal.py -q` → FAIL (module missing).

- [ ] **Step 3: Implement** — `src/answer_api/temporal.py`:

```python
"""When a fact counts as current (BACKLOG 41).

A fact whose end date is stated in the text and lies in the FUTURE -- "ADE is
retired in September 2028" -- is still true today. Treating any non-null
invalid_at as "no longer valid" hid such facts from answers.
"""
from __future__ import annotations

from datetime import datetime, timezone


def is_current(invalid_at: datetime | None, now: datetime | None = None) -> bool:
    """True while the fact has no end date or its end date is still ahead."""
    if invalid_at is None:
        return True
    if invalid_at.tzinfo is None:
        invalid_at = invalid_at.replace(tzinfo=timezone.utc)
    return invalid_at > (now or datetime.now(timezone.utc))
```

`search.py`: `from answer_api.temporal import is_current` and replace line 38 with
`edges = [e for e in edges if is_current(getattr(e, "invalid_at", None))]`.

`timeline.py`: import `is_current`; change `_fact_status` to

```python
def _fact_status(invalid_at, expired_by_sweep) -> str:
    if is_current(invalid_at):
        return "current"
    if expired_by_sweep:
        return "expired"
    return "superseded"
```

- [ ] **Step 4: Run to verify pass**, then mutation-test: (a) `return invalid_at > ...` → `return False` (after the None check); (b) drop the naive-to-UTC line; (c) revert `search.py:38` to `is None`. Each must be killed.

- [ ] **Step 5: Gate and commit** — `fix(answer-api): a future end date keeps a fact current (BACKLOG 41)`.

---

### Task 3: Optional chunk packing

**Files:**
- Modify: `src/graph_extract/episode_builder.py`
- Modify: `src/graph_extract/config.py` (next to `min_chunk_tokens`, ~line 52)
- Modify: `src/graph_extract/ingest_driver.py` (the `build_episodes(...)` call, ~line 184)
- Modify: `src/graph_extract/probe.py` (its `build_episodes(...)` call, ~line 37)
- Test: `tests/unit/test_episode_builder.py`

**Interfaces:**
- Produces: `build_episodes(..., pack_target_tokens: int = 0)`; `ExtractSettings.pack_target_tokens: int = 0`.

- [ ] **Step 1: Failing tests** — append to `tests/unit/test_episode_builder.py`:

```python
def _c(text: str, tok: int) -> Chunk:
    return Chunk(text=text, start_index=0, end_index=len(text), token_count=tok)


def _eps(chunks, *, pack=0, max_tokens=1800, min_tokens=128):
    return build_episodes(article_id="a", title="T", chapter_path="", content_hash="h" * 16,
                          chunks=chunks, max_chunk_tokens=max_tokens,
                          min_chunk_tokens=min_tokens, pack_target_tokens=pack)


def test_packing_off_is_todays_behaviour():
    chunks = [_c(f"c{i}", 300) for i in range(5)]
    assert [e.body for e in _eps(chunks)] == [e.body for e in _eps(chunks, pack=0)]
    assert len(_eps(chunks)) == 5


def test_packing_merges_consecutive_chunks_up_to_the_target():
    chunks = [_c(f"c{i}", 300) for i in range(5)]
    eps = _eps(chunks, pack=900)
    assert [e.token_count for e in eps] == [900, 600]
    assert eps[0].body.endswith("c0\nc1\nc2")


def test_packing_never_exceeds_the_tier_cap():
    chunks = [_c(f"c{i}", 300) for i in range(5)]
    assert max(e.token_count for e in _eps(chunks, pack=1500, max_tokens=900)) <= 900


def test_packing_preserves_content_and_order():
    chunks = [_c(f"c{i}", 250) for i in range(7)]
    body = lambda eps: "\n".join(e.body.split("\n", 1)[1] for e in eps)
    assert body(_eps(chunks, pack=1000)) == body(_eps(chunks))


def test_packing_leaves_a_chunk_already_at_the_target_alone():
    chunks = [_c("big", 1000), _c("small", 200)]
    assert [e.token_count for e in _eps(chunks, pack=1000)] == [1000, 200]
```

- [ ] **Step 2: Run to verify failure** — `uv run --extra dev pytest tests/unit/test_episode_builder.py -q` → FAIL (unexpected keyword `pack_target_tokens`).

- [ ] **Step 3: Implement** — in `episode_builder.py`, factor the join out of `_merge_tiny` so packing reuses it (no duplicated logic):

```python
def _join(p: Chunk, c: Chunk) -> Chunk:
    return Chunk(text=p.text + "\n" + c.text, start_index=p.start_index,
                 end_index=c.end_index, token_count=p.token_count + c.token_count)

def _merge_tiny(chunks: list[Chunk], min_tokens: int, max_tokens: int) -> list[Chunk]:
    out: list[Chunk] = []
    for c in chunks:
        if (out and c.token_count < min_tokens
                and out[-1].token_count + c.token_count <= max_tokens):
            out[-1] = _join(out[-1], c)
        else:
            out.append(c)
    return out

def _pack(chunks: list[Chunk], limit: int) -> list[Chunk]:
    """Greedily merge CONSECUTIVE chunks while the total stays <= limit (D6).
    ~70% of an episode's prompt volume is fixed per-episode overhead, so fewer,
    larger episodes cost less (pre-bootstrap-decisions-2026-09-23.md)."""
    out: list[Chunk] = []
    for c in chunks:
        if out and out[-1].token_count + c.token_count <= limit:
            out[-1] = _join(out[-1], c)
        else:
            out.append(c)
    return out
```

`build_episodes`: add keyword `pack_target_tokens: int = 0` and, directly after `sized = _merge_tiny(...)`:

```python
    if pack_target_tokens > 0:
        sized = _pack(sized, min(pack_target_tokens, max_chunk_tokens))
```

`config.py` after `min_chunk_tokens: int = 128`:

```python
    # Pack consecutive chunks into episodes of up to this many tokens (never above
    # the tier's max_chunk_tokens). 0 = off. Set ONCE before the bootstrap: changing
    # it re-keys an ingested article's episodes on its next re-ingest. Decided by the
    # D6 A/B (pre-bootstrap-decisions-2026-09-23.md).
    pack_target_tokens: int = 0
```

`ingest_driver.py` and `probe.py`: pass `pack_target_tokens=<settings>.pack_target_tokens` to their `build_episodes(...)` calls (`self._s` in IngestDriver; the settings object in scope in probe).

- [ ] **Step 4: Run to verify pass**; mutation-test: (a) `<= limit` → `< limit`; (b) drop the `min(..., max_chunk_tokens)` clamp; (c) make the `if pack_target_tokens > 0` block unconditional with limit `max_chunk_tokens`. Each killed.

- [ ] **Step 5: Gate and commit** — `feat(episode-builder): optional chunk packing, default off (D6)`.

---

### Task 4: The D6 A/B harness

The controller — not a subagent — runs it (it is paid). The subagent writes it and unit-tests its pure helpers.

**Files:**
- Create: `scripts/chunk_ab.py`
- Test: `tests/unit/test_chunk_ab.py`

**Interfaces:**
- Consumes: `graph_extract.cli._build_ingest_driver(settings) -> (IngestDriver, Graphiti, httpx.AsyncClient, AsyncDriver)`; `graph_extract.usage.get_tally()` (`.prompt_tokens`, `.completion_tokens`); `IngestDriver.ingest_article(article_id) -> IngestArticleResult` (`.episodes_added`, `.tier`, `.dedup`); `episode_builder.build_episodes`; `graph_extract.article_router.is_dense_matrix`; Task 3's `pack_target_tokens`.
- Produces: `select_articles(sample, n, rng, settings) -> list[dict]`, `today_episode_count(article, settings) -> int`, `article_cost(prompt, completion, tier) -> float`, `summarise(rows) -> dict`.

- [ ] **Step 1: Failing tests** — `tests/unit/test_chunk_ab.py`:

```python
import importlib.util
import random
from pathlib import Path

from graph_extract.config import ExtractSettings

_spec = importlib.util.spec_from_file_location(
    "chunk_ab", Path(__file__).parents[2] / "scripts" / "chunk_ab.py")
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)

S = ExtractSettings(_env_file=None, neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


def _art(i, vendor, n_chunks, tok=300):
    return {"id": f"a{i}", "vendor": vendor, "title": f"T{i}",
            "chunks": [{"t": f"chunk {j} " + "w " * 50, "tok": tok} for j in range(n_chunks)]}


def test_today_episode_count_uses_the_production_builder():
    assert ab.today_episode_count(_art(0, "V", 5), S) == 5


def test_select_prefers_multi_episode_articles_across_vendors():
    sample = [_art(i, v, n) for i, (v, n) in enumerate(
        [("A", 5), ("A", 5), ("A", 1), ("B", 4), ("B", 2), ("C", 6)])]
    picked = ab.select_articles(sample, 3, random.Random(0), S)
    assert {a["vendor"] for a in picked} == {"A", "B", "C"}
    assert all(ab.today_episode_count(a, S) >= 3 for a in picked)


def test_article_cost_uses_the_tier_price():
    assert ab.article_cost(1_000_000, 0, "cheap") == 0.09
    assert ab.article_cost(0, 1_000_000, "strong") == 2.00


def test_summarise_totals_and_ratios():
    rows = [{"episodes": 3, "facts": 10, "entities": 6, "seconds": 30.0, "cost": 0.02},
            {"episodes": 1, "facts": 5, "entities": 4, "seconds": 10.0, "cost": 0.01}]
    s = ab.summarise(rows)
    assert s["episodes"] == 4 and s["facts"] == 15 and s["entities"] == 10
    assert abs(s["cost"] - 0.03) < 1e-9 and s["seconds"] == 40.0
```

Adjust `S`'s constructor to whatever `ExtractSettings` requires (copy from `tests/unit/test_extract_config.py`). If `today_episode_count(_art(0,"V",5), S)` is not 5 because `is_dense_matrix` or `min_chunk_tokens` changes the count, fix the fixture (e.g. token size), not the function.

- [ ] **Step 2: Run to verify failure** — `uv run --extra dev pytest tests/unit/test_chunk_ab.py -q` → FAIL.

- [ ] **Step 3: Implement** — `scripts/chunk_ab.py`:

```python
"""D6 A/B: today's chunking vs packing to 1,200 tokens. PAID -- authorised 2026-09-23,
cap $5. Each arm writes into its OWN throwaway Neo4j testcontainer; the .env graph is
never touched. Articles run sequentially through IngestDriver.ingest_article (this
bypasses ingest_source's warm-up barrier and sibling spreading, which do not interact
with the chunking variable under test).

    uv run python scripts/chunk_ab.py --sample SAMPLE.json --out DIR [--articles 30] [--smoke-only]

SAMPLE.json is scripts/spikes/pre_bootstrap/d6_sample.py's output.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from testcontainers.neo4j import Neo4jContainer

from graph_extract.article_router import is_dense_matrix
from graph_extract.chonkie_client import Chunk
from graph_extract.cli import _build_ingest_driver
from graph_extract.config import ExtractSettings, get_extract_settings
from graph_extract.episode_builder import build_episodes
from graph_extract.usage import get_tally

PRICES = {"cheap": (0.09, 0.36), "strong": (0.25, 2.00)}  # $/1M in, out
SPEND_CAP_USD = 5.0
SMOKE_N = 2
ARMS: dict[str, dict[str, Any]] = {
    "today": {},
    "pack1200": {"pack_target_tokens": 1200, "cheap_max_chunk_tokens": 1200},
}
NEO4J_IMAGE = "neo4j:2026.07.1-community"


def today_episode_count(article: dict, s: ExtractSettings) -> int:
    chunks = [Chunk(text=c["t"], start_index=0, end_index=0, token_count=c["tok"])
              for c in article["chunks"]]
    md = "\n".join(c["t"] for c in article["chunks"])
    dense = is_dense_matrix(md, ratio_threshold=s.dense_table_line_ratio,
                            pipe_threshold=s.dense_pipe_count)
    cap = s.max_chunk_tokens if dense else s.cheap_max_chunk_tokens
    return len(build_episodes(article_id=article["id"], title=article["title"],
                              chapter_path="", content_hash="x" * 16, chunks=chunks,
                              max_chunk_tokens=cap, min_chunk_tokens=s.min_chunk_tokens))


def select_articles(sample: list[dict], n: int, rng: random.Random,
                    s: ExtractSettings, min_episodes: int = 3) -> list[dict]:
    """Articles with >= min_episodes episodes today, round-robin across vendors."""
    by_vendor: dict[str, list[dict]] = {}
    for a in sample:
        if today_episode_count(a, s) >= min_episodes:
            by_vendor.setdefault(a["vendor"] or "?", []).append(a)
    for group in by_vendor.values():
        rng.shuffle(group)
    out: list[dict] = []
    while len(out) < n and any(by_vendor.values()):
        for v in sorted(by_vendor):
            if by_vendor[v] and len(out) < n:
                out.append(by_vendor[v].pop())
    return out


def article_cost(prompt: int, completion: int, tier: str) -> float:
    pin, pout = PRICES[tier]
    return (prompt * pin + completion * pout) / 1_000_000


def summarise(rows: list[dict]) -> dict:
    keys = ("episodes", "facts", "entities", "seconds", "cost")
    return {k: sum(r[k] for r in rows) for k in keys} | {"articles": len(rows)}


async def _seed(driver, articles: list[dict]) -> None:
    await driver.execute_query(
        "UNWIND $rows AS r MERGE (a:Article {id: r.id}) "
        "SET a.source_url = 'https://example.invalid/' + r.id, a.title = r.title "
        "MERGE (c:Chapter {id: 'ch-' + r.id}) SET c.title = r.title "
        "MERGE (a)-[:IN_CHAPTER]->(c)",
        rows=[{"id": a["id"], "title": a["title"]} for a in articles])


async def _article_graph(driver, article_id: str) -> dict:
    r = await driver.execute_query(
        "MATCH (:Article {id: $a})-[:HAS_EPISODE]->(e:Episodic) "
        "OPTIONAL MATCH (e)-[:MENTIONS]->(n:Entity) "
        "WITH collect(DISTINCT e.uuid) AS eps, collect(DISTINCT n.name) AS ents "
        "OPTIONAL MATCH ()-[f:RELATES_TO]->() WHERE any(u IN f.episodes WHERE u IN eps) "
        "RETURN size(ents) AS entities, count(DISTINCT f) AS facts, "
        "collect(DISTINCT f.fact) AS fact_texts, ents",
        a=article_id)
    rec = r.records[0]
    return {"entities": rec["entities"], "facts": rec["facts"],
            "fact_texts": sorted(rec["fact_texts"]), "entity_names": sorted(rec["ents"])}


async def run_arm(name: str, articles: list[dict], spent_before: float) -> list[dict]:
    """Ingest `articles` in a fresh container under arm `name`'s settings."""
    rows: list[dict] = []
    with Neo4jContainer(NEO4J_IMAGE) as neo:
        s = get_extract_settings().model_copy(update={
            "neo4j_uri": neo.get_connection_url(), "neo4j_user": "neo4j",
            "neo4j_password": neo.password, "group_id": f"ab-{name}", **ARMS[name]})
        ingest, graphiti, docext, driver = await _build_ingest_driver(s)
        try:
            await _seed(driver, articles)
            for a in articles:
                t = get_tally()
                p0, c0 = t.prompt_tokens, t.completion_tokens
                start = time.perf_counter()
                res = await ingest.ingest_article(a["id"])
                seconds = time.perf_counter() - start
                t = get_tally()
                dp, dc = t.prompt_tokens - p0, t.completion_tokens - c0
                g = await _article_graph(driver, a["id"])
                rows.append({"arm": name, "article_id": a["id"], "vendor": a["vendor"],
                             "tier": res.tier, "episodes": res.episodes_added,
                             "facts": g["facts"], "entities": g["entities"],
                             "seconds": seconds, "prompt_tokens": dp,
                             "completion_tokens": dc, "cost": article_cost(dp, dc, res.tier),
                             "dedup": asdict(res.dedup), "fact_texts": g["fact_texts"],
                             "entity_names": g["entity_names"]})
                spent = spent_before + sum(r["cost"] for r in rows)
                print(f"[{name}] {len(rows)}/{len(articles)} {a['id']} episodes={res.episodes_added} "
                      f"facts={g['facts']} ${spent:.3f}", flush=True)
                if spent > SPEND_CAP_USD:
                    raise SystemExit(f"spend cap ${SPEND_CAP_USD} exceeded (${spent:.2f}); stopping")
        finally:
            await driver.close()
            await docext.aclose()
            await graphiti.close()
    return rows


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--articles", type=int, default=30)
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    s = get_extract_settings()
    articles = select_articles(json.loads(Path(args.sample).read_text()), args.articles,
                               random.Random(7), s)
    print(f"{len(articles)} articles selected", flush=True)

    # Smoke: prove the variable is engaged before the full spend (CLAUDE.md).
    smoke = articles[:SMOKE_N]
    tally_before = get_tally().prompt_tokens
    a_rows = await run_arm("today", smoke, 0.0)
    b_rows = await run_arm("pack1200", smoke, sum(r["cost"] for r in a_rows))
    a_eps, b_eps = sum(r["episodes"] for r in a_rows), sum(r["episodes"] for r in b_rows)
    spent = sum(r["cost"] for r in a_rows + b_rows)
    print(f"smoke: today episodes={a_eps} pack1200 episodes={b_eps} spent=${spent:.3f}", flush=True)
    if get_tally().prompt_tokens == tally_before:
        raise SystemExit("usage tally did not move: cost is not being measured -- refusing")
    if not b_eps < a_eps:
        raise SystemExit("packing did not reduce episodes on the smoke articles -- the "
                         "variable is not engaged; refusing to spend on the full run")
    (out / "smoke.json").write_text(json.dumps(a_rows + b_rows, indent=1))
    if args.smoke_only:
        return

    results: dict[str, Any] = {}
    for arm in ARMS:
        rows = await run_arm(arm, articles, spent)
        spent += sum(r["cost"] for r in rows)
        results[arm] = {"summary": summarise(rows), "rows": rows}
    results["spent_usd_including_smoke"] = spent
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))
    for arm in ARMS:
        print(arm, results[arm]["summary"], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run to verify pass** — `uv run --extra dev pytest tests/unit/test_chunk_ab.py -q` and `uv run ruff check scripts/chunk_ab.py`.

- [ ] **Step 5: Commit** — `feat(scripts): D6 chunk-packing A/B harness (paid; controller-run)`.

- [ ] **Step 6 (controller only):** run the smoke, then the full A/B; write `docs/superpowers/chunk-packing-ab-2026-09-23.md` with per-arm totals, paired per-article ratios, a hand-judged sample of facts present in one arm and missing from the other, and the recommendation per the spec's decision rule.
