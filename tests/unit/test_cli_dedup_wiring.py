"""`_build_ingest_driver` wires the dedup guard: detection on both tiers, retry from
the cheap tier onto the strong tier's UNGUARDED client, governed by
`dedup_retry_on_strong`. Every external constructor is stubbed; nothing connects.
"""
from __future__ import annotations

import pytest

from graph_extract import cli
from graph_extract.config import ExtractSettings
from graph_extract.dedup_guard import CURRENT_DEDUP_STATS, DEDUP_PROMPT_NAME, DedupIndexStats
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.lib import prompt_library

pytestmark = pytest.mark.asyncio


class _LLM:
    def __init__(self, reply: dict) -> None:
        self.reply = reply
        self.calls = 0

    async def generate_response(self, messages, *args, **kwargs):
        self.calls += 1
        return self.reply


class _G:
    def __init__(self, llm: _LLM) -> None:
        self.llm_client = llm

    async def close(self) -> None:
        return None


def _settings(**kw) -> ExtractSettings:
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                cheap_llm_api_key="or-key")
    base.update(kw)
    return ExtractSettings(**base)


@pytest.fixture
def stubs(monkeypatch):
    strong = _LLM({"duplicate_facts": [1], "contradicted_facts": []})
    cheap = _LLM({"duplicate_facts": [10], "contradicted_facts": []})

    async def init(_, **_kwargs):
        return None

    class _Docext:
        async def aclose(self):
            return None

    class _Driver:
        async def close(self):
            return None

    monkeypatch.setattr(cli, "build_graphiti", lambda s: _G(strong))
    monkeypatch.setattr(cli, "build_cheap_graphiti", lambda s: _G(cheap))
    monkeypatch.setattr(cli, "init_indices", init)
    monkeypatch.setattr(cli, "make_docext_client", lambda **k: _Docext())
    monkeypatch.setattr(cli.AsyncGraphDatabase, "driver", lambda *a, **k: _Driver())
    monkeypatch.setattr(cli, "Provenance", lambda d: object())
    return strong, cheap


def _dedup_messages():
    ctx = {"existing_edges": [{"idx": i, "fact": f"r{i}"} for i in range(10)],
           "edge_invalidation_candidates": [{"idx": 10 + i, "fact": f"e{i}"} for i in range(6)],
           "new_edge": "new"}
    return prompt_library.dedupe_edges.resolve_edge(ctx)


async def _dedup_call(graphiti) -> dict:
    return await graphiti.llm_client.generate_response(
        _dedup_messages(), response_model=None, model_size=ModelSize.small,
        prompt_name=DEDUP_PROMPT_NAME)


async def test_cheap_tier_retries_on_the_strong_client_and_is_counted_once(stubs):
    strong, cheap = stubs
    # Explicit: the default is OFF since the 2026-09-11 measurement showed the
    # strong tier repeats the mistake on 24% of retries.
    ingest, graphiti, docext, driver = await cli._build_ingest_driver(
        _settings(dedup_retry_on_strong=True))
    stats = DedupIndexStats()
    token = CURRENT_DEDUP_STATS.set(stats)
    try:
        out = await _dedup_call(ingest._cheap.graphiti)
    finally:
        CURRENT_DEDUP_STATS.reset(token)
    assert out == {"duplicate_facts": [1], "contradicted_facts": []}
    assert (cheap.calls, strong.calls) == (1, 1)
    # ONE dedup call in the scope: the retry went through the strong client's raw
    # method, not through the strong tier's own guard.
    assert (stats.calls, stats.invalid_calls, stats.retried, stats.retry_clean) == (1, 1, 1, 1)


async def test_strong_tier_is_guarded_for_detection_only(stubs):
    strong, cheap = stubs
    strong.reply = {"duplicate_facts": [15], "contradicted_facts": []}
    ingest, graphiti, docext, driver = await cli._build_ingest_driver(_settings())
    stats = DedupIndexStats()
    token = CURRENT_DEDUP_STATS.set(stats)
    try:
        out = await _dedup_call(ingest._strong.graphiti)
    finally:
        CURRENT_DEDUP_STATS.reset(token)
    assert out == strong.reply                      # no fallback: pass-through
    assert (stats.calls, stats.invalid_calls, stats.retried) == (1, 1, 0)
    assert stats.dup_in_invalidation_range == 1


async def test_suppress_contradictions_defaults_to_true_and_follows_the_flag(stubs, monkeypatch):
    """D4: both install_dedup_guard call sites must pass suppress_contradictions
    derived from ingest_same_pair_contradictions (inverted -- default False means
    suppression on)."""
    calls: list[dict] = []
    real = cli.install_dedup_guard

    def _capture(*args, **kwargs):
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "install_dedup_guard", _capture)

    await cli._build_ingest_driver(_settings())
    assert calls, "install_dedup_guard was never called"
    assert all(kw["suppress_contradictions"] is True for kw in calls)

    calls.clear()
    await cli._build_ingest_driver(_settings(ingest_same_pair_contradictions=True))
    assert calls, "install_dedup_guard was never called"
    assert all(kw["suppress_contradictions"] is False for kw in calls)


async def test_retry_switch_off_leaves_the_cheap_reply_alone(stubs):
    """Off is the DEFAULT: `_settings()` passes no flag, so this also pins that a
    plain configuration does not spend a strong-tier call per event."""
    strong, cheap = stubs
    ingest, *_ = await cli._build_ingest_driver(_settings())
    stats = DedupIndexStats()
    token = CURRENT_DEDUP_STATS.set(stats)
    try:
        out = await _dedup_call(ingest._cheap.graphiti)
    finally:
        CURRENT_DEDUP_STATS.reset(token)
    assert out == {"duplicate_facts": [10], "contradicted_facts": []}
    assert strong.calls == 0
    assert (stats.invalid_calls, stats.retried) == (1, 0)   # still detected and counted


@pytest.mark.parametrize(("detect", "same_pair", "warned"), [
    (True, False, True),     # scan runs, suppression discards every result
    (True, True, False),     # both on: cross-pair invalidation actually live
    (False, False, False),   # the defaults: no scan, nothing to discard
    (False, True, False),
])
async def test_warns_when_the_invalidation_scan_runs_but_is_discarded(
        stubs, caplog, detect, same_pair, warned):
    """I1: ingest_same_pair_contradictions=False clears the WHOLE contradicted_facts
    list, cross-pair indices included, so detection on + suppression on pays for the
    O(corpus) scan and invalidates nothing. That combination must be loud."""
    caplog.set_level("WARNING", logger=cli.logger.name)
    await cli._build_ingest_driver(_settings(
        ingest_detect_contradictions=detect,
        ingest_same_pair_contradictions=same_pair))
    hits = [r for r in caplog.records
            if r.name == cli.logger.name and r.levelname == "WARNING"
            and "results are discarded" in r.getMessage()]
    assert len(hits) == (1 if warned else 0)
