"""The edge verifier hook: dedup_guard filters every extract_edges.edge reply through
fact_verifier.EdgeVerifier when one is installed (cheap tier only, EDGE_VERIFIER_ENABLED),
and changes nothing when none is (spec 2026-10-29-verifier-trial-design.md section 4)."""
from __future__ import annotations

import pytest
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.prompts.models import Message

from graph_extract import cli, fact_verifier
from graph_extract.config import ExtractSettings
from graph_extract.dedup_guard import CURRENT_DEDUP_STATS, DedupIndexStats, install_dedup_guard
from tests.unit.test_cli_dedup_wiring import _G, _settings, stubs  # noqa: F401  (fixture)

pytestmark = pytest.mark.asyncio
EDGES = {"edges": [{"source_entity_name": "A", "target_entity_name": "B", "fact": "x"},
                   {"source_entity_name": "C", "target_entity_name": "D", "fact": "y"}]}
AUTHORED = "<CURRENT_MESSAGE>\nchunk\n</CURRENT_MESSAGE>"


class _MutatingLLM:
    """Like graphiti's clients: appends schema text to the messages IN PLACE."""

    def __init__(self, reply) -> None:
        self.reply, self.calls = reply, 0

    async def generate_response(self, messages, *args, **kwargs):
        self.calls += 1
        messages[-1].content += "\n\nRespond with a JSON object in the following format: ..."
        return self.reply


class _FakeVerifier:
    def __init__(self, drop_last: bool = True) -> None:
        self.seen: list[list[str]] = []
        self.drop_last = drop_last

    async def check(self, messages, response, stats):
        self.seen.append([m.content for m in messages])
        stats.verifier_calls += 1
        if self.drop_last:
            stats.verifier_dropped += 1
            return {**response, "edges": response["edges"][:-1]}
        return response


async def _edge_call(graphiti, name: str = "extract_edges.edge"):
    msgs = [Message(role="system", content="sys"), Message(role="user", content=AUTHORED)]
    return await graphiti.llm_client.generate_response(
        msgs, response_model=None, model_size=ModelSize.small, prompt_name=name)


async def _scoped(coro):
    stats = DedupIndexStats()
    token = CURRENT_DEDUP_STATS.set(stats)
    try:
        return await coro, stats
    finally:
        CURRENT_DEDUP_STATS.reset(token)


async def test_no_verifier_changes_nothing() -> None:
    g = _G(_MutatingLLM(EDGES))
    install_dedup_guard(g, fallback=None, unscoped=DedupIndexStats())
    out, stats = await _scoped(_edge_call(g))
    assert out == EDGES and stats.verifier_calls == 0


async def test_verifier_filters_edges_from_the_authored_prompt() -> None:
    g, v = _G(_MutatingLLM(EDGES)), _FakeVerifier()
    install_dedup_guard(g, fallback=None, unscoped=DedupIndexStats(), edge_verifier=v)
    out, stats = await _scoped(_edge_call(g))
    assert out == {"edges": EDGES["edges"][:1]}
    assert (stats.verifier_calls, stats.verifier_dropped) == (1, 1)
    assert v.seen == [["sys", AUTHORED]]          # not the schema-appended copy


async def test_verifier_only_sees_extract_edges() -> None:
    g, v = _G(_MutatingLLM({"anything": 1})), _FakeVerifier()
    install_dedup_guard(g, fallback=None, unscoped=DedupIndexStats(), edge_verifier=v)
    await _scoped(_edge_call(g, "extract_edges.extract_timestamps"))
    assert v.seen == []


async def test_cli_installs_the_verifier_on_the_cheap_tier_only(stubs, monkeypatch) -> None:  # noqa: F811
    strong, cheap = stubs
    strong.reply = cheap.reply = EDGES
    v = _FakeVerifier()
    monkeypatch.setattr(cli, "build_edge_verifier", lambda s: v)
    ingest, *_ = await cli._build_ingest_driver(_settings(
        edge_verifier_enabled=True, edge_verifier_base_url="http://v", edge_verifier_api_key="k"))
    out_cheap, s1 = await _scoped(_edge_call(ingest._cheap.graphiti))
    out_strong, s2 = await _scoped(_edge_call(ingest._strong.graphiti))
    assert len(out_cheap["edges"]) == 1 and s1.verifier_calls == 1
    assert out_strong == EDGES and s2.verifier_calls == 0


async def test_cli_default_installs_no_verifier(stubs, monkeypatch) -> None:  # noqa: F811
    strong, cheap = stubs
    cheap.reply = EDGES
    monkeypatch.setattr(cli, "build_edge_verifier",
                        lambda s: pytest.fail("built a verifier with the flag off"))
    ingest, *_ = await cli._build_ingest_driver(_settings())
    out, stats = await _scoped(_edge_call(ingest._cheap.graphiti))
    assert out == EDGES and stats.verifier_calls == 0


def test_build_edge_verifier_validates_settings() -> None:
    base = dict(_env_file=None, edge_verifier_enabled=True)
    with pytest.raises(ValueError, match="EDGE_VERIFIER_BASE_URL"):
        fact_verifier.build_edge_verifier(ExtractSettings(**base, edge_verifier_api_key="k"))
    with pytest.raises(ValueError, match="EDGE_VERIFIER_API_KEY"):
        fact_verifier.build_edge_verifier(ExtractSettings(**base, edge_verifier_base_url="http://v"))
    v = fact_verifier.build_edge_verifier(ExtractSettings(
        **base, edge_verifier_base_url="http://v/v1", edge_verifier_api_key="k"))
    assert v.model == "verifier" and v.timeout == 120.0


async def test_verifier_without_a_cheap_tier_refuses_to_start(stubs, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(cli, "build_edge_verifier", lambda s: _FakeVerifier())
    with pytest.raises(ValueError, match="cheap tier"):
        await cli._build_ingest_driver(_settings(
            extraction_routing=False, edge_verifier_enabled=True,
            edge_verifier_base_url="http://v", edge_verifier_api_key="k"))


def test_verifier_counters_reach_the_worker_batch_line() -> None:
    s = DedupIndexStats(verifier_calls=3, verifier_dropped=2, verifier_failed_open=1)
    other = DedupIndexStats(verifier_calls=1)
    s.merge(other)
    assert "verifier_calls=4 verifier_dropped=2 verifier_failed_open=1" in s.summary()
