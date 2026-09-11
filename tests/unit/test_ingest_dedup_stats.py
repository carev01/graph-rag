"""IngestDriver surfaces the dedup-index guard's counters per article.

The guard records into whatever `CURRENT_DEDUP_STATS` holds; the driver must set a
fresh `DedupIndexStats` for each article's scope, attach it to the result, and
aggregate across `ingest_source`. Stubbed like test_ingest_skip: no DB, no LLM.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from graph_extract import content_fetch, ingest_driver
from graph_extract.config import ExtractSettings
from graph_extract.dedup_guard import CURRENT_DEDUP_STATS
from graph_extract.graphiti_client import ExtractionTier
from graph_extract.ingest_driver import IngestDriver

pytestmark = pytest.mark.asyncio


class _Prov:
    async def already_ingested(self, *a, **k):
        return False

    async def link(self, *a, **k):
        return None


def _settings() -> ExtractSettings:
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


@pytest.fixture
def driver(monkeypatch):
    """A driver whose add_text_episode simulates the guard: it bumps the CURRENT
    scope's counters (as install_dedup_guard does inside add_episode)."""
    async def fake_fetch_article(client, article_id):
        return SimpleNamespace(id=article_id, title=f"Article {article_id}",
                               source_url="https://example.test/a", content_markdown="body",
                               last_updated_at=None, extracted_at="2026-07-12T16:00:17Z",
                               images=[])

    async def fake_chunk(*a, **k):
        return []

    def fake_build_episodes(**k):
        return [SimpleNamespace(name="n", body="b", chunk_index=0, heading_path="",
                                token_count=1, content_hash="h")]

    async def fake_add_text_episode(graphiti, s, **k):
        stats = CURRENT_DEDUP_STATS.get()
        assert stats is not None, "the driver must open a dedup-stats scope per article"
        stats.calls += 1
        stats.invalid_calls += 1
        stats.dup_in_invalidation_range += 2
        return SimpleNamespace(episode=SimpleNamespace(uuid="e"), nodes=[], edges=[])

    async def noop(self, *a, **k):
        return None

    monkeypatch.setattr(content_fetch, "fetch_article", fake_fetch_article)
    monkeypatch.setattr(ingest_driver.chonkie_client, "neural_chunk", fake_chunk)
    monkeypatch.setattr(ingest_driver.episode_builder, "build_episodes", fake_build_episodes)
    monkeypatch.setattr(ingest_driver, "add_text_episode", fake_add_text_episode)
    monkeypatch.setattr(IngestDriver, "_content_hash", noop)
    monkeypatch.setattr(IngestDriver, "_chapter_path", noop)
    monkeypatch.setattr(IngestDriver, "_supersede_trailing_episodes", noop)

    tier = ExtractionTier("strong", object(), "I", 1000)
    d = IngestDriver(settings=_settings(), strong_tier=tier, cheap_tier=None,
                     docext=object(), provenance=_Prov(), driver=object())

    async def ids(self, source_id):
        return ["a1", "a2"]

    monkeypatch.setattr(IngestDriver, "list_article_ids", ids)
    return d


async def test_each_article_gets_its_own_dedup_stats(driver):
    r1 = await driver.ingest_article("a1")
    r2 = await driver.ingest_article("a2")
    assert (r1.dedup.calls, r1.dedup.invalid_calls, r1.dedup.dup_in_invalidation_range) == (1, 1, 2)
    assert (r2.dedup.calls, r2.dedup.invalid_calls) == (1, 1)
    assert r1.dedup is not r2.dedup
    assert CURRENT_DEDUP_STATS.get() is None, "the scope must be closed after the article"


async def test_the_scope_is_closed_even_when_extraction_raises(driver, monkeypatch):
    async def boom(graphiti, s, **k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(ingest_driver, "add_text_episode", boom)
    with pytest.raises(RuntimeError):
        await driver.ingest_article("a1")
    assert CURRENT_DEDUP_STATS.get() is None


async def test_ingest_source_aggregates_dedup_stats(driver):
    out = await driver.ingest_source("src")
    assert out.articles == 2
    assert (out.dedup.calls, out.dedup.invalid_calls, out.dedup.dup_in_invalidation_range) == (2, 2, 4)


async def test_an_article_with_invalid_indices_is_logged_with_its_counts(driver, caplog):
    with caplog.at_level(logging.WARNING, logger="graph_extract.ingest_driver"):
        await driver.ingest_article("a1")
    recs = [r for r in caplog.records if r.name == "graph_extract.ingest_driver"]
    assert len(recs) == 1
    assert "a1" in recs[0].getMessage() and "invalid_calls=1" in recs[0].getMessage()


async def test_a_clean_article_is_not_logged(driver, monkeypatch, caplog):
    async def clean(graphiti, s, **k):
        CURRENT_DEDUP_STATS.get().calls += 1
        return SimpleNamespace(episode=SimpleNamespace(uuid="e"), nodes=[], edges=[])

    monkeypatch.setattr(ingest_driver, "add_text_episode", clean)
    with caplog.at_level(logging.WARNING, logger="graph_extract.ingest_driver"):
        res = await driver.ingest_article("a1")
    assert res.dedup.calls == 1 and res.dedup.invalid_calls == 0
    assert [r for r in caplog.records if r.name == "graph_extract.ingest_driver"] == []
