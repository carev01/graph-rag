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
from graph_extract.grounding import CURRENT_GROUNDING
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
                               # Post-2026-09-13 upstream contract. Without it the
                               # driver correctly warns about falling back to crawl
                               # ordering, which is a second record in caplog.
                               content_changed_at="2026-08-01T14:03:11Z",
                               content_changed_basis="exact",
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

    async def no_chain(self, *a, **k):
        return "", "", ""

    monkeypatch.setattr(content_fetch, "fetch_article", fake_fetch_article)
    monkeypatch.setattr(ingest_driver.chonkie_client, "neural_chunk", fake_chunk)
    monkeypatch.setattr(ingest_driver.episode_builder, "build_episodes", fake_build_episodes)
    monkeypatch.setattr(ingest_driver, "add_text_episode", fake_add_text_episode)
    monkeypatch.setattr(IngestDriver, "_content_hash", noop)
    monkeypatch.setattr(IngestDriver, "_chapter_path", noop)
    monkeypatch.setattr(IngestDriver, "_supersede_trailing_episodes", noop)
    monkeypatch.setattr(IngestDriver, "_document_context", no_chain)

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


# ------------------------------------------------- grounding context (2026-10-17)

def _two_episodes(**k):
    return [SimpleNamespace(name=f"n{i}", body=f"body {i}", chunk_index=i, heading_path="",
                            token_count=1, content_hash=f"h{i}") for i in range(2)]


async def test_each_episode_runs_inside_its_own_grounding_context(driver, monkeypatch):
    seen = []

    async def chain(self, *a, **k):
        return "Veeam", "Veeam Backup & Replication", "User Guide"

    async def record(graphiti, s, **k):
        ctx = CURRENT_GROUNDING.get()
        assert ctx is not None, "the driver must open a grounding context per episode"
        seen.append((ctx.text, ctx.allowed, k["body"]))
        ctx.stats.calls += 1
        ctx.stats.dropped += 2
        ctx.stats.dropped_names.extend(["Hyper-V", "East US"])
        return SimpleNamespace(episode=SimpleNamespace(uuid="e"), nodes=[], edges=[])

    monkeypatch.setattr(IngestDriver, "_document_context", chain)
    monkeypatch.setattr(ingest_driver.episode_builder, "build_episodes", _two_episodes)
    monkeypatch.setattr(ingest_driver, "add_text_episode", record)
    res = await driver.ingest_article("a1")
    allowed = ("Veeam Backup & Replication", "Veeam")
    assert seen == [("body 0", allowed, "body 0"), ("body 1", allowed, "body 1")]
    assert (res.grounding.calls, res.grounding.dropped) == (2, 4)
    assert CURRENT_GROUNDING.get() is None, "the context must be reset after each episode"


async def test_empty_document_names_are_not_allowed_names(driver, monkeypatch):
    seen = []

    async def record(graphiti, s, **k):
        seen.append(CURRENT_GROUNDING.get().allowed)
        return SimpleNamespace(episode=SimpleNamespace(uuid="e"), nodes=[], edges=[])

    monkeypatch.setattr(ingest_driver, "add_text_episode", record)
    await driver.ingest_article("a1")     # the fixture's chain is ("", "", "")
    assert seen == [()]


async def test_the_grounding_context_is_reset_when_extraction_raises(driver, monkeypatch):
    async def boom(graphiti, s, **k):
        raise RuntimeError("llm down")

    monkeypatch.setattr(ingest_driver, "add_text_episode", boom)
    with pytest.raises(RuntimeError):
        await driver.ingest_article("a1")
    assert CURRENT_GROUNDING.get() is None


async def test_dropped_names_are_logged_at_info_per_article(driver, monkeypatch, caplog):
    async def drop(graphiti, s, **k):
        st = CURRENT_GROUNDING.get().stats
        st.dropped += 25
        st.dropped_names.extend(f"E{i}" for i in range(25))
        return SimpleNamespace(episode=SimpleNamespace(uuid="e"), nodes=[], edges=[])

    monkeypatch.setattr(ingest_driver, "add_text_episode", drop)
    with caplog.at_level(logging.INFO, logger="graph_extract.ingest_driver"):
        await driver.ingest_article("a1")
    msgs = [r.getMessage() for r in caplog.records
            if r.name == "graph_extract.ingest_driver" and "grounding dropped" in r.getMessage()]
    assert len(msgs) == 1
    assert msgs[0].startswith("article a1: grounding dropped 25 entities: ")
    assert "E19" in msgs[0] and "E20" not in msgs[0]


async def test_ingest_source_aggregates_grounding_stats(driver, monkeypatch):
    async def drop(graphiti, s, **k):
        CURRENT_GROUNDING.get().stats.dropped += 3
        return SimpleNamespace(episode=SimpleNamespace(uuid="e"), nodes=[], edges=[])

    monkeypatch.setattr(ingest_driver, "add_text_episode", drop)
    out = await driver.ingest_source("src")
    assert out.grounding.dropped == 6
