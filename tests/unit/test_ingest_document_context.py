"""extraction-grounding (2026-10-17): every `add_text_episode` call for an
article carries that article's real vendor/product/document set, read once
from the structural chain, so extraction is told the subject instead of being
invited to invent one. With no structural chain, ingest logs a WARNING and
passes an empty context rather than failing the article.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from graph_extract import content_fetch, ingest_driver
from graph_extract.config import ExtractSettings
from graph_extract.graphiti_client import ExtractionTier, document_context
from graph_extract.ingest_driver import IngestDriver

pytestmark = pytest.mark.asyncio


class _Prov:
    async def already_ingested(self, *a, **k):
        return False

    async def link(self, *a, **k):
        return None


class _FakeResult:
    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record


class _FakeSession:
    def __init__(self, record):
        self._record = record

    async def run(self, query, **params):
        return _FakeResult(self._record)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeNeo4jDriver:
    """Returns `record` (a dict, or None for 'no structural chain') to every
    `session().run(...)` call -- exercises the REAL `_document_context` query,
    unlike the other tests here which stub `_document_context` directly."""

    def __init__(self, record):
        self._record = record

    def session(self):
        return _FakeSession(self._record)


def _settings() -> ExtractSettings:
    return ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                           neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


def _driver(monkeypatch, *, chain=None, episode_count=2, neo4j_driver=None):
    async def fake_fetch_article(client, article_id):
        return SimpleNamespace(id=article_id, title=f"Article {article_id}",
                               source_url="https://example.test/a", content_markdown="body",
                               last_updated_at=None, extracted_at="2026-07-12T16:00:17Z",
                               content_changed_at="2026-08-01T14:03:11Z",
                               content_changed_basis="exact", images=[])

    async def fake_chunk(*a, **k):
        return []

    def fake_build_episodes(**k):
        return [SimpleNamespace(name=f"n{i}", body="b", chunk_index=i, heading_path="",
                                token_count=1, content_hash="h")
                for i in range(episode_count)]

    calls: list[dict] = []

    async def fake_add_text_episode(graphiti, s, **k):
        calls.append(k)
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
    if chain is not None:
        async def fake_document_context(self, article_id):
            return chain
        monkeypatch.setattr(IngestDriver, "_document_context", fake_document_context)

    tier = ExtractionTier("strong", object(), "I", 1000)
    d = IngestDriver(settings=_settings(), strong_tier=tier, cheap_tier=None,
                     docext=object(), provenance=_Prov(),
                     driver=neo4j_driver if neo4j_driver is not None else object())
    return d, calls


async def test_every_episode_gets_the_articles_document_context(monkeypatch):
    chain = ("Veeam", "Veeam Backup & Replication", "VBR User Guide")
    driver, calls = _driver(monkeypatch, chain=chain, episode_count=2)

    await driver.ingest_article("a1")

    expected = document_context(*chain)
    assert expected != ""
    assert len(calls) == 2
    assert all(c["context"] == expected for c in calls)


async def test_missing_chain_warns_and_passes_empty_context(monkeypatch, caplog):
    # Exercises the REAL `_document_context` Cypher path (no record ->
    # ("", "", "") + WARNING), unlike the other tests here which stub
    # `_document_context` to isolate the per-episode wiring.
    driver, calls = _driver(monkeypatch, episode_count=1,
                            neo4j_driver=_FakeNeo4jDriver(record=None))

    with caplog.at_level(logging.WARNING, logger="graph_extract.ingest_driver"):
        await driver.ingest_article("a1")

    assert len(calls) == 1
    assert calls[0]["context"] == ""
    recs = [r for r in caplog.records
            if r.name == "graph_extract.ingest_driver" and "a1" in r.getMessage()
            and "structural chain" in r.getMessage()]
    assert len(recs) == 1


async def test_single_chain_returns_it_without_a_warning(caplog):
    # Exercises the REAL `_document_context` query directly -- chain_count=1,
    # no multiplicity WARNING.
    driver = IngestDriver(settings=_settings(), strong_tier=ExtractionTier("strong", object(), "I", 1000),
                          cheap_tier=None, docext=object(), provenance=_Prov(),
                          driver=_FakeNeo4jDriver(record={
                              "vendor": "Veeam", "product": "Veeam Backup & Replication",
                              "source": "VBR User Guide", "chain_count": 1}))
    with caplog.at_level(logging.WARNING, logger="graph_extract.ingest_driver"):
        chain = await driver._document_context("a1")
    assert chain == ("Veeam", "Veeam Backup & Replication", "VBR User Guide")
    assert [r for r in caplog.records if r.name == "graph_extract.ingest_driver"] == []


async def test_multiple_chains_logs_a_warning_and_uses_the_chosen_one(caplog):
    # chain_count=2 -> WARNING naming the article and the count; the query
    # itself (ORDER BY source id LIMIT 1) already picked the lowest-id chain,
    # so _document_context just has to report the multiplicity and use what
    # the query returned.
    driver = IngestDriver(settings=_settings(), strong_tier=ExtractionTier("strong", object(), "I", 1000),
                          cheap_tier=None, docext=object(), provenance=_Prov(),
                          driver=_FakeNeo4jDriver(record={
                              "vendor": "Veeam", "product": "Veeam Backup & Replication",
                              "source": "VBR User Guide (lowest id)", "chain_count": 2}))
    with caplog.at_level(logging.WARNING, logger="graph_extract.ingest_driver"):
        chain = await driver._document_context("a1")
    assert chain == ("Veeam", "Veeam Backup & Replication", "VBR User Guide (lowest id)")
    recs = [r for r in caplog.records
            if r.name == "graph_extract.ingest_driver" and "a1" in r.getMessage()
            and "2" in r.getMessage()]
    assert len(recs) == 1


async def test_document_context_is_computed_once_per_article(monkeypatch):
    chain = ("AWS", "AWS Backup", "AWS Backup Developer Guide")
    driver, calls = _driver(monkeypatch, chain=chain, episode_count=3)

    lookups = []
    orig = IngestDriver._document_context

    async def counting(self, article_id):
        lookups.append(article_id)
        return await orig(self, article_id)

    monkeypatch.setattr(IngestDriver, "_document_context", counting)

    await driver.ingest_article("a1")

    assert lookups == ["a1"]
    assert len(calls) == 3
