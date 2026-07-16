"""LLM-free unit test for the navigation-page skip guard in IngestDriver.

Stubs docext (via a monkeypatched content_fetch.fetch_article) and graphiti
so ingest_article() never touches a real DB or LLM: the guard must return
before chunking/extraction is attempted.
"""
from types import SimpleNamespace

import pytest

from graph_extract import content_fetch
from graph_extract.ingest_driver import IngestDriver

pytestmark = pytest.mark.asyncio


class _StubGraphiti:
    """Records whether add_episode-style extraction was ever invoked."""

    def __init__(self) -> None:
        self.add_episode_calls: list[tuple] = []

    async def add_episode(self, *args, **kwargs):
        self.add_episode_calls.append((args, kwargs))
        raise AssertionError("add_episode should never be called for a navigation page")


async def test_ingest_article_skips_navigation_page(monkeypatch):
    async def fake_fetch_article(client, article_id):
        return SimpleNamespace(
            id=article_id,
            title="Archived release notes",
            source_url="https://example.test/archived-release-notes",
            content_markdown="# Archived release notes\n\nsome content",
            last_updated_at=None,
            extracted_at="2026-07-12T16:00:17Z",
            images=[],
        )

    monkeypatch.setattr(content_fetch, "fetch_article", fake_fetch_article)

    graphiti = _StubGraphiti()
    driver = IngestDriver(
        settings=object(),
        graphiti=graphiti,
        docext=object(),
        provenance=object(),
        driver=object(),
    )

    res = await driver.ingest_article("some-id")

    assert res.skipped_navigation is True
    assert res.episodes_added == 0
    assert graphiti.add_episode_calls == []
