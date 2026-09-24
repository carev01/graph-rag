"""Unit tests for search_local filtering by invalid_at."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from answer_api.search import search_local


class FakeEdge:
    """Stub edge object with configurable invalid_at."""

    def __init__(self, uuid, fact, invalid_at=None, episodes=None):
        self.uuid = uuid
        self.fact = fact
        self.invalid_at = invalid_at
        self.episodes = episodes or []


class FakeProvenance:
    """Stub provenance resolver."""

    def __init__(self):
        self.driver = None

    async def resolve_citations(self, uuids):
        return {uuid: {"sources": [{"article_id": "a1", "source_url": "https://x/a1"}]}
                for uuid in uuids}


class FakeGraphiti:
    """Stub graphiti client."""

    def __init__(self, edges):
        self.edges = edges

    async def _search(self, q, config, group_ids=None, center_node_uuid=None):
        return MagicMock(edges=self.edges)


class FakeDriver:
    """Stub Neo4j driver."""

    async def session(self):
        return MagicMock()


@pytest.mark.asyncio
async def test_future_invalid_at_is_returned_with_include_invalid_false(monkeypatch):
    """A fact with a future end date is returned even with include_invalid=False."""
    # Arrange
    edges = [FakeEdge("f1", "future fact", datetime(2028, 9, 1, tzinfo=timezone.utc))]
    graphiti = FakeGraphiti(edges)
    driver = FakeDriver()
    monkeypatch.setattr("answer_api.search.Provenance", lambda d: FakeProvenance())

    # Act
    result = await search_local(graphiti, driver, q="test", k=10, include_invalid=False,
                                group_id="test-group")

    # Assert
    assert result["count"] == 1
    assert result["results"][0]["fact_uuid"] == "f1"
    assert result["results"][0]["invalid_at"] is not None


@pytest.mark.asyncio
async def test_past_invalid_at_is_filtered_with_include_invalid_false(monkeypatch):
    """A fact with a past end date is filtered out with include_invalid=False."""
    # Arrange
    edges = [FakeEdge("f1", "past fact", datetime(2026, 3, 31, tzinfo=timezone.utc))]
    graphiti = FakeGraphiti(edges)
    driver = FakeDriver()
    monkeypatch.setattr("answer_api.search.Provenance", lambda d: FakeProvenance())

    # Act
    result = await search_local(graphiti, driver, q="test", k=10, include_invalid=False,
                                group_id="test-group")

    # Assert
    assert result["count"] == 0
    assert result["results"] == []
