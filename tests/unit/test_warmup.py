"""The warm-up predicate is derived from the GRAPH, not from a run counter.

That is what makes warmth persist across runs, processes and days for free: a
source ingested over three days warms once, and a quarterly re-extraction of an
already-ingested source never warms at all -- correctly, since its hub entities
already exist.
"""
from __future__ import annotations

import pytest

from graph_extract.warmup import WarmupGate


class _FakeResult:
    def __init__(self, records):
        self.records = records


class _FakeDriver:
    """Records every query so the caching claims can be asserted on."""

    def __init__(self, live_counts: dict[str, int], sources: dict[str, str] | None = None):
        self.live_counts = live_counts
        self.sources = sources or {}
        self.queries: list[tuple[str, dict]] = []

    async def execute_query(self, query, **kwargs):
        self.queries.append((query, kwargs))
        if "a.source_id AS source_id" in query:
            sid = self.sources.get(kwargs["article_id"])
            return _FakeResult([{"source_id": sid}] if sid is not None else [])
        return _FakeResult([{"live": self.live_counts.get(kwargs["source_id"], 0)}])


async def test_a_source_below_the_threshold_is_cold():
    gate = WarmupGate(_FakeDriver({"s1": 3}), "backup-docs", 8)
    assert await gate.is_cold_source("s1") is True


async def test_a_source_at_the_threshold_is_warm():
    gate = WarmupGate(_FakeDriver({"s1": 8}), "backup-docs", 8)
    assert await gate.is_cold_source("s1") is False


async def test_threshold_zero_never_touches_the_driver():
    driver = _FakeDriver({"s1": 0})
    gate = WarmupGate(driver, "backup-docs", 0)
    assert await gate.is_cold_source("s1") is False
    assert driver.queries == [], "threshold 0 is off; it must not query"


async def test_a_warm_source_is_cached_and_a_cold_one_is_not():
    """Warmth is monotonic within a run (nothing removes live episodes mid-run),
    so a warm answer can be cached. A cold source must be re-queried -- it is
    about to become warm, and that is the whole point."""
    driver = _FakeDriver({"warm": 8, "cold": 1})
    gate = WarmupGate(driver, "backup-docs", 8)
    assert await gate.is_cold_source("warm") is False
    assert await gate.is_cold_source("warm") is False
    assert len(driver.queries) == 1, "a warm source must be queried once"
    driver.queries.clear()
    assert await gate.is_cold_source("cold") is True
    assert await gate.is_cold_source("cold") is True
    assert len(driver.queries) == 2, "a cold source must be re-queried each time"


async def test_a_missing_article_is_cold():
    """sync_core.apply enqueues the semantic job BEFORE apply_structural writes
    the Article node, so a worker can claim a job in that window. Cold is the
    conservative branch: treating it as warm would disable the protection
    exactly when the source is newest."""
    gate = WarmupGate(_FakeDriver({}, sources={}), "backup-docs", 8)
    assert await gate.is_cold_article("unknown-article") is True


async def test_an_article_resolves_through_its_source():
    driver = _FakeDriver({"s1": 2}, sources={"a1": "s1"})
    gate = WarmupGate(driver, "backup-docs", 8)
    assert await gate.is_cold_article("a1") is True


async def test_a_driver_failure_propagates():
    """The caller (run_concurrently) puts it in the item's slot. Swallowing it
    here would silently disable warm-up on a Neo4j blip."""
    class _Boom:
        async def execute_query(self, *a, **k):
            raise RuntimeError("neo4j down")

    gate = WarmupGate(_Boom(), "backup-docs", 8)
    with pytest.raises(RuntimeError, match="neo4j down"):
        await gate.is_cold_source("s1")
