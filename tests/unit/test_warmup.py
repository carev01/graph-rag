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


_SOURCE_LOOKUP = "a.source_id AS source_id"


class _FakeDriver:
    """Records every query so the caching claims can be asserted on.

    `live_counts` is keyed by whatever the count query is given as `$source_id`,
    so a gate that skips the article->source hop and counts under the ARTICLE id
    is observable: give the article id and its source id different counts.

    `groups`, when given, replaces `live_counts` with per-group counts and makes
    the fake honour the query's scope the way Neo4j would: a query whose text
    binds `$group_id` sees only that group, and one that does not sees every
    group's episodes. Asserting on the `group_id` kwarg alone cannot tell those
    apart -- a MATCH with the filter dropped still receives the kwarg.
    """

    def __init__(
        self,
        live_counts: dict[str, int],
        sources: dict[str, str] | None = None,
        *,
        groups: dict[str, dict[str, int]] | None = None,
    ):
        self.live_counts = live_counts
        self.sources = sources or {}
        self.groups = groups
        self.queries: list[tuple[str, dict]] = []

    def _live(self, query: str, kwargs: dict) -> int:
        sid = kwargs["source_id"]
        if self.groups is None:
            return self.live_counts.get(sid, 0)
        if "$group_id" not in query:
            return sum(counts.get(sid, 0) for counts in self.groups.values())
        return self.groups.get(kwargs.get("group_id"), {}).get(sid, 0)

    async def execute_query(self, query, **kwargs):
        self.queries.append((query, kwargs))
        if _SOURCE_LOOKUP in query:
            sid = self.sources.get(kwargs["article_id"])
            return _FakeResult([{"source_id": sid}] if sid is not None else [])
        return _FakeResult([{"live": self._live(query, kwargs)}])

    def count_queries(self) -> list[tuple[str, dict]]:
        return [(q, kw) for q, kw in self.queries if _SOURCE_LOOKUP not in q]


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


async def test_the_count_is_scoped_to_the_gate_group():
    """The A/B runner ingests into an isolated group so it never touches the
    baseline. A count that ignored the group would see the baseline's episodes,
    call the fresh group warm, and skip the warm-up on the one run that needs
    it. The fixture puts 8 live articles in the OTHER group and none in the
    gate's, so an unscoped or wrongly scoped count reports warm."""
    driver = _FakeDriver({}, groups={"ab-isolated": {"s1": 0}, "backup-docs": {"s1": 8}})
    gate = WarmupGate(driver, "ab-isolated", 8)
    assert await gate.is_cold_source("s1") is True
    counts = driver.count_queries()
    assert len(counts) == 1
    query, kwargs = counts[0]
    assert kwargs["group_id"] == "ab-isolated", "the gate's group must reach the query"
    assert "$group_id" in query, "the MATCH must bind the group, not ignore or hardcode it"


async def test_a_missing_article_is_cold():
    """sync_core.apply enqueues the semantic job BEFORE apply_structural writes
    the Article node, so a worker can claim a job in that window. Cold is the
    conservative branch: treating it as warm would disable the protection
    exactly when the source is newest.

    The fixture gives the raw article id a warm-looking count so that a gate
    which skips the lookup and counts under the article id answers warm; the
    correct gate never counts at all."""
    driver = _FakeDriver({"unknown-article": 8}, sources={})
    gate = WarmupGate(driver, "backup-docs", 8)
    assert await gate.is_cold_article("unknown-article") is True
    assert driver.count_queries() == [], "a missing article must not be counted, only looked up"


async def test_an_article_resolves_through_its_source():
    """The fixture must make the two-hop answer DIFFER from the no-hop answer:
    s1 is warm (8 >= 8) while the raw article id would resolve to 2 and look
    cold. A fixture where both paths agree pins nothing."""
    driver = _FakeDriver({"s1": 8, "a1": 2}, sources={"a1": "s1"})
    gate = WarmupGate(driver, "backup-docs", 8)
    assert await gate.is_cold_article("a1") is False
    assert any(_SOURCE_LOOKUP in q for q, _ in driver.queries), (
        "the article must be resolved through its source, not used as one")
    assert [kw["source_id"] for _, kw in driver.count_queries()] == ["s1"]


async def test_a_cold_source_is_cold_through_its_article():
    """The mirror image: s1 is genuinely below the threshold while the raw
    article id would look warm, so a no-hop gate answers warm here."""
    driver = _FakeDriver({"s1": 2, "a1": 8}, sources={"a1": "s1"})
    gate = WarmupGate(driver, "backup-docs", 8)
    assert await gate.is_cold_article("a1") is True
    assert [kw["source_id"] for _, kw in driver.count_queries()] == ["s1"]


async def test_threshold_zero_makes_is_cold_article_skip_the_driver():
    """Redundant with the check inside is_cold_source, so losing it would cost
    a round-trip per article rather than a wrong answer -- but 'off' should
    mean off, not 'one Neo4j query per article for nothing'."""
    driver = _FakeDriver({"s1": 0}, sources={"a1": "s1"})
    gate = WarmupGate(driver, "backup-docs", 0)
    assert await gate.is_cold_article("a1") is False
    assert driver.queries == [], "threshold 0 is off; it must not look the article up"


async def test_a_driver_failure_propagates():
    """The caller (run_concurrently) puts it in the item's slot. Swallowing it
    here would silently disable warm-up on a Neo4j blip."""
    class _Boom:
        async def execute_query(self, *a, **k):
            raise RuntimeError("neo4j down")

    gate = WarmupGate(_Boom(), "backup-docs", 8)
    with pytest.raises(RuntimeError, match="neo4j down"):
        await gate.is_cold_source("s1")
