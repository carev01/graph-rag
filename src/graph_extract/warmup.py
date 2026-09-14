"""Is this source still in its warm-up window?

Concurrency duplicates entities when two in-flight articles both extract an
entity that is not yet in the graph. Measured at N=4 on the pilot: 30 duplicates,
every one an exact-name collision, concentrated on hub entities (mean article
span 7.6 against 2.4 for the rest) -- `AWS Backup`, `Amazon EC2`, `Microsoft
Azure`. Those are the nodes cross-vendor questions resolve through, which is what
design invariant #4 exists to protect.

The risk is front-loaded: for an entity in k articles it scales with k-1, and
636 of 999 baseline entities are single-article and carry none. Per source in
`sort_order`, the first 8 articles carry 79.5% of the total risk weight, because
a documentation source opens with overview pages naming the product and its core
concepts.

**Warmth is a property of the graph, not of a run.** That is deliberate: it makes
the answer identical for `ingest_source` (which has an order) and the semantic
worker (which claims batches spanning sources and has no notion of a start), and
it persists across runs, restarts and days with no new state. A source ingested
over three days warms once; a quarterly re-extraction of an already-ingested
source never warms, which is right -- its hubs already exist; a semantic-layer
reset drops the count to zero and it warms again, which is also right.
"""
from __future__ import annotations

from neo4j import AsyncDriver

_LIVE_ARTICLES = (
    "MATCH (a:Article {source_id:$source_id})-[r:HAS_EPISODE]->"
    "(:Episodic {group_id:$group_id}) "
    "WHERE coalesce(r.superseded, false) = false "
    "RETURN count(DISTINCT a) AS live"
)

_ARTICLE_SOURCE = "MATCH (a:Article {id:$article_id}) RETURN a.source_id AS source_id"


class WarmupGate:
    """Per-run cold/warm predicate over sources.

    `Article.source_id` carries a RANGE index (`article_source`), so the count is
    an index seek plus a small expand -- negligible against a ~300 s article.
    """

    def __init__(self, driver: AsyncDriver, group_id: str, threshold: int) -> None:
        self._driver = driver
        self._group_id = group_id
        self._threshold = threshold
        self._warm: set[str] = set()

    async def is_cold_source(self, source_id: str) -> bool:
        if self._threshold <= 0:
            return False
        if source_id in self._warm:
            return False
        r = await self._driver.execute_query(
            _LIVE_ARTICLES, source_id=source_id, group_id=self._group_id)
        live = r.records[0]["live"] if r.records else 0
        if live >= self._threshold:
            # Warmth is monotonic within a run -- nothing removes live episodes
            # mid-run -- so a warm answer is cached for the run's lifetime. A
            # cold source is deliberately NOT cached: it is about to become warm.
            self._warm.add(source_id)
            return False
        return True

    async def is_cold_article(self, article_id: str) -> bool:
        if self._threshold <= 0:
            return False
        r = await self._driver.execute_query(_ARTICLE_SOURCE, article_id=article_id)
        if not r.records or r.records[0]["source_id"] is None:
            # `sync_core.apply` enqueues the semantic job BEFORE it writes the
            # Article node, so a worker can claim a job in that window. Cold is
            # the conservative branch; calling it warm would disable the
            # protection exactly when the source is newest.
            return True
        return await self.is_cold_source(r.records[0]["source_id"])
