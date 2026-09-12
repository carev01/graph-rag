from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
import httpx
from neo4j import AsyncDriver
from graph_extract.article_filter import is_navigation_article
from graph_extract.article_router import is_dense_matrix
from graph_extract.config import ExtractSettings
from graph_extract import content_fetch, chonkie_client, episode_builder
from graph_extract.dedup_guard import CURRENT_DEDUP_STATS, DedupIndexStats
from graph_extract.llm_timing import PromptTimings
from graph_extract.graphiti_client import add_text_episode, ExtractionTier
from graph_extract.provenance import Provenance

logger = logging.getLogger(__name__)


@dataclass
class IngestArticleResult:
    article_id: str
    episodes_added: int = 0
    episodes_skipped: int = 0
    entities: int = 0
    edges: int = 0
    skipped_navigation: bool = False
    tier: str = "strong"
    # Out-of-range dedup indices seen while extracting THIS article (dedup_guard).
    dedup: DedupIndexStats = field(default_factory=DedupIndexStats)


@dataclass
class IngestResult:
    articles: int = 0
    episodes_added: int = 0
    episodes_skipped: int = 0
    dedup: DedupIndexStats = field(default_factory=DedupIndexStats)


def _parse_ts(art) -> datetime:
    raw = art.last_updated_at or art.extracted_at
    if raw:
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.now(timezone.utc)


class IngestDriver:
    def __init__(self, settings: ExtractSettings, strong_tier: ExtractionTier,
                 cheap_tier: ExtractionTier | None, docext: httpx.AsyncClient,
                 provenance: Provenance, driver: AsyncDriver):
        self._s = settings
        self._strong = strong_tier
        self._cheap = cheap_tier
        self._docext = docext
        self._prov = provenance
        self._driver = driver
        # Set by the CLI to the PromptTimings shared across both tiers' guards, so
        # `ingest` can report where an episode's wall time went (BACKLOG 31).
        self.timings = PromptTimings()

    def _tier_for(self, markdown: str) -> ExtractionTier:
        if self._cheap is None:
            return self._strong
        dense = is_dense_matrix(
            markdown, ratio_threshold=self._s.dense_table_line_ratio,
            pipe_threshold=self._s.dense_pipe_count)
        return self._strong if dense else self._cheap

    async def list_article_ids(self, source_id: str) -> list[str]:
        async with self._driver.session() as s:
            r = await s.run(
                "MATCH (a:Article {source_id:$s}) WHERE coalesce(a.removed,false)=false "
                "RETURN a.id AS id ORDER BY a.sort_order", s=source_id)
            return [rec["id"] async for rec in r]

    async def ingest_article(self, article_id: str) -> IngestArticleResult:
        res = IngestArticleResult(article_id=article_id)
        # Open the per-article scope the dedup guard records into. Reset in
        # `finally` so a failed article never leaks its scope into the next one.
        token = CURRENT_DEDUP_STATS.set(res.dedup)
        try:
            return await self._ingest_article(article_id, res)
        finally:
            CURRENT_DEDUP_STATS.reset(token)
            if res.dedup.invalid_calls:
                logger.warning("article %s (%s tier): out-of-range dedup indices -- %s",
                               article_id, res.tier, res.dedup.summary())

    async def _ingest_article(self, article_id: str, res: IngestArticleResult) -> IngestArticleResult:
        art = await content_fetch.fetch_article(self._docext, article_id)
        if is_navigation_article(art.title or ""):
            res.skipped_navigation = True
            return res      # navigation page: no chunking, no extraction, 0 tokens
        tier = self._tier_for(art.content_markdown)
        res.tier = tier.name
        ref = _parse_ts(art)
        # content_hash: reuse the article's stored hash from the graph, else hash markdown
        content_hash = await self._content_hash(article_id) or _sha(art.content_markdown)
        chapter_path = await self._chapter_path(article_id)
        async with httpx.AsyncClient(base_url=self._s.chonkie_base_url, timeout=120) as ch:
            chunks = await chonkie_client.neural_chunk(ch, art.content_markdown, self._s.chonkie_model)
        episodes = episode_builder.build_episodes(
            article_id=art.id, title=art.title, chapter_path=chapter_path,
            content_hash=content_hash, chunks=chunks,
            max_chunk_tokens=tier.max_chunk_tokens, min_chunk_tokens=self._s.min_chunk_tokens)
        for e in episodes:
            if await self._prov.already_ingested(art.id, e.chunk_index, e.content_hash):
                res.episodes_skipped += 1
                continue
            r = await add_text_episode(tier.graphiti, self._s, name=e.name, body=e.body,
                                       source_description=art.source_url, reference_time=ref,
                                       instructions=tier.instructions)
            await self._prov.link(art.id, r.episode.uuid, chunk_index=e.chunk_index,
                                  heading_path=e.heading_path, token_count=e.token_count,
                                  content_hash=e.content_hash)
            res.episodes_added += 1
            res.entities += len(r.nodes)
            res.edges += len(r.edges)
        await self._supersede_trailing_episodes(art.id, len(episodes))
        return res

    async def tombstone_article_episodes(self, article_id: str) -> int:
        async with self._driver.session() as s:
            r = await s.run(
                "MATCH (:Article {id:$a})-[:HAS_EPISODE]->(e:Episodic) "
                "SET e.removed=true RETURN count(e) AS c", a=article_id)
            rec = await r.single()
            return rec["c"] if rec else 0

    async def _supersede_trailing_episodes(self, article_id: str, new_count: int) -> None:
        """A shrunk article drops trailing chunks. Flag the EDGE (not just the node)
        so the staleness sweep sees them as dead -- flagging only the node is what
        previously let facts from deleted content stay current forever."""
        async with self._driver.session() as s:
            await s.run(
                "MATCH (:Article {id:$a})-[r:HAS_EPISODE]->(e:Episodic) "
                "WHERE r.chunk_index >= $n "
                "SET r.superseded=true", a=article_id, n=new_count)

    async def ingest_source(self, source_id: str, limit: int | None = None) -> IngestResult:
        ids = await self.list_article_ids(source_id)
        if limit is not None:
            ids = ids[:limit]
        out = IngestResult()
        for aid in ids:
            r = await self.ingest_article(aid)
            out.articles += 1
            out.episodes_added += r.episodes_added
            out.episodes_skipped += r.episodes_skipped
            out.dedup.merge(r.dedup)
        return out

    async def _content_hash(self, article_id: str) -> str | None:
        async with self._driver.session() as s:
            r = await s.run("MATCH (a:Article {id:$a}) RETURN a.content_hash AS h", a=article_id)
            rec = await r.single()
            return rec["h"] if rec else None

    async def _chapter_path(self, article_id: str) -> str:
        async with self._driver.session() as s:
            r = await s.run("MATCH (a:Article {id:$a})-[:IN_CHAPTER]->(c:Chapter) "
                            "RETURN c.title AS t", a=article_id)
            rec = await r.single()
            return rec["t"] if rec and rec["t"] else ""


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()
