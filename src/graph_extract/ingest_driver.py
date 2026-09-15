from __future__ import annotations
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import httpx
from neo4j import AsyncDriver
from graph_extract.article_filter import is_navigation_article
from graph_extract.article_router import is_dense_matrix
from graph_extract.concurrent_ingest import run_concurrently
from graph_extract.config import ExtractSettings
from graph_extract import content_fetch, chonkie_client, episode_builder
from graph_extract.dedup_guard import CURRENT_DEDUP_STATS, DedupIndexStats
from graph_extract.llm_timing import PromptTimings
from graph_extract.graphiti_client import add_text_episode, ExtractionTier
from graph_extract.provenance import Provenance
from graph_extract.warmup import WarmupGate

logger = logging.getLogger(__name__)

# Reference-time basis when upstream supplies no content_changed_at: ordering
# by crawl sequence, which is the pre-2026-09-13 behaviour and a known defect.
CRAWL_FALLBACK = "crawl-fallback"


@dataclass
class IngestArticleResult:
    article_id: str
    episodes_added: int = 0
    episodes_skipped: int = 0
    entities: int = 0
    edges: int = 0
    skipped_navigation: bool = False
    tier: str = "strong"
    # Which timestamp the episodes' reference time came from: the upstream
    # content_changed_basis, or CRAWL_FALLBACK when upstream supplied none.
    reference_basis: str = CRAWL_FALLBACK
    # Out-of-range dedup indices seen while extracting THIS article (dedup_guard).
    dedup: DedupIndexStats = field(default_factory=DedupIndexStats)


@dataclass
class IngestResult:
    articles: int = 0
    episodes_added: int = 0
    episodes_skipped: int = 0
    dedup: DedupIndexStats = field(default_factory=DedupIndexStats)


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _reference_time(art) -> tuple[datetime, str]:
    """The episode's reference time, and the basis of that claim.

    `content_changed_at` is the ordering axis: it is the moment the SERVED markdown
    became current -- the same bytes `content_hash` covers and our re-ingest gate
    keys on -- so a change we act on is always a change we can date. It is the only
    timestamp with uniform semantics across the whole corpus, which is why it is
    preferred over `last_updated_at` even where the vendor declares a real date.
    Mixing the two would put "the vendor's declared day" and "the served bytes
    became current" in one field, switching by vendor: exactly the incoherence that
    produced this project's phantom invalidations.

    `last_updated_at` and `source_changed_at` are persisted elsewhere and used for
    display and for gating invalidation. They are never sorted by.

    The returned basis is `content_changed_basis` (exact / lower_bound / first_seen)
    or `CRAWL_FALLBACK`. The fallback is the pre-2026-09-13 behaviour -- ordering by
    when the crawler happened to visit -- and it is LOUD, because reverting to it
    silently is the original defect: 2,470 articles once shared a single crawl
    minute, so within such a block the ordering is not approximate, it is arbitrary.
    """
    parsed = _parse_iso(getattr(art, "content_changed_at", None))
    if parsed is not None:
        return parsed, (getattr(art, "content_changed_basis", None) or "unlabelled")
    logger.warning(
        "article %s has no usable content_changed_at; falling back to crawl time, "
        "which orders facts by crawl sequence rather than by content change",
        getattr(art, "id", "?"))
    return (_parse_iso(art.last_updated_at) or _parse_iso(art.extracted_at)
            or datetime.now(timezone.utc)), CRAWL_FALLBACK


class IngestDriver:
    def __init__(self, settings: ExtractSettings, strong_tier: ExtractionTier,
                 cheap_tier: ExtractionTier | None, docext: httpx.AsyncClient,
                 provenance: Provenance, driver: AsyncDriver,
                 warmup_gate: WarmupGate | None = None):
        self._s = settings
        self._strong = strong_tier
        self._cheap = cheap_tier
        self._docext = docext
        self._prov = provenance
        self._driver = driver
        # Set by the CLI to the PromptTimings shared across both tiers' guards, so
        # `ingest` can report where an episode's wall time went (BACKLOG 31).
        self.timings = PromptTimings()
        # The warm-up barrier `ingest_source` applies (see concurrent_ingest and
        # warmup for the measured rationale). None = no barrier, which is also
        # what `ingest_warmup_articles=0` builds: the fan-out then takes the
        # byte-for-byte pre-warm-up path rather than a predicate that always
        # answers False.
        self.warmup_gate = warmup_gate

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
        ref, res.reference_basis = _reference_time(art)
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
                                  content_hash=e.content_hash,
                                  superseded_at=ref.isoformat())
            res.episodes_added += 1
            res.entities += len(r.nodes)
            res.edges += len(r.edges)
        await self._supersede_trailing_episodes(art.id, len(episodes), ref.isoformat())
        return res

    async def tombstone_article_episodes(self, article_id: str) -> int:
        async with self._driver.session() as s:
            r = await s.run(
                "MATCH (:Article {id:$a})-[:HAS_EPISODE]->(e:Episodic) "
                "SET e.removed=true RETURN count(e) AS c", a=article_id)
            rec = await r.single()
            return rec["c"] if rec else 0

    async def _supersede_trailing_episodes(self, article_id: str, new_count: int,
                                           superseded_at: str | None = None) -> None:
        """A shrunk article drops trailing chunks. Flag the EDGE (not just the node)
        so the staleness sweep sees them as dead -- flagging only the node is what
        previously let facts from deleted content stay current forever."""
        async with self._driver.session() as s:
            await s.run(
                "MATCH (:Article {id:$a})-[r:HAS_EPISODE]->(e:Episodic) "
                "WHERE r.chunk_index >= $n "
                # coalesce: an episode dies once; a re-run must not move its death date.
                "SET r.superseded=true, "
                "    r.superseded_at=coalesce(r.superseded_at, $sat)",
                a=article_id, n=new_count, sat=superseded_at)

    async def ingest_source(self, source_id: str, limit: int | None = None) -> IngestResult:
        ids = await self.list_article_ids(source_id)
        if limit is not None:
            ids = ids[:limit]
        out = IngestResult()
        is_cold: Callable[[str], Awaitable[bool]] | None = None
        gate = self.warmup_gate
        if gate is not None:
            async def _source_is_cold(_article_id: str) -> bool:
                # Warmth is a property of the SOURCE, so every article of this
                # call asks the same question; the gate caches the warm answer.
                # "Warm" is "at least `threshold` articles have a live episode"
                # as the graph stands when asked -- not "that many finished".
                return await gate.is_cold_source(source_id)
            is_cold = _source_is_cold
        results = await run_concurrently(
            ids, self.ingest_article,
            limit=self._s.ingest_article_concurrency, is_cold=is_cold)
        failures: list[BaseException] = []
        for article_id, r in zip(ids, results):
            if isinstance(r, BaseException):
                failures.append(r)
                logger.error("article %s failed during ingest_source", article_id, exc_info=r)
                continue
            out.articles += 1
            out.episodes_added += r.episodes_added
            out.episodes_skipped += r.episodes_skipped
            out.dedup.merge(r.dedup)
        if failures:
            # Raised AFTER the batch rather than mid-loop. The pre-fan-out code
            # propagated immediately and silently abandoned every article after
            # the failure; this attempts every article and still surfaces the
            # error. Deliberate change, pinned by a test. "Every article" holds
            # because a worker exception lands in its own slot instead of
            # cancelling the batch: with no gate, gather creates all the tasks
            # up front, whatever the limit; with a gate, tasks are created up
            # front only BETWEEN cold boundaries (a cold article drains what is
            # in flight, runs alone, then the fan-out resumes). What does
            # abandon the remaining articles is a BaseException reaching
            # run_concurrently's loop -- an outer cancellation delivered while
            # it awaits a drain, or one escaping the predicate: it propagates
            # out of run_concurrently after every launched task has been
            # cancelled and awaited, and never reaches this loop. A plain
            # Exception from the predicate lands in its article's slot and is
            # handled here like any other failure. Every failure is logged
            # above (naming its article) so a multi-failure batch is fully
            # visible in the logs even though only the first one propagates --
            # raising all of them isn't an option, so this is the compromise
            # that keeps the rest from vanishing silently.
            raise failures[0]
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
