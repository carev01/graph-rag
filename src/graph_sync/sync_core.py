from __future__ import annotations
import asyncio
import logging
from dataclasses import dataclass, field
import httpx
from graph_sync.catalog import Catalog
from graph_sync.config import Settings
from graph_sync.delta_client import DeltaStream, build_delta_params
from graph_sync.mapper import map_content, map_tombstone
from graph_sync.models import (
    ContentRecord, TombstoneRecord, decode_cursor_seq, min_watermark,
)
from graph_sync.neo4j_repo import Neo4jRepo
from graph_sync.state_store import StateStore
from graph_sync.toc import fetch_toc
from graph_sync.toc_mapper import map_toc

log = logging.getLogger("graph_sync.sync")


@dataclass
class BootstrapResult:
    applied: int = 0
    skipped: int = 0
    requeued: int = 0
    sources: set[str] = field(default_factory=set)


@dataclass
class IncrementalResult:
    applied: int = 0
    removed: int = 0
    skipped: int = 0
    requeued: int = 0
    advanced: bool = False
    sources: set[str] = field(default_factory=set)


class SyncCore:
    def __init__(self, client: httpx.AsyncClient, catalog: Catalog,
                 repo: Neo4jRepo, store: StateStore, settings: Settings) -> None:
        self._client = client
        self._catalog = catalog
        self._repo = repo
        self._store = store
        self._settings = settings
        # In-process single-flight guard. The Postgres advisory lock alone is
        # not enough: the poll loop and the webhook both drive run_incremental
        # on this same instance/connection, and pg_try_advisory_lock is
        # reentrant per session (a second call on the same connection returns
        # True again) while two concurrent queries on the shared lock
        # connection raise InterfaceError. This flag is flipped synchronously
        # (no await between check and set), so under asyncio it admits exactly
        # one in-flight run_incremental; the advisory lock remains for
        # cross-process safety.
        self._incremental_running = False
        # Set when a nudge arrives mid-sync so the running sync does one more
        # pass (spec §5.3/§7) instead of dropping the trigger.
        self._incremental_dirty = False

    async def _lane(self, res: BootstrapResult | IncrementalResult, article_id: str) -> str:
        """CLAUDE.md ("incremental updates preempt bootstrap backfill"): the
        unbudgeted `incremental` lane exists ONLY for updates to articles that
        have already been semantically extracted. An incremental pull also
        carries records for articles never ingested before -- brand-new
        articles, or a changed/removed article from a vendor never
        bootstrapped -- and those must land in the budgeted `bootstrap` lane
        like any other backfill, or they bypass `claim_semantic_jobs`'s
        token-budget gate entirely (`AND ($2 OR lane='incremental')`). This is
        what queued 5,646 unbudgeted jobs on the cluster's first incremental
        pull (see BACKLOG). Applies to tombstones too: a remove for an article
        with no episodes never extracted anything, so it belongs in
        `bootstrap`.

        Called AFTER the hash gate for content records (an unchanged replay
        never reaches this) so a no-op replay never pays the `has_episodes`
        Neo4j round trip; tombstones have no hash gate, so they call it
        directly.
        """
        if isinstance(res, BootstrapResult):
            return "bootstrap"
        return "incremental" if await self._repo.has_episodes(article_id) else "bootstrap"

    async def _requeue_unextracted(self, rec: ContentRecord) -> bool:
        """Re-queue an unchanged article that was never extracted (BACKLOG 50).

        The hash gate skips an unchanged record before anything is enqueued, which
        is right only while Postgres and Neo4j agree. After a state reset, a lost
        `semantic_jobs` row, or one store restored without the other, every article
        already written structurally would be skipped forever -- no job, no
        episodes, no error. A plain re-run of `bootstrap --source-id` now repairs it.

        Any job row, in ANY status, means the article is accounted for: `done` with
        no episodes is legitimate (navigation pages, zero-chunk articles complete
        that way), and a `dead` job stays dead -- retrying those is a separate
        decision. Only with no row at all is Neo4j asked; if the article has
        episodes the graph is ahead of Postgres and redoing it would only cost.
        The common skip path therefore pays one indexed Postgres read and no Neo4j
        round trip.

        The lane is `bootstrap` by construction: the lane rule (`_lane`) sends an
        article without episodes there from either stream.
        """
        if await self._store.has_semantic_job(rec.id):
            return False
        if await self._repo.has_episodes(rec.id):
            return False
        # Never an overwrite: a row that appeared since the check wins.
        if not await self._store.enqueue_semantic_job_if_absent(
                rec.id, "upsert", rec.content_hash, rec.source_id):
            return False
        # WARNING, not INFO: firing at all means Postgres and Neo4j disagreed.
        log.warning("re-queued unextracted article %s (no job row, no episodes)", rec.id)
        return True

    async def _apply_record(
        self, rec: ContentRecord | TombstoneRecord, res: BootstrapResult | IncrementalResult
    ) -> str | None:
        if isinstance(rec, TombstoneRecord):
            lane = await self._lane(res, rec.id)
            await self._store.enqueue_semantic_job(rec.id, "remove", None, lane, rec.source_id)
            await self._repo.tombstone_article(map_tombstone(rec))
            # BootstrapResult has no `removed` counter (bootstrap streams never emit
            # tombstones); only IncrementalResult tracks it.
            if isinstance(res, IncrementalResult):
                res.removed += 1
            return rec.source_id

        existing = await self._repo.get_content_hash(rec.id)
        if existing == rec.content_hash:
            if await self._requeue_unextracted(rec):
                res.requeued += 1
            else:
                res.skipped += 1
            # Nothing structural changed either way: not a touched source.
            return None
        lane = await self._lane(res, rec.id)

        info = self._catalog.resolve(rec.source_id)
        if info is None:
            await self._catalog.refresh()
            info = self._catalog.resolve(rec.source_id)
        if info is None:
            # Spec §7: never DROP an unknown-source record (and never let the
            # batch advance the cursor past it as a silent loss). Persist a
            # minimal Article node flagged catalog_incomplete instead of dead-
            # lettering: it is then hash-gated on retry, and a later catalog
            # refresh + reconciliation can complete its vendor/product/source
            # wiring. A subsequent apply_structural fills the rest via
            # `SET a += $article` on the same id.
            await self._store.enqueue_semantic_job(
                rec.id, "upsert", rec.content_hash, lane, rec.source_id)
            await self._repo.apply_incomplete_article({
                "id": rec.id, "source_id": rec.source_id, "title": rec.title,
                "source_url": rec.source_url, "topic_key": rec.topic_key,
                "content_hash": rec.content_hash,
                "estimated_tokens": rec.estimated_tokens,
                "sort_order": rec.sort_order, "last_updated_at": rec.last_updated_at,
                "run_id": rec.run_id, "seq": rec.seq, "removed": False,
                "catalog_incomplete": True,
            })
            res.applied += 1
            return rec.source_id

        await self._store.enqueue_semantic_job(
            rec.id, "upsert", rec.content_hash, lane, rec.source_id)
        await self._repo.apply_structural(map_content(rec, info))
        res.applied += 1
        return rec.source_id

    async def refresh_toc(self, source_id: str) -> None:
        try:
            toc = await fetch_toc(self._client, source_id)
            await self._repo.apply_toc(map_toc(toc))
        except Exception:  # decoupled: TOC failures never block article sync
            log.exception("TOC refresh failed for source %s", source_id)

    # Dropped-stream handling (CLIENT-USAGE-GUIDE §6; BACKLOG 49). Class attributes
    # rather than settings so tests can tighten them per instance.
    BOOTSTRAP_MAX_STALLS = 5          # consecutive dropped attempts with no new record
    BOOTSTRAP_BACKOFF_SECONDS = 5.0   # x attempt number, capped below
    BOOTSTRAP_BACKOFF_CAP_SECONDS = 60.0
    BOOTSTRAP_PROGRESS_EVERY = 200    # applied records between in-stream progress saves
    _sleep = staticmethod(asyncio.sleep)

    async def bootstrap(self, *, source_id: str | None = None,
                        vendor_id: str | None = None) -> BootstrapResult:
        """Stream a shard's bootstrap, resuming a dropped stream from the highest
        applied id with the ORIGINAL watermark (CLIENT-USAGE-GUIDE §6).

        A dropped stream is one that ends without the terminal cursor line --
        cleanly truncated, cut by a transport error (a ReadTimeout when
        DocExtractor goes silent mid-stream; BACKLOG 49), or refused with a 5xx
        or 429. Any other 4xx fails immediately. Either way it resumes
        with `bootstrap_after=<last applied id>`; only BOOTSTRAP_MAX_STALLS
        consecutive attempts that apply nothing new give up, re-raising.

        `last_id` means "every id up to here is applied" because the stream is
        consumed in order and each record is applied before it is counted
        (CLAUDE.md, sync correctness). Progress is saved every
        BOOTSTRAP_PROGRESS_EVERY records as well as at the end of each attempt, so
        a killed process loses at most that many; the next run of an
        `in_progress` shard resumes from it. A `complete` shard is replayed from
        the start: re-running one is a deliberate repair (BACKLOG 50).
        """
        res = BootstrapResult()
        shard = source_id or vendor_id or "global"
        bootstrap_after: str | None = None
        watermark: str | None = None
        prior = await self._store.get_bootstrap(shard)
        if prior and prior.get("status") == "in_progress" and prior.get("last_id"):
            bootstrap_after, watermark = prior["last_id"], prior.get("watermark")
            log.info("bootstrap %s: resuming in-progress shard after %s", shard, bootstrap_after)
        stalls = 0
        while True:
            params = build_delta_params(source_id=source_id, vendor_id=vendor_id,
                                        bootstrap_after=bootstrap_after)
            stream = DeltaStream(self._client, params)
            last_id: str | None = None
            since_save = 0
            error: httpx.HTTPError | None = None
            try:
                async for rec in stream.records():
                    if watermark is None:
                        # Keep the ORIGINAL first-attempt watermark across resumes;
                        # a resumed stream's own bootstrap_start is not adopted.
                        watermark = stream.bootstrap_start_since
                    touched = await self._apply_record(rec, res)
                    if touched:
                        res.sources.add(touched)
                    if isinstance(rec, ContentRecord):
                        last_id = rec.id
                        since_save += 1
                        if since_save >= self.BOOTSTRAP_PROGRESS_EVERY:
                            await self._store.upsert_bootstrap(
                                shard, watermark, last_id, "in_progress")
                            since_save = 0
            except httpx.TransportError as exc:
                error = exc
            except httpx.HTTPStatusError as exc:
                # raise_for_status() is not a TransportError. A 5xx or 429 from the
                # ingress is transient -- a dropped stream; any other 4xx (bad key,
                # unknown id) will not fix itself and fails now.
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise
                error = exc
            if watermark is None:
                watermark = stream.bootstrap_start_since
            resume_from = last_id or bootstrap_after
            await self._store.upsert_bootstrap(
                shard, watermark, resume_from,
                "complete" if stream.terminated_clean else "in_progress")
            if stream.terminated_clean:
                break
            stalls = 0 if last_id else stalls + 1
            if stalls >= self.BOOTSTRAP_MAX_STALLS:
                log.error("bootstrap %s: %d consecutive dropped attempts with no progress "
                          "after %s; giving up (progress saved -- rerun resumes)",
                          shard, stalls, resume_from)
                if error is not None:
                    raise error
                raise RuntimeError(f"bootstrap {shard}: stream keeps ending without its "
                                   f"terminal cursor line after {resume_from}")
            delay = min(self.BOOTSTRAP_BACKOFF_SECONDS * max(stalls, 1),
                        self.BOOTSTRAP_BACKOFF_CAP_SECONDS)
            log.warning("bootstrap %s: stream dropped (%s) after %s; resuming in %.0fs",
                        shard, type(error).__name__ if error else "no terminal line",
                        resume_from, delay)
            await self._sleep(delay)
            bootstrap_after = resume_from  # resume from where the stream dropped
        for sid in res.sources:
            await self.refresh_toc(sid)
        marks = await self._store.all_bootstrap_watermarks()
        if marks:
            candidate = min_watermark(marks)
            # Only advance, never regress. bootstrap_progress rows are never
            # cleared, so a later repair-bootstrap of one shard would otherwise
            # reset the live incremental cursor back to that shard's original
            # (lower) watermark and force a huge redundant replay.
            current = await self._store.get_cursor()
            if current is None or decode_cursor_seq(candidate) > decode_cursor_seq(current):
                await self._store.set_cursor(candidate)
        return res

    async def run_incremental(self) -> IncrementalResult:
        # In-process single-flight: checked-and-set synchronously (no await
        # between), so a second concurrent caller — e.g. a webhook firing mid
        # poll-cycle — does not run a second overlapping sync. Instead it marks
        # the run dirty so the active sync does one more pass (never drops a
        # nudge; spec §5.3/§7).
        if self._incremental_running:
            self._incremental_dirty = True
            return IncrementalResult()
        self._incremental_running = True
        try:
            res = IncrementalResult()
            while True:
                self._incremental_dirty = False
                r = await self._run_incremental_once()
                # Accumulate across passes so the returned counts reflect ALL
                # work, not just the last (dirty re-pass) pass.
                res.applied += r.applied
                res.removed += r.removed
                res.skipped += r.skipped
                res.sources |= r.sources
                res.advanced = res.advanced or r.advanced
                if not self._incremental_dirty:
                    return res
        finally:
            self._incremental_running = False

    async def _run_incremental_once(self) -> IncrementalResult:
        res = IncrementalResult()
        if not await self._store.try_lock():
            return res  # another process is already running: single-flight
        try:
            cursor = await self._store.get_cursor()
            stream = DeltaStream(self._client, build_delta_params(since=cursor))
            async for rec in stream.records():
                touched = await self._apply_record(rec, res)
                if touched:
                    res.sources.add(touched)
            # Unparseable lines are dead-lettered and BLOCK cursor advance
            # (do not skip past a dropped line; spec §7).
            for bad in stream.malformed:
                await self._store.record_dead_letter(bad, "unparseable delta line")
            if stream.terminated_clean and stream.next_since and not stream.malformed:
                await self._store.set_cursor(stream.next_since)
                res.advanced = True
                for sid in res.sources:
                    await self.refresh_toc(sid)
            return res
        finally:
            await self._store.unlock()
