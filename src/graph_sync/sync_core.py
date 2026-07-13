from __future__ import annotations
import logging
from dataclasses import dataclass, field
import httpx
from graph_sync.catalog import Catalog
from graph_sync.config import Settings
from graph_sync.delta_client import DeltaStream, build_delta_params
from graph_sync.mapper import map_content, map_tombstone
from graph_sync.models import ContentRecord, TombstoneRecord, min_watermark
from graph_sync.neo4j_repo import Neo4jRepo
from graph_sync.state_store import StateStore
from graph_sync.toc import fetch_toc
from graph_sync.toc_mapper import map_toc

log = logging.getLogger("graph_sync.sync")


@dataclass
class BootstrapResult:
    applied: int = 0
    skipped: int = 0
    sources: set[str] = field(default_factory=set)


@dataclass
class IncrementalResult:
    applied: int = 0
    removed: int = 0
    skipped: int = 0
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

    async def _apply_record(
        self, rec: ContentRecord | TombstoneRecord, res: BootstrapResult | IncrementalResult
    ) -> str | None:
        if isinstance(rec, TombstoneRecord):
            await self._repo.tombstone_article(map_tombstone(rec))
            # BootstrapResult has no `removed` counter (bootstrap streams never emit
            # tombstones); only IncrementalResult tracks it.
            if isinstance(res, IncrementalResult):
                res.removed += 1
            return rec.source_id

        existing = await self._repo.get_content_hash(rec.id)
        if existing == rec.content_hash:
            res.skipped += 1
            return None

        info = self._catalog.resolve(rec.source_id)
        if info is None:
            await self._catalog.refresh()
            info = self._catalog.resolve(rec.source_id)
        if info is None:
            await self._store.record_dead_letter(
                rec.model_dump_json(), f"unknown source {rec.source_id}"
            )
            return None

        await self._repo.apply_structural(map_content(rec, info))
        res.applied += 1
        return rec.source_id

    async def refresh_toc(self, source_id: str) -> None:
        try:
            toc = await fetch_toc(self._client, source_id)
            await self._repo.apply_toc(map_toc(toc))
        except Exception:  # decoupled: TOC failures never block article sync
            log.exception("TOC refresh failed for source %s", source_id)

    async def bootstrap(self, *, source_id: str | None = None,
                        vendor_id: str | None = None) -> BootstrapResult:
        res = BootstrapResult()
        bootstrap_after: str | None = None
        watermark: str | None = None
        shard = source_id or vendor_id or "global"
        while True:
            params = build_delta_params(source_id=source_id, vendor_id=vendor_id,
                                        bootstrap_after=bootstrap_after)
            stream = DeltaStream(self._client, params)
            last_id: str | None = None
            async for rec in stream.records():
                touched = await self._apply_record(rec, res)
                if touched:
                    res.sources.add(touched)
                if isinstance(rec, ContentRecord):
                    last_id = rec.id
            if watermark is None:
                # Keep the ORIGINAL first-attempt watermark across resumes; a
                # truncated stream's own bootstrap_start is not recomputed.
                watermark = stream.bootstrap_start_since
            await self._store.upsert_bootstrap(
                shard, watermark, last_id,
                "complete" if stream.terminated_clean else "in_progress")
            if stream.terminated_clean:
                break
            bootstrap_after = last_id  # resume from where the stream dropped
        for sid in res.sources:
            await self.refresh_toc(sid)
        marks = await self._store.all_bootstrap_watermarks()
        if marks:
            await self._store.set_cursor(min_watermark(marks))
        return res

    async def run_incremental(self) -> IncrementalResult:
        res = IncrementalResult()
        if not await self._store.try_lock():
            return res  # another sync is already running: single-flight
        try:
            cursor = await self._store.get_cursor()
            stream = DeltaStream(self._client, build_delta_params(since=cursor))
            async for rec in stream.records():
                touched = await self._apply_record(rec, res)
                if touched:
                    res.sources.add(touched)
            if stream.terminated_clean and stream.next_since:
                await self._store.set_cursor(stream.next_since)
                res.advanced = True
                for sid in res.sources:
                    await self.refresh_toc(sid)
            return res
        finally:
            await self._store.unlock()
