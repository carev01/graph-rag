from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def run_worker_once(store, ingest, batch: int) -> int:
    jobs = await store.claim_semantic_jobs(batch, True)
    for job in jobs:
        try:
            if job["op"] == "upsert":
                await ingest.ingest_article(job["article_id"])
            elif job["op"] == "remove":
                await ingest.tombstone_article_episodes(job["article_id"])
            else:
                raise ValueError(f"unknown op {job['op']!r}")
            await store.complete_semantic_job(job["id"])
        except Exception as e:  # a poison job must not block the queue
            logger.exception("semantic job %s failed", job["id"])
            await store.fail_semantic_job(job["id"], str(e))
    return len(jobs)


async def run_worker(store, ingest, *, batch: int, poll_seconds: float,
                     stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        n = await run_worker_once(store, ingest, batch)
        if n == 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
            except asyncio.TimeoutError:
                pass
