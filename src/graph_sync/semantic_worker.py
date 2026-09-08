from __future__ import annotations

import asyncio
import logging

from graph_extract.usage import get_tally

logger = logging.getLogger(__name__)


def exp_backoff(attempts: int, *, base: float, cap: float) -> float:
    return min(base * (2 ** (attempts - 1)), cap)


async def run_worker_once(
    store, ingest, *, batch: int, budget: int, max_attempts: int,
    backoff_base: float, backoff_cap: float, lease: float,
) -> int:
    await store.reap_stale_jobs(lease, max_attempts)
    include_bootstrap = await store.today_token_total() < budget
    jobs = await store.claim_semantic_jobs(batch, include_bootstrap)
    for job in jobs:
        before = get_tally()
        t0 = before.prompt_tokens + before.completion_tokens
        try:
            if job["op"] == "upsert":
                await ingest.ingest_article(job["article_id"])
            elif job["op"] == "remove":
                await ingest.tombstone_article_episodes(job["article_id"])
            else:
                raise ValueError(f"unknown op {job['op']!r}")
            await store.complete_semantic_job(job["id"], job["claimed_at"])
        except Exception as e:  # a poison job must not block the queue
            logger.exception("semantic job %s failed", job["id"])
            await store.fail_semantic_job(
                job["id"], str(e), max_attempts=max_attempts,
                retry_delay_seconds=exp_backoff(
                    job["attempts"] + 1, base=backoff_base, cap=backoff_cap),
                claimed_at=job["claimed_at"])
        finally:
            after = get_tally()
            delta = (after.prompt_tokens + after.completion_tokens) - t0
            if delta:
                await store.record_tokens(delta)
    return len(jobs)


async def run_worker(
    store, ingest, *, batch: int, poll_seconds: float, stop_event: asyncio.Event,
    budget: int, max_attempts: int, backoff_base: float, backoff_cap: float,
    lease: float, max_batches: int | None = None,
) -> None:
    """Drain `semantic_jobs` until stopped.

    `max_batches` bounds the run to N iterations and then returns normally. Without
    it the only way out is a signal -- and `timeout ... uv run ...` does NOT forward
    SIGTERM through to the wrapped interpreter, so a supposedly time-boxed worker
    keeps running. That bit us for real: an unbounded run overshot its window,
    looped again, and claimed a live bootstrap-lane job that was never intended to
    be processed. Operators and tests need a stop condition that does not depend on
    OS signal plumbing.
    """
    batches = 0
    while not stop_event.is_set():
        n = await run_worker_once(
            store, ingest, batch=batch, budget=budget, max_attempts=max_attempts,
            backoff_base=backoff_base, backoff_cap=backoff_cap, lease=lease)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            return
        if n == 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
            except asyncio.TimeoutError:
                pass
