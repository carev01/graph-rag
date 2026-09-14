from __future__ import annotations

import asyncio
import logging
from collections import Counter

from graph_extract.concurrent_ingest import run_concurrently
from graph_extract.dedup_guard import DedupIndexStats
from graph_extract.ingest_driver import CRAWL_FALLBACK
from graph_extract.usage import get_tally

logger = logging.getLogger(__name__)


def exp_backoff(attempts: int, *, base: float, cap: float) -> float:
    return min(base * (2 ** (attempts - 1)), cap)


async def run_worker_once(
    store, ingest, *, batch: int, budget: int, max_attempts: int,
    backoff_base: float, backoff_cap: float, lease: float,
    concurrency: int = 1,
) -> int:
    # Everything the ingest driver measures is per-article and was being DISCARDED
    # here: `ingest_article` returns dedup counters, the reference-time basis and
    # (on the driver) per-prompt timings, and the production worker is the path
    # that actually runs at scale. The CLI printed them; this did not, so the
    # measurements built to ride on a real ingest were invisible exactly where it
    # matters. Accumulated and logged per batch below.
    batch_dedup = DedupIndexStats()
    basis_counts: Counter[str] = Counter()
    await store.reap_stale_jobs(lease, max_attempts)
    include_bootstrap = await store.today_token_total() < budget
    jobs = await store.claim_semantic_jobs(batch, include_bootstrap)

    async def _run_job(job) -> None:
        before = get_tally()
        t0 = before.prompt_tokens + before.completion_tokens
        try:
            if job["op"] == "upsert":
                res = await ingest.ingest_article(job["article_id"])
                # Observability must never fail the work it observes. A driver
                # double, or a future refactor returning None, must not raise in
                # here -- that would fail the job and poison the queue for the
                # sake of a counter. Explicit check, not a blanket try/except.
                if res is not None:
                    batch_dedup.merge(res.dedup)
                    basis_counts[res.reference_basis] += 1
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

    async def _run_group(group) -> None:
        for job in group:
            await _run_job(job)

    # Grouped by article_id, NOT fanned out over raw jobs: claim_semantic_jobs has
    # no DISTINCT on article_id (state_store.py:145-147), so one batch can hold an
    # upsert and a later remove for the same article, and applying those out of
    # order would tombstone episodes the upsert just created.
    groups: dict[str, list] = {}
    for job in jobs:
        groups.setdefault(job["article_id"], []).append(job)
    results = await run_concurrently(
        list(groups.values()), _run_group, limit=concurrency)
    # `_run_job` catches Exception, so anything that reaches a result slot has
    # escaped the per-job handler: a CancelledError, or an exception from the
    # failure path itself (fail_semantic_job / record_tokens -- i.e. Postgres
    # down). The sequential loop propagated those out of here; discarding the
    # slots would make a database outage vanish with no log and no failed job.
    # Same contract as ingest_source: every one is logged naming its article,
    # the first propagates.
    escaped: list[BaseException] = []
    for article_id, r in zip(groups, results):
        if isinstance(r, BaseException):
            escaped.append(r)
            logger.error("semantic jobs for article %s escaped the job handler",
                         article_id, exc_info=r)
    if escaped:
        raise escaped[0]
    if jobs:
        logger.info("semantic batch: jobs=%d reference_basis=%s dedup: %s",
                    len(jobs), dict(basis_counts), batch_dedup.summary())
        fallbacks = basis_counts.get(CRAWL_FALLBACK, 0)
        if fallbacks:
            logger.warning(
                "%d of %d articles had no usable content_changed_at and were ordered "
                "by crawl time -- their facts cannot be trusted for temporal ordering",
                fallbacks, len(jobs))
        timings = getattr(ingest, "timings", None)
        if timings is not None and timings.by_prompt:
            logger.info("%s", timings.report())
    return len(jobs)


async def run_worker(
    store, ingest, *, batch: int, poll_seconds: float, stop_event: asyncio.Event,
    budget: int, max_attempts: int, backoff_base: float, backoff_cap: float,
    lease: float, max_batches: int | None = None, concurrency: int = 1,
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
            backoff_base=backoff_base, backoff_cap=backoff_cap, lease=lease,
            concurrency=concurrency)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            return
        if n == 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
            except asyncio.TimeoutError:
                pass
