from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Awaitable, Callable

from graph_extract.concurrent_ingest import run_concurrently
from graph_extract.dedup_guard import DedupIndexStats
from graph_extract.ingest_driver import CRAWL_FALLBACK
from graph_extract.usage import CURRENT_USAGE_TALLY, UsageTally

logger = logging.getLogger(__name__)


def exp_backoff(attempts: int, *, base: float, cap: float) -> float:
    return min(base * (2 ** (attempts - 1)), cap)


async def run_worker_once(
    store, ingest, *, batch: int, budget: int, max_attempts: int,
    backoff_base: float, backoff_cap: float, lease: float,
    concurrency: int = 1,
    is_cold: Callable[[str], Awaitable[bool]] | None = None,
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
        # The job's spend is read from a scope opened for THIS job, not as a
        # before/after delta on the module-global tally: with N articles in
        # flight a global delta includes every sibling's tokens (two overlapping
        # 100-token jobs would record 400), and record_tokens feeds
        # today_token_total, which decides whether the bootstrap lane runs at
        # all. Inflated numbers fail that gate closed. The scope wraps the whole
        # try/except/finally so a job that fails still records what it spent.
        spent = UsageTally()
        scope = CURRENT_USAGE_TALLY.set(spent)
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
            CURRENT_USAGE_TALLY.reset(scope)
            delta = spent.prompt_tokens + spent.completion_tokens
            if delta:
                await store.record_tokens(delta)

    # `_run_job` catches Exception, so what escapes a group is the failure path
    # ITSELF raising -- fail_semantic_job / record_tokens, i.e. Postgres down --
    # or a CancelledError. That is not a bad article, it is broken
    # infrastructure, and carrying on ingests every remaining article at full
    # LLM cost with nowhere to record completion: all stay in_progress, all get
    # reaped and re-run. The sequential loop aborted the batch on the spot; this
    # flag reproduces that for the groups not yet started, while groups already
    # in flight finish. Skipped jobs stay in_progress for reap_stale_jobs, the
    # same outcome the old abort produced. The flag is read and written on the
    # event loop with no await between the check and the set, so it cannot
    # race.
    escaped_early = False

    async def _run_group(group) -> None:
        nonlocal escaped_early
        if escaped_early:
            return
        try:
            for job in group:
                await _run_job(job)
        except BaseException:
            escaped_early = True
            raise

    # Grouped by article_id, NOT fanned out over raw jobs. Today the queue cannot
    # hand us two jobs for one article: `ux_semantic_jobs_pending` is UNIQUE on
    # (article_id) WHERE status='pending' (state_store.py:28-29), enqueue
    # collapses an upsert-then-remove into one row via ON CONFLICT ... SET
    # op=excluded.op (:129-136), and claim_semantic_jobs selects only
    # status='pending' (:138-149). The grouping is defence in depth that does not
    # depend on that index staying: if it ever went, running an upsert and a
    # later remove for the same article out of order would tombstone episodes
    # the upsert just created.
    groups: dict[str, list] = {}
    for job in jobs:
        groups.setdefault(job["article_id"], []).append(job)
    # The warm-up barrier (concurrent_ingest / warmup). `is_cold` is per
    # ARTICLE -- WarmupGate.is_cold_article resolves the article to its source
    # -- and a group is one article, so it is asked once per group. An
    # Exception from the predicate (Neo4j unreachable for the lookup) lands in
    # that group's slot like an escape: logged naming the article, its jobs
    # left in_progress for the reaper, the first one re-raised after the batch.
    group_is_cold: Callable[[list], Awaitable[bool]] | None = None
    if is_cold is not None:
        async def _group_is_cold(group: list) -> bool:
            # Tombstoning extracts nothing, so a remove-only group cannot race
            # over an entity and never needs the barrier.
            if all(j["op"] == "remove" for j in group):
                return False
            return await is_cold(group[0]["article_id"])
        group_is_cold = _group_is_cold
    results = await run_concurrently(
        list(groups.values()), _run_group, limit=concurrency, is_cold=group_is_cold)
    # Anything in a result slot escaped the per-job handler (see the flag
    # above). The sequential loop propagated those out of here; discarding the
    # slots would make a database outage vanish with no log and no failed job.
    # Same contract as ingest_source: every one is logged naming its article,
    # the first propagates.
    escaped: list[BaseException] = []
    for article_id, r in zip(groups, results):
        if isinstance(r, BaseException):
            escaped.append(r)
            if isinstance(r, asyncio.CancelledError):
                # Only a CancelledError a worker raised ITSELF lands here: an
                # outer cancellation cancels the gather, which re-raises
                # without returning slots, so this loop never runs for a real
                # shutdown. WARNING because a cancellation is a stop, not a
                # fault -- there is no defect for a traceback to point at.
                logger.warning("semantic jobs for article %s were cancelled",
                               article_id)
            else:
                logger.error("semantic jobs for article %s escaped the job handler",
                             article_id, exc_info=r)
    # The batch summary below comes BEFORE the escaped exception propagates:
    # every group that ran has finished, and raising first would drop the
    # dedup/basis summary and the timing report for the whole batch.
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
    if escaped:
        raise escaped[0]
    return len(jobs)


async def run_worker(
    store, ingest, *, batch: int, poll_seconds: float, stop_event: asyncio.Event,
    budget: int, max_attempts: int, backoff_base: float, backoff_cap: float,
    lease: float, max_batches: int | None = None, concurrency: int = 1,
    is_cold: Callable[[str], Awaitable[bool]] | None = None,
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
            concurrency=concurrency, is_cold=is_cold)
        batches += 1
        if max_batches is not None and batches >= max_batches:
            return
        if n == 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
            except asyncio.TimeoutError:
                pass
