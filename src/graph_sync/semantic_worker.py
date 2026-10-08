from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager, AsyncExitStack

import httpx
import openai

from graph_extract import vector_search
from graph_extract.concurrent_ingest import run_concurrently
from graph_extract.dedup_guard import DedupIndexStats
from graph_extract.ingest_driver import CRAWL_FALLBACK
from graph_extract.usage import CURRENT_USAGE_TALLY, UsageTally
from graph_sync import metrics

logger = logging.getLogger(__name__)


_deferrals = 0


def _note_deferral(article_id: str, n_jobs: int) -> None:
    """INFO on the first deferral (proof the mechanism engaged -- CLAUDE.md, cost
    awareness) and every 100th after, DEBUG otherwise: with 8 workers the busy
    case repeats every few seconds and would drown the batch lines."""
    global _deferrals
    _deferrals += 1
    level = logging.INFO if _deferrals == 1 or _deferrals % 100 == 0 else logging.DEBUG
    logger.log(level, "deferred cold article %s (%d job(s)): warm-up lock busy in another "
               "worker; %d deferral(s) so far in this process", article_id, n_jobs, _deferrals)


class CreditsExhausted(Exception):
    """The LLM account is out of credits (HTTP 402). Raised out of
    `run_worker_once` after the batch's jobs were deferred; `run_worker` stops
    claiming for `credit_pause_seconds`."""


def _is_out_of_credits(exc: BaseException) -> bool:
    """HTTP 402 / insufficient credits anywhere in the exception chain.

    An empty account fails EVERY job the same way, so treating it as a poison job
    spends each job's attempts and marches on through the queue (2026-09-26: 155
    jobs in 15 minutes). graphiti may wrap the provider error, hence the chain walk.
    """
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen and len(seen) < 10:
        seen.add(id(e))
        if getattr(e, "status_code", None) == 402:
            return True
        text = str(e).lower()
        if "error code: 402" in text or ("insufficient" in text and "credit" in text):
            return True
        e = e.__cause__ or e.__context__
    return False


class ProviderUnreachable(Exception):
    """An upstream endpoint -- LLM, embedder or DocExtractor, all httpx-based -- cannot
    be reached. Raised out of `run_worker_once` after the batch's jobs were deferred;
    `run_worker` stops claiming for `unreachable_pause_seconds`. A Neo4j connection
    refused is caught too, through the chain: its driver raises `ServiceUnavailable`
    from the underlying `ConnectionRefusedError` -- also an outage, so also deferred."""


# Failing to CONNECT can never be the article's fault.
_CONNECT_FAILED = (httpx.ConnectError, httpx.ConnectTimeout, ConnectionRefusedError)
# A request dropped mid-flight is ambiguous: an endpoint going down, or a proxy (the SSH
# tunnel, vast.ai's proxy) cutting off one very long request. Settled by a probe.
_DROPPED = (httpx.RemoteProtocolError, httpx.ReadError, httpx.WriteError,
            ConnectionResetError, openai.APIConnectionError)
_TIMEOUTS = (openai.APITimeoutError, httpx.TimeoutException, TimeoutError)


def _chain(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    e: BaseException | None = exc
    while e is not None and e not in out and len(out) < 10:
        out.append(e)
        e = e.__cause__ or e.__context__
    return out


def _connection_failure(exc: BaseException) -> str | None:
    """"connect", "dropped" or None for an exception chain.

    An endpoint that is down fails EVERY job the same way, so treating it as a poison
    job spends each job's attempts and marches on through the queue (2026-10-08: the
    GPU tier's tunnel lost vast.ai's SSH proxy for ~10 minutes; 24 jobs in a few
    minutes). A read/write timeout is excluded -- it can be the article itself (a huge
    prompt) -- and `openai.APITimeoutError` subclasses `APIConnectionError`, so
    timeouts are checked before the dropped-connection types. A CONNECT timeout is an
    endpoint that is not answering, so it counts as "connect" (checked first).
    """
    chain = _chain(exc)
    if any(isinstance(x, _CONNECT_FAILED) for x in chain):
        return "connect"
    if any(isinstance(x, _TIMEOUTS) for x in chain):
        return None
    if any(isinstance(x, _DROPPED) for x in chain):
        return "dropped"
    return None


def _request_url(exc: BaseException) -> httpx.URL | None:
    for x in _chain(exc):
        try:
            req = getattr(x, "request", None)   # httpx raises if it was never set
        except RuntimeError:
            continue
        if isinstance(req, httpx.Request):
            return req.url
    return None


async def endpoint_answers(url: httpx.URL,
                           transport: httpx.AsyncBaseTransport | None = None) -> bool:
    """Is anything answering at `url`'s origin? Any HTTP response counts -- a 401 or a
    404 included -- except 502-504: that is a proxy whose backend is gone.

    `verify=False`: DocExtractor sits behind the private CA, and a probe that failed
    the certificate check would call a live endpoint "down" -- re-creating the
    infinite deferral this probe exists to prevent. Nothing is sent but a bare GET.
    Anything unexpected answers "up", so the job fails normally rather than looping
    or taking the worker down.
    """
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False, transport=transport) as client:
            r = await client.get(url.copy_with(path="/", query=None, fragment=None))
        return r.status_code not in (502, 503, 504)
    except httpx.TransportError:
        return False
    except Exception:  # noqa: BLE001 -- see docstring: unknown means fail the job
        return True


def exp_backoff(attempts: int, *, base: float, cap: float) -> float:
    return min(base * (2 ** (attempts - 1)), cap)


async def run_worker_once(
    store, ingest, *, batch: int, budget: int, max_attempts: int,
    backoff_base: float, backoff_cap: float, lease: float,
    concurrency: int = 1,
    is_cold: Callable[[str], Awaitable[bool]] | None = None,
    cold_lock: Callable[[], AbstractAsyncContextManager[bool]] | None = None,
    source_ids: list[str] | None = None,
    defer_seconds: float | None = None,
    credit_pause_seconds: float = 300.0,
    unreachable_pause_seconds: float = 60.0,
    probe: Callable[[httpx.URL], Awaitable[bool]] = endpoint_answers,
) -> int:
    """Claim and run one batch; returns the number of jobs PROCESSED.

    A job failing with HTTP 402 (the LLM account is out of credits) is DEFERRED by
    `credit_pause_seconds` instead of failed -- no attempt spent -- and so is every
    job of the batch not yet started; `CreditsExhausted` is then raised so
    `run_worker` stops claiming. A job whose upstream endpoint cannot be reached gets
    the same treatment with `unreachable_pause_seconds` and `ProviderUnreachable`:
    always when the connection could not be made, and for a request dropped
    mid-flight only when `probe` finds the endpoint not answering either --
    otherwise the drop was this article's doing and the job fails as before, so it
    can still reach `dead`.

    `defer_seconds` switches the global warm-up lock to defer mode: `cold_lock()`
    is expected to try once (`StateStore.warmup_lock(wait=False)`), and a cold
    article whose lock is held by another process is handed back with
    `defer_semantic_job` (no attempt spent) so this worker can claim warm work
    instead of idling. Deferred jobs do not count as processed: a batch that only
    deferred returns 0, and `run_worker` then sleeps its poll interval before the
    next claim -- which is what keeps an all-cold queue from becoming a busy
    loop. `None` keeps the waiting behaviour (and its give-up-and-run timeout).
    """
    # Everything the ingest driver measures is per-article and was being DISCARDED
    # here: `ingest_article` returns dedup counters, the reference-time basis and
    # (on the driver) per-prompt timings, and the production worker is the path
    # that actually runs at scale. The CLI printed them; this did not, so the
    # measurements built to ride on a real ingest were invisible exactly where it
    # matters. Accumulated and logged per batch below.
    batch_dedup = DedupIndexStats()
    # Summed per-job spend: prompt / cached / completion tokens for the batch line.
    batch_tokens = UsageTally()
    basis_counts: Counter[str] = Counter()
    await store.reap_stale_jobs(lease, max_attempts)
    include_bootstrap = await store.today_token_total() < budget
    # `source_ids` scopes this worker to a subset of sources (the k3s
    # bootstrap-first rehearsal, `SEMANTIC_CLAIM_SOURCE_IDS`); `None`/empty is
    # unscoped -- today's default behaviour.
    jobs = await store.claim_semantic_jobs(batch, include_bootstrap, source_ids)

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
        started = time.monotonic()
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
            metrics.job_outcome("done", time.monotonic() - started)
        except Exception as e:  # a poison job must not block the queue
            halt = "credits" if _is_out_of_credits(e) else None
            if halt is None:
                kind = _connection_failure(e)
                url = _request_url(e) if kind == "dropped" else None
                if kind == "connect" or (url is not None and not await probe(url)):
                    halt = "unreachable"
            if halt is not None:
                # Not the article's fault: no attempt spent, and the batch stops.
                halted.append((halt, f"{type(e).__name__}: {str(e)[:300]}"))
                await _defer_halted(job)
                return
            logger.exception("semantic job %s failed", job["id"])
            await store.fail_semantic_job(
                job["id"], str(e), max_attempts=max_attempts,
                retry_delay_seconds=exp_backoff(
                    job["attempts"] + 1, base=backoff_base, cap=backoff_cap),
                claimed_at=job["claimed_at"])
            metrics.job_outcome("failed", time.monotonic() - started)
        finally:
            CURRENT_USAGE_TALLY.reset(scope)
            metrics.job_tokens(spent.prompt_tokens, spent.cached_tokens,
                               spent.completion_tokens)
            batch_tokens.add("llm", prompt=spent.prompt_tokens,
                             completion=spent.completion_tokens, cached=spent.cached_tokens)
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

    # Articles the predicate called cold, so `_run_group` can hold the global
    # warm-up lock for exactly those. `run_concurrently` evaluates `is_cold`
    # before it launches the item's task, so the verdict is always recorded by
    # the time the task runs.
    cold_ids: set[str] = set()
    deferred: list[int] = []
    not_ours: list[int] = []
    # (kind, message) of the failures that halt the batch: "credits" (HTTP 402) or
    # "unreachable" (the endpoint is down).
    halted: list[tuple[str, str]] = []

    def _halt_kind() -> str:
        # Credits win: an empty account needs a human, and its longer pause.
        return "credits" if any(k == "credits" for k, _ in halted) else "unreachable"

    async def _defer_halted(job) -> None:
        delay = credit_pause_seconds if _halt_kind() == "credits" else unreachable_pause_seconds
        if await store.defer_semantic_job(job["id"], job["claimed_at"], delay):
            deferred.append(job["id"])
            metrics.job_outcome("deferred_halt")
        else:
            not_ours.append(job["id"])

    async def _run_group(group) -> None:
        nonlocal escaped_early
        if escaped_early:
            return
        if halted:
            # The account is empty or the endpoint is down: hand the rest of the
            # batch back unrun rather than send it somewhere that will refuse it.
            for job in group:
                await _defer_halted(job)
            return
        try:
            async with AsyncExitStack() as stack:
                if cold_lock is not None and group[0]["article_id"] in cold_ids:
                    # Serialised across worker PROCESSES, not just within this
                    # one (StateStore.warmup_lock). A failure to take the lock is
                    # Postgres being down, which is the same broken-infrastructure
                    # case `_run_job`'s escape handles: it sets escaped_early and
                    # the batch stops rather than ingesting at full LLM cost with
                    # nowhere to record completion.
                    waited_from = time.monotonic()
                    held = await stack.enter_async_context(cold_lock())
                    if defer_seconds is not None and not held:
                        # A job whose defer returns False was reaped and
                        # reclaimed meanwhile: not ours to count either way.
                        ours = [job for job in group if await store.defer_semantic_job(
                            job["id"], job["claimed_at"], defer_seconds)]
                        lost = len(group) - len(ours)
                        deferred.extend(job["id"] for job in ours)
                        not_ours.extend(job["id"] for job in group if job not in ours)
                        if ours:
                            _note_deferral(group[0]["article_id"], len(ours))
                            for _ in ours:
                                metrics.job_outcome("deferred_lock")
                        if lost:
                            logger.warning("%d job(s) for article %s were reclaimed by "
                                           "another worker before they could be deferred",
                                           lost, group[0]["article_id"])
                        return
                    # The only direct evidence the cross-process barrier fired
                    # (CLAUDE.md, cost awareness); without it, engagement can
                    # only be reconstructed from episode timestamps afterwards.
                    logger.info("warm-up lock held for cold article %s (waited %.1fs)",
                                group[0]["article_id"], time.monotonic() - waited_from)
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
    # -- and a group is one article, so it is asked once per group.
    #
    # An Exception from the predicate (Neo4j unreachable for the lookup) is
    # caught HERE and answered "cold": conservative, i.e. slower, never less
    # safe. The group then runs and, if Neo4j really is down, its job fails
    # through `_run_job`'s per-job backoff exactly as a blip mid-ingest did
    # before the barrier existed. Left to run_concurrently instead, the
    # exception would land in the group's slot: its jobs would never run
    # (left in_progress for the reaper), every OTHER group would still run at
    # full LLM cost -- a predicate failure is not an escape and does not set
    # `escaped_early` -- and the slot loop below would re-raise it after the
    # batch, which `run_worker` does not catch. A mitigation that ships on by
    # default must not turn a one-retry blip into a dead worker process.
    # BaseException (an outer cancellation) is not caught: it must propagate.
    group_is_cold: Callable[[list], Awaitable[bool]] | None = None
    if is_cold is not None:
        async def _group_is_cold(group: list) -> bool:
            # Tombstoning extracts nothing, so a remove-only group cannot race
            # over an entity and never needs the barrier. `all`, not `any`: a
            # mixed upsert+remove group for one article DOES extract.
            if all(j["op"] == "remove" for j in group):
                return False
            article_id = group[0]["article_id"]
            try:
                verdict = await is_cold(article_id)
            except Exception:
                logger.warning(
                    "warm-up lookup for article %s failed; treating it as cold",
                    article_id, exc_info=True)
                verdict = True
            if verdict:
                cold_ids.add(article_id)
            return verdict
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
    processed = len(jobs) - len(deferred) - len(not_ours)
    if processed:
        logger.info("semantic batch: jobs=%d deferred=%d reference_basis=%s dedup: %s",
                    processed, len(deferred), dict(basis_counts), batch_dedup.summary())
        # Phase B: per-batch proof the index path engaged; reset so each batch
        # reports its own counts.
        pt = batch_tokens.prompt_tokens
        logger.info("semantic batch llm tokens: prompt=%d cached=%d (%d%%) completion=%d",
                    pt, batch_tokens.cached_tokens,
                    round(100 * batch_tokens.cached_tokens / pt) if pt else 0,
                    batch_tokens.completion_tokens)
        logger.info("semantic batch vector search: %s", vector_search.stats_summary())
        vector_search.reset_stats()
        fallbacks = basis_counts.get(CRAWL_FALLBACK, 0)
        if fallbacks:
            logger.warning(
                "%d of %d articles had no usable content_changed_at and were ordered "
                "by crawl time -- their facts cannot be trusted for temporal ordering",
                fallbacks, processed)
        timings = getattr(ingest, "timings", None)
        if timings is not None and timings.by_prompt:
            logger.info("%s", timings.report())
    if escaped:
        raise escaped[0]
    if halted:
        kind = _halt_kind()
        message = next(m for k, m in halted if k == kind)
        raise (CreditsExhausted if kind == "credits" else ProviderUnreachable)(message)
    return processed


async def run_worker(
    store, ingest, *, batch: int, poll_seconds: float, stop_event: asyncio.Event,
    budget: int, max_attempts: int, backoff_base: float, backoff_cap: float,
    lease: float, max_batches: int | None = None, concurrency: int = 1,
    cold_lock: Callable[[], AbstractAsyncContextManager[bool]] | None = None,
    is_cold: Callable[[str], Awaitable[bool]] | None = None,
    source_ids: list[str] | None = None,
    defer_seconds: float | None = None,
    credit_pause_seconds: float = 300.0,
    unreachable_pause_seconds: float = 60.0,
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
        pause = 0.0
        try:
            n = await run_worker_once(
                store, ingest, batch=batch, budget=budget, max_attempts=max_attempts,
                backoff_base=backoff_base, backoff_cap=backoff_cap, lease=lease,
                concurrency=concurrency, is_cold=is_cold, cold_lock=cold_lock,
                source_ids=source_ids, defer_seconds=defer_seconds,
                credit_pause_seconds=credit_pause_seconds,
                unreachable_pause_seconds=unreachable_pause_seconds)
        except CreditsExhausted as e:
            logger.error("LLM provider out of credits (HTTP 402); jobs deferred without "
                         "spending attempts, pausing claims for %.0fs. Top up the account. "
                         "Provider said: %s", credit_pause_seconds, e)
            n, pause = 0, credit_pause_seconds
            metrics.pause("credits")
        except ProviderUnreachable as e:
            logger.error("upstream endpoint unreachable (LLM, embedder or DocExtractor); jobs "
                         "deferred without spending attempts, pausing claims for %.0fs. "
                         "Error: %s",
                         unreachable_pause_seconds, e)
            n, pause = 0, unreachable_pause_seconds
            metrics.pause("unreachable")
        batches += 1
        if max_batches is not None and batches >= max_batches:
            return
        if pause:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=pause)
            except asyncio.TimeoutError:
                pass
            continue
        if n == 0:
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_seconds)
            except asyncio.TimeoutError:
                pass
