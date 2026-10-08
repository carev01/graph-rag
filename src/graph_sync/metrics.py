"""Prometheus metrics for the ingestion pipeline (Grafana "graph-rag ingestion").

Two producers:

- **Workers** count what they do -- jobs by outcome, pauses, LLM tokens, job time --
  and serve them on `WORKER_METRICS_PORT` (`start_worker_metrics`). `WORKER_TIER`
  (`api` / `gpu`, set by the Helm chart) labels every series, so the dashboard can
  compare the OpenRouter tier with the vast.ai GPU tier.
- **The queue exporter** (`graph-sync metrics-exporter`) turns Postgres
  `semantic_jobs` into gauges per vendor / product / source / status, named through
  Neo4j's structural layer, plus the OpenRouter credit balance and the GPU's hourly
  rate. Read-only against both stores.

Prometheus finds the worker pods through a headless Service (deploy/monitoring/).
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Awaitable, Callable

import httpx
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger(__name__)

TIER = os.environ.get("WORKER_TIER", "api")

# --- worker side ------------------------------------------------------------------
JOBS = Counter("graphrag_worker_jobs_total",
               "Semantic jobs handled by this worker, by outcome: done, failed, "
               "deferred_lock (warm-up lock busy), deferred_halt (out of credits or an "
               "unreachable endpoint)", ["tier", "outcome"])
PAUSES = Counter("graphrag_worker_pauses_total",
                 "Claim pauses: credits (HTTP 402) or unreachable (endpoint down)",
                 ["tier", "reason"])
TOKENS = Counter("graphrag_worker_llm_tokens_total",
                 "LLM tokens spent by finished or failed jobs; kind=prompt|cached|completion "
                 "(cached is a subset of prompt)", ["tier", "kind"])
JOB_SECONDS = Histogram("graphrag_worker_job_seconds", "Wall time per job, by outcome",
                        ["tier", "outcome"],
                        buckets=(10, 30, 60, 120, 300, 600, 1200, 2400, 4800, 9600))


def job_outcome(outcome: str, seconds: float | None = None) -> None:
    JOBS.labels(TIER, outcome).inc()
    if seconds is not None:
        JOB_SECONDS.labels(TIER, outcome).observe(seconds)


def job_tokens(prompt: int, cached: int, completion: int) -> None:
    for kind, n in (("prompt", prompt), ("cached", cached), ("completion", completion)):
        if n:
            TOKENS.labels(TIER, kind).inc(n)


def pause(reason: str) -> None:
    PAUSES.labels(TIER, reason).inc()


def start_worker_metrics(port: int) -> None:
    """Serve this process's metrics; 0 disables. A port already in use is logged and
    skipped -- observability must never stop the worker from working."""
    if not port:
        return
    try:
        start_http_server(port)
        logger.info("worker metrics on :%d (tier=%s)", port, TIER)
    except OSError as e:
        logger.warning("worker metrics disabled: cannot bind :%d (%s)", port, e)


# --- queue exporter ---------------------------------------------------------------
QUEUE_JOBS = Gauge("graphrag_queue_jobs", "semantic_jobs rows",
                   ["vendor", "product", "source", "status"])
QUEUE_DONE_1H = Gauge("graphrag_queue_done_last_hour", "Jobs completed in the last hour",
                      ["vendor"])
QUEUE_RETRYING = Gauge("graphrag_queue_retrying_jobs",
                       "Pending jobs that have already spent at least one attempt")
TOKENS_TODAY = Gauge("graphrag_tokens_today", "LLM tokens recorded today (budget gate)")
CREDITS = Gauge("graphrag_openrouter_credits_remaining_usd",
                "OpenRouter balance (total credits minus usage)")
GPU_COST = Gauge("graphrag_gpu_hourly_cost_usd", "Configured hourly rate of the GPU instance")
REFRESHED = Gauge("graphrag_exporter_last_refresh_timestamp_seconds",
                  "Unix time of the last successful queue refresh")

_COUNTS_SQL = """
SELECT coalesce(source_id, '') AS source_id, status, count(*) AS n
FROM semantic_jobs GROUP BY 1, 2
"""
_DONE_1H_SQL = """
SELECT coalesce(source_id, '') AS source_id, count(*) AS n
FROM semantic_jobs WHERE status = 'done' AND updated_at > now() - interval '1 hour'
GROUP BY 1
"""
_RETRYING_SQL = "SELECT count(*) FROM semantic_jobs WHERE status = 'pending' AND attempts > 0"

SourceNames = dict[str, tuple[str, str, str]]    # source id -> (vendor, product, source)

# STRUCTURAL nodes only: graphiti's extracted entities also carry :Vendor/:Product
# labels, so the chain is matched through HAS_PRODUCT / HAS_SOURCE.
_NAMES_CYPHER = """
MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(s:Source)
RETURN s.id AS id, v.name AS vendor, p.name AS product, s.name AS source
"""


async def load_source_names(driver) -> SourceNames:
    records, _, _ = await driver.execute_query(_NAMES_CYPHER)
    return {r["id"]: (r["vendor"] or "?", r["product"] or "?", r["source"] or "?")
            for r in records}


async def openrouter_credits(api_key: str, base_url: str = "https://openrouter.ai/api/v1",
                             transport: httpx.AsyncBaseTransport | None = None
                             ) -> float | None:
    """Remaining balance, or None when it cannot be read (never raises)."""
    try:
        async with httpx.AsyncClient(timeout=10.0, transport=transport) as client:
            r = await client.get(f"{base_url}/credits",
                                 headers={"Authorization": f"Bearer {api_key}"})
            r.raise_for_status()
            d = r.json()["data"]
            return float(d["total_credits"]) - float(d["total_usage"])
    except Exception as e:  # noqa: BLE001 -- a missing gauge beats a dead exporter
        logger.warning("openrouter credits unavailable: %s", type(e).__name__)
        return None


def _name(names: SourceNames, source_id: str) -> tuple[str, str, str]:
    if not source_id:
        return ("(no source)", "(no source)", "(no source)")
    return names.get(source_id, ("(unknown)", "(unknown)", source_id))


async def refresh_queue(pool, names: SourceNames) -> None:
    """One pass over Postgres into the gauges. Label sets are rebuilt, so a status
    or source that disappears drops out instead of freezing at its last value."""
    counts = await pool.fetch(_COUNTS_SQL)
    done_1h = await pool.fetch(_DONE_1H_SQL)
    retrying = await pool.fetchval(_RETRYING_SQL)
    today = await pool.fetchval(
        "SELECT coalesce((SELECT tokens FROM token_ledger WHERE day = current_date), 0)")

    agg: dict[tuple[str, str, str, str], int] = {}
    for r in counts:
        key = (*_name(names, r["source_id"]), r["status"])
        agg[key] = agg.get(key, 0) + r["n"]
    by_vendor: dict[str, int] = {}
    for r in done_1h:
        vendor = _name(names, r["source_id"])[0]
        by_vendor[vendor] = by_vendor.get(vendor, 0) + r["n"]

    QUEUE_JOBS.clear()
    for (vendor, product, source, status), n in agg.items():
        QUEUE_JOBS.labels(vendor, product, source, status).set(n)
    QUEUE_DONE_1H.clear()
    for vendor, n in by_vendor.items():
        QUEUE_DONE_1H.labels(vendor).set(n)
    QUEUE_RETRYING.set(retrying or 0)
    TOKENS_TODAY.set(today or 0)
    REFRESHED.set(time.time())


async def run_exporter(
    *, pool, port: int, interval: float, stop: asyncio.Event,
    names_loader: Callable[[], Awaitable[SourceNames]],
    credits_loader: Callable[[], Awaitable[float | None]] | None,
    gpu_hourly_cost: float,
    names_every: float = 3600.0, credits_every: float = 300.0,
) -> None:
    """Refresh the gauges every `interval` until `stop`. A failed pass is logged and
    retried next tick: the last good values stay served, and REFRESHED shows how old
    they are."""
    if port:
        start_http_server(port)
    logger.info("queue exporter on :%d, refresh every %.0fs", port, interval)
    GPU_COST.set(gpu_hourly_cost)
    names: SourceNames = {}
    names_at = credits_at = float("-inf")
    while not stop.is_set():
        now = time.monotonic()
        if now - names_at >= names_every:
            try:
                names = await names_loader()
                names_at = now
            except Exception:  # noqa: BLE001 -- Neo4j down: keep the last names
                logger.exception("source names refresh failed; keeping the last ones")
        try:
            await refresh_queue(pool, names)
        except Exception:  # noqa: BLE001 -- keep serving the last good values
            logger.exception("queue refresh failed")
        if credits_loader is not None and now - credits_at >= credits_every:
            credits_at = now
            value = await credits_loader()
            if value is not None:
                CREDITS.set(value)
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass
