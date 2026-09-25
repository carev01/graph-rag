from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime

import asyncpg

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_cursor (
  id int PRIMARY KEY DEFAULT 1, cursor text, updated_at timestamptz DEFAULT now(),
  CHECK (id = 1));
CREATE TABLE IF NOT EXISTS bootstrap_progress (
  shard text PRIMARY KEY, watermark text, last_id text, status text,
  updated_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS webhook_delivery (
  signature text PRIMARY KEY, received_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS source_debounce (
  source_id text PRIMARY KEY, last_run_at timestamptz);
CREATE TABLE IF NOT EXISTS dead_letter (
  id bigserial PRIMARY KEY, raw text, context text, created_at timestamptz DEFAULT now());
CREATE TABLE IF NOT EXISTS semantic_jobs (
  id bigserial PRIMARY KEY, article_id text NOT NULL,
  op text NOT NULL, content_hash text,
  status text NOT NULL DEFAULT 'pending', attempts int NOT NULL DEFAULT 0,
  last_error text, enqueued_at timestamptz DEFAULT now(),
  claimed_at timestamptz, updated_at timestamptz DEFAULT now(),
  lane text NOT NULL DEFAULT 'incremental',
  next_attempt_at timestamptz NOT NULL DEFAULT now());
CREATE UNIQUE INDEX IF NOT EXISTS ux_semantic_jobs_pending
  ON semantic_jobs(article_id) WHERE status='pending';
ALTER TABLE semantic_jobs ADD COLUMN IF NOT EXISTS lane text NOT NULL DEFAULT 'incremental';
ALTER TABLE semantic_jobs ADD COLUMN IF NOT EXISTS next_attempt_at timestamptz NOT NULL DEFAULT now();
ALTER TABLE semantic_jobs ADD COLUMN IF NOT EXISTS source_id text;
CREATE INDEX IF NOT EXISTS ix_semantic_jobs_source ON semantic_jobs(source_id) WHERE status='pending';
CREATE INDEX IF NOT EXISTS ix_semantic_jobs_article ON semantic_jobs(article_id);
CREATE TABLE IF NOT EXISTS token_ledger (day date PRIMARY KEY, tokens bigint NOT NULL DEFAULT 0);
"""
_LOCK_KEY = 911_222_333
# Distinct from _LOCK_KEY: that one elects a single poller and is taken with
# try-once semantics; this one is contended and held for a whole article.
_WARMUP_LOCK_KEY = 911_222_334


class StateStore:
    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None
        self._lock_conn: asyncpg.Connection | None = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._dsn)
        return self._pool

    async def init_schema(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as c:
            await c.execute(_SCHEMA)

    async def close(self) -> None:
        # Idempotent: null out handles so a second close() is a no-op.
        if self._lock_conn is not None:
            await self._lock_conn.close()
            self._lock_conn = None
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def get_cursor(self) -> str | None:
        pool = await self._get_pool()
        return await pool.fetchval("SELECT cursor FROM sync_cursor WHERE id=1")

    async def set_cursor(self, cursor: str) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "INSERT INTO sync_cursor (id, cursor, updated_at) VALUES (1,$1,now()) "
            "ON CONFLICT (id) DO UPDATE SET cursor=$1, updated_at=now()", cursor)

    async def upsert_bootstrap(self, shard, watermark, last_id, status) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "INSERT INTO bootstrap_progress (shard,watermark,last_id,status,updated_at) "
            "VALUES ($1,$2,$3,$4,now()) ON CONFLICT (shard) DO UPDATE SET "
            "watermark=$2,last_id=$3,status=$4,updated_at=now()",
            shard, watermark, last_id, status)

    async def get_bootstrap(self, shard) -> dict | None:
        pool = await self._get_pool()
        row = await pool.fetchrow("SELECT * FROM bootstrap_progress WHERE shard=$1", shard)
        return dict(row) if row else None

    async def all_bootstrap_watermarks(self) -> list[str]:
        pool = await self._get_pool()
        rows = await pool.fetch(
            "SELECT watermark FROM bootstrap_progress WHERE watermark IS NOT NULL")
        return [r["watermark"] for r in rows]

    async def seen_delivery(self, signature: str) -> bool:
        pool = await self._get_pool()
        res = await pool.execute(
            "INSERT INTO webhook_delivery (signature) VALUES ($1) "
            "ON CONFLICT (signature) DO NOTHING", signature)
        return res == "INSERT 0 0"  # 0 rows inserted => already seen

    async def should_run_source(self, source_id: str, debounce_seconds: int) -> bool:
        pool = await self._get_pool()
        async with pool.acquire() as c, c.transaction():
            row = await c.fetchrow(
                "SELECT last_run_at FROM source_debounce WHERE source_id=$1 FOR UPDATE",
                source_id)
            if row and row["last_run_at"] is not None:
                due = await c.fetchval(
                    "SELECT now() - $1 > make_interval(secs => $2)",
                    row["last_run_at"], debounce_seconds)
                if not due:
                    return False
            await c.execute(
                "INSERT INTO source_debounce (source_id,last_run_at) VALUES ($1,now()) "
                "ON CONFLICT (source_id) DO UPDATE SET last_run_at=now()", source_id)
            return True

    async def record_dead_letter(self, raw: str, context: str) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "INSERT INTO dead_letter (raw,context) VALUES ($1,$2)", raw, context)

    async def dead_letter_count(self) -> int:
        pool = await self._get_pool()
        return await pool.fetchval("SELECT count(*) FROM dead_letter")

    async def enqueue_semantic_job(
        self, article_id: str, op: str, content_hash: str | None,
        lane: str = "incremental", source_id: str | None = None,
    ) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "INSERT INTO semantic_jobs (article_id, op, content_hash, lane, source_id) "
            "VALUES ($1,$2,$3,$4,$5) "
            "ON CONFLICT (article_id) WHERE status='pending' DO UPDATE SET "
            "op=excluded.op, content_hash=excluded.content_hash, "
            "lane=CASE WHEN excluded.lane='incremental' OR semantic_jobs.lane='incremental' "
            "THEN 'incremental' ELSE 'bootstrap' END, "
            "source_id=coalesce(excluded.source_id, semantic_jobs.source_id), "
            "enqueued_at=now(), updated_at=now()",
            article_id, op, content_hash, lane, source_id)

    async def enqueue_semantic_job_if_absent(
        self, article_id: str, op: str, content_hash: str | None, source_id: str | None,
    ) -> bool:
        """Insert a `bootstrap`-lane job only if the article has NO row in any status;
        never touch an existing one. Returns whether it inserted.

        For the hash gate's requeue (`SyncCore._requeue_unextracted`), which reads
        `has_semantic_job` and writes later: a producer may enqueue for the same
        article in between -- a tombstone, say -- and `enqueue_semantic_job`'s
        ON CONFLICT would flip that pending `remove` back to `upsert`. Here the
        existence test and the insert are one statement, and a concurrent pending
        insert that slips past the NOT EXISTS still loses to DO NOTHING."""
        pool = await self._get_pool()
        result = await pool.execute(
            "INSERT INTO semantic_jobs (article_id, op, content_hash, lane, source_id) "
            "SELECT $1, $2, $3, 'bootstrap', $4 "
            "WHERE NOT EXISTS (SELECT 1 FROM semantic_jobs WHERE article_id=$1) "
            "ON CONFLICT DO NOTHING",
            article_id, op, content_hash, source_id)
        return result == "INSERT 0 1"

    async def has_semantic_job(self, article_id: str) -> bool:
        """Whether the article has EVER had a semantic job, in any status (the hash
        gate's requeue check, `SyncCore._requeue_unextracted`). Served by
        `ix_semantic_jobs_article`; `ux_semantic_jobs_pending` only covers pending."""
        pool = await self._get_pool()
        return bool(await pool.fetchval(
            "SELECT EXISTS (SELECT 1 FROM semantic_jobs WHERE article_id=$1)", article_id))

    async def claim_semantic_jobs(
        self, batch: int, include_bootstrap: bool, source_ids: list[str] | None = None
    ) -> list[dict]:
        # `source_ids` scopes a worker to a subset of sources (the k3s
        # bootstrap-first rehearsal: `SEMANTIC_CLAIM_SOURCE_IDS`). `None`/empty
        # is unscoped -- today's behaviour exactly, and the `$3::text[] IS
        # NULL` branch below makes that the case whether the caller passes
        # `None` or `[]`.
        pool = await self._get_pool()
        rows = await pool.fetch(
            "UPDATE semantic_jobs SET status='in_progress', claimed_at=now(), updated_at=now() "
            "WHERE id IN (SELECT id FROM semantic_jobs "
            "WHERE status='pending' AND next_attempt_at <= now() "
            "AND ($2 OR lane='incremental') "
            "AND ($3::text[] IS NULL OR source_id = ANY($3::text[])) "
            "ORDER BY (lane='bootstrap'), next_attempt_at "
            "FOR UPDATE SKIP LOCKED LIMIT $1) "
            "RETURNING id, article_id, op, content_hash, attempts, lane, claimed_at",
            batch, include_bootstrap, source_ids or None)
        return [dict(r) for r in rows]

    async def complete_semantic_job(self, job_id: int, claimed_at: datetime | None) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "UPDATE semantic_jobs SET status='done', updated_at=now() "
            "WHERE id=$1 AND claimed_at=$2 AND status='in_progress'", job_id, claimed_at)

    async def fail_semantic_job(
        self, job_id: int, error: str, *, max_attempts: int, retry_delay_seconds: float,
        claimed_at: datetime | None
    ) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "UPDATE semantic_jobs SET attempts=attempts+1, last_error=$2, "
            "status=CASE WHEN attempts+1 >= $3 THEN 'dead' ELSE 'pending' END, "
            "next_attempt_at=CASE WHEN attempts+1 >= $3 THEN next_attempt_at "
            "ELSE now() + make_interval(secs => $4) END, updated_at=now() "
            "WHERE id=$1 AND claimed_at=$5 AND status='in_progress'",
            job_id, error, max_attempts, retry_delay_seconds, claimed_at)

    async def reap_stale_jobs(self, lease_seconds: float, max_attempts: int) -> int:
        pool = await self._get_pool()
        res = await pool.execute(
            "UPDATE semantic_jobs SET attempts=attempts+1, updated_at=now(), "
            "status=CASE WHEN attempts+1 >= $2 THEN 'dead' ELSE 'pending' END, "
            "next_attempt_at=CASE WHEN attempts+1 >= $2 THEN next_attempt_at ELSE now() END "
            "WHERE status='in_progress' AND claimed_at < now() - make_interval(secs => $1)",
            lease_seconds, max_attempts)
        return int(res.split()[-1])   # "UPDATE <n>"

    async def dead_semantic_job_count(self) -> int:
        pool = await self._get_pool()
        return await pool.fetchval("SELECT count(*) FROM semantic_jobs WHERE status='dead'")

    async def job_status_counts(self) -> dict[str, int]:
        pool = await self._get_pool()
        rows = await pool.fetch("SELECT status, count(*) AS n FROM semantic_jobs GROUP BY status")
        return {r["status"]: r["n"] for r in rows}

    async def record_tokens(self, delta: int) -> None:
        pool = await self._get_pool()
        await pool.execute(
            "INSERT INTO token_ledger (day, tokens) VALUES (current_date, $1) "
            "ON CONFLICT (day) DO UPDATE SET tokens = token_ledger.tokens + $1", delta)

    async def today_token_total(self) -> int:
        pool = await self._get_pool()
        return await pool.fetchval(
            "SELECT COALESCE((SELECT tokens FROM token_ledger WHERE day=current_date), 0)")

    @asynccontextmanager
    async def warmup_lock(self, *, timeout: float, poll: float = 5.0):
        """Serialise warm-up articles across worker PROCESSES. Yields True if held.

        `run_concurrently`'s barrier only serialises cold articles within ONE
        process. That is enough for a single worker, and it is also -- by
        accident rather than design -- what protects cross-source hub entities:
        while any source is warming up, nothing else runs anywhere.

        Scale to N workers and that protection disappears exactly where it costs
        most. Measured on the 999-entity baseline: entities appearing in the
        warm-up window of two or more sources carry 20.7% of all duplicate risk
        weight, and they are the biggest hubs in the graph (`Azure Backup` k=55,
        `Backup vault` k=33, `AWS Backup` k=30) -- the cross-vendor entities
        design invariant #4 exists to keep unified. This lock restores the
        single-process guarantee across processes.

        Held for the whole article, so ~5 minutes. The lock is session-scoped, so
        a worker that dies releases it when its connection drops -- no reaper is
        needed for the crash case.

        Two things release it, deliberately: the explicit unlock below, and
        asyncpg's pool reset, which runs `pg_advisory_unlock_all()` when the
        connection goes back to the pool (`asyncpg/connection.py:1748`). The
        second means removing the first would not break any test -- so do not
        read the tests as proof this line is load-bearing. It is defence in
        depth against a future change that stops using a pooled connection, and
        the `if held` guard only avoids a Postgres warning for unlocking a lock
        we never took.

        On timeout it logs and yields False rather than blocking forever: a
        wedged holder must not stall a multi-week bootstrap. That trades the
        guarantee for progress, so it is loud, and the duplicates it admits are
        exact-name ones `merge_duplicates` can still clean.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            deadline = time.monotonic() + timeout
            held = False
            while True:
                held = await conn.fetchval(
                    "SELECT pg_try_advisory_lock($1)", _WARMUP_LOCK_KEY)
                if held or time.monotonic() >= deadline:
                    break
                await asyncio.sleep(poll)
            if not held:
                logger.warning(
                    "warm-up lock not acquired after %.0fs; proceeding WITHOUT it. "
                    "Another worker is holding it for longer than one article "
                    "should take. Cross-source hub entities may be duplicated "
                    "until merge-duplicates runs.", timeout)
            try:
                yield held
            finally:
                if held:
                    await conn.execute(
                        "SELECT pg_advisory_unlock($1)", _WARMUP_LOCK_KEY)

    async def try_lock(self) -> bool:
        if self._lock_conn is None:
            self._lock_conn = await asyncpg.connect(self._dsn)
        return await self._lock_conn.fetchval("SELECT pg_try_advisory_lock($1)", _LOCK_KEY)

    async def unlock(self) -> None:
        if self._lock_conn is not None:
            await self._lock_conn.execute("SELECT pg_advisory_unlock($1)", _LOCK_KEY)
