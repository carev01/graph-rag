"""One-off repair for `semantic_jobs` rows enqueued before this change existed
(`python -m graph_sync.cli relane-jobs [--apply]`; docs/deploy/k3s.md rehearsal
step). Two independent gaps, one command, two steps:

Step A -- `source_id` backfill. `enqueue_semantic_job` did not persist
`source_id` before scoped claims existed, so every row queued before this
change has `source_id IS NULL`. `claim_semantic_jobs`'s `source_ids` filter
depends on that column, so pre-existing rows are backfilled by looking each
`article_id`'s `Article.source_id` up in Neo4j (READ only, batched -- one
`UNWIND` query per `batch_size` ids rather than a round trip per row).

Step B -- lane repair. `SyncCore._apply_record`'s lane rule (`has_episodes`)
did not exist before this change either, so every incremental-lane job queued
before it is unconditionally `lane='incremental'` -- including brand-new,
never-extracted articles that should have landed in the budgeted `bootstrap`
lane. That is the 5,646-job gap this whole change fixes for FUTURE pulls; this
step repairs the jobs the cluster's first (pre-fix) incremental pull already
queued. It moves `status='pending' AND lane='incremental'` jobs whose article
has no `HAS_EPISODE` edge to `lane='bootstrap'`.

Report-only by default; `--apply` performs the writes. Neo4j is read-only in
both steps; Postgres writes happen only under `--apply`, one transaction per
step. Both steps are idempotent -- an already-backfilled row (`source_id` no
longer NULL) or an already-moved job (`lane` no longer `'incremental'`) is
simply not selected on a re-run, so a partial run (e.g. Step A applied, Step B
not yet) can always be re-run safely.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from graph_sync.neo4j_repo import Neo4jRepo
from graph_sync.state_store import StateStore

_BATCH_SIZE = 500


@dataclass
class BackfillPlan:
    updates: list[tuple[str, str]] = field(default_factory=list)  # (article_id, source_id)
    missing: list[str] = field(default_factory=list)  # no Article / no source_id in Neo4j


@dataclass
class RelanePlan:
    kept: list[int] = field(default_factory=list)  # semantic_jobs.id: has episodes, stays incremental
    to_move: list[int] = field(default_factory=list)  # semantic_jobs.id: no episodes, moves to bootstrap


@dataclass
class BackfillReport:
    candidates: int
    backfilled: int
    missing_in_graph: int


@dataclass
class RelaneReport:
    checked: int
    kept_incremental: int
    moved_to_bootstrap: int


def plan_backfill(article_ids: list[str], found: dict[str, str]) -> BackfillPlan:
    """Pure planning step for Step A (no IO), so the one thing this repair must
    never do -- write a `source_id` for an id Neo4j did NOT actually return one
    for -- is a plain, unit-testable branch rather than buried inside a query
    and a transaction."""
    plan = BackfillPlan()
    for aid in article_ids:
        sid = found.get(aid)
        if sid is not None:
            plan.updates.append((aid, sid))
        else:
            plan.missing.append(aid)
    return plan


def plan_relane(jobs: list[tuple[int, str]], has_episodes: set[str]) -> RelanePlan:
    """Pure planning step for Step B (no IO). `jobs` is (semantic_jobs.id,
    article_id) pairs already filtered to `status='pending' AND
    lane='incremental'`; `has_episodes` is the subset of their article_ids
    Neo4j confirmed have at least one `HAS_EPISODE` edge. A job whose article
    IS in that set genuinely extracted something and stays `incremental`;
    everything else never extracted anything and must move to `bootstrap` (see
    module docstring, Step B) -- moving a job WITH episodes would silently move
    a real incremental update behind the daily token budget."""
    plan = RelanePlan()
    for job_id, article_id in jobs:
        if article_id in has_episodes:
            plan.kept.append(job_id)
        else:
            plan.to_move.append(job_id)
    return plan


async def backfill_source_ids(
    store: StateStore, repo: Neo4jRepo, *, apply: bool, batch_size: int = _BATCH_SIZE,
) -> BackfillReport:
    """Step A. Reads candidate ids and Neo4j's answers regardless of `apply`
    (report mode needs the same counts as apply mode); only the Postgres write
    is gated."""
    pool = await store._get_pool()
    article_ids = [r["article_id"] for r in await pool.fetch(
        "SELECT DISTINCT article_id FROM semantic_jobs WHERE source_id IS NULL")]
    updates: list[tuple[str, str]] = []
    missing = 0
    for i in range(0, len(article_ids), batch_size):
        chunk = article_ids[i:i + batch_size]
        found = await repo.article_source_ids(chunk)
        chunk_plan = plan_backfill(chunk, found)
        updates.extend(chunk_plan.updates)
        missing += len(chunk_plan.missing)
    if apply and updates:
        async with pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                "UPDATE semantic_jobs SET source_id=$2 "
                "WHERE article_id=$1 AND source_id IS NULL",
                updates,
            )
    return BackfillReport(
        candidates=len(article_ids), backfilled=len(updates), missing_in_graph=missing)


async def relane_incremental_jobs(
    store: StateStore, repo: Neo4jRepo, *, apply: bool, batch_size: int = _BATCH_SIZE,
) -> RelaneReport:
    """Step B. Same report/apply split as Step A: the read (and the plan it
    produces) always runs; only the write is gated on `apply`."""
    pool = await store._get_pool()
    jobs = [(r["id"], r["article_id"]) for r in await pool.fetch(
        "SELECT id, article_id FROM semantic_jobs "
        "WHERE status='pending' AND lane='incremental'")]
    article_ids = sorted({article_id for _, article_id in jobs})
    has_episodes: set[str] = set()
    for i in range(0, len(article_ids), batch_size):
        chunk = article_ids[i:i + batch_size]
        has_episodes |= await repo.articles_with_episodes(chunk)
    plan = plan_relane(jobs, has_episodes)
    if apply and plan.to_move:
        async with pool.acquire() as conn, conn.transaction():
            await conn.execute(
                "UPDATE semantic_jobs SET lane='bootstrap', updated_at=now() "
                "WHERE id = ANY($1::bigint[])",
                plan.to_move,
            )
    return RelaneReport(
        checked=len(jobs), kept_incremental=len(plan.kept),
        moved_to_bootstrap=len(plan.to_move))


def format_report(apply: bool, backfill: BackfillReport, relane: RelaneReport) -> str:
    mode = "APPLY" if apply else "REPORT (pass --apply to write)"
    return (
        f"relane-jobs [{mode}]\n"
        f"  source_id backfill: {backfill.backfilled} backfilled, "
        f"{backfill.missing_in_graph} missing in graph "
        f"(of {backfill.candidates} candidates)\n"
        f"  incremental lane repair: {relane.checked} pending incremental job(s) checked, "
        f"{relane.kept_incremental} kept incremental, "
        f"{relane.moved_to_bootstrap} moved to bootstrap"
    )
