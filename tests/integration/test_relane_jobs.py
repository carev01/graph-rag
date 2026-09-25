"""Integration tests for `graph_sync.relane`'s IO wrapper functions (real
testcontainer Neo4j + Postgres). tests/unit/test_relane.py pins the pure
planning decisions (`plan_backfill`/`plan_relane`) in isolation; these pin the
wiring around them: report mode reads and computes but never writes, apply
mode performs exactly the writes the plan decided on, and both steps are
idempotent."""
from __future__ import annotations

import pytest

from graph_sync.models import StructuralWrite
from graph_sync.relane import backfill_source_ids, relane_incremental_jobs

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _write(article_id: str, source_id: str) -> StructuralWrite:
    return StructuralWrite(
        vendor={"id": f"v-{source_id}", "name": "V", "website": None},
        product={"id": f"p-{source_id}", "name": "P", "version": None,
                 "vendor_id": f"v-{source_id}"},
        source={"id": source_id, "name": "S", "base_url": None, "source_type": None,
                "platform": None, "last_extracted_at": None},
        article={"id": article_id, "title": "T", "source_url": "u", "topic_key": "tk",
                 "content_hash": "h", "estimated_tokens": 10, "sort_order": 0,
                 "last_updated_at": None, "run_id": "r1", "seq": None,
                 "source_id": source_id, "removed": False},
    )


async def _mark_done(state_store, *article_ids: str) -> None:
    """Test hygiene: `state_store`/`neo4j_repo` are module-scoped and shared
    across every test in this file, and both relane queries select on
    `status='pending'` -- leaving a test's rows pending would make a LATER
    test's counts depend on execution order."""
    pool = await state_store._get_pool()
    await pool.execute(
        "UPDATE semantic_jobs SET status='done' WHERE article_id = ANY($1::text[])",
        list(article_ids))


async def test_backfill_report_mode_writes_nothing(neo4j_repo, state_store):
    await neo4j_repo.apply_structural(_write("relane-a1", "relane-src-1"))
    await state_store.enqueue_semantic_job("relane-a1", "upsert", "h", "incremental")
    try:
        report = await backfill_source_ids(state_store, neo4j_repo, apply=False)
        assert report.backfilled >= 1  # Neo4j had a match -> it WOULD be backfilled
        pool = await state_store._get_pool()
        assert await pool.fetchval(
            "SELECT source_id FROM semantic_jobs WHERE article_id='relane-a1'") is None
    finally:
        await _mark_done(state_store, "relane-a1")


async def test_backfill_apply_mode_writes_the_found_source_id(neo4j_repo, state_store):
    await neo4j_repo.apply_structural(_write("relane-a2", "relane-src-2"))
    await state_store.enqueue_semantic_job("relane-a2", "upsert", "h", "incremental")
    try:
        report = await backfill_source_ids(state_store, neo4j_repo, apply=True)
        assert report.backfilled >= 1
        pool = await state_store._get_pool()
        assert await pool.fetchval(
            "SELECT source_id FROM semantic_jobs WHERE article_id='relane-a2'") == "relane-src-2"
    finally:
        await _mark_done(state_store, "relane-a2")


async def test_backfill_counts_an_id_missing_from_neo4j(neo4j_repo, state_store):
    await state_store.enqueue_semantic_job("relane-ghost", "upsert", "h", "incremental")
    try:
        report = await backfill_source_ids(state_store, neo4j_repo, apply=True)
        assert report.missing_in_graph >= 1
        pool = await state_store._get_pool()
        assert await pool.fetchval(
            "SELECT source_id FROM semantic_jobs WHERE article_id='relane-ghost'") is None
    finally:
        await _mark_done(state_store, "relane-ghost")


async def test_backfill_is_idempotent_on_a_rerun(neo4j_repo, state_store):
    await neo4j_repo.apply_structural(_write("relane-a3", "relane-src-3"))
    await state_store.enqueue_semantic_job("relane-a3", "upsert", "h", "incremental")
    try:
        first = await backfill_source_ids(state_store, neo4j_repo, apply=True)
        assert first.backfilled >= 1
        # already backfilled -> no longer a NULL-source_id candidate, so a
        # re-run must not touch it again (idempotent: safe to re-run after a
        # partial failure).
        pool = await state_store._get_pool()
        assert await pool.fetchval(
            "SELECT source_id FROM semantic_jobs WHERE article_id='relane-a3'") == "relane-src-3"
        await backfill_source_ids(state_store, neo4j_repo, apply=True)
        assert await pool.fetchval(
            "SELECT source_id FROM semantic_jobs WHERE article_id='relane-a3'") == "relane-src-3"
    finally:
        await _mark_done(state_store, "relane-a3")


async def test_relane_report_mode_writes_nothing(neo4j_repo, state_store):
    await neo4j_repo.apply_structural(_write("relane-new1", "relane-src-4"))
    await state_store.enqueue_semantic_job("relane-new1", "upsert", "h", "incremental")
    try:
        await relane_incremental_jobs(state_store, neo4j_repo, apply=False)
        pool = await state_store._get_pool()
        assert await pool.fetchval(
            "SELECT lane FROM semantic_jobs WHERE article_id='relane-new1'") == "incremental"
    finally:
        await _mark_done(state_store, "relane-new1")


async def test_relane_apply_update_recheck_protects_a_row_claimed_after_the_plan(
    neo4j_repo, state_store, monkeypatch
):
    """The apply UPDATE re-checks `status='pending' AND lane='incremental'` at
    write time, not just at the read that built the plan. Simulates the race
    the runbook also warns operators never to create (never run a worker
    between the first pull and `relane-jobs --apply`): a plan that -- as if
    computed slightly earlier -- still says "move this job" for one that has,
    by write time, already been claimed by a worker. That row must be left
    untouched rather than silently reclassified out from under whatever the
    worker is doing with it."""
    from graph_sync import relane as relane_mod

    await neo4j_repo.apply_structural(_write("relane-race", "relane-src-6"))
    # no episode edge -> a fresh plan would decide to move it
    await state_store.enqueue_semantic_job("relane-race", "upsert", "h", "incremental")
    pool = await state_store._get_pool()
    [job_id] = [r["id"] for r in await pool.fetch(
        "SELECT id FROM semantic_jobs WHERE article_id='relane-race'")]
    try:
        real_plan_relane = relane_mod.plan_relane

        def _stale_plan(jobs, has_episodes):
            # The real plan, plus this job forced into to_move regardless --
            # standing in for a plan computed before the claim below happened.
            plan = real_plan_relane(jobs, has_episodes)
            if job_id not in plan.to_move:
                plan.to_move.append(job_id)
            return plan

        monkeypatch.setattr(relane_mod, "plan_relane", _stale_plan)
        # A worker claims the job AFTER the (simulated) plan was decided.
        await pool.execute(
            "UPDATE semantic_jobs SET status='in_progress', claimed_at=now() WHERE id=$1",
            job_id)

        report = await relane_incremental_jobs(state_store, neo4j_repo, apply=True)

        row = await pool.fetchrow(
            "SELECT status, lane FROM semantic_jobs WHERE id=$1", job_id)
        assert row["status"] == "in_progress" and row["lane"] == "incremental", (
            "a job claimed after the plan was built must be left untouched")
        assert report.moved_to_bootstrap == 0
    finally:
        await pool.execute("UPDATE semantic_jobs SET status='done' WHERE id=$1", job_id)


async def test_relane_apply_mode_moves_exactly_the_no_episode_jobs(neo4j_repo, state_store):
    await neo4j_repo.apply_structural(_write("relane-ext", "relane-src-5"))
    await neo4j_repo.add_episode_edge("relane-ext")  # already extracted -> stays incremental
    await neo4j_repo.apply_structural(_write("relane-new2", "relane-src-5"))
    # relane-new2 has NO episode edge -- never extracted -> must move

    await state_store.enqueue_semantic_job("relane-ext", "upsert", "h", "incremental")
    await state_store.enqueue_semantic_job("relane-new2", "upsert", "h", "incremental")
    try:
        report = await relane_incremental_jobs(state_store, neo4j_repo, apply=True)
        assert report.moved_to_bootstrap >= 1
        pool = await state_store._get_pool()
        lanes = {r["article_id"]: r["lane"] for r in await pool.fetch(
            "SELECT article_id, lane FROM semantic_jobs "
            "WHERE article_id IN ('relane-ext', 'relane-new2')")}
        assert lanes == {"relane-ext": "incremental", "relane-new2": "bootstrap"}
    finally:
        await _mark_done(state_store, "relane-ext", "relane-new2")
