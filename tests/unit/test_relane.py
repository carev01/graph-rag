"""Unit tests for the pure planning functions behind `relane-jobs`
(graph_sync.relane): no IO, so these pin the actual decision logic -- what
counts as "backfillable" and what counts as "never extracted" -- independent
of the Postgres/Neo4j wiring (covered by tests/integration/test_relane_jobs.py).
"""
from __future__ import annotations

from graph_sync.relane import (
    BackfillReport,
    RelaneReport,
    format_report,
    plan_backfill,
    plan_relane,
)


def test_plan_backfill_updates_ids_found_in_neo4j():
    plan = plan_backfill(["a1", "a2", "a3"], {"a1": "s1", "a3": "s2"})
    assert set(plan.updates) == {("a1", "s1"), ("a3", "s2")}
    assert plan.missing == ["a2"]


def test_plan_backfill_all_missing():
    plan = plan_backfill(["a1", "a2"], {})
    assert plan.updates == []
    assert plan.missing == ["a1", "a2"]


def test_plan_backfill_all_found():
    plan = plan_backfill(["a1", "a2"], {"a1": "s1", "a2": "s1"})
    assert set(plan.updates) == {("a1", "s1"), ("a2", "s1")}
    assert plan.missing == []


def test_plan_relane_moves_jobs_with_no_episodes():
    jobs = [(1, "a1"), (2, "a2"), (3, "a3")]
    plan = plan_relane(jobs, has_episodes={"a1", "a3"})
    assert plan.kept == [1, 3]
    assert plan.to_move == [2]


def test_plan_relane_keeps_everything_when_all_have_episodes():
    jobs = [(1, "a1"), (2, "a2")]
    plan = plan_relane(jobs, has_episodes={"a1", "a2"})
    assert plan.kept == [1, 2]
    assert plan.to_move == []


def test_plan_relane_moves_everything_when_none_have_episodes():
    jobs = [(1, "a1"), (2, "a2")]
    plan = plan_relane(jobs, has_episodes=set())
    assert plan.kept == []
    assert plan.to_move == [1, 2]


def test_format_report_report_mode_says_pass_apply():
    text = format_report(
        False,
        BackfillReport(candidates=10, backfilled=8, missing_in_graph=2),
        RelaneReport(checked=5, kept_incremental=3, moved_to_bootstrap=2),
    )
    assert "REPORT" in text and "--apply" in text
    assert "8 article(s) backfilled" in text
    assert "2 missing in graph" in text
    assert "2 moved to bootstrap" in text
    # missing_in_graph > 0 -> the never-structurally-written-remove-job note appears
    assert "never structurally written" in text


def test_format_report_apply_mode_says_apply():
    text = format_report(
        True,
        BackfillReport(candidates=0, backfilled=0, missing_in_graph=0),
        RelaneReport(checked=0, kept_incremental=0, moved_to_bootstrap=0),
    )
    assert "APPLY" in text
    assert "--apply" not in text
    # nothing missing -> the note is omitted rather than printed unconditionally
    assert "never structurally written" not in text
