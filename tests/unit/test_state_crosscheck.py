"""`graph_sync.state_crosscheck.evaluate`: the verdict the k3s runbook gates the
poller on after the state migration. The I/O half (`_read`) is read-only and is
exercised by the operator against the real stores; this pins the decision."""
from __future__ import annotations

from graph_sync.state_crosscheck import DoneJob, evaluate


def test_all_consistent_passes():
    lines, ok = evaluate({"src-1": 120, "global": 900},
                         [DoneJob("a1", True, "Configure backup", 3)])
    assert ok
    assert all(line.startswith(("OK", "  ")) for line in lines)


def test_a_complete_shard_with_no_articles_fails():
    lines, ok = evaluate({"src-1": 120, "src-2": 0}, [])
    assert not ok
    assert any(line.startswith("FAIL") and "src-2" in line for line in lines)


def test_a_done_job_whose_article_is_missing_fails_and_is_named():
    lines, ok = evaluate({}, [DoneJob("gone", False, None, 0)])
    assert not ok
    assert any("missing :Article gone" in line for line in lines)


def test_a_done_job_without_episodes_fails():
    lines, ok = evaluate({}, [DoneJob("a2", True, "Restore a VM", 0)])
    assert not ok
    assert any("no HAS_EPISODE   a2" in line for line in lines)


def test_a_navigation_page_with_no_episodes_is_expected_not_a_failure():
    lines, ok = evaluate({}, [DoneJob("nav", True, "Release notes 12.1", 0)])
    assert ok
    assert any("1 navigation" in line for line in lines)


def test_no_stranded_articles_passes():
    lines, ok = evaluate({}, [], stranded={})
    assert ok
    assert any(line.startswith("OK") and "0 stranded" in line for line in lines)


def test_stranded_articles_fail_per_source_and_name_the_repair():
    """BACKLOG 50: a graph Article with content, no episodes and NO job row of any
    status was never extracted and never will be unless re-queued. The line must
    say which sources, and that re-running their bootstrap is the repair."""
    lines, ok = evaluate({}, [], stranded={"src-1": ["a", "b"], "src-2": ["c"]})
    assert not ok
    fail = [line for line in lines if line.startswith("FAIL")]
    assert fail and "3 stranded" in fail[0]
    text = "\n".join(lines)
    assert "src-1: 2" in text and "src-2: 1" in text
    assert "bootstrap --source-id" in text
