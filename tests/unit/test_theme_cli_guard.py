"""`theme-build` calls the duplicate guard BEFORE it builds anything, and does
not call it at all for `--verify-pending`.

Hermetic on purpose: `theme-build --allow-duplicates` proceeds into a real
community build on the strong tier, so no test may ever reach that path with
real dependencies. Every dependency the command resolves at call time is
stubbed here -- settings, the Neo4j driver, the guard and all three runners --
and the assertions are on the ORDER of the recorded calls, not on their
presence. A guard that ran after the reports were generated has saved
nothing, and a presence-only assertion cannot tell the two apart.
"""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import theme_builder.cli as theme_cli
from graph_extract.config import ExtractSettings
from graph_extract.merge_duplicates import DuplicateEntitiesError
from theme_builder.cli import app as theme_app

RUNNERS = ("_run_theme_build_incremental", "_run_theme_build", "_run_verify_pending")


class _FakeDriver:
    async def close(self) -> None:
        return None


class _FakeGraphDatabase:
    """Records the driver it hands out so a test can check the guard got THAT
    one (the driver the build would use), not a second connection."""
    handed_out: list[_FakeDriver] = []

    @classmethod
    def driver(cls, uri, auth):
        driver = _FakeDriver()
        cls.handed_out.append(driver)
        return driver


@pytest.fixture
def hermetic(monkeypatch):
    settings = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                               neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                               group_id="g-theme")
    _FakeGraphDatabase.handed_out = []
    monkeypatch.setattr(theme_cli, "get_extract_settings", lambda: settings)
    monkeypatch.setattr(theme_cli, "AsyncGraphDatabase", _FakeGraphDatabase)
    return settings


# --- the brief's test, verbatim in its assertions ------------------------------
# Parametrised over both BUILD modes: the guard sits on the modes that detect
# communities and write reports, so `--full` must be refused the same way as
# the incremental default. `--verify-pending` is deliberately absent here and
# pinned ungated below. All three runners are stubbed with the same recorder,
# so whichever one a mutant routed to would still be seen.

@pytest.mark.parametrize("mode", [[], ["--full"]])
def test_theme_build_calls_the_guard_before_doing_any_work(monkeypatch, hermetic, mode):
    """Hermetic: every dependency is stubbed, so no LLM call can happen. Proves
    the guard runs BEFORE the build, which is the whole point -- a guard that
    runs after the reports are generated has saved nothing."""
    calls: list[str] = []

    async def _fake_guard(driver, group_id, *, allow):
        calls.append("guard")
        raise DuplicateEntitiesError("2 exact-name duplicate entities")

    async def _fake_build(*a, **k):
        calls.append("build")
        return {}

    monkeypatch.setattr(theme_cli, "assert_no_duplicates", _fake_guard)
    for runner in RUNNERS:
        monkeypatch.setattr(theme_cli, runner, _fake_build)
    result = CliRunner().invoke(theme_app, ["theme-build", *mode])
    assert result.exit_code != 0
    assert calls == ["guard"], f"the build must not start; got {calls}"
    assert "2 exact-name duplicate entities" in result.output, "the refusal names the count"
    assert result.exception is None or isinstance(result.exception, SystemExit), \
        "a refusal is a clean non-zero exit, not a traceback"


@pytest.mark.parametrize("flag, expected_allow", [([], False), (["--allow-duplicates"], True)])
def test_allow_duplicates_reaches_the_guard_and_the_build_follows_it(
        monkeypatch, hermetic, flag, expected_allow):
    """The other half of the order: when the guard lets the run through, the
    build runs AFTER it (not before, not instead), with the driver the guard
    saw and the configured group. `--allow-duplicates` is what the guard
    receives as `allow` -- a flag that were parsed and never passed would
    leave the guard always strict."""
    calls: list[str] = []
    seen: dict = {}

    async def _fake_guard(driver, group_id, *, allow):
        calls.append("guard")
        seen.update(driver=driver, group_id=group_id, allow=allow)
        return 2

    async def _fake_build(settings, *, driver):
        calls.append("build")
        seen["build_driver"] = driver
        return {"communities_detected": 4}

    monkeypatch.setattr(theme_cli, "assert_no_duplicates", _fake_guard)
    for runner in RUNNERS:
        monkeypatch.setattr(theme_cli, runner, _fake_build)
    result = CliRunner().invoke(theme_app, ["theme-build", *flag])
    assert result.exit_code == 0, result.output
    assert calls == ["guard", "build"]
    assert seen["allow"] is expected_allow
    assert seen["group_id"] == "g-theme"
    assert _FakeGraphDatabase.handed_out == [seen["driver"]]
    assert seen["build_driver"] is seen["driver"], "one driver: the guard checked the build's graph"
    assert json.loads(result.stdout) == {"communities_detected": 4}


def test_verify_pending_is_not_gated_by_the_duplicate_guard(monkeypatch, hermetic):
    """`--verify-pending` promotes reports STAGED by an earlier run; it detects
    and generates nothing. The hazard the guard names is fixed at staging
    time, and the graph's duplicate count now cannot tell "staged clean,
    duplicates arrived later" (harmless; a gate would refuse a legitimate
    recovery) from "staged over a fragmented graph, merged since" (the real
    hazard; the count is 0, so a gate would wave it through). So the guard is
    not consulted at all on this path -- not called-and-ignored, not called.
    A guard that raises is the sharpest probe: if a future edit re-adds the
    call, the refusal shows up as a non-zero exit and `calls == ["guard"]`."""
    calls: list[str] = []

    async def _fake_guard(driver, group_id, *, allow):
        calls.append("guard")
        raise DuplicateEntitiesError("2 exact-name duplicate entities")

    async def _fake_verify(settings, *, driver):
        calls.append("verify")
        return {"reports_promoted": 1}

    async def _fake_build(*a, **k):
        calls.append("build")
        return {}

    monkeypatch.setattr(theme_cli, "assert_no_duplicates", _fake_guard)
    monkeypatch.setattr(theme_cli, "_run_verify_pending", _fake_verify)
    for runner in ("_run_theme_build_incremental", "_run_theme_build"):
        monkeypatch.setattr(theme_cli, runner, _fake_build)
    result = CliRunner().invoke(theme_app, ["theme-build", "--verify-pending"])
    assert result.exit_code == 0, result.output
    assert calls == ["verify"], f"--verify-pending must not consult the guard; got {calls}"
    assert json.loads(result.stdout) == {"reports_promoted": 1}


def test_theme_build_help_offers_the_override():
    result = CliRunner().invoke(theme_app, ["theme-build", "--help"])
    assert result.exit_code == 0
    assert "--allow-duplicates" in result.stdout
