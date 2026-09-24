import pytest

from graph_sync import connectivity as c

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _no_real_settings(monkeypatch):
    """The unit conftest strips .env, so real settings cannot be built here."""
    monkeypatch.setattr(c, "_secret_values", lambda: [])


async def _ok() -> str:
    return "fine"


async def _boom() -> str:
    raise ConnectionError("refused at postgresql://u:hunter2@host/db")


async def test_run_checks_records_success_and_failure_without_raising():
    results = await c.run_checks([("a", _ok), ("b", _boom)])
    assert [(r.name, r.ok) for r in results] == [("a", True), ("b", False)]
    assert results[0].detail == "fine"
    assert "ConnectionError" in results[1].detail


def test_report_exit_code_is_nonzero_when_anything_failed():
    text, code = c.report([c.CheckResult("a", True, "x"), c.CheckResult("b", False, "y")])
    assert code == 1 and "FAIL" in text and "b" in text
    assert c.report([c.CheckResult("a", True, "x")])[1] == 0


def test_a_dsn_contributes_its_password_as_a_separate_secret():
    assert "hunter2" in c._dsn_passwords(["postgresql://u:hunter2@host:5432/db", "plain"])


def test_redact_removes_every_secret_value():
    assert c.redact("postgresql://u:hunter2@host/db key=sk-abc", ["hunter2", "sk-abc", ""]) \
        == "postgresql://u:***@host/db key=***"


async def test_failure_details_are_redacted_by_run_checks(monkeypatch):
    monkeypatch.setattr(c, "_secret_values", lambda: ["hunter2"])
    (r,) = await c.run_checks([("pg", _boom)])
    assert "hunter2" not in r.detail and "***" in r.detail
