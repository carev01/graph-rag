import asyncio

import pytest

from graph_sync import connectivity as c


@pytest.fixture(autouse=True)
def _no_real_settings(monkeypatch):
    """The unit conftest strips .env, so real settings cannot be built here."""
    monkeypatch.setattr(c, "_secret_values", lambda: [])


async def _ok() -> str:
    return "fine"


async def _boom() -> str:
    raise ConnectionError("refused at postgresql://u:hunter2@host/db")


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_failure_details_are_redacted_by_run_checks(monkeypatch):
    monkeypatch.setattr(c, "_secret_values", lambda: ["hunter2"])
    (r,) = await c.run_checks([("pg", _boom)])
    assert "hunter2" not in r.detail and "***" in r.detail


@pytest.mark.asyncio
async def test_optional_tier_with_empty_base_url_is_reported_ok_not_probed():
    (r,) = await c.run_checks([("judge", lambda: c._optional_http(""))])
    assert r.ok and r.detail == "not configured"


@pytest.mark.asyncio
async def test_optional_tier_with_a_base_url_is_probed_like_any_other(monkeypatch):
    async def _fake_http(url: str, verify: bool = True) -> str:
        return f"probed {url}"

    monkeypatch.setattr(c, "_http", _fake_http)
    (r,) = await c.run_checks([("judge", lambda: c._optional_http("http://x/v1"))])
    assert r.ok and r.detail == "probed http://x/v1/models"


@pytest.mark.asyncio
async def test_verify_tier_is_also_optional_and_skipped_when_unset():
    """`_checks()` wires the report/map/rerank/eval-judge/judge/verify tiers to
    this same `_optional_http`, so this is the same behavior under the name
    `verify` (`graph_extract.config.ExtractSettings.verify_llm_base_url`)."""
    (r,) = await c.run_checks([("verify", lambda: c._optional_http(""))])
    assert r.ok and r.detail == "not configured"


class _FakeResp:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def _fake_client(status: int, seen: list):
    class _Client:
        def __init__(self, **_kw) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a) -> None:
            return None

        async def get(self, url, headers=None):
            seen.append((url, headers))
            return _FakeResp(status)

    return _Client


@pytest.mark.asyncio
async def test_docextractor_checks_require_exactly_200(monkeypatch):
    seen: list = []
    monkeypatch.setattr(c.httpx, "AsyncClient", _fake_client(401, seen))
    (r,) = await c.run_checks([("docext-auth", lambda: c._http_200("http://d/api/articles?limit=1"))])
    assert not r.ok and "401" in r.detail


@pytest.mark.asyncio
async def test_authenticated_read_sends_the_key_in_a_header_and_never_prints_it(monkeypatch):
    key = "dxk_supersecretreadkey"
    monkeypatch.setattr(c, "_secret_values", lambda: [key])
    for status, ok in ((200, True), (403, False)):
        seen: list = []
        monkeypatch.setattr(c.httpx, "AsyncClient", _fake_client(status, seen))
        (r,) = await c.run_checks([("docext-auth", lambda: c._http_200(
            "http://d/api/articles?limit=1", headers={"X-API-Key": key}))])
        assert r.ok is ok
        assert seen == [("http://d/api/articles?limit=1", {"X-API-Key": key})]
        assert key not in r.detail and key not in seen[0][0]
        assert key not in c.report([r])[0]


def test_the_docextractor_auth_check_is_wired_with_the_read_key(monkeypatch):
    from types import SimpleNamespace
    fake = SimpleNamespace(
        docext_base_url="http://d/", docext_verify_tls=False, docext_read_key="k",
        neo4j_uri="", neo4j_user="", neo4j_password="", embed_base_url="",
        chonkie_base_url="", llm_base_url="", cheap_llm_base_url="", judge_base_url="",
        report_llm_base_url="", map_llm_base_url="", rerank_base_url="",
        eval_judge_base_url="", verify_llm_base_url="")
    monkeypatch.setattr(c, "get_extract_settings", lambda: fake)
    monkeypatch.setattr(c, "get_settings", lambda: SimpleNamespace(postgres_dsn=""))
    checks = dict(c._checks())
    assert "docextractor" in checks
    calls: list = []

    async def _rec(url, **kw):
        calls.append((url, kw))
        return "ok"

    monkeypatch.setattr(c, "_http_200", _rec)
    asyncio.run(checks["docextractor"]())
    asyncio.run(checks["docext-auth"]())
    assert calls[0][0] == "http://d/api/health"
    assert calls[1] == ("http://d/api/articles?limit=1",
                        {"verify": False, "headers": {"X-API-Key": "k"}})
