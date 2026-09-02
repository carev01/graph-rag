import pytest

from compat import runner
from compat.model import CallableCheck, CheckContext, CypherCheck, SkipCheck
from graph_extract.config import ExtractSettings


def _settings(**kw):
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://main", neo4j_user="mu", neo4j_password="mp")
    base.update(kw)
    return ExtractSettings(**base)


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        async def gen():
            for row in self._rows:
                yield row
        return gen()


class _FakeSession:
    """Records every statement; raises for statements listed in `fail_on`."""

    def __init__(self, calls, fail_on, rows):
        self.calls, self.fail_on, self.rows = calls, fail_on, rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, cypher, **params):
        self.calls.append(cypher)
        for needle in self.fail_on:
            if cypher.startswith(needle):
                raise RuntimeError("SyntaxError: nope")
        return _FakeResult(self.rows)


class _FakeDriver:
    def __init__(self, fail_on=(), rows=()):
        self.calls: list[str] = []
        self.fail_on, self.rows = fail_on, list(rows)

    def session(self):
        return _FakeSession(self.calls, self.fail_on, self.rows)


def test_compat_target_falls_back_to_main_neo4j_settings():
    assert runner.compat_target(_settings()) == ("bolt://main", "mu", "mp")


def test_compat_target_prefers_compat_overrides():
    s = _settings(compat_neo4j_uri="bolt://new", compat_neo4j_user="cu",
                  compat_neo4j_password="cp")
    assert runner.compat_target(s) == ("bolt://new", "cu", "cp")


def test_compat_target_falls_back_per_field():
    s = _settings(compat_neo4j_uri="bolt://new")
    assert runner.compat_target(s) == ("bolt://new", "mu", "mp")


def test_fabricate_embedding_has_requested_dim_and_is_deterministic():
    a, b = runner.fabricate_embedding(768), runner.fabricate_embedding(768)
    assert len(a) == 768
    assert a == b


@pytest.mark.asyncio
async def test_cypher_check_passes_when_statement_runs():
    d = _FakeDriver(rows=[{"v": 1}])
    res = await runner.run_cypher_check(d, CypherCheck("c", "g", "RETURN 1 AS v"))
    assert res.status == "pass"
    assert res.cypher5_retry is None


@pytest.mark.asyncio
async def test_cypher_check_fails_when_expect_predicate_rejects_rows():
    d = _FakeDriver(rows=[{"v": 0}])
    check = CypherCheck("c", "g", "RETURN 0 AS v", expect=lambda rows: rows[0]["v"] == 1)
    res = await runner.run_cypher_check(d, check)
    assert res.status == "fail"
    assert "expect" in res.detail.lower()


@pytest.mark.asyncio
async def test_cypher_check_retries_failure_under_cypher_5():
    d = _FakeDriver(fail_on=["MATCH"], rows=[{"v": 1}])
    res = await runner.run_cypher_check(d, CypherCheck("c", "g", "MATCH (n) RETURN 1 AS v"))
    assert res.status == "fail"
    assert res.cypher5_retry == "pass"
    assert d.calls[-1].startswith("CYPHER 5 MATCH")


@pytest.mark.asyncio
async def test_cypher_check_records_retry_failure():
    d = _FakeDriver(fail_on=["MATCH", "CYPHER 5 MATCH"])
    res = await runner.run_cypher_check(d, CypherCheck("c", "g", "MATCH (n) RETURN 1"))
    assert res.status == "fail"
    assert res.cypher5_retry == "fail"


@pytest.mark.asyncio
async def test_callable_check_skip_and_failure_are_distinguished():
    ctx = CheckContext(driver=None, graphiti=None, settings=_settings(), embedding=[0.1])

    async def skips(_):
        raise SkipCheck("embedder unreachable")

    async def boom(_):
        raise RuntimeError("kaboom")

    async def ok(_):
        return "all good"

    assert (await runner.run_callable_check(ctx, CallableCheck("s", "g", skips))).status == "skip"
    failed = await runner.run_callable_check(ctx, CallableCheck("b", "g", boom))
    assert failed.status == "fail"
    assert failed.cypher5_retry is None
    assert "kaboom" in failed.detail
    passed = await runner.run_callable_check(ctx, CallableCheck("o", "g", ok))
    assert passed.status == "pass"
    assert passed.detail == "all good"


@pytest.mark.asyncio
async def test_run_all_contains_failures_and_keeps_going():
    ctx = CheckContext(driver=_FakeDriver(rows=[{"v": 1}]), graphiti=None,
                       settings=_settings(), embedding=[0.1])

    async def boom(_):
        raise RuntimeError("kaboom")

    checks = [
        CallableCheck("first", "g", boom),
        CypherCheck("second", "g", "RETURN 1 AS v"),
    ]
    results = await runner.run_all(ctx, checks)
    assert [r.status for r in results] == ["fail", "pass"]


@pytest.mark.asyncio
async def test_run_all_preserves_informational_flag():
    ctx = CheckContext(driver=_FakeDriver(rows=[{"v": 1}]), graphiti=None,
                       settings=_settings(), embedding=[0.1])
    results = await runner.run_all(
        ctx, [CypherCheck("i", "g", "RETURN 1 AS v", informational=True)])
    assert results[0].informational is True


@pytest.mark.asyncio
async def test_teardown_scopes_deletes_to_the_compat_group_and_reports_errors():
    d = _FakeDriver()
    assert await runner.teardown(d) is None
    assert any("compat-check" in c for c in d.calls)
    assert not any(c.strip().startswith("MATCH (n) DETACH DELETE") for c in d.calls)

    boom = _FakeDriver(fail_on=["MATCH"])
    err = await runner.teardown(boom)
    assert err is not None and "nope" in err
