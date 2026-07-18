import pytest

pytestmark = pytest.mark.asyncio


class _FakeResult:
    def __init__(self, value):
        self._value = value
    async def single(self):
        return {"c": self._value}


class _FakeSession:
    def __init__(self, value):
        self._value = value
    async def __aenter__(self):
        return self
    async def __aexit__(self, *a):
        return False
    async def run(self, cypher, **kw):
        return _FakeResult(self._value)


class _FakeDriver:
    def __init__(self, values):
        self._values = list(values)
        self.session_calls = 0
    def session(self):
        self.session_calls += 1
        return _FakeSession(self._values.pop(0))


async def test_freshness_with_reports():
    from answer_api.freshness import freshness
    d = _FakeDriver(["2026-07-18T00:00:00Z", "2026-07-17T00:00:00Z"])
    out = await freshness(d, "backup-docs", reports=True)
    assert out["graph_cursor_time"] == "2026-07-18T00:00:00Z"
    assert out["reports_as_of"] == "2026-07-17T00:00:00Z"


async def test_freshness_without_reports_skips_community_query():
    from answer_api.freshness import freshness
    d = _FakeDriver(["2026-07-18T00:00:00Z"])   # only ONE value -> only one query allowed
    out = await freshness(d, "backup-docs", reports=False)
    assert out["graph_cursor_time"] == "2026-07-18T00:00:00Z"
    assert out["reports_as_of"] is None
    # regression-proof the gate: the community query must NOT run when reports=False
    # (this assertion lives outside freshness's broad except, so a gate regression
    # would surface here rather than being swallowed into a None).
    assert d.session_calls == 1


async def test_freshness_resilient_on_error():
    from answer_api.freshness import freshness

    class _Boom:
        def session(self):
            raise RuntimeError("neo4j down")

    out = await freshness(_Boom(), "g", reports=True)
    assert out == {"graph_cursor_time": None, "reports_as_of": None}
