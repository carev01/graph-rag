import pytest
from httpx import ASGITransport, AsyncClient

import answer_api.app as app_mod
import answer_api.search as search_mod
import answer_api.synthesize as synth_mod
import answer_api.timeline as timeline_mod

pytestmark = pytest.mark.asyncio


class FakeGraphiti:
    async def close(self) -> None:
        pass


class FakeDriver:
    async def close(self) -> None:
        pass


class FakeSynthClient:
    async def close(self) -> None:
        pass


async def _fake_search_local(graphiti, driver, *, q, k=10, vendor=None,
                              include_invalid=False, group_id):
    return {
        "query": q,
        "count": 1,
        "results": [
            {
                "fact": "AWS Backup Vault Lock requires compliance mode",
                "fact_uuid": "f1",
                "valid_at": None,
                "sources": [{"article_id": "art1", "source_url": "https://x/art1"}],
            }
        ],
    }


async def _fake_build_driver(settings) -> FakeDriver:
    return FakeDriver()


def _fake_synthesis_client_and_model(settings):
    return FakeSynthClient(), "glm"


async def _fake_answer_local(graphiti, driver, synth_client, synth_model, *,
                              q, k=15, vendor=None, group_id):
    return {
        "query": q,
        "answer": "AWS Backup Vault Lock requires compliance mode [1].",
        "citations": [
            {
                "marker": 1,
                "fact": "AWS Backup Vault Lock requires compliance mode",
                "fact_uuid": "f1",
                "sources": [{"article_id": "art1", "source_url": "https://x/art1"}],
            }
        ],
        "retrieved": 1,
        "cited": 1,
    }


async def _fake_timeline_local(graphiti, driver, *, q, limit=30, vendor=None,
                                group_id):
    return {
        "query": q,
        "count": 1,
        "timeline": [
            {
                "fact": "AWS Backup Vault Lock requires compliance mode",
                "fact_uuid": "f1",
                "valid_at": "2024-01-01T00:00:00Z",
                "invalid_at": None,
                "status": "current",
                "sources": [{"article_id": "art1", "source_url": "https://x/art1"}],
            }
        ],
    }


@pytest.fixture(autouse=True)
def _stub_deps(monkeypatch):
    """Avoid building a real Graphiti/Neo4jDriver/synthesis client in the
    lifespan and avoid real search_local/answer_local/timeline_local calls so
    the endpoint tests need no Neo4j and no GLM endpoint."""
    monkeypatch.setattr(app_mod, "build_graphiti", lambda settings: FakeGraphiti())
    monkeypatch.setattr(app_mod, "_build_driver", _fake_build_driver)
    monkeypatch.setattr(search_mod, "search_local", _fake_search_local)
    monkeypatch.setattr(
        synth_mod, "_synthesis_client_and_model", _fake_synthesis_client_and_model
    )
    monkeypatch.setattr(synth_mod, "answer_local", _fake_answer_local)
    monkeypatch.setattr(timeline_mod, "timeline_local", _fake_timeline_local)


async def test_health():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_search_local_returns_stubbed_results():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/local", params={"q": "vault lock"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "vault lock"
    assert body["count"] == 1
    assert body["results"][0]["fact_uuid"] == "f1"


async def test_search_local_requires_q():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/local")
    assert resp.status_code == 422


async def test_answer_returns_stubbed_synthesis():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/answer", params={"q": "vault lock"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"query", "answer", "citations", "retrieved", "cited"}
    assert body["query"] == "vault lock"
    assert body["retrieved"] == 1
    assert body["cited"] == 1
    assert body["citations"][0]["fact_uuid"] == "f1"


async def test_answer_requires_q():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/answer")
    assert resp.status_code == 422


async def test_timeline_returns_stubbed_results():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/timeline", params={"q": "vault lock"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"query", "count", "timeline"}
    assert body["query"] == "vault lock"
    assert body["count"] == 1
    assert body["timeline"][0]["fact_uuid"] == "f1"
    assert body["timeline"][0]["status"] == "current"


async def test_timeline_requires_q():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/timeline")
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "path,params",
    [
        ("/timeline", {"q": "x", "limit": 0}),
        ("/timeline", {"q": "x", "limit": -1}),
        ("/search/local", {"q": "x", "k": 0}),
        ("/search/local", {"q": "x", "k": -3}),
        ("/answer", {"q": "x", "k": 0}),
    ],
)
async def test_nonpositive_limit_is_422(path, params):
    # limit/k <= 0 must be rejected, not silently drop the last edge (edges[:-1]).
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get(path, params=params)
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "path,params",
    [
        ("/timeline", {"q": "x", "limit": 1}),
        ("/search/local", {"q": "x", "k": 1}),
        ("/answer", {"q": "x", "k": 1}),
    ],
)
async def test_limit_one_is_allowed(path, params):
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get(path, params=params)
    assert resp.status_code == 200
