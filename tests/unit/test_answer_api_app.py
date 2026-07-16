import pytest
from httpx import ASGITransport, AsyncClient

import answer_api.app as app_mod
import answer_api.search as search_mod

pytestmark = pytest.mark.asyncio


class FakeGraphiti:
    async def close(self) -> None:
        pass


class FakeDriver:
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


@pytest.fixture(autouse=True)
def _stub_deps(monkeypatch):
    """Avoid building a real Graphiti/Neo4jDriver in the lifespan and avoid a
    real search_local so the endpoint test needs no Neo4j."""
    monkeypatch.setattr(app_mod, "build_graphiti", lambda settings: FakeGraphiti())
    monkeypatch.setattr(app_mod, "_build_driver", _fake_build_driver)
    monkeypatch.setattr(search_mod, "search_local", _fake_search_local)


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
