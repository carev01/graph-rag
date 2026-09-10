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


class FakeEmbedder:
    class _Client:
        async def close(self) -> None:
            pass

    def __init__(self) -> None:
        self.client = FakeEmbedder._Client()


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


async def _fake_answer_router(graphiti, driver, embedder, synth_client, synth_model,
                              map_client, map_model, cheap_client, cheap_model, *,
                              q, mode_override, vendor, settings):
    return {"mode": mode_override or "drift", "query": q, "answer": "routed answer [1].",
            "citations": [{"marker": 1, "fact_uuid": "f1",
                           "sources": [{"url": "https://x/art1", "title": "T", "article_id": "art1"}]}],
            "routing": {"chosen": mode_override or "drift", "via": "override" if mode_override else "default",
                        "fallback_from": None}}


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


async def _fake_global_search(driver, embedder, map_client, map_model, synth_client,
                              synth_model, *, q, level, k, group_id, settings):
    return {"query": q, "answer": "AWS and Azure both back up S3 [1].",
            "citations": [{"marker": 1, "fact_uuid": "f1",
                           "sources": [{"article_id": "art1", "source_url": "https://x/art1"}]}],
            "communities_used": [{"community_id": "c1", "title": "S3", "relevance": 0.9}]}


async def _fake_drift_search(graphiti, driver, embedder, synth_client, synth_model, *,
                             q, level, iterations, primer_k, max_followups, followup_k,
                             group_id, settings):
    return {"query": q, "answer": "cross-vendor DRIFT answer [1].",
            "citations": [{"marker": 1, "fact_uuid": "f1",
                           "sources": [{"url": "https://x/art1", "title": "T", "article_id": "art1"}]}],
            "follow_ups": [{"query": "how retained", "community_id": "c1", "iteration": 1}],
            "communities_used": [{"community_id": "c1", "title": "S3"}]}


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
    monkeypatch.setattr(timeline_mod, "timeline_local", _fake_timeline_local)
    import answer_api.global_search as global_mod
    monkeypatch.setattr(app_mod, "build_embedder", lambda s: FakeEmbedder())
    monkeypatch.setattr(global_mod, "_map_client_and_model",
                        lambda s: (FakeSynthClient(), "map-model"))
    monkeypatch.setattr(global_mod, "global_search", _fake_global_search)
    import answer_api.drift as drift_mod
    monkeypatch.setattr(drift_mod, "drift_search", _fake_drift_search)
    import answer_api.router as router_mod
    monkeypatch.setattr(router_mod, "answer_router", _fake_answer_router)
    monkeypatch.setattr(router_mod, "_cheap_classify_client", lambda s: (None, ""))


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


async def test_answer_router_returns_envelope():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/answer", params={"q": "plan retention"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "drift"
    assert body["routing"]["chosen"] == "drift"
    assert body["citations"][0]["fact_uuid"] == "f1"


async def test_answer_mode_override_forces_mode():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/answer", params={"q": "x", "mode": "timeline"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "timeline"


async def test_answer_invalid_mode_is_422():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/answer", params={"q": "x", "mode": "bogus"})
    assert resp.status_code == 422


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


async def test_global_returns_stubbed_answer():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/global", params={"q": "compare vendors on S3"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"query", "answer", "citations", "communities_used"}
    assert body["citations"][0]["fact_uuid"] == "f1"


async def test_global_requires_q():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/global")
    assert resp.status_code == 422


async def test_drift_returns_stubbed_answer():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/drift", params={"q": "plan retention across vendors"})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"query", "answer", "citations", "follow_ups", "communities_used"}
    assert body["citations"][0]["fact_uuid"] == "f1"


async def test_drift_requires_q():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/drift")
    assert resp.status_code == 422


async def test_drift_iterations_over_two_is_422():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/drift", params={"q": "x", "iterations": 3})
    assert resp.status_code == 422


@pytest.mark.parametrize(
    "path,params",
    [
        ("/timeline", {"q": "x", "limit": 0}),
        ("/timeline", {"q": "x", "limit": -1}),
        ("/search/local", {"q": "x", "k": 0}),
        ("/search/local", {"q": "x", "k": -3}),
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
    ],
)
async def test_limit_one_is_allowed(path, params):
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get(path, params=params)
    assert resp.status_code == 200
