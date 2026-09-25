import asyncio
import logging

import pytest
from httpx import ASGITransport, AsyncClient

import answer_api.app as app_mod
import answer_api.search as search_mod
import answer_api.synthesize as synth_mod
import answer_api.timeline as timeline_mod
from answer_api.scope import ScopeResolver

pytestmark = pytest.mark.asyncio

_VENDORS = ["Veeam", "AWS"]
_PRODUCTS = [("Veeam Backup & Replication", "Veeam"), ("AWS Backup", "AWS")]


async def _fake_scope_resolver_load(driver, aliases_path=None):
    return ScopeResolver(_VENDORS, _PRODUCTS, {})


class FakeGraphiti:
    driver = None

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


async def _fake_search_local(graphiti, driver, *, q, k=10, scope=None,
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
                              q, mode_override, scope, settings):
    return {"mode": mode_override or "drift", "query": q, "answer": "routed answer [1].",
            "citations": [{"marker": 1, "fact_uuid": "f1",
                           "sources": [{"url": "https://x/art1", "title": "T", "article_id": "art1"}]}],
            "routing": {"chosen": mode_override or "drift", "via": "override" if mode_override else "default",
                        "fallback_from": None},
            "scope": scope.as_dict()}


async def _fake_timeline_local(graphiti, driver, *, q, limit=30, scope=None,
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
                              synth_model, *, q, level, k, group_id, settings,
                              scope=None):
    return {"query": q, "answer": "AWS and Azure both back up S3 [1].",
            "citations": [{"marker": 1, "fact_uuid": "f1",
                           "sources": [{"article_id": "art1", "source_url": "https://x/art1"}]}],
            "communities_used": [{"community_id": "c1", "title": "S3", "relevance": 0.9}]}


async def _fake_drift_search(graphiti, driver, embedder, synth_client, synth_model, *,
                             q, level, iterations, primer_k, max_followups, followup_k,
                             group_id, settings, scope=None):
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
    monkeypatch.setattr(app_mod.ScopeResolver, "load", _fake_scope_resolver_load)

    async def _fake_ensure_vector_indexes(driver, embed_dim, **_kwargs):
        return None

    monkeypatch.setattr(app_mod, "ensure_vector_indexes", _fake_ensure_vector_indexes)
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


async def test_lifespan_waits_for_vector_indexes_with_the_startup_budget(monkeypatch):
    from graph_extract.config import ExtractSettings

    settings = ExtractSettings(
        _env_file=None, neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
        docext_base_url="http://x", docext_read_key="k", embed_dim=8,
        vector_search_enabled=True, vector_index_startup_wait_seconds=7.5)
    monkeypatch.setattr(app_mod, "get_extract_settings", lambda: settings)
    calls: list[tuple[int, dict]] = []

    async def _ensure(driver, embed_dim, **kwargs):
        calls.append((embed_dim, kwargs))

    monkeypatch.setattr(app_mod, "ensure_vector_indexes", _ensure)
    app = app_mod.create_app()
    async with app.router.lifespan_context(app):
        pass
    assert calls == [(8, {"wait_seconds": 7.5})]


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


async def test_search_local_vendor_param_converts_to_scope(monkeypatch):
    """/search/local?vendor=aws must reach search_local with a resolved Scope,
    not a dropped/raw vendor string -- the app-side half of R3's conversion,
    now resolved (and canonicalised) via the ScopeResolver rather than built
    from the raw query string."""
    captured: dict = {}

    async def _capture_search_local(graphiti, driver, *, q, k=10, scope=None,
                                    include_invalid=False, group_id):
        captured["scope"] = scope
        return {"query": q, "count": 0, "results": []}

    monkeypatch.setattr(search_mod, "search_local", _capture_search_local)
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/local", params={"q": "x", "vendor": "aws"})
    assert resp.status_code == 200
    assert captured["scope"].vendors == ("AWS",)
    assert captured["scope"].source == "explicit"


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
    assert set(body.keys()) == {"query", "count", "timeline", "scope"}
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
    assert set(body.keys()) == {"query", "answer", "citations", "communities_used", "scope"}
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
    assert set(body.keys()) == {"query", "answer", "citations", "follow_ups", "communities_used", "scope"}
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


async def test_answer_vendor_and_product_params_reach_router_resolved():
    """A repeated vendor= plus a product= param resolve (canonicalised) via the
    ScopeResolver and reach the router as a Scope, not dropped raw strings."""
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/answer", params=[
                ("q", "x"), ("vendor", "veeam"), ("vendor", "aws"),
                ("product", "AWS Backup")])
    assert resp.status_code == 200
    assert resp.json()["scope"] == {
        "vendors": ["Veeam", "AWS"], "products": ["AWS Backup"], "source": "explicit"}


async def test_scope_none_disables_detection():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/answer", params={"q": "Veeam retention", "scope": "none"})
    assert resp.status_code == 200
    assert resp.json()["scope"] == {"vendors": [], "products": [], "source": "none"}


async def test_unknown_vendor_name_is_422_with_unknown_names():
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/answer", params={"q": "x", "vendor": "NoSuchVendor"})
    assert resp.status_code == 422
    assert resp.json()["detail"] == {"unknown": ["NoSuchVendor"]}


async def test_search_local_scope_none_and_unknown_name_too():
    """The same scope resolution/validation applies on /search/local, not just
    /answer."""
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            resp = await c.get("/search/local", params={"q": "x", "vendor": "NoSuchVendor"})
    assert resp.status_code == 422
    assert resp.json()["detail"] == {"unknown": ["NoSuchVendor"]}


async def test_scope_resolver_reloads_after_ttl(monkeypatch):
    from graph_extract.config import ExtractSettings

    settings = ExtractSettings(
        _env_file=None, neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
        docext_base_url="http://x", docext_read_key="k", scope_reload_seconds=100)
    monkeypatch.setattr(app_mod, "get_extract_settings", lambda: settings)

    load_calls = {"n": 0}

    async def _counting_load(driver, aliases_path=None):
        load_calls["n"] += 1
        return ScopeResolver(_VENDORS, _PRODUCTS, {})

    monkeypatch.setattr(app_mod.ScopeResolver, "load", _counting_load)

    clock = {"t": 1000.0}
    monkeypatch.setattr(app_mod.time, "monotonic", lambda: clock["t"])

    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            assert load_calls["n"] == 1                # lifespan load
            resp1 = await c.get("/answer", params={"q": "x"})
            assert load_calls["n"] == 1                 # still fresh
            clock["t"] += 101                            # past scope_reload_seconds=100
            resp2 = await c.get("/answer", params={"q": "x"})
            assert load_calls["n"] == 2                 # stale beyond TTL -> reload
    assert resp1.status_code == 200 and resp2.status_code == 200


async def test_scope_resolver_reload_failure_keeps_serving_old_resolver_and_backs_off(
        monkeypatch, caplog):
    """R6: a reload failure (e.g. a Neo4j blip) after the TTL must not 500 the
    request, must not wedge every later request into retrying-and-failing, and
    must log loudly rather than silently. The previous resolver keeps serving,
    and the next reload attempt is deferred 60s rather than retried on every
    request until it happens to succeed."""
    from graph_extract.config import ExtractSettings

    settings = ExtractSettings(
        _env_file=None, neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
        docext_base_url="http://x", docext_read_key="k", scope_reload_seconds=100)
    monkeypatch.setattr(app_mod, "get_extract_settings", lambda: settings)

    load_calls = {"n": 0}

    async def _failing_load(driver, aliases_path=None):
        load_calls["n"] += 1
        raise RuntimeError("neo4j blip")

    clock = {"t": 1000.0}
    monkeypatch.setattr(app_mod.time, "monotonic", lambda: clock["t"])

    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            # Past the TTL, and the resolver is now unreachable.
            clock["t"] = 1101.0
            monkeypatch.setattr(app_mod.ScopeResolver, "load", _failing_load)

            with caplog.at_level(logging.WARNING, logger="answer_api.app"):
                resp1 = await c.get("/answer", params={"q": "x"})
            assert resp1.status_code == 200                    # old resolver still served
            assert load_calls["n"] == 1
            assert any(r.levelno == logging.WARNING for r in caplog.records)
            assert any(r.exc_info for r in caplog.records)     # exc_info=True, not swallowed

            resp2 = await c.get("/answer", params={"q": "x"})   # immediate retry
            assert resp2.status_code == 200
            assert load_calls["n"] == 1                          # backed off, not retried

            clock["t"] += 61                                     # past the 60s backoff
            resp3 = await c.get("/answer", params={"q": "x"})
            assert resp3.status_code == 200
            assert load_calls["n"] == 2                          # retried after the backoff


async def test_concurrent_stale_requests_reload_the_resolver_once(monkeypatch):
    """The minor finding: two requests landing past the TTL concurrently must
    not both call `ScopeResolver.load` -- single-flight via a lock, with the
    second re-checking staleness inside the lock rather than reloading again."""
    from graph_extract.config import ExtractSettings

    settings = ExtractSettings(
        _env_file=None, neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
        docext_base_url="http://x", docext_read_key="k", scope_reload_seconds=100)
    monkeypatch.setattr(app_mod, "get_extract_settings", lambda: settings)

    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            clock = {"t": 1101.0}          # frozen, stale relative to loaded_at below
            monkeypatch.setattr(app_mod.time, "monotonic", lambda: clock["t"])
            app.state.scope_loaded_at = 1000.0

            load_calls = {"n": 0}
            started = asyncio.Event()
            release = asyncio.Event()

            async def _slow_load(driver, aliases_path=None):
                load_calls["n"] += 1
                started.set()
                await release.wait()
                return ScopeResolver(_VENDORS, _PRODUCTS, {})

            monkeypatch.setattr(app_mod.ScopeResolver, "load", _slow_load)

            async def _get():
                return await c.get("/answer", params={"q": "x"})

            t1 = asyncio.create_task(_get())
            await started.wait()           # t1 is inside the lock, blocked on release
            t2 = asyncio.create_task(_get())
            # Let t2 run forward to (and block on) the lock. A real delay
            # (asyncio.sleep(n>0)) would need the loop's own clock -- which is
            # time.monotonic(), frozen by this test -- to advance, so it would
            # hang forever; sleep(0) just yields to the next loop iteration.
            for _ in range(50):
                await asyncio.sleep(0)
            release.set()
            r1, r2 = await asyncio.gather(t1, t2)
    assert r1.status_code == 200 and r2.status_code == 200
    assert load_calls["n"] == 1
