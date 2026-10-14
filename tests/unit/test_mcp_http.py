"""/mcp over HTTP: bearer token, fail-closed mount, stateless transport (spec §3-4)."""
import httpx
import pytest

import answer_api.app as app_mod
from answer_api.mcp_auth import BearerTokenGate

pytestmark = pytest.mark.asyncio

_INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "t", "version": "0"}}}
_HDRS = {"accept": "application/json, text/event-stream",
         "content-type": "application/json", "host": "graphrag.k3s.home.lan"}


def _settings(monkeypatch, token):
    from graph_extract.config import ExtractSettings
    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                        mcp_bearer_token=token)
    monkeypatch.setattr(app_mod, "get_extract_settings", lambda: s)
    return s


async def _post(app, headers):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://graphrag.k3s.home.lan") as c:
        return await c.post("/mcp", json=_INIT, headers=headers)


async def test_no_token_configured_means_no_mcp_route(monkeypatch):
    _settings(monkeypatch, "")
    app = app_mod.create_app()
    assert app.state.mcp_server is None
    assert (await _post(app, {**_HDRS, "authorization": "Bearer x"})).status_code == 404


@pytest.mark.parametrize("auth", [None, "Bearer wrong", "Basic c2VjcmV0", "Bearer"])
async def test_missing_or_wrong_token_is_401(monkeypatch, auth):
    _settings(monkeypatch, "s3cret-token")
    app = app_mod.create_app()
    headers = dict(_HDRS) if auth is None else {**_HDRS, "authorization": auth}
    r = await _post(app, headers)
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"


async def test_right_token_reaches_the_mcp_server(monkeypatch):
    _settings(monkeypatch, "s3cret-token")
    app = app_mod.create_app()
    async with app.state.mcp_server.session_manager.run():
        r = await _post(app, {**_HDRS, "authorization": "Bearer s3cret-token"})
    assert r.status_code == 200
    assert "graph-rag" in r.text                 # serverInfo.name in the initialize reply


async def test_a_foreign_host_header_is_rejected(monkeypatch):
    _settings(monkeypatch, "s3cret-token")
    app = app_mod.create_app()
    async with app.state.mcp_server.session_manager.run():
        r = await _post(app, {**_HDRS, "host": "evil.example",
                              "authorization": "Bearer s3cret-token"})
    assert r.status_code == 421


async def test_rest_routes_do_not_need_the_token(monkeypatch):
    _settings(monkeypatch, "s3cret-token")
    app = app_mod.create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://graphrag.k3s.home.lan") as c:
        assert (await c.get("/health")).status_code == 200


async def test_gate_passes_other_paths_and_non_http_scopes():
    seen = []

    async def inner(scope, receive, send):
        seen.append(scope["path"] if scope["type"] == "http" else scope["type"])

    gate = BearerTokenGate(inner, "t")
    await gate({"type": "http", "path": "/other", "headers": []}, None, None)
    await gate({"type": "lifespan"}, None, None)
    assert seen == ["/other", "lifespan"]


async def _post_under_root_path(app, headers):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, root_path="/api"),
                                 base_url="http://graphrag.k3s.home.lan") as c:
        return await c.post("/api/mcp", json=_INIT, headers=headers)


async def test_a_root_path_does_not_bypass_the_gate(monkeypatch):
    _settings(monkeypatch, "s3cret-token")
    app = app_mod.create_app()
    async with app.state.mcp_server.session_manager.run():
        denied = await _post_under_root_path(app, dict(_HDRS))
        allowed = await _post_under_root_path(
            app, {**_HDRS, "authorization": "Bearer s3cret-token"})
    assert denied.status_code == 401
    assert denied.headers["www-authenticate"] == "Bearer"
    assert allowed.status_code == 200
    assert "graph-rag" in allowed.text
