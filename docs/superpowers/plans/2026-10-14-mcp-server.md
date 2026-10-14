# MCP Server for answer-api Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve three MCP tools (`answer_question`, `search_facts`, `list_vendors_products`) at `/mcp` on answer-api, behind a bearer token, so the user can query the graph from Claude Code on the LAN.

**Architecture:** An `MCPServer` (official MCP Python SDK 2.x) built in `src/answer_api/mcp_server.py`, its stateless streamable-HTTP Starlette app mounted inside the existing FastAPI app behind a small bearer-token ASGI wrapper; the SDK session manager runs in answer-api's lifespan. Tools call the same functions the REST routes call (`run_answer`, `run_search_local`, extracted from the route bodies).

**Tech Stack:** Python 3.12, FastAPI/Starlette, `mcp` 2.3 (`MCPServer`, `Context`, `ToolError`, `Client`), pytest + pytest-asyncio, uv.

**Spec:** `docs/superpowers/specs/2026-10-14-mcp-server-design.md`

## Global Constraints

- Dependency: `mcp>=2.3,<3` (2.x renamed FastMCP to `mcp.server.mcpserver.MCPServer`; v1 examples do not apply).
- Verified SDK surface (mcp 2.3.0): `from mcp.server.mcpserver import MCPServer, Context`; `from mcp.server.mcpserver.exceptions import ToolError`; `from mcp.server.transport_security import TransportSecuritySettings`; `from mcp.types import CallToolResult, TextContent, ToolAnnotations`; `from mcp import Client`. `@server.tool(name=, description=, annotations=ToolAnnotations(read_only_hint=True))`; a `ctx: Context` parameter is injected and omitted from the input schema; `await ctx.report_progress(progress, total, message)` is a no-op when the client sent no progress token; returning a `CallToolResult` passes through unchanged; raising `ToolError("msg")` returns `is_error=True` with text `Error executing tool <name>: msg`; any other exception returns only `Error executing tool <name>`. `server.streamable_http_app(streamable_http_path="/mcp", stateless_http=True, transport_security=...)` returns a Starlette app whose own lifespan (`server.session_manager.run()`) does NOT run when mounted — answer-api's lifespan must enter it. `Client(server)` connects in-process; `client.call_tool(name, args, progress_callback=cb)` with `cb(progress, total, message)`.
- Design decision #2: the LLM never writes a URL. Sources text is built only from citation `sources` (`{url, title, article_id, section, vendor, product}`).
- A failure never reads as an answer: errors become `ToolError` (message says what failed) and are logged; a refusal or partial answer is a normal result.
- `/mcp` requires `Authorization: Bearer <MCP_BEARER_TOKEN>` (constant-time compare, 401 + `WWW-Authenticate: Bearer`); with the token unset `/mcp` is not mounted (404). REST routes unchanged.
- No secret in the repo, image or chart: the token lives only in the cluster Secret `graph-rag-secret`.
- Lint gate is `uv run ruff check src tests` (E702: no semicolons in test fakes; E402: imports at top). Types: `uv run mypy src`. Tests: `uv run --extra dev pytest -m "not live"`. Unit tests use `ExtractSettings(_env_file=None, ...)` and never build settings at collection time.
- No paid calls in tests; every test uses fakes.

---

## File Structure

- Modify `pyproject.toml` / `uv.lock` — add `mcp`.
- Modify `src/graph_extract/config.py` — `mcp_bearer_token`, `mcp_allowed_hosts`, `mcp_progress_seconds`.
- Modify `src/answer_api/app.py` — extract `run_answer` / `run_search_local`; mount MCP behind auth; enter the session manager in `_lifespan`.
- Create `src/answer_api/mcp_server.py` — `build_mcp_server(app)`, the three tools, text formatting, catalog query, progress heartbeat.
- Create `src/answer_api/mcp_auth.py` — `BearerTokenGate` ASGI wrapper.
- Create `tests/unit/test_mcp_tools.py`, `tests/unit/test_mcp_http.py`.
- Create `docs/deploy/mcp.md`; modify `CLAUDE.md`, `docs/superpowers/BACKLOG.md`.

---

### Task 1: Dependency and settings

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Modify: `src/graph_extract/config.py` (next to the other answer-api settings)
- Test: `tests/unit/test_config.py`

**Interfaces:**
- Produces: `ExtractSettings.mcp_bearer_token: str = ""`, `ExtractSettings.mcp_allowed_hosts: list[str]`, `ExtractSettings.mcp_progress_seconds: float = 10.0`.

- [ ] **Step 1: Add the dependency**

Run: `uv add "mcp>=2.3,<3"`
Expected: `pyproject.toml` gains `"mcp>=2.3,<3"` in `dependencies`; `uv.lock` updated; `uv run python -c "from mcp.server.mcpserver import MCPServer"` prints nothing and exits 0.

- [ ] **Step 2: Write the failing test** (append to `tests/unit/test_config.py`)

```python
def test_mcp_defaults_fail_closed():
    s = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
    assert s.mcp_bearer_token == ""              # unset -> /mcp not served
    assert s.mcp_allowed_hosts == ["graphrag.k3s.home.lan", "localhost:*", "127.0.0.1:*"]
    assert s.mcp_progress_seconds == 10.0
```

(If `ExtractSettings` is imported under another name in that file, use the file's existing import.)

- [ ] **Step 3: Run it to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_config.py::test_mcp_defaults_fail_closed -q`
Expected: FAIL (`AttributeError` / no attribute `mcp_bearer_token`).

- [ ] **Step 4: Add the settings** in `src/graph_extract/config.py`, inside `ExtractSettings`, near the answer-api settings:

```python
    # --- MCP server (spec 2026-10-14) ---
    # Bearer token required on /mcp. Empty (the default) leaves /mcp unmounted: a
    # deployment without the token never exposes the MCP endpoint. Set only in the
    # cluster Secret, never in the repo, image or chart values.
    mcp_bearer_token: str = ""
    # Host header values the MCP transport accepts (DNS-rebinding protection): the
    # ingress host, plus localhost for a kubectl port-forward.
    mcp_allowed_hosts: list[str] = ["graphrag.k3s.home.lan", "localhost:*", "127.0.0.1:*"]
    # Seconds between progress notifications while answer_question runs (comparisons
    # and enumerations take 1-7 minutes; clients time out silent calls).
    mcp_progress_seconds: float = 10.0
```

- [ ] **Step 5: Run the test and the config suite**

Run: `uv run --extra dev pytest tests/unit/test_config.py -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/graph_extract/config.py tests/unit/test_config.py
git commit -m "feat(answer-api): mcp dependency and MCP settings (fail closed without a token)"
```

---

### Task 2: Extract the `/answer` and `/search/local` bodies into shared functions

**Files:**
- Modify: `src/answer_api/app.py` (the `search_local` and `answer` route handlers inside `create_app`)
- Test: `tests/unit/test_answer_api_app.py` (existing; add one test)

**Interfaces:**
- Consumes: existing `_resolve_scope(app, q, vendor, product, scope_mode, source)`, `classify_shape`, `retrieval_scope`, `router_mod.answer_router`, `search_mod.search_local`.
- Produces (module-level in `app.py`, used by Task 3):
  - `async def run_answer(app: FastAPI, *, q: str, mode: Mode | None = None, vendor: list[str] | None = None, product: list[str] | None = None, source: list[str] | None = None, scope: Literal["auto", "none"] = "auto") -> dict[str, Any]`
  - `async def run_search_local(app: FastAPI, *, q: str, k: int = 10, vendor: list[str] | None = None, product: list[str] | None = None, source: list[str] | None = None, scope: Literal["auto", "none"] = "auto", include_invalid: bool = False) -> dict[str, Any]`
  - Both raise `HTTPException(422, detail={"unknown": [...]})` on unknown names, exactly as today.

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_answer_api_app.py`; the file's autouse `_stub_deps` fixture stubs every backend, so `create_app()` + `lifespan_context` is all a test needs)

```python
async def test_run_answer_and_run_search_local_match_their_routes():
    # The REST routes are thin wrappers over run_answer / run_search_local (spec
    # 2026-10-14 §1); the MCP tools call the same functions.
    app = app_mod.create_app()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        async with app.router.lifespan_context(app):
            via_answer = (await c.get("/answer", params={"q": "plan retention"})).json()
            via_search = (await c.get("/search/local", params={"q": "plan retention"})).json()
            direct_answer = await app_mod.run_answer(app, q="plan retention")
            direct_search = await app_mod.run_search_local(app, q="plan retention")
    assert direct_answer == via_answer
    assert direct_search == via_search
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --extra dev pytest tests/unit/test_answer_api_app.py -k run_answer -q`
Expected: FAIL (`ImportError: cannot import name 'run_answer'`).

- [ ] **Step 3: Implement** — in `app.py`, add these module-level functions (above `create_app`) and make the routes call them:

```python
async def run_answer(app: FastAPI, *, q: str, mode: Mode | None = None,
                     vendor: list[str] | None = None, product: list[str] | None = None,
                     source: list[str] | None = None,
                     scope: Literal["auto", "none"] = "auto") -> dict[str, Any]:
    """What GET /answer runs; shared with the MCP `answer_question` tool."""
    st = app.state
    resolved = await _resolve_scope(app, q, vendor, product, scope, source)
    shape = classify_shape(q, st.scope_resolver, resolved,
                           scope_disabled=(scope == "none"))
    resolved = retrieval_scope(shape, resolved)
    return await router_mod.answer_router(
        st.graphiti, st.driver, st.embedder, st.synth_client, st.synth_model,
        st.map_client, st.map_model, st.cheap_client, st.cheap_model,
        q=q, mode_override=mode, scope=resolved, settings=st.settings,
        shape=shape)


async def run_search_local(app: FastAPI, *, q: str, k: int = 10,
                           vendor: list[str] | None = None,
                           product: list[str] | None = None,
                           source: list[str] | None = None,
                           scope: Literal["auto", "none"] = "auto",
                           include_invalid: bool = False) -> dict[str, Any]:
    """What GET /search/local runs; shared with the MCP `search_facts` tool."""
    resolved = await _resolve_scope(app, q, vendor, product, scope, source)
    result = await search_mod.search_local(
        app.state.graphiti, app.state.driver, q=q, k=k, scope=resolved,
        include_invalid=include_invalid, group_id=app.state.settings.group_id,
        fetch_cap=app.state.settings.local_scoped_fetch_cap,
        scan_max_episodes=app.state.settings.local_scoped_scan_max_episodes,
        settings=app.state.settings)
    result["scope"] = resolved.as_dict()
    return result
```

Route bodies become:

```python
    @app.get("/search/local")
    async def search_local(
        q: str, k: int = Query(10, ge=1),
        vendor: list[str] | None = Query(None), product: list[str] | None = Query(None),
        source: list[str] | None = Query(None),
        scope: Literal["auto", "none"] = "auto",
        include_invalid: bool = False
    ) -> dict[str, Any]:
        return await run_search_local(app, q=q, k=k, vendor=vendor, product=product,
                                      source=source, scope=scope,
                                      include_invalid=include_invalid)

    @app.get("/answer")
    async def answer(
        q: str, mode: Mode | None = None,
        vendor: list[str] | None = Query(None), product: list[str] | None = Query(None),
        source: list[str] | None = Query(None),
        scope: Literal["auto", "none"] = "auto",
    ) -> dict[str, Any]:
        return await run_answer(app, q=q, mode=mode, vendor=vendor, product=product,
                                source=source, scope=scope)
```

- [ ] **Step 4: Run the answer-api unit tests**

Run: `uv run --extra dev pytest tests/unit/test_answer_api_app.py tests/unit -q -k "answer or search_local or scope"`
Expected: all PASS (existing REST tests unchanged).

- [ ] **Step 5: Lint, types, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/answer_api/app.py tests/unit/test_answer_api_app.py
git commit -m "refactor(answer-api): /answer and /search/local bodies as run_answer / run_search_local"
```

---

### Task 3: The MCP server and its three tools

**Files:**
- Create: `src/answer_api/mcp_server.py`
- Test: `tests/unit/test_mcp_tools.py`

**Interfaces:**
- Consumes: `run_answer`, `run_search_local` (Task 2); `ExtractSettings.mcp_progress_seconds` (Task 1); `app.state.driver`, `app.state.scope_resolver`, `app.state.settings`.
- Produces:
  - `build_mcp_server(app: FastAPI) -> MCPServer` — the server with tools `answer_question`, `search_facts`, `list_vendors_products`.
  - `format_answer_text(env: dict) -> str`, `format_facts_text(result: dict) -> str` (pure, tested).
  - `CATALOG_QUERY: str` (Cypher).

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_mcp_tools.py`:

```python
"""MCP tools over answer-api (spec 2026-10-14 §2, §5), in-process client, fake state."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from mcp import Client

import answer_api.mcp_server as ms

pytestmark = pytest.mark.asyncio

_SRC = {"url": "https://docs.example/a", "title": "Hardened repository", "article_id": "a1",
        "section": "", "vendor": "Veeam", "product": "Veeam Backup & Replication"}
_ENV = {"mode": "local", "query": "q", "answer": "Immutable for a set period [1].",
        "citations": [{"marker": 1, "fact_uuid": "f1", "valid_at": None,
                       "invalid_at": None, "sources": [_SRC, {**_SRC, "url": "https://docs.example/b"}]}],
        "scope": {"vendors": ["Veeam"]}, "shape": "single", "facets": [],
        "applies_to": [{"vendor": "Veeam", "product": "Veeam Backup & Replication"}],
        "routing": {"chosen": "local"}}


class _Driver:
    async def execute_query(self, query, **params):
        assert query is ms.CATALOG_QUERY
        return ([{"vendor": "Veeam", "product": "Veeam Backup & Replication", "articles": 12},
                 {"vendor": "AWS", "product": "AWS Backup", "articles": 3}], None, None)


def _app(*, answer=None, search=None, progress=10.0):
    app = FastAPI()
    app.state.driver = _Driver()
    app.state.settings = SimpleNamespace(mcp_progress_seconds=progress)
    app.state.scope_resolver = SimpleNamespace(content_vendors=("AWS", "Veeam"))
    return app


@pytest.fixture
def patch(monkeypatch):
    def _set(answer=None, search=None):
        async def _answer(app, **kw):
            if isinstance(answer, BaseException):
                raise answer
            if callable(answer):
                return await answer(**kw)
            return answer or _ENV
        async def _search(app, **kw):
            if isinstance(search, BaseException):
                raise search
            return search or {"results": [{"fact": "Veeam keeps backups immutable.",
                                           "fact_uuid": "f1", "valid_at": "2026-01-01T00:00:00Z",
                                           "invalid_at": None, "sources": [_SRC]}],
                              "count": 1, "scope": {}}
        monkeypatch.setattr(ms, "run_answer", _answer)
        monkeypatch.setattr(ms, "run_search_local", _search)
    return _set


async def test_three_read_only_tools_are_listed(patch):
    patch()
    async with Client(ms.build_mcp_server(_app())) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert set(tools) == {"answer_question", "search_facts", "list_vendors_products"}
    assert tools["answer_question"].input_schema["required"] == ["question"]
    assert set(tools["answer_question"].input_schema["properties"]) == {
        "question", "vendors", "products", "mode"}
    assert "sources" in tools["answer_question"].description.lower()
    assert all(t.annotations.read_only_hint for t in tools.values())


async def test_answer_text_ends_with_sources_built_from_citations(patch):
    patch()
    async with Client(ms.build_mcp_server(_app())) as c:
        r = await c.call_tool("answer_question", {"question": "q"})
    text = r.content[0].text
    assert not r.is_error
    assert text.startswith("Immutable for a set period [1].")
    assert ("[1] Hardened repository — Veeam · Veeam Backup & Replication — "
            "https://docs.example/a (+1 more source)") in text
    assert r.structured_content["citations"] == _ENV["citations"]
    assert r.structured_content["mode"] == "local"


def test_format_answer_text_never_invents_a_url():
    env = {**_ENV, "answer": "No sources here.", "citations": []}
    assert ms.format_answer_text(env) == "No sources here."


async def test_mode_and_filters_reach_run_answer(patch):
    seen = {}
    async def _a(**kw):
        seen.update(kw)
        return _ENV
    patch(answer=_a)
    async with Client(ms.build_mcp_server(_app())) as c:
        await c.call_tool("answer_question", {"question": "q", "vendors": ["Veeam"],
                                              "products": ["AWS Backup"], "mode": "global"})
    assert seen == {"q": "q", "mode": "global", "vendor": ["Veeam"],
                    "product": ["AWS Backup"]}


async def test_auto_mode_lets_the_router_choose(patch):
    seen = {}
    async def _a(**kw):
        seen.update(kw)
        return _ENV
    patch(answer=_a)
    async with Client(ms.build_mcp_server(_app())) as c:
        await c.call_tool("answer_question", {"question": "q"})
    assert seen["mode"] is None


async def test_unknown_names_are_a_tool_error_naming_valid_vendors(patch):
    patch(answer=HTTPException(422, detail={"unknown": ["Veam"]}))
    async with Client(ms.build_mcp_server(_app())) as c:
        r = await c.call_tool("answer_question", {"question": "q", "vendors": ["Veam"]})
    assert r.is_error
    msg = r.content[0].text
    assert "Unknown vendor/product name(s): Veam" in msg
    assert "AWS, Veeam" in msg and "list_vendors_products" in msg


async def test_a_backend_failure_is_a_tool_error_saying_what_failed(patch, caplog):
    patch(answer=RuntimeError("neo4j unavailable"))
    async with Client(ms.build_mcp_server(_app())) as c:
        r = await c.call_tool("answer_question", {"question": "q"})
    assert r.is_error
    assert "answer_question failed: RuntimeError: neo4j unavailable" in r.content[0].text
    assert "answer_question failed" in caplog.text


async def test_a_refusal_is_a_result_not_an_error(patch):
    patch(answer={**_ENV, "answer": "I don't have enough information to answer that "
                  "from the available sources.", "citations": []})
    async with Client(ms.build_mcp_server(_app())) as c:
        r = await c.call_tool("answer_question", {"question": "q"})
    assert not r.is_error and r.content[0].text.startswith("I don't have enough")


async def test_a_slow_answer_sends_progress(patch):
    async def _slow(**kw):
        await asyncio.sleep(0.35)
        return _ENV
    patch(answer=_slow)
    got = []
    async def _progress(progress, total, message):
        got.append(message)
    async with Client(ms.build_mcp_server(_app(progress=0.1))) as c:
        r = await c.call_tool("answer_question", {"question": "q"},
                              progress_callback=_progress)
    assert not r.is_error
    assert len(got) >= 2 and all("still working" in m for m in got)


async def test_search_facts_lists_dates_and_sources_and_caps_k(patch, monkeypatch):
    seen = {}

    async def _s(app, **kw):
        seen.update(kw)
        return {"results": [{"fact": "Veeam keeps backups immutable.", "fact_uuid": "f1",
                             "valid_at": "2026-01-01T00:00:00Z", "invalid_at": None,
                             "sources": [_SRC]}], "count": 1, "scope": {}}
    patch()
    monkeypatch.setattr(ms, "run_search_local", _s)
    async with Client(ms.build_mcp_server(_app())) as c:
        r = await c.call_tool("search_facts", {"query": "immutability", "k": 99})
    assert seen["k"] == 25 and seen["include_invalid"] is False
    text = r.content[0].text
    assert "- Veeam keeps backups immutable. (valid from 2026-01-01)" in text
    assert "Hardened repository — Veeam · Veeam Backup & Replication — https://docs.example/a" in text
    assert r.structured_content["count"] == 1


async def test_list_vendors_products_reads_the_catalog(patch):
    patch()
    async with Client(ms.build_mcp_server(_app())) as c:
        r = await c.call_tool("list_vendors_products", {})
    assert r.structured_content == {"vendors": [
        {"vendor": "AWS", "products": [{"product": "AWS Backup", "articles": 3}]},
        {"vendor": "Veeam", "products": [
            {"product": "Veeam Backup & Replication", "articles": 12}]}]}
    assert "Veeam: Veeam Backup & Replication (12 articles)" in r.content[0].text
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_mcp_tools.py -q`
Expected: FAIL (`ModuleNotFoundError: answer_api.mcp_server`).

- [ ] **Step 3: Implement** `src/answer_api/mcp_server.py`:

```python
"""MCP server over answer-api (spec 2026-10-14): three read-only tools that call the
same functions the REST routes call. Mounted at /mcp by answer_api.app behind a
bearer token (answer_api.mcp_auth).

Design decision #2 holds: the Sources text is built only from citation `sources`
(graph-derived); nothing here or in the LLM writes a URL."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any, Literal, TypeVar

from fastapi import FastAPI, HTTPException
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from answer_api.app import run_answer, run_search_local

logger = logging.getLogger(__name__)
T = TypeVar("T")
MAX_K = 25

# Vendors/products with extracted content and their article counts -- what a filter
# can usefully name. Same structural chain as scope._CONTENT_PRODUCTS.
CATALOG_QUERY = (
    "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(:Source)"
    "-[:HAS_ARTICLE]->(a:Article) "
    "WHERE v.id IS NOT NULL AND p.id IS NOT NULL "
    "AND EXISTS { MATCH (a)-[:HAS_EPISODE]->(:Episodic) } "
    "RETURN v.name AS vendor, p.name AS product, count(DISTINCT a) AS articles "
    "ORDER BY vendor, product")

_ANSWER_DESC = (
    "Answer a question about backup-product documentation (vendors such as Veeam, AWS, "
    "Microsoft, Cohesity) from the knowledge graph. It picks the retrieval mode itself: "
    "single-product questions, comparisons between vendors, 'which vendors ...' questions "
    "(every vendor in the graph is checked) and change-over-time questions. Comparisons "
    "and 'which vendors' questions can take several minutes. The answer cites facts as [N]; "
    "always show the user the Sources list that follows it. Optional `vendors`/`products` "
    "restrict the answer to that documentation (see list_vendors_products for names); "
    "`mode` forces local, global, drift or timeline retrieval instead of auto.")
_SEARCH_DESC = (
    "Return the raw cited facts the graph holds for a query, with when each became valid "
    "and its source articles. No synthesis, fast, free. Use it to inspect the evidence "
    "behind an answer. Always show the sources.")
_LIST_DESC = (
    "List the vendors and products that have documentation in the knowledge graph, with "
    "article counts. Use it to find valid names for the vendors/products filters.")

_RO = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _source_line(src: dict) -> str:
    return (f"{src.get('title') or 'untitled'} — {src.get('vendor') or '?'} · "
            f"{src.get('product') or '?'} — {src.get('url') or 'no url'}")


def format_answer_text(env: dict) -> str:
    """The answer, then one Sources line per cited marker (first source, plus a count
    of the others). Every URL comes from `citations[].sources`."""
    answer = (env.get("answer") or "").rstrip()
    lines = []
    for c in env.get("citations") or []:
        sources = c.get("sources") or []
        if not sources:
            continue
        more = len({s.get("url") for s in sources}) - 1
        extra = f" (+{more} more source{'s' if more != 1 else ''})" if more > 0 else ""
        lines.append(f"[{c.get('marker')}] {_source_line(sources[0])}{extra}")
    return answer + ("\n\nSources:\n" + "\n".join(lines) if lines else "")


def _date(value: Any) -> str | None:
    return str(value)[:10] if value else None


def format_facts_text(result: dict) -> str:
    rows = []
    for r in result.get("results") or []:
        start, end = _date(r.get("valid_at")), _date(r.get("invalid_at"))
        when = (f"valid from {start}" if start else "no start date") + (
            f", until {end}" if end else "")
        rows.append(f"- {r.get('fact')} ({when})")
        rows += [f"    {_source_line(s)}" for s in r.get("sources") or []]
    return "\n".join(rows) if rows else "No facts found."


async def _with_heartbeat(ctx: Context, work: Awaitable[T], every: float) -> T:
    """Await `work`, reporting progress every `every` seconds so a client does not
    time out a long call; cancelling the call cancels the work."""
    task = asyncio.ensure_future(work)
    elapsed = 0.0
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=every)
            if done:
                return task.result()
            elapsed += every
            await ctx.report_progress(elapsed, None,
                                      f"still working ({int(elapsed)} s elapsed)")
    finally:
        if not task.done():
            task.cancel()


def _unknown_names(exc: HTTPException, app: FastAPI) -> ToolError:
    names = ", ".join((exc.detail or {}).get("unknown", [])) or "?"
    vendors = ", ".join(app.state.scope_resolver.content_vendors)
    return ToolError(f"Unknown vendor/product name(s): {names}. Vendors with content: "
                     f"{vendors}. Call list_vendors_products for product names.")


def build_mcp_server(app: FastAPI) -> MCPServer:
    server = MCPServer("graph-rag", instructions=(
        "Cited answers over backup-vendor documentation. Always show sources."))

    @server.tool(name="answer_question", description=_ANSWER_DESC, annotations=_RO)
    async def answer_question(
            question: str, ctx: Context, vendors: list[str] | None = None,
            products: list[str] | None = None,
            mode: Literal["auto", "local", "global", "drift", "timeline"] = "auto",
    ) -> CallToolResult:
        try:
            env = await _with_heartbeat(ctx, run_answer(
                app, q=question, mode=None if mode == "auto" else mode,
                vendor=vendors, product=products), app.state.settings.mcp_progress_seconds)
        except HTTPException as exc:
            if exc.status_code == 422:
                raise _unknown_names(exc, app) from exc
            logger.exception("answer_question failed")
            raise ToolError(f"answer_question failed: HTTP {exc.status_code}") from exc
        except Exception as exc:
            logger.exception("answer_question failed")
            raise ToolError(f"answer_question failed: {type(exc).__name__}: {exc}") from exc
        structured = {k: env.get(k) for k in
                      ("mode", "scope", "shape", "facets", "applies_to", "citations")}
        return CallToolResult(content=[TextContent(type="text", text=format_answer_text(env))],
                              structured_content=structured)

    @server.tool(name="search_facts", description=_SEARCH_DESC, annotations=_RO)
    async def search_facts(
            query: str, vendors: list[str] | None = None, products: list[str] | None = None,
            k: int = 10, include_superseded: bool = False,
    ) -> CallToolResult:
        try:
            result = await run_search_local(
                app, q=query, k=max(1, min(k, MAX_K)), vendor=vendors, product=products,
                include_invalid=include_superseded)
        except HTTPException as exc:
            if exc.status_code == 422:
                raise _unknown_names(exc, app) from exc
            logger.exception("search_facts failed")
            raise ToolError(f"search_facts failed: HTTP {exc.status_code}") from exc
        except Exception as exc:
            logger.exception("search_facts failed")
            raise ToolError(f"search_facts failed: {type(exc).__name__}: {exc}") from exc
        return CallToolResult(content=[TextContent(type="text", text=format_facts_text(result))],
                              structured_content=result)

    @server.tool(name="list_vendors_products", description=_LIST_DESC, annotations=_RO)
    async def list_vendors_products() -> CallToolResult:
        try:
            records, _, _ = await app.state.driver.execute_query(CATALOG_QUERY)
        except Exception as exc:
            logger.exception("list_vendors_products failed")
            raise ToolError(f"list_vendors_products failed: {type(exc).__name__}: {exc}") from exc
        by_vendor: dict[str, list[dict]] = {}
        for r in records:
            by_vendor.setdefault(r["vendor"], []).append(
                {"product": r["product"], "articles": r["articles"]})
        vendors = [{"vendor": v, "products": by_vendor[v]} for v in sorted(by_vendor)]
        text = "\n".join(
            f"{v['vendor']}: " + "; ".join(f"{p['product']} ({p['articles']} articles)"
                                          for p in v["products"]) for v in vendors)
        return CallToolResult(content=[TextContent(type="text", text=text or "No content yet.")],
                              structured_content={"vendors": vendors})

    return server
```

The tools resolve `run_answer` / `run_search_local` as `mcp_server` module globals at call time, which is what lets the tests' `monkeypatch.setattr(ms, ...)` replace them. No import cycle: `app.py` imports `mcp_server` only inside `create_app` (Task 4).

- [ ] **Step 4: Run the tests**

Run: `uv run --extra dev pytest tests/unit/test_mcp_tools.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint, types, commit**

```bash
uv run ruff check src tests && uv run mypy src
git add src/answer_api/mcp_server.py tests/unit/test_mcp_tools.py
git commit -m "feat(answer-api): MCP tools answer_question, search_facts, list_vendors_products"
```

---

### Task 4: Bearer-token gate, mount at /mcp, session manager in the lifespan

**Files:**
- Create: `src/answer_api/mcp_auth.py`
- Modify: `src/answer_api/app.py` (`create_app` mounts; `_lifespan` enters the session manager)
- Test: `tests/unit/test_mcp_http.py`

**Interfaces:**
- Consumes: `build_mcp_server(app)` (Task 3); `mcp_bearer_token`, `mcp_allowed_hosts` (Task 1).
- Produces: `BearerTokenGate(app: ASGIApp, token: str, path: str = "/mcp")`; `app.state.mcp_server: MCPServer | None` set by `create_app`.

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_mcp_http.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --extra dev pytest tests/unit/test_mcp_http.py -q`
Expected: FAIL (`ModuleNotFoundError: answer_api.mcp_auth`).

- [ ] **Step 3: Implement** `src/answer_api/mcp_auth.py`:

```python
"""Bearer-token gate for /mcp (spec 2026-10-14 §3). Constant-time comparison; a
missing or wrong token gets 401 with `WWW-Authenticate: Bearer`. Requests outside
`path` and non-HTTP scopes pass through untouched."""
from __future__ import annotations

import hmac

from starlette.types import ASGIApp, Receive, Scope, Send


class BearerTokenGate:
    def __init__(self, app: ASGIApp, token: str, path: str = "/mcp"):
        if not token:
            raise ValueError("BearerTokenGate needs a non-empty token")
        self.app, self._expected, self.path = app, f"Bearer {token}".encode(), path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and (scope["path"] == self.path
                                        or scope["path"].startswith(self.path + "/")):
            got = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(got, self._expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"www-authenticate", b"Bearer"),
                                        (b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return
        await self.app(scope, receive, send)
```

In `src/answer_api/app.py`:

1. Imports (top of file):

```python
from mcp.server.transport_security import TransportSecuritySettings

from answer_api.mcp_auth import BearerTokenGate
```

2. At the END of `create_app` (after every `@app.get` route, so REST routes match first), before `return app`:

```python
    # MCP (spec 2026-10-14): mounted only when a token is configured -- fail closed.
    settings = get_extract_settings()
    app.state.mcp_server = None
    if settings.mcp_bearer_token:
        from answer_api.mcp_server import build_mcp_server  # imports run_* from here

        server = build_mcp_server(app)
        mcp_app = server.streamable_http_app(
            streamable_http_path="/mcp", stateless_http=True,
            transport_security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=list(settings.mcp_allowed_hosts),
                allowed_origins=[f"https://{h}" for h in settings.mcp_allowed_hosts
                                 if not h.endswith(":*")]))
        app.state.mcp_server = server
        # Mounted at the root AFTER the REST routes: only /mcp reaches it; anything
        # else unmatched gets the sub-app's 404.
        app.mount("", BearerTokenGate(mcp_app, settings.mcp_bearer_token, "/mcp"))
```

(The import is local because `mcp_server` imports `run_answer` from this module.)

3. In `_lifespan`, wrap the `yield` so the SDK session manager runs for the app's lifetime (its own Starlette lifespan does not run when mounted). Find the single `yield` in `_lifespan` and replace it with:

```python
    mcp_server = getattr(app.state, "mcp_server", None)
    if mcp_server is not None:
        async with mcp_server.session_manager.run():
            yield
    else:
        yield
```

keeping the surrounding try/finally (shutdown closes) exactly as it is.

- [ ] **Step 4: Run the tests**

Run: `uv run --extra dev pytest tests/unit/test_mcp_http.py tests/unit/test_mcp_tools.py tests/unit/test_answer_api_app.py -q`
Expected: all PASS. If `test_right_token_reaches_the_mcp_server` returns 406/400, check the request headers match `_HDRS` (the transport requires `accept: application/json, text/event-stream` and `content-type: application/json`) before changing code.

- [ ] **Step 5: Full gate, commit**

```bash
uv run ruff check src tests && uv run mypy src && uv run --extra dev pytest tests/unit -q
git add src/answer_api/mcp_auth.py src/answer_api/app.py tests/unit/test_mcp_http.py
git commit -m "feat(answer-api): serve MCP at /mcp behind a bearer token (fail closed)"
```

---

### Task 5: Docs — setup guide, CLAUDE.md, BACKLOG

**Files:**
- Create: `docs/deploy/mcp.md`
- Modify: `CLAUDE.md` (Repository status + `answer_api/` module layout), `docs/superpowers/BACKLOG.md` (item 47 note)

- [ ] **Step 1: Write `docs/deploy/mcp.md`** with exactly these sections:

````markdown
# MCP server (answer-api /mcp) — LAN setup

answer-api serves an MCP endpoint at `https://graphrag.k3s.home.lan/mcp` (streamable
HTTP, stateless) when `MCP_BEARER_TOKEN` is set in `graph-rag-secret`. Without it, /mcp
is not served. Tools: `answer_question` (the /answer router; may take minutes),
`search_facts` (raw cited facts, free), `list_vendors_products` (catalog, free).

## 1. Add the token to the Secret (operator)

Generate and add it without echoing it (it never touches disk or the repo):

```bash
TOKEN=$(openssl rand -hex 32)
kubectl -n graph-rag get secret graph-rag-secret -o json \
  | T="$TOKEN" python3 -c 'import json,sys,base64,os; d=json.load(sys.stdin); d["data"]["MCP_BEARER_TOKEN"]=base64.b64encode(os.environ["T"].encode()).decode(); json.dump(d,sys.stdout)' \
  | kubectl apply -f -
kubectl -n graph-rag rollout restart deploy/graph-rag-answer
unset TOKEN
```

Read it back when configuring a client:
`kubectl -n graph-rag get secret graph-rag-secret -o jsonpath='{.data.MCP_BEARER_TOKEN}' | base64 -d`

## 2. Trust the internal CA

The endpoint uses the cluster's internal CA (`rootca.home.lan`). For Claude Code:
`export NODE_EXTRA_CA_CERTS=/path/to/rootca.pem` before starting `claude`.

## 3. Add the server to Claude Code

```bash
claude mcp add --transport http graph-rag https://graphrag.k3s.home.lan/mcp \
  --header "Authorization: Bearer <token>"
```

Comparisons and "which vendors" questions take 1-7 minutes. The server sends progress
notifications every 10 s; if your client still times out, raise its tool timeout
(Claude Code: `export MCP_TOOL_TIMEOUT=600000`, in ms).

## 4. Cost

`search_facts` and `list_vendors_products` make no LLM call. Each `answer_question` costs
what one /answer call costs (roughly $0.005-0.02; more for comparisons and enumerations).

## 5. Rotate or disable

Rotate: repeat step 1 and update clients. Disable: remove the key from the Secret and
restart the deployment — /mcp then returns 404.
````

- [ ] **Step 2: Update `CLAUDE.md`** — in "Repository status", change "**Not yet built:** Copilot/MCP exposure (Phase 5)." to: "**MCP (Phase 5, LAN slice) built:** `/mcp` on answer-api — `answer_question`, `search_facts`, `list_vendors_products` — behind `MCP_BEARER_TOKEN` (`docs/deploy/mcp.md`). **Not yet built:** public exposure (tunnel/APIM), OAuth, Copilot Studio onboarding." In the `answer_api/` module-layout bullet add: "`mcp_server.py` (`build_mcp_server`: three MCP tools over `run_answer`/`run_search_local`), `mcp_auth.py` (`BearerTokenGate`)."

- [ ] **Step 3: Update BACKLOG 47** — append to item 47's paragraph: "**2026-10-14:** `/mcp` is token-gated (`MCP_BEARER_TOKEN`, fail closed); the REST routes are still unauthenticated and LAN-only — this item stays open for them until public exposure."

- [ ] **Step 4: Commit**

```bash
git add docs/deploy/mcp.md CLAUDE.md docs/superpowers/BACKLOG.md
git commit -m "docs: MCP setup guide; CLAUDE.md and BACKLOG 47 for /mcp"
```

---

## Deployment and live check (controller, after the final review — each step needs the user's approval)

These touch shared state; they are not subagent tasks.

1. Full CI suite on the branch (`ruff`, `mypy`, `pytest -m "not live"`), gitleaks over `origin/main..HEAD`, merge to `main`, push (user approval). The image workflow builds `sha-<merge>`.
2. Add `MCP_BEARER_TOKEN` to `graph-rag-secret` per `docs/deploy/mcp.md` §1 (user approval: cluster mutation; token via stdin only, never printed).
3. `helm upgrade ... --reset-then-reuse-values --force-conflicts --set image.tag=sha-<merge>` from the merge commit's chart (user approval).
4. Live check from the dev host (token read into an env var, never echoed): `initialize` + `tools/list` over HTTPS; `list_vendors_products` and `search_facts` (free); one `answer_question` on a comparison (~1¢) with a progress callback — confirm progress events arrive and the call completes through the ingress. If the ingress cuts the stream, report the measured cut-off and propose the Traefik timeout change before making it.
5. Hand the user the Claude Code setup (`docs/deploy/mcp.md` §2-3).
