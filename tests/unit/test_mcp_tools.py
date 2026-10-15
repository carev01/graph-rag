"""MCP tools over answer-api (spec 2026-10-14 §2, §5), in-process client, fake state."""
import asyncio
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from mcp import Client

import answer_api.mcp_server as ms

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


async def test_answer_structured_output_carries_routing_and_list_fields(patch):
    env = {**_ENV, "routing": {"chosen": "local", "path": "list",
                               "degraded": ["retrieval failed for: q1"]},
           "list_mode": True, "expansion_queries": ["q1", "q2"]}
    patch(answer=env)
    async with Client(ms.build_mcp_server(_app())) as c:
        r = await c.call_tool("answer_question", {"question": "q"})
    assert r.structured_content["routing"] == env["routing"]
    assert r.structured_content["list_mode"] is True
    assert r.structured_content["expansion_queries"] == ["q1", "q2"]


async def test_answer_structured_output_omits_absent_list_fields(patch):
    patch()
    async with Client(ms.build_mcp_server(_app())) as c:
        r = await c.call_tool("answer_question", {"question": "q"})
    assert r.structured_content["routing"] == {"chosen": "local"}
    assert "list_mode" not in r.structured_content
    assert "expansion_queries" not in r.structured_content


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


@pytest.mark.parametrize("tool,args", [("answer_question", {"question": "q"}),
                                       ("search_facts", {"query": "q"})])
async def test_unknown_names_are_logged(patch, caplog, tool, args):
    err = HTTPException(422, detail={"unknown": ["Veam", "Comvault"]})
    patch(answer=err, search=err)
    caplog.set_level("INFO", logger=ms.logger.name)
    async with Client(ms.build_mcp_server(_app())) as c:
        r = await c.call_tool(tool, {**args, "vendors": ["Veam", "Comvault"]})
    assert r.is_error
    assert f"{tool}: unknown vendor/product name(s): Veam, Comvault" in caplog.text


async def test_unknown_names_with_no_content_vendors_reads_sensibly(patch):
    patch(answer=HTTPException(422, detail={"unknown": ["Veam"]}))
    app = _app()
    app.state.scope_resolver = SimpleNamespace(content_vendors=())
    async with Client(ms.build_mcp_server(app)) as c:
        r = await c.call_tool("answer_question", {"question": "q", "vendors": ["Veam"]})
    assert r.is_error
    assert "Vendors with content: none yet." in r.content[0].text


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
