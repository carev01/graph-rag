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
    detail: dict[str, Any] = exc.detail if isinstance(exc.detail, dict) else {}
    names = ", ".join(detail.get("unknown", [])) or "?"
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
        vendors: list[dict[str, Any]] = [
            {"vendor": v, "products": by_vendor[v]} for v in sorted(by_vendor)]
        text = "\n".join(
            f"{v['vendor']}: " + "; ".join(f"{p['product']} ({p['articles']} articles)"
                                          for p in v["products"]) for v in vendors)
        return CallToolResult(content=[TextContent(type="text", text=text or "No content yet.")],
                              structured_content={"vendors": vendors})

    return server
