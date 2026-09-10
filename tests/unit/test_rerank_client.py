"""The reranker returns None when it CANNOT score. None never means "nothing is
relevant" -- four defects in this codebase came from a failure coerced into a
legitimate-looking value."""
import httpx
import pytest

from answer_api.rerank import rerank, rerank_configured
from graph_extract.config import ExtractSettings


def _settings(**kw):
    base = dict(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                rerank_base_url="https://rr.example/v1",
                rerank_model="rerank-3", rerank_api_key="k")
    base.update(kw)
    return ExtractSettings(**base)


def _transport(handler):
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_parses_index_and_score_pairs():
    def handler(request):
        return httpx.Response(200, json={"data": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.4}]})
    got = await rerank("q", ["a", "b", "c"], top_k=3, settings=_settings(),
                       transport=_transport(handler))
    assert got == [(2, 0.9), (0, 0.4)]


@pytest.mark.asyncio
async def test_none_on_non_200():
    def handler(request):
        return httpx.Response(500, text="boom")
    assert await rerank("q", ["a"], top_k=1, settings=_settings(),
                        transport=_transport(handler)) is None


@pytest.mark.asyncio
async def test_none_on_transport_error():
    def handler(request):
        raise httpx.ConnectError("no route")
    assert await rerank("q", ["a"], top_k=1, settings=_settings(),
                        transport=_transport(handler)) is None


@pytest.mark.asyncio
async def test_none_on_malformed_payload():
    def handler(request):
        return httpx.Response(200, json={"unexpected": True})
    assert await rerank("q", ["a"], top_k=1, settings=_settings(),
                        transport=_transport(handler)) is None


@pytest.mark.asyncio
async def test_out_of_range_indices_are_dropped_not_crashed_on():
    def handler(request):
        return httpx.Response(200, json={"data": [
            {"index": 99, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.5}]})
    got = await rerank("q", ["a", "b"], top_k=2, settings=_settings(),
                       transport=_transport(handler))
    assert got == [(1, 0.5)]


@pytest.mark.asyncio
async def test_empty_documents_returns_empty_without_calling_the_api():
    called = []

    def handler(request):
        called.append(1)
        return httpx.Response(200, json={"data": []})
    got = await rerank("q", [], top_k=3, settings=_settings(),
                       transport=_transport(handler))
    assert got == [] and called == []


@pytest.mark.asyncio
async def test_sends_model_query_and_documents():
    seen = {}

    def handler(request):
        import json as _json
        seen.update(_json.loads(request.content))
        assert request.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"data": []})
    await rerank("what about encryption?", ["d1", "d2"], top_k=2,
                 settings=_settings(), transport=_transport(handler))
    assert seen["model"] == "rerank-3"
    assert seen["query"] == "what about encryption?"
    assert seen["documents"] == ["d1", "d2"]


def test_rerank_configured_reports_missing_config():
    assert rerank_configured(_settings()) is True
    assert rerank_configured(ExtractSettings(
        _env_file=None, docext_base_url="http://x", docext_read_key="k",
        neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")) is False
