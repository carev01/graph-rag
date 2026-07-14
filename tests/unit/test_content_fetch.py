import httpx
import json
import pytest
from graph_extract.content_fetch import fetch_article, ArticleContent

pytestmark = pytest.mark.asyncio

async def test_fetch_article_parses_detail():
    body = json.dumps({
        "id": "a1", "title": "What is AWS Backup?",
        "source_url": "https://x/whatis.html",
        "content_markdown": "# What is AWS Backup?\n\nAWS Backup is ...",
        "last_updated_at": None, "extracted_at": "2026-07-12T16:00:17Z",
        "images": [],
    }).encode()
    def handler(req):
        assert req.url.path == "/api/articles/a1"
        return httpx.Response(200, content=body)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://x")
    art = await fetch_article(client, "a1")
    assert isinstance(art, ArticleContent)
    assert art.title.startswith("What is") and art.content_markdown.startswith("#")
    assert art.extracted_at == "2026-07-12T16:00:17Z"
