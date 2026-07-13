from __future__ import annotations
from dataclasses import dataclass
import httpx

@dataclass
class ArticleContent:
    id: str
    title: str
    source_url: str
    content_markdown: str
    last_updated_at: str | None
    extracted_at: str | None
    images: list[dict]

async def fetch_article(client: httpx.AsyncClient, article_id: str) -> ArticleContent:
    resp = await client.get(f"/api/articles/{article_id}")
    resp.raise_for_status()
    d = resp.json()
    return ArticleContent(
        id=d["id"], title=d.get("title", ""), source_url=d.get("source_url", ""),
        content_markdown=d.get("content_markdown", ""),
        last_updated_at=d.get("last_updated_at"), extracted_at=d.get("extracted_at"),
        images=d.get("images", []) or [],
    )
