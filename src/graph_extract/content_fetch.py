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
    # Added upstream 2026-09-13. Optional so a pre-deploy DocExtractor still parses;
    # `_reference_time` logs loudly rather than silently reverting to crawl time.
    #   content_changed_at   -- the served markdown became current (the ordering axis)
    #   content_changed_basis-- exact | lower_bound | first_seen
    #   source_changed_at    -- the VENDOR's page changed, as opposed to a re-render
    #                           or an enrichment; gates invalidation, never ordering
    #   last_updated_source  -- vendor_meta | page_markup, non-null with last_updated_at
    content_changed_at: str | None = None
    content_changed_basis: str | None = None
    source_changed_at: str | None = None
    last_updated_source: str | None = None

async def fetch_article(client: httpx.AsyncClient, article_id: str) -> ArticleContent:
    resp = await client.get(f"/api/articles/{article_id}")
    resp.raise_for_status()
    d = resp.json()
    return ArticleContent(
        id=d["id"], title=d.get("title", ""), source_url=d.get("source_url", ""),
        content_markdown=d.get("content_markdown", ""),
        last_updated_at=d.get("last_updated_at"), extracted_at=d.get("extracted_at"),
        images=d.get("images", []) or [],
        content_changed_at=d.get("content_changed_at"),
        content_changed_basis=d.get("content_changed_basis"),
        source_changed_at=d.get("source_changed_at"),
        last_updated_source=d.get("last_updated_source"),
    )
