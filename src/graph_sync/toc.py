from __future__ import annotations
import httpx

async def fetch_toc(client: httpx.AsyncClient, source_id: str) -> dict:
    resp = await client.get(f"/api/articles/toc/{source_id}")
    resp.raise_for_status()
    return resp.json()
