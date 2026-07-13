from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    source_name: str
    base_url: str | None
    source_type: str | None
    platform: str | None
    last_extracted_at: str | None
    product_id: str
    product_name: str
    product_version: str | None
    vendor_id: str
    vendor_name: str
    vendor_website: str | None


class Catalog:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client
        self._by_source: dict[str, SourceInfo] = {}

    @classmethod
    def from_lists(
        cls,
        vendors: list[dict[str, Any]],
        products: list[dict[str, Any]],
        sources: list[dict[str, Any]],
    ) -> "Catalog":
        cat = cls()
        cat._build(vendors, products, sources)
        return cat

    def _build(
        self,
        vendors: list[dict[str, Any]],
        products: list[dict[str, Any]],
        sources: list[dict[str, Any]],
    ) -> None:
        vmap = {v["id"]: v for v in vendors}
        pmap = {p["id"]: p for p in products}
        by_source: dict[str, SourceInfo] = {}
        for s in sources:
            p = pmap.get(s["product_id"])
            if p is None:
                continue
            v = vmap.get(p["vendor_id"], {})
            by_source[s["id"]] = SourceInfo(
                source_id=s["id"],
                source_name=s.get("name", ""),
                base_url=s.get("base_url"),
                source_type=s.get("source_type"),
                platform=s.get("platform"),
                last_extracted_at=s.get("last_extracted_at"),
                product_id=p["id"],
                product_name=p.get("name", ""),
                product_version=p.get("version"),
                vendor_id=v.get("id", p["vendor_id"]),
                vendor_name=v.get("name", ""),
                vendor_website=v.get("website"),
            )
        self._by_source = by_source

    def resolve(self, source_id: str) -> SourceInfo | None:
        return self._by_source.get(source_id)

    async def load(self) -> None:
        await self.refresh()

    async def refresh(self) -> None:
        # NOTE: deviates from the original task brief, which called a single
        # `GET /api/sources?limit=1000`. Verified against the live API: the
        # /api/sources list endpoint caps out at 200 results and ignores
        # offset/page params, with unstable ordering across calls -- a
        # single bulk call silently drops sources (284 exist upstream, only
        # 200 come back, and *which* 200 isn't stable). /api/vendors (40)
        # and /api/products (79) are both under the 200 cap and return
        # fully in one call, so those stay as bulk calls. Sources are
        # instead enumerated per product via
        # `GET /api/sources?product_id=<pid>&limit=200` and unioned; each
        # product has far fewer than 200 sources, so no per-product call
        # hits the cap, and the union covers all sources. Verified this
        # yields all 284 sources against the live cluster.
        assert self._client is not None, "Catalog needs an httpx client to refresh"
        client = self._client
        vendors = (await client.get("/api/vendors", params={"limit": 200})).json()[
            "vendors"
        ]
        products = (await client.get("/api/products", params={"limit": 200})).json()[
            "products"
        ]

        async def fetch_sources(product_id: str) -> list[dict[str, Any]]:
            resp = await client.get(
                "/api/sources", params={"product_id": product_id, "limit": 200}
            )
            result: list[dict[str, Any]] = resp.json()["sources"]
            return result

        results = await asyncio.gather(*(fetch_sources(p["id"]) for p in products))
        sources: list[dict[str, Any]] = [s for batch in results for s in batch]

        self._build(vendors, products, sources)
