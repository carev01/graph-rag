import json
from pathlib import Path

import httpx

from graph_sync.catalog import Catalog

FIX = Path(__file__).parent.parent / "fixtures"


def test_resolve_source_to_product_and_vendor():
    vendors = json.loads((FIX / "vendors.json").read_text())["vendors"]
    products = json.loads((FIX / "products.json").read_text())["products"]
    sources = json.loads((FIX / "sources.json").read_text())["sources"]
    cat = Catalog.from_lists(vendors, products, sources)
    info = cat.resolve("21632f3b-5a4c-4c93-9f00-6701d0e9f677")  # AWS Backup Dev Guide
    assert info is not None
    assert info.product_name == "AWS Backup"
    assert info.vendor_name == "AWS"
    assert info.vendor_id and info.product_id


def test_resolve_unknown_source_returns_none():
    cat = Catalog.from_lists([], [], [])
    assert cat.resolve("does-not-exist") is None


async def test_refresh_unions_sources_across_products():
    """/api/sources caps at 200 and ignores offset/page, so refresh() must
    enumerate sources per product (see catalog.py:refresh docstring) rather
    than issue a single bulk /api/sources call. This test proves that a
    source belonging to a "later" product still resolves after refresh().
    """
    vendors = [
        {"id": "v1", "name": "Vendor One", "website": "https://v1.example"},
        {"id": "v2", "name": "Vendor Two", "website": "https://v2.example"},
    ]
    products = [
        {"id": "p1", "vendor_id": "v1", "name": "Product One", "version": None},
        {"id": "p2", "vendor_id": "v2", "name": "Product Two", "version": "3.0"},
    ]
    sources_by_product = {
        "p1": [
            {
                "id": "s1",
                "product_id": "p1",
                "name": "Source One",
                "base_url": "https://s1.example",
                "source_type": "web",
                "platform": "zendesk",
                "last_extracted_at": None,
            }
        ],
        "p2": [
            {
                "id": "s2",
                "product_id": "p2",
                "name": "Source Two",
                "base_url": "https://s2.example",
                "source_type": "web",
                "platform": "confluence",
                "last_extracted_at": None,
            }
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/vendors":
            return httpx.Response(200, json={"vendors": vendors})
        if request.url.path == "/api/products":
            return httpx.Response(200, json={"products": products})
        if request.url.path == "/api/sources":
            product_id = request.url.params.get("product_id")
            return httpx.Response(
                200, json={"sources": sources_by_product.get(product_id, [])}
            )
        raise AssertionError(f"unexpected request: {request.url}")

    client = httpx.AsyncClient(
        base_url="https://docextractor.example", transport=httpx.MockTransport(handler)
    )
    try:
        cat = Catalog(client)
        await cat.refresh()
    finally:
        await client.aclose()

    for source_id, product_id, vendor_id in (("s1", "p1", "v1"), ("s2", "p2", "v2")):
        info = cat.resolve(source_id)
        assert info is not None, f"source {source_id} was dropped by refresh()"
        assert info.product_id == product_id
        assert info.vendor_id == vendor_id
