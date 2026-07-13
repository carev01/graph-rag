import pytest
from httpx import ASGITransport, AsyncClient
from graph_sync.app import create_app

pytestmark = pytest.mark.asyncio

class FakeStatusStore:
    async def get_cursor(self): return "cur-1"
    async def dead_letter_count(self): return 0

async def test_health_and_status():
    app = create_app(status_store=FakeStatusStore(), start_background=False)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        assert (await c.get("/health")).json() == {"status": "ok"}
        s = (await c.get("/status")).json()
        assert s["cursor"] == "cur-1" and s["dead_letter_count"] == 0
