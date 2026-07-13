import hmac, hashlib, json, pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from graph_sync.webhook import verify_signature, build_router

pytestmark = pytest.mark.asyncio
SECRET = "s3cr3t"

def _sig(body: bytes) -> str:
    return "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()

def test_verify_signature_accepts_and_rejects():
    body = b'{"a":1}'
    assert verify_signature(SECRET, body, _sig(body)) is True
    assert verify_signature(SECRET, body, "sha256=deadbeef") is False
    assert verify_signature(SECRET, body, None) is False

class FakeStore:
    def __init__(self): self.seen=set(); self.ran=[]
    async def seen_delivery(self, sig):
        dup = sig in self.seen; self.seen.add(sig); return dup
    async def should_run_source(self, sid, deb): self.ran.append(sid); return True

async def _app(store, calls):
    from types import SimpleNamespace
    settings = SimpleNamespace(webhook_secret=SECRET, webhook_debounce_seconds=300)
    async def trigger(): calls.append(1)
    app = FastAPI(); app.include_router(build_router(store, settings, trigger))
    return app

async def test_webhook_rejects_bad_hmac():
    calls=[]; app = await _app(FakeStore(), calls)
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        r = await c.post("/webhooks/docextractor", content=b"{}",
                         headers={"X-DocExtractor-Signature":"sha256=bad"})
    assert r.status_code == 401 and calls == []

async def test_webhook_rejects_when_secret_unconfigured():
    # An empty secret must fail closed: verify_signature("", ...) would validate
    # against an empty-key HMAC, so a forged delivery would otherwise be accepted.
    from types import SimpleNamespace
    calls: list = []
    store = FakeStore()
    settings = SimpleNamespace(webhook_secret="", webhook_debounce_seconds=300)
    async def trigger(): calls.append(1)
    app = FastAPI(); app.include_router(build_router(store, settings, trigger))
    body = b'{"source_id":"s1"}'
    forged = "sha256=" + hmac.new(b"", body, hashlib.sha256).hexdigest()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        r = await c.post("/webhooks/docextractor", content=body,
                         headers={"X-DocExtractor-Signature": forged})
    assert r.status_code == 401 and calls == []

async def test_webhook_accepts_and_dedups():
    calls=[]; store=FakeStore(); app = await _app(store, calls)
    body = json.dumps({"event":"extraction_complete","source_id":"s1"}).encode()
    async with AsyncClient(transport=ASGITransport(app), base_url="http://t") as c:
        h = {"X-DocExtractor-Signature": _sig(body)}
        r1 = await c.post("/webhooks/docextractor", content=body, headers=h)
        r2 = await c.post("/webhooks/docextractor", content=body, headers=h)
    assert r1.status_code == 200 and r2.status_code == 200
    assert store.ran == ["s1"]  # deduped: second delivery did not enqueue again
