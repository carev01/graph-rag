from __future__ import annotations
import hashlib, hmac, json
from typing import Awaitable, Callable
from fastapi import APIRouter, BackgroundTasks, Request, Response

def verify_signature(secret: str, body: bytes, header: str | None) -> bool:
    if not header:
        return False
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)

def build_router(store, settings, trigger: Callable[[], Awaitable[None]]) -> APIRouter:
    router = APIRouter()

    @router.post("/webhooks/docextractor")
    async def receive(request: Request, background: BackgroundTasks) -> Response:
        # Fail closed when no secret is configured: an empty secret makes
        # verify_signature() validate against an empty-key HMAC, which anyone
        # reachable could forge. Reject rather than authenticate with no key.
        if not settings.webhook_secret:
            return Response(status_code=401)
        body = await request.body()
        sig = request.headers.get("X-DocExtractor-Signature")
        if not verify_signature(settings.webhook_secret, body, sig):
            return Response(status_code=401)
        if await store.seen_delivery(sig):
            return Response(content='{"status":"ok"}', media_type="application/json")
        payload = json.loads(body or b"{}")
        source_id = payload.get("source_id")
        run = True
        if source_id:
            run = await store.should_run_source(
                source_id, settings.webhook_debounce_seconds)
        if run:
            background.add_task(trigger)
        return Response(content='{"status":"ok"}', media_type="application/json")

    return router
