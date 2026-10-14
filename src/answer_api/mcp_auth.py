"""Bearer-token gate for /mcp (spec 2026-10-14 §3). Constant-time comparison; a
missing or wrong token gets 401 with `WWW-Authenticate: Bearer`. Requests outside
`path` and non-HTTP scopes pass through untouched."""
from __future__ import annotations

import hmac

from starlette.types import ASGIApp, Receive, Scope, Send


class BearerTokenGate:
    def __init__(self, app: ASGIApp, token: str, path: str = "/mcp"):
        if not token:
            raise ValueError("BearerTokenGate needs a non-empty token")
        self.app, self._expected, self.path = app, f"Bearer {token}".encode(), path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and (scope["path"] == self.path
                                        or scope["path"].startswith(self.path + "/")):
            got = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(got, self._expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"www-authenticate", b"Bearer"),
                                        (b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return
        await self.app(scope, receive, send)
