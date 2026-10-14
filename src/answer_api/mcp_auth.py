"""Bearer-token gate for /mcp (spec 2026-10-14 §3). Constant-time comparison; a
missing or wrong token gets 401 with `WWW-Authenticate: Bearer`. Requests outside
`path` and non-HTTP scopes pass through untouched.

The path tested is the one Starlette's router dispatches on (`get_route_path`: the
request path with `root_path` stripped), never raw `scope["path"]` -- otherwise a
deployment behind `--root-path` or a prefixing proxy would route `/prefix/mcp` to the
MCP app while the gate saw a path it does not guard."""
from __future__ import annotations

import hmac

from starlette._utils import get_route_path
from starlette.types import ASGIApp, Receive, Scope, Send


class BearerTokenGate:
    def __init__(self, app: ASGIApp, token: str, path: str = "/mcp"):
        if not token:
            raise ValueError("BearerTokenGate needs a non-empty token")
        self.app, self._expected, self.path = app, f"Bearer {token}".encode(), path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and self._guards(get_route_path(scope)):
            got = dict(scope.get("headers") or []).get(b"authorization", b"")
            if not hmac.compare_digest(got, self._expected):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"www-authenticate", b"Bearer"),
                                        (b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return
        await self.app(scope, receive, send)

    def _guards(self, route_path: str) -> bool:
        return route_path == self.path or route_path.startswith(self.path + "/")
