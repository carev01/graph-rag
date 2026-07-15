from __future__ import annotations
from typing import AsyncIterator
import httpx
from docext.client import make_docext_client
from graph_sync.config import Settings
from graph_sync.models import (
    ContentRecord, TombstoneRecord, ControlRecord, parse_delta_line,
)

def make_client(settings: Settings, *, admin: bool = False) -> httpx.AsyncClient:
    return make_docext_client(
        base_url=settings.docext_base_url, read_key=settings.docext_read_key,
        admin_key=settings.docext_admin_key, verify_tls=settings.docext_verify_tls, admin=admin)

def build_delta_params(*, since: str | None = None, source_id: str | None = None,
                       vendor_id: str | None = None,
                       bootstrap_after: str | None = None) -> dict:
    params = {"since": since, "source_id": source_id,
              "vendor_id": vendor_id, "bootstrap_after": bootstrap_after}
    return {k: v for k, v in params.items() if v is not None}

class DeltaStream:
    def __init__(self, client: httpx.AsyncClient, params: dict) -> None:
        self._client = client
        self._params = params
        self.bootstrap_start_since: str | None = None
        self.next_since: str | None = None
        self.count: int | None = None
        self.terminated_clean: bool = False
        self.malformed: list[str] = []  # raw lines that failed to parse

    async def records(self) -> AsyncIterator[ContentRecord | TombstoneRecord]:
        async with self._client.stream(
            "GET", "/api/articles/delta", params=self._params
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    rec = parse_delta_line(line)
                except Exception:
                    # Record the unparseable line and keep reading the stream
                    # (so remaining good records + the terminal control line are
                    # still seen); the caller dead-letters these and refuses to
                    # advance the cursor past them (spec §7).
                    self.malformed.append(line)
                    continue
                if isinstance(rec, ControlRecord):
                    if rec.control == "bootstrap_start":
                        self.bootstrap_start_since = rec.next_since
                    elif rec.control == "cursor":
                        self.next_since = rec.next_since
                        self.count = rec.count
                        self.terminated_clean = True
                    continue
                yield rec
