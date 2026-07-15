from __future__ import annotations
import httpx


def make_docext_client(*, base_url: str, read_key: str, admin_key: str,
                       verify_tls: bool, admin: bool = False) -> httpx.AsyncClient:
    key = admin_key if admin else read_key
    return httpx.AsyncClient(
        base_url=base_url,
        headers={"X-API-Key": key},
        verify=verify_tls,
        timeout=300.0,
    )
