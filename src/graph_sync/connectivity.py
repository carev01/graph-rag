"""Pre-flight connectivity check for a deployed graph-rag (docs/deploy/k3s.md).

    python -m graph_sync.connectivity

Read-only and free: no LLM tokens, no writes. Checks every dependency a worker or
the answer API needs and exits non-zero naming each failure. Error text is redacted
of every credential the settings hold before it is printed.
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import asyncpg
import httpx
from neo4j import AsyncGraphDatabase

from graph_extract.config import get_extract_settings
from graph_sync.config import get_settings

TIMEOUT = 15.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _secret_values() -> list[str]:
    """Every settings value whose field name marks it as a credential."""
    out: list[str] = []
    for model in (get_extract_settings(), get_settings()):
        for key, value in model.model_dump().items():
            if any(w in key.lower() for w in ("password", "key", "secret", "token", "dsn")):
                if isinstance(value, str) and value:
                    out.append(value)
    return out + _dsn_passwords(out)


def _dsn_passwords(values: list[str]) -> list[str]:
    """An error can quote a DSN partially; redact its password on its own too."""
    out = []
    for v in values:
        pw = urlparse(v).password if "://" in v else None
        if pw:
            out.append(pw)
    return out


def redact(text: str, secrets: list[str]) -> str:
    for s in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(s, "***")
    return text


async def run_checks(
        checks: list[tuple[str, Callable[[], Awaitable[str]]]]) -> list[CheckResult]:
    results: list[CheckResult] = []
    secrets = _secret_values()
    for name, check in checks:
        try:
            detail = await asyncio.wait_for(check(), TIMEOUT)
            results.append(CheckResult(name, True, redact(detail, secrets)))
        except Exception as exc:  # report every failure, never stop at the first
            results.append(CheckResult(
                name, False, redact(f"{type(exc).__name__}: {exc}", secrets)[:300]))
    return results


def report(results: list[CheckResult]) -> tuple[str, int]:
    lines = [f"{'OK  ' if r.ok else 'FAIL'}  {r.name:<12} {r.detail}" for r in results]
    return "\n".join(lines), 0 if all(r.ok for r in results) else 1


async def _http(url: str, verify: bool = True) -> str:
    """Reachable = any HTTP response below 500 (auth errors still prove the route)."""
    async with httpx.AsyncClient(timeout=TIMEOUT, verify=verify) as client:
        resp = await client.get(url)
    if resp.status_code >= 500:
        raise RuntimeError(f"HTTP {resp.status_code} from {url}")
    return f"HTTP {resp.status_code} {url}"


async def _http_200(url: str, *, verify: bool = True,
                    headers: dict[str, str] | None = None) -> str:
    """Strict variant: only HTTP 200 passes. Used for DocExtractor, where a
    401/403 means a rotated or wrong read key -- exactly what the check exists to
    catch before anything is scaled up. Credentials go in `headers` only, never in
    the URL, so neither the success detail nor an httpx error can quote them."""
    async with httpx.AsyncClient(timeout=TIMEOUT, verify=verify) as client:
        resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code} from {url} (expected 200)")
    return f"HTTP 200 {url}"


async def _optional_http(base_url: str) -> str:
    """judge/report/map/rerank/eval-judge/verify default to empty and fall back
    to another tier's client at call time -- an unset base URL is a valid
    deployment choice, not a failure, so skip the probe and say so."""
    if not base_url:
        return "not configured"
    return await _http(base_url.rstrip("/") + "/models")


def _checks() -> list[tuple[str, Callable[[], Awaitable[str]]]]:
    s = get_extract_settings()
    g = get_settings()

    async def neo4j() -> str:
        driver = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
        try:
            await driver.execute_query("RETURN 1")
        finally:
            await driver.close()
        return s.neo4j_uri

    async def postgres() -> str:
        conn = await asyncpg.connect(g.postgres_dsn, timeout=TIMEOUT)
        try:
            await conn.fetchval("SELECT 1")
        finally:
            await conn.close()
        return "SELECT 1 ok"

    return [
        ("neo4j", neo4j),
        ("postgres", postgres),
        ("docextractor", lambda: _http_200(s.docext_base_url.rstrip("/") + "/api/health",
                                           verify=s.docext_verify_tls)),
        # One AUTHENTICATED read: /api/health needs no key, so without this a
        # revoked/rotated read key would pass here and fail on the first poll.
        ("docext-auth", lambda: _http_200(
            s.docext_base_url.rstrip("/") + "/api/articles?limit=1",
            verify=s.docext_verify_tls, headers={"X-API-Key": s.docext_read_key})),
        ("embedder", lambda: _http(s.embed_base_url.rstrip("/") + "/models")),
        ("chunker", lambda: _http(s.chonkie_base_url.rstrip("/") + "/")),
        ("llm", lambda: _http(s.llm_base_url.rstrip("/") + "/models")),
        ("cheap-llm", lambda: _http(s.cheap_llm_base_url.rstrip("/") + "/models")),
        # Optional tiers: empty base URL means "falls back to another tier at
        # call time", not "misconfigured" -- _optional_http reports that as OK.
        ("judge", lambda: _optional_http(s.judge_base_url)),
        ("report", lambda: _optional_http(s.report_llm_base_url)),
        ("map", lambda: _optional_http(s.map_llm_base_url)),
        ("rerank", lambda: _optional_http(s.rerank_base_url)),
        ("eval-judge", lambda: _optional_http(s.eval_judge_base_url)),
        ("verify", lambda: _optional_http(s.verify_llm_base_url)),
    ]


def main() -> None:
    text, code = report(asyncio.run(run_checks(_checks())))
    print(text)
    sys.exit(code)


if __name__ == "__main__":
    main()
