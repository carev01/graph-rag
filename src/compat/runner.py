"""Executes compatibility checks against a target Neo4j.

Two guarantees: (1) one check's failure never aborts the run — every exception is
contained and recorded; (2) a failed CypherCheck is automatically re-run prefixed
`CYPHER 5`, so the report can distinguish "needs a server config change" from
"needs a code change"."""
from __future__ import annotations

import logging

from compat.model import (
    COMPAT_GROUP_ID, CallableCheck, Check, CheckContext, CheckResult, CypherCheck,
    SkipCheck, Status,
)

logger = logging.getLogger(__name__)

_MAX_DETAIL = 300


def compat_target(settings) -> tuple[str, str, str]:
    """(uri, user, password) for the harness target: the compat_* override where
    set, else the main neo4j_* value, resolved field by field."""
    return (settings.compat_neo4j_uri or settings.neo4j_uri,
            settings.compat_neo4j_user or settings.neo4j_user,
            settings.compat_neo4j_password or settings.neo4j_password)


def fabricate_embedding(dim: int) -> list[float]:
    """A deterministic unit-ish vector of `dim` floats. Deterministic so reruns are
    comparable; fabricated so groups 1-7 need no embedder."""
    return [((i % 97) + 1) / 100.0 for i in range(dim)]


def _detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:_MAX_DETAIL]


async def _execute(driver, cypher: str, params: dict) -> list[dict]:
    async with driver.session() as s:
        result = await s.run(cypher, **params)
        return [dict(rec) async for rec in result]


async def run_cypher_check(driver, check: CypherCheck) -> CheckResult:
    try:
        rows = await _execute(driver, check.cypher, check.params)
    except Exception as exc:  # noqa: BLE001 - containment is the point
        detail = _detail(exc)
        retry: Status
        try:
            await _execute(driver, "CYPHER 5 " + check.cypher, check.params)
            retry = "pass"
        except Exception:  # noqa: BLE001
            retry = "fail"
        return CheckResult(name=check.name, group=check.group, status="fail",
                           detail=detail, cypher5_retry=retry,
                           informational=check.informational)
    if check.expect is not None and not check.expect(rows):
        return CheckResult(
            name=check.name, group=check.group, status="fail",
            detail=f"expect predicate rejected rows: {str(rows)[:_MAX_DETAIL]}",
            informational=check.informational)
    return CheckResult(name=check.name, group=check.group, status="pass",
                       detail=str(rows[0])[:_MAX_DETAIL] if rows else "",
                       informational=check.informational)


async def run_callable_check(ctx: CheckContext, check: CallableCheck) -> CheckResult:
    try:
        detail = await check.fn(ctx)
    except SkipCheck as exc:
        return CheckResult(name=check.name, group=check.group, status="skip",
                           detail=str(exc)[:_MAX_DETAIL],
                           informational=check.informational)
    except Exception as exc:  # noqa: BLE001
        # A CallableCheck has no single statement to re-prefix, so a language-version
        # retry is not applicable; cypher5_retry stays None and renders as
        # "n/a (procedural)".
        return CheckResult(name=check.name, group=check.group, status="fail",
                           detail=_detail(exc), informational=check.informational)
    return CheckResult(name=check.name, group=check.group, status="pass",
                       detail=detail[:_MAX_DETAIL], informational=check.informational)


async def run_all(ctx: CheckContext, checks: list[Check]) -> list[CheckResult]:
    """Run every check in registry order. Order matters: group 5 writes the synthetic
    graph that groups 6 and 7 query."""
    results: list[CheckResult] = []
    for check in checks:
        if isinstance(check, CypherCheck):
            res = await run_cypher_check(ctx.driver, check)
        else:
            res = await run_callable_check(ctx, check)
        logger.info("compat check %s/%s -> %s", check.group, check.name, res.status)
        results.append(res)
    return results


async def teardown(driver) -> str | None:
    """Delete every harness artifact. Returns None on success, else an error string
    that the report surfaces as 'manual cleanup required'.

    Deletes are ALWAYS scoped to COMPAT_GROUP_ID — the harness must be safe to run
    against a populated instance."""
    try:
        async with driver.session() as s:
            # COMPAT_GROUP_ID is a hardcoded internal constant (never attacker- or
            # user-controlled), so inlining it is safe and lets tests/operators see
            # the exact scope in the query text rather than a bound parameter.
            await s.run(
                f"MATCH ()-[r {{group_id:'{COMPAT_GROUP_ID}'}}]-() DELETE r")
            await s.run(
                f"MATCH (n {{group_id:'{COMPAT_GROUP_ID}'}}) DETACH DELETE n")
            result = await s.run(
                "SHOW INDEXES YIELD name WHERE name STARTS WITH 'compat_' "
                "RETURN collect(name) AS names")
            rows = [dict(rec) async for rec in result]
            for name in (rows[0]["names"] if rows else []):
                await s.run(f"DROP INDEX {name} IF EXISTS")
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("compat teardown failed", exc_info=True)
        return _detail(exc)
