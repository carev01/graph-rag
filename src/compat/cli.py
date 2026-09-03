"""Run the Neo4j compatibility harness against the COMPAT_*-resolved target and
write docs/superpowers/neo4j-compat-report.md.

    uv run --extra dev python -m compat.cli
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from neo4j import AsyncGraphDatabase

from compat import report as report_mod
from compat.checks import all_checks
from compat.model import CheckContext
from compat.runner import compat_target, fabricate_embedding, run_all, teardown
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_graphiti

logger = logging.getLogger(__name__)
_OUT = Path(__file__).resolve().parents[2] / "docs" / "superpowers" / "neo4j-compat-report.md"


async def _target_facts(driver, uri: str) -> dict[str, str]:
    facts = {"uri": uri}
    probes = {
        "kernel": ("CALL dbms.components() YIELD name, versions, edition "
                   "WHERE name = 'Neo4j Kernel' "
                   "RETURN versions[0] + ' (' + edition + ')' AS v"),
        "default Cypher language": (
            "SHOW SETTINGS YIELD name, value "
            "WHERE name = 'db.query.default_language' RETURN value AS v"),
        "gds": "CALL gds.version() YIELD gdsVersion RETURN gdsVersion AS v",
    }
    for label, cypher in probes.items():
        try:
            async with driver.session() as s:
                result = await s.run(cypher)
                rows = [dict(rec) async for rec in result]
            facts[label] = str(rows[0]["v"]) if rows else "unknown"
        except Exception:  # noqa: BLE001
            facts[label] = "unavailable"
    return facts


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_extract_settings()
    uri, user, password = compat_target(settings)
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    try:
        await driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        await driver.close()
        print(f"cannot reach the target Neo4j at {uri}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    compat_settings = settings.model_copy(update={
        "neo4j_uri": uri, "neo4j_user": user, "neo4j_password": password})
    graphiti = build_graphiti(compat_settings)
    ctx = CheckContext(driver=driver, graphiti=graphiti, settings=compat_settings,
                       embedding=fabricate_embedding(settings.embed_dim))
    try:
        try:
            results = await run_all(ctx, all_checks(ctx.embedding))
        finally:
            # Always run teardown, even on KeyboardInterrupt during a slow live run
            # — otherwise compat-check data is left behind on a production instance.
            teardown_error = await teardown(driver)
        facts = await _target_facts(driver, uri)
        rendered = report_mod.render(results, target=facts,
                                     teardown_error=teardown_error)
        _OUT.write_text(rendered)
        print(rendered)
        print(f"\nwrote {_OUT}")
    finally:
        await graphiti.close()
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
