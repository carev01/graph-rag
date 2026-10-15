"""Every uvicorn launch runs the standard asyncio loop, never uvloop.

Under uvloop the Neo4j async driver cannot discard a pooled connection that was
dropped while idle: closing it writes to a closed transport, uvloop raises
RuntimeError("... the handler is closed"), and every later query failed until the
pod restarted (2026-10-15, answer-api idle ~15 h). Reproduced locally by aborting a
pooled connection's transport: uvloop fails, asyncio retries and succeeds."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = ["deploy/helm/graph-rag/templates/answer.yaml",
             "deploy/helm/graph-rag/templates/sync.yaml",
             "Dockerfile"]


@pytest.mark.parametrize("path", LAUNCHERS)
def test_uvicorn_runs_the_asyncio_loop(path):
    commands = [line for line in (ROOT / path).read_text().splitlines()
                if '"uvicorn"' in line]
    assert commands, f"no uvicorn command found in {path}"
    for line in commands:
        assert re.search(r'"--loop",\s*"asyncio"', line), line
