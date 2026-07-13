"""Live end-to-end verification against the real DocExtractor cluster and a
local Neo4j/Postgres brought up via `docker compose up -d`.

Opt-in only: excluded from the default `pytest` lane by `addopts = "-m 'not
live'"` in pyproject.toml. Run explicitly with:

    uv run pytest tests/e2e/test_live.py -m live -v

Requires a filled-in `.env` (copied from `.env.example`) with real
DOCEXT_READ_KEY / DOCEXT_ADMIN_KEY values and reachable Neo4j/Postgres.
"""
from __future__ import annotations

import pytest

from graph_sync.catalog import Catalog
from graph_sync.config import get_settings
from graph_sync.delta_client import make_client
from graph_sync.neo4j_repo import Neo4jRepo
from graph_sync.state_store import StateStore
from graph_sync.sync_core import SyncCore

pytestmark = [pytest.mark.live, pytest.mark.asyncio]

AWS_SRC = "21632f3b-5a4c-4c93-9f00-6701d0e9f677"
NESTED_ARTICLE_ID = "2c92266f-f84c-49c2-9e89-3b4376ec9043"


async def test_live_bootstrap_reconciles_and_builds_chapters():
    s = get_settings()
    client = make_client(s)
    cat = Catalog(client)
    await cat.load()
    repo = Neo4jRepo(s.neo4j_uri, s.neo4j_user, s.neo4j_password)
    await repo.init_schema()
    store = StateStore(s.postgres_dsn)
    await store.init_schema()
    try:
        core = SyncCore(client, cat, repo, store, s)
        res = await core.bootstrap(source_id=AWS_SRC)
        # Brief expects ~146; tolerate small drift in either direction since
        # the live cluster's content can change between task authoring and
        # this run.
        assert res.applied >= 140

        # Reconcile against the live dashboard's authoritative count for
        # this source, rather than hardcoding an expected number.
        dash = (await client.get("/api/dashboard/sources")).json()["sources"]
        expected = next(x["article_count"] for x in dash if x["id"] == AWS_SRC)
        actual = await repo.article_count_by_source(AWS_SRC)
        assert actual == expected, (
            f"graph has {actual} articles for source {AWS_SRC}, "
            f"dashboard reports {expected}"
        )

        # Authoritative chapters: a known nested article is linked to a
        # chapter node (TOC pass ran during bootstrap).
        assert await repo.article_chapter_id(NESTED_ARTICLE_ID) is not None

        # Idempotent replay: re-running bootstrap against unchanged content
        # applies nothing (content-hash gating).
        res2 = await core.bootstrap(source_id=AWS_SRC)
        assert res2.applied == 0
    finally:
        await repo.close()
        await store.close()
        await client.aclose()
