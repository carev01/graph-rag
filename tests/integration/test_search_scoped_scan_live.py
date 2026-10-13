"""Live: a Cohesity-scoped search is no longer empty next to Veeam's 20k articles.

Read-only against the live graph; embeds the query (TEI), calls no LLM."""
import re

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_cohesity_scoped_search_finds_datalock_live(live_extract_driver):
    from answer_api.scope import Scope
    from answer_api.search import search_local
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_graphiti
    s = get_extract_settings()
    graphiti = build_graphiti(s)
    try:
        res = await search_local(
            graphiti, live_extract_driver,
            q="Which backup vendors support immutable backups?",
            scope=Scope(vendors=("Cohesity",), source="explicit"), group_id=s.group_id,
            fetch_cap=s.local_scoped_fetch_cap,
            scan_max_episodes=s.local_scoped_scan_max_episodes)
        facts = [r["fact"] for r in res["results"]]
        assert facts, "Cohesity-scoped search returned no facts"
        assert any(re.search(r"data\s?lock|worm", f, re.IGNORECASE) for f in facts), facts
    finally:
        await graphiti.close()
