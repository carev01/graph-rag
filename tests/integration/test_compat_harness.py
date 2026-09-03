import pytest

from compat.checks import all_checks
from compat.model import CheckContext
from compat.report import render, verdict
from compat.runner import fabricate_embedding, run_all, teardown
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_graphiti

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_harness_runs_end_to_end_against_a_testcontainer(
        extract_driver, extract_neo4j):
    """The harness must produce a complete result set and a rendered report against a
    real Neo4j, and must SKIP (never fail) the checks whose dependencies are absent
    from this environment."""
    uri, user, password = extract_neo4j
    # Point graphiti at the CONTAINER: the graphiti-* groups reach Neo4j through
    # ctx.graphiti, whose driver comes from settings, not through ctx.driver. The
    # checks that build their own Neo4jRepo (structural schema/fixture, e2e) now
    # read ctx.settings.neo4j_uri/user/password directly rather than re-resolving
    # compat_target() themselves, so overriding neo4j_uri/user/password here is
    # sufficient to keep every write inside this test pointed at the container --
    # there is no second override left to clear.
    settings = get_extract_settings().model_copy(update={
        "neo4j_uri": uri, "neo4j_user": user, "neo4j_password": password})
    graphiti = build_graphiti(settings)
    embedding = fabricate_embedding(settings.embed_dim)
    ctx = CheckContext(driver=extract_driver, graphiti=graphiti, settings=settings,
                       embedding=embedding)
    try:
        results = await run_all(ctx, all_checks(embedding))
    finally:
        await teardown(extract_driver)
        await graphiti.close()

    groups = {r.group for r in results}
    assert groups == {"server", "bootstrap", "vector", "fulltext", "graphiti-write",
                      "graphiti-search", "our-cypher", "e2e"}

    # The e2e check only SKIPs on an unreachable LLM/embedder endpoint
    # (httpx.ConnectError/ConnectTimeout, openai.APIConnectionError) -- see
    # compat/checks.py's _ENDPOINT_DOWN. Those endpoints ARE reachable from this
    # dev/CI host (the cheap extraction tier), so the check actually runs the
    # real ingest-then-retrieve path here rather than skipping; it must never
    # silently fail. There is no GDS plugin in this container -> that check SKIPs.
    e2e = [r for r in results if r.group == "e2e"]
    assert all(r.status in ("pass", "skip") for r in e2e), [
        (r.name, r.status, r.detail) for r in e2e]
    gds = [r for r in results if "leiden" in r.name.lower() or "gds" in r.name.lower()]
    assert all(r.status == "skip" for r in gds), [
        (r.name, r.status, r.detail) for r in gds]

    rendered = render(results, target={"uri": "testcontainer"})
    assert "## Verdict:" in rendered
    assert verdict(results) in ("GO", "GO_WITH_CONFIG", "NO_GO")


async def test_teardown_leaves_no_harness_data(extract_driver):
    """Teardown must remove the structural and community nodes too, not just the
    graphiti ones -- they are created through the real write path and would
    otherwise be left behind on a production instance."""
    async with extract_driver.session() as s:
        result = await s.run(
            "MATCH (n {group_id:'compat-check'}) "
            "RETURN labels(n) AS labels, count(n) AS n")
        rows = [dict(rec) async for rec in result]
    assert rows == [], f"residual harness nodes: {rows}"
