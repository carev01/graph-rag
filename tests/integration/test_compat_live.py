import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_compat_harness_live_against_target():
    """Runs the full harness against the COMPAT_*-resolved target and writes
    docs/superpowers/neo4j-compat-report.md."""
    from neo4j import AsyncGraphDatabase

    from compat.checks import all_checks
    from compat.model import CheckContext
    from compat.report import render, verdict
    from compat.runner import compat_target, fabricate_embedding, run_all, teardown
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_graphiti

    settings = get_extract_settings()
    uri, user, password = compat_target(settings)
    driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
    compat_settings = settings.model_copy(update={
        "neo4j_uri": uri, "neo4j_user": user, "neo4j_password": password})
    graphiti = build_graphiti(compat_settings)
    embedding = fabricate_embedding(settings.embed_dim)
    ctx = CheckContext(driver=driver, graphiti=graphiti, settings=compat_settings,
                       embedding=embedding)
    try:
        results = await run_all(ctx, all_checks(embedding))
        teardown_error = await teardown(driver)
        rendered = render(results, target={"uri": uri},
                          teardown_error=teardown_error)
        assert teardown_error is None, teardown_error
        assert verdict(results) != "NO_GO", rendered
    finally:
        await graphiti.close()
        await driver.close()
