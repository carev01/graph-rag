import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_incremental_no_change_regenerates_nothing_live(live_extract_driver):
    """The cost guarantee on the real backup-docs graph: with a populated
    :Community layer and no new content, an incremental run regenerates ZERO
    reports (Leiden is seeded, so re-detection reproduces the same partition →
    every community matches + is untouched → all reused). Fast: no LLM calls.

    Assumes the layer was built by the seeded detector (a `theme-build --full`
    or a prior incremental run). A layer built before Leiden was seeded needs one
    reconciling run first.
    """
    import theme_builder.cli as cli
    from graph_extract.config import get_extract_settings
    s = get_extract_settings()
    res = await cli._run_theme_build_incremental(s, driver=live_extract_driver)
    assert res["reports_regenerated"] == 0, res
    assert res["reports_reused"] > 0, res
