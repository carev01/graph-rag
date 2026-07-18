import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_answer_router_live(live_extract_driver):
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_embedder, build_graphiti
    from answer_api.synthesize import _synthesis_client_and_model
    from answer_api.global_search import _map_client_and_model
    from answer_api.router import answer_router, _cheap_classify_client
    s = get_extract_settings()
    graphiti = build_graphiti(s)
    emb = build_embedder(s)
    sc, sm = _synthesis_client_and_model(s)
    mc, mm = _map_client_and_model(s)
    cc, cmodel = _cheap_classify_client(s)
    try:
        # a temporal question routes to timeline via the heuristic (no LLM needed)
        env = await answer_router(
            graphiti, live_extract_driver, emb, sc, sm, mc, mm, cc, cmodel,
            q="How has AWS Backup vault lock changed over time?",
            mode_override=None, vendor=None, settings=s)
        assert env["mode"] == "timeline"
        assert env["routing"]["via"] == "heuristic"
        assert "http" not in env["answer"]                    # timeline render authored no URL
        for c in env["citations"]:
            assert c["sources"], "cited fact must resolve to >=1 source"
    finally:
        await graphiti.close()
        await sc.close()
        await mc.close()
        if cc is not None:
            await cc.close()
