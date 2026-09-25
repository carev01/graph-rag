import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_router_niceties_live(live_extract_driver):
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_embedder, build_graphiti
    from answer_api.synthesize import _synthesis_client_and_model
    from answer_api.global_search import _map_client_and_model
    from answer_api.router import answer_router, _cheap_classify_client
    from answer_api.scope import Scope
    s = get_extract_settings()
    graphiti = build_graphiti(s)
    emb = build_embedder(s)
    sc, sm = _synthesis_client_and_model(s)
    mc, mm = _map_client_and_model(s)
    cc, cmodel = _cheap_classify_client(s)
    try:
        # cross-vendor -> global (heuristic); has communities -> reports_as_of set
        env = await answer_router(
            graphiti, live_extract_driver, emb, sc, sm, mc, mm, cc, cmodel,
            q="Compare how AWS Backup and Azure Backup handle retention",
            mode_override=None, scope=Scope(), settings=s)
        assert env["freshness"]["graph_cursor_time"]          # live graph watermark present
        assert env["freshness"]["reports_as_of"]              # global is community-based
        assert "http" not in env["answer"]
        srcs = [src for c in env["citations"] for src in c["sources"]]
        assert srcs, "expected >=1 resolved source"
        assert any(src.get("vendor") or src.get("product") for src in srcs)
    finally:
        await graphiti.close()
        await sc.close()
        await mc.close()
        if cc is not None:
            await cc.close()
