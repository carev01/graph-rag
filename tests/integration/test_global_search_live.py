import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_global_search_live(live_extract_driver):
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_embedder
    from answer_api.synthesize import _synthesis_client_and_model
    from answer_api.global_search import global_search, _map_client_and_model
    s = get_extract_settings()
    emb = build_embedder(s)
    mc, mm = _map_client_and_model(s)
    sc, sm = _synthesis_client_and_model(s)
    try:
        res = await global_search(
            live_extract_driver, emb, mc, mm, sc, sm,
            q="How do AWS Backup and Azure Backup handle backup retention?",
            level=s.global_default_level, k=s.global_shortlist_k,
            group_id=s.group_id, relevance_min=s.global_map_relevance_min)
        assert res["answer"] and res["communities_used"]
        assert "http" not in res["answer"]                       # no LLM-authored URL
        for c in res["citations"]:
            assert c["sources"], "cited fact must resolve to >=1 source"
    finally:
        await mc.close()
        await sc.close()
