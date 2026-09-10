import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_drift_search_live(live_extract_driver):
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_embedder, build_graphiti
    from answer_api.synthesize import _synthesis_client_and_model
    from answer_api.drift import drift_search
    s = get_extract_settings()
    graphiti = build_graphiti(s)
    emb = build_embedder(s)
    sc, sm = _synthesis_client_and_model(s)
    try:
        res = await drift_search(
            graphiti, live_extract_driver, emb, sc, sm,
            q="What should I consider when planning long-term backup retention across cloud vendors?",
            level=s.drift_primer_level, iterations=s.drift_iterations,
            primer_k=s.drift_primer_k, max_followups=s.drift_max_followups,
            followup_k=s.drift_followup_k, group_id=s.group_id, settings=s)
        assert res["answer"]
        assert "http" not in res["answer"]                    # no LLM-authored URL
        for c in res.get("citations", []):
            assert c["sources"], "cited fact must resolve to >=1 source"
    finally:
        await graphiti.close()
        await sc.close()
