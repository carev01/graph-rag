import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest.mark.live
async def test_router_eval_smoke_live(live_extract_driver):
    import answer_api.eval_router as er
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_embedder, build_graphiti
    from answer_api.synthesize import _synthesis_client_and_model
    from answer_api.global_search import _map_client_and_model
    from answer_api.router import _cheap_classify_client
    from answer_api.scope import ScopeResolver
    s = get_extract_settings()
    graphiti = build_graphiti(s)
    emb = build_embedder(s)
    sc, sm = _synthesis_client_and_model(s)
    mc, mm = _map_client_and_model(s)
    cc, cmodel = _cheap_classify_client(s)
    resolver = await ScopeResolver.load(live_extract_driver)
    # _score_one unpacks 11 clients: the faithfulness judge is independent of
    # synthesis (eval_router._eval_judge_client_and_model refuses a self-judge).
    jc, jm = er._eval_judge_client_and_model(s)
    clients = (graphiti, live_extract_driver, emb, sc, sm, mc, mm, cc, cmodel, jc, jm)
    # one factual + one broad (comparative fires for the broad one) — bounds cost
    questions = [
        {"question": "What does AWS Backup Vault Lock enforce on recovery points?",
         "intent": "local", "expected_modes": ["local"],
         "expected_article_ids": ["dad69d7c-189b-4d53-8d70-40697eae685a"]},
        {"question": "How has Azure Backup soft-delete retention changed over time?",
         "intent": "timeline", "expected_modes": ["timeline"], "expected_article_ids": []},
    ]
    try:
        summary = await er.run_eval(clients, questions, s, resolver)
        assert summary["n"] == 2
        assert summary["routing_accuracy"] >= 0.5
        for r in summary["per_question"]:
            # faithfulness can genuinely be unmeasurable (judge truncation) --
            # that must surface as None, never as a fabricated 0.
            assert r["faithfulness"] is None or 0 <= r["faithfulness"] <= 5
        report = er.format_report(summary)
        assert "Routing accuracy" in report
    finally:
        await graphiti.close()
        await sc.close()
        await mc.close()
        await jc.close()
        await emb.client.close()
        if cc is not None:
            await cc.close()
