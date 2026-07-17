import pytest

from answer_api.synthesize import _REFUSAL

pytestmark = pytest.mark.asyncio


async def test_answer_local_synthesizes_with_citations(monkeypatch):
    import answer_api.search as search_mod

    async def _fake_search(*a, **k):
        return {"query": "q", "count": 2, "results": [
            {"fact": "immutability", "fact_uuid": "f1", "valid_at": None,
             "sources": [{"url": "u1", "title": "t1", "article_id": "a1"}]},
            {"fact": "cross-region copy", "fact_uuid": "f2", "valid_at": None,
             "sources": [{"url": "u2", "title": "t2", "article_id": "a2"}]}]}
    monkeypatch.setattr(search_mod, "search_local", _fake_search)

    class _Msg:
        content = "Immutable [1] and cross-region [2]."

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Chat:
        class completions:
            @staticmethod
            async def create(**kw):
                return _Resp()

    class _StubGLM:
        chat = _Chat()

    from answer_api.synthesize import answer_local
    out = await answer_local(object(), object(), _StubGLM(), "glm", q="q", group_id="g")
    assert out["cited"] == 2
    assert out["citations"][0]["sources"][0]["article_id"] == "a1"
    assert out["citations"][1]["sources"][0]["article_id"] == "a2"
    assert "http" not in out["answer"]
    assert out["retrieved"] == 2


async def test_answer_local_url_guard_end_to_end(monkeypatch):
    import answer_api.search as search_mod

    async def _fake_search(*a, **k):
        return {"query": "q", "count": 2, "results": [
            {"fact": "immutability", "fact_uuid": "f1", "valid_at": None,
             "sources": [{"url": "u1", "title": "t1", "article_id": "a1"}]},
            {"fact": "cross-region copy", "fact_uuid": "f2", "valid_at": None,
             "sources": [{"url": "u2", "title": "t2", "article_id": "a2"}]}]}
    monkeypatch.setattr(search_mod, "search_local", _fake_search)

    class _Msg:
        content = "See https://evil/x [1]."

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Chat:
        class completions:
            @staticmethod
            async def create(**kw):
                return _Resp()

    class _StubGLM:
        chat = _Chat()

    from answer_api.synthesize import answer_local
    out = await answer_local(object(), object(), _StubGLM(), "glm", q="q", group_id="g")
    assert "http" not in out["answer"]
    assert [c["marker"] for c in out["citations"]] == [1]
    assert out["cited"] == 1


async def test_answer_local_zero_retrieval_short_circuits(monkeypatch):
    import answer_api.search as search_mod

    async def _fake_search(*a, **k):
        return {"query": "q", "count": 0, "results": []}
    monkeypatch.setattr(search_mod, "search_local", _fake_search)

    class _RaisingCompletions:
        @staticmethod
        async def create(**kw):
            raise AssertionError("synth client must not be called on zero retrieval")

    class _Chat:
        completions = _RaisingCompletions()

    class _StubGLM:
        chat = _Chat()

    from answer_api.synthesize import answer_local
    out = await answer_local(object(), object(), _StubGLM(), "glm", q="q", group_id="g")
    assert out["answer"] == _REFUSAL
    assert out["citations"] == []
    assert out["cited"] == 0
    assert out["retrieved"] == 0


@pytest.mark.live
async def test_answer_local_live_smoke(live_extract_driver):
    """Live GLM synthesis smoke test against the real compose Neo4j/Graphiti
    (already-bootstrapped AWS Backup graph). Opt-in only: excluded from the
    default `-m "not live"` lane.
    """
    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import build_graphiti
    from answer_api.synthesize import _synthesis_client_and_model, answer_local

    settings = get_extract_settings()
    graphiti = build_graphiti(settings)
    client, model = _synthesis_client_and_model(settings)
    try:
        out = await answer_local(
            graphiti, live_extract_driver, client, model,
            q="What does AWS Backup Vault Lock require?", k=5,
            group_id=settings.group_id)
        assert out["answer"]
        assert len(out["citations"]) >= 1
        assert any(c["sources"] for c in out["citations"])
        assert "http" not in out["answer"]

        off_corpus = await answer_local(
            graphiti, live_extract_driver, client, model,
            q="What is the capital of France?", k=5,
            group_id=settings.group_id)
        assert off_corpus["answer"] == _REFUSAL or off_corpus["cited"] == 0
    finally:
        await graphiti.close()
