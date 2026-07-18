import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


class _FakeEmbedder:
    def __init__(self, vec): self._vec = vec
    async def create_batch(self, texts): return [self._vec for _ in texts]


async def test_shortlist_ranks_and_filters_by_level(extract_driver):
    from answer_api.global_search import shortlist_communities
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (:Community {group_id:$g, level:1, community_id:'near', title:'Near', "
                    "summary:'x', rating:6.0, cited_fact_uuids:['f1'], full_report:'[]', embedding:[1.0,0.0]})", g=g)
        await s.run("CREATE (:Community {group_id:$g, level:1, community_id:'far', title:'Far', "
                    "summary:'x', rating:9.0, cited_fact_uuids:['f2'], full_report:'[]', embedding:[0.0,1.0]})", g=g)
        await s.run("CREATE (:Community {group_id:$g, level:0, community_id:'wronglvl', title:'L0', "
                    "summary:'x', rating:9.0, cited_fact_uuids:['f3'], full_report:'[]', embedding:[1.0,0.0]})", g=g)
    hits = await shortlist_communities(extract_driver, _FakeEmbedder([1.0, 0.0]), "q",
                                       level=1, k=5, group_id=g)
    assert [h.community_id for h in hits] == ["near", "far"]   # level-1 only, near first


async def test_global_search_end_to_end(extract_driver):
    from answer_api.global_search import global_search
    import json
    g = "backup-docs"
    # seed a community + a real fact whose provenance resolves to a URL
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")   # isolate: shared module-scoped container
        await s.run("CREATE (a:Article {id:'art1', source_url:'https://x/art1', title:'T'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1', group_id:$g})", g=g)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1'], "
                    "fact:'AWS Backup supports S3'}]->(y:Entity)", g=g)
        await s.run("CREATE (:Community {group_id:$g, level:1, community_id:'c1', title:'S3 backup', "
                    "summary:'s', rating:8.0, cited_fact_uuids:['f1'], full_report:'[]', embedding:[1.0,0.0]})", g=g)

    class _Emb:
        async def create_batch(self, texts): return [[1.0, 0.0] for _ in texts]

    class _MapClient:   # returns a valid map result citing f1
        def __init__(self):
            self.chat = self
            self.completions = self
        async def create(self, **kw):
            c = json.dumps({"relevance": 9, "key_points": ["S3 supported"], "fact_ids": ["f1"]})
            return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": c})()})()]})

    class _ReduceClient:  # cites [1] and (badly) writes a URL + an invalid marker
        def __init__(self):
            self.chat = self
            self.completions = self
        async def create(self, **kw):
            c = "AWS Backup supports S3 [1]. See https://evil/x [9]."
            return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": c})()})()]})

    res = await global_search(extract_driver, _Emb(), _MapClient(), "mm", _ReduceClient(), "rm",
                              q="how is S3 backed up", level=1, k=5, group_id=g, relevance_min=2)
    assert res["citations"][0]["fact_uuid"] == "f1"
    # Provenance.resolve_citations returns sources keyed {url, title, article_id}
    assert res["citations"][0]["sources"][0]["url"] == "https://x/art1"          # #2 chain
    assert "http" not in res["answer"]                     # URL stripped
    assert [c["marker"] for c in res["citations"]] == [1]  # invalid [9] dropped
    assert res["communities_used"][0]["community_id"] == "c1"


async def test_global_search_empty_shortlist_refuses(extract_driver):
    from answer_api.global_search import global_search, _REFUSAL
    g = "backup-docs-empty"

    class _Emb:
        async def create_batch(self, texts): return [[1.0, 0.0] for _ in texts]

    class _Boom:   # must NOT be called (no communities -> no LLM)
        def __init__(self):
            self.chat = self
            self.completions = self
        async def create(self, **kw): raise AssertionError("no LLM on empty shortlist")

    res = await global_search(extract_driver, _Emb(), _Boom(), "mm", _Boom(), "rm",
                              q="x", level=1, k=5, group_id=g, relevance_min=2)
    assert res["answer"] == _REFUSAL and res["citations"] == []


async def test_global_search_all_maps_filtered_refuses(extract_driver):
    # Shortlist is non-empty, but every map result is below the relevance floor,
    # so `results` is empty -> refusal, and the reduce tier must NOT be called.
    from answer_api.global_search import global_search, _REFUSAL
    import json
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (:Community {group_id:$g, level:1, community_id:'c1', title:'S3', "
                    "summary:'s', rating:8.0, cited_fact_uuids:['f1'], full_report:'[]', embedding:[1.0,0.0]})", g=g)

    class _Emb:
        async def create_batch(self, texts): return [[1.0, 0.0] for _ in texts]

    class _LowRelevanceMap:   # shortlisted + called, but relevance below the floor -> None
        def __init__(self):
            self.chat = self
            self.completions = self
        async def create(self, **kw):
            c = json.dumps({"relevance": 0, "key_points": ["x"], "fact_ids": ["f1"]})
            return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": c})()})()]})

    class _Boom:   # the reduce tier must NOT run when nothing survives the map
        def __init__(self):
            self.chat = self
            self.completions = self
        async def create(self, **kw): raise AssertionError("no reduce call when all maps filtered")

    res = await global_search(extract_driver, _Emb(), _LowRelevanceMap(), "mm", _Boom(), "rm",
                              q="x", level=1, k=5, group_id=g, relevance_min=2)
    assert res["answer"] == _REFUSAL
    assert res["citations"] == [] and res["communities_used"] == []
