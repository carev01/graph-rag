import json
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

GROUP_ID = "backup-docs"


class _FakeEmbedder:
    def __init__(self, vec):
        self._vec = vec
    async def create_batch(self, texts):
        return [self._vec for _ in texts]


class _FakeLLM:
    def __init__(self, contents):
        self._c = list(contents)
        self.chat = self
        self.completions = self
    async def create(self, **kw):
        c = self._c.pop(0)
        return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": c})()})()]})


async def _seed_community(driver, cid="c1", emb=(1.0, 0.0)):
    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (:Community {group_id:$g, level:1, community_id:$cid, title:'S3', "
                    "summary:'s', rating:8.0, cited_fact_uuids:['f1'], full_report:'[]', "
                    "embedding:$e})", g=GROUP_ID, cid=cid, e=list(emb))


async def test_primer_shortlists_and_budgets(extract_driver):
    from answer_api.drift import _primer
    await _seed_community(extract_driver)
    payload = json.dumps({"preliminary_answer": "draft",
                          "follow_ups": [{"query": "how retained", "community_id": "c1", "relevance": 9},
                                         {"query": "extra", "community_id": None, "relevance": 1}]})
    out = await _primer(_FakeEmbedder([1.0, 0.0]), _FakeLLM([payload]), "m", extract_driver,
                        q="retention", level=1, k=5, max_followups=1, group_id=GROUP_ID)
    assert out is not None
    preliminary, fus, hits = out
    assert preliminary == "draft"
    assert [f.query for f in fus] == ["how retained"]     # budgeted to 1
    assert fus[0].community_id == "c1"
    assert hits[0].community_id == "c1"


async def test_primer_empty_shortlist_signals_degrade(extract_driver):
    from answer_api.drift import _primer
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")   # no communities
    out = await _primer(_FakeEmbedder([1.0, 0.0]), _FakeLLM([]), "m", extract_driver,
                        q="q", level=1, k=5, max_followups=4, group_id=GROUP_ID)
    assert out is None


async def test_primer_bad_json_falls_back_to_single_query(extract_driver):
    from answer_api.drift import _primer
    await _seed_community(extract_driver)
    out = await _primer(_FakeEmbedder([1.0, 0.0]), _FakeLLM(["nope", "still nope"]), "m",
                        extract_driver, q="original q", level=1, k=5, max_followups=4,
                        group_id=GROUP_ID)
    assert out is not None
    preliminary, fus, hits = out
    assert preliminary == ""                              # unparseable -> empty draft
    assert [(f.query, f.community_id) for f in fus] == [("original q", None)]   # single fallback


async def test_top_member_entity_picks_highest_degree(extract_driver):
    from answer_api.drift import _top_member_entity
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (c:Community {group_id:$g, level:1, community_id:'c1'})", g=GROUP_ID)
        # hub has 2 RELATES_TO, leaf has 1; both members of c1
        await s.run(
            "MATCH (c:Community {community_id:'c1'}) "
            "CREATE (hub:Entity {uuid:'hub'})-[:IN_COMMUNITY]->(c) "
            "CREATE (leaf:Entity {uuid:'leaf'})-[:IN_COMMUNITY]->(c) "
            "CREATE (o1:Entity)-[:RELATES_TO {group_id:$g, uuid:'r1'}]->(hub) "
            "CREATE (hub)-[:RELATES_TO {group_id:$g, uuid:'r2'}]->(o2:Entity) "
            "CREATE (leaf)-[:RELATES_TO {group_id:$g, uuid:'r3'}]->(o3:Entity)", g=GROUP_ID)
    assert await _top_member_entity(extract_driver, GROUP_ID, "c1") == "hub"
    assert await _top_member_entity(extract_driver, GROUP_ID, "nope") is None


async def test_run_followup_biases_by_center_node(extract_driver):
    from answer_api.drift import _run_followup, FollowUp

    class _Edge:
        def __init__(self):
            self.uuid = "f1"
            self.fact = "fact"
            self.episodes = ["ep1"]
            self.valid_at = None
            self.invalid_at = None

    class _Results:
        def __init__(self, edges):
            self.edges = edges

    class _CapGraphiti:
        def __init__(self):
            self.last_center = "unset"
        async def _search(self, query, config, group_ids=None, **kw):
            self.last_center = kw.get("center_node_uuid")
            return _Results([_Edge()])

    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (a:Article {id:'art1', source_url:'https://x/art1', title:'T'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1'})")
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1']}]->(y:Entity)", g=GROUP_ID)
        await s.run("CREATE (c:Community {group_id:$g, level:1, community_id:'c1'}) "
                    "CREATE (m:Entity {uuid:'m1'})-[:IN_COMMUNITY]->(c) "
                    "CREATE (m)-[:RELATES_TO {group_id:$g, uuid:'r1'}]->(:Entity)", g=GROUP_ID)
    g = _CapGraphiti()
    rows = await _run_followup(g, extract_driver, FollowUp("q", "c1", 1), k=8, group_id=GROUP_ID)
    assert g.last_center == "m1"                    # community's top member used as center
    assert rows[0]["fact_uuid"] == "f1"

    g2 = _CapGraphiti()
    await _run_followup(g2, extract_driver, FollowUp("q", None, 1), k=8, group_id=GROUP_ID)
    assert g2.last_center is None                   # untagged -> plain local


async def _seed_fact_provenance(driver, cid="c1"):
    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (a:Article {id:'art1', source_url:'https://x/art1', title:'T'})"
                    "-[:HAS_EPISODE]->(:Episodic {uuid:'ep1', group_id:$g})", g=GROUP_ID)
        await s.run("CREATE (x:Entity)-[:RELATES_TO {group_id:$g, uuid:'f1', episodes:['ep1'], "
                    "fact:'AWS Backup supports S3'}]->(y:Entity)", g=GROUP_ID)
        await s.run("CREATE (c:Community {group_id:$g, level:1, community_id:$cid, title:'S3', "
                    "summary:'s', rating:8.0, cited_fact_uuids:['f1'], full_report:'[]', "
                    "embedding:[1.0,0.0]}) "
                    "CREATE (m:Entity {uuid:'m1'})-[:IN_COMMUNITY]->(c) "
                    "CREATE (m)-[:RELATES_TO {group_id:$g, uuid:'r1'}]->(:Entity)", g=GROUP_ID, cid=cid)


class _Edge:
    def __init__(self, uuid, fact):
        self.uuid = uuid
        self.fact = fact
        self.episodes = ["ep1"]
        self.valid_at = None
        self.invalid_at = None


class _Results:
    def __init__(self, edges):
        self.edges = edges


class _FactGraphiti:               # returns fact f1 for any follow-up search
    async def _search(self, query, config, group_ids=None, **kw):
        return _Results([_Edge("f1", "AWS Backup supports S3")])


async def test_drift_search_end_to_end(extract_driver):
    from answer_api.drift import drift_search
    await _seed_fact_provenance(extract_driver)
    primer = json.dumps({"preliminary_answer": "S3 is supported",
                         "follow_ups": [{"query": "how", "community_id": "c1", "relevance": 9}]})
    synth = "AWS Backup supports S3 [1]. See https://evil/x [9]."   # bad URL + invalid marker
    llm = _FakeLLM([primer, synth])
    res = await drift_search(_FactGraphiti(), extract_driver, _FakeEmbedder([1.0, 0.0]),
                             llm, "m", q="s3 retention", level=1, iterations=1,
                             primer_k=5, max_followups=4, followup_k=8, group_id=GROUP_ID)
    assert res["citations"][0]["fact_uuid"] == "f1"
    assert res["citations"][0]["sources"][0]["url"] == "https://x/art1"
    assert "http" not in res["answer"]                         # URL stripped
    assert [c["marker"] for c in res["citations"]] == [1]      # invalid [9] dropped
    assert res["follow_ups"][0]["community_id"] == "c1"
    assert res["communities_used"][0]["community_id"] == "c1"


async def test_drift_search_empty_shortlist_degrades_to_local(extract_driver):
    from answer_api.drift import drift_search
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")   # no communities
    # No community shortlist -> _primer short-circuits before any LLM call. The
    # degrade path still delegates to answer_local, which runs its own
    # search_local via the (fake) graphiti; _FactGraphiti always returns a fact
    # regardless of driver state, so one synthesis call happens -> 1 canned reply.
    llm = _FakeLLM(["n/a"])
    res = await drift_search(_FactGraphiti(), extract_driver, _FakeEmbedder([1.0, 0.0]),
                             llm, "m", q="q", level=1, iterations=1,
                             primer_k=5, max_followups=4, followup_k=8, group_id=GROUP_ID)
    assert res["degraded"] == "no-primer-communities"


async def test_drift_search_zero_facts_refuses(extract_driver):
    from answer_api.drift import drift_search, _REFUSAL

    class _NoFacts:
        async def _search(self, query, config, group_ids=None, **kw):
            return _Results([])

    await _seed_fact_provenance(extract_driver)
    primer = json.dumps({"preliminary_answer": "d",
                         "follow_ups": [{"query": "how", "community_id": "c1", "relevance": 9}]})

    class _Boom:
        def __init__(self, contents):
            self._c = list(contents)
            self.chat = self
            self.completions = self
        async def create(self, **kw):
            if not self._c:
                raise AssertionError("no synthesis call when zero facts")
            return type("R", (), {"choices": [type("m", (), {"message": type("mm", (), {"content": self._c.pop(0)})()})()]})

    res = await drift_search(_NoFacts(), extract_driver, _FakeEmbedder([1.0, 0.0]),
                             _Boom([primer]), "m", q="q", level=1, iterations=1,
                             primer_k=5, max_followups=4, followup_k=8, group_id=GROUP_ID)
    assert res["answer"] == _REFUSAL and res["citations"] == []


async def test_drift_search_iterations_two_runs_refinement(extract_driver):
    from answer_api.drift import drift_search
    await _seed_fact_provenance(extract_driver)
    primer = json.dumps({"preliminary_answer": "d",
                         "follow_ups": [{"query": "round1", "community_id": "c1", "relevance": 9}]})
    refine = json.dumps({"follow_ups": [{"query": "round2", "community_id": "c1", "relevance": 9}]})
    synth = "Answer [1]."
    # 3 LLM calls expected for iterations=2: primer, refine, synth
    llm = _FakeLLM([primer, refine, synth])
    res = await drift_search(_FactGraphiti(), extract_driver, _FakeEmbedder([1.0, 0.0]),
                             llm, "m", q="q", level=1, iterations=2,
                             primer_k=5, max_followups=4, followup_k=8, group_id=GROUP_ID)
    iters = {f["iteration"] for f in res["follow_ups"]}
    assert iters == {1, 2}                                   # both rounds executed
    assert res["citations"][0]["fact_uuid"] == "f1"


async def test_drift_search_one_followup_raises_does_not_abort(extract_driver):
    from answer_api.drift import drift_search

    class _FlakyGraphiti:   # first follow-up raises, second returns a fact
        def __init__(self):
            self._n = 0
        async def _search(self, query, config, group_ids=None, **kw):
            self._n += 1
            if self._n == 1:
                raise RuntimeError("boom")
            return _Results([_Edge("f1", "AWS Backup supports S3")])

    await _seed_fact_provenance(extract_driver)
    primer = json.dumps({"preliminary_answer": "d", "follow_ups": [
        {"query": "a", "community_id": "c1", "relevance": 9},
        {"query": "b", "community_id": None, "relevance": 8}]})
    synth = "Answer [1]."
    res = await drift_search(_FlakyGraphiti(), extract_driver, _FakeEmbedder([1.0, 0.0]),
                             _FakeLLM([primer, synth]), "m", q="q", level=1, iterations=1,
                             primer_k=5, max_followups=4, followup_k=8, group_id=GROUP_ID)
    assert res["citations"][0]["fact_uuid"] == "f1"          # survived the raising follow-up
