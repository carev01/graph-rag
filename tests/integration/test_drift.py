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
