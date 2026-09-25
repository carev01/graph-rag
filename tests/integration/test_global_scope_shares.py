"""`_scope_shares` (spec §4.3 step 2, BACKLOG 52): each community's in-scope
share of its cited_fact_uuids, computed by graph traversal over the SAME
provenance chain Provenance.resolve_citations walks -- fact -> episode ->
article -> source -> product -> vendor. Fixture: two vendors, a community
citing 3 Veeam facts + 1 Cohesity fact, and a community with no cited facts."""
from __future__ import annotations

import pytest

from answer_api.global_search import CommunityHit, _scope_shares
from answer_api.scope import Scope

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _hit(cid, facts):
    return CommunityHit(cid, cid, "sum", 1, 7.0, list(facts), "[]", 0.9)


async def test_scope_shares_veeam_cohesity_mix(extract_driver):
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run(
            "CREATE (:Vendor {id:'v1', name:'Veeam'})-[:HAS_PRODUCT]->"
            "(:Product {id:'p1', name:'VBR'})-[:HAS_SOURCE]->(:Source {id:'s1'})"
            "-[:HAS_ARTICLE]->(:Article {id:'a1'})-[:HAS_EPISODE]->"
            "(:Episodic {uuid:'ev1'})")
        await s.run(
            "CREATE (:Vendor {id:'v2', name:'Cohesity'})-[:HAS_PRODUCT]->"
            "(:Product {id:'p2', name:'DataProtect'})-[:HAS_SOURCE]->(:Source {id:'s2'})"
            "-[:HAS_ARTICLE]->(:Article {id:'a2'})-[:HAS_EPISODE]->"
            "(:Episodic {uuid:'ec1'})")
        # 3 Veeam facts + 1 Cohesity fact, all cited by community 'mix'
        for fid, epi in [("f1", "ev1"), ("f2", "ev1"), ("f3", "ev1"), ("f4", "ec1")]:
            await s.run(
                "CREATE (x:Entity)-[:RELATES_TO {group_id:'g', uuid:$u, "
                "episodes:[$e], fact:'fact'}]->(y:Entity)", u=fid, e=epi)

    hits = [_hit("mix", ["f1", "f2", "f3", "f4"]), _hit("empty", [])]

    veeam_scope = Scope(("Veeam",), (), "explicit")
    shares = await _scope_shares(extract_driver, "g", hits, veeam_scope)
    assert shares["mix"] == pytest.approx(0.75)
    assert shares["empty"] == 0.0

    cohesity_by_product = Scope((), ("DataProtect",), "explicit")
    shares2 = await _scope_shares(extract_driver, "g", hits, cohesity_by_product)
    assert shares2["mix"] == pytest.approx(0.25)
    assert shares2["empty"] == 0.0

    fortknox_scope = Scope((), ("FortKnox",), "explicit")
    shares3 = await _scope_shares(extract_driver, "g", hits, fortknox_scope)
    assert shares3["mix"] == 0.0


async def test_scope_shares_case_insensitive(extract_driver):
    """R3: names arriving in raw API-param casing must still match the
    canonically-cased structural names."""
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run(
            "CREATE (:Vendor {id:'v1', name:'Veeam'})-[:HAS_PRODUCT]->"
            "(:Product {id:'p1', name:'VBR'})-[:HAS_SOURCE]->(:Source {id:'s1'})"
            "-[:HAS_ARTICLE]->(:Article {id:'a1'})-[:HAS_EPISODE]->"
            "(:Episodic {uuid:'ev1'})")
        await s.run(
            "CREATE (x:Entity)-[:RELATES_TO {group_id:'g', uuid:'f1', "
            "episodes:['ev1'], fact:'fact'}]->(y:Entity)")

    hits = [_hit("c1", ["f1"])]
    shares = await _scope_shares(extract_driver, "g", hits, Scope(("veeam",), (), "x"))
    assert shares["c1"] == 1.0
    shares2 = await _scope_shares(extract_driver, "g", hits, Scope((), ("vbr",), "x"))
    assert shares2["c1"] == 1.0


async def test_scope_shares_no_hits_returns_empty_dict(extract_driver):
    assert await _scope_shares(extract_driver, "g", [], Scope(("Veeam",), (), "x")) == {}
