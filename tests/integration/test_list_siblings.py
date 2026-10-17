"""The relation-expansion sibling fetch (spec 2026-10-15 §10.1, Task 8) against a real
Neo4j: an unordered bounded fetch, then the scope filter (one bounded episode lookup)
and the uuid sort + cap in Python -- deterministic for a static graph, the scope
filter applied before the cap. Fixture: one subject entity with three current
`SUPPORTS` edges -- two from Veeam episodes, one from a Cohesity episode that sorts
first -- plus one invalidated edge. The keyword sweep's full-text query (spec §12.1) runs
against the same container, on the index graphiti creates, and so does the compact
INTEGER8 embedding fetch (Task 13)."""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
from neo4j.vector import Vector, VectorDType

import answer_api.fact_dedup as fd
import answer_api.list_mode as lm
import answer_api.list_sweep as ls
from answer_api.scope import Scope

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def _seed(driver):
    async with driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run(
            "CREATE (:Vendor {id:'v1', name:'Veeam'})-[:HAS_PRODUCT]->"
            "(:Product {id:'p1', name:'VBR'})-[:HAS_SOURCE]->(:Source {id:'s1', name:'VBR UG'})"
            "-[:HAS_ARTICLE]->(:Article {id:'a1'})-[:HAS_EPISODE]->(:Episodic {uuid:'ev1'})")
        await s.run(
            "CREATE (:Vendor {id:'v2', name:'Cohesity'})-[:HAS_PRODUCT]->"
            "(:Product {id:'p2', name:'DataProtect'})-[:HAS_SOURCE]->"
            "(:Source {id:'s2', name:'DP Guide'})"
            "-[:HAS_ARTICLE]->(:Article {id:'a2'})-[:HAS_EPISODE]->(:Episodic {uuid:'ec1'})")
        await s.run(
            "CREATE (a:Entity {uuid:'subj'}) "
            "CREATE (a)-[:RELATES_TO {uuid:'f0', name:'SUPPORTS', fact:'c', episodes:['ec1']}]"
            "->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'f2', name:'SUPPORTS', fact:'v2', episodes:['ev1']}]"
            "->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'f1', name:'SUPPORTS', fact:'v1', "
            "episodes:['ec1', 'ev1']}]->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'f3', name:'SUPPORTS', fact:'old', episodes:['ev1'], "
            "invalid_at: datetime('2020-01-01T00:00:00Z')}]->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'f4', name:'OTHER', fact:'o', episodes:['ev1']}]"
            "->(:Entity)")


async def test_the_siblings_query_is_bounded_and_skips_invalid_and_other_relations(
        extract_driver):
    await _seed(extract_driver)
    rows, _, _ = await extract_driver.execute_query(
        lm.SIBLINGS_QUERY, src="subj", rel="SUPPORTS", fetch=10)
    assert sorted(r["uuid"] for r in rows) == ["f0", "f1", "f2"]
    rows, _, _ = await extract_driver.execute_query(
        lm.SIBLINGS_QUERY, src="subj", rel="SUPPORTS", fetch=2)
    assert len(rows) == 2


async def test_unscoped_siblings_are_ordered_and_capped(extract_driver):
    await _seed(extract_driver)
    for _ in range(3):                                       # same answer every run
        rows = await lm._group_siblings(extract_driver, "subj", "SUPPORTS",
                                        scope=Scope(), cap=2)
        assert [r["uuid"] for r in rows] == ["f0", "f1"]


async def test_scoped_siblings_filter_before_the_cap(extract_driver):
    await _seed(extract_driver)
    rows = await lm._group_siblings(extract_driver, "subj", "SUPPORTS",
                                    scope=Scope(vendors=("Veeam",)), cap=2)
    assert [r["uuid"] for r in rows] == ["f1", "f2"]          # Cohesity-only f0 excluded
    rows = await lm._group_siblings(extract_driver, "subj", "SUPPORTS",
                                    scope=Scope(sources=("DP Guide",)), cap=5)
    assert [r["uuid"] for r in rows] == ["f0", "f1"]
    rows = await lm._group_siblings(extract_driver, "subj", "SUPPORTS",
                                    scope=Scope(products=("vbr",)), cap=1)
    assert [r["uuid"] for r in rows] == ["f1"]


# graphiti's own definition of the index the sweep queries.
_FULLTEXT = ("CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS "
             "FOR ()-[e:RELATES_TO]-() ON EACH [e.name, e.fact, e.group_id]")
_SWEEP_SETTINGS = SimpleNamespace(list_sweep_cap=10, list_sweep_scoped_overfetch=4)


async def _seed_sweep(driver):
    await _seed(driver)
    async with driver.session() as s:
        await s.run(_FULLTEXT)
        await s.run("CALL db.awaitIndexes(60)")
        await s.run(
            "CREATE (a:Entity {uuid:'sw'}) "
            "CREATE (a)-[:RELATES_TO {uuid:'plural', name:'RESTORES_TO', group_id:'g', "
            "fact:'Backups created by Veeam Agents can be restored to Proxmox VE.', "
            "episodes:['ev1']}]->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'hyperv', name:'RESTORES_TO', group_id:'g', "
            "fact:'Instant Recovery to Hyper-V hosts is supported.', episodes:['ev1']}]"
            "->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'cohesity', name:'RESTORES_TO', group_id:'g', "
            "fact:'The Cohesity agent restores to Nutanix AHV.', episodes:['ec1']}]"
            "->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'invalid', name:'RESTORES_TO', group_id:'g', "
            "fact:'The Veeam agent restored to Xen.', episodes:['ev1'], "
            "invalid_at: datetime('2020-01-01T00:00:00Z')}]->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'othergroup', name:'RESTORES_TO', "
            "group_id:'other', fact:'The Veeam agent restores to Oracle KVM.', "
            "episodes:['ev1']}]->(:Entity)")


async def test_the_sweep_returns_only_current_same_group_edges(extract_driver):
    await _seed_sweep(extract_driver)
    got = await ls.keyword_sweep(extract_driver, "Which hypervisors does the agent support?",
                                 scope=None, group_id="g", settings=_SWEEP_SETTINGS)
    # "agents" is found for the term "agent"; the invalidated and other-group edges never.
    assert sorted(f["fact_uuid"] for f in got) == ["cohesity", "plural"]
    plural = next(f for f in got if f["fact_uuid"] == "plural")
    assert plural["sources"][0]["vendor"] == "Veeam"          # real provenance chain


async def test_a_hyphenated_term_is_searched_without_error(extract_driver):
    await _seed_sweep(extract_driver)
    assert ls.question_terms("Which targets can Hyper-V restores reach?", None) == [
        "hyper-v", "restores", "reach"]
    got = await ls.keyword_sweep(extract_driver, "Which targets can Hyper-V restores reach?",
                                 scope=None, group_id="g", settings=_SWEEP_SETTINGS)
    uuids = [f["fact_uuid"] for f in got]
    assert "hyperv" in uuids
    assert "invalid" not in uuids and "othergroup" not in uuids


async def test_a_scoped_sweep_keeps_only_in_scope_edges(extract_driver):
    await _seed_sweep(extract_driver)
    got = await ls.keyword_sweep(extract_driver, "Which hypervisors does the agent support?",
                                 scope=Scope(vendors=("Veeam",)), group_id="g",
                                 settings=_SWEEP_SETTINGS)
    assert [f["fact_uuid"] for f in got] == ["plural"]


def _cos(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


async def test_fetched_embeddings_keep_the_float_cosine(extract_driver, monkeypatch):
    monkeypatch.setattr(fd, "_compact_unavailable", False)
    rng = random.Random(5)
    e1 = [rng.gauss(0, 0.05) for _ in range(1024)]
    e2 = [x + rng.gauss(0, 0.02) for x in e1]                 # a near-duplicate
    e3 = [rng.gauss(0, 0.05) for _ in range(1024)]            # unrelated
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run(
            "CREATE (a:Entity {uuid:'emb'}) "
            "CREATE (a)-[:RELATES_TO {uuid:'e1', fact_embedding:$e1}]->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'e2', fact_embedding:$e2}]->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'e3', fact_embedding:$e3}]->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'none'}]->(:Entity) "
            "CREATE (a)-[:RELATES_TO {uuid:'zero', fact_embedding:[0.0, 0.0]}]->(:Entity)",
            e1=e1, e2=e2, e3=e3)
    cache: dict = {}
    got = await fd.fetch_vectors(extract_driver, ["e1", "e2", "e3", "none", "zero"], cache)
    assert sorted(got) == ["e1", "e2", "e3"]
    assert sorted(cache) == ["e1", "e2", "e3", "none", "zero"]
    assert all(v.dtype == np.float32 and v.shape == (1024,) for v in got.values())
    floats = {"e1": e1, "e2": e2, "e3": e3}
    assert _cos(e1, e2) > 0.9                                # the pair worth deduplicating
    for a, b in (("e1", "e2"), ("e1", "e3")):
        assert abs(_cos(got[a], got[b]) - _cos(floats[a], floats[b])) < 0.01
    # Neo4j 2025+ (the fixture image, as production) must take the compact path: a
    # silent fallback would pass the cosine check above and lose the whole speed-up.
    info = await extract_driver.get_server_info()
    major = int(info.agent.split("/")[1].split(".")[0])
    if major >= 2025:
        assert fd._compact_unavailable is False
        rows, _, _ = await extract_driver.execute_query(fd.EMBEDDINGS_QUERY, uuids=["e1"])
        assert isinstance(rows[0]["embedding"], Vector)
        assert rows[0]["embedding"].dtype == VectorDType.I8
