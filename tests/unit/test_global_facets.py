"""Global search shortlists per facet and groups the reduce input (spec §3.2)."""
import asyncio
from types import SimpleNamespace

import pytest

import answer_api.global_search as gs
from answer_api.scope import Scope
from answer_api.shape import Facet
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_V = Facet("Veeam", Scope(vendors=("Veeam",)))
_A = Facet("AWS Backup", Scope(products=("AWS Backup",)))
_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")


def _hit(cid, facts):
    return gs.CommunityHit(community_id=cid, title=f"T{cid}", summary="s", level=1,
                           rating=5.0, cited_fact_uuids=facts, full_report="[]",
                           similarity=0.9)


_SRC = {"v1": [{"vendor": "Veeam", "product": "Veeam ONE", "article_id": "a"}],
        "a1": [{"vendor": "AWS", "product": "AWS Backup", "article_id": "b"}]}


class _Reducer:
    def __init__(self):
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="Veeam X [1]. AWS Y [2]."), finish_reason="stop")])


@pytest.fixture
def stubs(monkeypatch):
    calls = SimpleNamespace(shortlist=[], maps=[])

    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        calls.shortlist.append((scope, top_n, k))
        if scope == _V.scope:
            return [_hit("c1", ["v1", "a1"]), _hit("shared", ["v1"])]
        return [_hit("shared", ["v1"]), _hit("c2", ["a1"])]

    async def _map(client, model, q, hit):
        calls.maps.append(hit.community_id)
        return gs.MapResult(community_id=hit.community_id, title=hit.title,
                            relevance=None, key_points=["p"],
                            fact_ids=list(hit.cited_fact_uuids))

    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            return {u: {"sources": _SRC[u]} for u in uuids if u in _SRC}

    async def _texts(driver, group_id, uuids):
        return {u: f"text {u}" for u in uuids}

    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    monkeypatch.setattr(gs, "map_report", _map)
    monkeypatch.setattr(gs, "Provenance", _Prov)
    monkeypatch.setattr(gs, "_fact_texts", _texts)
    return calls


async def test_each_facet_gets_its_own_shortlist(stubs):
    await gs.global_search(None, None, None, "m", _Reducer(), "s", q="q", level=1, k=10,
                           group_id="g", settings=_S, facets=[_V, _A])
    assert stubs.shortlist == [(_V.scope, 3, 3), (_A.scope, 3, 3)]


async def test_map_output_is_filtered_per_facet_and_shared_communities_serve_all(stubs):
    r = _Reducer()
    res = await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                                 group_id="g", settings=_S, facets=[_V, _A])
    p = r.prompts[0]
    assert p.index("=== Veeam ===") < p.index("=== AWS Backup ===")
    veeam_part, aws_part = p.split("=== AWS Backup ===")
    # c1 (Veeam facet) keeps only its Veeam fact; "shared" (v1) serves Veeam and
    # contributes nothing under AWS because v1 is out of AWS scope.
    assert "text v1" in veeam_part and "text a1" not in veeam_part
    assert "text a1" in aws_part and "text v1" not in aws_part
    assert [c["community_id"] for c in res["communities_used"]] == ["c1", "shared", "c2"]
    assert sorted(stubs.maps) == ["c1", "c2", "shared"]
    assert "Address EVERY side" in p
    assert res["degraded_facets"] == []


async def test_a_shared_community_serves_every_facet_in_scope(stubs, monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        return [_hit("both", ["v1", "a1"])]
    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    r = _Reducer()
    res = await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                                 group_id="g", settings=_S, facets=[_V, _A])
    veeam_part, aws_part = r.prompts[0].split("=== AWS Backup ===")
    assert "text v1" in veeam_part and "text a1" not in veeam_part
    assert "text a1" in aws_part and "text v1" not in aws_part
    assert stubs.maps == ["both"]
    assert [c["community_id"] for c in res["communities_used"]] == ["both"]


@pytest.mark.parametrize("failure", ["none", "raise"])
async def test_a_facet_whose_maps_all_fail_is_degraded_not_absent(stubs, monkeypatch,
                                                                  failure):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        return [_hit("c1", ["v1"])] if scope == _V.scope else [_hit("c2", ["a1"])]

    async def _map(client, model, q, hit):
        if hit.community_id == "c2":
            if failure == "raise":
                raise RuntimeError("map down")
            return None
        return gs.MapResult(community_id=hit.community_id, title=hit.title,
                            relevance=None, key_points=["p"],
                            fact_ids=list(hit.cited_fact_uuids))
    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    monkeypatch.setattr(gs, "map_report", _map)
    r = _Reducer()
    res = await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                                 group_id="g", settings=_S, facets=[_V, _A])
    assert res["degraded_facets"] == ["AWS Backup"]
    assert "=== AWS Backup ===" not in r.prompts[0]
    assert "=== Veeam ===" in r.prompts[0]
    assert res["answer"].endswith("Could not check: AWS Backup.")


async def test_a_facet_whose_shortlist_fails_is_reported(stubs, monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        if scope == _A.scope:
            raise RuntimeError("rerank down")
        return [_hit("c1", ["v1"])]
    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    res = await gs.global_search(None, None, None, "m", _Reducer(), "s", q="q", level=1,
                                 k=10, group_id="g", settings=_S, facets=[_V, _A])
    assert res["degraded_facets"] == ["AWS Backup"]
    assert res["answer"].endswith("Could not check: AWS Backup.")


async def test_single_facet_is_unchanged(stubs):
    await gs.global_search(None, None, None, "m", _Reducer(), "s", q="q", level=1, k=10,
                           group_id="g", settings=_S, facets=[_V])
    assert stubs.shortlist == [(None, None, 10)]


async def test_a_cancelled_facet_shortlist_propagates_not_degrades(stubs, monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        if scope == _A.scope:
            raise asyncio.CancelledError()
        return [_hit("c1", ["v1"])]
    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    with pytest.raises(asyncio.CancelledError):
        await gs.global_search(None, None, None, "m", _Reducer(), "s", q="q", level=1,
                               k=10, group_id="g", settings=_S, facets=[_V, _A])


async def test_a_cancelled_map_propagates_not_degrades(stubs, monkeypatch):
    async def _map(client, model, q, hit):
        raise asyncio.CancelledError()
    monkeypatch.setattr(gs, "map_report", _map)
    with pytest.raises(asyncio.CancelledError):
        await gs.global_search(None, None, None, "m", _Reducer(), "s", q="q", level=1,
                               k=10, group_id="g", settings=_S, facets=[_V, _A])


async def test_each_facet_shortlists_and_maps_with_its_own_question(stubs, monkeypatch):
    asked = SimpleNamespace(shortlist=[], maps=[])

    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        asked.shortlist.append((q, scope))
        if scope == _V.scope:
            return [_hit("c1", ["v1"]), _hit("shared", ["v1", "a1"])]
        return [_hit("shared", ["v1", "a1"]), _hit("c2", ["a1"])]

    async def _map(client, model, q, hit):
        asked.maps.append((q, hit.community_id))
        return gs.MapResult(community_id=hit.community_id, title=hit.title,
                            relevance=None, key_points=["p"],
                            fact_ids=list(hit.cited_fact_uuids))

    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    monkeypatch.setattr(gs, "map_report", _map)
    r = _Reducer()
    facets = [Facet("Veeam", _V.scope, query="veeam q"),
              Facet("AWS Backup", _A.scope, query="aws q")]
    res = await gs.global_search(None, None, None, "m", r, "s", q="original q", level=1,
                                 k=10, group_id="g", settings=_S, facets=facets)
    assert asked.shortlist == [("veeam q", _V.scope), ("aws q", _A.scope)]
    # A community shared by facets with different questions is mapped once per question.
    assert sorted(asked.maps) == [("aws q", "c2"), ("aws q", "shared"),
                                  ("veeam q", "c1"), ("veeam q", "shared")]
    p = r.prompts[0]
    assert "original q" in p and "veeam q" not in p and "aws q" not in p
    veeam_part, aws_part = p.split("=== AWS Backup ===")
    assert "text v1" in veeam_part and "text a1" in aws_part
    assert [c["community_id"] for c in res["communities_used"]] == ["c1", "shared", "c2"]
