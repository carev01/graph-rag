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


# --- Task 15 (run 6): every live facet gets a scoped local supplement -------------

_G = object()   # a graphiti handle; the stubbed search_local never touches it
_LSRC = {"lv": [{"vendor": "Veeam", "product": "Veeam Backup & Replication",
                 "article_id": "lva"}],
         "la": [{"vendor": "AWS", "product": "AWS Backup", "article_id": "laa"}]}


def _local_fact(uuid):
    return {"fact": f"local {uuid}", "fact_uuid": uuid, "valid_at": None,
            "invalid_at": None, "sources": []}


class _CitingReducer(_Reducer):
    def __init__(self, text):
        super().__init__()
        self.text = text

    async def _create(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=self.text), finish_reason="stop")])


@pytest.fixture
def local(monkeypatch):
    """search_local stub: facts per facet scope; `raise_for` scopes raise."""
    state = SimpleNamespace(calls=[], facts={_V.scope: ["lv"], _A.scope: ["la"]},
                            raise_for=set())

    async def _search(graphiti, driver, *, q, k, scope=None, group_id, fetch_cap=450,
                      scan_max_episodes=6000, settings=None, rerank_pool=0):
        state.calls.append({"graphiti": graphiti, "q": q, "k": k, "scope": scope,
                            "group_id": group_id, "fetch_cap": fetch_cap,
                            "scan_max_episodes": scan_max_episodes,
                            "settings": settings, "rerank_pool": rerank_pool})
        if scope in state.raise_for:
            raise RuntimeError("local down")
        return {"results": [_local_fact(u) for u in state.facts.get(scope, [])]}

    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            src = {**_SRC, **_LSRC}
            state.resolved = list(uuids)
            return {u: {"sources": src[u]} for u in uuids if u in src}

    monkeypatch.setattr(gs, "search_local", _search)
    monkeypatch.setattr(gs, "Provenance", _Prov)
    return state


def _facets_with_queries():
    return [Facet("Veeam", _V.scope, query="veeam q"),
            Facet("AWS Backup", _A.scope, query="aws q")]


async def test_every_live_facet_gets_the_local_supplement(stubs, local):
    await gs.global_search(None, None, None, "m", _Reducer(), "s", q="original q",
                           level=1, k=10, group_id="g", settings=_S,
                           facets=_facets_with_queries(), graphiti=_G)
    assert [(c["q"], c["scope"]) for c in local.calls] == [("veeam q", _V.scope),
                                                           ("aws q", _A.scope)]
    for c in local.calls:
        assert c["graphiti"] is _G and c["group_id"] == "g" and c["settings"] is _S
        assert c["k"] == _S.global_facet_local_k == 8
        assert c["fetch_cap"] == _S.local_scoped_fetch_cap
        assert c["scan_max_episodes"] == _S.local_scoped_scan_max_episodes
        assert c["rerank_pool"] == _S.local_facet_rerank_pool


async def test_supplement_renders_after_community_blocks_and_cites_via_markers(stubs,
                                                                             local):
    r = _CitingReducer("Veeam X [1]. Veeam local [2]. AWS local [4].")
    res = await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                                 group_id="g", settings=_S, facets=[_V, _A], graphiti=_G)
    p = r.prompts[0]
    veeam_part, aws_part = p.split("=== AWS Backup ===")
    # community block first, then the facet's own facts under the same heading
    assert veeam_part.index('COMMUNITY "Tc1"') < veeam_part.index(
        "FACTS FROM Veeam DOCUMENTATION:")
    assert "[2] (Veeam · Veeam Backup & Replication) local lv" in veeam_part
    assert aws_part.index('COMMUNITY "Tc2"') < aws_part.index(
        "FACTS FROM AWS Backup DOCUMENTATION:")
    assert "local la" in aws_part and "local lv" not in aws_part
    # one resolve_citations call covers community AND local facts
    assert sorted(local.resolved) == ["a1", "la", "lv", "v1"]
    by_marker = {c["marker"]: c for c in res["citations"]}
    assert by_marker[2]["fact_uuid"] == "lv"
    assert by_marker[2]["sources"] == _LSRC["lv"]
    assert by_marker[4]["fact_uuid"] == "la"
    assert by_marker[4]["sources"] == _LSRC["la"]
    # communities_used lists communities only, never local facts
    assert [c["community_id"] for c in res["communities_used"]] == ["c1", "shared", "c2"]
    assert res["degraded_facets"] == []


async def test_a_shared_fact_keeps_one_marker(stubs, local):
    # AWS's local search returns a1 (already a community fact under AWS) and v1
    # (already numbered under Veeam): neither gets a second marker.
    local.facts[_A.scope] = ["a1", "v1", "la"]
    r = _Reducer()
    await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                           group_id="g", settings=_S, facets=[_V, _A], graphiti=_G)
    p = r.prompts[0]
    _, aws_part = p.split("=== AWS Backup ===")
    lines = p.split("FINDINGS:\n", 1)[1].splitlines()
    a1_lines = [ln for ln in lines if ln.endswith(" a1")]
    assert a1_lines == ["[3] (AWS · AWS Backup) local a1"]   # not repeated under AWS
    v1_lines = [ln for ln in lines if ln.endswith(" v1")]
    assert len(v1_lines) >= 2 and all(ln.startswith("[1] ") for ln in v1_lines)
    assert "[1] (Veeam · Veeam ONE) local v1" in aws_part   # v1 keeps marker 1
    assert "[4] (AWS · AWS Backup) local la" in aws_part


async def test_a_facet_with_no_communities_uses_its_local_facts(stubs, local,
                                                                monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        return [_hit("c1", ["v1"])] if scope == _V.scope else []
    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    r = _CitingReducer("Veeam X [1]. AWS local [3].")
    res = await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                                 group_id="g", settings=_S, facets=[_V, _A], graphiti=_G)
    _, aws_part = r.prompts[0].split("=== AWS Backup ===")
    assert "\n(no findings)" not in aws_part
    assert "FACTS FROM AWS Backup DOCUMENTATION:" in aws_part
    assert [c["fact_uuid"] for c in res["citations"]] == ["v1", "la"]


async def test_a_facet_whose_communities_are_all_out_of_scope_uses_local_facts(
        stubs, local, monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        return [_hit("c1", ["v1"])]   # AWS's only community cites a Veeam fact
    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    local.facts[_V.scope] = []
    r = _Reducer()
    res = await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                                 group_id="g", settings=_S, facets=[_V, _A], graphiti=_G)
    _, aws_part = r.prompts[0].split("=== AWS Backup ===")
    assert "\n(no findings)" not in aws_part
    assert "local la" in aws_part and "text v1" not in aws_part
    assert res["degraded_facets"] == []


async def test_no_communities_and_no_local_facts_is_no_findings(stubs, local,
                                                                monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        return [_hit("c1", ["v1"])] if scope == _V.scope else []
    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    local.facts[_A.scope] = []
    r = _Reducer()
    res = await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                                 group_id="g", settings=_S, facets=[_V, _A], graphiti=_G)
    assert "=== AWS Backup ===\n(no findings)" in r.prompts[0]
    assert res["degraded_facets"] == []


async def test_supplement_failure_keeps_community_findings(stubs, local, caplog):
    local.raise_for = {_A.scope}
    r = _Reducer()
    with caplog.at_level("WARNING"):
        res = await gs.global_search(None, None, None, "m", r, "s", q="q", level=1,
                                     k=10, group_id="g", settings=_S, facets=[_V, _A],
                                     graphiti=_G)
    _, aws_part = r.prompts[0].split("=== AWS Backup ===")
    assert "text a1" in aws_part
    assert "FACTS FROM AWS Backup DOCUMENTATION:" not in aws_part
    assert res["degraded_facets"] == []
    assert "Could not check" not in res["answer"]
    assert any("local down" in rec.getMessage() for rec in caplog.records)


async def test_supplement_failure_without_communities_degrades_never_absent(
        stubs, local, monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        return [_hit("c1", ["v1"])] if scope == _V.scope else []
    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    local.raise_for = {_A.scope}
    r = _Reducer()
    res = await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                                 group_id="g", settings=_S, facets=[_V, _A], graphiti=_G)
    assert "=== AWS Backup ===" not in r.prompts[0]
    assert "\n(no findings)" not in r.prompts[0]
    assert res["degraded_facets"] == ["AWS Backup"]
    assert res["answer"].endswith("Could not check: AWS Backup.")


async def test_a_degraded_facet_gets_no_supplement_block(stubs, local, monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        if scope == _A.scope:
            raise RuntimeError("rerank down")
        return [_hit("c1", ["v1"])]
    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    r = _Reducer()
    res = await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                                 group_id="g", settings=_S, facets=[_V, _A], graphiti=_G)
    assert "local la" not in r.prompts[0]
    assert res["degraded_facets"] == ["AWS Backup"]


async def test_a_cancelled_supplement_propagates(stubs, local, monkeypatch):
    async def _search(graphiti, driver, **kw):
        raise asyncio.CancelledError()
    monkeypatch.setattr(gs, "search_local", _search)
    with pytest.raises(asyncio.CancelledError):
        await gs.global_search(None, None, None, "m", _Reducer(), "s", q="q", level=1,
                               k=10, group_id="g", settings=_S, facets=[_V, _A],
                               graphiti=_G)


async def test_without_graphiti_the_supplement_is_skipped(stubs, local, caplog):
    r = _Reducer()
    with caplog.at_level("INFO"):
        await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                               group_id="g", settings=_S, facets=[_V, _A])
    assert local.calls == []
    assert "FACTS FROM" not in r.prompts[0]
    assert any("local supplement skipped" in rec.getMessage() for rec in caplog.records)


async def test_single_path_never_calls_the_supplement(stubs, local):
    r = _Reducer()
    await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                           group_id="g", settings=_S, facets=[_V], graphiti=_G)
    assert local.calls == []
    assert "FACTS FROM" not in r.prompts[0]
    assert r.prompts[0] == gs._REDUCE_PROMPT.format(q="q", blocks=r.prompts[0].split(
        "FINDINGS:\n", 1)[1].rsplit("\n\nAnswer:", 1)[0])


async def test_facet_reduce_prompt_forbids_false_absence_and_long_marker_lists(stubs,
                                                                               local):
    r = _Reducer()
    await gs.global_search(None, None, None, "m", r, "s", q="q", level=1, k=10,
                           group_id="g", settings=_S, facets=[_V, _A], graphiti=_G)
    p = r.prompts[0]
    assert ("Never state that the documentation does not cover a side when findings "
            "are listed under its heading; describe what those findings say. Say a side "
            "is not covered only when it shows \"(no findings)\".") in p
    assert ("Cite each claim with the one or two markers of the facts it rests on; do "
            "not attach long marker lists to a sentence.") in p
    assert "Address EVERY side" in p and "Do NOT write any URL." in p
