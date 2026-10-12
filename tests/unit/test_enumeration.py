# tests/unit/test_enumeration.py
"""Enumeration checks every content vendor (spec §3.4)."""
from types import SimpleNamespace

import pytest

import answer_api.enumeration as en
import answer_api.global_search as gs
from answer_api.scope import Scope
from answer_api.shape import Facet
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")
_VENDORS = ["AWS", "Cohesity", "Microsoft", "Veeam"]
_FACETS = [Facet(v, Scope(vendors=(v,), source="detected")) for v in _VENDORS]


def _src(v):
    return [{"vendor": v, "product": f"{v} P", "article_id": v}]


class _Client:
    """Check replies by vendor; synthesis returns one cited line per documented vendor."""

    def __init__(self, checks):
        self.checks = checks
        self.prompts = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        p = kw["messages"][0]["content"]
        self.prompts.append(p)
        if p.startswith("Decide whether"):
            vendor = next(v for v in _VENDORS if f"from {v}'s documentation" in p)
            text = self.checks[vendor]
        else:
            text = "AWS locks vaults [1]. Veeam hardens repos [2]."
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason="stop")])


@pytest.fixture(autouse=True)
def stubs(monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        v = scope.vendors[0]
        return [gs.CommunityHit(community_id=f"c{v}", title=f"{v} T", summary="s",
                                level=1, rating=5, cited_fact_uuids=[f"cf-{v}"],
                                full_report='[{"finding": "f", "fact_ids": ["cf-%s"]}]' % v,
                                similarity=0.9)]

    async def _search(graphiti, driver, *, q, k, scope=None, group_id, fetch_cap=450):
        v = scope.vendors[0]
        return {"results": [{"fact": f"{v} fact", "fact_uuid": f"lf-{v}",
                             "valid_at": None, "invalid_at": None, "sources": _src(v)}]}

    async def _texts(driver, group_id, uuids):
        return {u: f"text {u}" for u in uuids}

    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            return {u: {"sources": _src(u.split("-")[1])} for u in uuids}

    monkeypatch.setattr(en, "shortlist_communities", _shortlist)
    monkeypatch.setattr(en, "search_local", _search)
    monkeypatch.setattr(en, "_fact_texts", _texts)
    monkeypatch.setattr(en, "Provenance", _Prov)


def _checks(**over):
    base = {v: '{"documented": false}' for v in _VENDORS}
    base.update(over)
    return base


async def _run(client):
    return await en.enumerate_vendors(None, None, None, client, "map", client, "syn",
                                      q="Which vendors lock backups?", facets=_FACETS,
                                      group_id="g", settings=_S)


async def test_every_vendor_is_checked_and_documented_ones_are_listed():
    c = _Client(_checks(AWS='{"documented": true, "points": ["p"], "fact_ids": ["lf-AWS"]}',
                        Veeam='{"documented": true, "points": ["p"], "fact_ids": ["cf-Veeam"]}'))
    res = await _run(c)
    assert res["checked_vendors"] == _VENDORS
    assert res["documented_vendors"] == ["AWS", "Veeam"]
    assert res["answer"].endswith(
        "Checked: AWS, Cohesity, Microsoft, Veeam. Documented for: AWS, Veeam.")
    syn = c.prompts[-1]
    assert syn.index("=== AWS ===") < syn.index("=== Veeam ===")
    assert "=== Cohesity ===" not in syn
    assert {x["fact_uuid"] for x in res["citations"]} == {"lf-AWS", "cf-Veeam"}


async def test_fact_ids_outside_the_evidence_are_dropped_and_unverifiable_is_degraded():
    c = _Client(_checks(AWS='{"documented": true, "points": ["p"], "fact_ids": ["made-up"]}'))
    res = await _run(c)
    assert res["documented_vendors"] == []
    assert res["degraded_facets"] == ["AWS"]


async def test_an_unusable_check_after_retry_is_could_not_check_not_absence():
    c = _Client(_checks(Cohesity="not json"))
    res = await _run(c)
    assert "Cohesity" in res["degraded_facets"]
    assert "Could not check: Cohesity." in res["answer"]


async def test_none_documented_names_the_checked_vendors():
    res = await _run(_Client(_checks()))
    assert res["documented_vendors"] == []
    assert res["answer"].startswith(
        "None of the checked vendors (AWS, Cohesity, Microsoft, Veeam) document this")
    assert res["citations"] == []


class _EmptySynthClient(_Client):
    """Checks answer normally; the synthesis call returns empty content every time."""

    async def _create(self, **kw):
        p = kw["messages"][0]["content"]
        if p.startswith("Decide whether"):
            return await super()._create(**kw)
        self.prompts.append(p)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=""), finish_reason="length")])


async def test_unusable_synthesis_still_reports_the_vendor_that_could_not_be_checked():
    c = _EmptySynthClient(_checks(
        Cohesity="not json",
        AWS='{"documented": true, "points": ["p"], "fact_ids": ["lf-AWS"]}'))
    res = await _run(c)
    assert res["degraded"] == "synthesis-failed"
    assert res["degraded_facets"] == ["Cohesity"]
    assert "Could not check: Cohesity." in res["answer"]
    assert res["answer"].endswith(
        "Checked: AWS, Cohesity, Microsoft, Veeam. Documented for: AWS.")
    assert res["citations"] == []


class _RaisingCheckClient(_Client):
    """Microsoft's check call raises (endpoint error); every other call answers."""

    async def _create(self, **kw):
        if "from Microsoft's documentation" in kw["messages"][0]["content"]:
            raise RuntimeError("endpoint down")
        return await super()._create(**kw)


async def test_a_check_that_raises_is_could_not_check_and_the_rest_still_answer():
    c = _RaisingCheckClient(_checks(
        AWS='{"documented": true, "points": ["p"], "fact_ids": ["lf-AWS"]}'))
    res = await _run(c)
    assert res["degraded_facets"] == ["Microsoft"]
    assert res["documented_vendors"] == ["AWS"]
    assert "Could not check: Microsoft." in res["answer"]
