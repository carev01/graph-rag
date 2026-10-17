# tests/unit/test_enumeration.py
"""Enumeration checks every content vendor (spec §3.4)."""
import asyncio
from types import SimpleNamespace

import pytest

import answer_api.enumeration as en
import answer_api.global_search as gs
from answer_api.scope import Scope
from answer_api.shape import Facet
from graph_extract.config import ExtractSettings

pytestmark = pytest.mark.asyncio

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p",
                     # Hermetic: the relevance floor must stay OFF unless a test turns
                     # it on explicitly (rerank_configured(settings) reads these two).
                     rerank_base_url="", rerank_model="")
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
            if isinstance(text, list):   # a sequence of replies, one per call
                text = text.pop(0)
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

    async def _search(graphiti, driver, *, q, k, scope=None, group_id, fetch_cap=450, scan_max_episodes=6000, settings=None,
                      rerank_pool=0):
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


_INVENTED = '{"documented": true, "points": ["p"], "fact_ids": ["made-up"]}'
_GOOD_AWS = '{"documented": true, "points": ["p"], "fact_ids": ["lf-AWS"]}'


def _aws_checks(c):
    return sum(1 for p in c.prompts if "from AWS's documentation" in p)


async def test_invented_fact_ids_take_the_retry_and_a_good_second_reply_documents():
    c = _Client(_checks(AWS=[_INVENTED, _GOOD_AWS]))
    res = await _run(c)
    assert _aws_checks(c) == 2
    assert res["documented_vendors"] == ["AWS"]
    assert res["degraded_facets"] == []


async def test_invented_fact_ids_twice_is_degraded_not_absence():
    c = _Client(_checks(AWS=[_INVENTED, _INVENTED]))
    res = await _run(c)
    assert _aws_checks(c) == 2
    assert res["documented_vendors"] == []
    assert res["degraded_facets"] == ["AWS"]


async def test_an_unusable_check_after_retry_is_could_not_check_not_absence():
    c = _Client(_checks(Cohesity="not json"))
    res = await _run(c)
    assert "Cohesity" in res["degraded_facets"]
    assert "Could not check: Cohesity." in res["answer"]
    clause = res["answer"].split(")")[0]
    assert clause == "None of the checked vendors (AWS, Microsoft, Veeam"
    assert "Cohesity" not in clause
    assert res["checked_vendors"] == ["AWS", "Microsoft", "Veeam"]


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
        "Checked: AWS, Microsoft, Veeam. Documented for: AWS.")
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


async def test_every_vendor_degraded_is_a_refusal_not_none_documented():
    res = await _run(_Client({v: "not json" for v in _VENDORS}))
    assert res["degraded_facets"] == _VENDORS
    assert "None of the checked vendors" not in res["answer"]
    assert res["answer"] == (
        "I don't have enough information to answer that from the available sources. "
        "Could not check: AWS, Cohesity, Microsoft, Veeam.")


def _community(v, fact_id):
    return gs.CommunityHit(community_id=f"c{v}", title=f"{v} T", summary="s", level=1,
                           rating=5, cited_fact_uuids=[fact_id],
                           full_report='[{"finding": "f", "fact_ids": ["%s"]}]' % fact_id,
                           similarity=0.9)


async def test_a_vendor_is_never_documented_on_another_vendors_community_fact(monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        v = scope.vendors[0]
        return [_community(v, "cf-Veeam" if v == "AWS" else f"cf-{v}")]

    monkeypatch.setattr(en, "shortlist_communities", _shortlist)
    c = _Client(_checks(AWS='{"documented": true, "points": ["p"], "fact_ids": ["cf-Veeam"]}'))
    res = await _run(c)
    aws_prompt = next(p for p in c.prompts if "from AWS's documentation" in p)
    assert "cf-Veeam" not in aws_prompt
    assert "AWS" not in res["documented_vendors"]
    assert res["degraded_facets"] == ["AWS"]


async def test_documented_with_no_fact_text_is_could_not_check(monkeypatch):
    async def _no_texts(driver, group_id, uuids):
        return {}

    monkeypatch.setattr(en, "_fact_texts", _no_texts)
    c = _Client(_checks(AWS='{"documented": true, "points": ["p"], "fact_ids": ["lf-AWS"]}',
                        Veeam='{"documented": true, "points": ["p"], "fact_ids": ["cf-Veeam"]}'))
    res = await _run(c)
    assert res["documented_vendors"] == ["AWS"]
    assert res["degraded_facets"] == ["Veeam"]
    assert "=== Veeam ===" not in c.prompts[-1]
    assert res["answer"].endswith(
        "Could not check: Veeam.\n\nChecked: AWS, Cohesity, Microsoft. Documented for: AWS.")


async def test_a_fact_shared_by_two_vendors_renders_under_both_with_one_marker(monkeypatch):
    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        return [_community(scope.vendors[0], "cf-shared")]

    class _Prov:
        def __init__(self, driver):
            pass

        async def resolve_citations(self, uuids):
            return {u: {"sources": _src("AWS") + _src("Veeam") if u == "cf-shared"
                        else _src(u.split("-")[1])} for u in uuids}

    monkeypatch.setattr(en, "shortlist_communities", _shortlist)
    monkeypatch.setattr(en, "Provenance", _Prov)
    shared = '{"documented": true, "points": ["p"], "fact_ids": ["cf-shared"]}'
    c = _Client(_checks(AWS=shared, Veeam=shared))
    res = await _run(c)
    assert res["documented_vendors"] == ["AWS", "Veeam"]
    syn = c.prompts[-1]
    assert "=== AWS ===\n[1] " in syn
    assert "=== Veeam ===\n[1] " in syn


async def test_gather_and_check_use_the_facet_question_synthesis_keeps_the_original(
        monkeypatch):
    # Run 6: a single vendor's evidence cannot answer "...and how do their approaches
    # differ?", so the check asks the vendor's own question; synthesis keeps `q`.
    asked = []

    async def _shortlist(driver, embedder, q, *, level, k, group_id, settings=None,
                         stats=None, scope=None, top_n=None):
        asked.append(("shortlist", q))
        return []

    async def _search(graphiti, driver, *, q, k, scope=None, group_id, **kw):
        asked.append(("search", q))
        v = scope.vendors[0]
        return {"results": [{"fact": f"{v} fact", "fact_uuid": f"lf-{v}",
                             "valid_at": None, "invalid_at": None, "sources": _src(v)}]}

    monkeypatch.setattr(en, "shortlist_communities", _shortlist)
    monkeypatch.setattr(en, "search_local", _search)
    c = _Client({"AWS": '{"documented": true, "points": ["p"], "fact_ids": ["lf-AWS"]}'})
    res = await en.enumerate_vendors(None, None, None, c, "map", c, "syn",
                                     q="Which vendors lock backups?",
                                     facets=[Facet("AWS", Scope(vendors=("AWS",)),
                                                   query="Does AWS lock backups?")],
                                     group_id="g", settings=_S)
    # Facts are searched with the vendor's question AND the original (run 10), and
    # merged without duplicates (both fake searches return lf-AWS).
    assert asked == [("shortlist", "Does AWS lock backups?"),
                     ("search", "Does AWS lock backups?"),
                     ("search", "Which vendors lock backups?")]
    check = next(p for p in c.prompts if p.startswith("Decide whether"))
    assert "QUESTION: Does AWS lock backups?" in check
    assert "Which vendors lock backups?" not in check
    assert "answers the QUESTION about AWS" in check
    syn = c.prompts[-1]
    assert not syn.startswith("Decide whether")
    assert "QUESTION: Which vendors lock backups?" in syn
    assert "Does AWS lock backups?" not in syn
    assert res["documented_vendors"] == ["AWS"]


async def test_a_failed_rewrite_checks_a_per_vendor_form_never_the_bare_question():
    # Regression (final review): a facet whose rewrite failed has no query; its
    # check must not reintroduce the cross-vendor question run 6 failed on.
    c = _Client(_checks())
    await _run(c)
    checks = [p for p in c.prompts if p.startswith("Decide whether")]
    assert len(checks) == 4
    for v in _VENDORS:
        check = next(p for p in checks if f"from {v}'s documentation" in p)
        assert (f"QUESTION: Which vendors lock backups?\n(Consider only what {v}'s "
                f"evidence shows about this for {v}.)\n\nEVIDENCE:") in check


async def test_a_rewritten_facet_checks_its_own_question_verbatim():
    c = _Client(_checks())
    facets = [Facet("AWS", Scope(vendors=("AWS",)), query="Does AWS lock backups?")]
    await en.enumerate_vendors(None, None, None, c, "map", c, "syn",
                               q="Which vendors lock backups?", facets=facets,
                               group_id="g", settings=_S)
    check = next(p for p in c.prompts if p.startswith("Decide whether"))
    assert "QUESTION: Does AWS lock backups?\n\nEVIDENCE:" in check
    assert "Consider only" not in check


async def test_enum_concurrency_bounds_the_vendor_fan_out(monkeypatch):
    # At 40 vendors the unbounded fan-out issued up to 120 hybrid searches plus 40
    # scans at once against Neo4j: enum_concurrency bounds vendor gathers and checks.
    live = {"gather": 0, "check": 0}
    peak = {"gather": 0, "check": 0}

    async def _hold(kind):
        live[kind] += 1
        peak[kind] = max(peak[kind], live[kind])
        for _ in range(5):
            await asyncio.sleep(0)
        live[kind] -= 1

    async def _search(graphiti, driver, *, q, k, scope=None, group_id, **kw):
        await _hold("gather")
        v = scope.vendors[0]
        return {"results": [{"fact": f"{v} fact", "fact_uuid": f"lf-{v}",
                             "valid_at": None, "invalid_at": None, "sources": _src(v)}]}

    class _Counting(_Client):
        async def _create(self, **kw):
            await _hold("check")
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"documented": false}'),
                finish_reason="stop")])

    monkeypatch.setattr(en, "search_local", _search)
    vendors = [f"V{i}" for i in range(7)]
    facets = [Facet(v, Scope(vendors=(v,))) for v in vendors]
    monkeypatch.setattr(en, "Provenance", _AnyProv)
    c = _Counting({})
    settings = _S.model_copy(update={"enum_concurrency": 2})
    res = await en.enumerate_vendors(None, None, None, c, "map", c, "syn",
                                     q="Which vendors lock backups?", facets=facets,
                                     group_id="g", settings=settings)
    assert peak == {"gather": 2, "check": 2}
    assert res["checked_vendors"] == vendors   # results stay in facet order


class _AnyProv:
    def __init__(self, driver):
        pass

    async def resolve_citations(self, uuids):
        return {u: {"sources": _src(u.split("-", 1)[1])} for u in uuids}


async def test_enum_concurrency_defaults_to_8():
    assert _S.enum_concurrency == 8


async def test_gather_merges_the_vendor_question_and_original_question_facts(monkeypatch):
    # Run 10: the rewrite's wording alone dropped AWS's restore-testing facts.
    async def _shortlist(*a, **kw):
        return []

    async def _search(graphiti, driver, *, q, k, scope=None, group_id, **kw):
        tag = "own" if q.startswith("Does") else "orig"
        return {"results": [{"fact": f"{tag} {i}", "fact_uuid": f"{tag}-{i}"}
                            for i in range(2)]
                + [{"fact": "shared", "fact_uuid": "shared"}]}

    monkeypatch.setattr(en, "shortlist_communities", _shortlist)
    monkeypatch.setattr(en, "search_local", _search)
    _, facts = await en._gather(None, None, None, q="Which vendors lock backups?",
                                facet=Facet("AWS", Scope(vendors=("AWS",)),
                                            query="Does AWS lock backups?"),
                                group_id="g", settings=_S)
    assert [f["fact_uuid"] for f in facts] == ["own-0", "own-1", "shared",
                                               "orig-0", "orig-1"]


async def test_gather_without_a_rewrite_searches_once(monkeypatch):
    asked = []

    async def _shortlist(*a, **kw):
        return []

    async def _search(graphiti, driver, *, q, **kw):
        asked.append(q)
        return {"results": []}

    monkeypatch.setattr(en, "shortlist_communities", _shortlist)
    monkeypatch.setattr(en, "search_local", _search)
    await en._gather(None, None, None, q="Which vendors lock backups?",
                     facet=Facet("AWS", Scope(vendors=("AWS",))), group_id="g",
                     settings=_S)
    assert asked == ["Which vendors lock backups?"]


# -- Relevance floor (spec 2026-10-17: enumeration_min_score=0.55) ----------

_DOCUMENTED = _checks(AWS='{"documented": true, "points": ["p"], "fact_ids": ["lf-AWS"]}',
                      Veeam='{"documented": true, "points": ["p"], "fact_ids": ["cf-Veeam"]}')


def _rerank_by_vendor(scores: dict[str, float]):
    """A fake `rerank` that scores by which vendor's fact text the documents
    hold -- the real texts fixture renders facts as "text <fact_uuid>", and
    every fact_uuid in this file embeds its vendor's name."""
    async def _rerank(q, documents, *, top_k, settings):
        for v, score in scores.items():
            if any(v in d for d in documents):
                return [(0, score)]
        raise AssertionError(f"no score configured for documents {documents!r}")
    return _rerank


async def test_a_vendor_below_the_relevance_floor_is_checked_not_documented(monkeypatch):
    monkeypatch.setattr(en, "rerank", _rerank_by_vendor({"AWS": 0.448, "Veeam": 0.633}))
    monkeypatch.setattr(en, "rerank_configured", lambda s: True)
    res = await _run(_Client(_DOCUMENTED))
    assert res["checked_vendors"] == _VENDORS
    assert res["documented_vendors"] == ["Veeam"]
    assert res["answer"].endswith(
        "Checked: AWS, Cohesity, Microsoft, Veeam. Documented for: Veeam.")
    assert res["vendor_scores"] == {"AWS": 0.448, "Veeam": 0.633}
    assert res["relevance_unchecked"] == []


async def test_a_vendor_above_the_floor_stays_documented(monkeypatch):
    monkeypatch.setattr(en, "rerank", _rerank_by_vendor({"AWS": 0.633, "Veeam": 0.824}))
    monkeypatch.setattr(en, "rerank_configured", lambda s: True)
    res = await _run(_Client(_DOCUMENTED))
    assert res["documented_vendors"] == ["AWS", "Veeam"]
    assert res["vendor_scores"] == {"AWS": 0.633, "Veeam": 0.824}
    assert res["relevance_unchecked"] == []


async def test_relevance_check_unavailable_keeps_both_and_records_degraded():
    # Default _S has no reranker configured: both documented vendors are kept,
    # not scored, and recorded as unchecked rather than silently trusted.
    res = await _run(_Client(_DOCUMENTED))
    assert res["documented_vendors"] == ["AWS", "Veeam"]
    assert set(res["relevance_unchecked"]) == {"AWS", "Veeam"}
    assert res["vendor_scores"] == {}


async def test_a_rerank_exception_for_one_vendor_keeps_that_vendor(monkeypatch):
    async def _rerank(q, documents, *, top_k, settings):
        if any("AWS" in d for d in documents):
            raise RuntimeError("rerank endpoint down")
        return [(0, 0.824)]

    monkeypatch.setattr(en, "rerank", _rerank)
    monkeypatch.setattr(en, "rerank_configured", lambda s: True)
    res = await _run(_Client(_DOCUMENTED))
    assert res["documented_vendors"] == ["AWS", "Veeam"]
    assert res["relevance_unchecked"] == ["AWS"]
    assert res["vendor_scores"] == {"Veeam": 0.824}


async def test_enumeration_min_score_defaults_to_055():
    assert ExtractSettings(_env_file=None, docext_base_url="http://x",
                           docext_read_key="k", neo4j_uri="bolt://x", neo4j_user="u",
                           neo4j_password="p").enumeration_min_score == 0.55


async def test_check_prompt_rejects_adjacent_features():
    assert ("Mark it documented only when the evidence states the capability the "
            "question asks about itself -- a related or adjacent feature does not "
            "count.") in en._CHECK_PROMPT
