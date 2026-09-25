"""Unit tests for Scope-based filtering in search_local/timeline_local and
vendor/product attribution labels in answer_local (Task 3, spec §3.1/§3.2/§4.2).
"""
from __future__ import annotations

import pytest

from answer_api import search as search_mod
from answer_api import synthesize
from answer_api import timeline as timeline_mod
from answer_api.attribution import ATTRIBUTION_RULES
from answer_api.scope import Scope

pytestmark = pytest.mark.asyncio


class FakeEdge:
    """Stub edge object with configurable episodes/invalid_at."""

    def __init__(self, uuid, fact, episodes=None, invalid_at=None):
        self.uuid = uuid
        self.fact = fact
        self.episodes = episodes or []
        self.invalid_at = invalid_at
        self.valid_at = None


class FakeGraphiti:
    """Stub graphiti client returning a fixed edge list."""

    def __init__(self, edges):
        self.edges = edges

    async def _search(self, q, config, group_ids=None, center_node_uuid=None):
        class _Results:
            pass
        r = _Results()
        r.edges = self.edges
        return r


class FakeProvenance:
    """Stub provenance resolver; sources carry vendor/product so attribution
    can be exercised without a real graph."""

    def __init__(self):
        self.driver = None

    async def resolve_citations(self, uuids):
        return {u: {"sources": [{"vendor": "Veeam", "product": "Veeam Backup & Replication",
                                 "article_id": "a1", "url": "https://x/a1"}]}
                for u in uuids}


class FakeDriver:
    pass


def _counting_scope_fn(calls, allowed):
    async def _fn(driver, scope):
        calls["n"] += 1
        return set(allowed)
    return _fn


# ---------------------------------------------------------------------------
# search_local: scope filtering
# ---------------------------------------------------------------------------

async def test_search_local_scope_filters_by_episode_intersection(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(search_mod, "scope_episode_uuids",
                        _counting_scope_fn(calls, {"ep-veeam"}))
    monkeypatch.setattr(search_mod, "Provenance", lambda d: FakeProvenance())
    edges = [FakeEdge("f1", "veeam fact", ["ep-veeam"]),
             FakeEdge("f2", "other vendor fact", ["ep-other"])]
    graphiti = FakeGraphiti(edges)

    out = await search_mod.search_local(
        graphiti, FakeDriver(), q="q", k=10, group_id="g",
        scope=Scope(("Veeam",), (), "detected"))

    assert {r["fact_uuid"] for r in out["results"]} == {"f1"}
    assert calls["n"] == 1


@pytest.mark.parametrize("scope", [None, Scope()])
async def test_search_local_empty_scope_never_calls_scope_episode_uuids(monkeypatch, scope):
    calls = {"n": 0}
    monkeypatch.setattr(search_mod, "scope_episode_uuids",
                        _counting_scope_fn(calls, set()))
    monkeypatch.setattr(search_mod, "Provenance", lambda d: FakeProvenance())
    edges = [FakeEdge("f1", "fact", ["ep1"])]
    graphiti = FakeGraphiti(edges)

    out = await search_mod.search_local(
        graphiti, FakeDriver(), q="q", k=10, group_id="g", scope=scope)

    assert calls["n"] == 0
    assert {r["fact_uuid"] for r in out["results"]} == {"f1"}


# ---------------------------------------------------------------------------
# timeline_local: same filtering
# ---------------------------------------------------------------------------

async def test_timeline_local_scope_filters_by_episode_intersection(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(timeline_mod, "scope_episode_uuids",
                        _counting_scope_fn(calls, {"ep-veeam"}))
    monkeypatch.setattr(timeline_mod, "Provenance", lambda d: FakeProvenance())

    edges = [FakeEdge("f1", "veeam fact", ["ep-veeam"]),
             FakeEdge("f2", "other vendor fact", ["ep-other"])]

    async def _fake_retrieve(*a, **k):
        return list(edges)

    async def _fake_sweep(driver, uuids, group_id):
        return {}

    monkeypatch.setattr(timeline_mod, "_retrieve_edges", _fake_retrieve)
    monkeypatch.setattr(timeline_mod, "_sweep_flags", _fake_sweep)

    out = await timeline_mod.timeline_local(
        object(), FakeDriver(), q="q", limit=10, group_id="g",
        scope=Scope(("Veeam",), (), "detected"))

    assert {r["fact_uuid"] for r in out["timeline"]} == {"f1"}
    assert calls["n"] == 1


async def test_timeline_local_empty_scope_never_calls_scope_episode_uuids(monkeypatch):
    calls = {"n": 0}
    monkeypatch.setattr(timeline_mod, "scope_episode_uuids",
                        _counting_scope_fn(calls, set()))
    monkeypatch.setattr(timeline_mod, "Provenance", lambda d: FakeProvenance())

    edges = [FakeEdge("f1", "fact", ["ep1"])]

    async def _fake_retrieve(*a, **k):
        return list(edges)

    async def _fake_sweep(driver, uuids, group_id):
        return {}

    monkeypatch.setattr(timeline_mod, "_retrieve_edges", _fake_retrieve)
    monkeypatch.setattr(timeline_mod, "_sweep_flags", _fake_sweep)

    out = await timeline_mod.timeline_local(
        object(), FakeDriver(), q="q", limit=10, group_id="g", scope=None)

    assert calls["n"] == 0
    assert {r["fact_uuid"] for r in out["timeline"]} == {"f1"}


# ---------------------------------------------------------------------------
# answer_local: attribution-labelled prompt + applies_to from cited facts only
# ---------------------------------------------------------------------------

async def test_answer_local_prompt_has_labels_and_applies_to_cited_only(monkeypatch):
    async def _fake_search(*a, **k):
        return {"query": "q", "count": 2, "results": [
            {"fact": "Hardened repositories keep backups immutable for 30 days",
             "fact_uuid": "f1", "valid_at": None, "invalid_at": None,
             "sources": [{"vendor": "Veeam", "product": "Veeam Backup & Replication",
                         "article_id": "a1", "url": "https://x/a1"}]},
            {"fact": "Object lock requires versioning",
             "fact_uuid": "f2", "valid_at": None, "invalid_at": None,
             "sources": [{"vendor": "Commvault", "product": "Commvault Cloud",
                         "article_id": "a2", "url": "https://x/a2"}]},
        ]}
    monkeypatch.setattr(synthesize.search_mod, "search_local", _fake_search)

    calls: list[dict] = []

    class _Msg:
        content = "Hardened repositories keep backups immutable [1]."

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Chat:
        class completions:
            @staticmethod
            async def create(**kw):
                calls.append(kw)
                return _Resp()

    class _StubGLM:
        chat = _Chat()

    out = await synthesize.answer_local(object(), object(), _StubGLM(), "glm",
                                        q="q", group_id="g")

    prompt = calls[0]["messages"][0]["content"]
    assert ("[1] (Veeam · Veeam Backup & Replication) Hardened repositories keep "
            "backups immutable for 30 days") in prompt
    assert ATTRIBUTION_RULES in prompt

    # Only the cited fact's (marker 1, Veeam) vendor/product may appear -- the
    # uncited fact's Commvault label must not leak into applies_to.
    assert out["applies_to"] == [
        {"vendor": "Veeam", "products": ["Veeam Backup & Replication"], "facts": 1}]
    assert not any(a["vendor"] == "Commvault" for a in out["applies_to"])


async def test_answer_local_refusal_has_empty_applies_to(monkeypatch):
    async def _fake_search_empty(*a, **k):
        return {"query": "q", "count": 0, "results": []}
    monkeypatch.setattr(synthesize.search_mod, "search_local", _fake_search_empty)

    out = await synthesize.answer_local(object(), object(), object(), "glm",
                                        q="q", group_id="g")
    assert out["answer"] == synthesize._REFUSAL
    assert out["applies_to"] == []
