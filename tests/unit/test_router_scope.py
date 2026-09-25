"""Router-level scope wiring (Task 6, spec section 4.3's "the router passes the
scope to every mode"): the resolved Scope reaches every mode's call, the same
Scope survives a local->drift or global->local fallback, and the envelope
carries `scope` + `applies_to` (the mode's own when present, else derived from
citations -- timeline)."""
from __future__ import annotations

import pytest

import answer_api.synthesize as synth_mod
import answer_api.global_search as global_mod
import answer_api.drift as drift_mod
import answer_api.timeline as timeline_mod
import answer_api.freshness as freshness_mod
from answer_api import attribution
from answer_api.router import answer_router, _normalize, _render_timeline
from answer_api.scope import Scope
from graph_extract.config import ExtractSettings

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")

_SCOPE = Scope(("Veeam",), (), "explicit")

_captured: dict = {}


async def _fake_local(*a, **k):
    _captured["local"] = k.get("scope")
    return {"query": k["q"], "answer": "local [1]",
            "citations": [{"marker": 1, "fact_uuid": "f1", "sources": []}],
            "retrieved": 1, "cited": 1,
            "applies_to": [{"vendor": "Veeam", "products": [], "facts": 1}]}


async def _fake_local_empty(*a, **k):
    _captured["local"] = k.get("scope")
    return {"query": k["q"], "answer": "refusal", "citations": [],
            "retrieved": 0, "cited": 0, "applies_to": []}


async def _fake_global(*a, **k):
    _captured["global"] = k.get("scope")
    return {"query": k["q"], "answer": "global [1]",
            "citations": [{"marker": 1, "fact_uuid": "g1", "sources": []}],
            "communities_used": [{"community_id": "c1", "title": "T"}],
            "applies_to": []}


async def _fake_drift(*a, **k):
    _captured["drift"] = k.get("scope")
    return {"query": k["q"], "answer": "drift [1]",
            "citations": [{"marker": 1, "fact_uuid": "d1", "sources": []}],
            "communities_used": [{"community_id": "c1", "title": "T"}],
            "applies_to": []}


async def _fake_timeline(*a, **k):
    _captured["timeline"] = k.get("scope")
    return {"query": k["q"], "count": 1,
            "timeline": [{"fact": "F", "fact_uuid": "t1", "valid_at": "2023-01-01",
                          "invalid_at": None, "status": "current",
                          "sources": [{"vendor": "Veeam", "product": "VBR"}]}]}


async def _fake_freshness(driver, group_id, *, reports):
    return {"graph_cursor_time": "2026-07-18T00:00:00Z", "reports_as_of": None}


@pytest.fixture(autouse=True)
def _patch_modes(monkeypatch):
    _captured.clear()
    monkeypatch.setattr(synth_mod, "answer_local", _fake_local)
    monkeypatch.setattr(global_mod, "global_search", _fake_global)
    monkeypatch.setattr(drift_mod, "drift_search", _fake_drift)
    monkeypatch.setattr(timeline_mod, "timeline_local", _fake_timeline)
    monkeypatch.setattr(freshness_mod, "freshness", _fake_freshness)


async def _route(mode_override, scope=None):
    return await answer_router(None, None, None, None, "sm", None, "mm", None, "",
                               q="q", mode_override=mode_override,
                               scope=_SCOPE if scope is None else scope, settings=_S)


@pytest.mark.parametrize("mode", ["local", "global", "drift", "timeline"])
async def test_scope_reaches_every_mode(mode):
    await _route(mode)
    assert _captured[mode] == _SCOPE


async def test_local_to_drift_fallback_reuses_same_scope(monkeypatch):
    monkeypatch.setattr(synth_mod, "answer_local", _fake_local_empty)
    env = await _route("local")
    assert env["mode"] == "drift"
    assert _captured["drift"] == _SCOPE


async def test_global_to_local_fallback_reuses_same_scope(monkeypatch):
    async def _global_empty(*a, **k):
        _captured["global"] = k.get("scope")
        return {"query": k["q"], "answer": "r", "citations": [], "communities_used": [],
                "applies_to": []}
    monkeypatch.setattr(global_mod, "global_search", _global_empty)
    env = await _route("global")
    assert env["mode"] == "local"
    assert _captured["local"] == _SCOPE


async def test_global_that_cites_nothing_falls_back_to_scoped_local(monkeypatch):
    """Communities fed the reduce step but it refused (e.g. a scoped comparison
    whose second vendor's community failed to map): the answer grounds nothing,
    so the router must try local with the SAME scope rather than serve the
    refusal (2026-09-25 acceptance run: 2 of 5 scoped comparisons)."""
    async def _global_refused(*a, **k):
        _captured["global"] = k.get("scope")
        return {"query": k["q"], "answer": "refusal", "citations": [],
                "communities_used": [{"community_id": "c1", "title": "t", "relevance": None}],
                "applies_to": []}
    monkeypatch.setattr(global_mod, "global_search", _global_refused)
    env = await _route("global")
    assert env["mode"] == "local"
    assert env["routing"]["fallback_from"] == "global"
    assert _captured["local"] == _SCOPE


async def test_global_with_citations_does_not_fall_back(monkeypatch):
    async def _global_ok(*a, **k):
        return {"query": k["q"], "answer": "ok [1]",
                "citations": [{"marker": 1, "fact_uuid": "f", "sources": []}],
                "communities_used": [{"community_id": "c1", "title": "t", "relevance": None}],
                "applies_to": []}
    monkeypatch.setattr(global_mod, "global_search", _global_ok)
    env = await _route("global")
    assert env["mode"] == "global" and env["routing"]["fallback_from"] is None


async def test_a_detected_scope_that_grounds_nothing_is_relaxed(monkeypatch):
    """Detection is a guess: when the scoped path grounds nothing, the router
    re-runs unscoped rather than refusing where main answered, and says so."""
    detected = Scope(("Microsoft",), (), "detected")
    seen: list = []

    async def _local(*a, **k):
        seen.append(k.get("scope"))
        empty = not k.get("scope") or k["scope"].is_empty()
        return ({"query": k["q"], "answer": "ok [1]", "retrieved": 1,
                 "citations": [{"marker": 1, "fact_uuid": "f", "sources": []}],
                 "applies_to": []} if empty else
                {"query": k["q"], "answer": "refusal", "retrieved": 0, "citations": [],
                 "applies_to": []})

    async def _drift_empty(*a, **k):
        return {"query": k["q"], "answer": "refusal", "citations": [], "applies_to": []}
    monkeypatch.setattr(synth_mod, "answer_local", _local)
    monkeypatch.setattr(drift_mod, "drift_search", _drift_empty)
    env = await _route("local", scope=detected)
    assert env["citations"] and env["scope"] == {
        "vendors": ["Microsoft"], "products": [], "source": "detected-relaxed"}
    assert seen[-1].is_empty()


async def test_an_explicit_scope_is_never_relaxed(monkeypatch):
    explicit = Scope(("Microsoft",), (), "explicit")

    async def _local_empty(*a, **k):
        return {"query": k["q"], "answer": "refusal", "retrieved": 0, "citations": [],
                "applies_to": []}

    async def _drift_empty(*a, **k):
        return {"query": k["q"], "answer": "refusal", "citations": [], "applies_to": []}
    monkeypatch.setattr(synth_mod, "answer_local", _local_empty)
    monkeypatch.setattr(drift_mod, "drift_search", _drift_empty)
    env = await _route("local", scope=explicit)
    assert env["scope"]["source"] == "explicit" and env["citations"] == []


async def test_envelope_carries_scope_dict():
    env = await _route("local")
    assert env["scope"] == _SCOPE.as_dict()


async def test_envelope_uses_modes_own_applies_to_when_present():
    env = await _route("local")
    assert env["applies_to"] == [{"vendor": "Veeam", "products": [], "facts": 1}]


async def test_envelope_derives_applies_to_from_citations_for_timeline():
    env = await _route("timeline")
    assert env["applies_to"] == attribution.applies_to(env["citations"])
    assert env["applies_to"] == [{"vendor": "Veeam", "products": ["VBR"], "facts": 1}]


def test_render_timeline_includes_attribution_label():
    raw = {"timeline": [
        {"fact": "F", "fact_uuid": "f1", "valid_at": "2023-01-01", "invalid_at": None,
         "status": "current", "sources": [{"vendor": "Veeam", "product": "VBR"}]},
    ]}
    answer, _ = _render_timeline(raw)
    assert answer == "- **F** (Veeam · VBR) — valid_at 2023-01-01 (current) [1]"


def test_render_timeline_omits_label_with_no_double_space():
    raw = {"timeline": [
        {"fact": "F", "fact_uuid": "f1", "valid_at": "2023-01-01", "invalid_at": None,
         "status": "current", "sources": []},
    ]}
    answer, _ = _render_timeline(raw)
    assert answer == "- **F** — valid_at 2023-01-01 (current) [1]"


def test_normalize_carries_scope_and_modes_own_applies_to():
    scope = Scope(("Veeam",), (), "explicit")
    raw = {"query": "q", "answer": "A [1]",
           "citations": [{"marker": 1, "fact_uuid": "f1", "sources": []}],
           "applies_to": [{"vendor": "Veeam", "products": [], "facts": 1}]}
    env = _normalize("local", "heuristic", None, raw, "q", scope)
    assert env["scope"] == scope.as_dict()
    assert env["applies_to"] == raw["applies_to"]


def test_normalize_derives_applies_to_when_raw_has_none():
    scope = Scope()
    raw = {"timeline": [{"fact": "F", "fact_uuid": "f1", "valid_at": None,
                         "invalid_at": None, "status": "current",
                         "sources": [{"vendor": "AWS", "product": "AWS Backup"}]}]}
    env = _normalize("timeline", "heuristic", None, raw, "q", scope)
    assert env["applies_to"] == [{"vendor": "AWS", "products": ["AWS Backup"], "facts": 1}]
    assert env["scope"] == {"vendors": [], "products": [], "source": "none"}
