"""BACKLOG 0b: the reduce step must see marker-bound FACT LINES, not key points
plus a bag of markers.

The per-claim audit (faithfulness-investigation-2026-09-10.md §3.2) found 6 of 6
claims fully supported by the facts the reducer was given and 0 correctly cited:
`map_report` returns `key_points[]` and `fact_ids[]` as two unrelated lists, and
the old block rendered `Supporting facts: [1] [2] ... [19]` with no fact text at
all, so the reducer numbered sentences by position. These tests pin the new
block shape, the unchanged marker numbering, the single batched fact-text read,
and that a fact whose text cannot be found is dropped loudly rather than
rendered as an empty line.
"""
from __future__ import annotations

import logging

from answer_api import global_search as gs
from graph_extract.config import ExtractSettings

_S = ExtractSettings(_env_file=None, docext_base_url="http://x", docext_read_key="k",
                     neo4j_uri="bolt://x", neo4j_user="u", neo4j_password="p")

TEXTS = {"fa": "AWS Backup encrypts recovery points using AWS KMS keys.",
         "fb": "Azure Backup encrypts data at rest using 256-bit AES encryption.",
         "fc": "Vault Lock in compliance mode cannot be removed by any user."}


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for r in self._rows:
            yield r


class _FakeSession:
    """Answers the fact-text query from `texts`; every other query (provenance)
    yields nothing. Records each call so a test can count the round trips."""

    def __init__(self, texts, calls):
        self._texts = texts
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def run(self, cypher, **kw):
        self._calls.append((cypher, kw))
        if "f.fact AS fact" in cypher:
            return _Rows([{"uuid": u, "fact": self._texts[u]} for u in kw["u"]
                          if u in self._texts])
        return _Rows([])


class _FakeDriver:
    def __init__(self, texts):
        self.calls: list[tuple[str, dict]] = []
        self._texts = texts

    def session(self):
        return _FakeSession(self._texts, self.calls)

    def fact_text_calls(self):
        return [(c, kw) for c, kw in self.calls if "f.fact AS fact" in c]


class _FakeSynthClient:
    def __init__(self, content):
        self._content = content
        self.calls: list[dict] = []
        self.chat = self
        self.completions = self

    async def create(self, **kw):
        self.calls.append(kw)
        choice = type("C", (), {"message": type("M", (), {"content": self._content})(),
                                "finish_reason": "stop"})()
        return type("R", (), {"choices": [choice]})()


def _hit(cid, title, fids):
    return gs.CommunityHit(cid, title, "sum", 1, 7.0, list(fids), "[]", 0.9, None)


async def _run(monkeypatch, maps: dict[str, tuple[list[str], list[str]]], texts,
               synth_content="answer [1]."):
    """`maps`: community_id -> (key_points, fact_ids). Shortlist and map step are
    faked; the fact-text read and provenance go through the fake driver."""
    hits = [_hit(cid, f"Title {cid}", fids) for cid, (_, fids) in maps.items()]

    async def _shortlist(driver, embedder, q, *, level, k, group_id, rating_boost=0.1,
                         settings=None, stats=None, scope=None):
        return hits

    async def _map(client, model, q, hit):
        kps, fids = maps[hit.community_id]
        return gs.MapResult(community_id=hit.community_id, title=hit.title,
                            relevance=None, key_points=list(kps), fact_ids=list(fids))

    monkeypatch.setattr(gs, "shortlist_communities", _shortlist)
    monkeypatch.setattr(gs, "map_report", _map)
    driver = _FakeDriver(texts)
    synth = _FakeSynthClient(synth_content)
    res = await gs.global_search(driver, None, object(), "mm", synth, "sm",
                                 q="q", level=1, k=3, group_id="g", settings=_S)
    return res, synth, driver


def _reduce_prompt(synth) -> str:
    return synth.calls[0]["messages"][0]["content"]


async def test_block_is_marker_bound_fact_lines_not_key_points_plus_marker_bag(monkeypatch):
    """The fix itself: each community block lists `[N] <fact text>` lines. The
    key points and the `Supporting facts:` bag are gone, so a claim can be
    traced to the fact it rests on. A fact selected by two communities keeps
    ONE marker (the ordered-unique union numbering is untouched)."""
    res, synth, _ = await _run(monkeypatch, {
        "c1": (["kp about kms"], ["fa", "fb"]),
        "c2": (["kp about vault lock"], ["fb", "fc"]),
    }, TEXTS, synth_content="Compliance mode is immutable [3].")
    prompt = _reduce_prompt(synth)
    findings = prompt.split("FINDINGS:", 1)[1]
    c1, c2 = findings.split('COMMUNITY "Title c2"')
    assert f"[1] {TEXTS['fa']}" in c1 and f"[2] {TEXTS['fb']}" in c1
    assert f"[2] {TEXTS['fb']}" in c2 and f"[3] {TEXTS['fc']}" in c2   # same marker for fb
    assert "Supporting facts:" not in findings
    assert "kp about" not in findings and "\n- " not in findings
    # numbering unchanged end to end: [3] still resolves to fc
    assert [c["fact_uuid"] for c in res["citations"]] == ["fc"]


async def test_fact_text_is_read_in_one_batched_query_for_the_union(monkeypatch):
    """Not one query per community and not one per fact: a single round trip
    for the ordered-unique union of every surviving map result's fact_ids."""
    _, _, driver = await _run(monkeypatch, {
        "c1": ([], ["fa", "fb"]),
        "c2": ([], ["fb", "fc"]),
        "c3": ([], ["fc"]),
    }, TEXTS)
    calls = driver.fact_text_calls()
    assert len(calls) == 1
    assert calls[0][1]["u"] == ["fa", "fb", "fc"]
    assert calls[0][1]["g"] == "g"


async def test_fact_without_text_is_dropped_and_logged_not_rendered_empty(monkeypatch, caplog):
    """A missing fact must not become `[2] ` -- an empty, legitimate-looking
    line. It is dropped from the block AND from marker_map (so the reducer
    cannot cite what it never saw), and the loss is logged."""
    texts = {k: v for k, v in TEXTS.items() if k != "fb"}
    with caplog.at_level(logging.WARNING, logger="answer_api.global_search"):
        res, synth, _ = await _run(monkeypatch, {"c1": ([], ["fa", "fb", "fc"])}, texts,
                                   synth_content="Claim [1]. Other [2]. Third [3].")
    findings = _reduce_prompt(synth).split("FINDINGS:", 1)[1]
    assert "[2]" not in findings
    assert f"[1] {TEXTS['fa']}" in findings and f"[3] {TEXTS['fc']}" in findings
    assert [c["marker"] for c in res["citations"]] == [1, 3]
    assert "[2]" not in res["answer"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "fb" in warnings[0] and "1" in warnings[0]


async def test_no_fact_text_at_all_refuses_without_calling_the_reducer(monkeypatch):
    """Every fact unreadable means there is no evidence to hand the reducer; an
    empty FINDINGS block must not reach the model. The communities WERE
    shortlisted and mapped, so `communities_used` still says so."""
    res, synth, _ = await _run(monkeypatch, {"c1": (["kp"], ["fa"])}, {})
    assert synth.calls == []
    assert res["answer"] == gs._REFUSAL and res["citations"] == []
    assert res["communities_used"] == [{"community_id": "c1", "title": "Title c1",
                                        "relevance": None}]
