"""Attribution check (spec §6): does the answer attribute a claim to a
vendor/product that is not among the (Vendor · Product) labels of the facts it
cites? `judge_attribution` mirrors `_faithfulness_judge`'s bounded max_tokens
retry pattern on the SAME independent eval-judge client, and must never coerce
an unusable reply to 0 (memory: "LLM empty reply coerced to a value")."""
from __future__ import annotations

import types

import pytest

import answer_api.eval_router as er

pytestmark = pytest.mark.asyncio


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        content, finish_reason = self._responses.pop(0)
        message = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
        return types.SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self, responses):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions(responses))


# ---------------------------------------------------------------------------
# judge_attribution
# ---------------------------------------------------------------------------

async def test_genuine_zero_is_preserved():
    client = _FakeClient([("0", "stop")])
    result = await er.judge_attribution(client, "m", "q", "answer", "[1] (V · P) fact")
    assert result == 0
    assert len(client.chat.completions.calls) == 1


async def test_genuine_count_is_returned():
    client = _FakeClient([("2", "stop")])
    result = await er.judge_attribution(client, "m", "q", "answer", "[1] (V · P) fact")
    assert result == 2


async def test_unusable_both_attempts_yields_none_not_zero():
    client = _FakeClient([(None, "length"), (None, "length")])
    result = await er.judge_attribution(client, "m", "q", "answer", "[1] (V · P) fact")
    assert result is None
    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 8000
    assert calls[1]["max_tokens"] == 16000


async def test_retry_recovers_on_second_attempt():
    client = _FakeClient([(None, "length"), ("1", "stop")])
    result = await er.judge_attribution(client, "m", "q", "answer", "[1] (V · P) fact")
    assert result == 1
    assert len(client.chat.completions.calls) == 2


async def test_prompt_carries_the_labelled_facts_verbatim():
    client = _FakeClient([("0", "stop")])
    await er.judge_attribution(client, "m", "which vendor?", "the answer text",
                                "[1] (Veeam · B&R) does X")
    prompt = client.chat.completions.calls[0]["messages"][0]["content"]
    assert "which vendor?" in prompt
    assert "the answer text" in prompt
    assert "[1] (Veeam · B&R) does X" in prompt
    assert "Reply with ONLY the integer" in prompt


# ---------------------------------------------------------------------------
# _score_one wiring: labelled facts built with attribution.fact_line, in
# citation order, paired by fact_uuid (R7) via `_cited_fact_texts_by_uuid` --
# never by position against `_cited_fact_texts`'s list.
# ---------------------------------------------------------------------------

def _settings_and_resolver():
    from answer_api.scope import ScopeResolver
    from graph_extract.config import ExtractSettings
    settings = ExtractSettings(_env_file=None, docext_base_url="http://x",
                                docext_read_key="k", neo4j_uri="bolt://x",
                                neo4j_user="u", neo4j_password="p")
    return settings, ScopeResolver([], [], {})


_CITATIONS = [
    {"marker": 1, "fact_uuid": "f1",
     "sources": [{"vendor": "Veeam", "product": "B&R", "url": "u1", "title": "t1"}]},
    {"marker": 2, "fact_uuid": "f2",
     "sources": [{"vendor": "Rubrik", "product": "RSC", "url": "u2", "title": "t2"}]},
]


async def test_score_one_builds_labelled_facts_from_citation_sources(monkeypatch):
    settings, resolver = _settings_and_resolver()
    env = {"mode": "local", "answer": "ans [1] [2].", "citations": _CITATIONS,
           "routing": {"chosen": "local", "via": "llm", "fallback_from": None}}

    async def _fake_router(*a, q, mode_override, scope, settings):
        return env

    async def _fake_facts(driver, g, uuids):
        return ["Veeam fact", "Rubrik fact"]

    async def _fake_texts_by_uuid(driver, g, uuids):
        return {"f1": "Veeam fact", "f2": "Rubrik fact"}

    captured: dict = {}

    async def _fake_judge(client, model, q, answer, facts):
        return 3

    async def _fake_attribution(client, model, q, answer, labelled_facts):
        captured["labelled_facts"] = labelled_facts
        return 1

    monkeypatch.setattr(er.router_mod, "answer_router", _fake_router)
    monkeypatch.setattr(er, "_cited_fact_texts", _fake_facts)
    monkeypatch.setattr(er, "_cited_fact_texts_by_uuid", _fake_texts_by_uuid)
    monkeypatch.setattr(er, "_faithfulness_judge", _fake_judge)
    monkeypatch.setattr(er, "judge_attribution", _fake_attribution)

    q = {"question": "who does X?", "expected_article_ids": []}
    result = await er._score_one((None,) * 11, q, None, settings, resolver)
    assert len(result) == 5
    _env, _ghit, _faith, _facts, misattributed = result
    assert misattributed == 1
    assert captured["labelled_facts"] == (
        "[1] (Veeam · B&R) Veeam fact\n[2] (Rubrik · RSC) Rubrik fact")


# --- R7: pair by fact_uuid, never by position --------------------------------
# `_cited_fact_texts`'s Cypher `IN`-scan order is not guaranteed to match the
# requested uuid list. A fake driver that deliberately returns rows in REVERSE
# of the requested order must still produce correctly paired labelled lines.

class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for r in self._rows:
            yield r


class _ReverseOrderSession:
    def __init__(self, texts):
        self._texts = texts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def run(self, cypher, **kw):
        # Deliberately the REVERSE of the requested order -- Cypher's `IN`-scan
        # order is not guaranteed to match the input list.
        reversed_uuids = list(reversed(kw["u"]))
        return _Rows([{"uuid": u, "fact": self._texts[u]} for u in reversed_uuids])


class _ReverseOrderDriver:
    def __init__(self, texts):
        self._texts = texts

    def session(self):
        return _ReverseOrderSession(self._texts)


async def test_cited_fact_texts_by_uuid_pairs_correctly_under_reverse_row_order():
    driver = _ReverseOrderDriver({"f1": "Veeam fact", "f2": "Rubrik fact"})
    texts = await er._cited_fact_texts_by_uuid(driver, "g", ["f1", "f2"])
    assert texts == {"f1": "Veeam fact", "f2": "Rubrik fact"}


async def test_cited_fact_texts_by_uuid_empty_uuids_short_circuits():
    driver = _ReverseOrderDriver({})
    assert await er._cited_fact_texts_by_uuid(driver, "g", []) == {}


async def test_score_one_pairs_labelled_facts_by_uuid_despite_reverse_row_order(monkeypatch):
    """End-to-end through the REAL `_cited_fact_texts_by_uuid` (not mocked) with
    a driver that returns rows in reverse order: before R7, positional zip
    against the query's row order would have swapped the two facts' labels."""
    settings, resolver = _settings_and_resolver()
    env = {"mode": "local", "answer": "ans [1] [2].", "citations": _CITATIONS,
           "routing": {"chosen": "local", "via": "llm", "fallback_from": None}}
    driver = _ReverseOrderDriver({"f1": "Veeam fact", "f2": "Rubrik fact"})

    async def _fake_router(*a, q, mode_override, scope, settings):
        return env

    async def _fake_facts(driver, g, uuids):
        return ["irrelevant"]

    async def _fake_judge(client, model, q, answer, facts):
        return 3

    captured: dict = {}

    async def _fake_attribution(client, model, q, answer, labelled_facts):
        captured["labelled_facts"] = labelled_facts
        return 0

    monkeypatch.setattr(er.router_mod, "answer_router", _fake_router)
    monkeypatch.setattr(er, "_cited_fact_texts", _fake_facts)
    monkeypatch.setattr(er, "_faithfulness_judge", _fake_judge)
    monkeypatch.setattr(er, "judge_attribution", _fake_attribution)

    clients = (None, driver, None, None, None, None, None, None, None, None, None)
    q = {"question": "who does X?", "expected_article_ids": []}
    await er._score_one(clients, q, None, settings, resolver)
    assert captured["labelled_facts"] == (
        "[1] (Veeam · B&R) Veeam fact\n[2] (Rubrik · RSC) Rubrik fact")


async def test_labelled_facts_skips_a_citation_with_no_text():
    """A citation whose uuid has no text (edge gone) is dropped, not
    mislabeled against a neighbour's fact."""
    citations = [
        {"marker": 1, "fact_uuid": "f1",
         "sources": [{"vendor": "Veeam", "product": "B&R"}]},
        {"marker": 2, "fact_uuid": "gone",
         "sources": [{"vendor": "Rubrik", "product": "RSC"}]},
    ]
    result = er._labelled_facts(citations, {"f1": "Veeam fact"})
    assert result == "[1] (Veeam · B&R) Veeam fact"
