"""fact_verifier: the production side of the fact verifier -- the prompt the model was
trained on, the strict reply parser, the P2 filter and the fail-open client wrapper (spec
2026-10-29-verifier-trial-design.md section 4)."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from graph_extract import fact_verifier as fv
from graph_extract.dedup_guard import DedupIndexStats
from scripts.distill import dpo_labels, verifier_data
from scripts.distill.dpo_pairs import document_context

CUR = "Veeam Backup supports Amazon S3 as a backup repository. Restores go to the original VM."
CTX = 'DOCUMENT CONTEXT: This chunk comes from Veeam documentation for X. The documented product is "X".'


def _messages() -> list[dict]:
    return [{"role": "system", "content": "Extract facts.\n" + CTX},
            {"role": "user", "content": "<PREVIOUS_MESSAGES>\n[\"earlier\"]\n</PREVIOUS_MESSAGES>\n"
             "<CURRENT_MESSAGE>\n" + CUR + "\n</CURRENT_MESSAGE>\n<ENTITIES>\n[]\n</ENTITIES>\n"}]


def _edge(i: int) -> dict:
    return {"source_entity_name": f"S{i}", "target_entity_name": f"T{i}",
            "relation_type": "R", "fact": f"fact {i}"}


def _reply(*verdicts: str) -> str:
    return json.dumps({"labels": [{"n": n, "verdict": v, **({"quote": "supports Amazon S3 as a"}
                                                            if v == "stated" else {})}
                                  for n, v in enumerate(verdicts, 1)]})


def test_instructions_are_the_training_instructions() -> None:
    old = dpo_labels.LABEL_INSTRUCTIONS.split("Return only JSON:")[0] + verifier_data._OUTPUT
    assert fv.VERIFIER_INSTRUCTIONS == old == verifier_data.VERIFIER_INSTRUCTIONS


def test_prompt_is_byte_identical_to_the_training_prompt() -> None:
    msgs = _messages()
    rec = {"messages": msgs, "context": document_context(msgs)}
    facts = [fv.fact_line(_edge(i)) for i in range(3)]
    assert fv.verifier_user(msgs, facts) == verifier_data.verifier_user(rec, facts)
    as_objects = [SimpleNamespace(**m) for m in msgs]          # graphiti's Message objects
    assert fv.verifier_user(as_objects, facts) == verifier_data.verifier_user(rec, facts)


def test_keep_edges_drops_only_unsupported_and_none_keeps_all() -> None:
    edges = [_edge(i) for i in range(3)]
    parsed = fv.parse_reply(_reply("stated", "unsupported", "previous_only"), 3)
    kept, dropped = fv.keep_edges(parsed, edges)
    assert kept == [edges[0], edges[2]] and dropped == [edges[1]]
    assert fv.keep_edges(None, edges) == (edges, [])


class _Client:
    def __init__(self, content: str | None = None, exc: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self.content, self.exc = content, exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def _check(client: _Client, response, stats: DedupIndexStats):
    v = fv.EdgeVerifier(client, model="verifier", timeout=5.0)
    return asyncio.run(v.check(_messages(), response, stats))


def test_check_drops_unsupported_and_counts() -> None:
    client, stats = _Client(_reply("stated", "unsupported")), DedupIndexStats()
    out = _check(client, {"edges": [_edge(0), _edge(1)], "extra": 1}, stats)
    assert out == {"edges": [_edge(0)], "extra": 1}
    assert (stats.verifier_calls, stats.verifier_dropped, stats.verifier_failed_open) == (1, 1, 0)
    kw = client.calls[0]
    assert kw["model"] == "verifier" and kw["temperature"] == 0 and kw["max_tokens"] == 64 + 160 * 2
    assert kw["messages"][0] == {"role": "system", "content": fv.VERIFIER_INSTRUCTIONS}
    assert "1. S0 -> T0: fact 0\n2. S1 -> T1: fact 1" in kw["messages"][1]["content"]


def test_check_fails_open_and_counts() -> None:
    resp = {"edges": [_edge(0), _edge(1)]}
    for client in (_Client(exc=RuntimeError("down")), _Client(content=""), _Client(content=None),
                   _Client(content=_reply("unsupported")), _Client(content="not json")):
        stats = DedupIndexStats()
        assert _check(client, resp, stats) == resp
        assert (stats.verifier_calls, stats.verifier_dropped, stats.verifier_failed_open) == (1, 0, 1)


def test_check_passes_through_replies_it_cannot_read() -> None:
    client, stats = _Client(_reply("unsupported")), DedupIndexStats()
    assert _check(client, {"edges": []}, stats) == {"edges": []}
    assert client.calls == [] and stats.verifier_calls == 0 and stats.verifier_empty == 1
    for resp in ({"no_edges": 1}, ["not", "a", "dict"], {"edges": "x"}):   # contract drift
        client, stats = _Client(_reply("unsupported")), DedupIndexStats()
        assert _check(client, resp, stats) == resp
        assert client.calls == []
        assert (stats.verifier_calls, stats.verifier_failed_open) == (1, 1)
    stats = DedupIndexStats()
    assert _check(_Client(_reply()), {"edges": [_edge(0), "junk"]}, stats) == {"edges": [_edge(0), "junk"]}
    assert stats.verifier_failed_open == 1
