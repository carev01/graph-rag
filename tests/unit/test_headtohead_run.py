from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from scripts.headtohead import run as runmod
from vendorkb.answer import REFUSAL


def _qs(n):
    return [{"id": f"q{i}", "question": f"question {i}?"} for i in range(n)]


def _envelope(answer="A [1]"):
    return {"query": "q", "answer": answer, "retrieved": 5, "cited": 1, "mode": "local",
            "citations": [{"marker": 1, "fact_uuid": "u", "valid_at": None, "invalid_at": None,
                           "sources": [{"url": "http://x"}]}]}


class Transport:
    def __init__(self, handler=None):
        self.calls = 0
        self.handler = handler or (lambda n, req: httpx.Response(200, json=_envelope()))

    def __call__(self, request):
        self.calls += 1
        return self.handler(self.calls, request)

    def client(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self), base_url="http://api")


def _vk(calls=None, **over):
    async def fn(q):
        if calls is not None:
            calls.append(q)
        out = {"query": q, "answer": "VK [1]", "citations": [{"marker": 1}],
               "unknown_markers": [], "uncited": False,
               "scope": {"vendors": [], "products": [], "relaxed": False},
               "rerank_ok": True, "retrieved": 10,
               "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
               "error": None}
        return out | over
    return fn


async def _ok():
    return None


async def _meta():
    return {"image": "img:1"}


async def _go(qs, transport, vk, out, **kw):
    kw.setdefault("graph_state_check", _ok)
    kw.setdefault("meta_collector", _meta)
    async with transport.client() as c:
        return await runmod.run(qs, answer_api=c, vendorkb_answer=vk, out_dir=out,
                                limit=kw.pop("limit", None), **kw)


def _lines(out: Path):
    return [json.loads(x) for x in (out / "answers.jsonl").read_text().splitlines()]


@pytest.mark.asyncio
async def test_each_question_yields_one_line_per_side(tmp_path):
    summary = await _go(_qs(3), Transport(), _vk(), tmp_path)
    lines = _lines(tmp_path)
    assert len(lines) == 6
    assert {(r["qid"], r["side"]) for r in lines} == {
        (f"q{i}", s) for i in range(3) for s in ("graph", "vendorkb")}
    g = next(r for r in lines if r["side"] == "graph")
    assert g["flags"] == {"mode": "local", "retrieved": 5, "cited": 1}
    assert g["usage"] is None and g["latency_s"] >= 0
    v = next(r for r in lines if r["side"] == "vendorkb")
    assert set(v["flags"]) == {"uncited", "unknown_markers", "scope_relaxed", "rerank_ok",
                               "retrieved"}
    assert summary["graph"] == {"ok": 3, "errors": 0, "refusals": 0}
    assert (tmp_path / "run_meta.json").exists()


@pytest.mark.asyncio
async def test_failure_is_retried_once_then_recorded_not_skipped(tmp_path):
    t = Transport(lambda n, req: httpx.Response(500, text="boom"))
    summary = await _go(_qs(1), t, _vk(), tmp_path)
    assert t.calls == 2
    g = next(r for r in _lines(tmp_path) if r["side"] == "graph")
    assert g["answer"] is None and g["error"]
    assert summary["graph"]["errors"] == 1
    t2 = Transport(lambda n, req: httpx.Response(500) if n == 1
                   else httpx.Response(200, json=_envelope()))
    out2 = tmp_path / "two"
    await _go(_qs(1), t2, _vk(), out2)
    assert next(r for r in _lines(out2) if r["side"] == "graph")["answer"] == "A [1]"


@pytest.mark.asyncio
async def test_empty_envelope_answer_is_a_failure(tmp_path):
    t = Transport(lambda n, req: httpx.Response(200, json=_envelope(answer="")))
    await _go(_qs(1), t, _vk(), tmp_path)
    g = next(r for r in _lines(tmp_path) if r["side"] == "graph")
    assert t.calls == 2 and g["answer"] is None and g["error"]


@pytest.mark.asyncio
async def test_vendorkb_error_retried_and_recorded(tmp_path):
    calls: list[str] = []
    await _go(_qs(1), Transport(), _vk(calls, answer=None, error="empty reply"), tmp_path)
    v = next(r for r in _lines(tmp_path) if r["side"] == "vendorkb")
    assert len(calls) == 2 and v["answer"] is None and v["error"] == "empty reply"


@pytest.mark.asyncio
async def test_limit_runs_only_n_questions(tmp_path):
    t = Transport()
    await _go(_qs(5), t, _vk(), tmp_path, limit=2)
    assert t.calls == 2
    assert {r["qid"] for r in _lines(tmp_path)} == {"q0", "q1"}


@pytest.mark.asyncio
async def test_refuses_when_graph_changed(tmp_path):
    async def changed():
        raise RuntimeError("graph changed")

    t = Transport()
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="graph changed"):
        await _go(_qs(2), t, _vk(calls), tmp_path, graph_state_check=changed)
    assert t.calls == 0 and calls == []


@pytest.mark.asyncio
async def test_resume_skips_completed_pairs(tmp_path):
    done = [
        {"qid": "q0", "side": "graph", "answer": "old", "citations": [], "latency_s": 1,
         "usage": None, "error": None, "flags": {}},
        {"qid": "q0", "side": "vendorkb", "answer": None, "citations": [], "latency_s": 1,
         "usage": None, "error": "final failure", "flags": {}},
        {"qid": "q1", "side": "graph", "answer": None, "citations": [], "latency_s": 1,
         "usage": None, "error": None, "flags": {}},
    ]
    (tmp_path / "answers.jsonl").write_text("\n".join(json.dumps(r) for r in done) + "\n")
    t = Transport()
    calls: list[str] = []
    summary = await _go(_qs(2), t, _vk(calls), tmp_path)
    assert t.calls == 1  # only (q1, graph) is rerun
    assert calls == ["question 1?"]
    assert len(_lines(tmp_path)) == 5
    assert summary["graph"]["ok"] == 2 and summary["vendorkb"]["errors"] == 1


@pytest.mark.asyncio
async def test_refusal_counted(tmp_path):
    summary = await _go(_qs(1), Transport(), _vk(answer=REFUSAL, citations=[]), tmp_path)
    assert summary["vendorkb"]["refusals"] == 1


def test_smoke_check():
    rows = [
        {"qid": "q0", "side": "graph", "answer": "a", "citations": [1], "flags": {}},
        {"qid": "q0", "side": "vendorkb", "answer": "b", "citations": [1],
         "flags": {"rerank_ok": True}},
    ]
    runmod.smoke_assert(rows)
    with pytest.raises(SystemExit):
        runmod.smoke_assert([rows[0] | {"citations": []}, rows[1]])
    with pytest.raises(SystemExit):
        runmod.smoke_assert([rows[0], rows[1] | {"flags": {"rerank_ok": False}}])
    with pytest.raises(SystemExit):
        runmod.smoke_assert([rows[0], rows[1] | {"answer": REFUSAL}])


def test_settings_dump_drops_secret_fields():
    from vendorkb.config import VendorkbSettings
    d = runmod.settings_dump(VendorkbSettings(
        _env_file=None, judge_api_key="sekret", vendorkb_dsn="postgresql://u:pw@h/db"))
    assert d["judge_model"] and "sekret" not in json.dumps(d) and "pw" not in json.dumps(d)
    assert not [k for k in d if any(w in k for w in ("key", "dsn", "password", "token"))]
