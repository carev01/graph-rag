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


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Record retry backoffs instead of sleeping."""
    slept: list[float] = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(runmod, "_sleep", fake_sleep)
    return slept


def _parity_ok(meta):
    return None


def _smoke_ok(out: Path) -> None:
    (out / "smoke").mkdir(parents=True, exist_ok=True)
    (out / "smoke" / "SMOKE_PASSED").write_text("{}")


async def _go(qs, transport, vk, out, **kw):
    kw.setdefault("graph_state_check", _ok)
    kw.setdefault("index_check", _ok)
    kw.setdefault("parity_check", _parity_ok)
    kw.setdefault("meta_collector", _meta)
    if kw.get("limit") is None:
        _smoke_ok(out)
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
    assert g["flags"] == {"mode": "local", "retrieved": 5, "cited": 1, "degraded": None,
                          "degraded_facets": None, "relevance_unchecked": None, "scope": None,
                          "applies_to": None, "item_grades": None, "checked_vendors": None}
    assert "raw" not in g
    assert g["usage"] is None and g["latency_s"] >= 0
    v = next(r for r in lines if r["side"] == "vendorkb")
    assert set(v["flags"]) == {"uncited", "unknown_markers", "scope_relaxed", "rerank_ok",
                               "retrieved", "scope"}
    assert v["flags"]["scope"] == {"vendors": [], "products": []}
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
    assert {r["qid"] for r in _lines(tmp_path / "smoke")} == {"q0", "q1"}
    assert not (tmp_path / "answers.jsonl").exists()  # a smoke run never shares --out


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
    errored = {"qid": "q1", "side": "graph", "answer": None, "citations": [], "flags": {},
               "error": "HTTP 500"}
    with pytest.raises(SystemExit, match="q1"):
        runmod.smoke_assert([*rows, errored])
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


@pytest.mark.asyncio
async def test_pool_gives_concurrent_calls_distinct_connections():
    import asyncio

    class Conn:
        closed = False

        async def close(self):
            self.closed = True

    async def factory():
        return Conn()

    pool = await runmod.ConnPool.create(factory, 2)
    seen = []

    async def call():
        async with pool.acquire() as c:
            seen.append(c)
            await asyncio.sleep(0.01)

    await asyncio.gather(call(), call())
    assert len(seen) == 2 and seen[0] is not seen[1]
    conns = list(pool._all)
    await pool.close()
    assert all(c.closed for c in conns)


@pytest.mark.asyncio
async def test_retried_vendorkb_usage_is_summed(tmp_path):
    n = {"i": 0}

    async def fn(q):
        n["i"] += 1
        first = n["i"] == 1
        return {"answer": None if first else "ok [1]", "error": "empty reply" if first else None,
                "citations": [1], "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                            "cost": 0.5}, "scope": {}}

    await _go(_qs(1), Transport(), fn, tmp_path)
    v = next(r for r in _lines(tmp_path) if r["side"] == "vendorkb")
    assert v["usage"] == {"prompt_tokens": 20, "completion_tokens": 10, "cost": 1.0}


@pytest.mark.asyncio
async def test_resume_tolerates_torn_line_and_keeps_run_meta(tmp_path):
    await _go(_qs(1), Transport(), _vk(), tmp_path)
    p = tmp_path / "answers.jsonl"
    p.write_text(p.read_text() + '{"qid": "q1", "si')  # torn, no newline
    await _go(_qs(2), Transport(), _vk(), tmp_path)
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.startswith("{\"qid\": \"q")
            and x.endswith("}")]
    assert {(r["qid"], r["side"]) for r in rows} >= {("q1", "graph"), ("q1", "vendorkb")}
    assert (tmp_path / "run_meta.json").exists()
    assert list(tmp_path.glob("run_meta.resume-*.json"))


# --- final-review fixes (I2-I8, M1) ---------------------------------------------------


@pytest.mark.asyncio
async def test_retry_waits_backoff_before_second_attempt(tmp_path, no_backoff):
    t = Transport(lambda n, req: httpx.Response(500, text="boom"))
    calls: list[str] = []
    await _go(_qs(1), t, _vk(calls, answer=None, error="empty reply"), tmp_path)
    assert no_backoff == [runmod.RETRY_BACKOFF_S, runmod.RETRY_BACKOFF_S]
    assert runmod.RETRY_BACKOFF_S == 5.0


@pytest.mark.asyncio
async def test_retry_errors_reasks_pairs_whose_last_row_is_an_error(tmp_path):
    rows = [
        {"qid": "q0", "side": "graph", "answer": None, "citations": [], "latency_s": 1,
         "usage": None, "error": "first", "flags": {}},
        {"qid": "q0", "side": "graph", "answer": "fixed", "citations": [], "latency_s": 1,
         "usage": None, "error": None, "flags": {}},
        {"qid": "q0", "side": "vendorkb", "answer": None, "citations": [], "latency_s": 1,
         "usage": None, "error": "boom", "flags": {}},
    ]
    (tmp_path / "answers.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    t = Transport()
    calls: list[str] = []
    summary = await _go(_qs(1), t, _vk(calls), tmp_path)
    assert t.calls == 0 and calls == []  # without the flag an error row is final
    summary = await _go(_qs(1), t, _vk(calls), tmp_path, retry_errors=True)
    assert t.calls == 0  # graph's LAST row is an answer: not re-asked
    assert calls == ["question 0?"]
    assert summary["vendorkb"]["ok"] == 1 and summary["graph"]["ok"] == 1


@pytest.mark.asyncio
async def test_circuit_breaker_aborts_after_five_consecutive_errors(tmp_path):
    t = Transport(lambda n, req: httpx.Response(500, text="down"))
    with pytest.raises(runmod.CircuitBreakerOpen, match="graph"):
        await _go(_qs(20), t, _vk(), tmp_path, concurrency=1)
    graph_rows = [r for r in _lines(tmp_path) if r["side"] == "graph"]
    assert len(graph_rows) == runmod.MAX_CONSECUTIVE_ERRORS
    assert t.calls == 2 * runmod.MAX_CONSECUTIVE_ERRORS  # one retry each, then stop


@pytest.mark.asyncio
async def test_an_answer_resets_the_consecutive_error_count(tmp_path):
    def handler(n, req):
        q = req.url.params["q"]
        return httpx.Response(200, json=_envelope()) if q.endswith("4?") else httpx.Response(500)
    t = Transport(handler)
    summary = await _go(_qs(8), t, _vk(), tmp_path, concurrency=1)
    assert summary["graph"]["errors"] == 7 and summary["graph"]["ok"] == 1


def test_circuit_breaker_exits_non_zero_from_main(monkeypatch, capsys):
    async def fake_amain(args):
        raise runmod.CircuitBreakerOpen("graph: 5 consecutive errors")
    monkeypatch.setattr(runmod, "_run_all", fake_amain)
    assert runmod.main(["--out", "x"]) != 0
    assert "5 consecutive errors" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_full_run_refuses_without_smoke_marker(tmp_path):
    t = Transport()
    calls: list[str] = []
    async with t.client() as c:
        with pytest.raises(runmod.RunRefused, match="SMOKE_PASSED"):
            await runmod.run(_qs(2), answer_api=c, vendorkb_answer=_vk(calls), out_dir=tmp_path,
                             limit=None, graph_state_check=_ok, index_check=_ok,
                             parity_check=_parity_ok, meta_collector=_meta)
    assert t.calls == 0 and calls == []


def test_smoke_marker_written_after_a_passing_smoke(tmp_path):
    runmod.write_smoke_marker(tmp_path / "smoke", "abc123")
    data = json.loads((tmp_path / "smoke" / "SMOKE_PASSED").read_text())
    assert data["git_commit"] == "abc123" and data["timestamp"]


@pytest.mark.asyncio
async def test_index_check_runs_before_any_request(tmp_path):
    async def incomplete():
        runmod.check_index_counts(documents=10, corpus_n=20, failed_n=3)

    t = Transport()
    calls: list[str] = []
    with pytest.raises(runmod.RunRefused, match="10.*17"):
        await _go(_qs(2), t, _vk(calls), tmp_path, index_check=incomplete)
    assert t.calls == 0 and calls == []


def test_index_counts():
    runmod.check_index_counts(documents=17, corpus_n=20, failed_n=3)
    with pytest.raises(runmod.RunRefused):
        runmod.check_index_counts(documents=20, corpus_n=20, failed_n=3)


def test_index_check_reads_corpus_and_failed(tmp_path):
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps([{"article_id": str(i)} for i in range(5)]))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"failed": ["1: HTTP 404", "2: timeout"]}))
    assert runmod.expected_documents(corpus, report) == (5, 2)


def _settings():
    from vendorkb.config import VendorkbSettings
    return VendorkbSettings(_env_file=None, judge_model="z-ai/glm-5.3-flash",
                            rerank_model="rerank-3", answer_reasoning_effort="low")


def test_parity_passes_and_unset_effort_means_low():
    s = _settings()
    meta = {"JUDGE_MODEL": "z-ai/glm-5.3-flash", "RERANK_MODEL": "rerank-3",
            "SYNTHESIS_REASONING_EFFORT": None, "unset_env": ["SYNTHESIS_REASONING_EFFORT"]}
    runmod.check_parity(meta, s)
    runmod.check_parity(meta | {"SYNTHESIS_REASONING_EFFORT": "low", "unset_env": []}, s)


@pytest.mark.parametrize("over,name", [
    ({"JUDGE_MODEL": "other/model"}, "JUDGE_MODEL"),
    ({"RERANK_MODEL": "rerank-2"}, "RERANK_MODEL"),
    ({"SYNTHESIS_REASONING_EFFORT": "high"}, "SYNTHESIS_REASONING_EFFORT"),
    ({"JUDGE_MODEL": None}, "JUDGE_MODEL"),
    ({"SYNTHESIS_REASONING_EFFORT": None, "unset_env": []}, "SYNTHESIS_REASONING_EFFORT"),
])
def test_parity_refuses_mismatch_or_null_read(over, name):
    meta = {"JUDGE_MODEL": "z-ai/glm-5.3-flash", "RERANK_MODEL": "rerank-3",
            "SYNTHESIS_REASONING_EFFORT": "low", "unset_env": []} | over
    with pytest.raises(runmod.RunRefused, match=name):
        runmod.check_parity(meta, _settings())


@pytest.mark.asyncio
async def test_parity_check_runs_before_any_request(tmp_path):
    def bad(meta):
        raise runmod.RunRefused("model parity: JUDGE_MODEL")

    t = Transport()
    with pytest.raises(runmod.RunRefused):
        await _go(_qs(1), t, _vk(), tmp_path, parity_check=bad)
    assert t.calls == 0


FROZEN = {"max_episode_created_at": "e1", "min_community_generated_at": "m0",
          "max_community_generated_at": "c1", "community_count": 883,
          "stale_community_count": 2}


def test_graph_state_compares_communities_too():
    runmod.compare_graph_state(FROZEN, dict(FROZEN))
    for key, val in (("max_episode_created_at", "e2"), ("max_community_generated_at", "c2"),
                     ("community_count", 884), ("stale_community_count", 3)):
        with pytest.raises(runmod.RunRefused, match=key):
            runmod.compare_graph_state(FROZEN, FROZEN | {key: val})


def test_summary_counts_rerank_failures_and_uncited():
    rows = [
        {"qid": "q0", "side": "vendorkb", "answer": "a", "flags": {"rerank_ok": False,
                                                                   "uncited": True}},
        {"qid": "q1", "side": "vendorkb", "answer": "b", "flags": {"rerank_ok": True,
                                                                   "uncited": False}},
        {"qid": "q0", "side": "graph", "answer": "c", "flags": {}},
    ]
    s = runmod.summarize(rows)
    assert s["vendorkb"]["rerank_not_ok"] == 1 and s["vendorkb"]["uncited"] == 1
    assert s["graph"] == {"ok": 1, "errors": 0, "refusals": 0}


def test_graph_flags_keep_diagnostics():
    env = _envelope() | {"degraded": "no-primer-communities", "degraded_facets": ["x"],
                         "relevance_unchecked": ["Veeam"], "scope": {"vendors": ["Veeam"]},
                         "applies_to": ["Veeam · VBR"], "item_grades": [{"item": "a"}],
                         "checked_vendors": ["Veeam", "AWS"]}
    f = runmod.graph_flags(env)
    assert f["degraded"] == "no-primer-communities" and f["checked_vendors"] == ["Veeam", "AWS"]
    assert f["item_grades"] == [{"item": "a"}] and f["mode"] == "local"


@pytest.mark.parametrize("text", [
    REFUSAL,
    "I don't have enough thematic coverage to answer that from the community reports.",
    REFUSAL + "\n\nChecked: Veeam, AWS.",
])
def test_every_answer_api_refusal_wording_counts(text):
    assert runmod._refusal(text)


def test_non_refusal_not_counted():
    assert not runmod._refusal("Veeam supports this [1].")


def _key_transport(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_openrouter_key_usage_reads_usage_with_bearer():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers.get("authorization")
        seen["url"] = str(req.url)
        return httpx.Response(200, json={"data": {"usage": 12.5, "limit": None}})

    async with _key_transport(handler) as c:
        assert await runmod.openrouter_key_usage(c, "k-123") == 12.5
    assert seen == {"auth": "Bearer k-123", "url": runmod.OPENROUTER_KEY_URL}


@pytest.mark.asyncio
async def test_openrouter_key_usage_failure_is_none():
    async with _key_transport(lambda req: httpx.Response(401, json={})) as c:
        assert await runmod.openrouter_key_usage(c, "k") is None
    async with _key_transport(lambda req: httpx.Response(200, json={"data": {}})) as c:
        assert await runmod.openrouter_key_usage(c, "k") is None

    def boom(req):
        raise httpx.ConnectError("no route")

    async with _key_transport(boom) as c:
        assert await runmod.openrouter_key_usage(c, "k") is None
    async with _key_transport(lambda req: httpx.Response(200, json={})) as c:
        assert await runmod.openrouter_key_usage(c, "") is None  # no key: no request


def test_cost_lines_measured():
    lines = runmod.cost_lines(10.0, 10.5, vendorkb_cost=0.1, n_questions=2, total_questions=100)
    text = "\n".join(lines)
    assert ("OpenRouter key usage delta: $0.5000 (covers graph-side GLM calls only if the "
            "answer-api pod uses the same key; vendorkb GLM measured separately: $0.1000)"
            ) in text
    assert "Voyage" in text and "not measurable" in text
    assert "NOT measured" not in text


def test_cost_lines_not_measured():
    text = "\n".join(runmod.cost_lines(None, 3.0, vendorkb_cost=0.1, n_questions=2,
                                       total_questions=100))
    assert "graph side cost NOT measured" in text and "Voyage" in text


def test_frozen_keys_are_exported_state_queries():
    from scripts.headtohead.export_corpus import STATE_QUERIES
    assert set(runmod.FROZEN_KEYS) <= set(STATE_QUERIES)
