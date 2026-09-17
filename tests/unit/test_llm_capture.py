"""Opt-in JSONL prompt/response capture, added to graph_extract.usage.instrument.

Capture exists to build a fine-tuning dataset for the extraction/router/map
tiers (docs/proposals/2026-09-16-extraction-finetune-dataset.md); usage.py
previously recorded only integer token counts. Requirements under test:

  1. off by default -- no file opened, nothing written, when llm_capture_path
     (threaded in as `capture_path=`) is empty;
  2. on -- one parseable JSON record per call, carrying the request messages
     and the full completion text;
  3. a capture failure never breaks the LLM call it observed;
  4. no captured record ever contains the client's API key;
  5. token tallying (`CURRENT_USAGE_TALLY` / the global tally) is byte-identical
     whether capture is on or off -- this is the regression that matters most,
     per CLAUDE.md's note on CURRENT_USAGE_TALLY.

Also covers `prompt_name` (graphiti's prompt identifier, e.g. "extract_edges.edge"),
added to the capture record so the fine-tuning dataset builder can bucket examples
by prompt and weight them separately (docs/proposals/2026-09-16-extraction-finetune-
dataset.md §6). It travels via `CURRENT_PROMPT_NAME`, a ContextVar set by
dedup_guard.py's `generate_response` wrapper -- a module global would misattribute
names across the concurrent articles an ingest run has in flight (exactly the bug
CURRENT_USAGE_TALLY exists to avoid; see its comment in usage.py):

  6. a captured record carries the prompt name that was in scope for that call;
  7. two interleaved concurrent calls with different prompt names each get their
     own name -- the test a module global would fail;
  8. a call with no prompt name in scope (global-search map, router classify --
     neither goes through graphiti's LLM client) records "", never "unknown" and
     never a missing key;
  9. the ContextVar is reset after a dedup_guard-wrapped call, including when the
     wrapped call raises.
"""
from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace

import pytest

from graph_extract import usage as usage_mod
from graph_extract.dedup_guard import DedupIndexStats, install_dedup_guard

_API_KEY = "sk-do-not-leak-this-12345"


class _Msg:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None, finish_reason: str) -> None:
        self.message = _Msg(content)
        self.finish_reason = finish_reason


def _chat_resp(content: str, *, prompt: int = 10, completion: int = 5,
              cached: int = 0, finish_reason: str = "stop") -> SimpleNamespace:
    u = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                        prompt_tokens_details=SimpleNamespace(cached_tokens=cached))
    return SimpleNamespace(choices=[_Choice(content, finish_reason)], usage=u,
                           model="fake-chat-model")


class _FakeCompletions:
    def __init__(self, resp: SimpleNamespace) -> None:
        self._resp = resp
        self.seen: dict = {}

    async def create(self, **kwargs):
        self.seen = kwargs
        return self._resp


class _FakeResponsesAPI:
    def __init__(self, resp: SimpleNamespace) -> None:
        self._resp = resp
        self.seen: dict = {}

    async def parse(self, **kwargs):
        self.seen = kwargs
        return self._resp


class _FakeAsyncOpenAI:
    """Minimal stand-in for openai.AsyncOpenAI: a chat surface plus a Responses
    (structured `.parse`) surface, holding an api_key the way a real client does
    (instrument() must never read or leak it)."""

    def __init__(self, chat_resp: SimpleNamespace, responses_resp: SimpleNamespace | None = None,
                api_key: str = _API_KEY) -> None:
        self.api_key = api_key
        self.chat = SimpleNamespace(completions=_FakeCompletions(chat_resp))
        if responses_resp is not None:
            self.responses = _FakeResponsesAPI(responses_resp)


@pytest.fixture(autouse=True)
def _reset_tally():
    usage_mod.reset_tally()
    yield
    usage_mod.reset_tally()


def _read_jsonl(path) -> list[dict]:
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]


@pytest.mark.asyncio
async def test_capture_disabled_writes_nothing_and_opens_no_file(tmp_path, monkeypatch):
    would_be_path = tmp_path / "capture.jsonl"

    def _forbidden_open(*a, **kw):
        raise AssertionError("open() must not be called when capture_path is empty")

    monkeypatch.setattr(usage_mod, "open", _forbidden_open, raising=False)

    client = usage_mod.instrument(_FakeAsyncOpenAI(_chat_resp("hi")), capture_path="")
    resp = await client.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])

    assert resp.choices[0].message.content == "hi"
    assert not would_be_path.exists()


@pytest.mark.asyncio
async def test_capture_enabled_writes_one_parseable_record_per_call(tmp_path):
    path = tmp_path / "capture.jsonl"
    client = usage_mod.instrument(
        _FakeAsyncOpenAI(_chat_resp("the completion text"),
                         responses_resp=SimpleNamespace(
                             output_text="parsed completion", usage=SimpleNamespace(
                                 input_tokens=20, output_tokens=8), status="completed")),
        tier="cheap", capture_path=str(path))

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
    await client.chat.completions.create(model="m1", messages=messages, max_tokens=100)
    await client.responses.parse(model="m2", input=[{"role": "user", "content": "resp-input"}],
                                 text_format="SomeSchema")

    records = _read_jsonl(path)
    assert len(records) == 2

    chat_rec, resp_rec = records
    assert chat_rec["messages"] == messages
    assert chat_rec["completion"] == "the completion text"
    assert chat_rec["finish_reason"] == "stop"
    assert chat_rec["tier"] == "cheap"
    assert chat_rec["model"] == "m1"
    assert chat_rec["api"] == "chat.completions"
    assert chat_rec["prompt_tokens"] == 10 and chat_rec["completion_tokens"] == 5
    assert "call_id" in chat_rec and chat_rec["call_id"]
    assert "timestamp" in chat_rec and chat_rec["timestamp"]

    assert resp_rec["input"] == [{"role": "user", "content": "resp-input"}]
    assert resp_rec["completion"] == "parsed completion"
    assert resp_rec["api"] == "responses.parse"
    assert resp_rec["prompt_tokens"] == 20 and resp_rec["completion_tokens"] == 8


@pytest.mark.asyncio
async def test_capture_failure_does_not_propagate(tmp_path, monkeypatch, caplog):
    path = tmp_path / "capture.jsonl"

    def _boom(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(usage_mod, "capture_llm_call", _boom)
    client = usage_mod.instrument(_FakeAsyncOpenAI(_chat_resp("still works")),
                                  capture_path=str(path))

    with caplog.at_level(logging.WARNING):
        resp = await client.chat.completions.create(model="m", messages=[])

    assert resp.choices[0].message.content == "still works"
    assert any("capture" in r.message.lower() for r in caplog.records)


@pytest.mark.asyncio
async def test_no_captured_record_contains_the_api_key(tmp_path):
    path = tmp_path / "capture.jsonl"
    client = usage_mod.instrument(_FakeAsyncOpenAI(_chat_resp("hello"), api_key=_API_KEY),
                                  capture_path=str(path))
    await client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hello"}])

    raw_text = path.read_text()
    assert _API_KEY not in raw_text
    for record in _read_jsonl(path):
        assert _API_KEY not in json.dumps(record)


@pytest.mark.asyncio
async def test_token_tallying_identical_with_capture_on_and_off(tmp_path):
    resp = _chat_resp("x", prompt=1000, completion=200, cached=768)

    usage_mod.reset_tally()
    off_client = usage_mod.instrument(_FakeAsyncOpenAI(resp), capture_path="")
    await off_client.chat.completions.create(model="m", messages=[])
    off_tally = usage_mod.get_tally()
    off_totals = (off_tally.prompt_tokens, off_tally.completion_tokens,
                 off_tally.cached_tokens, off_tally.calls)

    usage_mod.reset_tally()
    on_client = usage_mod.instrument(_FakeAsyncOpenAI(resp),
                                     capture_path=str(tmp_path / "capture.jsonl"))
    await on_client.chat.completions.create(model="m", messages=[])
    on_tally = usage_mod.get_tally()
    on_totals = (on_tally.prompt_tokens, on_tally.completion_tokens,
                on_tally.cached_tokens, on_tally.calls)

    assert off_totals == on_totals == (1000, 200, 768, 1)


# ------------------------------------------------------------------- prompt_name

@pytest.mark.asyncio
async def test_captured_record_carries_the_prompt_name_in_scope(tmp_path):
    path = tmp_path / "capture.jsonl"
    client = usage_mod.instrument(_FakeAsyncOpenAI(_chat_resp("hi")), capture_path=str(path))

    token = usage_mod.CURRENT_PROMPT_NAME.set("extract_edges.edge")
    try:
        await client.chat.completions.create(model="m", messages=[])
    finally:
        usage_mod.CURRENT_PROMPT_NAME.reset(token)

    records = _read_jsonl(path)
    assert records[0]["prompt_name"] == "extract_edges.edge"


@pytest.mark.asyncio
async def test_call_with_no_prompt_name_in_scope_records_empty_string(tmp_path):
    path = tmp_path / "capture.jsonl"
    client = usage_mod.instrument(_FakeAsyncOpenAI(_chat_resp("hi")), capture_path=str(path))

    assert usage_mod.CURRENT_PROMPT_NAME.get() == ""  # nothing set this scope
    await client.chat.completions.create(model="m", messages=[])

    records = _read_jsonl(path)
    assert "prompt_name" in records[0]
    assert records[0]["prompt_name"] == ""
    assert records[0]["prompt_name"] != "unknown"


@pytest.mark.asyncio
async def test_concurrent_calls_each_get_their_own_prompt_name(tmp_path):
    """The test a module-global CURRENT_PROMPT_NAME would fail.

    Interleaving, driven by three events so the order is deterministic rather than
    scheduler-luck: A sets its name, then (with a real await in between -- the
    required gap between "set" and "use") waits for B to set ITS name before A
    makes its own call; B, in turn, waits for A to finish (call + reset) before
    making its own call. So by construction: A opens -> B opens -> A calls+closes
    -> B calls+closes -- B's window is open while A performs its call, and A's
    close lands before B performs its.

    Under a real ContextVar (per-task copied context) each task's call reads back
    exactly the name it set, regardless of the other task's writes to ITS OWN
    context. Under a plain module-level variable this fails on both sides: A's
    call would read B's name (set into the shared variable while A's window was
    still open), and B's call would then read the empty string A's reset left
    behind -- verified against a throwaway reimplementation of this exact
    set/get/reset shape backed by a plain module global instead of
    `contextvars.ContextVar`, which reproduces precisely
    `{"a": "dedupe_nodes.nodes", "b": ""}` for this interleaving.
    """
    path = tmp_path / "capture.jsonl"
    client = usage_mod.instrument(_FakeAsyncOpenAI(_chat_resp("x")), capture_path=str(path))
    a_set = asyncio.Event()
    b_set = asyncio.Event()
    a_done = asyncio.Event()

    async def _task_a():
        token = usage_mod.CURRENT_PROMPT_NAME.set("extract_edges.edge")
        a_set.set()
        await b_set.wait()  # the required gap between "set" and "use"
        await client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "a"}])
        usage_mod.CURRENT_PROMPT_NAME.reset(token)
        a_done.set()

    async def _task_b():
        await a_set.wait()
        token = usage_mod.CURRENT_PROMPT_NAME.set("dedupe_nodes.nodes")
        b_set.set()
        await a_done.wait()
        await client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "b"}])
        usage_mod.CURRENT_PROMPT_NAME.reset(token)

    await asyncio.gather(_task_a(), _task_b())

    by_input = {r["messages"][0]["content"]: r["prompt_name"] for r in _read_jsonl(path)}
    assert by_input == {"a": "extract_edges.edge", "b": "dedupe_nodes.nodes"}


@pytest.mark.asyncio
async def test_dedup_guard_resets_prompt_name_after_a_call():
    class _Primary:
        def __init__(self) -> None:
            self.seen_during_call: str | None = None

        async def generate_response(self, messages, *a, **kw):
            self.seen_during_call = usage_mod.CURRENT_PROMPT_NAME.get()
            return {"duplicate_facts": [], "contradicted_facts": []}

    primary = _Primary()
    holder = SimpleNamespace(llm_client=primary)
    install_dedup_guard(holder, fallback=None, unscoped=DedupIndexStats())

    assert usage_mod.CURRENT_PROMPT_NAME.get() == ""
    await holder.llm_client.generate_response([], prompt_name="extract_edges.edge")
    assert primary.seen_during_call == "extract_edges.edge"
    assert usage_mod.CURRENT_PROMPT_NAME.get() == ""


@pytest.mark.asyncio
async def test_dedup_guard_resets_prompt_name_after_a_call_that_raises():
    class _Boom:
        async def generate_response(self, messages, *a, **kw):
            raise RuntimeError("strong tier down")

    holder = SimpleNamespace(llm_client=_Boom())
    install_dedup_guard(holder, fallback=None, unscoped=DedupIndexStats())

    with pytest.raises(RuntimeError, match="strong tier down"):
        await holder.llm_client.generate_response([], prompt_name="extract_edges.edge")
    assert usage_mod.CURRENT_PROMPT_NAME.get() == ""
