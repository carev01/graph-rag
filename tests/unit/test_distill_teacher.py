"""`TeacherTransport`/`AnswerStore`: the fake AsyncOpenAI that serves stored
teacher answers to graphiti's cheap-tier client, or raises `TeacherNeeded` so
the harness can collect the prompt for the teacher to answer out of band."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import pytest
from graphiti_core.llm_client import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.prompts.models import Message

from graph_extract import graphiti_client as gc
from graph_extract.config import ExtractSettings
from graph_extract.usage import CURRENT_PROMPT_NAME
from scripts.distill.teacher import AnswerStore, TeacherDropped, TeacherNeeded, TeacherTransport

ANSWER = {"extracted_entities": [{"name": "Veeam Backup", "entity_type_id": 0,
                                  "episode_indices": [0]}]}


def _messages() -> list[Message]:
    """A fresh list every call: `OpenAIGenericClient.generate_response` mutates
    `messages[0].content` in place (appends the language instruction), so
    reusing one list across two calls would change the second call's key."""
    return [Message(role="system", content="You extract entities."),
            Message(role="user", content="TEXT: Veeam Backup protects VMs.")]


def _settings(**over: Any) -> ExtractSettings:
    base: dict[str, Any] = dict(_env_file=None, neo4j_uri="bolt://x", neo4j_user="u",
                                neo4j_password="p", docext_base_url="http://x",
                                docext_read_key="k", cheap_llm_api_key="or-key",
                                embed_base_url="http://e")
    base.update(over)
    return ExtractSettings(**base)


# --- AnswerStore.key_for ---

def test_key_for_is_stable_across_dict_key_order(tmp_path):
    store = AnswerStore(tmp_path)
    messages_a = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    messages_b = [{"content": "sys", "role": "system"}, {"content": "hi", "role": "user"}]
    fmt_a = {"type": "json_schema", "json_schema": {"name": "N", "schema": {"a": 1, "b": 2}}}
    fmt_b = {"json_schema": {"schema": {"b": 2, "a": 1}, "name": "N"}, "type": "json_schema"}
    assert store.key_for(messages_a, fmt_a) == store.key_for(messages_b, fmt_b)


def test_key_for_changes_with_any_message_byte(tmp_path):
    store = AnswerStore(tmp_path)
    messages = [{"role": "user", "content": "hi"}]
    tweaked = [{"role": "user", "content": "hI"}]
    assert store.key_for(messages, None) != store.key_for(tweaked, None)


# --- AnswerStore: pending / put / drop lifecycle ---

def test_add_pending_is_idempotent(tmp_path):
    store = AnswerStore(tmp_path)
    request = {"prompt_name": "p", "messages": [], "response_format": None, "model": "m"}
    store.add_pending("k1", request)
    store.add_pending("k1", request)
    assert [k for k, _ in store.pending()] == ["k1"]


def test_put_removes_the_pending_file(tmp_path):
    store = AnswerStore(tmp_path)
    request = {"prompt_name": "p", "messages": [], "response_format": None, "model": "m"}
    store.add_pending("k1", request)
    assert (tmp_path / "pending" / "k1.json").exists()
    store.put("k1", json.dumps(ANSWER), {"validated": True})
    assert not (tmp_path / "pending" / "k1.json").exists()
    assert store.get("k1") == json.dumps(ANSWER)


def test_put_persists_across_a_new_store_instance(tmp_path):
    store = AnswerStore(tmp_path)
    store.put("k1", json.dumps(ANSWER), {})
    reloaded = AnswerStore(tmp_path)
    assert reloaded.get("k1") == json.dumps(ANSWER)


def test_drop_then_get_dropped_reason(tmp_path):
    store = AnswerStore(tmp_path)
    store.add_pending("k1", {"prompt_name": "p", "messages": [], "response_format": None,
                            "model": "m"})
    store.drop("k1", "schema mismatch twice")
    assert store.dropped_reason("k1") == "schema mismatch twice"
    assert not (tmp_path / "pending" / "k1.json").exists()
    assert store.get("k1") is None


# --- TeacherTransport over a raw dict request (no graphiti involved) ---

def test_a_missing_answer_adds_exactly_one_pending_file_even_when_called_twice(tmp_path):
    store = AnswerStore(tmp_path)
    transport = TeacherTransport(store, lambda: "extract_nodes.extract_text")
    kwargs = dict(model="m", messages=[{"role": "user", "content": "hi"}],
                 response_format={"type": "json_object"})

    with pytest.raises(TeacherNeeded) as exc1:
        asyncio.run(transport.chat.completions.create(**kwargs))
    with pytest.raises(TeacherNeeded) as exc2:
        asyncio.run(transport.chat.completions.create(**kwargs))

    assert exc1.value.key == exc2.value.key
    assert exc1.value.request["prompt_name"] == "extract_nodes.extract_text"
    assert [k for k, _ in store.pending()] == [exc1.value.key]


def test_a_dropped_key_raises_teacher_dropped(tmp_path):
    store = AnswerStore(tmp_path)
    transport = TeacherTransport(store, lambda: "extract_nodes.extract_text")
    kwargs = dict(model="m", messages=[{"role": "user", "content": "hi"}],
                 response_format=None)

    with pytest.raises(TeacherNeeded) as exc:
        asyncio.run(transport.chat.completions.create(**kwargs))
    store.drop(exc.value.key, "bad json twice")

    with pytest.raises(TeacherDropped) as dropped:
        asyncio.run(transport.chat.completions.create(**kwargs))
    assert dropped.value.key == exc.value.key
    assert dropped.value.reason == "bad json twice"


def test_a_stored_answer_returns_a_completion_shaped_like_openais(tmp_path):
    store = AnswerStore(tmp_path)
    transport = TeacherTransport(store, lambda: "extract_nodes.extract_text")
    kwargs = dict(model="m", messages=[{"role": "user", "content": "hi"}],
                 response_format=None)
    key = store.key_for(kwargs["messages"], kwargs["response_format"])
    store.put(key, json.dumps(ANSWER), {})

    resp = asyncio.run(transport.chat.completions.create(**kwargs))
    assert resp.choices[0].message.content == json.dumps(ANSWER)
    assert resp.choices[0].message.refusal is None
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens == 0
    assert resp.usage.completion_tokens == 0
    assert resp.usage.prompt_tokens_details is None
    assert resp.model == "claude-opus-teacher"


# --- Round-trip through a real OpenAIGenericClient ---

def test_a_stored_answer_parses_through_a_real_openai_generic_client(tmp_path):
    store = AnswerStore(tmp_path)
    transport = TeacherTransport(store, lambda: "extract_nodes.extract_text")
    client = OpenAIGenericClient(config=LLMConfig(model="m"), client=transport,
                                 structured_output_mode="json_schema")

    # Round 1: nothing stored yet -> TeacherNeeded, with exactly the request
    # the production cheap tier would have sent.
    with pytest.raises(TeacherNeeded) as exc:
        asyncio.run(client.generate_response(
            _messages(), response_model=ExtractedEntities,
            prompt_name="extract_nodes.extract_text"))

    store.put(exc.value.key, json.dumps(ANSWER), {"validated": True})

    # Round 2: same prompt (a fresh Message list, so no double-mutation) now
    # resolves to the stored answer and parses into the response model's shape.
    out = asyncio.run(client.generate_response(
        _messages(), response_model=ExtractedEntities,
        prompt_name="extract_nodes.extract_text"))
    assert out == ANSWER


def test_a_stored_answer_parses_through_a_real_cheap_graphiti_client(tmp_path):
    """The whole production cheap-tier stack (instrument, OpenRouter provider
    injection, _bound_index_arrays, ...) wraps the transport via
    `build_cheap_graphiti(transport=...)`. The dedup guard (which normally sets
    CURRENT_PROMPT_NAME) is not installed by this bare client call, so the
    test sets it directly -- exactly what the harness's round runner does
    around every graphiti call it drives."""
    store = AnswerStore(tmp_path)
    token = CURRENT_PROMPT_NAME.set("extract_nodes.extract_text")
    try:
        transport = TeacherTransport(store, CURRENT_PROMPT_NAME.get)
        g = gc.build_cheap_graphiti(_settings(), transport=transport)

        with pytest.raises(TeacherNeeded) as exc:
            asyncio.run(g.llm_client.generate_response(
                _messages(), response_model=ExtractedEntities,
                prompt_name="extract_nodes.extract_text"))
        assert exc.value.request["prompt_name"] == "extract_nodes.extract_text"

        store.put(exc.value.key, json.dumps(ANSWER), {"validated": True})

        out = asyncio.run(g.llm_client.generate_response(
            _messages(), response_model=ExtractedEntities,
            prompt_name="extract_nodes.extract_text"))
        assert out == ANSWER
    finally:
        CURRENT_PROMPT_NAME.reset(token)


# --- Crash safety: a kill mid-write must never lose readable data or raise
# out of AnswerStore()/pending() on restart. ---

def test_a_truncated_last_line_of_answers_jsonl_is_tolerated(tmp_path, caplog):
    good = json.dumps({"key": "k1", "answer": json.dumps(ANSWER), "meta": {}, "ts": "t"})
    truncated = '{"key": "k2", "answ'  # no trailing newline: a kill mid-write
    (tmp_path / "answers.jsonl").write_text(good + "\n" + truncated, encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        store = AnswerStore(tmp_path)

    assert store.get("k1") == json.dumps(ANSWER)
    assert store.get("k2") is None
    assert any("malformed last line" in r.message for r in caplog.records)


def test_a_truncated_last_line_of_dropped_jsonl_is_tolerated(tmp_path, caplog):
    good = json.dumps({"key": "k1", "reason": "bad json twice", "ts": "t"})
    (tmp_path / "dropped.jsonl").write_text(good + "\n" + '{"key": "k2"', encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        store = AnswerStore(tmp_path)

    assert store.dropped_reason("k1") == "bad json twice"
    assert store.dropped_reason("k2") is None


def test_a_malformed_middle_line_of_answers_jsonl_raises_with_file_and_line(tmp_path):
    bad = "{not valid json}"
    good = json.dumps({"key": "k1", "answer": json.dumps(ANSWER), "meta": {}, "ts": "t"})
    path = tmp_path / "answers.jsonl"
    path.write_text(bad + "\n" + good + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=rf"{re.escape(str(path))}:1"):
        AnswerStore(tmp_path)


def test_a_half_written_pending_file_is_skipped_not_raised(tmp_path, caplog):
    store = AnswerStore(tmp_path)
    (store.pending_dir / "halfwritten.json").write_text('{"prompt_name": "p", "mess',
                                                         encoding="utf-8")
    store.add_pending("good", {"prompt_name": "p", "messages": [], "response_format": None,
                               "model": "m"})

    with caplog.at_level(logging.WARNING):
        result = store.pending()

    assert [k for k, _ in result] == ["good"]
    assert any("malformed pending file" in r.message for r in caplog.records)


def test_add_pending_leaves_no_temp_files_behind(tmp_path):
    store = AnswerStore(tmp_path)
    store.add_pending("k1", {"prompt_name": "p", "messages": [], "response_format": None,
                            "model": "m"})
    assert list(store.pending_dir.glob("*.tmp")) == []
    assert (store.pending_dir / "k1.json").exists()


# --- note_asker / askers ---

def test_note_asker_records_every_chunk_that_asked(tmp_path):
    store = AnswerStore(tmp_path)
    store.note_asker("k1", "chunk-a")
    store.note_asker("k1", "chunk-b")
    store.note_asker("k2", "chunk-a")

    assert store.askers("k1") == ["chunk-a", "chunk-b"]
    assert store.askers("k2") == ["chunk-a"]
    assert store.askers("k3") == []

    reloaded = AnswerStore(tmp_path)
    assert reloaded.askers("k1") == ["chunk-a", "chunk-b"]


def test_drop_stores_an_optional_prompt_name(tmp_path):
    store = AnswerStore(tmp_path)
    store.drop("k1", "bad", prompt_name="extract_edges.edge")
    store.drop("k2", "bad")
    recs = [json.loads(line) for line in store.dropped_path.read_text().splitlines()]
    assert recs[0]["prompt_name"] == "extract_edges.edge"
    assert "prompt_name" not in recs[1]
    assert AnswerStore(tmp_path).dropped_reason("k1") == "bad"
