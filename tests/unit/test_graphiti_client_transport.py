"""`build_cheap_graphiti(transport=, graph_driver=)`: the injected transport gets
every production wrapper, in production order; the defaults are unchanged."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.prompts.models import Message

from graph_extract import graphiti_client as gc
from graph_extract.config import ExtractSettings
from scripts.distill.readonly_driver import ReadOnlyNeo4jDriver, ReadOnlyViolation

ANSWER = {"extracted_entities": [{"name": "Veeam Backup", "entity_type_id": 0,
                                  "episode_indices": [0]}]}


def _settings(**over: Any) -> ExtractSettings:
    base: dict[str, Any] = dict(_env_file=None, neo4j_uri="bolt://x", neo4j_user="u",
                                neo4j_password="p", docext_base_url="http://x",
                                docext_read_key="k", cheap_llm_api_key="or-key",
                                embed_base_url="http://e")
    base.update(over)
    return ExtractSettings(**base)


class FakeTransport:
    """A stand-in AsyncOpenAI: `chat.completions.create` records every call and
    returns a canned chat completion. The wrappers re-assign `create`, so the
    attribute must be writable -- as on the real client."""

    def __init__(self, answer: dict = ANSWER) -> None:
        self.calls: list[dict] = []
        self.closed = False

        async def create(*args: Any, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)),
                                         finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5,
                                      prompt_tokens_details=None),
                model="fake-model")

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    async def close(self) -> None:
        self.closed = True


def _messages() -> list[Message]:
    return [Message(role="system", content="You extract entities."),
            Message(role="user", content="TEXT: Veeam Backup protects VMs.")]


def _extract(g: Any) -> dict:
    return asyncio.run(g.llm_client.generate_response(
        _messages(), response_model=ExtractedEntities,
        prompt_name="extract_nodes.extract_text"))


def test_the_injected_transport_receives_the_production_wrapped_request():
    fake = FakeTransport()
    g = gc.build_cheap_graphiti(_settings(), transport=fake)
    out = _extract(g)
    assert out == ANSWER
    assert len(fake.calls) == 1
    call = fake.calls[0]
    # _bound_index_arrays ran on the way down: episode_indices is bounded.
    schema = call["response_format"]["json_schema"]["schema"]
    item = schema["$defs"]["ExtractedEntity"]["properties"]["episode_indices"]
    assert item["maxItems"] == 25
    # The default cheap base URL is OpenRouter: the provider preference is attached.
    assert call["extra_body"]["provider"]["require_parameters"] is True
    # The messages are the ones OpenAIGenericClient built (its suffix is appended
    # to the system message), not the caller's originals.
    assert call["messages"][0]["role"] == "system"
    assert call["messages"][0]["content"].startswith("You extract entities.")
    assert call["messages"][0]["content"] != "You extract entities."
    assert call["messages"][1] == {"role": "user",
                                   "content": "TEXT: Veeam Backup protects VMs."}
    assert call["model"] == _settings().cheap_llm_model
    # The transport IS the client graphiti holds (wrapped in place, not copied).
    assert g.llm_client.client is fake


def test_the_wrappers_run_in_production_order(monkeypatch):
    """Same order as the default path: instrument, openrouter, penalties, bound,
    strict, drop temperature, cache layout."""
    order: list[str] = []

    def rec(name: str):
        def wrap(client: Any, *args: Any, **kwargs: Any) -> Any:
            order.append(name)
            return client
        return wrap

    monkeypatch.setattr(gc, "instrument", rec("instrument"))
    monkeypatch.setattr(gc, "_inject_openrouter_provider", rec("provider"))
    monkeypatch.setattr(gc, "_inject_openrouter_reasoning", rec("reasoning"))
    monkeypatch.setattr(gc, "_inject_penalties", rec("penalties"))
    monkeypatch.setattr(gc, "_bound_index_arrays", rec("bound"))
    monkeypatch.setattr(gc, "_strict_schemas", rec("strict"))
    monkeypatch.setattr(gc, "_drop_temperature", rec("drop_temperature"))
    monkeypatch.setattr(gc, "wrap_client", rec("cache_layout"))
    s = _settings(cheap_llm_cache_layout=True, llm_openrouter_reasoning_effort="low",
                  llm_frequency_penalty=0.1, llm_strict_schema=True,
                  llm_send_temperature=False)
    expected = ["instrument", "provider", "reasoning", "penalties", "bound", "strict",
                "drop_temperature", "cache_layout"]
    fake = FakeTransport()
    gc._llm_client(gc._cheap_view_settings(s), tier="cheap", transport=fake)
    assert order == expected
    order.clear()
    gc._llm_client(gc._cheap_view_settings(s), tier="cheap")      # the default path
    assert order == expected


def test_a_capture_record_is_written_for_the_transport_call(tmp_path):
    cap = tmp_path / "capture.jsonl"
    fake = FakeTransport()
    g = gc.build_cheap_graphiti(_settings(llm_capture_path=str(cap)), transport=fake)
    _extract(g)
    lines = cap.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["tier"] == "cheap"
    assert json.loads(rec["completion"]) == ANSWER
    assert rec["model"] == _settings().cheap_llm_model


def test_the_graph_driver_is_injected_instead_of_the_uri():
    ro = ReadOnlyNeo4jDriver("bolt://localhost:1", "u", "p")
    g = gc.build_cheap_graphiti(_settings(), transport=FakeTransport(), graph_driver=ro)
    assert g.driver is ro
    asyncio.run(g.close())


def test_close_also_closes_the_injected_transport():
    fake = FakeTransport()
    ro = ReadOnlyNeo4jDriver("bolt://localhost:1", "u", "p")
    g = gc.build_cheap_graphiti(_settings(), transport=fake, graph_driver=ro)
    asyncio.run(g.close())
    assert fake.closed is True


def test_the_default_build_passes_no_injection(monkeypatch):
    """No transport/graph_driver: build_graphiti is called exactly as before."""
    seen: list[tuple] = []
    monkeypatch.setattr(gc, "build_graphiti",
                        lambda cfg, *, tier="strong": seen.append((cfg.llm_model, tier)) or "G")
    assert gc.build_cheap_graphiti(_settings()) == "G"
    assert seen == [(_settings().cheap_llm_model, "cheap")]


def test_the_default_llm_client_builds_a_real_async_openai():
    from openai import AsyncOpenAI
    llm = gc._llm_client(gc._cheap_view_settings(_settings()), tier="cheap")
    assert isinstance(llm.client, AsyncOpenAI)


@pytest.mark.parametrize("base_url", ["https://api.anthropic.com",
                                      "https://x.openai.azure.com"])
def test_a_transport_on_the_anthropic_or_azure_branch_is_refused(base_url):
    s = _settings(cheap_llm_base_url=base_url)
    with pytest.raises(ValueError, match="transport"):
        gc.build_cheap_graphiti(s, transport=FakeTransport())


def test_add_text_episode_passes_previous_episode_uuids_through():
    seen: list[dict] = []

    class _G:
        async def add_episode(self, **kwargs: Any) -> str:
            seen.append(kwargs)
            return "R"

    s = _settings()
    when = datetime(2026, 1, 1, tzinfo=timezone.utc)
    asyncio.run(gc.add_text_episode(_G(), s, name="n", body="b", source_description="d",
                                    reference_time=when, previous_episode_uuids=["e1"]))
    asyncio.run(gc.add_text_episode(_G(), s, name="n", body="b", source_description="d",
                                    reference_time=when))
    assert seen[0]["previous_episode_uuids"] == ["e1"]
    assert seen[1]["previous_episode_uuids"] is None


# --- ReadOnlyNeo4jDriver: hermetic checks (no server; the integration file
# proves the server-side half). ---

def test_constructing_inside_a_running_loop_runs_no_index_build(monkeypatch):
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    calls: list[str] = []

    async def _build(self: Any, delete_existing: bool = False) -> None:
        calls.append("built")

    monkeypatch.setattr(Neo4jDriver, "build_indices_and_constraints", _build)

    async def main() -> Any:
        ro = ReadOnlyNeo4jDriver("bolt://localhost:1", "u", "p")
        await asyncio.sleep(0.05)          # give a scheduled task every chance to run
        assert ro._init_task is None
        await ro.close()

    asyncio.run(main())
    assert calls == []


def test_explicit_index_build_and_delete_are_refused():
    ro = ReadOnlyNeo4jDriver("bolt://localhost:1", "u", "p")
    with pytest.raises(ReadOnlyViolation):
        asyncio.run(ro.build_indices_and_constraints())
    with pytest.raises(ReadOnlyViolation):
        ro.delete_all_indexes()
    asyncio.run(ro.close())


def test_write_routing_and_write_sessions_are_refused():
    from neo4j import RoutingControl, WRITE_ACCESS
    ro = ReadOnlyNeo4jDriver("bolt://localhost:1", "u", "p")
    with pytest.raises(ReadOnlyViolation):
        asyncio.run(ro.execute_query("RETURN 1", routing_=RoutingControl.WRITE))
    with pytest.raises(ReadOnlyViolation):
        asyncio.run(ro.execute_query("RETURN 1", routing_="w"))
    with pytest.raises(ReadOnlyViolation):
        ro.client.session(default_access_mode=WRITE_ACCESS)
    with pytest.raises(ReadOnlyViolation):
        asyncio.run(ro.session().execute_write(lambda tx: None))
    asyncio.run(ro.close())


def test_every_query_and_session_is_read_routed(monkeypatch):
    from neo4j import READ_ACCESS, RoutingControl
    ro = ReadOnlyNeo4jDriver("bolt://localhost:1", "u", "p", database="db1")
    inner = ro.client._driver
    seen: dict[str, Any] = {}

    async def _eq(query: str, **kwargs: Any) -> str:
        seen["eq"] = kwargs
        return "R"

    def _session(**kwargs: Any) -> Any:
        seen["session"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(inner, "execute_query", _eq)
    monkeypatch.setattr(inner, "session", _session)
    assert asyncio.run(ro.execute_query("MATCH (n) RETURN n")) == "R"
    assert seen["eq"]["routing_"] == RoutingControl.READ
    assert seen["eq"]["database_"] == "db1"
    ro.session()
    assert seen["session"] == {"database": "db1", "default_access_mode": READ_ACCESS}
    # graphiti's own transaction() opens its session through the same client.
    seen.clear()
    ro.client.session(database="db1")
    assert seen["session"]["default_access_mode"] == READ_ACCESS
    asyncio.run(ro.close())


def test_a_clone_stays_read_only():
    ro = ReadOnlyNeo4jDriver("bolt://localhost:1", "u", "p")
    other = ro.with_database("other")
    assert isinstance(other, ReadOnlyNeo4jDriver)
    with pytest.raises(ReadOnlyViolation):
        asyncio.run(other.execute_query("RETURN 1", routing_="w"))
    asyncio.run(ro.close())
