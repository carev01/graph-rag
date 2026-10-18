"""The round runner: chunk states (pending / complete / dropped / error), the grounding
and chunk ContextVars around each episode, previous episodes = the earlier sibling
chunks only, provenance of asks, self-match recording, and `status`.

Two levels of fake:

- `FakeGraphiti` stands in for graphiti itself; it asks the REAL observed
  `TeacherTransport` for one answer, then runs graphiti's REAL final save
  (`bulk_utils.add_nodes_and_edges_bulk`) on a `ReadOnlyNeo4jDriver`, which refuses it.
- `test_real_pipeline_*` run graphiti's real `add_episode`, built by the harness's own
  `build_teacher_graphiti`, over a `HarnessGraphDriver` whose network layer (`_execute`)
  is faked and an in-memory embedder: every prompt is the production prompt.

Nothing here touches a network: the Neo4j drivers point at 127.0.0.1:1 and are never
connected (the read-only layer refuses the only write before any I/O).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from graphiti_core.prompts import prompt_library
from graphiti_core.utils import bulk_utils
from neo4j import READ_ACCESS, EagerResult, RoutingControl
from neo4j.exceptions import Neo4jError

from graph_extract.config import ExtractSettings
from graph_extract.grounding import CURRENT_GROUNDING
from graph_extract.ontology import CHEAP_TIER_SALIENCE, EXTRACTION_INSTRUCTIONS
from graph_extract.usage import CURRENT_PROMPT_NAME
from scripts.distill import harness as h
from scripts.distill.readonly_driver import ReadOnlyNeo4jDriver, ReadOnlyViolation
from scripts.distill.sample import SampleArticle
from scripts.distill.teacher import AnswerStore, TeacherDropped, TeacherNeeded, TeacherTransport

VENDOR, PRODUCT, SOURCE = "Veeam", "Veeam Backup", "Veeam Backup User Guide"
REF = "2026-03-01T10:00:00+00:00"
UNREACHABLE = "bolt://127.0.0.1:1"


@pytest.fixture(autouse=True)
def _no_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GRAPHITI_TELEMETRY_ENABLED", "false")


def _base_settings() -> ExtractSettings:
    return ExtractSettings(_env_file=None, neo4j_uri=UNREACHABLE, neo4j_user="u",
                           neo4j_password="p", docext_base_url="http://x",
                           docext_read_key="k", cheap_llm_api_key="or-key",
                           embed_base_url="http://127.0.0.1:9")


def _content(k: int, article_id: str = "art-1") -> str:
    tag = "" if article_id == "art-1" else f" ({article_id})"
    return f"Veeam Backup protects workloads. Marker chunk-{k:02d} text{tag}."


def _article(article_id: str = "art-1") -> SampleArticle:
    return SampleArticle(article_id=article_id, vendor=VENDOR, product=PRODUCT,
                         source=SOURCE, split="train", stratum=f"{VENDOR} / {PRODUCT}")


def _uuid(article_id: str, k: int) -> str:
    return f"{article_id}-ep-{k:02d}"


REF_DT = datetime.fromisoformat(REF)
INGEST_T0 = datetime(2026, 4, 1, 9, tzinfo=timezone.utc)   # when art-1 was ingested
GROUP = "backup-docs"


def _graph_episodes(n_chunks: dict[str, int],
                    text_of: dict[str, str] | None = None) -> list[dict]:
    """Every :Episodic the fake graph holds. Each sampled article's chunks share the
    reference time and were created one minute apart, in chunk order; other
    articles' episodes surround them."""
    eps = []
    for a_i, (a, n) in enumerate(sorted(n_chunks.items())):
        for k in range(n):
            eps.append({"uuid": _uuid(a, k), "content": _content(k, (text_of or {}).get(a, a)),
                        "valid_at": REF_DT,
                        "created_at": INGEST_T0 + timedelta(hours=a_i, minutes=k),
                        "group_id": GROUP, "source": "text"})
    for i in range(12):            # older documents, ingested earlier
        eps.append({"uuid": f"other-old-{i:02d}", "content": f"Older document {i:02d}.",
                    "valid_at": REF_DT - timedelta(days=12 - i),
                    "created_at": INGEST_T0 - timedelta(days=1, minutes=i),
                    "group_id": GROUP, "source": "text"})
    eps += [
        {"uuid": "other-future", "content": "Future document.",
         "valid_at": REF_DT + timedelta(days=1), "created_at": INGEST_T0 - timedelta(days=2),
         "group_id": GROUP, "source": "text"},
        {"uuid": "other-later-created", "content": "Ingested after art-1.",
         "valid_at": REF_DT - timedelta(hours=1), "created_at": INGEST_T0 + timedelta(days=9),
         "group_id": GROUP, "source": "text"},
        {"uuid": "other-message", "content": "A message episode.",
         "valid_at": REF_DT - timedelta(hours=2), "created_at": INGEST_T0 - timedelta(days=3),
         "group_id": GROUP, "source": "message"},
        {"uuid": "other-group", "content": "Another group.",
         "valid_at": REF_DT - timedelta(hours=3), "created_at": INGEST_T0 - timedelta(days=3),
         "group_id": "elsewhere", "source": "text"},
    ]
    return eps


class FakeRead:
    """The separate read-only driver: episode rows, production's previous-episode
    query (evaluated over `_graph_episodes`), the footprint and production's
    document-context query (through a READ session, as IngestDriver runs it)."""

    def __init__(self, n_chunks: dict[str, int], *, names: frozenset[str] = frozenset(),
                 facts: frozenset[str] = frozenset(),
                 text_of: dict[str, str] | None = None) -> None:
        """`text_of` maps an article to the article whose chunk texts it repeats."""
        self.n_chunks = n_chunks
        self.text_of = text_of or {}
        self.episodes = _graph_episodes(n_chunks, self.text_of)
        self.names = names
        self.facts = facts
        self.session_modes: list[Any] = []
        self.previous_params: list[dict] = []

    async def execute_query(self, query: str, parameters_: dict | None = None,
                            routing_: Any = None) -> Any:
        assert routing_ == RoutingControl.READ
        params = parameters_ or {}
        if query == h.EPISODES_QUERY:
            a = params["a"]
            by_uuid = {e["uuid"]: e for e in self.episodes}
            rows = [{"title": "T", "source_url": f"https://docs/{a}",
                     "content_changed_at": REF, "content_changed_basis": "exact",
                     "last_updated_at": None, "chunk_index": k, "uuid": _uuid(a, k),
                     "name": f"{a}:{k}:abcdef12",
                     "content": _content(k, self.text_of.get(a, a)),
                     "valid_at": REF_DT, "created_at": by_uuid[_uuid(a, k)]["created_at"]}
                    for k in range(self.n_chunks.get(a, 0))]
            return SimpleNamespace(records=rows)
        if query == h.PREVIOUS_EPISODES_QUERY:
            self.previous_params.append(params)
            hits = [e for e in self.episodes
                    if e["valid_at"] <= params["reference_time"]
                    and e["group_id"] in params["group_ids"]
                    and e["source"] == params["source"]
                    and e["created_at"] <= params["created_at"]
                    and e["uuid"] != params["own"]
                    and e["uuid"] not in params["later_siblings"]]
            hits.sort(key=lambda e: e["uuid"])
            hits.sort(key=lambda e: (e["valid_at"], e["created_at"]), reverse=True)
            return SimpleNamespace(records=[{"uuid": e["uuid"]}
                                            for e in hits[:params["num_episodes"]]])
        if query == h.FOOTPRINT_QUERY:
            return SimpleNamespace(records=[{"names": list(self.names),
                                             "facts": list(self.facts)}])
        raise AssertionError(f"unexpected query {query[:60]}")

    def session(self, **config: Any) -> Any:
        self.session_modes.append(config.get("default_access_mode"))

        class _Result:
            async def single(self) -> dict:
                return {"vendor": VENDOR, "product": PRODUCT, "source": SOURCE,
                        "chain_count": 1}

        class _Session:
            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

            async def run(self, query: str, **params: Any) -> Any:
                assert "HAS_PRODUCT" in query and params["a"]
                return _Result()

        return _Session()


class FakeGraphiti:
    """graphiti stand-in: one teacher call, then graphiti's REAL final save on a
    read-only driver (or a scripted exception)."""

    def __init__(self, store: AnswerStore, log: h.RoundLog, *,
                 raise_instead: BaseException | None = None,
                 driver: Any = None) -> None:
        self.transport = h.observe_transport(TeacherTransport(store), log)
        self.calls: list[dict] = []
        self.raise_instead = raise_instead
        self.driver = driver

    async def add_episode(self, **kwargs: Any) -> Any:
        ctx = CURRENT_GROUNDING.get()
        self.calls.append({**kwargs, "grounding": ctx, "chunk": h.CURRENT_CHUNK.get()})
        if self.raise_instead is not None:
            raise self.raise_instead
        token = CURRENT_PROMPT_NAME.set("extract_nodes.extract_text")
        try:
            await self.transport.chat.completions.create(
                model="m", messages=[{"role": "user", "content": kwargs["episode_body"]}])
        finally:
            CURRENT_PROMPT_NAME.reset(token)
        await bulk_utils.add_nodes_and_edges_bulk(self.driver, [], [], [], [], None)
        raise AssertionError("the read-only final save did not refuse")


async def _round(graphiti: Any, store: AnswerStore, log: h.RoundLog, read: FakeRead,
                 articles: list[SampleArticle]) -> h.RoundSummary:
    settings = h.harness_settings(_base_settings(), store.root)
    return await h.run_round(articles, graphiti=graphiti, settings=settings, store=store,
                             log=log, read=read)


def _stores(tmp_path: Path) -> tuple[AnswerStore, h.RoundLog]:
    store = AnswerStore(tmp_path / "store")
    return store, h.RoundLog(store)


# --- the brief's round-trip ---------------------------------------------------------

async def test_pending_in_round_one_complete_in_round_two(tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, driver=ReadOnlyNeo4jDriver(UNREACHABLE, "u", "p"))
    read = FakeRead({"art-1": 1})

    s1 = await _round(g, store, log, read, [_article()])
    assert s1.round == 1 and s1.states["pending"] == 1 and s1.new_pending == 1
    (key, request), = store.pending()
    assert request["prompt_name"] == "extract_nodes.extract_text"
    assert store.askers(key) == ["art-1:0"]
    assert log.latest_states()["art-1:0"]["state"] == "pending"
    assert log.latest_states()["art-1:0"]["pending_keys"] == [key]

    store.put(key, json.dumps({"extracted_entities": []}), {"by": "test"})
    s2 = await _round(g, store, log, read, [_article()])
    assert s2.round == 2 and s2.states["complete"] == 1
    rec = log.latest_states()["art-1:0"]
    assert rec["state"] == "complete" and rec["replayed"] == 1 and rec["error"] is None
    assert store.askers(key) == ["art-1:0"]            # noted once, not per round

    s3 = await _round(g, store, log, read, [_article()])
    assert s3.skipped_final == 1 and not s3.states     # complete is final


async def test_identical_chunks_share_one_key_and_both_are_askers(tmp_path: Path) -> None:
    """Two chunks with the same text send the same prompt: ONE pending key, with both
    chunks recorded as its askers -- what the unqueue-unless-shared rule relies on."""
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, driver=ReadOnlyNeo4jDriver(UNREACHABLE, "u", "p"))
    read = FakeRead({"art-1": 2, "art-2": 1}, text_of={"art-2": "art-1"})
    await _round(g, store, log, read, [_article("art-1"), _article("art-2")])
    states = log.latest_states()
    (shared,) = states["art-2:0"]["pending_keys"]
    assert states["art-1:0"]["pending_keys"] == [shared]
    assert len(store.pending()) == 2                     # art-1:0/art-2:0 share one
    assert store.askers(shared) == ["art-1:0", "art-2:0"]
    # answered once, it serves both chunks
    store.put(shared, json.dumps({"extracted_entities": []}), {})
    await _round(g, store, log, read, [_article("art-1"), _article("art-2")])
    states = log.latest_states()
    assert states["art-1:0"]["state"] == states["art-2:0"]["state"] == "complete"
    assert store.askers(shared) == ["art-1:0", "art-2:0"]   # not re-noted on replay


async def test_dropped_chunk_keeps_a_prompt_another_chunk_asked(tmp_path: Path) -> None:
    """The shared-asker rule end to end: art-2:0 asks the same prompt as art-1:0 and
    then ends dropped; the shared prompt stays queued for art-1:0."""
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, driver=ReadOnlyNeo4jDriver(UNREACHABLE, "u", "p"))
    read = FakeRead({"art-1": 1, "art-2": 1}, text_of={"art-2": "art-1"})
    await _round(g, store, log, read, [_article("art-1")])
    (shared,) = [k for k, _ in store.pending()]

    class AsksThenDrops(FakeGraphiti):
        async def add_episode(self, **kwargs: Any) -> Any:
            try:
                await FakeGraphiti.add_episode(self, **kwargs)
            except TeacherNeeded:
                pass
            raise TeacherDropped("dead", "invalid twice")

    g2 = AsksThenDrops(store, log, driver=ReadOnlyNeo4jDriver(UNREACHABLE, "u", "p"))
    await _round(g2, store, log, read, [_article("art-2")])
    assert log.latest_states()["art-2:0"]["state"] == "dropped"
    assert store.askers(shared) == ["art-1:0", "art-2:0"]
    assert [k for k, _ in store.pending()] == [shared]


async def test_add_episode_arguments_match_production(tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, raise_instead=TeacherNeeded("k", {"prompt_name": "x"}))
    await _round(g, store, log, FakeRead({"art-1": 1}), [_article()])
    (call,) = g.calls
    assert call["episode_body"] == _content(0)
    assert call["name"] == "art-1:0:abcdef12"
    assert call["source_description"] == "https://docs/art-1"
    assert call["reference_time"] == datetime(2026, 3, 1, 10, tzinfo=timezone.utc)
    assert call["custom_extraction_instructions"] == (
        EXTRACTION_INSTRUCTIONS + CHEAP_TIER_SALIENCE
        + f'\n\nDOCUMENT CONTEXT: This chunk comes from {VENDOR} documentation for '
          f'{PRODUCT} (document set: {SOURCE}). The documented product is "{PRODUCT}".')
    assert log.latest_states()["art-1:0"]["ref_matches_episode"] is True


async def test_grounding_and_chunk_contextvars_set_and_reset(tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, raise_instead=TeacherNeeded("k", {"prompt_name": "x"}))
    read = FakeRead({"art-1": 2})
    assert CURRENT_GROUNDING.get() is None
    await _round(g, store, log, read, [_article()])
    assert CURRENT_GROUNDING.get() is None and h.CURRENT_CHUNK.get() is None
    assert [c["grounding"].text for c in g.calls] == [_content(0), _content(1)]
    for c in g.calls:
        assert c["grounding"].allowed == (PRODUCT, VENDOR)
        assert c["grounding"].product == PRODUCT
        assert c["grounding"].text == c["episode_body"]
    assert [c["chunk"].chunk_id for c in g.calls] == ["art-1:0", "art-1:1"]
    # the document context was read through a READ session
    assert read.session_modes in ([None], [READ_ACCESS])


async def test_unexpected_exception_marks_error_and_is_logged(
        tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, raise_instead=RuntimeError("boom"))
    with caplog.at_level(logging.ERROR, logger="distill.harness"):
        s = await _round(g, store, log, FakeRead({"art-1": 1}), [_article()])
    assert s.states["error"] == 1
    rec = log.latest_states()["art-1:0"]
    assert rec["state"] == "error" and "RuntimeError: boom" in rec["error"]
    assert any("art-1:0" in r.getMessage() and "boom" in r.getMessage()
               for r in caplog.records)
    assert any(r.exc_info for r in caplog.records)       # with its traceback
    # error is not final: the next round retries the chunk
    s2 = await _round(g, store, log, FakeRead({"art-1": 1}), [_article()])
    assert s2.states["error"] == 1 and s2.skipped_final == 0


async def test_read_only_refusal_outside_the_final_save_is_an_error(tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, raise_instead=ReadOnlyViolation("execute_write elsewhere"))
    await _round(g, store, log, FakeRead({"art-1": 1}), [_article()])
    assert log.latest_states()["art-1:0"]["state"] == "error"


def _access_mode_error() -> Neo4jError:
    return Neo4jError._hydrate_neo4j(code=h.ACCESS_MODE_CODE,
                                     message="Writing in read access mode not allowed.")


async def test_server_read_only_error_from_the_final_save_is_complete(tmp_path: Path) -> None:
    class ServerRefusesDriver(ReadOnlyNeo4jDriver):
        def session(self, database: str | None = None) -> Any:
            class _S:
                async def execute_write(self, *a: Any, **k: Any) -> Any:
                    raise _access_mode_error()

                async def close(self) -> None:
                    return None
            return _S()

    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, driver=ServerRefusesDriver(UNREACHABLE, "u", "p"))
    read = FakeRead({"art-1": 1})
    await _round(g, store, log, read, [_article()])
    (key, _), = store.pending()
    store.put(key, json.dumps({"extracted_entities": []}), {})
    await _round(g, store, log, read, [_article()])
    assert log.latest_states()["art-1:0"]["state"] == "complete"


async def test_server_read_only_error_elsewhere_is_an_error(tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, raise_instead=_access_mode_error())
    await _round(g, store, log, FakeRead({"art-1": 1}), [_article()])
    assert log.latest_states()["art-1:0"]["state"] == "error"


async def test_dropped_key_marks_dropped_and_is_final(tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, driver=ReadOnlyNeo4jDriver(UNREACHABLE, "u", "p"))
    read = FakeRead({"art-1": 1})
    await _round(g, store, log, read, [_article()])
    (key, _), = store.pending()
    store.drop(key, "invalid twice")
    await _round(g, store, log, read, [_article()])
    rec = log.latest_states()["art-1:0"]
    assert rec["state"] == "dropped" and rec["dropped_keys"] == [key]
    s3 = await _round(g, store, log, read, [_article()])
    assert s3.skipped_final == 1


def test_classify_swallowed_pending_is_not_complete() -> None:
    """graphiti swallows extract_timestamps failures, so the final save can be reached
    with a prompt still pending: the chunk must stay pending."""
    async def _raise() -> None:
        await bulk_utils.add_nodes_and_edges_bulk(_RaisingDriver(), [], [], [], [], None)

    try:
        asyncio.run(_raise())
    except ReadOnlyViolation as e:
        exc = e
    run = h.ChunkRun(chunk_id="a:0", episode_uuid="u", needed={"k": "p"})
    assert h.classify(exc, run) == ("pending", None)
    assert h.classify(exc, h.ChunkRun(chunk_id="a:0", episode_uuid="u")) == ("complete", None)
    state, error = h.classify(None, run)
    assert state == "error" and "returned normally" in (error or "")


class _RaisingDriver:
    def session(self) -> Any:
        class _S:
            async def execute_write(self, *a: Any, **k: Any) -> Any:
                raise ReadOnlyViolation("refused")

            async def close(self) -> None:
                return None
        return _S()


async def test_previous_episodes_mirror_production_retrieve_episodes(tmp_path: Path) -> None:
    """graphiti's own lookup over the group as it stood at original ingest: the 10
    most recent text episodes at or before the reference time, oldest first --
    never the chunk itself, its later siblings, or anything created after it."""
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, raise_instead=TeacherNeeded("k", {"prompt_name": "x"}))
    read = FakeRead({"art-1": 13, "art-0": 2})
    await _round(g, store, log, read, [_article()])
    assert len(g.calls) == 13
    olds = [f"other-old-{i:02d}" for i in range(12)]
    # art-0 was ingested an hour before art-1, at the same reference time
    art0 = [_uuid("art-0", 0), _uuid("art-0", 1)]
    for k, call in enumerate(g.calls):
        prev = call["previous_episode_uuids"]
        expected = (olds + art0 + [_uuid("art-1", j) for j in range(k)])[-10:]
        assert prev == expected, k
        assert _uuid("art-1", k) not in prev
        assert not {_uuid("art-1", j) for j in range(k + 1, 13)} & set(prev)
        assert not {"other-future", "other-later-created", "other-message",
                    "other-group"} & set(prev)
    # chunk 0 sees other articles' episodes
    assert g.calls[0]["previous_episode_uuids"] == olds[4:] + art0
    # the query was bounded by this chunk's own creation time and reference time
    p0 = read.previous_params[0]
    assert p0["own"] == _uuid("art-1", 0) and p0["num_episodes"] == 10
    assert p0["group_ids"] == [GROUP] and p0["source"] == "text"
    assert p0["later_siblings"] == [_uuid("art-1", j) for j in range(1, 13)]


async def test_reference_time_is_the_episode_valid_at(tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, raise_instead=TeacherNeeded("k", {"prompt_name": "x"}))
    read = FakeRead({"art-1": 1})
    later = REF_DT + timedelta(days=3)
    for e in read.episodes:
        if e["uuid"] == _uuid("art-1", 0):
            e["valid_at"] = later

    async def rows(query: str, parameters_: dict | None = None, routing_: Any = None) -> Any:
        r = await FakeRead.execute_query(read, query, parameters_, routing_)
        if query == h.EPISODES_QUERY:
            r.records[0]["valid_at"] = later
        return r

    read.execute_query = rows  # type: ignore[method-assign]
    await _round(g, store, log, read, [_article()])
    assert g.calls[0]["reference_time"] == later
    # the article's content_changed_at no longer agrees: recorded as a diagnostic
    assert log.latest_states()["art-1:0"]["ref_matches_episode"] is False


async def test_dropped_chunk_unqueues_its_own_pending_prompts(tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    store.add_pending("only-mine", {"prompt_name": "p"})
    store.add_pending("shared", {"prompt_name": "p"})
    store.note_asker("shared", "art-2:0")

    class DropsAfterAsking(FakeGraphiti):
        async def add_episode(self, **kwargs: Any) -> Any:
            run = h.CURRENT_CHUNK.get()
            assert run is not None
            log.asked(run, "only-mine", "p", [], needed=True)
            log.asked(run, "shared", "p", [], needed=True)
            raise TeacherDropped("dead", "invalid twice")

    g = DropsAfterAsking(store, log)
    await _round(g, store, log, FakeRead({"art-1": 1}), [_article()])
    assert log.latest_states()["art-1:0"]["state"] == "dropped"
    assert [k for k, _ in store.pending()] == ["shared"]


# --- self-matches ---------------------------------------------------------------------

def _dicts(messages: list[Any]) -> list[dict]:
    return [{"role": m.role, "content": m.content} for m in messages]


def test_dedup_candidates_parse_graphiti_prompts() -> None:
    nodes = prompt_library.dedupe_nodes.nodes({
        "extracted_nodes": [{"id": 0, "name": "Backup job", "entity_type": ["Entity"]}],
        "existing_nodes": [{"candidate_id": 0, "name": "Backup job", "entity_types": ["Entity"],
                            "summary": ""},
                           {"candidate_id": 1, "name": "Backup copy job",
                            "entity_types": ["Entity"], "summary": ""}],
        "episode_content": "x", "previous_episodes": [], "entity_types": [],
        "custom_instructions": ""} | {"entity_type_description": []})
    assert h.dedup_candidates("dedupe_nodes.nodes", _dicts(nodes)) == [
        "Backup job", "Backup copy job"]
    edges = prompt_library.dedupe_edges.resolve_edge({
        "existing_edges": [{"idx": 0, "fact": "Veeam Backup protects VMs"},
                           {"idx": 1, "fact": "It's a fact, with 'quotes'"}],
        "edge_invalidation_candidates": [{"idx": 2, "fact": "Old fact"}],
        "new_edge": "Veeam Backup protects VMs"})
    assert h.dedup_candidates("dedupe_edges.resolve_edge", _dicts(edges)) == [
        "Veeam Backup protects VMs", "It's a fact, with 'quotes'", "Old fact"]
    assert h.dedup_candidates("extract_edges.edge", _dicts(edges)) is None


def test_dedup_candidates_survive_a_moved_closing_tag() -> None:
    """The cache layout can move a prompt's static tail -- closing tag included --
    into the system message."""
    msgs = [{"role": "system", "content": "rules\n</EXISTING ENTITIES>\nmore rules"},
            {"role": "user", "content": '<EXISTING ENTITIES>\n[{"candidate_id": 0, '
                                        '"name": "Veeam Agent"}]\n'}]
    assert h.dedup_candidates("dedupe_nodes.nodes", msgs) == ["Veeam Agent"]


async def test_self_matches_recorded_once_and_reported(tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    run = h.ChunkRun(chunk_id="art-1:0", episode_uuid="ep", prod_names=frozenset({"Veeam Agent"}),
                     prod_facts=frozenset({"Veeam Backup protects VMs"}))
    node_msgs = [{"role": "user", "content": '<EXISTING ENTITIES>\n[{"candidate_id": 0, '
                  '"name": "Veeam Agent"}, {"candidate_id": 1, "name": "Other"}]\n'
                  '</EXISTING ENTITIES>'}]
    edge_msgs = [{"role": "user", "content":
                  "<EXISTING FACTS>\n[{'idx': 0, 'fact': 'Veeam Backup protects VMs'}]\n"
                  "</EXISTING FACTS>\n<FACT INVALIDATION CANDIDATES>\n[]\n"
                  "</FACT INVALIDATION CANDIDATES>"}]
    log.asked(run, "k-node", "dedupe_nodes.nodes", node_msgs, needed=True)
    log.asked(run, "k-node", "dedupe_nodes.nodes", node_msgs, needed=False)   # replay
    log.asked(run, "k-edge", "dedupe_edges.resolve_edge", edge_msgs, needed=False)
    recs = h._read_jsonl(log.selfmatch_path)
    assert [(r["key"], r["candidates"], r["self_matches"]) for r in recs] == [
        ("k-node", 2, 1), ("k-edge", 1, 1)]
    assert store.askers("k-node") == ["art-1:0"]
    # a new RoundLog (a later process) does not record the same (key, chunk) again
    h.RoundLog(store).asked(run, "k-node", "dedupe_nodes.nodes", node_msgs, needed=False)
    assert len(h._read_jsonl(log.selfmatch_path)) == 2
    text = h.status_text(store.root)
    assert "dedupe_nodes.nodes: 1 of 2 candidates (50.0%)" in text
    assert "dedupe_edges.resolve_edge: 1 of 1 candidates (100.0%)" in text


async def test_status_counts(tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, driver=ReadOnlyNeo4jDriver(UNREACHABLE, "u", "p"))
    read = FakeRead({"art-1": 3, "art-2": 1})
    arts = [_article("art-1"), _article("art-2")]
    await _round(g, store, log, read, arts)
    keys = [k for k, _ in store.pending()]
    assert len(keys) == 4
    store.put(keys[0], json.dumps({"extracted_entities": []}), {})
    store.drop(keys[1], "bad")
    text = h.status_text(store.root)
    assert "rounds run: 1" in text
    assert "chunks: 4 -- pending 4, complete 0, dropped 0, error 0" in text
    assert "pending prompts: 2" in text and "extract_nodes.extract_text: 2" in text
    assert "answered: 1" in text and "dropped: 1" in text
    assert h.main(["status", "--store", str(store.root)]) == 0


# --- the real graphiti pipeline --------------------------------------------------------

class FakeEmbedder:
    async def create(self, input_data: Any) -> list[float]:
        return [0.1] * 768

    async def create_batch(self, input_data_list: list[Any]) -> list[list[float]]:
        return [[0.1] * 768 for _ in input_data_list]


class FakeGraphDriver(h.HarnessGraphDriver):
    """`HarnessGraphDriver` with its network layer faked over the same episode table
    as `FakeRead`: the episode lookup returns the requested episodes in REVERSED
    order (the harness must restore it); every other read (vector/fulltext candidate
    searches) finds nothing."""

    def __init__(self, episodes: list[dict]) -> None:
        super().__init__(UNREACHABLE, "u", "p")
        self.episodes = {e["uuid"]: e for e in episodes}
        self.queries: list[str] = []
        self.index_builds = 0

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        self.index_builds += 1
        await super().build_indices_and_constraints(delete_existing)

    async def _execute(self, cypher_query_: Any, **kwargs: Any) -> Any:
        self.queries.append(str(cypher_query_))
        if h._is_episode_lookup(str(cypher_query_), kwargs):
            recs = [{"uuid": u, "name": u, "group_id": e["group_id"],
                     "created_at": e["created_at"], "source": e["source"],
                     "source_description": "d", "content": e["content"],
                     "valid_at": e["valid_at"], "entity_edges": []}
                    for u in kwargs["uuids"] for e in [self.episodes[u]]]
            return EagerResult(list(reversed(recs)), None, [])
        return EagerResult([], None, [])


ANSWERS = {
    "extract_nodes.extract_text": {"extracted_entities": [
        {"name": "Veeam Backup", "entity_type_id": 0}]},
    "extract_edges.edge": {"edges": []},
    "extract_nodes.extract_summaries_batch": {"summaries": []},
}


def _teacher_graphiti(settings: ExtractSettings, store: AnswerStore, log: h.RoundLog,
                      driver: FakeGraphDriver) -> Any:
    graphiti, _ = h.build_teacher_graphiti(settings, store, log, graph_driver=driver)
    graphiti.embedder = FakeEmbedder()
    graphiti.clients.embedder = graphiti.embedder
    return graphiti


async def test_real_pipeline_rounds_converge_with_production_previous_episodes(
        tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    settings = h.harness_settings(_base_settings(), store.root)
    n = 3
    read = FakeRead({"art-1": n})
    driver = FakeGraphDriver(read.episodes)
    graphiti = _teacher_graphiti(settings, store, log, driver)
    seen: dict[str, list[dict]] = {}            # prompt_name -> requests
    for _ in range(6):
        await h.run_round([_article()], graphiti=graphiti, settings=settings, store=store,
                          log=log, read=read)
        pending = store.pending()
        if not pending:
            break
        for key, req in pending:
            seen.setdefault(req["prompt_name"], []).append({"key": key, **req})
            store.put(key, json.dumps(ANSWERS[req["prompt_name"]]), {})
    states = {cid: r["state"] for cid, r in log.latest_states().items()}
    assert states == {f"art-1:{k}": "complete" for k in range(n)}, log.latest_states()
    assert set(seen) == set(ANSWERS)             # every prompt carried its type
    assert driver.index_builds == 0
    assert not any("CREATE" in q or "MERGE" in q for q in driver.queries)

    # The edge prompt of chunk k shows production's previous episodes -- older
    # documents, then chunks 0..k-1 -- oldest first, and chunk k only once (as the
    # current message).
    for req in seen["extract_edges.edge"]:
        (chunk_id,) = store.askers(req["key"])
        k = int(chunk_id.split(":")[1])
        text = "\n".join(m["content"] for m in req["messages"])
        for j in range(n):
            assert text.count(f"chunk-{j:02d}") == (1 if j <= k else 0), (k, j)
        markers = [f"Older document {i:02d}" for i in range(2 + k, 12)]
        markers += [f"chunk-{j:02d}" for j in range(k + 1)]
        positions = [text.find(m) for m in markers]
        assert -1 not in positions and positions == sorted(positions), k
        assert "Older document 01" not in text   # beyond the 10 most recent

    # Replayed answers were captured in the production capture format, with prompt type.
    capture = [json.loads(line) for line in
               (store.root / "capture.jsonl").read_text().splitlines()]
    assert capture and all(r["prompt_name"] in ANSWERS for r in capture)
    assert all(r["tier"] == "cheap" for r in capture)


async def test_real_pipeline_keys_do_not_depend_on_the_cache_layout(tmp_path: Path) -> None:
    """Production's cheap tier runs the cache layout, which learns per process: in a
    fresh round a prompt type's first 5 calls go out unmoved, later ones moved. With
    chunks progressing unevenly, the same prompt is sent moved in one round and
    unmoved in another -- and must keep ONE key."""
    store, log = _stores(tmp_path)
    settings = h.harness_settings(
        _base_settings().model_copy(update={"cheap_llm_cache_layout": True}), store.root)
    n = 9
    read = FakeRead({"art-1": n})
    driver = FakeGraphDriver(read.episodes)
    asked: dict[tuple[str, str], set[str]] = {}       # (chunk, prompt) -> keys
    answered: list[str] = []
    for round_i in range(30):
        graphiti = _teacher_graphiti(settings, store, log, driver)   # a fresh learner
        await h.run_round([_article()], graphiti=graphiti, settings=settings, store=store,
                          log=log, read=read)
        pending = store.pending()
        if not pending:
            break
        for key, req in pending:
            for chunk_id in store.askers(key):
                asked.setdefault((chunk_id, req["prompt_name"]), set()).add(key)
        # Round 1: answer everything (all 9 extract_text prompts; in round 2 chunks
        # 5-8 replay theirs as calls 6-9, i.e. moved). Then uneven progress: answer
        # only what the first 3 unfinished chunks asked, so finished chunks drop out
        # and a later round replays chunk 5's extract_text as an early call: unmoved.
        states = log.latest_states()
        unfinished = sorted(c for c, r in states.items() if r["state"] == "pending")
        if round_i:
            unfinished = unfinished[:3]
        for key, req in pending:
            if set(store.askers(key)) & set(unfinished):
                answered.append(key)
                store.put(key, json.dumps(ANSWERS[req["prompt_name"]]), {})
    states = {c: r["state"] for c, r in log.latest_states().items()}
    assert set(states.values()) == {"complete"}, states
    # one key per chunk and prompt type, never a re-ask
    assert all(len(keys) == 1 for keys in asked.values()), asked
    assert len(answered) == len(set(answered)) == n * len(ANSWERS)
    calls = h._read_jsonl(log.calls_path)
    layouts: dict[str, set[str]] = {}
    for c in calls:
        layouts.setdefault(c["key"], set()).add(c["layout"])
    both = [k for k, ls in layouts.items() if ls == {"moved", "unmoved"}]
    assert both, layouts                           # the layout really varied
    # capture records carry both layouts' wire messages, mapped to the same key
    wire_to_call = {c["wire_key"]: c for c in calls}
    capture = [json.loads(line) for line in
               (store.root / "capture.jsonl").read_text().splitlines()]
    by_key: dict[str, set[str]] = {}
    for r in capture:
        c = wire_to_call[store.key_for(r["messages"], r.get("response_format"))]
        by_key.setdefault(c["key"], set()).add(c["layout"])
    assert any(by_key.get(k) == {"moved", "unmoved"} for k in both)


def test_harness_settings_overrides(tmp_path: Path) -> None:
    base = _base_settings().model_copy(update={"ingest_same_pair_contradictions": True})
    s = h.harness_settings(base, tmp_path)
    assert s.llm_capture_path == str(tmp_path / "capture.jsonl")
    assert s.cheap_llm_model == "claude-opus-teacher"
    assert s.extraction_routing is True
    assert s.ingest_detect_contradictions is False
    assert s.ingest_same_pair_contradictions is False
    assert s.dedup_retry_on_strong is False


# --- batches / intake / review-sample / export (task 4) -------------------------------

def _write_sample_json(path: Path, articles: list[SampleArticle]) -> None:
    from dataclasses import asdict
    path.write_text(json.dumps({"articles": [asdict(a) for a in articles]}), encoding="utf-8")


def test_batches_groups_by_prompt_type_with_at_most_size_per_file(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "store")
    for i in range(5):
        store.add_pending(f"edge-{i}", {"prompt_name": "extract_edges.edge",
                                         "messages": [{"role": "user", "content": f"e{i}"}],
                                         "response_format": {}, "model": "m"})
    for i in range(3):
        store.add_pending(f"sum-{i}", {"prompt_name": "extract_nodes.extract_summaries_batch",
                                        "messages": [{"role": "user", "content": f"s{i}"}],
                                        "response_format": {}, "model": "m"})
    paths = h.write_batches(store.root, tmp_path / "batches", size=2)
    all_lines = [line for p in paths for line in h._read_jsonl(p)]
    assert len(all_lines) == 8
    assert {line["key"] for line in all_lines} == {f"edge-{i}" for i in range(5)} | {
        f"sum-{i}" for i in range(3)}
    for p in paths:
        lines = h._read_jsonl(p)
        assert len(lines) <= 2
        assert len({line["prompt_name"] for line in lines}) == 1
        for line in lines:
            assert "response_schema" in line
            assert "previous_answer" not in line and "rejection" not in line


def test_intake_accepts_requeues_once_then_drops_on_a_second_rejection(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = AnswerStore(tmp_path / "store")
    sample_path = tmp_path / "sample.json"
    _write_sample_json(sample_path, [_article("art-1")])
    request = {"prompt_name": "extract_edges.edge",
              "messages": [{"role": "system", "content": "s"},
                           {"role": "user", "content": "u"}],
              "response_format": {}, "model": "m"}
    for key in ("k1", "k2", "k3-unknown"):
        if key != "k3-unknown":
            store.add_pending(key, request)
            store.note_asker(key, "art-1:0")

    def fake_validate(prompt_name: str, messages: list, answer: dict, *,
                      grounding_ctx: Any = None) -> list[str]:
        return [] if answer.get("ok") else [f"bad:{answer}"]

    monkeypatch.setattr(h, "validate_answer", fake_validate)

    answers_dir = tmp_path / "round-1"
    answers_dir.mkdir()
    (answers_dir / "batch-000.answers.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in [
            {"key": "k1", "answer": {"ok": True}},
            {"key": "k2", "answer": {"ok": False, "n": 1}},
            {"key": "k3-unknown", "answer": {"ok": True}}]), encoding="utf-8")

    summary = h.run_intake(store.root, answers_dir, sample_path)
    assert (summary.accepted, summary.requeued, summary.dropped, summary.skipped) == (1, 1, 0, 1)
    store = AnswerStore(store.root)
    assert store.get("k1") is not None
    assert store.dropped_reason("k2") is None
    assert any(k for k, _ in store.pending() if k == "k2")
    rejections = h._rejections_by_key(store.root)
    assert len(rejections["k2"]) == 1

    answers_dir2 = tmp_path / "round-2"
    answers_dir2.mkdir()
    (answers_dir2 / "batch-000.answers.jsonl").write_text(
        json.dumps({"key": "k2", "answer": {"ok": False, "n": 2}}) + "\n", encoding="utf-8")
    summary2 = h.run_intake(store.root, answers_dir2, sample_path)
    assert (summary2.accepted, summary2.requeued, summary2.dropped) == (0, 0, 1)
    store = AnswerStore(store.root)
    assert store.dropped_reason("k2") is not None
    assert store.pending() == []


def test_export_prefers_the_moved_layout_and_excludes_dropped_keys(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "store")
    store.put("ka", json.dumps({"x": 1}), {"prompt_name": "extract_edges.edge"})
    store.drop("kb", "bad")

    msgs_unmoved = [{"role": "system", "content": "s"}, {"role": "user", "content": "u-unmoved"}]
    msgs_moved = [{"role": "system", "content": "s+tail"}, {"role": "user", "content": "u-moved"}]
    msgs_dropped = [{"role": "system", "content": "s"}, {"role": "user", "content": "u-dropped"}]
    wk_unmoved = store.key_for(msgs_unmoved, None)
    wk_moved = store.key_for(msgs_moved, None)
    wk_dropped = store.key_for(msgs_dropped, None)
    h._append_jsonl(store.root / "calls.jsonl", {
        "key": "ka", "wire_key": wk_unmoved, "layout": "unmoved",
        "prompt_name": "extract_edges.edge", "chunk_id": "art-1:0"})
    h._append_jsonl(store.root / "calls.jsonl", {
        "key": "ka", "wire_key": wk_moved, "layout": "moved",
        "prompt_name": "extract_edges.edge", "chunk_id": "art-1:0"})
    h._append_jsonl(store.root / "calls.jsonl", {
        "key": "kb", "wire_key": wk_dropped, "layout": "unmoved",
        "prompt_name": "extract_edges.edge", "chunk_id": "art-1:1"})
    h._append_jsonl(store.root / "capture.jsonl", {
        "call_id": "1", "messages": msgs_unmoved, "response_format": None,
        "prompt_name": "extract_edges.edge", "completion": "{}", "finish_reason": "stop"})
    h._append_jsonl(store.root / "capture.jsonl", {
        "call_id": "2", "messages": msgs_moved, "response_format": None,
        "prompt_name": "extract_edges.edge", "completion": "{}", "finish_reason": "stop"})
    h._append_jsonl(store.root / "capture.jsonl", {
        "call_id": "3", "messages": msgs_dropped, "response_format": None,
        "prompt_name": "extract_edges.edge", "completion": "{}", "finish_reason": "stop"})

    out_path = tmp_path / "export" / "capture.jsonl"
    summary = h.export_capture(store.root, out_path)
    assert summary.exported == 1 and summary.missing == 0
    assert summary.by_layout == {"moved": 1}
    assert summary.by_prompt == {"extract_edges.edge": 1}
    exported = h._read_jsonl(out_path)
    assert len(exported) == 1
    assert exported[0]["call_id"] == "2"
    assert exported[0]["messages"] == msgs_moved


def test_review_sample_is_deterministic(tmp_path: Path) -> None:
    store = AnswerStore(tmp_path / "store")
    for i in range(20):
        name = "extract_edges.edge" if i % 2 else "extract_nodes.extract_text"
        store.put(f"k{i}", json.dumps({"i": i}), {
            "prompt_name": name,
            "messages": [{"role": "system", "content": "s"},
                        {"role": "user", "content": f"u{i}"}]})
    path1 = h.write_review_sample(store.root, tmp_path / "r1", fraction=0.5, seed=20261017)
    path2 = h.write_review_sample(store.root, tmp_path / "r2", fraction=0.5, seed=20261017)
    lines1 = h._read_jsonl(path1)
    lines2 = h._read_jsonl(path2)
    assert lines1 == lines2
    assert lines1                               # some non-trivial fraction got sampled
    assert all({"key", "prompt_name", "messages", "answer"} <= set(line) for line in lines1)
