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
from datetime import datetime, timezone
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
from scripts.distill.teacher import AnswerStore, TeacherNeeded, TeacherTransport

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


def _content(k: int) -> str:
    return f"Veeam Backup protects workloads. Marker chunk-{k:02d} text."


def _article(article_id: str = "art-1", n_chunks: int = 3) -> SampleArticle:
    return SampleArticle(article_id=article_id, vendor=VENDOR, product=PRODUCT,
                         source=SOURCE, split="train", stratum=f"{VENDOR} / {PRODUCT}")


def _uuid(article_id: str, k: int) -> str:
    return f"{article_id}-ep-{k:02d}"


class FakeRead:
    """The separate read-only driver: episode rows, the footprint and production's
    document-context query (through a READ session, as IngestDriver runs it)."""

    def __init__(self, n_chunks: dict[str, int], *, names: frozenset[str] = frozenset(),
                 facts: frozenset[str] = frozenset()) -> None:
        self.n_chunks = n_chunks
        self.names = names
        self.facts = facts
        self.session_modes: list[Any] = []

    async def execute_query(self, query: str, parameters_: dict | None = None,
                            routing_: Any = None) -> Any:
        assert routing_ == RoutingControl.READ
        params = parameters_ or {}
        if query == h.EPISODES_QUERY:
            a = params["a"]
            # shuffled order: chunk order must come from chunk_index, not row order
            rows = [{"title": "T", "source_url": f"https://docs/{a}",
                     "content_changed_at": REF, "content_changed_basis": "exact",
                     "last_updated_at": None, "chunk_index": k, "uuid": _uuid(a, k),
                     "name": f"{a}:{k}:abcdef12", "content": _content(k), "valid_at": REF}
                    for k in range(self.n_chunks.get(a, 0))]
            return SimpleNamespace(records=rows)
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


async def test_previous_episodes_are_the_earlier_siblings_only(tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    g = FakeGraphiti(store, log, raise_instead=TeacherNeeded("k", {"prompt_name": "x"}))
    await _round(g, store, log, FakeRead({"art-1": 13}), [_article()])
    assert len(g.calls) == 13
    for k, call in enumerate(g.calls):
        prev = call["previous_episode_uuids"]
        assert prev == [_uuid("art-1", j) for j in range(max(0, k - 10), k)]
        assert _uuid("art-1", k) not in prev
    # chunk 0 passes [] -- None would let graphiti look up the group's last episodes
    assert g.calls[0]["previous_episode_uuids"] == []


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
    # art-2's only chunk has the same text as art-1's first: one shared prompt
    read = FakeRead({"art-1": 3, "art-2": 1})
    arts = [_article("art-1"), _article("art-2")]
    await _round(g, store, log, read, arts)
    keys = [k for k, _ in store.pending()]
    assert len(keys) == 3
    shared = log.latest_states()["art-2:0"]["pending_keys"][0]
    assert store.askers(shared) == ["art-1:0", "art-2:0"]
    others = [k for k in keys if k != shared]
    store.put(others[0], json.dumps({"extracted_entities": []}), {})
    store.drop(others[1], "bad")
    text = h.status_text(store.root)
    assert "rounds run: 1" in text
    assert "chunks: 4 -- pending 4, complete 0, dropped 0, error 0" in text
    assert "pending prompts: 1" in text and "extract_nodes.extract_text: 1" in text
    assert "answered: 1" in text and "dropped: 1" in text
    assert h.main(["status", "--store", str(store.root)]) == 0


# --- the real graphiti pipeline --------------------------------------------------------

class FakeEmbedder:
    async def create(self, input_data: Any) -> list[float]:
        return [0.1] * 768

    async def create_batch(self, input_data_list: list[Any]) -> list[list[float]]:
        return [[0.1] * 768 for _ in input_data_list]


class FakeGraphDriver(h.HarnessGraphDriver):
    """`HarnessGraphDriver` with its network layer faked: the episode lookup returns
    the requested episodes in REVERSED order (the harness must restore it); every
    other read (vector/fulltext candidate searches) finds nothing."""

    def __init__(self, contents: dict[str, str]) -> None:
        super().__init__(UNREACHABLE, "u", "p")
        self.contents = contents
        self.queries: list[str] = []
        self.index_builds = 0

    async def build_indices_and_constraints(self, delete_existing: bool = False) -> None:
        self.index_builds += 1
        await super().build_indices_and_constraints(delete_existing)

    async def _execute(self, cypher_query_: Any, **kwargs: Any) -> Any:
        self.queries.append(str(cypher_query_))
        if h._is_episode_lookup(str(cypher_query_), kwargs):
            when = datetime.fromisoformat(REF)
            recs = [{"uuid": u, "name": u, "group_id": "backup-docs", "created_at": when,
                     "source": "text", "source_description": "d", "content": self.contents[u],
                     "valid_at": when, "entity_edges": []} for u in kwargs["uuids"]]
            return EagerResult(list(reversed(recs)), None, [])
        return EagerResult([], None, [])


ANSWERS = {
    "extract_nodes.extract_text": {"extracted_entities": [
        {"name": "Veeam Backup", "entity_type_id": 0}]},
    "extract_edges.edge": {"edges": []},
    "extract_nodes.extract_summaries_batch": {"summaries": []},
}


async def test_real_pipeline_rounds_converge_with_sibling_previous_episodes(
        tmp_path: Path) -> None:
    store, log = _stores(tmp_path)
    settings = h.harness_settings(_base_settings(), store.root)
    n = 3
    driver = FakeGraphDriver({_uuid("art-1", k): _content(k) for k in range(n)})
    graphiti, _ = h.build_teacher_graphiti(settings, store, log, graph_driver=driver)
    graphiti.embedder = FakeEmbedder()          # type: ignore[assignment]
    graphiti.clients.embedder = graphiti.embedder
    read = FakeRead({"art-1": n})
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

    # The edge prompt of chunk k shows chunks 0..k-1 as previous episodes, oldest
    # first, and chunk k only once (as the current message).
    for req in seen["extract_edges.edge"]:
        (chunk_id,) = store.askers(req["key"])
        k = int(chunk_id.split(":")[1])
        text = "\n".join(m["content"] for m in req["messages"])
        for j in range(n):
            assert text.count(f"chunk-{j:02d}") == (1 if j <= k else 0), (k, j)
        positions = [text.find(f"chunk-{j:02d}") for j in range(k + 1)]
        assert positions == sorted(positions)

    # Replayed answers were captured in the production capture format, with prompt type.
    capture = [json.loads(line) for line in
               (store.root / "capture.jsonl").read_text().splitlines()]
    assert capture and all(r["prompt_name"] in ANSWERS for r in capture)
    assert all(r["tier"] == "cheap" for r in capture)


def test_harness_settings_overrides(tmp_path: Path) -> None:
    base = _base_settings().model_copy(update={"ingest_same_pair_contradictions": True})
    s = h.harness_settings(base, tmp_path)
    assert s.llm_capture_path == str(tmp_path / "capture.jsonl")
    assert s.cheap_llm_model == "claude-opus-teacher"
    assert s.extraction_routing is True
    assert s.ingest_detect_contradictions is False
    assert s.ingest_same_pair_contradictions is False
    assert s.dedup_retry_on_strong is False
