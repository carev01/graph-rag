"""The Opus-distillation harness: `select`, `round`, `status`.

    uv run --extra dev python -m scripts.distill.harness select --out DIR/sample.json
    uv run --extra dev python -m scripts.distill.harness round \
        --sample DIR/sample.json --store DIR/store [--articles N]
    uv run --extra dev python -m scripts.distill.harness status --store DIR/store

**`round`** runs graphiti's REAL `add_episode` on each sampled article's production
chunks (`Episodic.content`, in `HAS_EPISODE.chunk_index` order), with the production
cheap-tier client and every production wrapper, and two substitutions:

- the LLM transport is `TeacherTransport` (stored teacher answer, or `TeacherNeeded`);
- the graph driver is `HarnessGraphDriver`, a `ReadOnlyNeo4jDriver` -- the ONLY Neo4j
  connection graphiti gets. Sample selection, episode reads and the self-match
  footprint use a SEPARATE read-only driver (`_ReadOnlyAsyncDriver`: READ routing on
  every query, READ access mode on every session).

No `init_indices`, no `ensure_vector_indexes`: the vector-search routing is the one
`build_cheap_graphiti` installs (`install_vector_search`), which only reads the
existing tuned indexes. `install_dedup_guard` is installed as production installs it
on the cheap tier with retries off (`fallback=None`) and contradictions suppressed;
it is also what publishes `CURRENT_PROMPT_NAME`, so pending requests and capture
records carry their prompt type.

Each chunk ends in one state, appended to `<store>/chunks.jsonl` (latest record wins):

- `pending`  -- a prompt has no teacher answer yet (`TeacherNeeded`); recorded, and the
  round moves on to the next chunk;
- `complete` -- every prompt was answered and graphiti reached its final bulk save,
  which the read-only driver refuses (`ReadOnlyViolation`, or the server's
  `Neo.ClientError.Statement.AccessMode`) -- accepted ONLY when raised from inside
  `graphiti_core.utils.bulk_utils.add_nodes_and_edges_bulk`;
- `dropped`  -- a prompt's answer was dropped by intake (`TeacherDropped`);
- `error`    -- anything else, logged with the chunk id and its traceback.

`complete` and `dropped` are final: later rounds skip those chunks.

**Previous episodes** are what production's graphiti saw. Production passes no
`previous_episode_uuids`, so graphiti ran `retrieve_episodes` over the whole group:
the `PREVIOUS_EPISODES_MAX` most recent text episodes with `valid_at <=
reference_time`, `ORDER BY valid_at DESC`, then reversed (chronological). The live
graph has moved on since, so `PREVIOUS_EPISODES_QUERY` mirrors that query (read
only) EXCLUDING this chunk's own production episode, its later siblings, and every
episode created after this chunk's production episode (the graph as it stood at
original ingest -- an upper-bound approximation under concurrent workers, see the
query's comment), with a deterministic tie-break (`valid_at DESC, created_at DESC,
uuid`). The result goes to graphiti as `previous_episode_uuids`, oldest first.
graphiti fetches them with `EpisodicNode.get_by_uuids`, whose query has no ORDER BY;
`HarnessGraphDriver` returns that query's rows in the order of the uuids passed, so
the prompt is the same bytes every round.

**Reference time** is the chunk's production `Episodic.valid_at` -- the time
production's graphiti actually used. `ingest_driver._reference_time` over the
article's stored fields is computed as a diagnostic only (`ref_matches_episode`).

**Answer keys ignore the wire layout.** Production runs the cheap tier with the cache
layout on, whose `StaticTailLearner` learns per process: a fresh round sends a prompt
type's first calls unmoved and later ones with the static tail moved into the system
message. Keys are therefore computed on the PRE-layout request (`teacher.
install_pre_layout_keys`, the outermost wrapper), so one prompt has one key in every
round. Capture still records the wire messages; `<store>/calls.jsonl` maps each
answered call's wire request (`wire_key` = `AnswerStore.key_for` over the captured
messages and response_format) to its `key` and `layout` ("moved" when the system
message differs from graphiti's, else "unmoved") for export.

**Self-matches** (the accepted caveat). Dedup candidates come from the live graph,
which already holds what production extracted from this very chunk. For every
`dedupe_nodes.nodes` / `dedupe_edges.resolve_edge` prompt a chunk asks (pending or
replayed, once per key and chunk), the candidates are parsed out of the prompt and
compared with the chunk's production footprint: the names of the entities its
production episode MENTIONS, and the facts that cite it (`RELATES_TO.episodes`).
Exact string matches are self-matches. Recorded in `<store>/selfmatch.jsonl`;
`status` prints the share.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import json
import logging
import os
import sys
import traceback
from collections import Counter
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from graphiti_core import Graphiti
from neo4j import AsyncGraphDatabase, EagerResult, RoutingControl
from neo4j.exceptions import Neo4jError

from graph_extract.config import ExtractSettings, get_extract_settings
from graph_extract.dedup_guard import DEDUP_PROMPT_NAME, DedupIndexStats, install_dedup_guard
from graph_extract.graphiti_client import add_text_episode, build_cheap_graphiti, document_context
from graph_extract.grounding import CURRENT_GROUNDING, GroundingContext, GroundingStats
from graph_extract.ingest_driver import IngestDriver, _reference_time
from graph_extract.llm_timing import PromptTimings
from graph_extract.ontology import CHEAP_TIER_SALIENCE, EXTRACTION_INSTRUCTIONS
from graph_extract.usage import CURRENT_PROMPT_NAME
from scripts.distill.readonly_driver import (
    ReadOnlyNeo4jDriver,
    ReadOnlyViolation,
    _ReadOnlyAsyncDriver,
)
from scripts.distill.sample import (
    DEFAULT_VENDORS,
    RoutingThresholds,
    SampleArticle,
    load_sample,
    select_sample_with_counts,
    write_sample,
)
from scripts.distill.teacher import (
    PRE_LAYOUT_REQUEST,
    AnswerStore,
    TeacherDropped,
    TeacherNeeded,
    TeacherTransport,
    install_pre_layout_keys,
)

logger = logging.getLogger("distill.harness")

READ = RoutingControl.READ
PREVIOUS_EPISODES_MAX = 10          # graphiti's RELEVANT_SCHEMA_LIMIT, what production saw
ACCESS_MODE_CODE = "Neo.ClientError.Statement.AccessMode"
FINAL_SAVE_MODULE = "graphiti_core.utils.bulk_utils"
FINAL_SAVE_FUNCTION = "add_nodes_and_edges_bulk"
NODE_DEDUP_PROMPT_NAME = "dedupe_nodes.nodes"
SELF_MATCH_PROMPTS = (NODE_DEDUP_PROMPT_NAME, DEDUP_PROMPT_NAME)
STATES = ("pending", "complete", "dropped", "error")
FINAL_STATES = frozenset({"complete", "dropped"})
# How long stray graphiti tasks (siblings of the call that raised, still running
# inside an asyncio.gather) may take to finish before they are cancelled.
DRAIN_TIMEOUT_SECONDS = 60.0

EPISODES_QUERY = (
    "MATCH (a:Article {id: $a})-[he:HAS_EPISODE]->(e:Episodic) "
    "WHERE coalesce(he.superseded, false) = false AND coalesce(e.removed, false) = false "
    "RETURN a.title AS title, a.source_url AS source_url, "
    "       a.content_changed_at AS content_changed_at, "
    "       a.content_changed_basis AS content_changed_basis, "
    "       a.last_updated_at AS last_updated_at, "
    "       he.chunk_index AS chunk_index, e.uuid AS uuid, e.name AS name, "
    "       e.content AS content, e.valid_at AS valid_at, e.created_at AS created_at "
    "ORDER BY chunk_index, uuid")

# graphiti's retrieve_episodes (graph_data_operations.py, the non-saga branch, as
# add_episode calls it: last_n=RELEVANT_SCHEMA_LIMIT, group_ids=[group_id],
# source=text), minus this chunk's own episode, its later siblings and anything
# created after it, with a deterministic tie-break. Reversed by the caller, as
# graphiti reverses it.
#
# The created_at cut-off is an APPROXIMATION (an upper bound on what production
# saw). graphiti stamps an episode's created_at when add_episode STARTS but saves
# the episode only at its END (the final bulk save), and no saved-at timestamp
# exists. Under concurrent production workers, an episode created just before this
# chunk's -- by another worker, still mid-extraction -- was not yet in the graph
# when this chunk's retrieve_episodes ran, yet passes `created_at <= $created_at`
# here. Sequential ingest (one article at a time) has no such window.
PREVIOUS_EPISODES_QUERY = (
    "MATCH (e:Episodic) "
    "WHERE e.valid_at <= $reference_time "
    "AND e.group_id IN $group_ids "
    "AND e.source = $source "
    "AND e.created_at <= $created_at "
    "AND e.uuid <> $own "
    "AND NOT e.uuid IN $later_siblings "
    "RETURN e.uuid AS uuid "
    "ORDER BY e.valid_at DESC, e.created_at DESC, e.uuid "
    "LIMIT $num_episodes")

# What production extracted from one episode: the entities it MENTIONS and the facts
# that cite it (through MENTIONS, so the fact edges are not scanned one by one).
FOOTPRINT_QUERY = (
    "MATCH (e:Episodic {uuid: $u}) "
    "OPTIONAL MATCH (e)-[:MENTIONS]->(n:Entity) "
    "WITH e, collect(DISTINCT n.name) AS names "
    "OPTIONAL MATCH (e)-[:MENTIONS]->(:Entity)-[f:RELATES_TO]->(:Entity) "
    "WHERE e.uuid IN f.episodes "
    "RETURN names, collect(DISTINCT f.fact) AS facts")


# --- small jsonl helpers (same crash-safety contract as teacher.AnswerStore) ---------

def _append_jsonl(path: Path, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _read_jsonl(path: Path) -> list[dict]:
    """Tolerates a malformed LAST line (a kill mid-append), raises on any other."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if i == len(lines) - 1:
                logger.warning("skipping malformed last line of %s:%d: %s", path, i + 1, exc)
                continue
            raise ValueError(f"malformed line in {path}:{i + 1}") from exc
    return out


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- the graph driver graphiti gets --------------------------------------------------

def _is_episode_lookup(query: str, kwargs: dict[str, Any]) -> bool:
    """`EpisodicNode.get_by_uuids`'s query (graphiti 0.30.1, nodes.py)."""
    return ("MATCH (e:Episodic)" in query and "e.uuid IN $uuids" in query
            and isinstance(kwargs.get("uuids"), list))


class HarnessGraphDriver(ReadOnlyNeo4jDriver):
    """The read-only driver, plus a stable order for graphiti's episode lookup: its
    rows come back in the order of the `uuids` passed (oldest chunk first)."""

    async def execute_query(self, cypher_query_: Any, **kwargs: Any) -> Any:
        result = await self._execute(cypher_query_, **kwargs)
        if not _is_episode_lookup(str(cypher_query_), kwargs):
            return result
        position = {u: i for i, u in enumerate(kwargs["uuids"])}
        records, summary, keys = result
        ordered = sorted(records, key=lambda r: position.get(r["uuid"], len(position)))
        return EagerResult(ordered, summary, keys)

    async def _execute(self, cypher_query_: Any, **kwargs: Any) -> Any:
        return await super().execute_query(cypher_query_, **kwargs)


# --- per-chunk bookkeeping -----------------------------------------------------------

@dataclass
class ChunkRun:
    chunk_id: str
    episode_uuid: str
    prod_names: frozenset[str] = frozenset()
    prod_facts: frozenset[str] = frozenset()
    needed: dict[str, str] = field(default_factory=dict)      # key -> prompt_name
    dropped: dict[str, str] = field(default_factory=dict)     # key -> reason
    replayed: int = 0


# The chunk whose add_episode is in flight. A ContextVar so graphiti's inner tasks
# (asyncio.gather children) inherit it and the transport can attribute their calls.
CURRENT_CHUNK: ContextVar[ChunkRun | None] = ContextVar("distill_current_chunk", default=None)


def _message_texts(messages: Sequence[Any]) -> list[str]:
    out = []
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if isinstance(content, str):
            out.append(content)
    return out


def _tag_segments(messages: Sequence[Any], open_tag: str, close_tag: str) -> list[str]:
    """Each message's text from `open_tag` to `close_tag` (or the message's end:
    the cache layout may have moved the closing tag into another message)."""
    segs = []
    for text in _message_texts(messages):
        start = text.find(open_tag)
        if start < 0:
            continue
        start += len(open_tag)
        end = text.find(close_tag, start)
        segs.append(text[start:end if end >= 0 else len(text)])
    return segs


def _literal_list(segment: str, *, python_repr: bool) -> list | None:
    seg = segment.strip()
    if not seg.startswith("["):
        return None
    try:
        if python_repr:
            seg = seg[:seg.rfind("]") + 1]
            value = ast.literal_eval(seg)
        else:
            value, _ = json.JSONDecoder().raw_decode(seg)
    except (ValueError, SyntaxError):
        return None
    return value if isinstance(value, list) else None


def dedup_candidates(prompt_name: str, messages: Sequence[Any]) -> list[str] | None:
    """The candidate names (node dedup) or facts (edge dedup) a dedup prompt shows,
    or None when the prompt cannot be parsed. Coupled to graphiti 0.30.1's prompt
    text: `<EXISTING ENTITIES>` is JSON (`to_prompt_json`), the edge lists are Python
    reprs of `[{'idx': i, 'fact': ...}]`."""
    if prompt_name == NODE_DEDUP_PROMPT_NAME:
        segs = _tag_segments(messages, "<EXISTING ENTITIES>", "</EXISTING ENTITIES>")
        if not segs:
            return None
        items = _literal_list(segs[0], python_repr=False)
        if items is None:
            return None
        return [str(c.get("name")) for c in items if isinstance(c, dict)]
    if prompt_name == DEDUP_PROMPT_NAME:
        facts: list[str] = []
        found = False
        for tags in (("<EXISTING FACTS>", "</EXISTING FACTS>"),
                     ("<FACT INVALIDATION CANDIDATES>", "</FACT INVALIDATION CANDIDATES>")):
            segs = _tag_segments(messages, *tags)
            if not segs:
                continue
            items = _literal_list(segs[0], python_repr=True)
            if items is None:
                return None
            found = True
            facts.extend(str(c.get("fact")) for c in items if isinstance(c, dict))
        return facts if found else None
    return None


class RoundLog:
    """Everything the transport observer and the round runner record on disk."""

    def __init__(self, store: AnswerStore) -> None:
        self.store = store
        self.chunks_path = store.root / "chunks.jsonl"
        self.selfmatch_path = store.root / "selfmatch.jsonl"
        self.calls_path = store.root / "calls.jsonl"
        self._selfmatch_seen = {(r["key"], r["chunk_id"])
                                for r in _read_jsonl(self.selfmatch_path)}
        self._calls_seen = {(r["key"], r["wire_key"]) for r in _read_jsonl(self.calls_path)}

    def answered_call(self, key: str, wire_kwargs: dict, chunk_id: str | None) -> None:
        """Map an answered call's WIRE request (what capture records) to its key and
        layout, once per distinct (key, wire request)."""
        wire_messages = wire_kwargs.get("messages") or []
        wire_key = self.store.key_for(wire_messages, wire_kwargs.get("response_format"))
        if (key, wire_key) in self._calls_seen:
            return
        self._calls_seen.add((key, wire_key))
        _append_jsonl(self.calls_path, {
            "key": key, "wire_key": wire_key, "layout": wire_layout(wire_messages),
            "prompt_name": CURRENT_PROMPT_NAME.get(), "chunk_id": chunk_id,
            "ts": _utcnow()})

    def asked(self, run: ChunkRun, key: str, prompt_name: str, messages: Sequence[Any], *,
              needed: bool) -> None:
        # Provenance of a shared answer: every chunk that asked for this key, once.
        if run.chunk_id not in self.store.askers(key):
            self.store.note_asker(key, run.chunk_id)
        if needed:
            run.needed[key] = prompt_name
        else:
            run.replayed += 1
        if prompt_name in SELF_MATCH_PROMPTS and (key, run.chunk_id) not in self._selfmatch_seen:
            self._record_self_match(run, key, prompt_name, messages)

    def _record_self_match(self, run: ChunkRun, key: str, prompt_name: str,
                           messages: Sequence[Any]) -> None:
        cands = dedup_candidates(prompt_name, messages)
        footprint = run.prod_names if prompt_name == NODE_DEDUP_PROMPT_NAME else run.prod_facts
        record = {"key": key, "chunk_id": run.chunk_id, "prompt_name": prompt_name,
                  "parsed": cands is not None,
                  "candidates": len(cands) if cands is not None else 0,
                  "self_matches": sum(1 for c in cands if c in footprint) if cands else 0,
                  "ts": _utcnow()}
        _append_jsonl(self.selfmatch_path, record)
        self._selfmatch_seen.add((key, run.chunk_id))

    def chunk_done(self, record: dict) -> None:
        _append_jsonl(self.chunks_path, record)

    def latest_states(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for rec in _read_jsonl(self.chunks_path):
            out[rec["chunk_id"]] = rec
        return out

    def next_round(self) -> int:
        return max((int(r.get("round", 0)) for r in _read_jsonl(self.chunks_path)),
                   default=0) + 1


def _system_text(messages: Sequence[Any]) -> str | None:
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role == "system":
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            return content if isinstance(content, str) else None
    return None


def wire_layout(wire_messages: Sequence[Any]) -> str:
    """"moved" when the cache layout rewrote the system message of this call (the
    static tail went into it), "unmoved" when the wire system message is graphiti's,
    "unknown" outside `install_pre_layout_keys`."""
    pre = PRE_LAYOUT_REQUEST.get()
    if pre is None:
        return "unknown"
    return "unmoved" if _system_text(wire_messages) == _system_text(pre["messages"]) else "moved"


def observe_transport(transport: TeacherTransport, log: RoundLog) -> TeacherTransport:
    """Wrap the transport's `create` (innermost, before any production wrapper is
    applied on top) so every prompt a chunk asks -- answered or not -- is attributed
    to the chunk in flight."""
    orig = transport.chat.completions.create

    async def create(**kwargs: Any) -> Any:
        run = CURRENT_CHUNK.get()
        messages = kwargs.get("messages") or []
        try:
            response = await orig(**kwargs)
        except TeacherNeeded as exc:
            if run is not None:
                log.asked(run, exc.key, str(exc.request.get("prompt_name")), messages,
                          needed=True)
            raise
        except TeacherDropped as exc:
            if run is not None:
                run.dropped[exc.key] = exc.reason
            raise
        key = transport.request_key(kwargs)
        if run is not None:
            log.asked(run, key, CURRENT_PROMPT_NAME.get(), messages, needed=False)
        log.answered_call(key, kwargs, run.chunk_id if run is not None else None)
        return response

    transport.chat.completions.create = create
    return transport


# --- the stack -----------------------------------------------------------------------

def harness_settings(base: ExtractSettings, store_root: Path) -> ExtractSettings:
    """Production settings with the harness's overrides: capture into the store, the
    teacher's model label on the cheap tier, the cheap tier only (no dense-matrix
    router: only the cheap graphiti is built), contradictions as production runs them
    (detection off, suppression on), no strong-tier dedup retry."""
    for name, want in (("ingest_detect_contradictions", False),
                       ("ingest_same_pair_contradictions", False),
                       ("dedup_retry_on_strong", False)):
        if getattr(base, name) != want:
            logger.warning("overriding %s=%s from the environment to %s (production's "
                           "documented value)", name, getattr(base, name), want)
    return base.model_copy(update=dict(
        llm_capture_path=str(Path(store_root) / "capture.jsonl"),
        cheap_llm_model=TeacherTransport.MODEL,
        extraction_routing=True,
        ingest_detect_contradictions=False,
        ingest_same_pair_contradictions=False,
        dedup_retry_on_strong=False))


def build_teacher_graphiti(settings: ExtractSettings, store: AnswerStore, log: RoundLog, *,
                           graph_driver: ReadOnlyNeo4jDriver | None = None,
                           ) -> tuple[Graphiti, TeacherTransport]:
    """The production cheap graphiti over the teacher transport and a read-only
    driver, with production's dedup guard. Never builds indices."""
    transport = observe_transport(TeacherTransport(store), log)
    if graph_driver is None:
        graph_driver = HarnessGraphDriver(settings.neo4j_uri, settings.neo4j_user,
                                          settings.neo4j_password)
    graphiti = build_cheap_graphiti(settings, transport=transport, graph_driver=graph_driver)
    # Every production wrapper re-assigns create ON the transport object; the LLM
    # client's raw client must be that same object for the next wrapper to be the
    # outermost one.
    if graphiti.llm_client.client is not transport:  # type: ignore[attr-defined]
        raise RuntimeError("the cheap LLM client does not wrap the injected transport")
    install_pre_layout_keys(transport)
    install_dedup_guard(graphiti, fallback=None, unscoped=DedupIndexStats(),
                        timings=PromptTimings(), suppress_contradictions=True,
                        grounding_unscoped=GroundingStats())
    return graphiti, transport


# --- reads (separate read-only driver) ------------------------------------------------

@dataclass
class Chunk:
    chunk_index: int
    uuid: str
    name: str
    content: str
    valid_at: Any
    created_at: Any = None


@dataclass
class ArticleChunks:
    article_id: str
    title: str
    source_url: str
    fields: dict[str, Any]
    chunks: list[Chunk]


def _iso(value: Any) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if hasattr(value, "to_native"):           # neo4j.time.DateTime
        value = value.to_native()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


async def read_article_chunks(read: Any, article_id: str) -> ArticleChunks | None:
    result = await read.execute_query(EPISODES_QUERY, parameters_={"a": article_id},
                                      routing_=READ)
    recs = list(result.records)
    if not recs:
        return None
    first = recs[0]
    return ArticleChunks(
        article_id=article_id, title=first["title"] or "",
        source_url=first["source_url"] or "",
        fields={k: first[k] for k in ("content_changed_at", "content_changed_basis",
                                      "last_updated_at")},
        chunks=[Chunk(chunk_index=int(r["chunk_index"]), uuid=r["uuid"], name=r["name"],
                      content=r["content"], valid_at=r["valid_at"],
                      created_at=r["created_at"]) for r in recs])


async def read_document_context(read: Any, article_id: str) -> tuple[str, str, str]:
    """Production's own (vendor, product, source) rule, `IngestDriver._document_context`,
    run over the read-only driver (its sessions are READ access mode)."""
    return await IngestDriver._document_context(
        SimpleNamespace(_driver=read), article_id)  # type: ignore[arg-type]


async def read_footprint(read: Any, episode_uuid: str) -> tuple[frozenset[str], frozenset[str]]:
    result = await read.execute_query(FOOTPRINT_QUERY, parameters_={"u": episode_uuid},
                                      routing_=READ)
    recs = list(result.records)
    if not recs:
        return frozenset(), frozenset()
    return (frozenset(n for n in recs[0]["names"] if n),
            frozenset(f for f in recs[0]["facts"] if f))


def _as_datetime(value: Any) -> datetime:
    if hasattr(value, "to_native"):           # neo4j.time.DateTime
        value = value.to_native()
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    raise ValueError(f"not a datetime: {value!r}")


def reference_time(chunk: Chunk) -> datetime:
    """The chunk's production `Episodic.valid_at`: the reference time production's
    graphiti used for this episode."""
    if chunk.valid_at is None:
        raise ValueError(f"episode {chunk.uuid} has no valid_at")
    return _as_datetime(chunk.valid_at)


def article_reference_time(art: ArticleChunks) -> datetime | None:
    """`ingest_driver._reference_time` over the article's stored fields -- a
    diagnostic only (does production's valid_at still agree with the article?).
    None when it would fall back to crawl time or the wall clock."""
    if not art.fields.get("content_changed_at"):
        return None
    ref, _basis = _reference_time(SimpleNamespace(
        id=art.article_id,
        content_changed_at=_iso(art.fields.get("content_changed_at")),
        content_changed_basis=art.fields.get("content_changed_basis"),
        last_updated_at=_iso(art.fields.get("last_updated_at")),
        extracted_at=None))
    return ref


async def read_previous_episodes(read: Any, settings: ExtractSettings, art: ArticleChunks,
                                 chunk: Chunk) -> list[str]:
    """Production's previous episodes for `chunk`, oldest first (module docstring)."""
    later = [c.uuid for c in art.chunks if c.chunk_index > chunk.chunk_index]
    if chunk.created_at is None:
        raise ValueError(f"episode {chunk.uuid} has no created_at")
    result = await read.execute_query(PREVIOUS_EPISODES_QUERY, parameters_={
        "reference_time": chunk.valid_at, "group_ids": [settings.group_id],
        "source": "text", "created_at": chunk.created_at, "own": chunk.uuid,
        "later_siblings": later, "num_episodes": PREVIOUS_EPISODES_MAX}, routing_=READ)
    return list(reversed([r["uuid"] for r in result.records]))


def _same_instant(a: datetime | None, b: datetime) -> bool | None:
    if a is None or (a.tzinfo is None) != (b.tzinfo is None):
        return None
    return a == b


# --- outcome classification ------------------------------------------------------------

def from_final_save(exc: BaseException) -> bool:
    """Was `exc` raised from inside graphiti's final bulk save?"""
    for frame, _ in traceback.walk_tb(exc.__traceback__):
        if (frame.f_code.co_name == FINAL_SAVE_FUNCTION
                and frame.f_globals.get("__name__") == FINAL_SAVE_MODULE):
            return True
    return False


def is_read_only_refusal(exc: BaseException) -> bool:
    return isinstance(exc, ReadOnlyViolation) or (
        isinstance(exc, Neo4jError) and getattr(exc, "code", None) == ACCESS_MODE_CODE)


def classify(exc: BaseException | None, run: ChunkRun) -> tuple[str, str | None]:
    """(state, error message)."""
    if exc is None:
        return "error", ("add_episode returned normally: the final save did NOT fail on "
                         "the read-only driver")
    if isinstance(exc, TeacherNeeded):
        return "pending", None
    if isinstance(exc, TeacherDropped):
        return "dropped", None
    if is_read_only_refusal(exc) and from_final_save(exc):
        # A pending or dropped prompt whose exception graphiti swallowed (it does for
        # extract_edges.extract_timestamps) must not let the chunk count as complete.
        if run.dropped:
            return "dropped", None
        if run.needed:
            return "pending", None
        return "complete", None
    return "error", f"{type(exc).__name__}: {exc}"


async def _drain(tasks: set[asyncio.Task], chunk_id: str) -> None:
    """Let graphiti's stray tasks (gather siblings of the call that raised) finish, so
    nothing from this chunk is still running -- or still recording -- once its state
    is decided."""
    if not tasks:
        return
    done, pending = await asyncio.wait(tasks, timeout=DRAIN_TIMEOUT_SECONDS)
    for t in pending:
        logger.warning("chunk %s: cancelling a graphiti task still running after %.0fs",
                       chunk_id, DRAIN_TIMEOUT_SECONDS)
        t.cancel()
    for t in done:
        if t.cancelled():
            continue
        exc = t.exception()
        if exc is not None and not isinstance(exc, (TeacherNeeded, TeacherDropped)):
            logger.warning("chunk %s: a sibling graphiti task failed: %s: %s",
                           chunk_id, type(exc).__name__, exc)


# --- the round -----------------------------------------------------------------------

@dataclass
class RoundSummary:
    round: int
    states: Counter = field(default_factory=Counter)
    skipped_final: int = 0
    articles_missing: list[str] = field(default_factory=list)
    new_pending: int = 0


async def run_chunk(*, graphiti: Graphiti, settings: ExtractSettings, store: AnswerStore,
                    log: RoundLog, read: Any, art: ArticleChunks, chunk: Chunk,
                    vendor: str, product: str, source: str, round_no: int) -> dict:
    chunk_id = f"{art.article_id}:{chunk.chunk_index}"
    run = ChunkRun(chunk_id=chunk_id, episode_uuid=chunk.uuid)
    exc: BaseException | None = None
    ref: datetime | None = None
    previous: list[str] = []
    before = asyncio.all_tasks()
    try:
        if not chunk.content:
            raise ValueError(f"episode {chunk.uuid} has no stored content")
        run.prod_names, run.prod_facts = await read_footprint(read, chunk.uuid)
        ref = reference_time(chunk)
        previous = await read_previous_episodes(read, settings, art, chunk)
        stats = GroundingStats()
        ctoken = CURRENT_CHUNK.set(run)
        gtoken = CURRENT_GROUNDING.set(GroundingContext(
            text=chunk.content, allowed=tuple(n for n in (product, vendor) if n),
            stats=stats, product=product))
        try:
            await add_text_episode(
                graphiti, settings, name=chunk.name, body=chunk.content,
                source_description=art.source_url, reference_time=ref,
                instructions=EXTRACTION_INSTRUCTIONS + CHEAP_TIER_SALIENCE,
                context=document_context(vendor, product, source),
                previous_episode_uuids=list(previous))
        finally:
            CURRENT_GROUNDING.reset(gtoken)
            CURRENT_CHUNK.reset(ctoken)
    except Exception as e:  # classified below; nothing is swallowed
        exc = e
    current = asyncio.current_task()
    await _drain({t for t in asyncio.all_tasks() - before if t is not current and not t.done()},
                 chunk_id)
    if isinstance(exc, TeacherNeeded):
        # The transport already queued it; idempotent, and covers a TeacherNeeded
        # raised anywhere else.
        store.add_pending(exc.key, exc.request)
        if exc.key not in run.needed:
            prompt_name = str(exc.request.get("prompt_name"))
            log.asked(run, exc.key, prompt_name, exc.request.get("messages") or [],
                      needed=True)
    if isinstance(exc, TeacherDropped):
        run.dropped.setdefault(exc.key, exc.reason)
    state, error = classify(exc, run)
    if state == "dropped":
        # A dropped chunk will never ask again: take its other pending prompts off
        # the teacher's queue, unless another chunk asked them too.
        for key in run.needed:
            if set(store.askers(key)) <= {chunk_id}:
                store.unqueue(key)
    if state == "error":
        logger.error("chunk %s: %s", chunk_id, error,
                     exc_info=(type(exc), exc, exc.__traceback__) if exc else None)
    record = {
        "round": round_no, "chunk_id": chunk_id, "article_id": art.article_id,
        "chunk_index": chunk.chunk_index, "episode_uuid": chunk.uuid, "state": state,
        "pending_keys": sorted(run.needed), "dropped_keys": sorted(run.dropped),
        "replayed": run.replayed, "previous_episode_uuids": list(previous),
        "reference_time": ref.isoformat() if ref else None,
        "ref_matches_episode": (_same_instant(article_reference_time(art), ref)
                                if ref else None),
        "error": error, "ts": _utcnow()}
    log.chunk_done(record)
    return record


async def run_round(articles: Sequence[SampleArticle], *, graphiti: Graphiti,
                    settings: ExtractSettings, store: AnswerStore, log: RoundLog,
                    read: Any) -> RoundSummary:
    round_no = log.next_round()
    latest = log.latest_states()
    pending_before = len(store.pending())
    summary = RoundSummary(round=round_no)
    for sa in articles:
        art = await read_article_chunks(read, sa.article_id)
        if art is None:
            logger.warning("article %s: no live production episodes; skipped", sa.article_id)
            summary.articles_missing.append(sa.article_id)
            continue
        vendor, product, source = await read_document_context(read, sa.article_id)
        if (vendor, product) != (sa.vendor, sa.product):
            logger.warning("article %s: document context %s / %s differs from the sample's "
                           "%s / %s; using production's", sa.article_id, vendor, product,
                           sa.vendor, sa.product)
        for chunk in art.chunks:
            chunk_id = f"{sa.article_id}:{chunk.chunk_index}"
            if latest.get(chunk_id, {}).get("state") in FINAL_STATES:
                summary.skipped_final += 1
                continue
            rec = await run_chunk(graphiti=graphiti, settings=settings, store=store, log=log,
                                  read=read, art=art, chunk=chunk,
                                  vendor=vendor, product=product, source=source,
                                  round_no=round_no)
            summary.states[rec["state"]] += 1
    summary.new_pending = len(store.pending()) - pending_before
    return summary


# --- status ----------------------------------------------------------------------------

def status_text(store_root: Path) -> str:
    store = AnswerStore(store_root)
    log = RoundLog(store)
    latest = log.latest_states()
    states = Counter(r["state"] for r in latest.values())
    pending = store.pending()
    by_type = Counter(str(req.get("prompt_name")) for _, req in pending)
    lines = [f"store: {store_root}",
             f"rounds run: {log.next_round() - 1}",
             f"chunks: {len(latest)} -- " + ", ".join(f"{s} {states.get(s, 0)}" for s in STATES)]
    errors = [r for r in latest.values() if r["state"] == "error"]
    for r in errors[:10]:
        lines.append(f"  error {r['chunk_id']}: {r.get('error')}")
    lines.append(f"pending prompts: {len(pending)}")
    for name, n in sorted(by_type.items()):
        lines.append(f"  {name}: {n}")
    lines.append(f"answered: {len(store.answered_keys())}")
    lines.append(f"dropped: {len(store.dropped_keys())}")
    sm = _read_jsonl(log.selfmatch_path)
    lines.append("dedup self-matches (candidate = this chunk's own production extraction):")
    if not sm:
        lines.append("  none recorded yet")
    for name in SELF_MATCH_PROMPTS:
        recs = [r for r in sm if r["prompt_name"] == name]
        if not recs:
            continue
        cands = sum(r["candidates"] for r in recs)
        hits = sum(r["self_matches"] for r in recs)
        with_hit = sum(1 for r in recs if r["self_matches"])
        unparsed = sum(1 for r in recs if not r["parsed"])
        share = f"{hits / cands:.1%}" if cands else "n/a"
        lines.append(f"  {name}: {hits} of {cands} candidates ({share}); "
                     f"{with_hit} of {len(recs)} prompts with >=1; unparsed {unparsed}")
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------------------

def _read_driver(settings: ExtractSettings) -> Any:
    """A plain neo4j driver behind the read-only wrapper: READ routing on every
    query, READ access mode on every session."""
    raw = AsyncGraphDatabase.driver(settings.neo4j_uri,
                                    auth=(settings.neo4j_user, settings.neo4j_password))
    return _ReadOnlyAsyncDriver(raw)


async def _cmd_select(args: argparse.Namespace) -> int:
    settings = get_extract_settings()
    read = _read_driver(settings)
    try:
        sample, counts = await select_sample_with_counts(
            read, n_total=args.n_total, n_heldout=args.n_heldout, seed=args.seed,
            vendors=args.vendors, thresholds=RoutingThresholds.from_settings(settings))
    finally:
        await read.close()
    write_sample(args.out, sample, counts, seed=args.seed, n_total=args.n_total,
                 n_heldout=args.n_heldout, vendors=args.vendors)
    train = sum(1 for a in sample if a.split == "train")
    excluded = sum(c["excluded_strong_tier"] for c in counts.values())
    print(f"wrote {args.out}: {len(sample)} articles ({train} train, "
          f"{len(sample) - train} held out) over {len(counts)} strata; "
          f"{excluded} strong-tier articles excluded")
    return 0


async def _cmd_round(args: argparse.Namespace) -> int:
    store_root = Path(args.store)
    store = AnswerStore(store_root)
    log = RoundLog(store)
    articles = load_sample(args.sample)
    if args.articles is not None:
        articles = articles[:args.articles]
    settings = harness_settings(get_extract_settings(), store_root)
    graphiti, _ = build_teacher_graphiti(settings, store, log)
    read = _read_driver(settings)
    try:
        summary = await run_round(articles, graphiti=graphiti, settings=settings,
                                  store=store, log=log, read=read)
    finally:
        await read.close()
        await graphiti.close()
    print(f"round {summary.round}: " + ", ".join(
        f"{s} {summary.states.get(s, 0)}" for s in STATES)
          + f"; skipped (final) {summary.skipped_final}; new pending prompts "
          f"{summary.new_pending}; articles without episodes {len(summary.articles_missing)}")
    return 1 if summary.states.get("error") else 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m scripts.distill.harness")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select", help="pick the stratified sample (read-only)")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--n-total", type=int, default=100)
    s.add_argument("--n-heldout", type=int, default=20)
    s.add_argument("--seed", type=int, default=20261017)
    s.add_argument("--vendors", nargs="+", default=list(DEFAULT_VENDORS))
    r = sub.add_parser("round", help="replay every sampled chunk once")
    r.add_argument("--sample", type=Path, required=True)
    r.add_argument("--store", type=Path, required=True)
    r.add_argument("--articles", type=int, default=None,
                   help="only the first N articles of the sample (pilot)")
    st = sub.add_parser("status", help="chunk states, pending prompts, self-matches")
    st.add_argument("--store", type=Path, required=True)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parser().parse_args(argv)
    if args.cmd == "status":
        print(status_text(args.store))
        return 0
    if args.cmd == "select":
        return asyncio.run(_cmd_select(args))
    return asyncio.run(_cmd_round(args))


if __name__ == "__main__":
    sys.exit(main())
