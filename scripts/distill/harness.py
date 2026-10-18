"""The Opus-distillation harness: `select`, `round`, `status`, `batches`, `intake`,
`review-sample`, `report`, `export`.

    uv run --extra dev python -m scripts.distill.harness select --out DIR/sample.json
    uv run --extra dev python -m scripts.distill.harness round \
        --sample DIR/sample.json --store DIR/store [--articles N]
    uv run --extra dev python -m scripts.distill.harness status --store DIR/store
    uv run --extra dev python -m scripts.distill.harness batches \
        --store DIR/store --size 10 --out DIR/batches/round-N/
    # a teacher subagent answers each batch-XXX.jsonl into batch-XXX.answers.jsonl
    uv run --extra dev python -m scripts.distill.harness intake \
        --store DIR/store --answers DIR/batches/round-N/ --sample DIR/sample.json
    uv run --extra dev python -m scripts.distill.harness review-sample \
        --store DIR/store --fraction 0.05 --seed 20261017 --out DIR/reviews/
    uv run --extra dev python -m scripts.distill.harness report \
        --store DIR/store --batches DIR/batches/ [--out DIR/pilot-report.md]
    uv run --extra dev python -m scripts.distill.harness export \
        --store DIR/store --out DIR/capture.jsonl

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
import hashlib
import json
import logging
import math
import os
import random
import sys
import traceback
from collections import Counter, defaultdict
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
from scripts.distill.validate import build_grounding_context, validate_answer

logger = logging.getLogger("distill.harness")

READ = RoutingControl.READ
PREVIOUS_EPISODES_MAX = 10          # graphiti's RELEVANT_SCHEMA_LIMIT, what production saw
ACCESS_MODE_CODE = "Neo.ClientError.Statement.AccessMode"
FINAL_SAVE_MODULE = "graphiti_core.utils.bulk_utils"
FINAL_SAVE_FUNCTION = "add_nodes_and_edges_bulk"
NODE_DEDUP_PROMPT_NAME = "dedupe_nodes.nodes"
SELF_MATCH_PROMPTS = (NODE_DEDUP_PROMPT_NAME, DEDUP_PROMPT_NAME)
STATES = ("pending", "complete", "dropped", "error")
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

# What the harness passes as the cheap tier's (and so the cross-encoder's) API key: a
# non-secret placeholder, the same one production uses for keyless local endpoints.
NO_API_KEY = "not-needed"


def harness_settings(base: ExtractSettings, store_root: Path) -> ExtractSettings:
    """Production settings with the harness's overrides:

    - capture into the store (`llm_capture_path`);
    - the teacher's model label on the cheap tier (`cheap_llm_model`);
    - `extraction_routing=True`, production's value. The harness never consults it:
      it only ever builds the cheap graphiti (`build_teacher_graphiti`), so no
      dense-matrix router exists here, whatever the flag says;
    - contradictions as production runs them (detection off, suppression on) and no
      strong-tier dedup retry -- a differing environment value is overridden, with
      a WARNING;
    - the cheap tier's cache layout ON (`cheap_llm_cache_layout`, production's
      CHEAP_LLM_CACHE_LAYOUT=true): with it off every capture is "unmoved" and the
      student trains on a layout production never sends it in steady state;
    - NO real API key on the cheap tier or the reranker: `cheap_llm_api_key` is the
      placeholder `NO_API_KEY` and `rerank_api_key=""`. The LLM transport is the fake
      `TeacherTransport`, and graphiti's cross-encoder -- built by `build_graphiti`
      from the cheap view, i.e. from `cheap_llm_api_key` -- is never called during
      `add_episode`. Defence in depth: nothing the harness builds can authenticate to
      a paid endpoint. (A placeholder, not "": the openai SDK refuses to construct a
      client whose api_key is empty, and the cross-encoder constructs one.)"""
    for name, want in (("ingest_detect_contradictions", False),
                       ("ingest_same_pair_contradictions", False),
                       ("dedup_retry_on_strong", False),
                       ("cheap_llm_cache_layout", True)):
        if getattr(base, name) != want:
            logger.warning("overriding %s=%s from the environment to %s (production's "
                           "documented value)", name, getattr(base, name), want)
    return base.model_copy(update=dict(
        llm_capture_path=str(Path(store_root) / "capture.jsonl"),
        cheap_llm_model=TeacherTransport.MODEL,
        extraction_routing=True,
        ingest_detect_contradictions=False,
        ingest_same_pair_contradictions=False,
        dedup_retry_on_strong=False,
        cheap_llm_cache_layout=True,
        cheap_llm_api_key=NO_API_KEY,
        rerank_api_key=""))


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
    # The raw `HAS_EPISODE.chunk_index` as read; `run_round` parses it to an int per
    # chunk (`_parse_chunk_index`), so one bad row errors one chunk, not the round.
    chunk_index: Any
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
        chunks=[Chunk(chunk_index=r["chunk_index"], uuid=r["uuid"], name=r["name"],
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
    stale_unqueued: int = 0


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


def _parse_chunk_index(raw: Any) -> int:
    """`HAS_EPISODE.chunk_index` as an int; raises TypeError/ValueError on a row that
    has none (null) or a non-integer one."""
    if isinstance(raw, bool):
        raise TypeError(f"chunk_index is a bool: {raw!r}")
    if isinstance(raw, float) and not raw.is_integer():
        raise ValueError(f"chunk_index is not an integer: {raw!r}")
    return int(raw)


def _bad_chunk_record(round_no: int, art: ArticleChunks, chunk: Chunk,
                      exc: BaseException) -> dict:
    """The `error` record of a chunk whose row could not even be parsed."""
    return {
        "round": round_no, "chunk_id": f"{art.article_id}:{chunk.chunk_index}",
        "article_id": art.article_id, "chunk_index": chunk.chunk_index,
        "episode_uuid": chunk.uuid, "state": "error", "pending_keys": [],
        "dropped_keys": [], "replayed": 0, "previous_episode_uuids": [],
        "reference_time": None, "ref_matches_episode": None,
        "error": f"bad chunk_index {chunk.chunk_index!r}: {type(exc).__name__}: {exc}",
        "ts": _utcnow()}


def _parse_article_chunks(art: ArticleChunks, round_no: int, log: RoundLog,
                          summary: RoundSummary) -> None:
    """Parse every chunk's index, per chunk: a bad row is recorded `error` and left
    out of `art.chunks` (so it is neither replayed nor ordered as a sibling); the
    other chunks of the article still run."""
    good: list[Chunk] = []
    for chunk in art.chunks:
        try:
            chunk.chunk_index = _parse_chunk_index(chunk.chunk_index)
        except (TypeError, ValueError) as exc:
            rec = _bad_chunk_record(round_no, art, chunk, exc)
            logger.error("chunk %s: %s", rec["chunk_id"], rec["error"])
            log.chunk_done(rec)
            summary.states["error"] += 1
            continue
        good.append(chunk)
    art.chunks = good


def skip_in_round(state: str | None, *, replay_final: bool) -> bool:
    """A `dropped` chunk is never replayed; a `complete` one only with `replay_final`
    (which recaptures its stored answers, e.g. after a capture-affecting fix)."""
    if state == "dropped":
        return True
    return state == "complete" and not replay_final


async def run_round(articles: Sequence[SampleArticle], *, graphiti: Graphiti,
                    settings: ExtractSettings, store: AnswerStore, log: RoundLog,
                    read: Any, replay_final: bool = False) -> RoundSummary:
    round_no = log.next_round()
    latest = log.latest_states()
    pending_before = len(store.pending())
    summary = RoundSummary(round=round_no)
    ran: set[str] = set()
    asked_now: set[str] = set()
    for sa in articles:
        art = await read_article_chunks(read, sa.article_id)
        if art is None:
            logger.warning("article %s: no live production episodes; skipped", sa.article_id)
            summary.articles_missing.append(sa.article_id)
            continue
        _parse_article_chunks(art, round_no, log, summary)
        vendor, product, source = await read_document_context(read, sa.article_id)
        if (vendor, product) != (sa.vendor, sa.product):
            logger.warning("article %s: document context %s / %s differs from the sample's "
                           "%s / %s; using production's", sa.article_id, vendor, product,
                           sa.vendor, sa.product)
        for chunk in art.chunks:
            chunk_id = f"{sa.article_id}:{chunk.chunk_index}"
            if skip_in_round(latest.get(chunk_id, {}).get("state"),
                             replay_final=replay_final):
                summary.skipped_final += 1
                continue
            rec = await run_chunk(graphiti=graphiti, settings=settings, store=store, log=log,
                                  read=read, art=art, chunk=chunk,
                                  vendor=vendor, product=product, source=source,
                                  round_no=round_no)
            summary.states[rec["state"]] += 1
            ran.add(chunk_id)
            asked_now.update(rec["pending_keys"])
    summary.stale_unqueued = unqueue_stale(store, ran=ran, asked_now=asked_now)
    summary.new_pending = len(store.pending()) - pending_before
    return summary


def unqueue_stale(store: AnswerStore, *, ran: set[str], asked_now: set[str]) -> int:
    """Take off the teacher's queue every pending prompt that no chunk asked for this
    round although every chunk that ever asked for it was replayed: its prompt bytes
    changed (an unpinned hash seed, a code change), so its answer could never replay.
    A prompt also asked by a chunk this round did not run is kept. Returns the count."""
    n = 0
    for key, _ in store.pending():
        askers = set(store.askers(key))
        if key not in asked_now and askers and askers <= ran:
            store.unqueue(key)
            n += 1
    return n


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
    lines.append(f"{NOT_REPLAYED_LABEL}: {len(answered_not_replayed(store_root))}")
    lines.append("dedup self-matches (candidate = this chunk's own production extraction):")
    lines.extend(_selfmatch_lines(_read_jsonl(log.selfmatch_path)))
    return "\n".join(lines)


NOT_REPLAYED_LABEL = "answered but not replayed in the latest round"


def answered_not_replayed(store_root: Path) -> list[str]:
    """Accepted (not dropped) keys that no round has replayed since intake accepted
    them -- every answer accepted after the latest round. `calls.jsonl` gets a
    record the first time an answered call goes through the transport (a replay),
    and a key can only be replayed once it is answered, so an accepted key with no
    `calls.jsonl` record has never been replayed: it has no capture record yet, and
    `export` would count it `missing`. Also catches a key whose every asker chunk
    ended `dropped` on another key (it will never be replayed)."""
    store = AnswerStore(store_root)
    replayed = {r["key"] for r in _read_jsonl(store_root / "calls.jsonl")}
    accepted = set(store.answered_keys()) - set(store.dropped_keys())
    return sorted(accepted - replayed)


def _selfmatch_lines(sm: list[dict]) -> list[str]:
    """The self-match share per dedup prompt type (`status` and `report`)."""
    if not sm:
        return ["  none recorded yet"]
    lines = []
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
    return lines


# --- batches / intake / review-sample / report / export (task 4) ---------------------
#
# Five more stages on the same on-disk store, run between rounds:
#
# - `batches` writes the teacher's work: every still-pending request, grouped so one
#   file never mixes prompt types, carrying a re-ask's `previous_answer`/`rejection`,
#   and on every line its `ordinal` (1-based position in the file) and `check` (the
#   first 12 hex characters of sha256 of the user message), which the teacher echoes.
# - `intake` reads the teacher's `*.answers.jsonl` replies, checks the echoed
#   `ordinal`/`check` against the key's batch line (an answer filed under the wrong
#   key is rejected), validates each (`scripts.distill.validate.validate_answer`) and
#   accepts, re-queues (first rejection) or drops (second) it.
# - `review-sample` draws ~5% of accepted answers, stratified by prompt type (at
#   least one per type that has any), for a reviewer subagent.
# - `report` writes the pilot report: per prompt type, keys asked / accepted /
#   re-asked / dropped, validity and re-ask rates, prompt and answer sizes.
# - `export` copies the capture record of every accepted key into the production
#   capture format, preferring the "moved" wire layout (`calls.jsonl` tells layouts
#   apart; see the module docstring).
#
# A rejection's history lives in its own append-only log (`AnswerStore` has no notion
# of "rejected once already" -- that is this task's own bookkeeping, not the
# store's): `<store>/rejections.jsonl`, one record per rejection, `key` ->
# `{"reason", "previous_answer", "prompt_name"}`. A SECOND rejection of a key is what
# turns into `store.drop` -- and only a genuine second answer counts as one: the
# answer to a re-ask (its batch line carried `rejection`) or an answer that differs
# from the one already rejected. Re-running `intake` over the same answers directory
# sees the same rejected answer again and leaves the key pending (`already_rejected`).

REJECTIONS_FILENAME = "rejections.jsonl"
WRONG_PROMPT_REASON = "answer filed under the wrong prompt"
CHECK_HEX_CHARS = 12
INTAKE_OUTCOMES = ("accepted", "requeued", "dropped", "unanswered")


def _rejections_by_key(store_root: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for rec in _read_jsonl(store_root / REJECTIONS_FILENAME):
        out[rec["key"]].append(rec)
    return out


def _append_rejection(store_root: Path, key: str, *, reason: str, previous_answer: Any,
                      prompt_name: str | None = None) -> None:
    _append_jsonl(store_root / REJECTIONS_FILENAME, {
        "key": key, "reason": reason, "previous_answer": previous_answer,
        "prompt_name": prompt_name, "ts": _utcnow()})


def user_message_check(messages: Sequence[Any]) -> str:
    """The first `CHECK_HEX_CHARS` hex characters of sha256 over the request's user
    message (every user-role message's content, joined by a newline -- graphiti
    sends one). A string content is hashed as is, anything else as canonical JSON."""
    parts = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role != "user":
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        parts.append(content if isinstance(content, str)
                     else json.dumps(content, sort_keys=True, ensure_ascii=False))
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:CHECK_HEX_CHARS]


def write_batches(store_root: Path, out_dir: Path, *, size: int) -> list[Path]:
    """`<out_dir>/batch-NNN.jsonl`: every pending request, grouped by prompt type (one
    file never mixes types) with at most `size` lines each. Every line carries
    `ordinal` (its 1-based position in its file) and `check` (`user_message_check`);
    a key already rejected once carries `previous_answer`/`rejection` so the teacher
    fixes exactly that."""
    store = AnswerStore(store_root)
    rejections = _rejections_by_key(store_root)
    by_prompt: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for key, request in store.pending():
        by_prompt[str(request.get("prompt_name"))].append((key, request))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    n = 0
    for prompt_name in sorted(by_prompt):
        items = sorted(by_prompt[prompt_name], key=lambda kv: kv[0])
        for i in range(0, len(items), size):
            lines = []
            for ordinal, (key, request) in enumerate(items[i:i + size], start=1):
                messages = request.get("messages") or []
                line = {"key": key, "ordinal": ordinal,
                        "check": user_message_check(messages),
                        "prompt_name": prompt_name,
                        "messages": messages,
                        "response_schema": request.get("response_format")}
                history = rejections.get(key)
                if history:
                    line["previous_answer"] = history[-1]["previous_answer"]
                    line["rejection"] = history[-1]["reason"]
                lines.append(line)
            path = out_dir / f"batch-{n:03d}.jsonl"
            path.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n"
                                    for line in lines), encoding="utf-8")
            paths.append(path)
            n += 1
    return paths


@dataclass
class IntakeSummary:
    accepted: int = 0
    requeued: int = 0           # first rejection: back on the teacher's queue
    dropped: int = 0            # second rejection: given up on
    skipped: int = 0            # answer for a key no longer pending (already resolved)
    unanswered: int = 0         # a batch key with no parseable answer line at all
    already_rejected: int = 0   # the same rejected answer seen again (intake re-run)
    # prompt_name -> Counter over INTAKE_OUTCOMES
    by_prompt: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    def count(self, outcome: str, prompt_name: str) -> None:
        setattr(self, outcome, getattr(self, outcome) + 1)
        self.by_prompt[prompt_name][outcome] += 1


def _answer_files(answers_dir: Path) -> list[Path]:
    return sorted(answers_dir.glob("*.answers.jsonl"))


def _batch_files(answers_dir: Path) -> list[Path]:
    """The `batches`-written request files sitting alongside the teacher's
    `*.answers.jsonl` replies in the same directory -- everything ending in
    `.jsonl` that is NOT an answers file."""
    return sorted(p for p in answers_dir.glob("*.jsonl") if not p.name.endswith(".answers.jsonl"))


def _batch_lines(answers_dir: Path) -> dict[str, dict]:
    """key -> its batch line, for every key `batches` asked for in this round, read
    from our OWN files (`_read_jsonl`'s strict parser is fine here -- we wrote them)."""
    return {rec["key"]: rec for path in _batch_files(answers_dir) for rec in _read_jsonl(path)}


def _requested_keys(answers_dir: Path) -> set[str]:
    return set(_batch_lines(answers_dir))


def _filed_under_wrong_key(rec: dict, line: dict | None) -> bool:
    """Does the answer's echoed `ordinal`/`check` disagree with its key's batch line?
    A batch line written before `ordinal`/`check` existed (or no batch line in this
    directory) cannot be checked and passes."""
    if line is None or "ordinal" not in line or "check" not in line:
        return False
    return (str(rec.get("ordinal")).strip() != str(line["ordinal"])
            or str(rec.get("check")).strip().lower() != str(line["check"]).lower())


def _read_answer_lines(path: Path) -> list[dict]:
    """A teacher's `*.answers.jsonl` file, TOLERANTLY -- this is untrusted LLM output,
    never the harness's own, so it must never raise: a line that is not itself valid
    JSON, or is valid JSON but not an object carrying both "key" and "answer", is
    skipped with a WARNING naming the file and line number (a markdown code fence, a
    blank line, or a line of prose all fail this check harmlessly and are skipped the
    same way). An "answer" that is itself a JSON-encoded string (double-encoded) is
    decoded once; if that inner string is not valid JSON it is passed through as-is --
    `validate_answer`'s schema check, not this reader, is what rejects a bad answer
    shape."""
    if not path.exists():
        return []
    out: list[dict] = []
    for i, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("intake: %s:%d is not valid JSON; skipped: %.200r",
                           path, i + 1, line)
            continue
        if not isinstance(rec, dict) or "key" not in rec or "answer" not in rec:
            logger.warning(
                'intake: %s:%d is not a {"key", "answer"} object; skipped: %.200r',
                path, i + 1, line)
            continue
        answer = rec["answer"]
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except json.JSONDecodeError:
                pass          # left as the raw string; validate_answer rejects it
            else:
                rec = {**rec, "answer": answer}
        out.append(rec)
    return out


def _articles_for_key(store: AnswerStore, articles_by_id: dict[str, SampleArticle],
                      key: str) -> list[SampleArticle]:
    """Every DISTINCT (vendor, product) context behind this key's askers, in a
    deterministic order (the asker chunk ids sorted first) -- almost always one, but a
    prompt with identical wire bytes asked by chunks of different articles can carry
    more than one. `harness.run_chunk` names a chunk id `"<article_id>:<chunk_index>"`.
    An asker whose article is not in the sample contributes nothing (grounding is
    skipped for it, never guessed)."""
    seen: dict[tuple[str, str], SampleArticle] = {}
    for chunk_id in sorted(store.askers(key)):
        article = articles_by_id.get(chunk_id.rsplit(":", 1)[0])
        if article is not None:
            seen.setdefault((article.vendor, article.product), article)
    return list(seen.values())


def _validate_across_contexts(prompt_name: str, messages: list, answer: Any,
                              articles: list[SampleArticle]) -> list[str]:
    """`validate_answer` against every distinct context in `articles` (or just the
    ungrounded context when `articles` is empty) -- accepted only when EVERY context
    accepts (conservative). On any rejection, the combined, de-duplicated reasons from
    every context, prefixed with the multi-product case when there was one."""
    contexts = articles or [None]
    per_context: list[list[str]] = []
    for article in contexts:
        ctx = (build_grounding_context(prompt_name, messages, product=article.product,
                                       vendor=article.vendor)
              if article is not None else None)
        per_context.append(validate_answer(prompt_name, messages, answer, grounding_ctx=ctx))
    combined = [r for reasons in per_context for r in reasons]
    if not combined:
        return []
    reasons = list(dict.fromkeys(combined))      # de-duplicated, order preserved
    if len(articles) > 1:
        where = [(a.vendor, a.product) for a in articles]
        reasons = [f"multi-product key ({len(articles)} contexts {where}): " + reasons[0],
                  *reasons[1:]]
    return reasons


def run_intake(store_root: Path, answers_dir: Path, sample_path: Path) -> IntakeSummary:
    """Validate every teacher-answered key in `answers_dir`'s `*.answers.jsonl` files
    against the request still pending under that key. A key `batches` asked for in
    this round (`_batch_lines`) that never turns up as a parseable, correctly-shaped
    answer line in any of them stays pending untouched -- it is simply re-batched next
    round -- and is counted as `unanswered`.

    An answer whose echoed `ordinal`/`check` does not match its key's batch line is
    rejected (`WRONG_PROMPT_REASON`) without being validated. A rejection of a key
    already rejected once is the SECOND (a drop) only when this answer is a genuine
    second one: its batch line carried `rejection` (a re-ask), or the answer differs
    from the stored `previous_answer`. Otherwise it is the same rejected answer seen
    again -- `intake` re-run over the same directory -- and the key stays pending
    with no new rejection record (`already_rejected`)."""
    store = AnswerStore(store_root)
    pending = dict(store.pending())
    rejections = _rejections_by_key(store_root)
    articles_by_id = {a.article_id: a for a in load_sample(sample_path)}
    batch_lines = _batch_lines(answers_dir)
    summary = IntakeSummary()
    found_keys: set[str] = set()
    for path in _answer_files(answers_dir):
        for rec in _read_answer_lines(path):
            key, answer = rec["key"], rec["answer"]
            found_keys.add(key)
            request = pending.get(key)
            if request is None:
                summary.skipped += 1
                logger.warning(
                    "intake: %s has no pending request (already resolved); skipped", key)
                continue
            prompt_name = str(request.get("prompt_name"))
            messages = request.get("messages") or []
            line = batch_lines.get(key)
            if _filed_under_wrong_key(rec, line):
                assert line is not None
                logger.warning(
                    "intake: %s: echoed ordinal/check %r/%r != its batch line's %r/%r",
                    key, rec.get("ordinal"), rec.get("check"), line["ordinal"],
                    line["check"])
                reasons = [WRONG_PROMPT_REASON]
            else:
                articles = _articles_for_key(store, articles_by_id, key)
                if len(articles) > 1:
                    logger.warning(
                        "intake: %s has askers across %d distinct (vendor, product) "
                        "contexts: %s", key, len(articles),
                        [(a.vendor, a.product) for a in articles])
                reasons = _validate_across_contexts(prompt_name, messages, answer, articles)
            if not reasons:
                store.put(key, json.dumps(answer, ensure_ascii=False),
                         {"prompt_name": prompt_name, "messages": messages})
                summary.count("accepted", prompt_name)
                pending.pop(key, None)
                continue
            reason = "; ".join(reasons)
            history = rejections.get(key)
            if history:
                reask = bool(line is not None and line.get("rejection"))
                if not reask and answer == history[-1]["previous_answer"]:
                    summary.already_rejected += 1
                    logger.info("intake: %s: the same answer was already rejected once "
                                "(intake re-run?); left pending, not dropped", key)
                    continue
                store.drop(key, f"re-ask rejected again: first={history[-1]['reason']!r}; "
                                f"second={reason!r}", prompt_name=prompt_name)
                summary.count("dropped", prompt_name)
                pending.pop(key, None)
            else:
                summary.count("requeued", prompt_name)
            _append_rejection(store_root, key, reason=reason, previous_answer=answer,
                              prompt_name=prompt_name)
            rejections.setdefault(key, []).append({"reason": reason, "previous_answer": answer})
    for key in sorted(set(batch_lines) - found_keys):
        summary.count("unanswered", str(batch_lines[key].get("prompt_name")))
    return summary


def _accepted_prompt_names(store_root: Path) -> dict[str, str]:
    """key -> prompt_name, from the meta `intake` recorded at accept time (the latest
    record per key wins, same as every other append-only log here)."""
    out: dict[str, str] = {}
    for rec in _read_jsonl(store_root / "answers.jsonl"):
        name = (rec.get("meta") or {}).get("prompt_name")
        if name:
            out[rec["key"]] = name
    return out


def write_review_sample(store_root: Path, out_dir: Path, *, fraction: float,
                        seed: int) -> Path:
    """`<out_dir>/review-sample.jsonl`: about `fraction` of accepted answers, sampled
    independently within each prompt type with a seeded shuffle keyed on
    `(seed, "review", prompt_name)` -- deterministic for a fixed store and seed. Every
    prompt type with at least one accepted answer contributes at least one sample
    (a rare type would otherwise round to zero and never be reviewed)."""
    store = AnswerStore(store_root)
    accepted = set(store.answered_keys())
    meta_by_key = {rec["key"]: rec.get("meta") or {}
                  for rec in _read_jsonl(store_root / "answers.jsonl")}
    prompt_by_key = {k: v for k, v in _accepted_prompt_names(store_root).items()
                     if k in accepted}
    by_prompt: dict[str, list[str]] = defaultdict(list)
    for key, name in prompt_by_key.items():
        by_prompt[name].append(key)
    selected: list[str] = []
    for name in sorted(by_prompt):
        keys = sorted(by_prompt[name])
        random.Random(f"{seed}|review|{name}").shuffle(keys)
        selected.extend(sorted(keys[:max(1, round(len(keys) * fraction))]))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "review-sample.jsonl"
    lines = [{"key": key, "prompt_name": prompt_by_key.get(key),
             "messages": meta_by_key.get(key, {}).get("messages"),
             "answer": json.loads(store.get(key) or "null")}
            for key in selected]
    path.write_text("".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
                    encoding="utf-8")
    return path


# --- report ----------------------------------------------------------------------------

CHARS_PER_TOKEN_ESTIMATE = 4


def _percentile(values: Sequence[int], pct: float) -> int:
    """Nearest-rank percentile (0 for no values)."""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def _size_cell(values: Sequence[int]) -> str:
    """"p50 / p95 / max" chars, then the same in estimated tokens (chars / 4)."""
    if not values:
        return "n/a | n/a"
    stats = [_percentile(values, 50), _percentile(values, 95), max(values)]
    chars = " / ".join(str(v) for v in stats)
    tokens = " / ".join(str(round(v / CHARS_PER_TOKEN_ESTIMATE)) for v in stats)
    return f"{chars} | {tokens}"


def _rate(num: int, den: int) -> str:
    return f"{num / den:.1%}" if den else "n/a"


def _prompt_chars(messages: Sequence[Any]) -> int:
    return sum(len(t) for t in _message_texts(messages))


def report_text(store_root: Path, batches_dir: Path) -> str:
    """The pilot report (markdown). Every batch file under `batches_dir` (searched
    recursively, so the whole `DIR/batches/` with its `round-N/` folders or a single
    round folder both work) and the store's own logs.

    Per prompt type, all counts are DISTINCT KEYS:

    - asked: keys in any batch file;
    - answered: keys `intake` resolved at least once (accepted, or rejected at least
      once -- `rejections.jsonl`);
    - accepted: keys in the store's answers (not dropped);
    - re-asked: keys rejected at least once (each first rejection is sent back);
    - dropped: keys in `dropped.jsonl`;
    - validity = accepted / answered; re-ask rate = re-asked / answered.

    Sizes: prompt = every message's text of a key's latest batch line; answer = the
    JSON of every teacher answer line under `batches_dir` (rejected ones included).
    Tokens are estimated as chars / 4."""
    store = AnswerStore(store_root)
    log = RoundLog(store)
    batch_paths = sorted(p for p in batches_dir.rglob("*.jsonl")
                         if not p.name.endswith(".answers.jsonl"))
    answer_paths = sorted(batches_dir.rglob("*.answers.jsonl"))

    line_by_key: dict[str, dict] = {}
    for p in batch_paths:
        for rec in _read_jsonl(p):
            line_by_key[rec["key"]] = rec
    accepted_names = _accepted_prompt_names(store_root)
    dropped_recs = {r["key"]: r for r in _read_jsonl(store.dropped_path)}
    rejections = _rejections_by_key(store_root)

    def name_of(key: str, *hints: Any) -> str:
        for hint in hints:
            if hint:
                return str(hint)
        if key in line_by_key:
            return str(line_by_key[key].get("prompt_name"))
        return accepted_names.get(key) or "unknown"

    asked: dict[str, set[str]] = defaultdict(set)
    prompt_sizes: dict[str, list[int]] = defaultdict(list)
    for key, rec in line_by_key.items():
        name = name_of(key, rec.get("prompt_name"))
        asked[name].add(key)
        prompt_sizes[name].append(_prompt_chars(rec.get("messages") or []))
    answer_sizes: dict[str, list[int]] = defaultdict(list)
    for p in answer_paths:
        for rec in _read_answer_lines(p):
            answer_sizes[name_of(rec["key"])].append(
                len(json.dumps(rec["answer"], ensure_ascii=False)))
    dropped: dict[str, set[str]] = defaultdict(set)
    for key, rec in dropped_recs.items():
        last_rej = (rejections.get(key) or [{}])[-1]
        dropped[name_of(key, rec.get("prompt_name"), last_rej.get("prompt_name"))].add(key)
    accepted: dict[str, set[str]] = defaultdict(set)
    for key in set(store.answered_keys()) - set(dropped_recs):
        accepted[name_of(key, accepted_names.get(key))].add(key)
    reasked: dict[str, set[str]] = defaultdict(set)
    for key, recs in rejections.items():
        reasked[name_of(key, recs[-1].get("prompt_name"))].add(key)

    names = sorted(set(asked) | set(accepted) | set(reasked) | set(dropped)
                   | set(answer_sizes))
    out = ["# Distillation pilot report", "",
           f"- store: `{store_root}`",
           f"- batches: `{batches_dir}` ({len(batch_paths)} batch file(s), "
           f"{len(answer_paths)} answers file(s))",
           f"- rounds run: {log.next_round() - 1}",
           f"- generated: {_utcnow()}", "",
           "## Per prompt type (distinct keys)", "",
           "| prompt type | asked | answered | accepted | re-asked | dropped "
           "| validity (accepted / answered) | re-ask rate (re-asked / answered) |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    totals: Counter = Counter()
    for name in names:
        answered = len(accepted[name] | reasked[name])
        row = {"asked": len(asked[name]), "answered": answered,
               "accepted": len(accepted[name]), "reasked": len(reasked[name]),
               "dropped": len(dropped[name])}
        totals.update(row)
        out.append(f"| {name} | {row['asked']} | {answered} | {row['accepted']} "
                   f"| {row['reasked']} | {row['dropped']} "
                   f"| {_rate(row['accepted'], answered)} | {_rate(row['reasked'], answered)} |")
    out.append(f"| **total** | {totals['asked']} | {totals['answered']} "
               f"| {totals['accepted']} | {totals['reasked']} | {totals['dropped']} "
               f"| {_rate(totals['accepted'], totals['answered'])} "
               f"| {_rate(totals['reasked'], totals['answered'])} |")
    out += ["", "## Prompt and answer sizes", "",
            "p50 / p95 / max; tokens estimated as chars / "
            f"{CHARS_PER_TOKEN_ESTIMATE}.", "",
            "| prompt type | prompt chars | prompt tokens (est.) | answer chars "
            "| answer tokens (est.) |",
            "|---|---:|---:|---:|---:|"]
    for name in names:
        out.append(f"| {name} | {_size_cell(prompt_sizes[name])} "
                   f"| {_size_cell(answer_sizes[name])} |")

    latest = log.latest_states()
    states = Counter(r["state"] for r in latest.values())
    not_replayed = answered_not_replayed(store_root)
    out += ["", "## Chunks", "",
            f"{len(latest)} chunks: " + ", ".join(f"{s} {states.get(s, 0)}" for s in STATES),
            "", f"{NOT_REPLAYED_LABEL}: {len(not_replayed)} (run one more `round` "
            "before `export`)", "",
            "## Dedup self-matches", "",
            "Candidate = this chunk's own production extraction (the accepted caveat).", "",
            "```", *_selfmatch_lines(_read_jsonl(log.selfmatch_path)), "```", "",
            "## Controller notes (fill in by hand)", "",
            "Wall time and Agent usage per teacher batch -- not recorded by the harness.",
            "",
            "| batch file | wall time | Agent usage (tokens / tool calls) | notes |",
            "|---|---|---|---|"]
    for p in batch_paths:
        out.append(f"| {p.relative_to(batches_dir)} |  |  |  |")
    return "\n".join(out) + "\n"


@dataclass
class ExportSummary:
    exported: int = 0
    missing: int = 0                             # accepted key with no capture record
    by_prompt: Counter = field(default_factory=Counter)
    by_layout: Counter = field(default_factory=Counter)


def export_capture(store_root: Path, out_path: Path) -> ExportSummary:
    """`<out_path>`: one production-format capture record per accepted key. Every
    `capture.jsonl` record is matched back to its logical key by recomputing its
    WIRE hash (`AnswerStore.key_for` over the record's own `messages`/
    `response_format`) and looking it up in `calls.jsonl`'s wire_key -> (key, layout)
    map; among a key's matches the "moved" layout wins (production steady state),
    else "unmoved". Dropped keys are excluded. Records are copied unchanged."""
    store = AnswerStore(store_root)
    accepted = set(store.answered_keys()) - set(store.dropped_keys())
    wire_index: dict[str, tuple[str, str]] = {}
    for rec in _read_jsonl(store_root / "calls.jsonl"):
        wire_index[rec["wire_key"]] = (rec["key"], rec["layout"])
    by_key: dict[str, dict[str, dict]] = defaultdict(dict)
    for rec in _read_jsonl(store_root / "capture.jsonl"):
        wire_key = store.key_for(rec.get("messages") or [], rec.get("response_format"))
        found = wire_index.get(wire_key)
        if found is None:
            continue
        key, layout = found
        if key in accepted:
            by_key[key].setdefault(layout, rec)
    summary = ExportSummary()
    chosen: list[dict] = []
    for key in sorted(accepted):
        layouts = by_key.get(key, {})
        if "moved" in layouts:
            rec, layout = layouts["moved"], "moved"
        elif "unmoved" in layouts:
            rec, layout = layouts["unmoved"], "unmoved"
        elif layouts:
            layout, rec = next(iter(layouts.items()))
        else:
            summary.missing += 1
            continue
        chosen.append(rec)
        summary.exported += 1
        summary.by_prompt[str(rec.get("prompt_name"))] += 1
        summary.by_layout[layout] += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in chosen),
        encoding="utf-8")
    return summary


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
                                  store=store, log=log, read=read,
                                  replay_final=args.replay_final)
    finally:
        await read.close()
        await graphiti.close()
    print(f"round {summary.round}: " + ", ".join(
        f"{s} {summary.states.get(s, 0)}" for s in STATES)
          + f"; skipped (final) {summary.skipped_final}; new pending prompts "
          f"{summary.new_pending}; stale prompts unqueued {summary.stale_unqueued}; "
          f"articles without episodes {len(summary.articles_missing)}")
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
    r.add_argument("--replay-final", action="store_true",
                   help="also replay complete chunks (recapture their stored answers)")
    st = sub.add_parser("status", help="chunk states, pending prompts, self-matches")
    st.add_argument("--store", type=Path, required=True)
    b = sub.add_parser("batches", help="write the teacher's pending-request batch files")
    b.add_argument("--store", type=Path, required=True)
    b.add_argument("--size", type=int, default=10)
    b.add_argument("--out", type=Path, required=True)
    i = sub.add_parser("intake", help="validate and store the teacher's answers")
    i.add_argument("--store", type=Path, required=True)
    i.add_argument("--answers", type=Path, required=True)
    i.add_argument("--sample", type=Path, required=True)
    rs = sub.add_parser("review-sample", help="sample accepted answers for the reviewer")
    rs.add_argument("--store", type=Path, required=True)
    rs.add_argument("--fraction", type=float, default=0.05)
    rs.add_argument("--seed", type=int, default=20261017)
    rs.add_argument("--out", type=Path, required=True)
    rp = sub.add_parser("report", help="write the pilot report (markdown)")
    rp.add_argument("--store", type=Path, required=True)
    rp.add_argument("--batches", type=Path, required=True,
                    help="the batches directory (searched recursively: DIR/batches/ "
                         "with its round-N/ folders, or one round's folder)")
    rp.add_argument("--out", type=Path, default=None,
                    help="write the report here (default: print it)")
    ex = sub.add_parser("export", help="copy accepted keys' capture records")
    ex.add_argument("--store", type=Path, required=True)
    ex.add_argument("--out", type=Path, required=True)
    ex.add_argument("--allow-missing", action="store_true",
                    help="exit 0 even when some accepted keys have no capture record yet")
    return p


def _cmd_batches(args: argparse.Namespace) -> int:
    paths = write_batches(args.store, args.out, size=args.size)
    total = sum(len(_read_jsonl(p)) for p in paths)
    print(f"wrote {len(paths)} batch file(s) to {args.out}: {total} pending request(s)")
    return 0


def _cmd_intake(args: argparse.Namespace) -> int:
    summary = run_intake(args.store, args.answers, args.sample)
    print(f"intake: accepted {summary.accepted}, requeued {summary.requeued}, "
         f"dropped {summary.dropped}, skipped {summary.skipped}, "
         f"unanswered {summary.unanswered}, already rejected {summary.already_rejected}")
    for name in sorted(summary.by_prompt):
        counts = summary.by_prompt[name]
        print(f"  {name}: " + ", ".join(f"{o} {counts.get(o, 0)}" for o in INTAKE_OUTCOMES))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    text = report_text(args.store, args.batches)
    if args.out is None:
        print(text, end="")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def _cmd_review_sample(args: argparse.Namespace) -> int:
    path = write_review_sample(args.store, args.out, fraction=args.fraction, seed=args.seed)
    print(f"wrote {path}: {len(_read_jsonl(path))} answer(s) for review")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    summary = export_capture(args.store, args.out)
    print(f"exported {summary.exported} record(s) to {args.out} "
         f"(missing {summary.missing}); by prompt: {dict(summary.by_prompt)}; "
         f"by layout: {dict(summary.by_layout)}")
    if summary.missing and not args.allow_missing:
        print("hint: run one more `round` to replay accepted answers before export")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parser().parse_args(argv)
    if args.cmd == "status":
        print(status_text(args.store))
        return 0
    if args.cmd == "select":
        return asyncio.run(_cmd_select(args))
    if args.cmd == "batches":
        return _cmd_batches(args)
    if args.cmd == "intake":
        return _cmd_intake(args)
    if args.cmd == "review-sample":
        return _cmd_review_sample(args)
    if args.cmd == "report":
        return _cmd_report(args)
    if args.cmd == "export":
        return _cmd_export(args)
    return asyncio.run(_cmd_round(args))


# graphiti builds each entity's labels from a set ({"Entity", <type>}), so their order in
# the dedupe_nodes and extract_edges prompts follows the per-process string-hash seed.
# Answers are keyed on the prompt bytes: a round under another seed renders different
# bytes and can never replay them. Pin the seed, re-executing the CLI once if needed.
HASH_SEED = "0"


def ensure_fixed_hash_seed(environ: Any = None, execve: Any = None) -> None:
    environ = os.environ if environ is None else environ
    if environ.get("PYTHONHASHSEED") == HASH_SEED:
        return
    (execve or os.execve)(sys.executable, sys.orig_argv, {**environ, "PYTHONHASHSEED": HASH_SEED})


if __name__ == "__main__":
    ensure_fixed_hash_seed()
    sys.exit(main())
