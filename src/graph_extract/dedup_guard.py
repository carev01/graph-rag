"""Guard graphiti's edge-dedup LLM call against out-of-range candidate indices.

The mechanism (graphiti_core/utils/maintenance/edge_operations.py,
`resolve_extracted_edge`, 0.30.1): for each newly extracted fact graphiti sends ONE
prompt with two candidate lists numbered continuously -- `related_edges` (duplicate
candidates, idx 0..N-1, same endpoints as the new fact) then `existing_edges`
(invalidation candidates, idx N..N+M-1, any endpoints). The model answers with two
index lists. `duplicate_facts` is validated against 0..N-1 only, `contradicted_facts`
against 0..N+M-1; anything outside is logged at WARNING and DROPPED:

  * duplicates -- only the FIRST valid index is used, so a drop costs nothing unless
    EVERY index is invalid; then the edge is not deduplicated and a duplicate fact
    enters the graph with its own episode, permanently.
  * contradictions -- every valid index is used, so each dropped one is a missed
    invalidation: a superseded fact stays current (design decision #3 territory).

Both N and M are capped by graphiti's DEFAULT_SEARCH_LIMIT (10), so a valid reply
never contains an index above 19.

What this module does, per dedup call, BEFORE graphiti sees the reply (nothing has
been written to Neo4j at that point, so a retry has no side effects):

  1. recover N and M from the prompt itself (`parse_candidate_counts`) -- with a
     contiguity check, so a mis-parse yields None rather than a wrong N;
  2. classify every out-of-range index: inside the invalidation range (the
     index-space-confusion signature) or beyond it / negative (hallucinated);
  3. record the counts into the per-article `DedupIndexStats` published through
     `CURRENT_DEDUP_STATS` (IngestDriver opens one scope per article);
  4. optionally re-issue the SAME prompt on a fallback client (the strong tier)
     and hand graphiti that reply instead.
  5. when suppress_contradictions is set (D4; ingest_same_pair_contradictions=False,
     the default), clear the WHOLE contradicted_facts list after counting --
     same-pair indices (0..N-1) and cross-pair ones (N..N+M-1) alike, so no
     ingest-time contradiction invalidation of any kind survives. See
     pre-bootstrap-decisions-2026-09-23.md and the config.py comments.

What it deliberately does NOT do: rewrite, filter or reinterpret the model's
DUPLICATE indices, and it never adds a per-call `maximum` to the schema. Under constrained
decoding a value bound would turn `[10, 11]` into an in-range token sequence such as
`[1, 1]` -- a wrong dedup that merges two distinct facts and is invisible afterwards.
A detectable drop is strictly better than an undetectable merge.
"""
from __future__ import annotations

import logging
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from typing import Any

from graphiti_core.prompts.models import Message

from graph_extract.llm_timing import PromptTimings
from graph_extract.usage import CURRENT_PROMPT_NAME

logger = logging.getLogger(__name__)

DEDUP_PROMPT_NAME = "dedupe_edges.resolve_edge"


@dataclass
class DedupIndexStats:
    """Counters for one scope (an article, or a whole run when merged)."""
    calls: int = 0                      # dedup LLM calls observed
    parse_failures: int = 0             # N/M not recoverable -> call not checked
    invalid_calls: int = 0              # calls with >=1 out-of-range index (either field)
    dup_in_invalidation_range: int = 0  # duplicate_facts entries in N..N+M-1
    dup_beyond_range: int = 0           # duplicate_facts entries > N+M-1 or < 0
    contradicted_beyond_range: int = 0  # contradicted_facts entries outside 0..N+M-1
    # contradicted_facts entries in 0..N-1 -- indices into the DUPLICATE candidate
    # list. graphiti routes these to invalidation_candidates too
    # (edge_operations.py:769-776), so they survive the contradiction gate, which
    # only removes the cross-pair candidates. This is the residual same-pair
    # invalidation path (BACKLOG 33) and it is counted on EVERY call, not just
    # out-of-range ones -- a same-pair contradiction is perfectly in range.
    # Counted from the PRIMARY reply only: when a dirty call is retried on the
    # fallback, the fallback reply's same-pair indices are NOT added here.
    contradicted_same_pair: int = 0
    # contradicted_facts entries cleared before graphiti saw them (D4 suppression,
    # ingest_same_pair_contradictions=False). ALL entries, not only same-pair ones:
    # cross-pair and out-of-range indices are cleared and counted too. Counted from
    # the reply graphiti would actually have received -- on the retry path that is
    # the FALLBACK reply, so on retried calls this and contradicted_same_pair
    # describe different replies and must not be subtracted from each other.
    contradictions_suppressed: int = 0
    retried: int = 0                    # invalid calls re-issued on the fallback
    retry_clean: int = 0                # ... whose fallback reply was in range
    retry_dirty: int = 0                # ... whose fallback reply was ALSO out of range

    def merge(self, other: DedupIndexStats) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def summary(self) -> str:
        return " ".join(f"{f.name}={getattr(self, f.name)}" for f in fields(self))


# Set by IngestDriver for the duration of one article; read by the guard. A
# ContextVar (not a shared attribute) so attribution stays correct even if articles
# are ever processed concurrently -- add_episode's inner tasks inherit the context.
CURRENT_DEDUP_STATS: ContextVar[DedupIndexStats | None] = ContextVar(
    "current_dedup_stats", default=None)

_RELATED_TAGS = ("<EXISTING FACTS>", "</EXISTING FACTS>")
_EXISTING_TAGS = ("<FACT INVALIDATION CANDIDATES>", "</FACT INVALIDATION CANDIDATES>")
# graphiti renders each candidate list with Python's repr: [{'idx': 0, 'fact': '...'}, ...]
_IDX = re.compile(r"\{'idx': (\d+), 'fact': ")


def _block_indices(text: str, tags: tuple[str, str]) -> list[int] | None:
    start = text.find(tags[0])
    if start < 0:
        return None
    end = text.find(tags[1], start)
    if end < 0:
        return None
    return [int(x) for x in _IDX.findall(text[start:end])]


def parse_candidate_counts(messages: list[Message]) -> tuple[int, int] | None:
    """(N, M) = (len(related_edges), len(existing_edges)) as graphiti numbered them.

    Coupled to graphiti's prompt text (prompts/dedupe_edges.py) -- the tripwire
    test builds messages with the real prompt function. Returns None unless both
    blocks are present AND the idx runs are exactly 0..N-1 and N..N+M-1, so a fact
    that happens to contain an idx-shaped string cannot produce a wrong count.
    """
    text = "\n".join(m.content for m in messages)
    related = _block_indices(text, _RELATED_TAGS)
    existing = _block_indices(text, _EXISTING_TAGS)
    if related is None or existing is None:
        return None
    n, m = len(related), len(existing)
    if related != list(range(n)) or existing != list(range(n, n + m)):
        return None
    return n, m


def _int_list(value: object) -> list[int] | None:
    if isinstance(value, list) and all(
            isinstance(v, int) and not isinstance(v, bool) for v in value):
        return value
    return None


@dataclass
class _Verdict:
    dup_in_range: list[int]        # duplicate indices that fell in the invalidation range
    dup_beyond: list[int]          # duplicate indices beyond both ranges, or negative
    contradicted_beyond: list[int]
    contradicted_same_pair: list[int]   # contradicted indices in 0..N-1 (BACKLOG 33)

    @property
    def clean(self) -> bool:
        return not (self.dup_in_range or self.dup_beyond or self.contradicted_beyond)

    @property
    def invalid_duplicates(self) -> list[int]:
        return self.dup_in_range + self.dup_beyond


def _classify(response: dict[str, Any], n: int, m: int) -> _Verdict | None:
    """None when the reply is not even shaped like EdgeDuplicate -- graphiti's own
    `EdgeDuplicate(**response)` will raise on it, which is the right outcome."""
    dup = _int_list(response.get("duplicate_facts"))
    contra = _int_list(response.get("contradicted_facts"))
    if dup is None or contra is None:
        return None
    top = n + m - 1
    return _Verdict(
        dup_in_range=[i for i in dup if n <= i <= top],
        dup_beyond=[i for i in dup if i < 0 or i > top],
        contradicted_beyond=[i for i in contra if i < 0 or i > top],
        contradicted_same_pair=[i for i in contra if 0 <= i < n],
    )


class _Raw:
    """The unguarded `generate_response` of a guarded client, in the object-with-
    method shape graphiti and this module expect. Used as another tier's fallback
    so a retry is not double-counted by the strong tier's own guard."""

    def __init__(self, generate_response: Any) -> None:
        self.generate_response = generate_response


@dataclass
class DedupIndexGuard:
    fallback: Any | None      # object with async generate_response(messages, *a, **kw)
    unscoped: DedupIndexStats  # sink when no article scope is open
    raw: _Raw
    # Every LLM call this client makes, keyed by prompt_name -- not just dedup.
    # `_extract_edge_timestamps` issues a second call per new fact that nothing
    # counted before (BACKLOG 31).
    timings: PromptTimings = field(default_factory=PromptTimings)

    def stats(self) -> DedupIndexStats:
        return CURRENT_DEDUP_STATS.get() or self.unscoped


def install_dedup_guard(graphiti: Any, *, fallback: Any | None,
                        unscoped: DedupIndexStats,
                        timings: PromptTimings | None = None,
                        suppress_contradictions: bool = False) -> DedupIndexGuard:
    """Wrap `graphiti.llm_client.generate_response` in place (the same instance-
    attribute pattern as the chat.completions wrappers in graphiti_client.py).
    `graphiti` only needs an `llm_client` attribute."""
    client = graphiti.llm_client
    orig = client.generate_response
    guard = DedupIndexGuard(fallback=fallback, unscoped=unscoped, raw=_Raw(orig),
                            timings=timings or PromptTimings())

    def _suppressed(response: Any, stats: DedupIndexStats) -> Any:
        """Clear the WHOLE contradicted_facts list -- same-pair and cross-pair
        indices alike, despite the setting's "same-pair" name -- so graphiti
        invalidates nothing on this fact: neither the old edge
        (resolve_edge_contradictions) nor the new one (the inline later-valid_at
        block). duplicate_facts is never touched."""
        if not suppress_contradictions:
            return response
        if not isinstance(response, dict):
            raise TypeError(
                f"dedup reply is {type(response).__name__}, not dict; cannot suppress "
                "contradictions (graphiti's generate_response contract changed?)")
        contra = response.get("contradicted_facts") or []
        if not contra:
            return response
        stats.contradictions_suppressed += len(contra)
        return {**response, "contradicted_facts": []}

    async def _timed(fn, label: str, messages, *args: Any, **kwargs: Any) -> Any:
        """Time one call and record the in-flight count AT DISPATCH. The pairing
        is what tests whether the provider actually runs our concurrent calls in
        parallel -- flat latency across in-flight buckets means it does."""
        at = guard.timings.start()
        t0 = time.perf_counter()
        try:
            return await fn(messages, *args, **kwargs)
        finally:
            guard.timings.finish(label, (time.perf_counter() - t0) * 1000.0, at)

    async def generate_response(messages: list[Message], *args: Any, **kwargs: Any) -> Any:
        name = kwargs.get("prompt_name") or "unknown"
        # Published for the duration of this call so usage.py's capture (several
        # calls deeper, at the raw chat.completions.create layer) can tag its record
        # with the prompt that produced it -- see CURRENT_PROMPT_NAME in usage.py for
        # why this must be a ContextVar reset in a finally, not a module global.
        token = CURRENT_PROMPT_NAME.set(name)
        try:
            if name != DEDUP_PROMPT_NAME:
                return await _timed(orig, name, messages, *args, **kwargs)
            stats = guard.stats()
            stats.calls += 1
            counts = parse_candidate_counts(messages)
            if counts is None:
                stats.parse_failures += 1
                logger.warning("dedup prompt not parseable (graphiti prompt format changed?); "
                               "index check skipped for this call")
                return _suppressed(
                    await _timed(orig, DEDUP_PROMPT_NAME, messages, *args, **kwargs), stats)
            n, m = counts
            # graphiti's clients append schema/language text to the messages IN PLACE;
            # keep a pristine copy so the fallback sees the prompt as authored.
            pristine = [Message(role=msg.role, content=msg.content) for msg in messages]
            response = await _timed(orig, DEDUP_PROMPT_NAME, messages, *args, **kwargs)
            verdict = _classify(response, n, m)
            if verdict is not None:
                # Recorded BEFORE the clean-path return: a same-pair contradiction is
                # an in-range index, so the call it arrives on is usually "clean" and
                # would never reach the counters below.
                stats.contradicted_same_pair += len(verdict.contradicted_same_pair)
            if verdict is None or verdict.clean:
                return _suppressed(response, stats)
            stats.invalid_calls += 1
            stats.dup_in_invalidation_range += len(verdict.dup_in_range)
            stats.dup_beyond_range += len(verdict.dup_beyond)
            stats.contradicted_beyond_range += len(verdict.contradicted_beyond)
            retry_clean: bool | None = None
            if guard.fallback is not None:
                stats.retried += 1
                # Not wrapped in try/except on purpose: a dead fallback tier must fail the
                # episode loudly, like any other LLM error in the pipeline.
                response = await _timed(guard.fallback.generate_response,
                                        f"{DEDUP_PROMPT_NAME}:retry", pristine, *args, **kwargs)
                v2 = _classify(response, n, m)
                retry_clean = v2 is not None and v2.clean
                if retry_clean:
                    stats.retry_clean += 1
                else:
                    stats.retry_dirty += 1
            logger.warning(
                "dedup indices out of range: duplicate_facts=%s contradicted_facts=%s "
                "related=%d existing=%d (invalidation idx %d-%d) in_invalidation_range=%d "
                "beyond=%d retried=%s retry_clean=%s",
                verdict.invalid_duplicates, verdict.contradicted_beyond, n, m, n, n + m - 1,
                len(verdict.dup_in_range), len(verdict.dup_beyond) + len(verdict.contradicted_beyond),
                guard.fallback is not None, retry_clean)
            return _suppressed(response, stats)
        finally:
            CURRENT_PROMPT_NAME.reset(token)

    client.generate_response = generate_response
    return guard
