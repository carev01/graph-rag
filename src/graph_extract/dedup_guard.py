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

What it deliberately does NOT do: rewrite, filter or reinterpret the model's
indices, and it never adds a per-call `maximum` to the schema. Under constrained
decoding a value bound would turn `[10, 11]` into an in-range token sequence such as
`[1, 1]` -- a wrong dedup that merges two distinct facts and is invisible afterwards.
A detectable drop is strictly better than an undetectable merge.
"""
from __future__ import annotations

import logging
import re
from contextvars import ContextVar
from dataclasses import dataclass, fields
from typing import Any

from graphiti_core.prompts.models import Message

logger = logging.getLogger(__name__)

DEDUP_PROMPT_NAME = "dedupe_edges.resolve_edge"


@dataclass
class DedupIndexStats:
    """Counters for one scope (an article, or a whole run when merged)."""
    calls: int = 0                      # dedup calls observed (incl. short-circuited)
    parse_failures: int = 0             # N/M not recoverable -> call not checked
    no_candidate_skips: int = 0         # zero candidates -> answered without an LLM call
    invalid_calls: int = 0              # calls with >=1 out-of-range index (either field)
    dup_in_invalidation_range: int = 0  # duplicate_facts entries in N..N+M-1
    dup_beyond_range: int = 0           # duplicate_facts entries > N+M-1 or < 0
    contradicted_beyond_range: int = 0  # contradicted_facts entries outside 0..N+M-1
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

    def stats(self) -> DedupIndexStats:
        return CURRENT_DEDUP_STATS.get() or self.unscoped


def install_dedup_guard(graphiti: Any, *, fallback: Any | None,
                        unscoped: DedupIndexStats) -> DedupIndexGuard:
    """Wrap `graphiti.llm_client.generate_response` in place (the same instance-
    attribute pattern as the chat.completions wrappers in graphiti_client.py).
    `graphiti` only needs an `llm_client` attribute."""
    client = graphiti.llm_client
    orig = client.generate_response
    guard = DedupIndexGuard(fallback=fallback, unscoped=unscoped, raw=_Raw(orig))

    async def generate_response(messages: list[Message], *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("prompt_name") != DEDUP_PROMPT_NAME:
            return await orig(messages, *args, **kwargs)
        stats = guard.stats()
        stats.calls += 1
        counts = parse_candidate_counts(messages)
        if counts is None:
            stats.parse_failures += 1
            logger.warning("dedup prompt not parseable (graphiti prompt format changed?); "
                           "index check skipped for this call")
            return await orig(messages, *args, **kwargs)
        n, m = counts
        if n == 0 and m == 0:
            # No duplicate candidates and no invalidation candidates: there is
            # nothing to be a duplicate OF. graphiti calls the LLM anyway
            # (edge_operations.resolve_extracted_edge issues it unconditionally)
            # and then DISCARDS the answer -- `duplicate_fact_ids` filters every
            # index against `len(related_edges) == 0`, and the contradiction block
            # is skipped by `if related_edges or existing_edges`. So the reply
            # cannot affect the outcome, and the only correct answer is the empty
            # one. This is exact, not an approximation.
            stats.no_candidate_skips += 1
            return {"duplicate_facts": [], "contradicted_facts": []}
        # graphiti's clients append schema/language text to the messages IN PLACE;
        # keep a pristine copy so the fallback sees the prompt as authored.
        pristine = [Message(role=msg.role, content=msg.content) for msg in messages]
        response = await orig(messages, *args, **kwargs)
        verdict = _classify(response, n, m)
        if verdict is None or verdict.clean:
            return response
        stats.invalid_calls += 1
        stats.dup_in_invalidation_range += len(verdict.dup_in_range)
        stats.dup_beyond_range += len(verdict.dup_beyond)
        stats.contradicted_beyond_range += len(verdict.contradicted_beyond)
        retry_clean: bool | None = None
        if guard.fallback is not None:
            stats.retried += 1
            # Not wrapped in try/except on purpose: a dead fallback tier must fail the
            # episode loudly, like any other LLM error in the pipeline.
            response = await guard.fallback.generate_response(pristine, *args, **kwargs)
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
        return response

    client.generate_response = generate_response
    return guard
