"""Out-of-range dedup indices (slice B, BACKLOG 4).

graphiti's `resolve_extracted_edge` asks the LLM which candidate facts the new fact
duplicates / contradicts, by `idx`. The idx space is continuous across two lists:
`related_edges` (duplicate candidates, idx 0..N-1) then `existing_edges`
(invalidation candidates, idx N..N+M-1). `duplicate_facts` is validated against
0..N-1 only; out-of-range entries are logged at WARNING and DROPPED.

Three sections:
  1. Library pins -- what graphiti 0.30.1 actually does with a bad index. These
     document the impact claims and trip on a graphiti upgrade; they are NOT tests
     of our fix and are not expected to fail with the guard removed.
  2. The prompt parser -- recovers N and M from the real prompt, with a contiguity
     check so a mis-parse is detectable rather than silent.
  3. The guard -- classifies out-of-range indices against the invalidation range
     (the decisive check for the index-space-confusion hypothesis), records them
     per article through a ContextVar, and optionally re-issues the call on the
     strong tier BEFORE graphiti sees the answer (so nothing has been written yet).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.prompts.lib import prompt_library
from graphiti_core.prompts.models import Message
from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge

from graph_extract.dedup_guard import (
    CURRENT_DEDUP_STATS,
    DEDUP_PROMPT_NAME,
    DedupIndexStats,
    install_dedup_guard,
    parse_candidate_counts,
)

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_T_OLD = datetime(2025, 1, 1, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- helpers

def _edge(fact: str, *, src: str = "s", dst: str = "t", valid_at: datetime | None = None) -> EntityEdge:
    return EntityEdge(group_id="g", source_node_uuid=src, target_node_uuid=dst,
                      created_at=_T0, name="RELATES_TO", fact=fact, valid_at=valid_at)


def _episode() -> EpisodicNode:
    return EpisodicNode(name="ep", group_id="g", source=EpisodeType.text,
                        source_description="d", content="c", valid_at=_T0)


def _dedup_messages(n: int, m: int, facts: list[str] | None = None) -> list[Message]:
    """Messages exactly as graphiti builds them (the REAL prompt function)."""
    related = [{"idx": i, "fact": (facts[i] if facts else f"related fact {i}")}
               for i in range(n)]
    existing = [{"idx": n + i, "fact": f"invalidation candidate {i}"} for i in range(m)]
    ctx = {"existing_edges": related, "edge_invalidation_candidates": existing,
           "new_edge": "the new fact"}
    return prompt_library.dedupe_edges.resolve_edge(ctx)


class _ScriptedLLM:
    """A graphiti-shaped LLM client: returns scripted dicts and, like the real
    clients, MUTATES the messages it is given (schema / language suffixes)."""

    def __init__(self, *responses: dict) -> None:
        self.responses = list(responses)
        self.seen: list[dict] = []          # kwargs of each call
        self.seen_content: list[list[str]] = []  # message content BEFORE mutation

    async def generate_response(self, messages, *args, **kwargs):
        self.seen.append(kwargs)
        self.seen_content.append([m.content for m in messages])
        messages[0].content += " <lang-suffix>"
        messages[-1].content += " <schema-suffix>"
        return self.responses.pop(0)


def _dedup_response(dup: list[int], contra: list[int]) -> dict:
    return {"duplicate_facts": dup, "contradicted_facts": contra}


async def _call(client, n: int, m: int, prompt_name: str = DEDUP_PROMPT_NAME):
    return await client.generate_response(
        _dedup_messages(n, m), response_model=None, model_size=ModelSize.small,
        prompt_name=prompt_name)


# ------------------------------------------------- 1. library pins (graphiti 0.30.1)

async def test_pin_graphiti_uses_the_first_VALID_duplicate_index():
    """Impact claim 1, first half: only the first valid index is ever used, so a
    dropped index is harmless as long as at least one valid index survives."""
    related = [_edge("a"), _edge("b"), _edge("c")]
    llm = _ScriptedLLM(_dedup_response([10, 1], []))
    resolved, _, _ = await resolve_extracted_edge(
        llm, _edge("new", valid_at=_T0), related, [], _episode())
    assert resolved is related[1]


async def test_pin_graphiti_drops_all_invalid_duplicates_and_keeps_the_new_edge():
    """Impact claim 1, second half: when EVERY index is invalid the edge is not
    deduplicated -- a duplicate fact enters the graph with its own episode."""
    related = [_edge("a"), _edge("b"), _edge("c")]
    new = _edge("new", valid_at=_T0)
    llm = _ScriptedLLM(_dedup_response([10, 11], []))
    resolved, _, _ = await resolve_extracted_edge(llm, new, related, [], _episode())
    assert resolved is new


async def test_pin_graphiti_drops_an_invalid_contradicted_index_silently():
    """Impact claim 2: a dropped contradicted index is a missed invalidation; the
    valid ones in the same reply are still honoured (no `break` here).

    Precondition pinned too: graphiti only invalidates a candidate whose valid_at
    is EARLIER than the new fact's (resolve_edge_contradictions); a candidate with
    no valid_at is never invalidated at ingest, dropped index or not."""
    related = [_edge("a", valid_at=_T_OLD), _edge("b", valid_at=_T_OLD),
               _edge("c", valid_at=_T_OLD)]
    existing = [_edge("x", src="p", dst="q", valid_at=_T_OLD),
                _edge("y", src="p", dst="q", valid_at=_T_OLD)]
    llm = _ScriptedLLM(_dedup_response([], [2, 4, 99]))
    _, invalidated, _ = await resolve_extracted_edge(
        llm, _edge("new", valid_at=_T0), related, existing, _episode())
    assert invalidated == [related[2], existing[1]]
    assert all(e.invalid_at == _T0 for e in invalidated)


async def test_pin_graphiti_never_invalidates_a_candidate_without_valid_at():
    related = [_edge("a"), _edge("b")]
    llm = _ScriptedLLM(_dedup_response([], [0, 1]))
    _, invalidated, _ = await resolve_extracted_edge(
        llm, _edge("new", valid_at=_T0), related, [], _episode())
    assert invalidated == []


async def test_pin_graphiti_accepts_a_duplicate_range_index_in_contradicted_facts():
    """The 'opposite direction' does not exist: contradicted_facts legally spans
    BOTH ranges, so the only invalid contradicted indices are hallucinated ones."""
    related = [_edge("a", valid_at=_T_OLD), _edge("b", valid_at=_T_OLD)]
    existing = [_edge("x", src="p", dst="q", valid_at=_T_OLD)]
    llm = _ScriptedLLM(_dedup_response([], [0]))
    _, invalidated, _ = await resolve_extracted_edge(
        llm, _edge("new", valid_at=_T0), related, existing, _episode())
    assert invalidated == [related[0]]


# ------------------------------------------------------------ 2. the prompt parser

def test_parse_counts_from_the_real_prompt_with_hostile_facts():
    """Tripwire on graphiti's prompt format. Facts contain quotes, braces, a
    newline and a partial idx-shaped fragment -- the parser must still recover the
    true counts (a FULL spoof is the separate contiguity test below)."""
    facts = ["it's \"quoted\"", "braces {'idx': 7} and 'fact': inside", "line1\nline2"]
    assert parse_candidate_counts(_dedup_messages(3, 2, facts)) == (3, 2)


def test_parse_counts_with_no_duplicate_candidates():
    """N may be 0 (graphiti still calls the LLM when only invalidation candidates
    exist); then every duplicate index is out of range."""
    assert parse_candidate_counts(_dedup_messages(0, 2)) == (0, 2)


def test_parse_counts_with_no_invalidation_candidates():
    assert parse_candidate_counts(_dedup_messages(4, 0)) == (4, 0)


def test_parse_returns_none_when_the_index_run_is_not_contiguous():
    """A fact that spoofs an idx entry breaks contiguity -> None, never a wrong N."""
    msgs = _dedup_messages(2, 1, ["ok", "spoof {'idx': 5, 'fact': 'z'}"])
    assert parse_candidate_counts(msgs) is None


def test_parse_returns_none_for_a_non_dedup_prompt():
    assert parse_candidate_counts([Message(role="user", content="no tags here")]) is None


# ------------------------------------------------------------------- 3. the guard

async def test_guard_ignores_prompts_other_than_dedup():
    primary = _ScriptedLLM({"edges": [{"episode_indices": [10, 11, 14, 15]}]})
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=None, unscoped=stats)
    await _call(primary, 10, 6, prompt_name="extract_edges.edge")
    assert stats.calls == 0 and stats.invalid_calls == 0


async def test_guard_classifies_duplicate_indices_inside_the_invalidation_range():
    """The decisive check for the confusion hypothesis: [10,11,14,15] with N=10, M=6
    all land in the invalidation range 10..15."""
    primary = _ScriptedLLM(_dedup_response([10, 11, 14, 15], []))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=None, unscoped=stats)
    out = await _call(primary, 10, 6)
    assert out == _dedup_response([10, 11, 14, 15], [])   # pass-through, no fallback
    assert stats.calls == 1
    assert stats.invalid_calls == 1
    assert stats.dup_in_invalidation_range == 4
    assert stats.dup_beyond_range == 0
    assert stats.contradicted_beyond_range == 0
    assert stats.retried == 0


async def test_guard_classifies_indices_beyond_both_ranges_and_negatives():
    primary = _ScriptedLLM(_dedup_response([3, 16, -1], [15, 16, -2]))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=None, unscoped=stats)
    await _call(primary, 10, 6)
    assert stats.invalid_calls == 1
    assert stats.dup_in_invalidation_range == 0
    assert stats.dup_beyond_range == 2          # 16 and -1
    assert stats.contradicted_beyond_range == 2  # 16 and -2 (15 is legal)


async def test_guard_counts_a_clean_call_without_flagging_it():
    primary = _ScriptedLLM(_dedup_response([1], [12]))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=None, unscoped=stats)
    await _call(primary, 10, 6)
    assert (stats.calls, stats.invalid_calls) == (1, 0)


async def test_guard_retries_on_the_fallback_and_returns_its_answer():
    primary = _ScriptedLLM(_dedup_response([10], []))
    fallback = _ScriptedLLM(_dedup_response([1], [12]))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=fallback, unscoped=stats)
    out = await _call(primary, 10, 6)
    assert out == _dedup_response([1], [12])
    assert stats.retried == 1 and stats.retry_clean == 1
    # same call shape forwarded
    assert fallback.seen[0]["prompt_name"] == DEDUP_PROMPT_NAME
    assert fallback.seen[0]["model_size"] is ModelSize.small


async def test_guard_hands_the_fallback_UNMUTATED_messages():
    """graphiti's clients append schema/language text to the messages in place; the
    fallback must see the original prompt, not the cheap tier's mutated copy."""
    primary = _ScriptedLLM(_dedup_response([10], []))
    fallback = _ScriptedLLM(_dedup_response([], []))
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=fallback,
                        unscoped=DedupIndexStats())
    await _call(primary, 10, 6)
    original = [m.content for m in _dedup_messages(10, 6)]
    assert fallback.seen_content[0] == original
    assert primary.seen_content[0] == original


async def test_guard_counts_a_retry_that_is_also_dirty_and_still_uses_it():
    primary = _ScriptedLLM(_dedup_response([10], []))
    fallback = _ScriptedLLM(_dedup_response([12], []))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=fallback, unscoped=stats)
    out = await _call(primary, 10, 6)
    assert out == _dedup_response([12], [])
    assert (stats.retried, stats.retry_clean, stats.retry_dirty) == (1, 0, 1)


async def test_guard_does_not_retry_a_clean_call():
    primary = _ScriptedLLM(_dedup_response([2], []))
    fallback = _ScriptedLLM()
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=fallback, unscoped=stats)
    await _call(primary, 10, 6)
    assert fallback.seen == [] and stats.retried == 0


async def test_guard_records_a_parse_failure_and_passes_through_without_retrying():
    """When N/M cannot be recovered the guard cannot judge the reply: it must not
    guess, must not retry, and must leave graphiti's own validation to run."""
    primary = _ScriptedLLM(_dedup_response([10], []))
    fallback = _ScriptedLLM()
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=fallback, unscoped=stats)
    msgs = _dedup_messages(2, 1, ["ok", "spoof {'idx': 5, 'fact': 'z'}"])
    out = await primary.generate_response(msgs, prompt_name=DEDUP_PROMPT_NAME)
    assert out == _dedup_response([10], [])
    assert stats.parse_failures == 1 and stats.retried == 0 and fallback.seen == []


async def test_guard_fallback_errors_propagate():
    """Never swallow the retry's failure: a dead strong tier must fail the episode
    loudly, exactly like any other LLM error in the pipeline."""
    class _Boom:
        async def generate_response(self, *a, **k):
            raise RuntimeError("strong tier down")

    primary = _ScriptedLLM(_dedup_response([10], []))
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=_Boom(),
                        unscoped=DedupIndexStats())
    with pytest.raises(RuntimeError, match="strong tier down"):
        await _call(primary, 10, 6)


async def test_guard_records_into_the_context_scoped_stats_when_set():
    primary = _ScriptedLLM(_dedup_response([10], []), _dedup_response([11], []))
    unscoped = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=None, unscoped=unscoped)
    scoped = DedupIndexStats()
    token = CURRENT_DEDUP_STATS.set(scoped)
    try:
        await _call(primary, 10, 6)
    finally:
        CURRENT_DEDUP_STATS.reset(token)
    await _call(primary, 10, 6)
    assert (scoped.invalid_calls, unscoped.invalid_calls) == (1, 1)


async def test_guard_raw_bypasses_the_guard_so_a_retry_is_not_double_counted():
    """The strong tier is guarded for detection too; the cheap tier's fallback must
    be the strong client's UNGUARDED method or every retry would count twice."""
    strong = _ScriptedLLM(_dedup_response([10], []))
    strong_stats = DedupIndexStats()
    guard = install_dedup_guard(SimpleNamespace(llm_client=strong), fallback=None,
                                unscoped=strong_stats)
    cheap = _ScriptedLLM(_dedup_response([11], []))
    cheap_stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=cheap), fallback=guard.raw,
                        unscoped=cheap_stats)
    await _call(cheap, 10, 6)
    assert (cheap_stats.retried, cheap_stats.retry_dirty) == (1, 1)
    assert strong_stats.calls == 0


async def test_guard_logs_each_event_with_structured_args(caplog):
    primary = _ScriptedLLM(_dedup_response([10, 11], [40]))
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=None,
                        unscoped=DedupIndexStats())
    with caplog.at_level(logging.WARNING, logger="graph_extract.dedup_guard"):
        await _call(primary, 10, 6)
    recs = [r for r in caplog.records if r.name == "graph_extract.dedup_guard"]
    assert len(recs) == 1
    msg = recs[0].getMessage()
    assert "duplicate_facts=[10, 11]" in msg and "contradicted_facts=[40]" in msg
    assert "related=10" in msg and "existing=6" in msg


async def test_guard_recovers_dedup_inside_graphiti_s_real_pipeline():
    """End to end through the real resolve_extracted_edge: the cheap tier answers
    out of range, the guard re-asks the strong tier, graphiti dedups correctly.
    Compare with the library pin above where the same cheap reply lost the dedup."""
    related = [_edge("a"), _edge("b"), _edge("c")]
    new = _edge("new", valid_at=_T0)
    cheap = _ScriptedLLM(_dedup_response([10, 11], []))
    strong = _ScriptedLLM(_dedup_response([1], []))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=cheap), fallback=strong, unscoped=stats)
    resolved, _, _ = await resolve_extracted_edge(cheap, new, related, [], _episode())
    assert resolved is related[1]
    assert stats.retried == 1


def test_stats_merge_sums_every_counter():
    a = DedupIndexStats(calls=2, invalid_calls=1, dup_in_invalidation_range=3,
                        dup_beyond_range=1, contradicted_beyond_range=1,
                        parse_failures=1, retried=1, retry_clean=1, retry_dirty=0)
    b = DedupIndexStats(calls=1, invalid_calls=1, dup_in_invalidation_range=0,
                        dup_beyond_range=2, contradicted_beyond_range=0,
                        parse_failures=0, retried=1, retry_clean=0, retry_dirty=1)
    a.merge(b)
    assert (a.calls, a.invalid_calls, a.dup_in_invalidation_range, a.dup_beyond_range,
            a.contradicted_beyond_range, a.parse_failures, a.retried, a.retry_clean,
            a.retry_dirty) == (3, 2, 3, 3, 1, 1, 2, 1, 1)


# --- graphiti's own zero-candidate early return ------------------------------
# NOT our code: `resolve_extracted_edge` returns at line 653 when both candidate
# lists are empty, BEFORE the dedup prompt is built. A short-circuit in our
# wrapper for that case is unreachable, and was removed. This pin fails if the
# library ever drops the early return, which would make the case worth handling.

def test_graphiti_returns_early_when_there_are_no_candidates():
    import inspect
    from graphiti_core.utils.maintenance import edge_operations
    src = inspect.getsource(edge_operations.resolve_extracted_edge)
    guard = "if len(related_edges) == 0 and len(existing_edges) == 0:"
    assert guard in src, "graphiti no longer short-circuits the no-candidate case"
    # and it returns before ever naming the dedup prompt
    assert src.index(guard) < src.index("dedupe_edges.resolve_edge")


def test_graphiti_resolves_verbatim_duplicates_without_an_llm_call():
    """The brief's 'option 3' already exists in the library: a normalised exact
    fact+endpoints match returns before the dedup call."""
    import inspect
    from graphiti_core.utils.maintenance import edge_operations
    src = inspect.getsource(edge_operations.resolve_extracted_edge)
    assert "_normalize_string_exact" in src
    assert src.index("_normalize_string_exact") < src.index("dedupe_edges.resolve_edge")


# --- BACKLOG 33: the residual same-pair invalidation path --------------------
# The contradiction gate removes the cross-pair candidates by skipping the
# unfiltered search. It does NOT stop `contradicted_facts` indices in 0..N-1,
# which graphiti routes into invalidation_candidates from the DUPLICATE list
# (edge_operations.py:769-776). Those indices are perfectly in range, so the call
# carrying them is "clean" -- the counter must not live behind the invalid-call
# branch.

def _messages(n_related: int, n_invalidation: int):
    return prompt_library.dedupe_edges.resolve_edge({
        "existing_edges": [{"idx": i, "fact": f"r{i}"} for i in range(n_related)],
        "edge_invalidation_candidates": [
            {"idx": n_related + i, "fact": f"e{i}"} for i in range(n_invalidation)],
        "new_edge": "a new fact"})


@pytest.mark.asyncio
async def test_a_same_pair_contradiction_is_counted_on_an_otherwise_clean_call():
    async def _reply(messages, *a, **kw):
        return {"duplicate_facts": [], "contradicted_facts": [0, 2]}

    stats = DedupIndexStats()
    primary = SimpleNamespace(generate_response=_reply)
    holder = SimpleNamespace(llm_client=primary)
    install_dedup_guard(holder, fallback=None, unscoped=stats)
    token = CURRENT_DEDUP_STATS.set(stats)
    try:
        await holder.llm_client.generate_response(
            _messages(5, 0), prompt_name=DEDUP_PROMPT_NAME)
    finally:
        CURRENT_DEDUP_STATS.reset(token)
    assert stats.contradicted_same_pair == 2
    assert stats.invalid_calls == 0, "in-range indices are not an out-of-range event"
    assert stats.contradicted_beyond_range == 0


@pytest.mark.asyncio
async def test_cross_pair_contradictions_are_not_counted_as_same_pair():
    """Indices >= N point at the invalidation list, which the gate removes. They
    must not inflate the residual-path number."""
    async def _reply(messages, *a, **kw):
        return {"duplicate_facts": [], "contradicted_facts": [5, 6]}

    stats = DedupIndexStats()
    holder = SimpleNamespace(llm_client=SimpleNamespace(generate_response=_reply))
    install_dedup_guard(holder, fallback=None, unscoped=stats)
    token = CURRENT_DEDUP_STATS.set(stats)
    try:
        await holder.llm_client.generate_response(
            _messages(5, 3), prompt_name=DEDUP_PROMPT_NAME)
    finally:
        CURRENT_DEDUP_STATS.reset(token)
    assert stats.contradicted_same_pair == 0
    assert stats.contradicted_beyond_range == 0


# --- D4: suppress same-pair contradictions (pre-bootstrap-decisions-2026-09-23) --

async def test_suppression_clears_contradictions_and_keeps_duplicates():
    primary = _ScriptedLLM(_dedup_response([1], [0, 2]))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=None, unscoped=stats,
                        suppress_contradictions=True)
    reply = await _call(primary, 3, 0)
    assert reply["contradicted_facts"] == []
    assert reply["duplicate_facts"] == [1]
    assert stats.contradicted_same_pair == 2      # counted BEFORE suppression
    assert stats.contradictions_suppressed == 2


async def test_no_suppression_by_default():
    primary = _ScriptedLLM(_dedup_response([], [0]))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=None, unscoped=stats)
    reply = await _call(primary, 3, 0)
    assert reply["contradicted_facts"] == [0]
    assert stats.contradictions_suppressed == 0


async def test_suppression_applies_on_the_parse_failure_path():
    primary = _ScriptedLLM(_dedup_response([], [0]))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=None, unscoped=stats,
                        suppress_contradictions=True)
    reply = await primary.generate_response(
        [Message(role="user", content="not a parseable dedup prompt")],
        prompt_name=DEDUP_PROMPT_NAME)
    assert reply["contradicted_facts"] == []
    assert stats.parse_failures == 1 and stats.contradictions_suppressed == 1


async def test_suppression_applies_to_the_fallback_reply():
    primary = _ScriptedLLM(_dedup_response([15], [0]))       # out of range -> retry
    fallback = _ScriptedLLM(_dedup_response([], [1]))
    stats = DedupIndexStats()
    install_dedup_guard(SimpleNamespace(llm_client=primary), fallback=fallback,
                        unscoped=stats, suppress_contradictions=True)
    reply = await _call(primary, 3, 0)
    assert reply["contradicted_facts"] == []
    assert stats.retried == 1 and stats.contradictions_suppressed == 1


async def test_suppressed_same_pair_contradiction_invalidates_nothing_end_to_end():
    """Through graphiti's real resolve_extracted_edge: the related edge is NOT
    invalidated and the new edge is NOT expired, although the model said 'contradicts'."""
    related = [_edge("a", valid_at=_T_OLD), _edge("b", valid_at=_T_OLD)]
    existing = [_edge("x", src="p", dst="q", valid_at=_T_OLD)]
    llm = _ScriptedLLM(_dedup_response([], [0]))
    install_dedup_guard(SimpleNamespace(llm_client=llm), fallback=None,
                        unscoped=DedupIndexStats(), suppress_contradictions=True)
    resolved, invalidated, _ = await resolve_extracted_edge(
        llm, _edge("new", valid_at=_T0), related, existing, _episode())
    assert invalidated == []
    assert resolved.expired_at is None
