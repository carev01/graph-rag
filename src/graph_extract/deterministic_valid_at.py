"""Set a fact's `valid_at` from its episode's reference time, with no LLM call.

`content_changed_at` is the moment the SERVED markdown became current -- the bytes
`content_hash` covers and our re-ingest gate keys on. `IngestDriver` already makes
it the episode's reference time, so it is the right `valid_at` for every fact that
episode produced. This module makes that actually happen.

**Why the library's version is the wrong tool here.** graphiti derives `valid_at`
with a per-fact LLM call (`extract_edges.extract_timestamps`) that reads the fact
text and resolves relative expressions against the reference time. Measured on the
2026-09-14 pilot re-ingest (`docs/superpowers/pilot-reingest-2026-09-14.md`):

  * it dated **817 of 3,590 facts (23%)** -- three quarters of the graph got
    nothing, so there was no usable ordering axis at all;
  * it cost **21.7% of all LLM time, 5,476 s of a 6.9 h run** -- the second
    largest component in the pipeline;
  * and it returned 9 strings it could not itself parse.

Worse, the dates it *did* produce were semantically mixed: an in-world date read
out of the fact text ("supported since 2019") on roughly half, the reference time
on the rest. Comparing those two kinds is what produced this project's phantom
invalidations. A field holding two kinds of date is worse than a field holding the
less precise one.

**What is given up.** That call also extracts in-text END dates ("deprecated in
2024") into `invalid_at` -- but only for edges the combined extraction prompt left
undated, since that prompt already sets `invalid_at` itself (and such an edge takes
the early return below). We lose the fallback's end dates. It is a real cost,
accepted because
`invalid_at` should record *evidence of supersession* -- a newer document, a
tombstone, the staleness sweep -- rather than a model's reading of prose, and
because the same call's `valid_at` half was measured unusable.

**What is preserved.** Both of the original's early returns: an edge that already
carries either timestamp is untouched (graphiti's combined-extraction path sets
them from the extraction prompt), and an episode with no reference time yields an
undated edge. Inventing `now()` there would be the crawl-clock defect over again.
"""
from __future__ import annotations

import logging
from typing import Any

from graphiti_core.utils.maintenance import edge_operations

logger = logging.getLogger(__name__)

_MARKER = "_graph_extract_deterministic_valid_at"
_ORIGINAL = "_graph_extract_original_extract_timestamps"


def install_deterministic_valid_at(*, enabled: bool) -> bool:
    """Replace `edge_operations._extract_edge_timestamps` with the deterministic
    assignment. Both call sites live in that module and resolve the name through
    module globals at call time, so patching the module attribute reaches them.

    Returns True when installed. Idempotent, and calling it with `enabled=False`
    restores graphiti's own function if ours is in place.
    """
    current = edge_operations._extract_edge_timestamps
    if not enabled:
        original = getattr(current, _ORIGINAL, None)
        if original is not None:
            edge_operations._extract_edge_timestamps = original
            logger.info("deterministic valid_at uninstalled; graphiti's LLM "
                        "timestamp extraction is active again")
        return False
    if getattr(current, _MARKER, False):
        return True

    async def _deterministic(llm_client: Any, edge: Any, episode: Any) -> None:
        # Same early returns as the function being replaced, for the same reasons.
        if edge.valid_at is not None or edge.invalid_at is not None:
            return
        if episode is None or episode.valid_at is None:
            return
        edge.valid_at = episode.valid_at

    _deterministic.__dict__[_MARKER] = True
    _deterministic.__dict__[_ORIGINAL] = current
    edge_operations._extract_edge_timestamps = _deterministic
    logger.info(
        "valid_at is now the episode reference time (content_changed_at); "
        "graphiti's per-fact timestamp LLM call is skipped")
    return True


def is_installed() -> bool:
    return bool(getattr(edge_operations._extract_edge_timestamps, _MARKER, False))
