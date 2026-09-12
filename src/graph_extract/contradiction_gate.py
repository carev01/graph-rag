"""Suspend graphiti's contradiction detection, and the O(corpus) scan feeding it.

`resolve_extracted_edges` issues TWO searches per extracted fact, and they are not
equivalent. Measured with PROFILE on the live graph:

    duplicate candidates   SearchFilters(edge_uuids=[...])
        -> DirectedRelationshipIndexSeek, 10 rows scored. Bounded by the candidate
           list; does NOT grow with the corpus.

    invalidation candidates  SearchFilters()
        -> NodeByLabelScan + Expand(All), 3,469 rows scored. O(facts):
           scan_ms = 241 + 0.0699 x facts, i.e. 42.9 s at the projected 610k-fact
           corpus, overtaking the 1.8 s dedup LLM call at ~3,900 articles -- under
           4% of the corpus.

That scan exists solely to feed contradiction detection, which does not currently
work: every one of the 140 ingest-time invalidations measured was contradiction
driven, and the inspectable ones are refinements and near-duplicates, ordered by
the sequence DocExtractor happened to crawl the pages. There is no document
revision date in the corpus to order them properly
(docs/proposals/2026-09-12-docextractor-article-timestamps.md).

So the scan is the scale blocker AND it serves the one feature that is broken.
Suspending it removes both, plus the dedup index-space confusion: with no
invalidation candidates the dedup prompt carries ONE index range instead of two,
and all 267 observed out-of-range indices landed in the second one.

Scope: `edge_operations` is ingest-only. The answer path searches via
`graphiti.search()` / `search_utils`, so retrieval is untouched by construction
rather than by a flag someone must remember to check.

Re-enablement is NOT simply flipping the flag -- see
docs/superpowers/specs/2026-09-12-suspend-contradiction-detection-design.md section 7.
"""
from __future__ import annotations

import logging
from typing import Any

from graphiti_core.search.search_config import SearchResults
from graphiti_core.utils.maintenance import edge_operations

logger = logging.getLogger(__name__)

_MARKER = "_graph_extract_contradiction_gate"


def install_contradiction_gate(*, detect_contradictions: bool) -> bool:
    """Patch `edge_operations`' own `search` reference so the unfiltered
    invalidation search never runs.

    `edge_operations` does `from graphiti_core.search.search import search`, so the
    patch must target THAT module's attribute; patching the defining module would
    not be seen.

    Returns True when the gate is installed (contradiction detection suspended),
    False when detection is left enabled. Idempotent.
    """
    if detect_contradictions:
        return False
    if getattr(edge_operations.search, _MARKER, False):
        return True

    original = edge_operations.search

    async def gated(*args: Any, **kwargs: Any) -> Any:
        # `search_filter` is the 5th positional parameter and is passed as a
        # keyword by both call sites today. Read both, so a change in graphiti's
        # call style degrades to "still skipped" rather than silently resuming a
        # corpus scan.
        search_filter = kwargs.get("search_filter")
        if search_filter is None and len(args) >= 5:
            search_filter = args[4]
        if search_filter is not None and getattr(search_filter, "edge_uuids", None) is None:
            # The invalidation search. Empty, well-formed -- never None, which
            # would fail confusingly deep inside graphiti.
            return SearchResults()
        return await original(*args, **kwargs)

    gated.__dict__[_MARKER] = True
    edge_operations.search = gated
    logger.info(
        "contradiction detection suspended: the unfiltered invalidation-candidate "
        "search is skipped (scale blocker; see BACKLOG 6/8/30)")
    return True


def is_gate_installed() -> bool:
    """True when the invalidation-candidate search is currently being skipped."""
    return bool(getattr(edge_operations.search, _MARKER, False))
