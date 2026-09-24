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

So the scan is the scale blocker AND it serves the feature that is broken.
Suspending it removes the scan, and with it CROSS-PAIR invalidation and the dedup
index-space confusion: with no invalidation candidates the dedup prompt carries
ONE index range instead of two, and all 267 observed out-of-range indices landed
in the second one.

What it does NOT remove -- read this before claiming otherwise (BACKLOG 33):
SAME-PAIR contradiction stays live. `resolve_extracted_edge` routes
`contradicted_facts` indices below `len(related_edges)` into
`invalidation_candidates`, and `related_edges` comes from the FILTERED duplicate
search this gate deliberately delegates. graphiti's dedup prompt invites exactly
that ("idx values from EITHER list"; its worked example returns a contradiction
on a same-pair refinement). A same-pair candidate with a later `valid_at` can
also mark the NEW edge invalid on arrival. The six hand-inspected bad
invalidations that justified this change were all same-endpoint, same-relation
pairs -- that is this path, not the one suspended here. Its share is unmeasured.

Since 2026-09-23 that path is suppressed too, by default, but NOT here:
graph_extract.dedup_guard clears the whole `contradicted_facts` list before graphiti
sees it (`ingest_same_pair_contradictions=False`; 0 of 14 live cases were genuine,
pre-bootstrap-decisions-2026-09-23.md D4). Everything above still describes what
graphiti does when that suppression is off. Note the coupling: re-enabling this
gate's search while suppression stays on runs the scan and discards its results --
the ingest CLI warns about that combination at startup.

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
_ORIGINAL = "_graph_extract_contradiction_gate_original"


def install_contradiction_gate(*, detect_contradictions: bool) -> bool:
    """Patch `edge_operations`' own `search` reference so the unfiltered
    invalidation search never runs -- or, called with `detect_contradictions=True`
    while a gate is installed, undo that patch.

    `edge_operations` does `from graphiti_core.search.search import search`, so the
    patch must target THAT module's attribute; patching the defining module would
    not be seen.

    Returns True when the gate is installed (contradiction detection suspended),
    False when detection is left enabled (or has just been restored). Idempotent
    in both directions.
    """
    if detect_contradictions:
        current = edge_operations.search
        if getattr(current, _MARKER, False):
            edge_operations.search = current.__dict__[_ORIGINAL]
            logger.info(
                "contradiction detection restored: the invalidation-candidate "
                "search is delegated again")
        return False
    if getattr(edge_operations.search, _MARKER, False):
        return True

    original = edge_operations.search

    async def gated(*args: Any, **kwargs: Any) -> Any:
        # `search_filter` is the 5th positional parameter and is passed as a
        # keyword by both call sites today. Read both. The failure modes are NOT
        # symmetric: a switch to positional still discriminates ("still
        # skipped"); a call site that OMITS `search_filter` altogether reads as
        # None here, is delegated, and the corpus scan silently resumes. The
        # call-site tripwire in tests/unit/test_contradiction_gate.py is what
        # guards the second case, not this code.
        search_filter = kwargs.get("search_filter")
        if search_filter is None and len(args) >= 5:
            search_filter = args[4]
        if search_filter is not None and getattr(search_filter, "edge_uuids", None) is None:
            # The invalidation search. Empty, well-formed -- never None, which
            # would fail confusingly deep inside graphiti.
            return SearchResults()
        return await original(*args, **kwargs)

    gated.__dict__[_MARKER] = True
    gated.__dict__[_ORIGINAL] = original
    edge_operations.search = gated
    logger.info(
        "contradiction detection suspended: the unfiltered invalidation-candidate "
        "search is skipped (scale blocker; see BACKLOG 6/8/30)")
    return True


def is_gate_installed() -> bool:
    """True when the invalidation-candidate search is currently being skipped."""
    return bool(getattr(edge_operations.search, _MARKER, False))
