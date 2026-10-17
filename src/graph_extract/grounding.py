"""Deterministic grounding gate for graphiti's entity extraction.

Traced 2026-10-17 (docs/superpowers/specs/2026-10-17-extraction-grounding-design.md):
the extraction model invented entities the chunk never mentions -- a forced subject
("AWS Backup" over a bare Securiti.ai connector list) and the instructions' own
example vocabulary ("Hyper-V", "East US", "LRS" in a short AWS Security Hub control).
Facts then attached to those inventions.

`dedup_guard`'s wrapper already intercepts every graphiti LLM call; for the
`extract_nodes.extract_text` reply it calls `filter_entities`, which keeps an
extracted entity only when the chunk text mentions it (`is_grounded`). graphiti then
extracts edges only among the kept entities, so no fact can attach to an invented one.

The chunk text and the document's product/vendor names arrive through
`CURRENT_GROUNDING`, which IngestDriver sets around each `add_text_episode`. With no
context open (probe, compat, pilot scripts), the gate is a no-op counted as unscoped.

Never reads a failure as absence: a reply the gate cannot parse is passed through
unchanged (and counted), never emptied.
"""
from __future__ import annotations

import logging
import math
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

EXTRACT_TEXT_PROMPT_NAME = "extract_nodes.extract_text"

DROPPED_NAMES_CAP = 50

_STOPWORDS = frozenset(
    {"the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with", "by"})
_TOKEN = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("es", "s", "ed", "ing")


@dataclass
class GroundingStats:
    """Counters for one scope (an article, or a batch/run when merged)."""
    calls: int = 0            # extract_text replies filtered
    dropped: int = 0          # entities removed from those replies
    parse_failures: int = 0   # replies not shaped like ExtractedEntities -> passed through
    unscoped: int = 0         # extract_text calls with no grounding context open
    dropped_names: list[str] = field(default_factory=list)   # first DROPPED_NAMES_CAP

    def _add_names(self, names: list[str]) -> None:
        room = DROPPED_NAMES_CAP - len(self.dropped_names)
        if room > 0:
            self.dropped_names.extend(names[:room])

    def merge(self, other: GroundingStats) -> None:
        self.calls += other.calls
        self.dropped += other.dropped
        self.parse_failures += other.parse_failures
        self.unscoped += other.unscoped
        self._add_names(other.dropped_names)


@dataclass
class GroundingContext:
    text: str                    # the episode body exactly as graphiti receives it
    allowed: tuple[str, ...]     # the document's product and vendor names
    stats: GroundingStats


# Set by IngestDriver around each add_text_episode; read by dedup_guard's wrapper.
# A ContextVar for the same reason as CURRENT_DEDUP_STATS: articles run concurrently,
# and add_episode's inner tasks inherit the context.
CURRENT_GROUNDING: ContextVar[GroundingContext | None] = ContextVar(
    "current_grounding", default=None)


def _distinctive_tokens(name: str) -> list[str]:
    toks = (t for t in _TOKEN.findall(name.lower()) if len(t) >= 2 and t not in _STOPWORDS)
    return list(dict.fromkeys(toks))


def _stem(token: str) -> str | None:
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return None


def _at_word_start(token: str, text: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(token), text) is not None


def _present(token: str, text: str) -> bool:
    if _at_word_start(token, text):
        return True
    stem = _stem(token)
    return stem is not None and _at_word_start(stem, text)


def is_grounded(name: str, text: str, allowed: tuple[str, ...]) -> bool:
    """True when the text mentions `name` -- verbatim, or by at least ceil(2n/3) of
    its n distinctive words at word starts (a simple inflection counts) -- or when
    `name` is the document's product or vendor (the implied subject)."""
    lname = name.strip().lower()
    ltext = text.lower()
    if lname in ltext:
        return True
    if any(a.strip() and a.strip().lower() == lname for a in allowed):
        return True
    toks = _distinctive_tokens(lname)
    if not toks:
        return False
    present = sum(1 for t in toks if _present(t, ltext))
    return present >= math.ceil(2 * len(toks) / 3)


def filter_entities(response: Any, ctx: GroundingContext) -> Any:
    """`response` with `extracted_entities` reduced to the grounded ones. A reply
    not shaped like ExtractedEntities is returned unchanged and counted; graphiti's
    own `ExtractedEntities(**response)` then decides what it is worth."""
    stats = ctx.stats
    entities = response.get("extracted_entities") if isinstance(response, dict) else None
    if not isinstance(entities, list):
        stats.parse_failures += 1
        logger.warning(
            "grounding: extract_text reply is not {'extracted_entities': [...]} (%s); "
            "passed through unfiltered", type(response).__name__)
        return response
    stats.calls += 1
    kept: list[Any] = []
    dropped: list[str] = []
    for ent in entities:
        name = ent.get("name") if isinstance(ent, dict) else None
        # An item without a string name is not ours to judge: keep it.
        if not isinstance(name, str) or is_grounded(name, ctx.text, ctx.allowed):
            kept.append(ent)
        else:
            dropped.append(name)
    if not dropped:
        return response
    stats.dropped += len(dropped)
    stats._add_names(dropped)
    logger.debug("grounding dropped %d of %d entities: %s",
                 len(dropped), len(entities), dropped)
    return {**response, "extracted_entities": kept}
