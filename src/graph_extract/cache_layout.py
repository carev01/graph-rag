"""Move each prompt type's static user-message tail into the system message.

graphiti assembles its extraction prompts as a short static system message plus a
user message that puts the per-call content FIRST (previous/current messages,
entities, facts) and the static rules LAST. A provider prefix cache can only reuse
an identical PREFIX, so consecutive calls share almost nothing: ~7% cache hits on
the cheap tier (OpenRouter, 2026-09-25), where a cached input token costs $0.018/M
instead of $0.09/M.

Where the block goes matters and is empirical (scripts/reorder_prompts.py, measured
on llama.cpp): moved to the FRONT of the user message it gave no reuse at all
(8.88 s vs 9.24 s cold); moved into the SYSTEM message the same tokens served from
cache (1.37 s, 6.4x). So the block goes into the system message.

The static tail is LEARNED ONLINE, per prompt type, never committed as text: for the
first `min_samples` calls of a type nothing changes and the learner accumulates the
longest common suffix of their user messages; it then freezes that suffix (trimmed
to a section boundary by `clean_block`) and moves it on every later call whose user
message ends with it. A graphiti upgrade or an ontology change therefore re-learns
in a new process instead of drifting from a stale copy. Library-owned prompts are
not modified -- only the request's message list, at the transport, like
`graphiti_client._bound_index_arrays` does for the schema.

extraction-grounding (2026-10-17) appends a PER-ARTICLE "DOCUMENT CONTEXT: ..."
sentence to the extraction instructions (`graphiti_client.document_context`),
INSIDE `custom_extraction_instructions`. graphiti does NOT put that at the end of
the user message: `extract_nodes.extract_text` follows it with ~700 chars of
static <EXAMPLE> blocks, and `extract_edges.edge` follows it with ~4,000 chars of
static "# EXTRACTION RULES" text -- both otherwise-static tails this module exists
to cache. Left alone, the per-article sentence sits in the MIDDLE of what this
module compares, which breaks the mechanism in two ways: a same-article warm-up
bakes that one article's sentence into the "static" block, so every later,
different-article call then fails to match (nothing moves); a mixed-article
warm-up never accumulates a full common suffix either, because the differing
sentence's position still has to be walked past character-by-character before the
comparison reaches the static text behind it, and depending on sentence length
that erodes or (if a later sentence is longer) corrupts the learned block.
(An earlier version of this fix wrongly assumed the sentence was the very end of
the user message and just split there -- that silently discarded graphiti's real
trailing static blocks into the "varying" half instead of the learned one.)

The fix: before learning AND before matching, CUT the DOCUMENT CONTEXT sentence
out of the user message using `graphiti_client.DOCUMENT_CONTEXT_PATTERN` (the LAST
match, in case a chunk's own text happens to contain the bare marker -- the pattern
requires the full sentence opener, so a body line with just the marker text and no
"This chunk comes from ..." never matches), closing the gap it leaves so the text
before and after it become contiguous; learn/match on that. Once a block is moved
to the system message, the cut sentence is re-appended to what remains of the user
message (position doesn't matter to the model, only presence) -- it is never lost,
just excluded from what the learner compares across calls. A user message with no
match is untouched by this (today's behaviour exactly).
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def common_suffix(a: str, b: str) -> str:
    return os.path.commonprefix([a[::-1], b[::-1]])[::-1]


def _cut_document_context(user: str) -> tuple[str, str]:
    """`(user_with_context_cut_out, context_sentence)`: the LAST match of
    `graphiti_client.DOCUMENT_CONTEXT_PATTERN` is removed from `user`, closing
    the gap (NOT split at a position -- the sentence can sit in the middle of
    the user message, between `custom_extraction_instructions` and graphiti's
    own trailing static text). `context_sentence` is `""` (and `user` is
    returned unchanged) when there is no match -- the no-context path this
    module behaved on before extraction-grounding. The pattern requires the
    full sentence opener, so a chunk body line that merely contains the bare
    "DOCUMENT CONTEXT:" marker text never matches.

    Deferred import: `graphiti_client` imports this module (`wrap_client`), so
    importing it back at module scope here would be a circular import. The
    import resolves fine by the time this runs, since it only runs per-call,
    long after both modules have finished loading.
    """
    from graph_extract.graphiti_client import DOCUMENT_CONTEXT_PATTERN
    matches = list(DOCUMENT_CONTEXT_PATTERN.finditer(user))
    if not matches:
        return user, ""
    m = matches[-1]
    return user[:m.start()] + user[m.end():], m.group(0)


def clean_block(suffix: str) -> str:
    """Trim the leading fragment so the moved text starts at a section boundary.

    The longest common suffix can begin MID-TOKEN (a varying timestamp ending in a
    constant `+00:00`), so always advance to the next line; what remains often opens
    with the closer of the preceding variable block (`</CURRENT MESSAGE>`), which
    belongs to the content staying behind, so skip past it too.
    """
    i = suffix.find("\n")
    s = suffix[i + 1:] if i >= 0 else suffix
    if s.lstrip().startswith("</"):
        j = s.find("\n\n")
        if j >= 0:
            s = s[j + 2:]
    return s.lstrip("\n")


@dataclass
class _Type:
    samples: int = 0
    running: str | None = None
    block: str | None = None          # None while learning; "" = nothing worth moving


class StaticTailLearner:
    def __init__(self, *, min_samples: int = 5, min_chars: int = 400) -> None:
        self._min_samples = min_samples
        self._min_chars = min_chars
        self._types: dict[str, _Type] = {}
        self._moved = 0
        self._skipped = 0

    def stats(self) -> dict:
        learned = sum(1 for t in self._types.values() if t.block)
        return {"learned": learned, "moved": self._moved, "skipped": self._skipped}

    def transform(self, prompt_key: str, messages: list[dict]) -> list[dict]:
        si = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)
        ui = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
        if si is None or ui is None:
            return messages
        system, user = messages[si].get("content"), messages[ui].get("content")
        if not isinstance(system, str) or not isinstance(user, str):
            return messages
        # The per-article DOCUMENT CONTEXT sentence (if any) varies call to call
        # and can sit in the MIDDLE of graphiti's rendered user message (between
        # custom_extraction_instructions and graphiti's own trailing static
        # <EXAMPLE>/EXTRACTION RULES text) -- cut it out so what's left is
        # contiguous, then restore it below.
        user_main, context = _cut_document_context(user)
        # A prompt type = the response schema name AND its system message; two
        # prompt functions sharing a schema still learn separately.
        key = f"{prompt_key}:{hashlib.sha1(system.encode()).hexdigest()[:12]}"
        t = self._types.setdefault(key, _Type())
        if t.block is None:
            t.running = user_main if t.running is None else common_suffix(t.running, user_main)
            t.samples += 1
            if t.samples >= self._min_samples:
                block = clean_block(t.running)
                t.block = block if len(block) >= self._min_chars else ""
                t.running = None
                if t.block:
                    logger.info("prompt cache layout: static block learned for %s "
                                "(%d chars, from %d calls)", prompt_key, len(t.block),
                                t.samples)
            return messages
        if not t.block:
            return messages
        if not user_main.endswith(t.block):
            self._skipped += 1
            return messages
        head = user_main[: len(user_main) - len(t.block)]
        out = [dict(m) for m in messages]
        out[si]["content"] = f"{system.rstrip()}\n\n{t.block.rstrip()}\n"
        out[ui]["content"] = head.rstrip() + context if context else head.rstrip() + "\n"
        self._moved += 1
        return out


def _schema_name(kwargs: dict) -> str:
    fmt = kwargs.get("response_format")
    if isinstance(fmt, dict):
        js = fmt.get("json_schema")
        if isinstance(js, dict) and js.get("name"):
            return str(js["name"])
    return "-"


def wrap_client(client, learner: StaticTailLearner):
    """Rewrite `messages` on every `chat.completions.create` through `learner`."""
    orig = client.chat.completions.create

    async def create(*args, **kwargs):
        msgs = kwargs.get("messages")
        if isinstance(msgs, list):
            kwargs["messages"] = learner.transform(_schema_name(kwargs), msgs)
        return await orig(*args, **kwargs)

    client.chat.completions.create = create
    return client
