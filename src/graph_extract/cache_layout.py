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
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def common_suffix(a: str, b: str) -> str:
    return os.path.commonprefix([a[::-1], b[::-1]])[::-1]


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
        # A prompt type = the response schema name AND its system message; two
        # prompt functions sharing a schema still learn separately.
        key = f"{prompt_key}:{hashlib.sha1(system.encode()).hexdigest()[:12]}"
        t = self._types.setdefault(key, _Type())
        if t.block is None:
            t.running = user if t.running is None else common_suffix(t.running, user)
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
        if not user.endswith(t.block):
            self._skipped += 1
            return messages
        head = user[: len(user) - len(t.block)]
        out = [dict(m) for m in messages]
        out[si]["content"] = f"{system.rstrip()}\n\n{t.block.rstrip()}\n"
        out[ui]["content"] = head.rstrip() + "\n"
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
