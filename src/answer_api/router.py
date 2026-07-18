"""The /answer router: classify a query to a retrieval mode (heuristics first, a
cheap-tier LLM otherwise, default drift), dispatch to the existing mode function,
and normalize every mode's output into one uniform envelope. The router authors
no prose and no URL (design decision #2) — it only picks a mode label and
reshapes; answers/citations come from the mode functions or the deterministic
timeline render."""
from __future__ import annotations

import logging
import re
from typing import Literal, cast

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.usage import instrument

logger = logging.getLogger(__name__)

Mode = Literal["local", "global", "drift", "timeline"]
_MODE_ORDER: tuple[str, ...] = ("local", "global", "drift", "timeline")
_MODES = frozenset(_MODE_ORDER)

# High-confidence heuristic guardrails (checked before the LLM). Case-insensitive.
_TEMPORAL_RE = re.compile(
    r"\b(chang(e|ed|es|ing)|history|used to|since \d|over time|evolv|"
    r"deprecat|no longer|when did|previously)\b", re.IGNORECASE)
_CROSS_VENDOR_RE = re.compile(
    r"\b(compare|comparison|across (all )?vendors|all vendors|which vendors|"
    r"every vendor)\b", re.IGNORECASE)

_CLASSIFY_PROMPT = (
    "Classify the QUESTION about backup products into exactly one retrieval mode. "
    "Reply with ONLY the lowercase label, nothing else.\n"
    "- local: a specific factual question about one product/feature\n"
    "- global: a broad thematic/cross-corpus question spanning many vendors\n"
    "- drift: a broad question that still expects concrete, sourced detail\n"
    "- timeline: how something changed over time\n\n"
    "QUESTION: {q}\nLabel:"
)


def _cheap_classify_client(settings: ExtractSettings) -> tuple[AsyncOpenAI | None, str]:
    """The cheap classifier client (ling), or (None, '') when no cheap key is set
    -> the router runs heuristics-only and defaults uncaught queries to drift."""
    if not settings.cheap_llm_api_key:
        return None, ""
    client = instrument(AsyncOpenAI(
        api_key=settings.cheap_llm_api_key, base_url=settings.cheap_llm_base_url))
    return client, settings.cheap_llm_model


async def classify(q: str, *, cheap_client, cheap_model,
                   mode_override: str | None = None,
                   default_mode: str = "drift") -> tuple[Mode, str]:
    chosen, via = default_mode, "default"
    if mode_override in _MODES:
        chosen, via = cast(str, mode_override), "override"
    elif _TEMPORAL_RE.search(q):
        chosen, via = "timeline", "heuristic"
    elif _CROSS_VENDOR_RE.search(q):
        chosen, via = "global", "heuristic"
    elif cheap_client is not None:
        try:
            resp = await cheap_client.chat.completions.create(
                model=cheap_model, temperature=0, max_tokens=8,
                messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(q=q)}])
            label = (resp.choices[0].message.content or "").strip().lower()
            match = next((m for m in _MODE_ORDER if m in label), None)
            if match is not None:
                chosen, via = match, "llm"
        except Exception:
            logger.warning("cheap classifier failed; defaulting", exc_info=True)
    return cast(Mode, chosen), via
