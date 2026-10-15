"""List mode (spec 2026-10-15): exhaustive answers to single-vendor list questions.

Detection is deterministic. Expansion asks the cheap tier for retrieval queries that
name likely KINDS of item (hypervisors, clouds, ...): a paraphrase expansion did not
find the missing items (measured 2026-10-15), item-kind queries did. The model's
domain knowledge only steers which facts are searched for; the answer states only
what retrieved facts say."""
from __future__ import annotations

import json
import logging
import re

from graph_extract.usage import usable_content

logger = logging.getLogger(__name__)

_LIST_RE = re.compile(
    r"^\s*(which|what)\s+(\w+\s+){0,3}?\w+s\b(?!\s+(is|was)\b)"     # which/what ... <plural>
    r"|^\s*where\s+can\s+(i|you|we)\b"                             # where can I ...
    r"|^\s*what\s+are\s+the\s+(\w+\s+)?(options|ways|methods|types|kinds|destinations|targets)\b"
    r"|^\s*list\b"
    r"|\bsupported\s+\w+s\b",
    re.IGNORECASE)
# "What should I consider ..." / "What is ..." start with what + a non-plural; the
# pattern needs a plural noun within the first words. Questions that are clearly
# single-answer are excluded explicitly.
_NOT_LIST_RE = re.compile(r"^\s*(what|which)\s+(is|was|should|does|do|did|can)\b",
                          re.IGNORECASE)

_LIMIT_RE = re.compile(
    r"\b(not supported|does not|doesn't|do not|don't|cannot|can't|can not|is not|are not|"
    r"unsupported|only (available|supported|for|when|if|with|on|in)|not available)\b",
    re.IGNORECASE)

_EXPAND_PROMPT = (
    "You help search vendor backup documentation. The QUESTION asks for a list. Write "
    "up to {n} short search queries that together would find EVERY item the "
    "documentation might list as an answer. Name the likely kinds of item explicitly "
    "(for example, for restore destinations: specific hypervisors, clouds, physical "
    "machines, file and application level; for workloads: VMs, physical machines, "
    "databases, SaaS, Kubernetes). Keep the vendor and product names from the "
    "question in every query. Reply with ONLY a JSON list of strings.\n\n"
    "QUESTION: {q}\nJSON:")


def is_list_question(q: str) -> bool:
    return bool(_LIST_RE.search(q)) and not _NOT_LIST_RE.search(q)


def is_limitation(fact: str) -> bool:
    return bool(_LIMIT_RE.search(fact))


async def expand_queries(client, model: str, q: str, max_n: int) -> list[str]:
    """Up to `max_n` item-kind retrieval queries; [] (logged) when unusable."""
    try:
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=800,
            messages=[{"role": "user", "content": _EXPAND_PROMPT.format(n=max_n, q=q)}])
        content = usable_content(resp)
        if content is None:
            logger.warning("list expansion returned no usable content; using the question alone")
            return []
        start, end = content.find("["), content.rfind("]")
        parsed = json.loads(content[start:end + 1]) if start >= 0 and end > start else None
    except Exception as exc:
        logger.warning("list expansion failed (%s); using the question alone", exc)
        return []
    if not isinstance(parsed, list):
        logger.warning("list expansion reply was not a JSON list; using the question alone")
        return []
    out: list[str] = []
    for item in parsed:
        if isinstance(item, str) and item.strip() and item.strip() not in out:
            out.append(item.strip())
    return out[:max_n]
