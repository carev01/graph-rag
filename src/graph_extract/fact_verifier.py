"""Fact verifier, production side (spec docs/superpowers/specs/2026-10-29-verifier-trial-
design.md section 4). A LoRA on Qwen3.5-4B (verifier1, trained by notebooks/
distill_verifier.ipynb on the dpo4 Opus labels) labels each fact of one
`extract_edges.edge` reply as stated / previous_only / unsupported. `EdgeVerifier.check`
drops the unsupported ones (policy P2) before graphiti sees the reply.

The prompt is byte-identical to the training prompt (scripts/distill/verifier_data.py
imports VERIFIER_INSTRUCTIONS, fact_line and the user-text builder from here; a unit test
pins it). The reply parser is the scorer's strict one: a reply that is not exactly the
trained JSON keeps every edge. Every failure -- error, timeout, empty, unparseable, wrong
count, a reply shape it cannot read -- FAILS OPEN and is counted.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

VERDICTS = frozenset({"stated", "previous_only", "unsupported"})
VERIFIER_INSTRUCTIONS = 'You are auditing knowledge-graph facts extracted from ONE chunk of a vendor\ndocumentation page. CURRENT MESSAGE is the chunk. PREVIOUS MESSAGES are earlier chunks of\nthe same page, given only as context. DOCUMENT CONTEXT names the documented product.\n\nFor EVERY numbered fact below, in order, decide:\n- "stated": the CURRENT MESSAGE states it, or implies it unambiguously (resolving a\n  reference with PREVIOUS MESSAGES or DOCUMENT CONTEXT is fine). Give "quote": the\n  shortest span of the CURRENT MESSAGE that states it, copied character for character\n  (a few words at least -- never just an entity name).\n- "previous_only": the claim is supported only by the PREVIOUS MESSAGES, not the CURRENT MESSAGE.\n- "unsupported": neither supports it. Be strict: plausible general knowledge the text does\n  not state is unsupported.\nFor an unsupported fact give a category:\n- "over_inference": a claim the text does not make (generalisation, assumed capability, cause).\n- "subject_substitution": a claim the text makes about something else, attributed to the wrong\n  product, component or entity.\n- "taxonomy_as_fact": a list/menu/heading membership turned into a relationship the text does not assert.\n- "wrong_detail": right subject and gist, but a wrong number, version, condition or limit.\n- "other".\n\nReturn only JSON: {"labels": [{"n": <fact number>, "verdict": "...", "category": "..."\n(unsupported only), "quote": "..." (stated only)}]} with exactly one entry per fact, in order.\n'
_CONTEXT_LINE = re.compile(r"DOCUMENT CONTEXT: This chunk comes from [^\n]*")


def _role_content(m: Any) -> tuple[str, str]:
    """A chat message as (role, content): graphiti's Message objects or plain dicts."""
    if isinstance(m, dict):
        return str(m.get("role")), str(m.get("content"))
    return str(m.role), str(m.content)


def tag(text: str, name: str) -> str:
    m = re.search(rf"<{name}>\n?(.*?)</{name}>", text, re.S)
    return m.group(1).strip() if m else ""


def document_context(messages: list[Any]) -> str:
    for m in messages:
        hit = _CONTEXT_LINE.search(_role_content(m)[1])
        if hit:
            return hit.group(0)
    return ""


def fact_line(e: dict) -> str:
    return f"{e.get('source_entity_name')} -> {e.get('target_entity_name')}: {e.get('fact')}"


def verifier_user(messages: list[Any], facts: list[str]) -> str:
    """The verifier's user text from one extract_edges call's messages (system, user)."""
    user = next((c for r, c in map(_role_content, messages) if r == "user"), "")
    numbered = "\n".join(f"{n}. {f}" for n, f in enumerate(facts, 1))
    return "\n\n".join(["PREVIOUS MESSAGES:\n" + tag(user, "PREVIOUS_MESSAGES"),
                        "CURRENT MESSAGE:\n" + tag(user, "CURRENT_MESSAGE"),
                        document_context(messages), "FACTS:\n" + numbered])


def parse_reply(text: str | None, n: int) -> list[dict] | None:
    """Exactly n labels numbered 1..n with known verdicts, in n order; else None."""
    if not text:
        return None
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return None
    labels = obj.get("labels") if isinstance(obj, dict) else None
    if not isinstance(labels, list) or len(labels) != n:
        return None
    out: list[dict | None] = [None] * n
    for lab in labels:
        k = lab.get("n") if isinstance(lab, dict) else None
        if type(k) is not int or not 1 <= k <= n or out[k - 1] is not None:
            return None
        verdict = lab.get("verdict")
        if not isinstance(verdict, str) or verdict not in VERDICTS:
            return None
        out[k - 1] = lab
    return [lab for lab in out if lab is not None]


def keep_edges(parsed: list[dict] | None, edges: list) -> tuple[list, list]:
    """Policy P2: drop the edges labelled unsupported. None (unusable reply) keeps all."""
    if parsed is None:
        return list(edges), []
    kept = [e for e, lab in zip(edges, parsed) if lab["verdict"] != "unsupported"]
    dropped = [e for e, lab in zip(edges, parsed) if lab["verdict"] == "unsupported"]
    return kept, dropped


def max_tokens(n: int) -> int:
    """The prediction cap the verifier was scored with (notebook section 10)."""
    return 64 + 160 * n


class EdgeVerifier:
    """Wraps an AsyncOpenAI-compatible client serving the verifier adapter."""

    def __init__(self, client: Any, model: str, timeout: float) -> None:
        self.client, self.model, self.timeout = client, model, timeout

    async def check(self, messages: list[Any], response: Any, stats: Any) -> Any:
        """The reply with its unsupported edges removed; any failure returns it as is.
        `messages` must be the call's messages AS AUTHORED (before the client appended
        its schema text)."""
        if not isinstance(response, dict) or not isinstance(response.get("edges"), list) \
                or not response["edges"]:
            return response                  # nothing to check, or a shape we cannot read
        edges = response["edges"]
        stats.verifier_calls += 1
        if not all(isinstance(e, dict) for e in edges):
            stats.verifier_failed_open += 1
            return response
        user = verifier_user(messages, [fact_line(e) for e in edges])
        try:
            reply = await self.client.chat.completions.create(
                model=self.model, temperature=0, max_tokens=max_tokens(len(edges)),
                timeout=self.timeout,
                messages=[{"role": "system", "content": VERIFIER_INSTRUCTIONS},
                          {"role": "user", "content": user}])
            content = reply.choices[0].message.content
        except Exception as exc:             # fail open: the verifier must never fail an episode
            stats.verifier_failed_open += 1
            logger.warning("edge verifier call failed (%s: %s); keeping all %d edges",
                           type(exc).__name__, str(exc)[:200], len(edges))
            return response
        parsed = parse_reply(content, len(edges))
        if parsed is None:
            stats.verifier_failed_open += 1
            logger.warning("edge verifier reply unusable; keeping all %d edges", len(edges))
            return response
        kept, dropped = keep_edges(parsed, edges)
        stats.verifier_dropped += len(dropped)
        return {**response, "edges": kept}
