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
from graph_extract.usage import bounded_llm_client, usable_content

from answer_api import synthesize as synth_mod
from answer_api import global_search as global_mod
from answer_api import drift as drift_mod
from answer_api import timeline as timeline_mod
from answer_api import freshness as freshness_mod
from answer_api import attribution
from answer_api.scope import Scope

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
    # No reasoning parameter, deliberately and by measurement (2026-09-11,
    # upstage/solar-pro4): the classifier's max_tokens=8 fits a 2-token label
    # because the model does not reason by default; `reasoning: {effort: low}`
    # switched thinking ON and every reply came back content=None. A reasoning
    # cheap model would need BOTH a bound and a far larger max_tokens here.
    # 20s cuts a slow route (measured 36s for a 2-token reply) rather than
    # waiting on it; a timeout retries on a new route twice, then classify()
    # falls back to the default mode.
    # max_retries=1, not 2: on timeout the classifier falls back to via="default",
    # which routes to DRIFT -- the most expensive mode. A 36s route was measured on
    # this very model, so 20s x 3 attempts (60s worst case) makes a spurious default
    # plausible under provider slowdown. One retry halves that exposure.
    # reasoning_effort="" deliberately: solar-pro4 does not reason by default, and
    # sending `reasoning` would ENABLE it -- at max_tokens=8 every reply then comes
    # back content=None. See config.py and the 2026-09-11 hardening report.
    client = bounded_llm_client(settings.cheap_llm_base_url, settings.cheap_llm_api_key,
                                reasoning_effort="", timeout=20.0, max_retries=1,
                                tier="router-classify", capture_path=settings.llm_capture_path)
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
            # BACKLOG 5: an empty reply (no choices, or content=None) is the
            # same empty-reply class fixed at four other sites. The default is
            # still the right outcome, but it must be logged, not coerced to
            # "" and silently fallen through.
            content = usable_content(resp)
            if content is None:
                finish = (getattr(resp.choices[0], "finish_reason", None)
                          if resp.choices else "no-choices")
                logger.warning("cheap classifier %s returned no usable content "
                               "(finish_reason=%s); defaulting", cheap_model, finish)
            else:
                label = content.strip().lower()
                match = next((m for m in _MODE_ORDER if m in label), None)
                if match is not None:
                    chosen, via = match, "llm"
        except Exception:
            logger.warning("cheap classifier failed; defaulting", exc_info=True)
    return cast(Mode, chosen), via


def _render_timeline(timeline_result: dict) -> tuple[str, list[dict]]:
    events = timeline_result.get("timeline", [])
    if not events:
        return "No recorded changes for that query.", []
    lines: list[str] = []
    citations: list[dict] = []
    for i, e in enumerate(events, 1):
        valid = e.get("valid_at")
        invalid = e.get("invalid_at")
        span = f"valid_at {valid}" if valid else "no recorded start"
        if invalid:
            span += f", invalid_at {invalid}"
        label = attribution.label(e.get("sources", []))
        fact_part = f"**{e['fact']}** {label}" if label else f"**{e['fact']}**"
        lines.append(f"- {fact_part} — {span} ({e.get('status', '')}) [{i}]")
        citations.append({"marker": i, "fact_uuid": e["fact_uuid"],
                          "valid_at": e.get("valid_at"), "invalid_at": e.get("invalid_at"),
                          "sources": e.get("sources", [])})
    return "\n".join(lines), citations


def _normalize(mode: Mode, via: str, fallback_from: str | None, raw: dict,
               q: str, scope: Scope) -> dict:
    routing: dict = {"chosen": mode, "via": via, "fallback_from": fallback_from}
    if "degraded" in raw:
        routing["degraded"] = raw["degraded"]
    if mode == "timeline":
        answer, citations = _render_timeline(raw)
    else:
        answer = raw.get("answer", "")
        citations = raw.get("citations", [])
    env: dict = {"mode": mode, "query": q, "answer": answer,
                 "citations": citations, "routing": routing,
                 "scope": scope.as_dict(),
                 "applies_to": raw["applies_to"] if "applies_to" in raw
                 else attribution.applies_to(citations)}
    if mode == "timeline":
        env["timeline"] = raw.get("timeline", [])
    if "communities_used" in raw:
        env["communities_used"] = raw["communities_used"]
    if "follow_ups" in raw:
        env["follow_ups"] = raw["follow_ups"]
    return env


async def _dispatch(mode, graphiti, driver, embedder, synth_client, synth_model,
                    map_client, map_model, *, q, scope: Scope, settings) -> dict:
    g = settings.group_id
    if mode == "local":
        return await synth_mod.answer_local(
            graphiti, driver, synth_client, synth_model, q=q, scope=scope, group_id=g)
    if mode == "global":
        return await global_mod.global_search(
            driver, embedder, map_client, map_model, synth_client, synth_model,
            q=q, level=settings.global_default_level, k=settings.global_shortlist_k,
            group_id=g, settings=settings, scope=scope)
    if mode == "drift":
        return await drift_mod.drift_search(
            graphiti, driver, embedder, synth_client, synth_model, q=q,
            level=settings.drift_primer_level, iterations=settings.drift_iterations,
            primer_k=settings.drift_primer_k, max_followups=settings.drift_max_followups,
            followup_k=settings.drift_followup_k, group_id=g, settings=settings,
            scope=scope)
    return await timeline_mod.timeline_local(
        graphiti, driver, q=q, scope=scope, group_id=g)


async def answer_router(graphiti, driver, embedder, synth_client, synth_model,
                        map_client, map_model, cheap_client, cheap_model, *,
                        q, mode_override, scope: Scope, settings) -> dict:
    mode, via = await classify(q, cheap_client=cheap_client, cheap_model=cheap_model,
                               mode_override=mode_override,
                               default_mode=settings.router_default_mode)
    raw = await _dispatch(mode, graphiti, driver, embedder, synth_client, synth_model,
                          map_client, map_model, q=q, scope=scope, settings=settings)
    fallback_from: str | None = None
    if mode == "local" and raw.get("retrieved") == 0:
        fallback_from = "local"
        mode = "drift"
        raw = await _dispatch("drift", graphiti, driver, embedder, synth_client,
                              synth_model, map_client, map_model, q=q, scope=scope,
                              settings=settings)
    elif mode == "global" and not raw.get("communities_used"):
        fallback_from = "global"
        mode = "local"
        raw = await _dispatch("local", graphiti, driver, embedder, synth_client,
                              synth_model, map_client, map_model, q=q, scope=scope,
                              settings=settings)
    env = _normalize(mode, via, fallback_from, raw, q, scope)
    env["freshness"] = await freshness_mod.freshness(
        driver, settings.group_id, reports=mode in ("global", "drift"))
    return env
