"""The /answer router: classify a query to a retrieval mode (heuristics first, a
cheap-tier LLM otherwise, default drift), dispatch to the existing mode function,
and normalize every mode's output into one uniform envelope. The router authors
no prose and no URL (design decision #2) — it only picks a mode label and
reshapes; answers/citations come from the mode functions or the deterministic
timeline render."""
from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import Literal, cast

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.usage import bounded_llm_client, usable_content

from answer_api import synthesize as synth_mod
from answer_api import global_search as global_mod
from answer_api import drift as drift_mod
from answer_api import timeline as timeline_mod
from answer_api import freshness as freshness_mod
from answer_api import enumeration as enum_mod
from answer_api import attribution
from answer_api.scope import Scope
from answer_api.shape import QuestionShape, could_not_check_note, facet_queries

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
                    map_client, map_model, *, q, scope: Scope, settings,
                    facets=None) -> dict:
    g = settings.group_id
    if mode == "local":
        return await synth_mod.answer_local(
            graphiti, driver, synth_client, synth_model, q=q, scope=scope, group_id=g,
            facets=facets, min_per_facet=settings.local_min_per_facet,
            fetch_cap=settings.local_scoped_fetch_cap,
            scan_max_episodes=settings.local_scoped_scan_max_episodes, settings=settings)
    if mode == "global":
        return await global_mod.global_search(
            driver, embedder, map_client, map_model, synth_client, synth_model,
            q=q, level=settings.global_default_level, k=settings.global_shortlist_k,
            group_id=g, settings=settings, scope=scope, facets=facets)
    if mode == "drift":
        return await drift_mod.drift_search(
            graphiti, driver, embedder, synth_client, synth_model, q=q,
            level=settings.drift_primer_level, iterations=settings.drift_iterations,
            primer_k=settings.drift_primer_k, max_followups=settings.drift_max_followups,
            followup_k=settings.drift_followup_k, group_id=g, settings=settings,
            scope=scope, facets=facets)
    return await timeline_mod.timeline_local(
        graphiti, driver, q=q, scope=scope, group_id=g)


def _grounded(mode: Mode, raw: dict) -> bool:
    if mode == "timeline":
        return bool(raw.get("timeline"))
    return bool(raw.get("citations"))


def _shape_fields(env: dict, shape: QuestionShape | None, raw: dict,
                  requested: str) -> None:
    """`shape`/`facets` describe how the question was ANSWERED (single after an
    override, relaxation or a timeline run, which ignores facets);
    `shape_requested` is the classified kind it was asked as."""
    env["shape"] = shape.kind if shape is not None else "single"
    env["shape_requested"] = requested
    # `query` (the facet's rewritten retrieval question) only when one was set, so a
    # single question's envelope is unchanged.
    env["facets"] = [{"label": f.label, "scope": f.scope.as_dict(),
                      **({"query": f.query} if f.query else {})}
                     for f in (shape.facets if shape is not None else ())]
    for key in ("degraded_facets", "checked_vendors", "documented_vendors"):
        if key in raw:
            env[key] = raw[key]


async def answer_router(graphiti, driver, embedder, synth_client, synth_model,
                        map_client, map_model, cheap_client, cheap_model, *,
                        q, mode_override, scope: Scope, settings,
                        shape: QuestionShape | None = None) -> dict:
    mode, via = await classify(q, cheap_client=cheap_client, cheap_model=cheap_model,
                               mode_override=mode_override,
                               default_mode=settings.router_default_mode)
    requested = shape.kind if shape is not None else "single"
    if shape is not None and shape.kind == "enumeration" and via == "override":
        # An explicit mode is the caller's instruction, like an explicit scope
        # (the eval harness forces each mode in turn): run that mode unfaceted,
        # and report the run as the single question it was answered as.
        shape = None
    # A comparison routed to timeline runs unfaceted (timeline ignores facets,
    # see _shape_fields/_grounded below): skip the rewrite call. An enumeration
    # always rewrites regardless of `mode` -- it never dispatches through the
    # mode functions, so "the mode that will run" for it is never timeline.
    if (shape is not None and shape.kind in ("comparison", "enumeration")
            and not (shape.kind == "comparison" and mode == "timeline")):
        # Each side retrieves with its own standalone question (one cheap map-tier
        # call); a side whose rewrite failed retrieves with the original question.
        # Answering prompts keep the original question.
        shape = replace(shape, facets=await facet_queries(map_client, map_model, q,
                                                          shape.facets))
    if shape is not None and shape.kind == "enumeration":
        # The classifier still runs (it labels the envelope's mode), but an
        # enumeration checks every content vendor instead of running that mode.
        # It has no detected scope, so there is nothing to relax. Temporal wording
        # classifies as timeline, but the enumeration's result is a prose answer,
        # not a timeline -- normalizing it as one would discard the answer -- so
        # it is labelled global, the map-reduce it most resembles.
        raw = await enum_mod.enumerate_vendors(
            graphiti, driver, embedder, map_client, map_model, synth_client,
            synth_model, q=q, facets=shape.facets, group_id=settings.group_id,
            settings=settings)
        label: Mode = "global" if mode == "timeline" else mode
        env = _normalize(label, via, None, raw, q, scope)
        env["routing"]["path"] = "enumeration"
        _shape_fields(env, shape, raw, requested)
        env["freshness"] = await freshness_mod.freshness(
            driver, settings.group_id, reports=True)
        return env
    facets = shape.facets if shape is not None and shape.kind == "comparison" else None
    args = (graphiti, driver, embedder, synth_client, synth_model, map_client, map_model)
    first = mode
    mode, fallback_from, raw = await _run_mode(first, *args, q=q, scope=scope,
                                               settings=settings, facets=facets)
    env_scope = scope
    relaxed = False
    first_degraded: list[str] = []
    if scope.source == "detected" and not _grounded(mode, raw):
        first_degraded = list(raw.get("degraded_facets") or [])
        # A DETECTED scope is a guess from names in the question; when the scoped
        # path grounds nothing, answer unscoped rather than refuse where the
        # unscoped system would have answered (final review). An explicit scope
        # is the caller's instruction and is never relaxed. The re-run is single
        # and unscoped: the facets were derived from the same detected names.
        mode, fallback_from, raw = await _run_mode(first, *args, q=q, scope=Scope(),
                                                   settings=settings)
        relaxed = True
    env = _normalize(mode, via, fallback_from, raw, q, env_scope)
    if relaxed:
        env["scope"] = {**scope.as_dict(), "source": "detected-relaxed"}
    # A relaxed re-run, or a timeline run (which ignores facets), never balanced
    # across facets: report it as the single question it was answered as.
    answered_as = None if relaxed or mode == "timeline" else shape
    _shape_fields(env, answered_as, raw, requested)
    if relaxed and first_degraded:
        # A failure never reads as absence: the scoped run's could-not-check
        # facets survive the unscoped re-run.
        env["degraded_facets"] = first_degraded
        note = could_not_check_note(first_degraded)
        if note not in env["answer"]:
            env["answer"] = f"{env['answer']}\n\n{note}"
    env["freshness"] = await freshness_mod.freshness(
        driver, settings.group_id, reports=mode in ("global", "drift"))
    return env


async def _run_mode(mode: Mode, graphiti, driver, embedder, synth_client, synth_model,
                    map_client, map_model, *, q, scope: Scope, settings,
                    facets=None) -> tuple[Mode, str | None, dict]:
    """One mode plus its fallback (local->drift when nothing was retrieved;
    global->local when global grounded nothing), all with the same scope."""
    raw = await _dispatch(mode, graphiti, driver, embedder, synth_client, synth_model,
                          map_client, map_model, q=q, scope=scope, settings=settings,
                          facets=facets)
    fallback_from: str | None = None
    if mode == "local" and raw.get("retrieved") == 0:
        fallback_from = "local"
        mode = "drift"
        raw = await _dispatch("drift", graphiti, driver, embedder, synth_client,
                              synth_model, map_client, map_model, q=q, scope=scope,
                              settings=settings, facets=facets)
    # Global falls back to local when it grounded nothing: no community used, OR
    # communities used but no citation (the reduce step refused -- e.g. a scoped
    # comparison whose second vendor's community failed to map). Local runs with
    # the same scope and answers such questions from facts directly.
    elif mode == "global" and (not raw.get("communities_used") or not raw.get("citations")):
        fallback_from = "global"
        mode = "local"
        raw = await _dispatch("local", graphiti, driver, embedder, synth_client,
                              synth_model, map_client, map_model, q=q, scope=scope,
                              settings=settings, facets=facets)
    return mode, fallback_from, raw
