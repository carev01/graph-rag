"""Question shape and facets (spec 2026-10-12 §2).

A question is SINGLE (zero or one named side, or cross-vendor wording about one
named subject: "Which vendors does Veeam integrate with?"), a COMPARISON (two or
more named sides), or an ENUMERATION (cross-vendor wording with no explicit scope:
every vendor with content is checked). Each side is a FACET -- a label plus its own Scope -- and
every retrieval mode balances across facets. Classification is deterministic (no
LLM); `facet_queries` then makes ONE cheap-tier call per comparison/enumeration to
give each facet its own retrieval question (single questions never call it).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from graph_extract.usage import usable_content

from answer_api.attribution import in_scope
from answer_api.scope import _CROSS_VENDOR, Scope, ScopeResolver


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Facet:
    label: str
    scope: Scope
    # The facet's own retrieval question ("" = not rewritten: retrieve with the
    # original question). Only retrieval uses it; answering prompts keep the
    # original question.
    query: str = ""
    # The side's documented products (ENUMERATION facets only; comparison and
    # single facets keep ()). Only the side-question rewrite uses them: a vendor's
    # documentation is organised by product, so its side question names one.
    products: tuple[str, ...] = ()


@dataclass(frozen=True)
class QuestionShape:
    kind: Literal["single", "comparison", "enumeration"]
    facets: tuple[Facet, ...]


def _label_of(scope: Scope) -> str:
    return ", ".join((*scope.vendors, *scope.products, *scope.sources))


def _vendors_covered(scope: Scope, resolver: ScopeResolver) -> set[str]:
    out = {resolver.vendor_of(p) for p in scope.products}
    out |= {resolver.vendor_of_source(s) for s in scope.sources}
    return {v for v in out if v}


def _facets_from_mentions(resolver: ScopeResolver, q: str) -> list[Facet]:
    facets: list[Facet] = []
    for label, scope in resolver.detect_mentions(q):
        if all(f.scope != scope for f in facets):
            facets.append(Facet(label, scope))
    # A bare vendor facet is subsumed by a product/source facet of that vendor:
    # "Veeam's Veeam ONE" is one side, not two.
    covered = set().union(*(_vendors_covered(f.scope, resolver) for f in facets)) \
        if facets else set()
    return [f for f in facets
            if not (f.scope.vendors and not f.scope.products and not f.scope.sources
                    and set(f.scope.vendors) <= covered)]


# A cross-vendor question whose grammatical SUBJECT is a named vendor or product
# ("Which vendors does Veeam integrate with?") asks about that one entity: it is
# SINGLE, scoped to it. The named mention must start right where this match ends;
# a named entity in any other position (object, workload: "Which vendors integrate
# with Veeam?") leaves the question an enumeration (run 8, task 17).
_NAMED_SUBJECT = re.compile(
    r"\b(which|what)\s+(\w+\s+){0,2}vendors?\s+"
    r"(does|do|did|can|could|is|are|was|were|has|have|will|would)\s+",
    re.IGNORECASE)


def _named_subject(q: str, resolver: ScopeResolver) -> Facet | None:
    """The facet of the named vendor/product a cross-vendor question is asked
    ABOUT, or None (see _NAMED_SUBJECT)."""
    ends = {m.end() for m in _NAMED_SUBJECT.finditer(q)}
    if not ends:
        return None
    for start, label, scope in resolver.detect_mention_spans(q):
        if start in ends:
            return Facet(label, scope)
    return None


def classify_shape(q: str, resolver: ScopeResolver, scope: Scope, *,
                   scope_disabled: bool = False) -> QuestionShape:
    if scope_disabled:
        return QuestionShape("single", (Facet("", Scope()),))
    if scope.source == "explicit":
        return QuestionShape("single", (Facet(_label_of(scope), scope),))
    if _CROSS_VENDOR.search(q):
        subject = _named_subject(q, resolver)
        if subject is not None:
            return QuestionShape("single", (subject,))
    if _CROSS_VENDOR.search(q) and resolver.content_vendors:
        return QuestionShape("enumeration", tuple(
            Facet(v, Scope(vendors=(v,), source="detected"),
                  products=resolver.products_of(v))
            for v in resolver.content_vendors))
    facets = _facets_from_mentions(resolver, q)
    if len(facets) >= 2:
        return QuestionShape("comparison", tuple(facets))
    if len(facets) == 1:
        # The single facet carries the caller's (detected) scope unchanged, so the
        # single path is exactly today's.
        return QuestionShape("single", (Facet(facets[0].label, scope),))
    return QuestionShape("single", (Facet("", scope),))


def retrieval_scope(shape: QuestionShape, scope: Scope) -> Scope:
    """The scope to retrieve with: a single shape whose facet carries a scope the
    request lacks (a named-subject cross-vendor question -- `resolve` returns an
    empty Scope for cross-vendor wording) retrieves with the facet's scope;
    otherwise `scope` unchanged."""
    if (shape.kind == "single" and scope.is_empty()
            and not shape.facets[0].scope.is_empty()):
        return shape.facets[0].scope
    return scope


def group_by_facet(items: list[dict], facets: Sequence[Facet]
                   ) -> list[tuple[Facet, list[dict]]]:
    groups: list[tuple[Facet, list[dict]]] = [(f, []) for f in facets]
    for item in items:
        for f, bucket in groups:
            if in_scope(item.get("sources", []), f.scope):
                bucket.append(item)
                break
    return groups


def could_not_check_note(labels: Sequence[str]) -> str:
    return f"Could not check: {', '.join(labels)}." if labels else ""


def raise_if_cancelled(results: Iterable[object]) -> None:
    """After `asyncio.gather(..., return_exceptions=True)`: re-raise the first
    result that is a BaseException but not an Exception (CancelledError,
    KeyboardInterrupt, ...). Cancellation propagates; it is never a degraded facet.
    Callers then treat only `Exception` instances as degraded."""
    for r in results:
        if isinstance(r, BaseException) and not isinstance(r, Exception):
            raise r


_FACET_QUERIES_PROMPT = (
    "Each SIDE below is a vendor or product whose OWN documentation will be searched "
    "separately. Rewrite the QUESTION into one standalone question per SIDE, asked "
    "directly about that side alone, as you would ask its documentation: start with "
    "the side's name as the subject (e.g. \"How does <side> ...\", \"Does <side> ...\"). "
    "Keep the QUESTION's intent and its specific terms (feature names, workloads, "
    "settings). Drop cross-vendor wording such as \"which vendors\", \"compare\", "
    "\"differ\" or \"each\". Do not answer, do not add facts, and do not mention any other "
    "side or vendor. When a side lists its documented products, make the product the "
    "question concerns the subject (e.g. \"How does Azure Backup …\"); if several "
    "apply, name the most relevant one or two. Respond with ONLY JSON mapping each "
    "side's label, exactly as written below, to its question: "
    "{{\"<label>\": \"<question>\", ...}}.\n\n"
    "SIDES:\n{sides}\n\nQUESTION: {q}\nJSON:")


def _side_line(f: Facet) -> str:
    if f.products:
        return f"- {f.label} (its documented products: {', '.join(f.products)})"
    return f"- {f.label}"


async def facet_queries(client, model: str, q: str,
                        facets: Sequence[Facet]) -> tuple[Facet, ...]:
    """One cheap-tier call: rewrite `q` into a standalone question per facet, each
    about that side only, and return the facets with `query` set. Each facet whose
    rewrite is unusable (failed call, empty reply, no JSON, missing/empty label)
    keeps an empty `query` with a warning -- a fallback, never an absence: that
    facet's retrieval still runs, with the original question (every caller uses
    `f.query or q`). The query stays empty rather than becoming `q` so a caller can
    tell a fallback from a rewrite: the enumeration check must not ask a single
    vendor the bare cross-vendor question (final review)."""
    # Deferred: global_search imports this module.
    from answer_api.global_search import _extract_json

    prompt = _FACET_QUERIES_PROMPT.format(
        sides="\n".join(_side_line(f) for f in facets), q=q)
    obj: dict | None = None
    try:
        # No reasoning parameter: the map tier is a non-reasoning model (sending
        # one can switch thinking on and blank the reply). Headroom anyway, so a
        # reasoning model is not starved: ~100 tokens per side, at least 1500.
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=max(1500, 100 * len(facets)),
            messages=[{"role": "user", "content": prompt}])
        content = usable_content(resp)
        if content is None:
            logger.warning("facet question rewrite: %s returned no usable content; "
                           "every facet retrieves with the original question", model)
        else:
            obj = _extract_json(content)
            if obj is None:
                logger.warning("facet question rewrite: reply is not JSON; every facet "
                               "retrieves with the original question: %r", content[:200])
    except Exception:
        logger.warning("facet question rewrite failed; every facet retrieves with the "
                       "original question", exc_info=True)
    if obj is None:
        return tuple(replace(f, query="") for f in facets)
    out: list[Facet] = []
    for f in facets:
        rewritten = obj.get(f.label)
        if rewritten is None:
            # A model reply sometimes echoes the label with different case or
            # stray whitespace ("aws backup " for "AWS Backup"): match before
            # falling back to the original question.
            target = f.label.strip().casefold()
            rewritten = next(
                (v for k, v in obj.items()
                 if isinstance(k, str) and k.strip().casefold() == target), None)
        if isinstance(rewritten, str) and rewritten.strip():
            out.append(replace(f, query=rewritten.strip()))
        else:
            logger.warning("facet question rewrite: no question for facet %r; it "
                           "retrieves with the original question", f.label)
            out.append(replace(f, query=""))
    return tuple(out)
