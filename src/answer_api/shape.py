"""Question shape and facets (spec 2026-10-12 §2).

A question is SINGLE (zero or one named side), a COMPARISON (two or more named
sides), or an ENUMERATION (cross-vendor wording with no explicit scope: every vendor
with content is checked). Each side is a FACET -- a label plus its own Scope -- and
every retrieval mode balances across facets. Deterministic: no LLM.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

from answer_api.attribution import in_scope
from answer_api.scope import _CROSS_VENDOR, Scope, ScopeResolver


@dataclass(frozen=True)
class Facet:
    label: str
    scope: Scope


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


def classify_shape(q: str, resolver: ScopeResolver, scope: Scope, *,
                   scope_disabled: bool = False) -> QuestionShape:
    if scope_disabled:
        return QuestionShape("single", (Facet("", Scope()),))
    if scope.source == "explicit":
        return QuestionShape("single", (Facet(_label_of(scope), scope),))
    if _CROSS_VENDOR.search(q) and resolver.content_vendors:
        return QuestionShape("enumeration", tuple(
            Facet(v, Scope(vendors=(v,), source="detected"))
            for v in resolver.content_vendors))
    facets = _facets_from_mentions(resolver, q)
    if len(facets) >= 2:
        return QuestionShape("comparison", tuple(facets))
    if len(facets) == 1:
        # The single facet carries the caller's (detected) scope unchanged, so the
        # single path is exactly today's.
        return QuestionShape("single", (Facet(facets[0].label, scope),))
    return QuestionShape("single", (Facet("", scope),))


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
