"""Vendor/product attribution for fact lines and answers (spec §3).

Every label is derived from provenance sources (Provenance.resolve_citations:
fact -> episode -> article -> source -> product -> vendor); the LLM only reads
them (invariant #2 extended to labels)."""
from __future__ import annotations

from answer_api.scope import Scope

ATTRIBUTION_RULES = (
    "Each fact is prefixed with the vendor and product whose documentation states "
    "it, as (Vendor · Product). Attribute every claim to the vendor/product shown "
    "on the facts it cites. Never apply a fact labelled with one vendor or product "
    "to another. When comparing vendors, keep each vendor's claims separate.\n")


def pairs_of(sources: list[dict]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for s in sources:
        v, p = s.get("vendor"), s.get("product")
        if v and p and (v, p) not in out:
            out.append((v, p))
    return out


def label(sources: list[dict]) -> str:
    pairs = pairs_of(sources)
    return "(" + "; ".join(f"{v} · {p}" for v, p in pairs) + ")" if pairs else ""


def fact_line(n: int, fact: str, sources: list[dict]) -> str:
    lab = label(sources)
    return f"[{n}] {lab} {fact}" if lab else f"[{n}] {fact}"


def applies_to(citations: list[dict]) -> list[dict]:
    facts: dict[str, int] = {}
    products: dict[str, list[str]] = {}
    for c in citations:
        seen: set[str] = set()
        for v, p in pairs_of(c.get("sources", [])):
            products.setdefault(v, [])
            if p not in products[v]:
                products[v].append(p)
            if v not in seen:
                facts[v] = facts.get(v, 0) + 1
                seen.add(v)
    order = sorted(facts, key=lambda v: (-facts[v], list(facts).index(v)))
    return [{"vendor": v, "products": products[v], "facts": facts[v]} for v in order]


def in_scope(sources: list[dict], scope: Scope) -> bool:
    if scope.is_empty():
        return True
    return any(v in scope.vendors or p in scope.products for v, p in pairs_of(sources))
