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
    "to another. When comparing vendors, keep each vendor's claims separate. Keep "
    "products apart within a vendor too: a sentence that names one product cites "
    "only facts labelled with that product; a fact labelled with another product is "
    "stated under that product's name.\n")

# Run 9 (2026-10-13): the judge's 4s were a fact stripped of its qualifier (object
# storage immutability presented as the hardened repository's), names no fact states
# (NetApp from "SVM"/"cluster" facts), and 5-11 near-duplicate markers on one claim.
# Every synthesis prompt carries this, single and multi-side alike.
CLAIM_RULES = (
    "State only what each fact says, keeping its qualifiers: a fact about a specific "
    "component, storage type, mode, version or condition stays about that one; never "
    "present it as true more generally. Do not infer names, capabilities or "
    "relationships that no fact states. Cite each claim with the one or two markers "
    "of the facts that state it most directly; when several facts say the same thing, "
    "cite two of them at most. A sentence never carries more than three markers: "
    "split it so each detail cites its own fact. Write each marker individually, "
    "never as a range such as [1]-[3].\n")

# Run 9: the one misattribution and an unsupported inference were both in a closing
# "Comparison" paragraph -- "Both aim to protect against accidental deletion and
# malicious activity [5] [15]" pinned a Veeam point on an Azure fact. Multi-side
# prompts only.
COMPARISON_RULES = (
    "If you compare the sides, write contrasts only: each comparison sentence sets one "
    "side's point against another side's, each in its own clause citing only that "
    "side's markers. Never make one claim about several sides together (\"Both ...\", "
    "\"Each ...\", \"All ...\"), and introduce no point that the side sections did "
    "not already state.\n")


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
    if any(v in scope.vendors or p in scope.products for v, p in pairs_of(sources)):
        return True
    return bool(scope.sources) and any(s.get("source") in scope.sources for s in sources)
