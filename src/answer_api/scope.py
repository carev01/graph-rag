"""Which vendors/products/sources a question is about (spec §4.1).

Built from the STRUCTURAL layer only: graphiti's extracted entities also carry
:Vendor/:Product labels, so nodes are selected by the HAS_PRODUCT edge and a
non-null `id`, never by label alone. Detection is deterministic: whole-word,
case-insensitive, longest match first, so a product name is never split into
the vendor names it contains ("Veeam Backup for Microsoft 365").

Sources (a product's document sets) scope too, but only DISTINCTIVE ones are detected:
a source name must be unique in the catalog and contain its vendor's name ("Veeam
Agent for Linux", "Veeam App for Splunk"). Most source names are generic ("User
Guide", "Help Center", "Docs") and detecting them would scope unrelated questions;
those are reachable only through an alias or an explicit `source=`. Added 2026-10-10:
the Veeam Agent guides are sources under Veeam Backup & Replication, so product
scoping could not isolate them (Tier 1 eval).
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)
_ALIASES = Path(__file__).with_name("scope_aliases.json")

_CATALOG = (
    "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product) "
    "WHERE v.id IS NOT NULL AND p.id IS NOT NULL "
    "OPTIONAL MATCH (p)-[:HAS_SOURCE]->(s:Source) WHERE s.id IS NOT NULL "
    "RETURN v.name AS vendor, p.name AS product, collect(s.name) AS sources")
_EPISODES = (
    "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(s:Source)"
    "-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(e:Episodic) "
    "WHERE v.id IS NOT NULL AND p.id IS NOT NULL "
    "AND (toLower(v.name) IN $vendors OR toLower(p.name) IN $products "
    "OR toLower(s.name) IN $sources) "
    "RETURN collect(DISTINCT e.uuid) AS u")
# Candidate-bounded (request path): check only the episodes of the retrieved
# edges, walking UP from each via the Episodic uuid index -- the unbounded form
# above returns every episode a vendor has (~10^5 at bootstrap scale) per call.
_EPISODES_AMONG = (
    "UNWIND $eps AS u "
    "MATCH (e:Episodic {uuid: u})<-[:HAS_EPISODE]-(:Article)<-[:HAS_ARTICLE]-(s:Source)"
    "<-[:HAS_SOURCE]-(p:Product)<-[:HAS_PRODUCT]-(v:Vendor) "
    "WHERE v.id IS NOT NULL AND p.id IS NOT NULL "
    "AND (toLower(v.name) IN $vendors OR toLower(p.name) IN $products "
    "OR toLower(s.name) IN $sources) "
    "RETURN collect(DISTINCT u) AS u")


# Wording that makes a question explicitly cross-vendor. Such a question stays
# unscoped even when it names a platform or product: "Which backup vendors can
# protect Azure VMs?" is about Azure as a WORKLOAD, and scoping it to Microsoft's
# documentation would drop every other vendor's answer (final review, 2026-09-25).
#
# Deliberately narrow: "across regions/accounts", "a third-party KMS key", "which
# options in Azure Backup" are everyday SINGLE-vendor wording and must stay scoped
# (scoped re-review). Only a cross-vendor noun makes it cross-vendor.
_CROSS_VENDOR = re.compile(
    r"\bvendors\b"
    r"|\b(which|what|any|other|different|competing) vendor\b(?!-)"
    r"|\bacross (\w+ )?(vendors|providers|clouds|platforms|products|solutions)\b"
    r"|\bthird[- ]party (vendors?|tools|products|solutions|backup)\b"
    r"|\b(all|other|alternative|competing) (backup )?(products|solutions|vendors)\b",
    re.IGNORECASE)


# Words that, left over once the vendor's name is removed from a source name, make it
# generic rather than distinctive: "Keepit Platform" must not narrow "Does the Keepit
# platform back up Microsoft 365?" from the Keepit vendor to one of its sources (review).
_GENERIC_SOURCE_WORDS = frozenset({
    "platform", "docs", "documentation", "doc", "cloud", "console", "portal", "help",
    "center", "centre", "guide", "guides", "user", "admin", "administrator", "manual",
    "reference", "kb", "knowledge", "base", "support", "release", "notes", "the", "and"})

_warned_aliases: set[tuple[str, str]] = set()


def _distinctive(source: str, vendor: str) -> bool:
    """A source is detected only if its name carries the vendor's name AND something
    specific beyond it: "Veeam Agent for Linux" yes, "Keepit Platform" no."""
    name = source.lower()
    if not vendor or vendor.lower() not in name:
        return False
    rest = re.findall(r"[\w&+.-]+", name.replace(vendor.lower(), " "))
    return any(w not in _GENERIC_SOURCE_WORDS for w in rest)


@dataclass(frozen=True)
class Scope:
    vendors: tuple[str, ...] = ()
    products: tuple[str, ...] = ()
    source: str = "none"            # how the scope was set: explicit / detected / none
    sources: tuple[str, ...] = ()   # document sets (structural :Source names)

    def is_empty(self) -> bool:
        return not self.vendors and not self.products and not self.sources

    def as_dict(self) -> dict:
        return {"vendors": list(self.vendors), "products": list(self.products),
                "sources": list(self.sources), "source": self.source}


class UnknownScopeName(ValueError):
    def __init__(self, names: list[str]) -> None:
        super().__init__(f"unknown vendor/product/source: {', '.join(names)}")
        self.names = names


class ScopeResolver:
    def __init__(self, vendors: list[str], products: list[tuple[str, str]],
                 aliases: dict[str, str | list[str]],
                 sources: list[tuple[str, str]] | None = None) -> None:
        """`sources`: (source name, product name). Only sources whose name is unique
        across the catalog are addressable at all (explicitly or by alias); of those,
        only names containing their vendor's name are DETECTED (module docstring)."""
        self._vendors = {v.lower(): v for v in vendors}
        self._products = {p.lower(): p for p, _ in products}
        self._vendor_of = {p: v for p, v in products}
        counts: dict[str, int] = {}
        for name, _ in sources or []:
            counts[name.lower()] = counts.get(name.lower(), 0) + 1
        self._sources = {n.lower(): n for n, _ in sources or [] if counts[n.lower()] == 1}
        # Detection and alias-target resolution: on a name collision (a vendor and a
        # product sharing the same name, e.g. "Keepit"/"Keepit"), the vendor wins --
        # scoping to the vendor also covers the same-named product. A source never
        # takes a vendor's or product's name.
        terms: dict[str, list[tuple[str, str]]] = {}    # lower term -> [(kind, canonical)]
        for v in vendors:
            terms[v.lower()] = [("vendor", v)]
        for p, _ in products:
            terms.setdefault(p.lower(), [("product", p)])
        for name, product in sources or []:
            if name.lower() in self._sources and _distinctive(
                    name, self._vendor_of.get(product, "")):
                terms.setdefault(name.lower(), [("source", name)])
        self._alias_targets: dict[str, list[tuple[str, str]]] = {}
        for alias, target in aliases.items():
            hits = []
            for t in [target] if isinstance(target, str) else target:
                hit = self._canonical(t)
                if hit is None:
                    # Once per process: the resolver reloads every scope_reload_seconds,
                    # and an alias may name sources not ingested yet.
                    if (alias, t) not in _warned_aliases:
                        _warned_aliases.add((alias, t))
                        logger.warning("scope alias %r -> %r: target is not an ingested "
                                       "vendor/product/source; ignored", alias, t)
                else:
                    hits.append(hit)
            if hits:
                terms[alias.lower()] = hits
                self._alias_targets[alias.lower()] = hits
        self._terms = terms
        alternation = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
        self._re = re.compile(rf"(?<![\w-])({alternation})(?![\w])", re.IGNORECASE) \
            if terms else None

    def _canonical(self, name: str) -> tuple[str, str] | None:
        """Resolve an alias TARGET to (kind, canonical name); vendor wins a collision."""
        n = name.strip().lower()
        if n in self._vendors:
            return ("vendor", self._vendors[n])
        if n in self._products:
            return ("product", self._products[n])
        if n in self._sources:
            return ("source", self._sources[n])
        return None

    def _explicit(self, name: str, kind: str) -> list[str] | None:
        """Resolve an explicit vendors=/products=/sources= name by its DECLARED kind:
        the name (or an alias of it) must be of that kind, collisions notwithstanding."""
        n = name.strip().lower()
        table = {"vendor": self._vendors, "product": self._products,
                 "source": self._sources}[kind]
        if n in table:
            return [table[n]]
        hits = [c for k, c in self._alias_targets.get(n, []) if k == kind]
        return hits or None

    def vendor_of(self, product: str) -> str | None:
        return self._vendor_of.get(product)

    def detect(self, q: str) -> Scope:
        if self._re is None or _CROSS_VENDOR.search(q):
            return Scope()
        found: dict[str, list[str]] = {"vendor": [], "product": [], "source": []}
        for m in self._re.finditer(q):
            for kind, name in self._terms[m.group(1).lower()]:
                if name not in found[kind]:
                    found[kind].append(name)
        if not any(found.values()):
            return Scope()
        return Scope(tuple(found["vendor"]), tuple(found["product"]), "detected",
                     tuple(found["source"]))

    def resolve(self, q: str, *, vendors: list[str] | None = None,
                products: list[str] | None = None, sources: list[str] | None = None,
                disabled: bool = False) -> Scope:
        if disabled:
            return Scope()
        if vendors or products or sources:
            unknown: list[str] = []
            out: dict[str, list[str]] = {"vendor": [], "product": [], "source": []}
            for kind, names in (("vendor", vendors), ("product", products),
                                ("source", sources)):
                for name in names or []:
                    hits = self._explicit(name, kind)
                    if hits is None:
                        unknown.append(name)
                        continue
                    for h in hits:
                        if h not in out[kind]:
                            out[kind].append(h)
            if unknown:
                raise UnknownScopeName(unknown)
            return Scope(tuple(out["vendor"]), tuple(out["product"]), "explicit",
                         tuple(out["source"]))
        return self.detect(q)

    @classmethod
    async def load(cls, driver, aliases_path: Path | None = None) -> "ScopeResolver":
        records, _, _ = await driver.execute_query(_CATALOG)
        pairs = [(r["product"], r["vendor"]) for r in records]
        vendors = sorted({v for _, v in pairs})
        sources = [(name, r["product"]) for r in records for name in r["sources"] if name]
        aliases = json.loads((aliases_path or _ALIASES).read_text())
        return cls(vendors, pairs, aliases, sources)


async def scope_episode_uuids(driver, scope: Scope,
                              candidates: list[str] | None = None) -> set[str]:
    """Episode uuids of every article in scope (vendor named directly OR product
    named OR source named). Empty scope -> empty set; callers skip filtering on an empty scope.

    Case-insensitive: matched against `toLower(v.name)`/`toLower(p.name)` in
    `_EPISODES`, so the vendor/product names here are lower-cased on this side
    too -- a caller passing a query-param-cased name (`vendor=aws`) must still
    match the canonical structural name ("AWS")."""
    if scope.is_empty():
        return set()
    params = {"vendors": [v.lower() for v in scope.vendors],
              "products": [p.lower() for p in scope.products],
              "sources": [x.lower() for x in scope.sources]}
    if candidates is not None:
        # `candidates` bounds the answer: which of THESE episodes are in scope.
        # Every request-path caller passes its retrieved edges' episodes; only
        # compat checks ask for the whole scope.
        if not candidates:
            return set()
        records, _, _ = await driver.execute_query(
            _EPISODES_AMONG, eps=list(dict.fromkeys(candidates)), **params)
    else:
        records, _, _ = await driver.execute_query(_EPISODES, **params)
    return set(records[0]["u"]) if records else set()
