"""Which vendors/products a question is about (spec §4.1).

Built from the STRUCTURAL layer only: graphiti's extracted entities also carry
:Vendor/:Product labels, so nodes are selected by the HAS_PRODUCT edge and a
non-null `id`, never by label alone. Detection is deterministic: whole-word,
case-insensitive, longest match first, so a product name is never split into
the vendor names it contains ("Veeam Backup for Microsoft 365").
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
    "RETURN v.name AS vendor, p.name AS product")
_EPISODES = (
    "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(:Source)"
    "-[:HAS_ARTICLE]->(:Article)-[:HAS_EPISODE]->(e:Episodic) "
    "WHERE v.id IS NOT NULL AND (v.name IN $vendors OR p.name IN $products) "
    "RETURN collect(DISTINCT e.uuid) AS u")


@dataclass(frozen=True)
class Scope:
    vendors: tuple[str, ...] = ()
    products: tuple[str, ...] = ()
    source: str = "none"

    def is_empty(self) -> bool:
        return not self.vendors and not self.products

    def as_dict(self) -> dict:
        return {"vendors": list(self.vendors), "products": list(self.products),
                "source": self.source}


class UnknownScopeName(ValueError):
    def __init__(self, names: list[str]) -> None:
        super().__init__(f"unknown vendor/product: {', '.join(names)}")
        self.names = names


class ScopeResolver:
    def __init__(self, vendors: list[str], products: list[tuple[str, str]],
                 aliases: dict[str, str]) -> None:
        self._vendors = {v.lower(): v for v in vendors}
        self._products = {p.lower(): p for p, _ in products}
        self._vendor_of = {p: v for p, v in products}
        terms: dict[str, tuple[str, str]] = {}          # lower term -> (kind, canonical)
        for v in vendors:
            terms[v.lower()] = ("vendor", v)
        for p, _ in products:
            terms[p.lower()] = ("product", p)
        for alias, target in aliases.items():
            hit = self._canonical(target)
            if hit is None:
                logger.warning("scope alias %r -> %r: target is not an ingested "
                                "vendor/product; ignored", alias, target)
                continue
            terms[alias.lower()] = hit
        self._terms = terms
        alternation = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
        self._re = re.compile(rf"(?<![\w-])({alternation})(?![\w])", re.IGNORECASE) \
            if terms else None

    def _canonical(self, name: str) -> tuple[str, str] | None:
        n = name.strip().lower()
        if n in self._products:
            return ("product", self._products[n])
        if n in self._vendors:
            return ("vendor", self._vendors[n])
        return None

    def vendor_of(self, product: str) -> str | None:
        return self._vendor_of.get(product)

    def detect(self, q: str) -> Scope:
        if self._re is None:
            return Scope()
        vendors: list[str] = []
        products: list[str] = []
        for m in self._re.finditer(q):
            kind, name = self._terms[m.group(1).lower()]
            bucket = products if kind == "product" else vendors
            if name not in bucket:
                bucket.append(name)
        if not vendors and not products:
            return Scope()
        return Scope(tuple(vendors), tuple(products), "detected")

    def resolve(self, q: str, *, vendors: list[str] | None = None,
                products: list[str] | None = None, disabled: bool = False) -> Scope:
        if disabled:
            return Scope()
        if vendors or products:
            unknown: list[str] = []
            vs: list[str] = []
            ps: list[str] = []
            for name in vendors or []:
                hit = self._canonical(name)
                if hit is None or hit[0] != "vendor":
                    unknown.append(name)
                elif hit[1] not in vs:
                    vs.append(hit[1])
            for name in products or []:
                hit = self._canonical(name)
                if hit is None or hit[0] != "product":
                    unknown.append(name)
                elif hit[1] not in ps:
                    ps.append(hit[1])
            if unknown:
                raise UnknownScopeName(unknown)
            return Scope(tuple(vs), tuple(ps), "explicit")
        return self.detect(q)

    @classmethod
    async def load(cls, driver, aliases_path: Path | None = None) -> "ScopeResolver":
        records, _, _ = await driver.execute_query(_CATALOG)
        pairs = [(r["product"], r["vendor"]) for r in records]
        vendors = sorted({v for _, v in pairs})
        aliases = json.loads((aliases_path or _ALIASES).read_text())
        return cls(vendors, pairs, aliases)


async def scope_episode_uuids(driver, scope: Scope) -> set[str]:
    """Episode uuids of every article in scope (vendor named directly OR product
    named). Empty scope -> empty set; callers skip filtering on an empty scope."""
    if scope.is_empty():
        return set()
    records, _, _ = await driver.execute_query(
        _EPISODES, vendors=list(scope.vendors), products=list(scope.products))
    return set(records[0]["u"]) if records else set()
