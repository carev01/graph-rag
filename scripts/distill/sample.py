"""Deterministic stratified sample for the Opus distillation (spec section 5).

`select_sample(driver, ...)` picks about `n_total` articles that production has
already extracted (a live `HAS_EPISODE` edge to a non-removed episode), split by
article into train and held-out. Every query is read-only (`routing_=READ`; the
harness passes a driver whose sessions are READ access mode as well).

**Strata** are vendor x product (the structural `Product` node is the product
family: the graph carries no coarser grouping). Each article's chain is the one
production's ingest driver uses for its DOCUMENT CONTEXT: the lowest source id
(`IngestDriver._document_context`).

**Allocation.** Each vendor first gets a floor of `VENDOR_FLOOR` articles (or all
it has), the rest of `n_total` is apportioned to vendors in proportion to their
article counts (largest remainder, ties by size then name), and each vendor's
share is apportioned the same way over its products.

**Hard cases** are oversampled 2:1 inside each stratum: a weighted shuffle
(Efraimidis-Spirakis keys `u ** (1 / w)`, `w = 2` for a hard article, 1 for the
rest) over the stratum's articles in `ORDER BY a.id` order, with a `random.Random`
seeded by `(seed, stratum)`; the first k of that shuffle are taken. An article is
hard when any of these holds:

- `affected`: it is in the 2026-10-17 affected set -- the foreign-unnamed-subject
  predicate, `scripts/extraction_proof.AFFECTED_ARTICLES_QUERY`, reused verbatim
  (both of its halves: example-product subjects and other-product subjects);
- `unnamed_chunk`: one of its live chunks names no structural product at all;
- `restore_title`: its title contains "Restore" or "Restoring" (any case);
- `table`: one of its live chunks holds a markdown table rule (`|---`).

**Held out.** `n_heldout` is apportioned over the strata in proportion to what
each contributed; inside a stratum a second seeded permutation is taken and its
LAST h articles are held out.

**Order of the returned list** (and of `sample.json`): train first, then held
out; within each, a round-robin over vendors (by name) and, within a vendor, over
its strata, so the first few articles -- what `harness round --articles N` runs
for a pilot -- span vendors and are all train.
"""
from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neo4j import RoutingControl

from scripts.extraction_proof import (
    AFFECTED_ARTICLES_QUERY,
    EXAMPLE_SUBJECTS,
    PRODUCT_NAMES_QUERY,
)

logger = logging.getLogger(__name__)

READ = RoutingControl.READ
DEFAULT_VENDORS = ("Veeam", "Microsoft", "AWS", "Cohesity")
VENDOR_FLOOR = 5
HARD_WEIGHT = 2.0
# AFFECTED_ARTICLES_QUERY takes a LIMIT; the whole set is wanted here.
AFFECTED_LIMIT = 1_000_000
TABLE_RULE = "|---"
RESTORE_WORDS = ("restore", "restoring")

# Every article of the requested vendors with at least one live episode, with the
# chain production's ingest driver would pick (lowest source id; its query has no
# id filter either, so neither does this one) and the chunk-level hard-case flags,
# computed server-side so chunk text never leaves the database.
ARTICLES_QUERY = (
    "MATCH (v0:Vendor)-[:HAS_PRODUCT]->(:Product)-[:HAS_SOURCE]->(:Source)"
    "-[:HAS_ARTICLE]->(a:Article) "
    "WHERE v0.id IS NOT NULL AND toLower(v0.name) IN $vendors "
    "AND coalesce(a.removed, false) = false "
    "WITH DISTINCT a "
    "CALL (a) { "
    "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(so:Source)"
    "-[:HAS_ARTICLE]->(a) "
    "RETURN v.name AS vendor, p.name AS product, so.name AS source "
    "ORDER BY so.id LIMIT 1 "
    "} "
    "WITH a, vendor, product, source WHERE toLower(vendor) IN $vendors "
    "MATCH (a)-[he:HAS_EPISODE]->(e:Episodic) "
    "WHERE coalesce(he.superseded, false) = false AND coalesce(e.removed, false) = false "
    "WITH a, vendor, product, source, collect(toLower(coalesce(e.content, ''))) AS texts "
    "RETURN a.id AS id, a.title AS title, vendor, product, source, "
    "       any(t IN texts WHERE t CONTAINS $table_rule) AS has_table, "
    "       any(t IN texts WHERE none(pn IN $product_names WHERE t CONTAINS pn)) "
    "         AS unnamed_chunk "
    "ORDER BY id")


@dataclass(frozen=True)
class SampleArticle:
    article_id: str
    vendor: str
    product: str
    source: str
    split: str                      # "train" | "heldout"
    stratum: str                    # "<vendor> / <product>"
    hard_cases: tuple[str, ...] = field(default=())


@dataclass
class _Candidate:
    article_id: str
    title: str
    vendor: str
    product: str
    source: str
    hard_cases: tuple[str, ...]

    @property
    def stratum(self) -> str:
        return stratum_of(self.vendor, self.product)


def stratum_of(vendor: str, product: str) -> str:
    return f"{vendor} / {product}"


def apportion(total: int, sizes: dict[str, int],
              floors: dict[str, int] | None = None) -> dict[str, int]:
    """Split `total` over `sizes`' keys: each first gets its floor (capped at its
    size), the rest goes in proportion to size, largest remainder first (ties by
    larger size, then key). Never more than a key's size; the result sums to
    `min(total, sum(sizes))`. Deterministic."""
    floors = floors or {}
    alloc = {k: min(floors.get(k, 0), n) for k, n in sizes.items()}
    remaining = min(total, sum(sizes.values())) - sum(alloc.values())
    if remaining < 0:
        raise ValueError(f"floors {sum(alloc.values())} exceed the total {total}")
    while remaining > 0:
        open_ = sorted(k for k in sizes if alloc[k] < sizes[k])
        if not open_:
            break
        weight = sum(sizes[k] for k in open_)
        quotas = {k: remaining * sizes[k] / weight for k in open_}
        given = 0
        for k in open_:
            g = min(int(quotas[k]), sizes[k] - alloc[k])
            alloc[k] += g
            given += g
        left = remaining - given
        for k in sorted(open_, key=lambda k: (-(quotas[k] % 1), -sizes[k], k)):
            if left == 0:
                break
            if alloc[k] < sizes[k]:
                alloc[k] += 1
                left -= 1
        remaining = left
    return alloc


def _weighted_shuffle(items: Sequence[_Candidate], rng: random.Random) -> list[_Candidate]:
    """Efraimidis-Spirakis: hard articles carry weight 2, so they come early about
    twice as often. `items` must already be in a deterministic order (by id)."""
    keyed = []
    for c in items:
        w = HARD_WEIGHT if c.hard_cases else 1.0
        keyed.append((rng.random() ** (1.0 / w), c))
    keyed.sort(key=lambda kc: -kc[0])
    return [c for _, c in keyed]


async def _records(driver: Any, query: str, params: dict[str, Any]) -> list[Any]:
    result = await driver.execute_query(query, parameters_=params, routing_=READ)
    return list(result.records)


async def read_product_names(driver: Any) -> list[str]:
    recs = await _records(driver, PRODUCT_NAMES_QUERY, {})
    return sorted(recs[0]["names"]) if recs else []


async def read_affected_ids(driver: Any, product_names: list[str]) -> set[str]:
    """The whole 2026-10-17 affected set: both halves of extraction_proof's query."""
    out: set[str] = set()
    for example in (True, False):
        recs = await _records(driver, AFFECTED_ARTICLES_QUERY, {
            "product_names": list(product_names),
            "example_subjects": [s.lower() for s in EXAMPLE_SUBJECTS],
            "example": example, "exclude": [], "limit": AFFECTED_LIMIT})
        out.update(rec["id"] for rec in recs)
    return out


def _hard_cases(rec: Any, affected: set[str]) -> tuple[str, ...]:
    title = (rec["title"] or "").lower()
    flags = []
    if rec["id"] in affected:
        flags.append("affected")
    if rec["unnamed_chunk"]:
        flags.append("unnamed_chunk")
    if any(w in title for w in RESTORE_WORDS):
        flags.append("restore_title")
    if rec["has_table"]:
        flags.append("table")
    return tuple(flags)


async def read_candidates(driver: Any, vendors: Sequence[str]) -> list[_Candidate]:
    product_names = await read_product_names(driver)
    affected = await read_affected_ids(driver, product_names)
    recs = await _records(driver, ARTICLES_QUERY, {
        "vendors": [v.lower() for v in vendors], "product_names": product_names,
        "table_rule": TABLE_RULE})
    out = [_Candidate(article_id=rec["id"], title=rec["title"] or "",
                      vendor=rec["vendor"] or "", product=rec["product"] or "",
                      source=rec["source"] or "", hard_cases=_hard_cases(rec, affected))
           for rec in recs]
    out.sort(key=lambda c: c.article_id)       # ORDER BY a.id, whatever the driver did
    return out


def _round_robin(groups: Iterable[list[SampleArticle]]) -> list[SampleArticle]:
    queues = [list(g) for g in groups if g]
    out: list[SampleArticle] = []
    while queues:
        for q in queues:
            out.append(q.pop(0))
        queues = [q for q in queues if q]
    return out


def _ordered(articles: list[SampleArticle]) -> list[SampleArticle]:
    """Round-robin over vendors, and over each vendor's strata within it."""
    by_vendor: dict[str, dict[str, list[SampleArticle]]] = defaultdict(lambda: defaultdict(list))
    for a in articles:
        by_vendor[a.vendor][a.stratum].append(a)
    per_vendor = [_round_robin(by_vendor[v][s] for s in sorted(by_vendor[v]))
                  for v in sorted(by_vendor)]
    return _round_robin(per_vendor)


def choose(candidates: list[_Candidate], *, n_total: int, n_heldout: int, seed: int,
           vendor_floor: int = VENDOR_FLOOR) -> tuple[list[SampleArticle], dict[str, dict]]:
    """The pure selection over an already-read candidate table (see module doc).
    Returns the ordered sample and per-stratum counts."""
    if n_heldout > n_total:
        raise ValueError(f"n_heldout {n_heldout} exceeds n_total {n_total}")
    by_stratum: dict[str, list[_Candidate]] = defaultdict(list)
    for c in sorted(candidates, key=lambda c: c.article_id):
        by_stratum[c.stratum].append(c)
    vendor_of = {s: cs[0].vendor for s, cs in by_stratum.items()}
    vendor_sizes: dict[str, int] = defaultdict(int)
    for s, cs in by_stratum.items():
        vendor_sizes[vendor_of[s]] += len(cs)
    vendor_alloc = apportion(n_total, dict(vendor_sizes),
                             {v: vendor_floor for v in vendor_sizes})

    selected: dict[str, list[_Candidate]] = {}
    for vendor, share in sorted(vendor_alloc.items()):
        strata = {s: len(cs) for s, cs in by_stratum.items() if vendor_of[s] == vendor}
        for s, k in sorted(apportion(share, strata).items()):
            rng = random.Random(f"{seed}|select|{s}")
            selected[s] = _weighted_shuffle(by_stratum[s], rng)[:k]

    sizes = {s: len(cs) for s, cs in selected.items() if cs}
    held_alloc = apportion(min(n_heldout, sum(sizes.values())), sizes)
    sample: list[SampleArticle] = []
    counts: dict[str, dict] = {}
    for s in sorted(by_stratum):
        picked = selected.get(s, [])
        perm = list(picked)
        random.Random(f"{seed}|heldout|{s}").shuffle(perm)
        h = held_alloc.get(s, 0)
        held = {c.article_id for c in perm[len(perm) - h:]} if h else set()
        for c in picked:
            sample.append(SampleArticle(
                article_id=c.article_id, vendor=c.vendor, product=c.product,
                source=c.source, split="heldout" if c.article_id in held else "train",
                stratum=s, hard_cases=c.hard_cases))
        counts[s] = {
            "population": len(by_stratum[s]),
            "hard_population": sum(1 for c in by_stratum[s] if c.hard_cases),
            "selected": len(picked),
            "hard_selected": sum(1 for c in picked if c.hard_cases),
            "train": len(picked) - len(held),
            "heldout": len(held),
        }
    train = _ordered([a for a in sample if a.split == "train"])
    heldout = _ordered([a for a in sample if a.split == "heldout"])
    return train + heldout, counts


async def select_sample(driver: Any, *, n_total: int = 100, n_heldout: int = 20,
                        seed: int = 20261017,
                        vendors: Sequence[str] = DEFAULT_VENDORS) -> list[SampleArticle]:
    """Read-only selection over `driver` (anything with neo4j's async
    `execute_query(query, parameters_=, routing_=)`). Raises when a requested
    vendor has no article with live episodes: a silent zero would quietly drop a
    stratum the spec requires."""
    sample, _ = await select_sample_with_counts(driver, n_total=n_total,
                                                n_heldout=n_heldout, seed=seed,
                                                vendors=vendors)
    return sample


async def select_sample_with_counts(
        driver: Any, *, n_total: int = 100, n_heldout: int = 20, seed: int = 20261017,
        vendors: Sequence[str] = DEFAULT_VENDORS,
) -> tuple[list[SampleArticle], dict[str, dict]]:
    candidates = await read_candidates(driver, vendors)
    found = {c.vendor.lower() for c in candidates}
    missing = [v for v in vendors if v.lower() not in found]
    if missing:
        raise ValueError(
            f"no article with live episodes for vendor(s) {missing}; vendors found: "
            f"{sorted({c.vendor for c in candidates})} (pass the structural Vendor names)")
    return choose(candidates, n_total=n_total, n_heldout=n_heldout, seed=seed)


def write_sample(path: Path, sample: list[SampleArticle], counts: dict[str, dict], *,
                 seed: int, n_total: int, n_heldout: int, vendors: Sequence[str]) -> None:
    payload = {
        "seed": seed, "n_total": n_total, "n_heldout": n_heldout, "vendors": list(vendors),
        "selected": len(sample),
        "train": sum(1 for a in sample if a.split == "train"),
        "heldout": sum(1 for a in sample if a.split == "heldout"),
        "strata": counts,
        "articles": [asdict(a) for a in sample],
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")


def load_sample(path: Path) -> list[SampleArticle]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SampleArticle(**{**a, "hard_cases": tuple(a.get("hard_cases") or ())})
            for a in payload["articles"]]
