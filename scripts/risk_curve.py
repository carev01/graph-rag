"""Recompute the duplicate-risk curve (spec §1.1) from the graph, read-only.

Decides `ingest_warmup_articles` (`W`, spec §3.3) from evidence instead of a
second paid A/B: run it against a group after an ingest and read off the `c` at
which the cumulative risk share stops climbing steeply.

The model (`2026-09-14-concurrency-duplicate-mitigation-design.md` §1):

- An entity's duplicate risk under concurrent ingest scales with `k - 1`, where
  `k` is the number of distinct articles that mention it (an entity seen in one
  article can never collide with itself). Total risk weight is `Σ(k-1)`.
- Spans come from `(:Article)-[:HAS_EPISODE]->(:Episodic)-[:MENTIONS]->(:Entity)`,
  episodes and entities scoped to one `group_id`.
- An article's position is its 0-based rank within its OWN source by
  `Article.sort_order` -- the unit `ingest_source` processes, in the order it
  processes it. An entity "first appears at c" when the earliest per-source
  position among its articles is `c - 1`.
- The curve is the cumulative share of `Σ(k-1)` carried by entities that first
  appear within the first `c` articles of their source, for `c` in 1..--max-c.

Against the 999-entity baseline (`backup-docs`) it must print the spec's row:
`c=4 -> 71.3%`, `c=8 -> 79.5%`, `c=12 -> 82.5%` over `Σ(k-1) = 1538`. If it
does not, the script is wrong; the spec's numbers were computed from that graph.

Writes nothing: the session is opened READ-only, so the server itself refuses
a write.

    uv run --extra dev python scripts/risk_curve.py [--group G] [--max-c 30]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from typing import Any

from neo4j import AsyncGraphDatabase

from graph_extract.config import get_extract_settings

_ARTICLES = """
MATCH (a:Article)-[:HAS_EPISODE]->(:Episodic {group_id: $group_id})
RETURN DISTINCT a.id AS id, a.source_id AS source_id, a.sort_order AS sort_order
"""

_SPANS = """
MATCH (a:Article)-[:HAS_EPISODE]->(:Episodic {group_id: $group_id})
      -[:MENTIONS]->(e:Entity {group_id: $group_id})
RETURN e.uuid AS uuid, e.name AS name, collect(DISTINCT a.id) AS articles
"""


def per_source_positions(articles: list[dict[str, Any]]) -> dict[str, int]:
    """0-based rank of each article within its source, by `sort_order`
    (ties, and a missing `sort_order`, broken by id so the rank is stable)."""
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for article in articles:
        by_source[article["source_id"]].append(article)
    positions: dict[str, int] = {}
    for members in by_source.values():
        members.sort(key=lambda a: (a["sort_order"] is None, a["sort_order"] or 0, a["id"]))
        for rank, article in enumerate(members):
            positions[article["id"]] = rank
    return positions


def risk_curve(spans: list[dict[str, Any]], positions: dict[str, int],
               max_c: int) -> tuple[int, list[float]]:
    """Return `(total, shares)`: `Σ(k-1)` and, for c in 1..max_c, the
    percentage of it carried by entities first seen within the first `c`
    articles of their source."""
    weight_at_first: dict[int, int] = defaultdict(int)
    total = 0
    for span in spans:
        weight = len(span["articles"]) - 1
        total += weight
        weight_at_first[min(positions[a] for a in span["articles"])] += weight
    shares: list[float] = []
    cumulative = 0
    for c in range(1, max_c + 1):
        cumulative += weight_at_first.get(c - 1, 0)
        shares.append(100.0 * cumulative / total if total else 0.0)
    return total, shares


async def _load(group_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = get_extract_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    try:
        async with driver.session(default_access_mode="READ") as session:
            articles = [dict(r) async for r in await session.run(_ARTICLES, group_id=group_id)]
            spans = [dict(r) async for r in await session.run(_SPANS, group_id=group_id)]
    finally:
        await driver.close()
    return articles, spans


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--group", default=None,
                        help="group_id to measure (default: ExtractSettings.group_id)")
    parser.add_argument("--max-c", type=int, default=30,
                        help="largest warm-up size to tabulate (default 30)")
    args = parser.parse_args()
    group_id = args.group or get_extract_settings().group_id
    if args.max_c < 1:
        parser.error("--max-c must be >= 1")

    articles, spans = asyncio.run(_load(group_id))
    if not spans:
        print(f"no article->episode->entity spans in group {group_id!r}; nothing to measure",
              file=sys.stderr)
        return 1
    positions = per_source_positions(articles)
    total, shares = risk_curve(spans, positions, args.max_c)

    per_source: dict[str, int] = defaultdict(int)
    for article in articles:
        per_source[article["source_id"]] += 1
    print(f"group {group_id!r}: {len(articles)} articles over {len(per_source)} source(s) "
          f"({', '.join(str(n) for n in sorted(per_source.values(), reverse=True))}), "
          f"{len(spans)} entities, risk weight sum(k-1) = {total}")
    print()
    print("cumulative share of risk weight whose entity first appears within the")
    print("first c articles of its source (positions by Article.sort_order):")
    print()
    print("   c   share")
    for c, share in enumerate(shares, start=1):
        print(f"  {c:>2}   {share:5.1f}%")

    hubs = sorted(spans, key=lambda s: -len(s["articles"]))[:10]
    print()
    print("top entities by span (k articles; first per-source position, 0-based):")
    for span in hubs:
        first = min(positions[a] for a in span["articles"])
        print(f"  k={len(span['articles']):>3}  first={first:>3}  {span['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
