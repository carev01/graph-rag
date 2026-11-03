"""Export the head-to-head comparison corpus (graph side) and the graph's frozen state.

The corpus is exactly the `:Article` nodes with a `HAS_EPISODE` edge: what Graphiti has
extracted. The output is a JSON list read by `vendorkb.clients.load_corpus`, so each row
carries exactly the `CorpusEntry` keys. Every query is read-only.

Run: uv run --extra dev python -m scripts.headtohead.export_corpus \
    --out eval/headtohead/corpus.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

from neo4j import AsyncGraphDatabase, RoutingControl

from graph_extract.config import ExtractSettings

KEYS = ("article_id", "vendor", "product", "source", "title", "url", "graph_content_hash")

# Structural chain only (graphiti entities also carry :Vendor/:Product labels, but have no
# `id`). The variable-length part is bounded to 1..2 hops from the source: an unbounded or
# Vendor-rooted [*1..4] path timed out on 2026-11-03.
CORPUS_QUERY = (
    "MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->(s:Source) "
    "WHERE v.id IS NOT NULL AND p.id IS NOT NULL AND s.id IS NOT NULL "
    "MATCH (s)-[*1..2]->(a:Article) "
    "WHERE EXISTS { (a)-[:HAS_EPISODE]->() } "
    "RETURN DISTINCT a.id AS article_id, v.name AS vendor, p.name AS product, "
    "s.name AS source, a.title AS title, a.source_url AS url, "
    "a.content_hash AS graph_content_hash")

STATE_QUERIES = {
    "max_episode_created_at": "MATCH (e:Episodic) RETURN max(e.created_at) AS v",
    "min_community_generated_at": "MATCH (c:Community) RETURN min(c.generated_at) AS v",
    "max_community_generated_at": "MATCH (c:Community) RETURN max(c.generated_at) AS v",
    "community_count": "MATCH (c:Community) RETURN count(c) AS v",
    # theme_builder.writeback sets `stale: true` on a preserved (carried-over) report.
    "stale_community_count": "MATCH (c:Community) WHERE c.stale = true RETURN count(c) AS v",
}


def rows_to_corpus(rows: list[dict]) -> list[dict]:
    """One entry per article id, sorted by id; a duplicate chain keeps the first row after
    sorting by (vendor, product, source). Missing optional fields become None."""
    def chain(r: dict) -> tuple[str, str, str]:
        return (str(r.get("vendor") or ""), str(r.get("product") or ""),
                str(r.get("source") or ""))

    seen: dict[str, dict] = {}
    for r in sorted(rows, key=chain):
        aid = str(r["article_id"])
        if aid not in seen:
            seen[aid] = {k: r.get(k) for k in KEYS} | {"article_id": aid}
    return [seen[k] for k in sorted(seen)]


def _iso(v: Any) -> Any:
    if isinstance(v, (datetime, date)) or hasattr(v, "iso_format"):
        return v.isoformat() if hasattr(v, "isoformat") else v.iso_format()
    return v


async def _fetch(out: Path, state_out: Path) -> Counter:
    s = ExtractSettings()
    driver = AsyncGraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
    try:
        res = await driver.execute_query(CORPUS_QUERY, routing_=RoutingControl.READ)
        corpus = rows_to_corpus([r.data() for r in res.records])
        state: dict[str, Any] = {}
        for name, q in STATE_QUERIES.items():
            r = await driver.execute_query(q, routing_=RoutingControl.READ)
            state[name] = _iso(r.records[0]["v"])
    finally:
        await driver.close()
    for path, payload in ((out, corpus), (state_out, state)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"graph_state: {json.dumps(state)}")
    return Counter(e["vendor"] for e in corpus)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("eval/headtohead/corpus.json"))
    ap.add_argument("--state-out", type=Path, default=None)
    a = ap.parse_args(argv)
    state_out = a.state_out or a.out.with_name("graph_state.json")
    counts = asyncio.run(_fetch(a.out, state_out))
    print(f"total: {sum(counts.values())}")
    for vendor, n in counts.most_common():
        print(f"  {vendor}: {n}")


if __name__ == "__main__":
    main()
