"""Live acceptance for index-backed similarity search (spec §7).

Against the .env graph: creates/verifies the two vector indexes (the ONLY write,
schema, approved 2026-09-23), then runs the same searches through graphiti's
exact scan and through the index and reports agreement and latency.
  - retrieval: the 29 router golden questions, edge search, top 10
  - dedup:     up to 200 live entity names, node search, top 15, min 0.6
Free: the embedder is local and no LLM is called.

    uv run python scripts/vector_search_acceptance.py
"""
from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

from graphiti_core.search.search_filters import SearchFilters
from neo4j import RoutingControl

from graph_extract import vector_search as vs
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_embedder, build_graphiti

GOLDEN = Path("src/answer_api/router_golden.json")
EDGE_K, NODE_K, MIN_SCORE, MAX_NAMES = 10, 15, 0.6, 200


def overlap_at(a: list[str], b: list[str], k: int) -> float:
    a, b = a[:k], b[:k]
    if not a and not b:
        return 1.0
    return len(set(a) & set(b)) / max(len(a), len(b))


async def _timed(coro) -> tuple[float, list]:
    t = time.perf_counter()
    out = await coro
    return (time.perf_counter() - t) * 1000.0, out


async def main() -> None:
    s = get_extract_settings()
    graphiti = build_graphiti(s)
    embedder = build_embedder(s)
    d = graphiti.driver
    try:
        await vs.ensure_vector_indexes(d, s.embed_dim)
        vs.reset_stats()
        questions = [q["question"] for q in json.loads(GOLDEN.read_text())]
        qvecs = await embedder.create_batch(questions)
        e_ov, e_ms_x, e_ms_i = [], [], []
        for v in qvecs:
            tx, ex = await _timed(vs._ORIG_EDGE(d, v, None, None, SearchFilters(),
                                                [s.group_id], EDGE_K, MIN_SCORE))
            ti, ei = await _timed(vs.index_edge_similarity_search(
                d, v, None, None, SearchFilters(), [s.group_id], EDGE_K, MIN_SCORE))
            e_ov.append(overlap_at([x.uuid for x in ex], [x.uuid for x in ei], EDGE_K))
            e_ms_x.append(tx)
            e_ms_i.append(ti)
        r = await d.execute_query(
            "MATCH (n:Entity {group_id:$g}) RETURN n.name AS name ORDER BY n.uuid LIMIT $k",
            g=s.group_id, k=MAX_NAMES, routing_=RoutingControl.READ)
        names = [rec["name"].replace("\n", " ") for rec in r.records]
        nvecs = await embedder.create_batch(names)
        n_ov, n_ms_x, n_ms_i = [], [], []
        for v in nvecs:
            tx, nx = await _timed(vs._ORIG_NODE(d, v, SearchFilters(), [s.group_id],
                                                NODE_K, MIN_SCORE))
            ti, ni = await _timed(vs.index_node_similarity_search(
                d, v, SearchFilters(), [s.group_id], NODE_K, MIN_SCORE))
            n_ov.append(overlap_at([x.uuid for x in nx], [x.uuid for x in ni], NODE_K))
            n_ms_x.append(tx)
            n_ms_i.append(ti)
        stats = vs.stats_snapshot()
        print(f"retrieval: {len(e_ov)} questions, top-{EDGE_K} overlap mean "
              f"{statistics.fmean(e_ov):.3f} min {min(e_ov):.3f}; median ms exact "
              f"{statistics.median(e_ms_x):.0f} index {statistics.median(e_ms_i):.0f}")
        print(f"dedup:     {len(n_ov)} names, top-{NODE_K} overlap mean "
              f"{statistics.fmean(n_ov):.3f} min {min(n_ov):.3f}; median ms exact "
              f"{statistics.median(n_ms_x):.0f} index {statistics.median(n_ms_i):.0f}")
        print(f"counters:  {stats}")
        if stats["edge"]["routed"] != len(e_ov) or stats["node"]["routed"] != len(n_ov):
            raise SystemExit("index path did not engage on every call -- see counters")
    finally:
        await embedder.client.close()
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
