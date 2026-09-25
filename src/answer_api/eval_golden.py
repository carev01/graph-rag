"""Run the golden question set through search_local and report citation precision@k.

Live: needs the semantic graph + the embedder. Run as:
    uv run --extra dev python -m answer_api.eval_golden [--k 10]
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from neo4j import AsyncGraphDatabase

from answer_api.golden import first_hit_rank, precision_at_k
from answer_api.scope import Scope
from answer_api.search import search_local
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_graphiti

_QUESTIONS = Path(__file__).with_name("golden_questions.json")


async def main(k: int) -> None:
    settings = get_extract_settings()
    questions = json.loads(_QUESTIONS.read_text())
    graphiti = build_graphiti(settings)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    hits = 0
    ranks: list[int] = []
    try:
        print(f"# Golden retrieval eval (k={k}, {len(questions)} questions)\n")
        for q in questions:
            v = q.get("vendor")
            scope = Scope((v,), (), "explicit") if v else None
            out = await search_local(
                graphiti, driver, q=q["question"], k=k,
                scope=scope, group_id=settings.group_id)
            hit = precision_at_k(out["results"], q["expected_article_ids"])
            rank = first_hit_rank(out["results"], q["expected_article_ids"])
            hits += 1 if hit else 0
            if rank is not None:
                ranks.append(rank)
            mark = "HIT " if hit else "MISS"
            print(f"[{mark} rank={rank}] {q['question']}  "
                  f"(results={out['count']})")
    finally:
        await graphiti.close()
        await driver.close()
    n = len(questions)
    mrr = sum(1.0 / r for r in ranks) / n if n else 0.0
    print(f"\ncitation_precision@{k} = {hits}/{n} = {hits / n:.3f}")
    print(f"mean_reciprocal_rank    = {mrr:.3f}  (first-hit ranks: {sorted(ranks)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    asyncio.run(main(ap.parse_args().k))
