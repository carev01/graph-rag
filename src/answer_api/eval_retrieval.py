"""Free retrieval gate (spec 2026-10-12 §6): per golden question, does per-facet
retrieval find evidence for every facet, an expected article, and every expected
vendor? No LLM calls -- the local embedder, Neo4j and the reranker only. Run before
any paid eval:

    uv run python -m answer_api.eval_retrieval --questions src/answer_api/router_golden_tier1.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

from neo4j import AsyncGraphDatabase

from answer_api.global_search import shortlist_communities
from answer_api.scope import ScopeResolver
from answer_api.search import search_local
from answer_api.shape import classify_shape
from graph_extract.config import get_extract_settings
from graph_extract.graphiti_client import build_embedder, build_graphiti


def check_question(q: dict, facet_evidence: list[dict]) -> dict:
    expected = set(q.get("expected_article_ids") or [])
    articles = set().union(*(e["articles"] for e in facet_evidence)) if facet_evidence else set()
    vendors = set().union(*(e["vendors"] for e in facet_evidence)) if facet_evidence else set()
    expected_hit = bool(expected & articles) if expected else None
    with_ev = sum(1 for e in facet_evidence if e["communities"] + e["facts"] > 0)
    want = q.get("expected_vendors")
    vendors_ok = want is None or all(
        any(e["label"] == v and e["communities"] + e["facts"] > 0 for e in facet_evidence)
        for v in want)
    return {"question": q["question"], "shape": q.get("_shape"),
            "expected_hit": expected_hit, "facets_with_evidence": with_ev,
            "facets": len(facet_evidence), "vendors_found": sorted(vendors),
            "vendors_expected": want,
            "pass": with_ev == len(facet_evidence) and expected_hit is not False
            and vendors_ok}


async def _evidence(graphiti, driver, embedder, settings, q: str, shape) -> list[dict]:
    n = len(shape.facets)
    if shape.kind == "enumeration":
        k, top_n = settings.enum_facts_per_vendor, settings.enum_communities_per_vendor
    elif shape.kind == "comparison":
        k = max(settings.local_min_per_facet, math.ceil(15 / n))
        top_n = settings.global_facet_top_n
    else:
        k, top_n = 15, None
    out = []
    for f in shape.facets:
        scope = f.scope if not f.scope.is_empty() else None
        loc = await search_local(graphiti, driver, q=q, k=k, scope=scope,
                                 group_id=settings.group_id,
                                 fetch_cap=settings.local_scoped_fetch_cap)
        hits = await shortlist_communities(
            driver, embedder, q, level=settings.global_default_level,
            k=top_n or settings.global_shortlist_k, group_id=settings.group_id,
            settings=settings, scope=scope, top_n=top_n)
        srcs = [s for r in loc["results"] for s in r["sources"]]
        out.append({"label": f.label,
                    "articles": {s.get("article_id") for s in srcs},
                    "vendors": {s.get("vendor") for s in srcs if s.get("vendor")},
                    "communities": len(hits), "facts": loc["count"]})
    return out


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True)
    args = ap.parse_args()
    settings = get_extract_settings()
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    graphiti = build_graphiti(settings)
    embedder = build_embedder(settings)
    try:
        resolver = await ScopeResolver.load(driver)
        results = []
        for q in json.loads(args.questions.read_text()):
            shape = classify_shape(q["question"], resolver, resolver.resolve(q["question"]))
            ev = await _evidence(graphiti, driver, embedder, settings, q["question"], shape)
            r = check_question({**q, "_shape": shape.kind}, ev)
            results.append(r)
            print(f"{'PASS' if r['pass'] else 'FAIL'} {shape.kind:11} "
                  f"facets {r['facets_with_evidence']}/{r['facets']} "
                  f"expected={r['expected_hit']} vendors={r['vendors_found']} "
                  f"| {q['question']}", flush=True)
        passed = sum(r["pass"] for r in results)
        print(f"PASS {passed}/{len(results)}")
        return 0 if passed == len(results) else 1
    finally:
        await graphiti.close()
        await driver.close()
        await embedder.client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
