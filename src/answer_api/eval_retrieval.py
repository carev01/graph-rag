"""Free retrieval gate (spec 2026-10-12 §6, list check added 2026-10-15 §7): per
golden question, does per-facet retrieval find evidence for every facet, an
expected article, and every expected vendor? No generative calls except the
per-facet question rewrite (one cheap map-tier call per comparison/enumeration
question, exactly as /answer makes it) and, for list questions,
`list_mode.retrieve_list`'s cheap-tier item-kind expansion -- otherwise the
local embedder, Neo4j and the reranker only. Run before any paid eval.

Single and comparison questions pass when every facet has evidence and (if the
golden entry lists expected articles) one of them is retrieved. Enumerations pass
when every `expected_vendors` vendor has evidence (plus the same expected-article
check); the other content vendors' evidence counts are reported as information
only, because a vendor that does not document the feature legitimately has none.
This gate therefore cannot tell whether an enumeration answer is COMPLETE (no
vendor missing, none extra): only the paid eval's enumeration_completeness metric
measures that. A question carrying `expected_items` (a golden list question) must
ALSO have `list_mode.retrieve_list` find every expected item among its results'
and limitations' fact text -- a missing item fails the question even if its
facets otherwise have evidence.

    uv run python -m answer_api.eval_retrieval --questions src/answer_api/router_golden_tier1.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

from neo4j import AsyncGraphDatabase

from answer_api import list_mode
from answer_api.fact_dedup import collapse_near_duplicates
from answer_api.global_search import _map_client_and_model, shortlist_communities
from answer_api.item_coverage import match_items
from answer_api.scope import ScopeResolver
from answer_api.search import search_local
from answer_api.shape import QuestionShape, classify_shape, facet_queries
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
    out = {"question": q["question"], "shape": q.get("_shape"),
           "expected_hit": expected_hit, "facets_with_evidence": with_ev,
           "facets": len(facet_evidence), "vendors_found": sorted(vendors),
           "vendors_expected": want}
    if q.get("_shape") == "enumeration":
        # Only the expected vendors must have evidence; the rest is information.
        out["other_vendors"] = {e["label"]: e["communities"] + e["facts"]
                                for e in facet_evidence if e["label"] not in (want or [])}
        out["pass"] = vendors_ok and expected_hit is not False
    else:
        out["pass"] = (with_ev == len(facet_evidence) and expected_hit is not False
                       and vendors_ok)
    return out


async def _with_facet_queries(client, model: str, q: str,
                              shape: QuestionShape) -> QuestionShape:
    """The router's rewrite, reproduced: comparison/enumeration facets get their
    own retrieval question; a single question is returned unchanged (no call)."""
    if shape.kind not in ("comparison", "enumeration"):
        return shape
    return replace(shape, facets=await facet_queries(client, model, q, shape.facets))


async def _evidence(graphiti, driver, embedder, settings, q: str, shape) -> list[dict]:
    n = len(shape.facets)
    if shape.kind == "enumeration":
        k, top_n = settings.enum_facts_per_vendor, settings.enum_communities_per_vendor
    elif shape.kind == "comparison":
        k = max(settings.local_min_per_facet, math.ceil(15 / n))
        top_n = settings.global_facet_top_n
    else:
        k, top_n = 30, None
    # Facet shapes rerank a wider pool, as the router's facet callers do; a single
    # question mirrors /answer's single path: rerank a pool of
    # local_single_rerank_pool candidates, collapse near-duplicates, and keep 15.
    if shape.kind in ("comparison", "enumeration"):
        pool: dict = {"rerank_pool": settings.local_facet_rerank_pool}
    elif shape.kind == "single":
        pool = {"rerank_pool": settings.local_single_rerank_pool}
    else:
        pool = {}
    out = []
    for f in shape.facets:
        scope = f.scope if not f.scope.is_empty() else None
        fq = f.query or q
        loc = await search_local(graphiti, driver, q=fq, k=k, scope=scope,
                                 group_id=settings.group_id,
                                 fetch_cap=settings.local_scoped_fetch_cap,
                                 scan_max_episodes=settings.local_scoped_scan_max_episodes,
                                 settings=settings, **pool)
        results = loc["results"]
        if shape.kind == "single":
            results = (await collapse_near_duplicates(
                driver, results, cosine=settings.list_dedup_cosine))[:15]
        hits = await shortlist_communities(
            driver, embedder, fq, level=settings.global_default_level,
            k=top_n or settings.global_shortlist_k, group_id=settings.group_id,
            settings=settings, scope=scope, top_n=top_n)
        srcs = [s for r in results for s in r["sources"]]
        out.append({"label": f.label,
                    "articles": {s.get("article_id") for s in srcs},
                    "vendors": {s.get("vendor") for s in srcs if s.get("vendor")},
                    "communities": len(hits), "facts": len(results)})
    return out


async def check_list_items(graphiti, driver, map_client, map_model, settings,
                           q: dict, shape: QuestionShape) -> dict:
    """List-retrieval check (spec 2026-10-15 §7): does `list_mode.retrieve_list`
    find evidence for every `expected_items` entry? A single-shaped question's
    one facet already carries the scope `retrieval_scope` would resolve to
    (classify_shape fills it from the detected or named-subject scope), so it is
    used directly rather than recomputed. `retrieve_list` raising (every
    expansion/retrieval attempt failed) must read as FAIL with the error, never
    as a pass with zero items found."""
    items = q["expected_items"]
    scope = None
    if shape.facets and not shape.facets[0].scope.is_empty():
        scope = shape.facets[0].scope
    try:
        got = await list_mode.retrieve_list(
            graphiti, driver, map_client, map_model, q=q["question"], scope=scope,
            group_id=settings.group_id, settings=settings)
    except Exception as exc:
        return {"pass": False, "found": 0, "total": len(items),
                "missing": [i["item"] for i in items], "error": str(exc)}
    text = " || ".join(r["fact"] for r in got.results + got.limitations)
    found, missing = match_items(text, items)
    return {"pass": not missing, "found": len(found), "total": len(items),
            "missing": missing}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True)
    args = ap.parse_args()
    settings = get_extract_settings()
    # Built first: an unconfigured map tier fails before anything else is opened.
    map_client, map_model = _map_client_and_model(settings)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    graphiti = build_graphiti(settings)
    embedder = build_embedder(settings)
    try:
        resolver = await ScopeResolver.load(driver)
        results = []
        for q in json.loads(args.questions.read_text()):
            shape = classify_shape(q["question"], resolver, resolver.resolve(q["question"]))
            shape = await _with_facet_queries(map_client, map_model, q["question"], shape)
            ev = await _evidence(graphiti, driver, embedder, settings, q["question"], shape)
            r = check_question({**q, "_shape": shape.kind}, ev)
            extra = ""
            if q.get("expected_items"):
                items = await check_list_items(graphiti, driver, map_client, map_model,
                                               settings, q, shape)
                r["items"] = items
                r["pass"] = r["pass"] and items["pass"]
                extra = f"items {items['found']}/{items['total']} missing={items['missing']} "
                if items.get("error"):
                    extra += f"error={items['error']!r} "
            results.append(r)
            print(f"{'PASS' if r['pass'] else 'FAIL'} {shape.kind:11} "
                  f"facets {r['facets_with_evidence']}/{r['facets']} "
                  f"expected={r['expected_hit']} vendors={r['vendors_found']} "
                  + (f"other={r['other_vendors']} " if "other_vendors" in r else "")
                  + extra
                  + f"| {q['question']}", flush=True)
        passed = sum(r["pass"] for r in results)
        print(f"PASS {passed}/{len(results)}")
        return 0 if passed == len(results) else 1
    finally:
        await graphiti.close()
        await driver.close()
        await embedder.client.close()
        await map_client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
