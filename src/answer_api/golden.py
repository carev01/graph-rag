"""Golden-set scoring for local-retrieval citation precision.

Pure functions over `search_local` result dicts — no I/O. A question "hits"
when an expected article-id appears among the returned facts' citations.
"""
from __future__ import annotations


def _cited_article_ids(results: list[dict]) -> set[str]:
    return {s["article_id"] for r in results for s in r.get("sources", [])}


def precision_at_k(results: list[dict], expected_article_ids: list[str]) -> bool:
    """True if any expected article-id is cited anywhere in the results."""
    got = _cited_article_ids(results)
    return any(e in got for e in expected_article_ids)


def first_hit_rank(results: list[dict], expected_article_ids: list[str]) -> int | None:
    """1-indexed rank of the first result citing an expected article, else None."""
    expected = set(expected_article_ids)
    for i, r in enumerate(results, 1):
        if any(s["article_id"] in expected for s in r.get("sources", [])):
            return i
    return None
