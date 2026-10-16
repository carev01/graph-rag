"""Collapse near-duplicate facts (spec 2026-10-15 §4.3, amended §10): the same statement
extracted from several articles in different wordings filled answer slots (four variants
of "Instant Recovery ... as a Hyper-V VM"). The first occurrence is kept; every source
of the merged facts is kept on it, so no citation is lost. Facts merge only within a
vendor; a fact with no vendor merges only with another vendor-less fact (no vendorless
bridge between two named vendors)."""
from __future__ import annotations

import asyncio
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDINGS_QUERY = (
    "MATCH ()-[f:RELATES_TO]->() WHERE f.uuid IN $uuids "
    "RETURN f.uuid AS uuid, f.fact_embedding AS embedding")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_fact(text: str) -> str:
    return _NON_WORD.sub(" ", (text or "").lower()).strip()


def _merge_sources(into: dict, other: dict) -> None:
    seen = {s.get("url") or s.get("article_id") for s in into["sources"]}
    for s in other.get("sources") or []:
        key = s.get("url") or s.get("article_id")
        if key not in seen:
            seen.add(key)
            into["sources"].append(s)


def _vendors(r: dict) -> frozenset[str]:
    return frozenset(s["vendor"] for s in r.get("sources") or [] if s.get("vendor"))


def _same_vendor(a: frozenset[str], b: frozenset[str]) -> bool:
    """Facts merge only within a vendor: an unscoped question can retrieve the same
    statement for two vendors, and those are two claims. A fact with no vendor
    merges only with another fact that also has no vendor -- a vendorless fact must
    never bridge two named vendors into one merged entry."""
    return (not a and not b) or bool(a & b)


def _unit_rows(results: list[dict], vectors: dict[str, list[float]]
               ) -> tuple[np.ndarray, list[bool]]:
    """One unit-length row per result (zeros where it has no usable vector), so a
    row dot product is the raw cosine. Vectors of a different length than the
    first one are treated as missing."""
    dim = next((len(v) for v in vectors.values()), 0)
    has = [len(vectors.get(r["fact_uuid"]) or ()) == dim > 0 for r in results]
    mat = np.zeros((len(results), dim), dtype=np.float64)
    for i, r in enumerate(results):
        if has[i]:
            mat[i] = vectors[r["fact_uuid"]]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.divide(mat, norms, out=mat, where=norms > 0)
    return mat, has


def _collapse(results: list[dict], vectors: dict[str, list[float]],
              cosine: float) -> list[dict]:
    """CPU part (run off the event loop): the first occurrence wins; a later fact
    merges into the first kept one with the same normalised text or a raw cosine
    >= `cosine`, within a vendor."""
    mat, has = _unit_rows(results, vectors)
    sims = mat @ mat.T if any(has) else None
    texts = [normalize_fact(r["fact"]) for r in results]
    vendors = [_vendors(r) for r in results]
    kept: list[int] = []
    out: list[dict] = []
    for i, r in enumerate(results):
        twin = None
        for slot, j in enumerate(kept):
            if not _same_vendor(vendors[i], vendors[j]):
                continue
            if texts[i] == texts[j] or (
                    sims is not None and has[i] and has[j] and sims[i, j] >= cosine):
                twin = slot
                break
        if twin is None:
            kept.append(i)
            out.append({**r, "sources": list(r.get("sources") or [])})
        else:
            _merge_sources(out[twin], r)
    return out


async def fetch_vectors(driver, uuids: list[str], cache: dict[str, list[float]]
                        ) -> dict[str, list[float]]:
    """fact_embedding of each of `uuids`, fetched at most once per uuid: only the
    uuids missing from `cache` are queried (one EMBEDDINGS_QUERY), and the cache is
    updated -- a fact with no embedding is remembered as an empty list, so it is
    never asked for again. Returns the usable vectors of `uuids` only (empty ones
    left out). A failed query raises and leaves the cache unchanged."""
    wanted = list(dict.fromkeys(uuids))
    missing = [u for u in wanted if u not in cache]
    if missing:
        records, _, _ = await driver.execute_query(EMBEDDINGS_QUERY, uuids=missing)
        got = {r["uuid"]: list(r["embedding"]) for r in records if r["embedding"]}
        cache.update({u: got.get(u, []) for u in missing})
    return {u: cache[u] for u in wanted if cache[u]}


async def collapse_near_duplicates(driver, results: list[dict], *, cosine: float,
                                   vectors: dict[str, list[float]] | None = None
                                   ) -> list[dict]:
    """`vectors` is an optional per-question embedding cache (list mode collapses
    twice and selects by MMR over the same facts): only uuids it lacks are fetched,
    into it. Without one, every vector is fetched (the single-question path). A
    failed fetch collapses identical text only (plus any vectors already cached)."""
    if not results:
        return []
    cache = vectors if vectors is not None else {}
    uuids = [r["fact_uuid"] for r in results]
    try:
        usable = await fetch_vectors(driver, uuids, cache)
    except Exception as exc:
        logger.warning("fact embedding fetch failed (%s); collapsing identical text only", exc)
        usable = {u: cache[u] for u in dict.fromkeys(uuids) if cache.get(u)}
    # O(n^2) over ~200+ facts of 768-1024 dims: off the event loop.
    return await asyncio.to_thread(_collapse, results, usable, cosine)
