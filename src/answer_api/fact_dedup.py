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
from neo4j.exceptions import ClientError, CypherSyntaxError, CypherTypeError
from neo4j.vector import Vector

logger = logging.getLogger(__name__)

# Task 13 (2026-10-16): sending 1000 fact embeddings as float lists took ~20 s from
# the answer-api pod (the cost scales with bytes); quantised in Cypher to a
# per-vector max-abs-scaled INTEGER8 vector, the same 1000 take ~3 s. Cosine is
# scale-invariant, so the per-vector scale need not be sent back. Measured on 44,850
# real pairs: mean |dcos| 0.0008, at most 0.0018 for pairs above 0.9.
EMBEDDINGS_QUERY = (
    "MATCH ()-[f:RELATES_TO]->() WHERE f.uuid IN $uuids "
    "WITH f, f.fact_embedding AS e WHERE e IS NOT NULL "
    "WITH f, e, reduce(m = 0.0, x IN e | CASE WHEN abs(x) > m THEN abs(x) ELSE m END) AS mx "
    "RETURN f.uuid AS uuid, CASE WHEN mx = 0 THEN null "
    "ELSE vector([x IN e | toInteger(round(x * 127.0 / mx))], size(e), INTEGER8) END "
    "AS embedding")
# Servers without vector()/INTEGER8 (Neo4j 5.x, or a database not on Cypher 25).
EMBEDDINGS_QUERY_FLOAT = (
    "MATCH ()-[f:RELATES_TO]->() WHERE f.uuid IN $uuids "
    "RETURN f.uuid AS uuid, f.fact_embedding AS embedding")
# Set once the compact query has been rejected by the server: every later fetch in
# this process goes straight to EMBEDDINGS_QUERY_FLOAT.
_compact_unavailable = False
_EMPTY = np.empty(0, dtype=np.float32)       # "no embedding", shared by every cache
_EMPTY.setflags(write=False)
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


def _usable(v) -> bool:
    """A cached vector with values (an array or, from older callers, a list)."""
    return v is not None and len(v) > 0


def _unit_rows(results: list[dict], vectors: dict[str, np.ndarray]
               ) -> tuple[np.ndarray, list[bool]]:
    """One unit-length row per result (zeros where it has no usable vector), so a
    row dot product is the raw cosine. Vectors of a different length than the
    first one are treated as missing."""
    dim = next((len(v) for v in vectors.values()), 0)
    rows = [vectors.get(r["fact_uuid"]) for r in results]
    has = [dim > 0 and v is not None and len(v) == dim for v in rows]
    mat = np.zeros((len(results), dim), dtype=np.float64)
    picked = [v for v in rows if v is not None and len(v) == dim]
    if dim > 0 and picked:
        mat[np.flatnonzero(has)] = np.stack(picked)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.divide(mat, norms, out=mat, where=norms > 0)
    return mat, has


def _collapse(results: list[dict], vectors: dict[str, np.ndarray],
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


def _as_array(value) -> np.ndarray:
    """A returned embedding as a float32 array: an INTEGER8 `Vector` (the compact
    query; its values are the scaled integers, which cosine does not mind) or a plain
    float list (the fallback query). Empty when the fact has none, or an all-zero one
    (the compact query returns null for those, so both queries agree)."""
    if value is None:
        return _EMPTY
    raw = value.to_numpy() if isinstance(value, Vector) else value
    arr = np.asarray(raw, dtype=np.float32)
    return arr if arr.any() else _EMPTY


def _vector_type_unsupported(exc: ClientError) -> bool:
    """The server rejected the compact query itself (no vector()/INTEGER8): a Cypher
    syntax/type error or another statement-level client error. Auth, transaction and
    transport errors are not -- they propagate to the caller's degraded handling."""
    return (isinstance(exc, (CypherSyntaxError, CypherTypeError))
            or (exc.code or "").startswith("Neo.ClientError.Statement."))


async def _query_embeddings(driver, uuids: list[str]) -> list:
    global _compact_unavailable
    if not _compact_unavailable:
        try:
            records, _, _ = await driver.execute_query(EMBEDDINGS_QUERY, uuids=uuids)
            return records
        except ClientError as exc:
            if not _vector_type_unsupported(exc):
                raise
            _compact_unavailable = True
            logger.warning("compact embedding transfer unavailable (%s); "
                           "fetching float vectors", exc)
    records, _, _ = await driver.execute_query(EMBEDDINGS_QUERY_FLOAT, uuids=uuids)
    return records


async def fetch_vectors(driver, uuids: list[str], cache: dict[str, np.ndarray]
                        ) -> dict[str, np.ndarray]:
    """fact_embedding of each of `uuids` as a float32 array, fetched at most once per
    uuid: only the uuids missing from `cache` are queried (one EMBEDDINGS_QUERY, or
    EMBEDDINGS_QUERY_FLOAT on a server without INTEGER8 vectors), and the cache is
    updated -- a fact with no embedding is remembered as an empty array, so it is
    never asked for again. Returns the usable vectors of `uuids` only (empty ones
    left out). A failed query raises and leaves the cache unchanged."""
    wanted = list(dict.fromkeys(uuids))
    missing = [u for u in wanted if u not in cache]
    if missing:
        records = await _query_embeddings(driver, missing)
        got = {r["uuid"]: _as_array(r["embedding"]) for r in records}
        cache.update({u: got.get(u, _EMPTY) for u in missing})
    return {u: cache[u] for u in wanted if _usable(cache[u])}


async def collapse_near_duplicates(driver, results: list[dict], *, cosine: float,
                                   vectors: dict[str, np.ndarray] | None = None
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
        usable = {u: cache[u] for u in dict.fromkeys(uuids) if _usable(cache.get(u))}
    # O(n^2) over ~200+ facts of 768-1024 dims: off the event loop.
    return await asyncio.to_thread(_collapse, results, usable, cosine)
