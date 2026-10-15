"""Collapse near-duplicate facts (spec 2026-10-15 §4.3): the same statement extracted
from several articles in different wordings filled answer slots (four variants of
"Instant Recovery ... as a Hyper-V VM"). The first occurrence is kept; every source
of the merged facts is kept on it, so no citation is lost."""
from __future__ import annotations

import logging
import math
import re

logger = logging.getLogger(__name__)

EMBEDDINGS_QUERY = (
    "MATCH ()-[f:RELATES_TO]->() WHERE f.uuid IN $uuids "
    "RETURN f.uuid AS uuid, f.fact_embedding AS embedding")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize_fact(text: str) -> str:
    return _NON_WORD.sub(" ", (text or "").lower()).strip()


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _merge_sources(into: dict, other: dict) -> None:
    seen = {s.get("url") or s.get("article_id") for s in into["sources"]}
    for s in other.get("sources") or []:
        key = s.get("url") or s.get("article_id")
        if key not in seen:
            seen.add(key)
            into["sources"].append(s)


async def collapse_near_duplicates(driver, results: list[dict], *, cosine: float) -> list[dict]:
    if not results:
        return []
    vectors: dict[str, list[float]] = {}
    try:
        records, _, _ = await driver.execute_query(
            EMBEDDINGS_QUERY, uuids=[r["fact_uuid"] for r in results])
        vectors = {r["uuid"]: list(r["embedding"]) for r in records if r["embedding"]}
    except Exception as exc:
        logger.warning("fact embedding fetch failed (%s); collapsing identical text only", exc)
    kept: list[dict] = []
    for r in results:
        norm = normalize_fact(r["fact"])
        vec = vectors.get(r["fact_uuid"])
        twin = None
        for k in kept:
            if normalize_fact(k["fact"]) == norm:
                twin = k
                break
            kv = vectors.get(k["fact_uuid"])
            if vec is not None and kv is not None and _cos(vec, kv) >= cosine:
                twin = k
                break
        if twin is None:
            kept.append({**r, "sources": list(r.get("sources") or [])})
        else:
            _merge_sources(twin, r)
    return kept
