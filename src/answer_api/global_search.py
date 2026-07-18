"""Global map-reduce search over the :Community report layer. Shortlist reports
by query-embedding cosine + rating, MAP each to query-relevant key points (with
validated fact IDs), REDUCE into one cited answer. Citations are graph traversal
(design decision #2): the LLM only ever emits fact UUIDs / [N] markers."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from graph_extract.config import ExtractSettings
from graph_extract.provenance import Provenance
from graph_extract.usage import instrument
from answer_api.synthesize import _finalize_answer

logger = logging.getLogger(__name__)

_REFUSAL = "I don't have enough thematic coverage to answer that from the community reports."
_URL_RE = re.compile(r"https?://[^\s\[\]]+", re.IGNORECASE)   # local copy (answer_api-independent)


@dataclass
class CommunityHit:
    community_id: str
    title: str
    summary: str
    level: int
    rating: float
    cited_fact_uuids: list[str]
    full_report: str
    similarity: float


def _cosine(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _rank_hits(query_vec: list[float], rows: list[dict], *, k: int,
               rating_boost: float) -> list[CommunityHit]:
    scored: list[tuple[float, CommunityHit]] = []
    for r in rows:
        if not r.get("cited_fact_uuids"):
            continue                        # nothing citable -> useless for a cited answer
        sim = _cosine(query_vec, r["embedding"])
        score = sim + rating_boost * (r.get("rating") or 0.0) / 10.0
        scored.append((score, CommunityHit(
            community_id=r["community_id"], title=r["title"], summary=r["summary"],
            level=r["level"], rating=r.get("rating") or 0.0,
            cited_fact_uuids=list(r["cited_fact_uuids"]), full_report=r["full_report"],
            similarity=sim)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [h for _, h in scored[:k]]


async def shortlist_communities(driver, embedder, q: str, *, level: int, k: int,
                                group_id: str, rating_boost: float = 0.1) -> list[CommunityHit]:
    query_vec = (await embedder.create_batch([q]))[0]
    async with driver.session() as s:
        r = await s.run(
            "MATCH (c:Community {group_id:$g, level:$lvl}) "
            "RETURN c.community_id AS community_id, c.title AS title, "
            "coalesce(c.summary,'') AS summary, c.level AS level, "
            "coalesce(c.rating,0.0) AS rating, "
            "coalesce(c.cited_fact_uuids,[]) AS cited_fact_uuids, "
            "coalesce(c.full_report,'[]') AS full_report, c.embedding AS embedding",
            g=group_id, lvl=level)
        rows = [dict(rec) async for rec in r if rec["embedding"]]
    return _rank_hits(query_vec, rows, k=k, rating_boost=rating_boost)
