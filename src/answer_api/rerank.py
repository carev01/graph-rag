"""Cross-encoder reranking for community selection (Voyage AI, Cohere-compatible).

The global map step used to ask an LLM for `"relevance": 0-10 (how useful)` with no
anchors and no definition of relevance. Measured 2026-09-10, it inverted: a community
about KMS key policies scored 3 on an encryption question while an Azure BCDR platform
overview with no encryption facts scored 6. A cross-encoder is trained and calibrated
for exactly this comparison; an LLM rating 0-10 is improvising a scale.
"""
from __future__ import annotations

import logging

import httpx

from graph_extract.config import ExtractSettings

logger = logging.getLogger(__name__)


def rerank_configured(settings: ExtractSettings) -> bool:
    return bool(settings.rerank_base_url and settings.rerank_model)


async def rerank(query: str, documents: list[str], *, top_k: int,
                 settings: ExtractSettings,
                 transport: httpx.AsyncBaseTransport | None = None,
                 ) -> list[tuple[int, float]] | None:
    """Score `documents` against `query`, best first.

    Returns None when relevance COULD NOT BE SCORED -- a non-200, a transport error,
    or a payload we cannot read. None is not "nothing is relevant": the caller must
    degrade visibly rather than treat an outage as an empty result.
    """
    if not documents:
        return []
    payload = {"model": settings.rerank_model, "query": query,
               "documents": documents, "top_k": min(top_k, len(documents))}
    url = settings.rerank_base_url.rstrip("/") + "/rerank"
    try:
        async with httpx.AsyncClient(timeout=30.0, transport=transport) as client:
            resp = await client.post(
                url, json=payload,
                headers={"Authorization": f"Bearer {settings.rerank_api_key}"})
    except httpx.HTTPError as exc:
        logger.warning("rerank transport error (%s); caller must degrade visibly", exc)
        return None
    if resp.status_code != 200:
        logger.warning("rerank returned HTTP %s; caller must degrade visibly",
                       resp.status_code)
        return None
    try:
        rows = resp.json()["data"]
        out = [(int(r["index"]), float(r["relevance_score"])) for r in rows]
    except (ValueError, KeyError, TypeError) as exc:
        logger.warning("rerank payload unreadable (%s); caller must degrade visibly", exc)
        return None
    return [(i, s) for i, s in out if 0 <= i < len(documents)]
