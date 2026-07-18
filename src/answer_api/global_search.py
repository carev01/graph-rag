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


@dataclass
class MapResult:
    community_id: str
    title: str
    relevance: int
    key_points: list[str]
    fact_ids: list[str]


_MAP_PROMPT = (
    "You are assessing one COMMUNITY report for relevance to a QUESTION. Using ONLY "
    "the report, respond with JSON: {{\"relevance\": 0-10 (how useful for the question), "
    "\"key_points\": [short strings relevant to the question], \"fact_ids\": [the fact "
    "uuids from the report that support those points]}}. fact_ids MUST be uuids that "
    "appear in the report. Do NOT write URLs.\n\n"
    "QUESTION: {q}\n\nCOMMUNITY \"{title}\": {summary}\nFINDINGS: {full_report}"
)


def _map_client_and_model(settings: ExtractSettings) -> tuple[AsyncOpenAI, str]:
    base = settings.map_llm_base_url or settings.judge_base_url
    model = settings.map_llm_model or settings.judge_model
    key = settings.map_llm_api_key or settings.judge_api_key or "not-needed"
    if not base or not model:
        raise ValueError("No map model configured. Set map_llm_* or judge_* (GLM-5.2).")
    return instrument(AsyncOpenAI(api_key=key, base_url=base)), model


def _extract_json(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


async def map_report(client: AsyncOpenAI, model: str, q: str, hit: CommunityHit, *,
                     relevance_min: int) -> MapResult | None:
    prompt = _MAP_PROMPT.format(q=q, title=hit.title, summary=hit.summary,
                                full_report=hit.full_report)
    obj: dict | None = None
    for _ in range(2):
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}])
        obj = _extract_json(resp.choices[0].message.content or "")
        if obj is not None:
            break
    if obj is None:
        return None
    try:
        relevance = int(obj.get("relevance", 0) or 0)
    except (TypeError, ValueError):
        relevance = 0
    if relevance < relevance_min:
        return None
    valid = {u for u in hit.cited_fact_uuids}
    fact_ids = [f for f in (obj.get("fact_ids") or []) if f in valid]
    key_points = [str(p) for p in (obj.get("key_points") or [])]
    return MapResult(community_id=hit.community_id, title=hit.title, relevance=relevance,
                     key_points=key_points, fact_ids=fact_ids)


_REDUCE_PROMPT = (
    "Answer the QUESTION by synthesizing across these community findings, organized "
    "by theme and vendor. Cite every claim with the [N] fact markers shown. Use ONLY "
    "these findings. Do NOT write any URL. If nothing is relevant, reply exactly: "
    "\"" + _REFUSAL + "\"\n\nQUESTION: {q}\n\nFINDINGS:\n{blocks}\n\nAnswer:"
)


async def global_search(driver, embedder, map_client: AsyncOpenAI, map_model: str,
                        synth_client: AsyncOpenAI, synth_model: str, *, q: str,
                        level: int, k: int, group_id: str, relevance_min: int) -> dict:
    hits = await shortlist_communities(driver, embedder, q, level=level, k=k, group_id=group_id)
    if not hits:
        return {"query": q, "answer": _REFUSAL, "citations": [], "communities_used": []}
    maps = await asyncio.gather(
        *[map_report(map_client, map_model, q, h, relevance_min=relevance_min) for h in hits],
        return_exceptions=True)
    results: list[MapResult] = []
    for m in maps:
        if isinstance(m, BaseException):
            logger.warning("global map failed: %s", m)
        elif m is not None:
            results.append(m)
    if not results:
        return {"query": q, "answer": _REFUSAL, "citations": [], "communities_used": []}
    # number the ordered-unique union of fact ids -> marker_map
    marker_map: dict[int, dict] = {}
    fact_to_marker: dict[str, int] = {}
    for m in results:
        for fid in m.fact_ids:
            if fid not in fact_to_marker:
                idx = len(fact_to_marker) + 1
                fact_to_marker[fid] = idx
                marker_map[idx] = {"fact_uuid": fid}
    blocks = []
    for m in results:
        markers = " ".join(f"[{fact_to_marker[f]}]" for f in m.fact_ids)
        pts = "\n".join(f"- {p}" for p in m.key_points)
        blocks.append(f"COMMUNITY \"{m.title}\" (relevance {m.relevance}):\n{pts}\n"
                      f"Supporting facts: {markers}")
    resp = await synth_client.chat.completions.create(
        model=synth_model, temperature=0, max_tokens=3000,
        messages=[{"role": "user", "content": _REDUCE_PROMPT.format(q=q, blocks="\n\n".join(blocks))}])
    answer, cited = _finalize_answer(resp.choices[0].message.content or "", marker_map)
    resolved = await Provenance(driver).resolve_citations(
        [marker_map[m]["fact_uuid"] for m in cited])
    citations = [{"marker": m, "fact_uuid": marker_map[m]["fact_uuid"],
                  "sources": resolved.get(marker_map[m]["fact_uuid"], [])} for m in cited]
    return {"query": q, "answer": answer, "citations": citations,
            "communities_used": [{"community_id": m.community_id, "title": m.title,
                                  "relevance": m.relevance} for m in results]}
