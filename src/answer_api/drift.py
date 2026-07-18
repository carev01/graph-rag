"""DRIFT search: broad->deep. Primer on the :Community report layer drafts a
preliminary answer + follow-up queries; each follow-up runs a graph-biased local
fact search; a final synthesis merges the evidence into one cited answer.
Citations are graph traversal (design decision #2): the LLM emits only [N]
markers / fact UUIDs, never a URL."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from answer_api.search import search_local
from answer_api.synthesize import _finalize_answer, answer_local
from answer_api.global_search import shortlist_communities, _extract_json
from graph_extract.provenance import Provenance

logger = logging.getLogger(__name__)

_REFUSAL = "I don't have enough information to answer that from the available sources."


@dataclass
class FollowUp:
    query: str
    community_id: str | None
    iteration: int = 1


def _parse_followups(obj: dict, hit_ids: set[str], max_followups: int,
                     iteration: int) -> list[FollowUp]:
    scored: list[tuple[float, FollowUp]] = []
    for item in obj.get("follow_ups") or []:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        cid = item.get("community_id")
        cid = cid if (isinstance(cid, str) and cid in hit_ids) else None
        try:
            rel = float(item.get("relevance", 0) or 0)
        except (TypeError, ValueError):
            rel = 0.0
        scored.append((rel, FollowUp(query=query, community_id=cid, iteration=iteration)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [fu for _, fu in scored[:max_followups]]


_PRIMER_PROMPT = (
    "You are starting a broad investigation of a QUESTION about backup products. "
    "Use the COMMUNITY themes below as leads (not as the final answer). Draft a "
    "brief preliminary answer and 3-6 targeted follow-up questions that would "
    "retrieve concrete supporting facts. Tag each follow-up with the community_id "
    "it draws from, or null. Respond with ONLY JSON: {{\"preliminary_answer\": str, "
    "\"follow_ups\": [{{\"query\": str, \"community_id\": str|null, "
    "\"relevance\": 0-10}}]}}. Do NOT write URLs.\n\n"
    "QUESTION: {q}\n\nCOMMUNITIES:\n{blocks}"
)


async def _primer(embedder, synth_client, synth_model, driver, *, q, level, k,
                  max_followups, group_id):
    hits = await shortlist_communities(driver, embedder, q, level=level, k=k,
                                       group_id=group_id)
    if not hits:
        return None
    blocks = "\n".join(f'- {h.community_id} "{h.title}": {h.summary}' for h in hits)
    obj: dict | None = None
    for _ in range(2):
        resp = await synth_client.chat.completions.create(
            model=synth_model, temperature=0, max_tokens=2000,
            messages=[{"role": "user",
                       "content": _PRIMER_PROMPT.format(q=q, blocks=blocks)}])
        obj = _extract_json(resp.choices[0].message.content or "")
        if obj is not None:
            break
    hit_ids = {h.community_id for h in hits}
    if obj is None:
        return "", [FollowUp(query=q, community_id=None, iteration=1)], hits
    preliminary = str(obj.get("preliminary_answer", "")).strip()
    fus = _parse_followups(obj, hit_ids, max_followups, iteration=1)
    if not fus:
        fus = [FollowUp(query=q, community_id=None, iteration=1)]
    return preliminary, fus, hits
