"""List mode (spec 2026-10-15): exhaustive answers to single-vendor list questions.

Detection is deterministic. Expansion asks the cheap tier for retrieval queries that
name likely KINDS of item (hypervisors, clouds, ...): a paraphrase expansion did not
find the missing items (measured 2026-10-15), item-kind queries did. The model's
domain knowledge only steers which facts are searched for; the answer states only
what retrieved facts say."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from answer_api.attribution import ATTRIBUTION_RULES, CLAIM_RULES, applies_to, fact_line
from answer_api.fact_dedup import collapse_near_duplicates
from answer_api.rerank import rerank, rerank_configured
from answer_api.search import search_local
from answer_api.shape import raise_if_cancelled
from answer_api.synthesize import _REFUSAL, _complete_or_none, _finalize_answer
from graph_extract.usage import usable_content

logger = logging.getLogger(__name__)

# A plural noun: ends in -s but not -ss/-us/-is ("process", "status", "analysis").
_PLURAL = r"\w+s(?<!ss)(?<!us)(?<!is)"
# What may follow "which/what <plural>" in a list question: a helper or support verb
# ("Which hypervisors does ...", "What destinations can ...", "Which databases are
# supported?"). "What happens when ..." / "What causes a job to fail?" have none.
_HELPER = (r"(does|do|did|can|could|are|were|will|would|should|must|may|might|"
           r"support|supports|is\s+supported)")
_ITEM_NOUNS = (r"(options|ways|methods|types|kinds|destinations|targets|choices|"
               r"alternatives|modes|tiers|levels|formats|platforms|locations)")
_LIST_RE = re.compile(
    r"^\s*(which|what)\s+(\w+\s+){0,3}?" + _PLURAL + r"\s+" + _HELPER + r"\b"
    r"|^\s*where\s+can\s+(i|you|we)\b"                             # where can I ...
    r"|^\s*what\s+are\s+the\s+(\w+\s+)?" + _ITEM_NOUNS + r"\b"
    r"|^\s*list\b"
    r"|^\s*supported\s+" + _PLURAL + r"\b",
    re.IGNORECASE)
# Questions that are clearly single-answer ("What is ...", "What should I consider
# ...") are excluded explicitly.
_NOT_LIST_RE = re.compile(r"^\s*(what|which)\s+(is|was|should|does|do|did|can)\b",
                          re.IGNORECASE)

_LIMIT_RE = re.compile(
    r"\b(not supported|does not|doesn't|do not|don't|cannot|can't|can not|is not|are not|"
    r"unsupported|only (available|supported|for|when|if|with|on|in)|not available)\b",
    re.IGNORECASE)

_EXPAND_PROMPT = (
    "You help search vendor backup documentation. The QUESTION asks for a list. Write "
    "up to {n} short search queries that together would find EVERY item the "
    "documentation might list as an answer. Name the likely kinds of item explicitly "
    "(for example, for restore destinations: specific hypervisors, clouds, physical "
    "machines, file and application level; for workloads: VMs, physical machines, "
    "databases, SaaS, Kubernetes). Keep the vendor and product names from the question "
    "in every query; when the question names no vendor or product, keep the question's "
    "own terms (for example 'Windows agent'). Never write placeholder words such as "
    "'vendor', 'product' or 'software' in place of a name. Reply with ONLY a JSON list "
    "of strings.\n\n"
    "QUESTION: {q}\nJSON:")

_LIST_PROMPT = (
    "You are answering a question that asks for a LIST, about backup products, using "
    "ONLY the numbered facts below. " + ATTRIBUTION_RULES + CLAIM_RULES +
    "List EVERY item the SUPPORTED FACTS state as an answer -- no item limit -- grouped "
    "under a few short headings that the facts suggest. Give each item one short line "
    "with its [N] marker; when several facts state the same item, put it on one line "
    "with up to three of their markers. After the list, add a section titled \"Not "
    "supported / limitations\" summarising the LIMITATION FACTS briefly with their markers; omit "
    "that section when there are none. Do not pad, do not repeat items, and do not "
    "claim anything is undocumented beyond what the facts show. Do NOT use outside "
    "knowledge. Do NOT write any URL or link. If no fact answers the question, reply "
    "exactly: \"" + _REFUSAL + "\"\n\n"
    "SUPPORTED FACTS:\n{supported}\n\nLIMITATION FACTS:\n{limitations}\n\n"
    "QUESTION: {q}\n\nAnswer:")


def is_list_question(q: str) -> bool:
    return bool(_LIST_RE.search(q)) and not _NOT_LIST_RE.search(q)


def is_limitation(fact: str) -> bool:
    return bool(_LIMIT_RE.search(fact))


async def expand_queries(client, model: str, q: str, max_n: int) -> list[str]:
    """Up to `max_n` item-kind retrieval queries; [] (logged) when unusable."""
    try:
        resp = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=800,
            messages=[{"role": "user", "content": _EXPAND_PROMPT.format(n=max_n, q=q)}])
        content = usable_content(resp)
        if content is None:
            logger.warning("list expansion returned no usable content; using the question alone")
            return []
        start, end = content.find("["), content.rfind("]")
        parsed = json.loads(content[start:end + 1]) if start >= 0 and end > start else None
    except Exception as exc:
        logger.warning("list expansion failed (%s); using the question alone", exc)
        return []
    if not isinstance(parsed, list):
        logger.warning("list expansion reply was not a JSON list; using the question alone")
        return []
    out: list[str] = []
    for item in parsed:
        if isinstance(item, str) and item.strip() and item.strip() not in out:
            out.append(item.strip())
    return out[:max_n]


FINAL_RERANK_DEGRADED = "final rerank unavailable: items ordered by retrieval"
RELATION_EXPANSION_DEGRADED = "relation expansion unavailable"
_warned_unconfigured = False


async def _final_order(q: str, facts: list[dict], *, settings,
                       degraded: list[str]) -> list[dict]:
    """Every fact, best first by the cross-encoder against the original question.
    An unconfigured, failed or unreadable rerank keeps the merge order and records
    FINAL_RERANK_DEGRADED in `degraded`: the budgets then cut by retrieval order,
    and the answer must say so."""
    global _warned_unconfigured
    if not facts:
        return []
    if not rerank_configured(settings):
        if not _warned_unconfigured:
            _warned_unconfigured = True
            logger.warning("no reranker configured; list answers keep retrieval order")
        degraded.append(FINAL_RERANK_DEGRADED)
        return facts
    try:
        ranked = await rerank(q, [r["fact"] for r in facts], top_k=len(facts),
                              settings=settings)
    except Exception as exc:
        logger.warning("list answer rerank failed (%s); keeping retrieval order", exc)
        ranked = None
    if ranked is None:
        degraded.append(FINAL_RERANK_DEGRADED)
        return facts
    order = list(dict.fromkeys(i for i, _ in ranked if 0 <= i < len(facts)))
    chosen = set(order)
    # A short rerank reply keeps the unranked facts, after the ranked ones.
    order += [i for i in range(len(facts)) if i not in chosen]
    return [facts[i] for i in order]


def degraded_note(degraded: list[str]) -> str:
    """Fixed wording only: degraded entries can carry model-written query text,
    which must never reach the answer (the LLM never writes a URL)."""
    if not degraded:
        return ""
    parts = []
    failed = sum(1 for d in degraded if d.startswith("retrieval failed for:"))
    if failed:
        parts.append(f"{failed} search{'es' if failed != 1 else ''} could not be completed")
    if RELATION_EXPANSION_DEGRADED in degraded:
        parts.append("related facts could not be gathered")
    if FINAL_RERANK_DEGRADED in degraded:
        parts.append("results were ranked by retrieval order")
    return " and ".join(parts).capitalize() + "; this list may be incomplete." if parts else ""


@dataclass
class ListRetrieval:
    results: list[dict]
    limitations: list[dict]
    expansion_queries: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)


async def retrieve_list(graphiti, driver, map_client, map_model, *, q, scope, group_id,
                        settings) -> ListRetrieval:
    expansions = await expand_queries(map_client, map_model, q,
                                      settings.list_expansion_queries_max)
    queries = [q] + [x for x in expansions if x != q]
    found = await asyncio.gather(*[
        search_local(graphiti, driver, q=sq, k=settings.list_per_query_k, scope=scope,
                     group_id=group_id, fetch_cap=settings.local_scoped_fetch_cap,
                     scan_max_episodes=settings.local_scoped_scan_max_episodes,
                     settings=settings, rerank_pool=settings.list_per_query_pool)
        for sq in queries], return_exceptions=True)
    raise_if_cancelled(found)
    merged: list[dict] = []
    degraded: list[str] = []
    for sq, res in zip(queries, found):
        if isinstance(res, BaseException):
            logger.warning("list retrieval failed for %r: %s", sq, res)
            degraded.append(f"retrieval failed for: {sq}")
            continue
        merged.extend(res["results"])
    if len(degraded) == len(queries):
        raise RuntimeError("every list retrieval failed")
    distinct = await collapse_near_duplicates(driver, merged,
                                              cosine=settings.list_dedup_cosine)
    ranked = await _final_order(q, distinct, settings=settings, degraded=degraded)
    supported: list[dict] = []
    limits: list[dict] = []
    for r in ranked:
        (limits if is_limitation(r["fact"]) else supported).append(r)
    return ListRetrieval(results=supported[:settings.list_supported_k],
                         limitations=limits[:settings.list_limitations_k],
                         expansion_queries=expansions, degraded=degraded)


async def answer_list(graphiti, driver, map_client, map_model, synth_client, synth_model,
                      *, q, scope, group_id, settings) -> dict:
    got = await retrieve_list(graphiti, driver, map_client, map_model, q=q, scope=scope,
                              group_id=group_id, settings=settings)
    facts = got.results + got.limitations
    base: dict = {"query": q, "list_mode": True, "expansion_queries": got.expansion_queries}
    if got.degraded:
        base["degraded"] = got.degraded
    if not facts:
        return {**base, "answer": _REFUSAL, "citations": [], "retrieved": 0, "cited": 0,
                "applies_to": []}
    marker_map = {i: r for i, r in enumerate(facts, 1)}
    n_sup = len(got.results)
    sup = "\n".join(fact_line(i, r["fact"], r["sources"])
                    for i, r in marker_map.items() if i <= n_sup) or "(none)"
    lim = "\n".join(fact_line(i, r["fact"], r["sources"])
                    for i, r in marker_map.items() if i > n_sup) or "(none)"
    raw = await _complete_or_none(synth_client, synth_model,
                                  _LIST_PROMPT.format(supported=sup, limitations=lim, q=q),
                                  max_tokens=6000)
    if raw is None:
        return {**base, "answer": _REFUSAL, "citations": [], "retrieved": len(facts),
                "cited": 0, "applies_to": []}
    answer, cited = _finalize_answer(raw, marker_map)
    note = degraded_note(got.degraded)
    if note and cited:
        answer = f"{answer}\n\n{note}"
    citations = [{"marker": m, "fact": marker_map[m]["fact"],
                  "fact_uuid": marker_map[m]["fact_uuid"],
                  "valid_at": marker_map[m].get("valid_at"),
                  "invalid_at": marker_map[m].get("invalid_at"),
                  "sources": marker_map[m]["sources"]} for m in cited]
    return {**base, "answer": answer, "citations": citations, "retrieved": len(facts),
            "cited": len(cited), "applies_to": applies_to(citations)}
