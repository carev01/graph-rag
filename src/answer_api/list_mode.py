"""List mode (spec 2026-10-15): exhaustive answers to single-vendor list questions.

Detection is deterministic. Expansion asks the cheap tier for retrieval queries that
name likely KINDS of item (hypervisors, clouds, ...): a paraphrase expansion did not
find the missing items (measured 2026-10-15), item-kind queries did. The model's
domain knowledge only steers which facts are searched for; the answer states only
what retrieved facts say."""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import re
import time
from dataclasses import dataclass, field

import numpy as np

from answer_api.attribution import ATTRIBUTION_RULES, CLAIM_RULES, applies_to, fact_line
from answer_api.fact_dedup import collapse_near_duplicates, fetch_vectors
from answer_api.item_grades import CONFIRMED, NEEDS_VALIDATION, grade_items
from answer_api.list_sweep import (
    _ITEM_NOUNS,
    _PLURAL,
    grading_terms,
    is_limitation,
    keyword_sweep,
)
from answer_api.rerank import rerank, rerank_configured
from answer_api.scope import scope_episode_uuids
from answer_api.search import search_local
from answer_api.shape import raise_if_cancelled
from answer_api.synthesize import _REFUSAL, _complete_with_finish_reason, _finalize_answer
from graph_extract.provenance import Provenance
from graph_extract.usage import usable_content

logger = logging.getLogger(__name__)

# What may follow "which/what <plural>" in a list question: a helper or support verb
# ("Which hypervisors does ...", "What destinations can ...", "Which databases are
# supported?"). "What happens when ..." / "What causes a job to fail?" have none.
_HELPER = (r"(does|do|did|can|could|are|were|will|would|should|must|may|might|"
           r"support|supports|is\s+supported)")
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
    "under a few short headings that the facts suggest. Write each item as one line in "
    "exactly this form: - **<item name>** — <short description> [N]. Name each item the "
    "way the facts name it (the product, platform or feature the fact names), not by a "
    "broader category. Include an item even "
    "when its fact states it more generally than the question asks (for example for every "
    "agent rather than one); such items are checked and flagged automatically. When "
    "several facts state the same item, put it on one line "
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
DIVERSITY_DEGRADED = "diversity selection unavailable"
SWEEP_DEGRADED = "keyword search unavailable"
LIST_TRUNCATED = "list answer cut short"
# The list synthesis budget (spec §11), and the one wider retry when the reply is cut
# short at it (finish_reason "length").
_LIST_MAX_TOKENS = 10000
_LIST_RETRY_MAX_TOKENS = 16000
_warned_unconfigured = False


def _degrade(degraded: list[str], note: str) -> None:
    """Record a fixed degraded note at most once (two rerank rounds can both fail)."""
    if note not in degraded:
        degraded.append(note)


async def _final_order(q: str, facts: list[dict], *, settings,
                       degraded: list[str]) -> list[tuple[dict, float | None]]:
    """Every fact with its cross-encoder score against the original question, best
    first; reranked in batches of `list_rerank_batch` (Voyage takes ~1000 documents
    per call) and merged by score. An unconfigured, failed or unreadable rerank keeps
    the merge order with no scores and records FINAL_RERANK_DEGRADED in `degraded`:
    the budgets then cut by retrieval order, and the answer must say so."""
    global _warned_unconfigured
    if not facts:
        return []
    unranked: list[tuple[dict, float | None]] = [(f, None) for f in facts]
    if not rerank_configured(settings):
        if not _warned_unconfigured:
            _warned_unconfigured = True
            logger.warning("no reranker configured; list answers keep retrieval order")
        _degrade(degraded, FINAL_RERANK_DEGRADED)
        return unranked
    size = max(1, settings.list_rerank_batch)
    batches = [facts[start:start + size] for start in range(0, len(facts), size)]
    # The batches run concurrently: scores are absolute (same model, same question),
    # so merging them by score does not depend on which batch finished first.
    replies = await asyncio.gather(*[
        rerank(q, [r["fact"] for r in batch], top_k=len(batch), settings=settings)
        for batch in batches], return_exceptions=True)
    raise_if_cancelled(replies)
    scored: list[tuple[dict, float | None]] = []
    for batch, ranked in zip(batches, replies):
        if isinstance(ranked, BaseException):
            logger.warning("list answer rerank failed (%s); keeping retrieval order", ranked)
            ranked = None
        if ranked is None:
            _degrade(degraded, FINAL_RERANK_DEGRADED)
            return unranked
        seen: set[int] = set()
        batch_scored: list[tuple[dict, float | None]] = []
        for i, score in ranked:
            if 0 <= i < len(batch) and i not in seen:
                seen.add(i)
                batch_scored.append((batch[i], score))
        if not seen:
            # A reply without one valid index ranked nothing: a failed rerank, not a
            # batch of facts that all scored nothing.
            logger.warning("list answer rerank returned no valid index for a batch of %d; "
                           "keeping retrieval order", len(batch))
            _degrade(degraded, FINAL_RERANK_DEGRADED)
            return unranked
        scored += batch_scored
        # A short rerank reply keeps the unranked facts, after every ranked one.
        scored += [(batch[i], None) for i in range(len(batch)) if i not in seen]
    # Stable: equal scores keep reply order; unscored facts go last.
    scored.sort(key=lambda fs: (fs[1] is None, -(fs[1] or 0.0)))
    return scored


SEED_EDGES_QUERY = (
    "MATCH (a:Entity)-[f:RELATES_TO]->(:Entity) WHERE f.uuid IN $uuids "
    "RETURN f.uuid AS uuid, a.uuid AS src, f.name AS rel")
# Unordered and unscoped on purpose: an ORDER BY or an in-query scope EXISTS walks
# EVERY edge of a big group before the limit (434 s for one product scope,
# 2026-10-16). The group is over-fetched (`fetch` = 4 x the group cap), filtered by
# scope in one bounded lookup, then sorted by uuid and capped in Python.
SIBLINGS_QUERY = (
    "MATCH (a:Entity {uuid: $src})-[f:RELATES_TO]->(:Entity) WHERE f.name = $rel "
    "AND (f.invalid_at IS NULL OR f.invalid_at > datetime()) "
    "RETURN f.uuid AS uuid, f.fact AS fact, f.episodes AS episodes LIMIT $fetch")


async def _group_siblings(driver, src: str, rel: str, *, scope, cap: int) -> list[dict]:
    """One (subject, relation) group: up to 4 x `cap` current sibling edges; when
    scoped, only those with an episode in scope (the lookup is bounded by the fetched
    edges' episodes); then the `cap` lowest uuids, so a static graph gives the same
    siblings on every run."""
    rows, _, _ = await driver.execute_query(SIBLINGS_QUERY, src=src, rel=rel, fetch=cap * 4)
    if scope is not None and not scope.is_empty():
        allowed = await scope_episode_uuids(
            driver, scope, candidates=[e for r in rows for e in r["episodes"] or ()])
        rows = [r for r in rows if allowed.intersection(r["episodes"] or ())]
    return sorted(rows, key=lambda r: r["uuid"])[:cap]


async def relation_siblings(driver, seeds: list[dict], *, scope, settings,
                            exclude: set[str], degraded: list[str]) -> list[dict]:
    """Same subject entity + same relation name as the top round-one facts (spec §10.1):
    grounded recall for long-tail items the expansion never named. The groups run
    concurrently; a failed one records RELATION_EXPANSION_DEGRADED (once) and the
    other groups still count. At most `list_relation_total_cap` siblings are kept,
    in group order. Sources resolve deterministically through the provenance chain;
    no LLM is involved."""
    top = seeds[:settings.list_relation_seed_facts]
    if not top:
        return []
    meta, _, _ = await driver.execute_query(
        SEED_EDGES_QUERY, uuids=[r["fact_uuid"] for r in top])
    groups = collections.Counter((m["src"], m["rel"]) for m in meta if m["src"] and m["rel"])
    pairs = [pair for pair, _n in groups.most_common(settings.list_relation_groups)]
    found = await asyncio.gather(*[
        _group_siblings(driver, src, rel, scope=scope, cap=settings.list_relation_group_cap)
        for src, rel in pairs], return_exceptions=True)
    raise_if_cancelled(found)
    rows: dict[str, dict] = {}
    for got in found:
        if isinstance(got, BaseException):
            logger.warning("list relation expansion failed for one group (%s)", got)
            _degrade(degraded, RELATION_EXPANSION_DEGRADED)
            continue
        for r in got:
            if len(rows) >= settings.list_relation_total_cap:
                break
            if r["uuid"] not in exclude and r["uuid"] not in rows:
                rows[r["uuid"]] = r
    if not rows:
        return []
    resolved = await Provenance(driver).resolve_citations(list(rows))
    return [{"fact": r["fact"], "fact_uuid": u,
             "valid_at": resolved.get(u, {}).get("valid_at"),
             "invalid_at": resolved.get(u, {}).get("invalid_at"),
             "sources": resolved.get(u, {}).get("sources", [])} for u, r in rows.items()]


async def _merge_round_two(q: str, round_one: list[tuple[dict, float | None]],
                           new: list[dict], *, settings,
                           degraded: list[str]) -> list[tuple[dict, float | None]]:
    """Rerank only the NEW (sibling) facts and merge them with round one's scores --
    same model, same question, so the scores are comparable. When round one has no
    scores at all (degraded rerank, already recorded) the siblings follow in
    retrieval order without a second call; a failed round two keeps round one's
    scores and puts the siblings last, unscored (FINAL_RERANK_DEGRADED recorded)."""
    if not new:
        return round_one
    if all(sc is None for _, sc in round_one):
        return round_one + [(f, None) for f in new]
    merged = round_one + await _final_order(q, new, settings=settings, degraded=degraded)
    # Stable: equal scores keep round one first; unscored facts go last.
    merged.sort(key=lambda fs: (fs[1] is None, -(fs[1] or 0.0)))
    return merged


async def _fact_vectors(driver, uuids: list[str], *, cache: dict[str, np.ndarray],
                        degraded: list[str]) -> dict[str, np.ndarray]:
    """fact_embedding per uuid, from the question's cache (the collapses already
    fetched nearly all of them; only uuids not yet cached are queried). {} when the
    fetch fails -- selection then keeps rerank order, and DIVERSITY_DEGRADED is
    recorded so the answer says so."""
    if not uuids:
        return {}
    try:
        return await fetch_vectors(driver, uuids, cache)
    except Exception as exc:
        logger.warning("fact embedding fetch failed (%s); list selection keeps rerank order",
                       exc)
        _degrade(degraded, DIVERSITY_DEGRADED)
        return {}


def select_diverse(scored: list[tuple[dict, float | None]], vectors: dict[str, np.ndarray],
                   *, k: int, lam: float) -> list[dict]:
    """Maximal marginal relevance (spec §10.3): greedily the fact with the best rerank
    score minus `lam` x its highest RAW cosine to an already chosen fact. A fact
    without a usable vector is never penalised and never penalises; with no vectors
    at all, or no scores at all (degraded rerank), this is the given order. Ties go to the earlier (better-ranked) fact."""
    if not scored or k <= 0:
        return []
    if not vectors or all(sc is None for _, sc in scored):
        # No relevance signal (degraded rerank): MMR would become farthest-point
        # sampling and favour outliers; keep retrieval order, as the note says.
        return [f for f, _ in scored[:k]]
    dim = len(next(iter(vectors.values())))
    n = len(scored)
    rows = [vectors.get(f["fact_uuid"]) for f, _ in scored]
    has = np.array([v is not None and len(v) == dim for v in rows], dtype=bool)
    mat = np.zeros((n, dim))
    picked = [v for v in rows if v is not None and len(v) == dim]
    if picked:
        mat[has] = np.stack(picked)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.divide(mat, norms, out=mat, where=norms > 0)
    rel = np.array([s if s is not None else 0.0 for _, s in scored], dtype=np.float64)
    best_sim = np.zeros(n)
    taken = np.zeros(n, dtype=bool)
    chosen: list[int] = []
    while len(chosen) < min(k, n):
        gain = np.where(taken, -np.inf, rel - lam * best_sim)
        i = int(np.argmax(gain))           # argmax returns the first (best-ranked) tie
        chosen.append(i)
        taken[i] = True
        if has[i]:
            best_sim = np.maximum(best_sim, np.where(has, mat @ mat[i], 0.0))
    return [scored[i][0] for i in chosen]


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
    if DIVERSITY_DEGRADED in degraded:
        parts.append("repeated items could not be filtered out")
    if SWEEP_DEGRADED in degraded:
        parts.append("the keyword search could not be completed")
    if LIST_TRUNCATED in degraded:
        parts.append("the answer was cut short")
    return " and ".join(parts).capitalize() + "; this list may be incomplete." if parts else ""


@dataclass
class ListRetrieval:
    results: list[dict]
    limitations: list[dict]
    expansion_queries: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    # Final rerank score per fact_uuid (None: unscored); the item grades' floor.
    scores: dict[str, float | None] = field(default_factory=dict)


async def retrieve_list(graphiti, driver, map_client, map_model, *, q, scope, group_id,
                        settings) -> ListRetrieval:
    stages: dict[str, float] = collections.defaultdict(float)
    clock = time.monotonic()

    def lap(stage: str) -> None:
        nonlocal clock
        now = time.monotonic()
        stages[stage] += now - clock
        clock = now

    expansions = await expand_queries(map_client, map_model, q,
                                      settings.list_expansion_queries_max)
    lap("expansion")
    queries = [q] + [x for x in expansions if x != q]
    found = await asyncio.gather(*[
        search_local(graphiti, driver, q=sq, k=settings.list_per_query_k, scope=scope,
                     group_id=group_id, fetch_cap=settings.local_scoped_fetch_cap,
                     scan_max_episodes=settings.local_scoped_scan_max_episodes,
                     settings=settings, rerank_pool=settings.list_per_query_pool)
        for sq in queries] + [
        keyword_sweep(driver, q, scope=scope, group_id=group_id, settings=settings)],
        return_exceptions=True)
    raise_if_cancelled(found)
    lap("searches")
    per_query, swept_result = found[:-1], found[-1]
    merged: list[dict] = []
    degraded: list[str] = []
    for sq, res in zip(queries, per_query):
        if isinstance(res, BaseException):
            logger.warning("list retrieval failed for %r: %s", sq, res)
            degraded.append(f"retrieval failed for: {sq}")
            continue
        merged.extend(res["results"])
    if len(degraded) == len(queries):
        raise RuntimeError("every list retrieval failed")
    if isinstance(swept_result, BaseException):
        logger.warning("list keyword sweep failed (%s); continuing without it", swept_result)
        _degrade(degraded, SWEEP_DEGRADED)
        swept: list[dict] = []
    else:
        swept = swept_result
        merged.extend(swept)
    # One embedding cache per question: both collapses and the MMR selection read the
    # same fact vectors, each fetched at most once (was three fetches, ~50-70 s).
    vectors: dict[str, np.ndarray] = {}
    distinct = await collapse_near_duplicates(driver, merged,
                                              cosine=settings.list_dedup_cosine,
                                              vectors=vectors)
    lap("collapse")
    ranked = await _final_order(q, distinct, settings=settings, degraded=degraded)
    lap("rerank")
    try:
        siblings = await relation_siblings(
            driver, [f for f, _ in ranked], scope=scope, settings=settings,
            exclude={f["fact_uuid"] for f in distinct}, degraded=degraded)
    except Exception as exc:
        logger.warning("list relation expansion failed (%s); answering from round one", exc)
        _degrade(degraded, RELATION_EXPANSION_DEGRADED)
        siblings = []
    lap("relations")
    if siblings:
        scores = {f["fact_uuid"]: sc for f, sc in ranked}
        distinct = await collapse_near_duplicates(driver, [f for f, _ in ranked] + siblings,
                                                  cosine=settings.list_dedup_cosine,
                                                  vectors=vectors)
        lap("collapse")
        # A sibling merged into a round-one fact keeps that fact's slot and score.
        ranked = await _merge_round_two(
            q, [(f, scores[f["fact_uuid"]]) for f in distinct if f["fact_uuid"] in scores],
            [f for f in distinct if f["fact_uuid"] not in scores],
            settings=settings, degraded=degraded)
        lap("rerank")
    supported: list[tuple[dict, float | None]] = []
    limits: list[tuple[dict, float | None]] = []
    for fs in ranked:
        (limits if is_limitation(fs[0]["fact"]) else supported).append(fs)
    selection_vectors = await _fact_vectors(driver, [f["fact_uuid"] for f, _ in supported],
                                            cache=vectors, degraded=degraded)
    results = await asyncio.to_thread(select_diverse, supported, selection_vectors,
                                      k=settings.list_supported_k,
                                      lam=settings.list_mmr_lambda)
    lap("selection")
    logger.info("list stages (s): %s; candidates=%d siblings=%d swept=%d",
                " ".join(f"{name}={stages[name]:.1f}" for name in
                         ("expansion", "searches", "collapse", "rerank", "relations",
                          "selection")),
                len(ranked), len(siblings), len(swept))
    return ListRetrieval(results=results,
                         limitations=[f for f, _ in limits][:settings.list_limitations_k],
                         expansion_queries=expansions, degraded=degraded,
                         scores={f["fact_uuid"]: sc for f, sc in ranked})


async def _list_synthesis(client, model: str, prompt: str, *,
                          degraded: list[str]) -> str | None:
    """The list answer; None when the model gave nothing usable. A reply cut short at
    its budget (finish_reason "length") is retried ONCE at `_LIST_RETRY_MAX_TOKENS`,
    unless its budget was already that large; cut short again (or unusable), the
    longer partial text is kept and LIST_TRUNCATED recorded, so the answer says it
    was cut short rather than reading as the whole list."""
    first = await _complete_with_finish_reason(client, model, prompt,
                                               max_tokens=_LIST_MAX_TOKENS)
    if first.content is None or first.finish_reason != "length":
        return first.content
    if first.max_tokens >= _LIST_RETRY_MAX_TOKENS:
        logger.warning("list answer cut short (finish_reason=length) at max_tokens=%d; "
                       "prompt %d chars; no wider retry, keeping the partial list",
                       first.max_tokens, len(prompt))
        _degrade(degraded, LIST_TRUNCATED)
        return first.content
    logger.warning("list answer cut short (finish_reason=length) at max_tokens=%d; "
                   "prompt %d chars; retrying with max_tokens=%d",
                   first.max_tokens, len(prompt), _LIST_RETRY_MAX_TOKENS)
    retry = await _complete_with_finish_reason(client, model, prompt,
                                               max_tokens=_LIST_RETRY_MAX_TOKENS)
    if retry.content is not None and retry.finish_reason != "length":
        return retry.content
    logger.warning("list answer still cut short at max_tokens=%d; prompt %d chars; "
                   "keeping the longer partial list", retry.max_tokens, len(prompt))
    _degrade(degraded, LIST_TRUNCATED)
    return max((first.content, retry.content or ""), key=len)


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
                "applies_to": [], "item_grades": [], "items_ungraded": 0}
    marker_map = {i: r for i, r in enumerate(facts, 1)}
    n_sup = len(got.results)
    sup = "\n".join(fact_line(i, r["fact"], r["sources"])
                    for i, r in marker_map.items() if i <= n_sup) or "(none)"
    lim = "\n".join(fact_line(i, r["fact"], r["sources"])
                    for i, r in marker_map.items() if i > n_sup) or "(none)"
    # Up to 250 supported facts: room for a long grouped list plus the synthesis
    # tier's reasoning tokens.
    raw = await _list_synthesis(synth_client, synth_model,
                                _LIST_PROMPT.format(supported=sup, limitations=lim, q=q),
                                degraded=got.degraded)
    if got.degraded:
        base["degraded"] = got.degraded
    if raw is None:
        return {**base, "answer": _REFUSAL, "citations": [], "retrieved": len(facts),
                "cited": 0, "applies_to": [], "item_grades": [], "items_ungraded": 0}
    # Deterministic per-item grades (spec §12.2) BEFORE marker resolution, so a
    # dropped item's markers are never cited.
    graded = grade_items(raw, marker_map, q=q, terms=grading_terms(q, scope),
                         scores=got.scores, floor=settings.list_item_floor,
                         unnamed_floor=settings.list_item_unnamed_floor, scope=scope,
                         flag_group_min=settings.list_flag_group_min)
    item_grades = [g.as_dict(marker_map) for g in graded.grades]
    if graded.grades and not any(g.grade in (CONFIRMED, NEEDS_VALIDATION)
                                 for g in graded.grades):
        # Every item was dropped: whatever text remains is not a grounded list, so
        # refuse and let the router take its normal route.
        logger.info("list answer: every item dropped by grading; taking the normal route")
        return {**base, "answer": _REFUSAL, "citations": [], "retrieved": len(facts),
                "cited": 0, "applies_to": [], "item_grades": item_grades,
                "items_ungraded": graded.ungraded}
    answer, cited = _finalize_answer(graded.text, marker_map)
    note = degraded_note(got.degraded)
    if note and cited:
        answer = f"{answer}\n\n{note}"
    citations = [{"marker": m, "fact": marker_map[m]["fact"],
                  "fact_uuid": marker_map[m]["fact_uuid"],
                  "valid_at": marker_map[m].get("valid_at"),
                  "invalid_at": marker_map[m].get("invalid_at"),
                  "sources": marker_map[m]["sources"]} for m in cited]
    return {**base, "answer": answer, "citations": citations, "retrieved": len(facts),
            "cited": len(cited), "applies_to": applies_to(citations),
            "item_grades": item_grades, "items_ungraded": graded.ungraded}
