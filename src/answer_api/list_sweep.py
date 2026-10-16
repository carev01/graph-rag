"""Keyword sweep for list mode (spec 2026-10-15 §12.1): a full-text search of fact text
for the question's own distinctive words. Every word comes from the question and every
hit is a stored fact -- nothing is inferred. It reaches long-tail items whose facts hang
off subjects the relation expansion never visits (2026-10-16: every item still missing
at 250 facts had a fact naming it, e.g. "... backups created by Veeam Agents to the HPE
Morpheus VM Essentials environment")."""
from __future__ import annotations

import re

from answer_api.scope import Scope, scope_episode_uuids
from graph_extract.provenance import Provenance

# A plural noun: ends in -s but not -ss/-us/-is ("process", "status", "analysis").
_PLURAL = r"\w+s(?<!ss)(?<!us)(?<!is)"
_ITEM_NOUNS = (r"(options|ways|methods|types|kinds|destinations|targets|choices|"
               r"alternatives|modes|tiers|levels|formats|platforms|locations)")

STOPWORDS = frozenset((
    "a an the and or not of for to in on at by with from into via as is are was were be "
    "been can could do does did will would should must may might shall i you we they it "
    "its my our your me us which what where when how who whom whose why list all any "
    "every each some there their this that these those than then also about between "
    "other support supports supported supporting provide provides offer offers use uses "
    "using allow allows enable enables available possible").split())
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")
# The list's own item noun ("Which workloads ...", "What recovery options ...").
_ITEM_NOUN_RE = re.compile(r"^\s*(?:which|what)\s+(?:[\w\-]+\s+){0,3}?(" + _PLURAL + r")\b",
                           re.IGNORECASE)
_ITEM_NOUN_WORDS = frozenset(_ITEM_NOUNS.strip("()").split("|"))
# Longest first; a stem keeps at least 4 characters.
_SUFFIXES = ("ility", "ations", "ation", "ions", "ion", "ings", "ing", "ies", "es", "ed",
             "er", "s", "e", "y")

FULLTEXT_INDEX = "edge_name_and_fact"
SWEEP_QUERY = (
    "CALL db.index.fulltext.queryRelationships($index, $query, {limit: $fetch}) "
    "YIELD relationship AS f, score "
    "WHERE f.group_id = $group_id AND (f.invalid_at IS NULL OR f.invalid_at > datetime()) "
    "RETURN f.uuid AS uuid, f.fact AS fact, f.episodes AS episodes, score")


def stem(word: str) -> str:
    """Crude suffix strip for prefix matching: immutability -> immutab, restore ->
    restor, recovery -> recover, databases -> databas; agent stays agent."""
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


def question_terms(q: str, scope: Scope | None) -> list[str]:
    """The question's distinctive words: no stopwords or generic verbs, no list item
    noun, no word of the scope's vendor/product/source names (the scope already
    restricts to them)."""
    drop = set(STOPWORDS) | _ITEM_NOUN_WORDS
    m = _ITEM_NOUN_RE.search(q)
    if m:
        drop.add(m.group(1).lower())
    if scope is not None:
        for name in (*scope.vendors, *scope.products, *scope.sources):
            drop.update(_WORD_RE.findall(name.lower()))
    out: list[str] = []
    for w in _WORD_RE.findall(q.lower()):
        if w not in drop and w not in out:
            out.append(w)
    return out


def lucene_query(terms: list[str]) -> str:
    """OR of every term, plus its stem as a prefix when the stem is a plain word of at
    least 4 characters; '-' is escaped (Lucene syntax) and never wildcarded."""
    parts: list[str] = []
    for t in terms:
        if "-" in t:
            parts.append(t.replace("-", "\\-"))
            continue
        parts.append(t)
        s = stem(t)
        if len(s) >= 4:
            parts.append(f"{s}*")
    return " ".join(parts)


async def keyword_sweep(driver, q: str, *, scope: Scope | None, group_id: str,
                        settings) -> list[dict]:
    """Up to `list_sweep_cap` current facts matching the question's words, best BM25
    first; scoped questions over-fetch `list_sweep_scoped_overfetch` x the cap and keep
    only facts with an episode in scope. Sources resolve through the provenance chain."""
    terms = question_terms(q, scope)
    cap = settings.list_sweep_cap
    if not terms or cap <= 0:
        return []
    scoped = scope is not None and not scope.is_empty()
    fetch = cap * (settings.list_sweep_scoped_overfetch if scoped else 1)
    rows, _, _ = await driver.execute_query(
        SWEEP_QUERY, index=FULLTEXT_INDEX, query=lucene_query(terms), fetch=fetch,
        group_id=group_id)
    rows = sorted(rows, key=lambda r: -r["score"])        # stable: BM25 order
    if scoped:
        assert scope is not None
        allowed = await scope_episode_uuids(
            driver, scope, candidates=[e for r in rows for e in r["episodes"] or ()])
        rows = [r for r in rows if allowed.intersection(r["episodes"] or ())]
    kept: dict[str, dict] = {}
    for r in rows:
        if len(kept) >= cap:
            break
        kept.setdefault(r["uuid"], r)
    if not kept:
        return []
    resolved = await Provenance(driver).resolve_citations(list(kept))
    return [{"fact": r["fact"], "fact_uuid": u,
             "valid_at": resolved.get(u, {}).get("valid_at"),
             "invalid_at": resolved.get(u, {}).get("invalid_at"),
             "sources": resolved.get(u, {}).get("sources", [])} for u, r in kept.items()]
