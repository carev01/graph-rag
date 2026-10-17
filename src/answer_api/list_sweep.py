"""Keyword sweep for list mode (spec 2026-10-15 §12.1): a full-text search of fact text
for the question's own distinctive words. Every word comes from the question and every
hit is a stored fact -- nothing is inferred. It reaches long-tail items whose facts hang
off subjects the relation expansion never visits (2026-10-16: every item still missing
at 250 facts had a fact naming it, e.g. "... backups created by Veeam Agents to the HPE
Morpheus VM Essentials environment")."""
from __future__ import annotations

import logging
import re

from answer_api.scope import Scope, scope_episode_uuids
from graph_extract.provenance import Provenance

logger = logging.getLogger(__name__)

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

# A fact stating a limitation ("not supported", "only for ..."): list mode budgets these
# separately and the item grades never let one confirm an item.
_LIMIT_RE = re.compile(
    r"\b(not supported|does not|doesn't|do not|don't|cannot|can't|can not|is not|are not|"
    r"unsupported|only (available|supported|for|when|if|with|on|in)|not available)\b",
    re.IGNORECASE)

FULLTEXT_INDEX = "edge_name_and_fact"
SWEEP_QUERY = (
    "CALL db.index.fulltext.queryRelationships($index, $query, {limit: $fetch}) "
    "YIELD relationship AS f, score "
    "WHERE f.group_id = $group_id AND (f.invalid_at IS NULL OR f.invalid_at > datetime()) "
    "RETURN f.uuid AS uuid, f.fact AS fact, f.episodes AS episodes, score")


def is_limitation(fact: str) -> bool:
    return bool(_LIMIT_RE.search(fact))


def stem(word: str) -> str:
    """Crude suffix strip for prefix matching: immutability -> immutab, restore ->
    restor, recovery -> recover, databases -> databas, copies -> copy, policies ->
    policy; agent stays agent. The result keeps at least 4 characters."""
    for suf in _SUFFIXES:
        if word.endswith(suf):
            out = word[: -len(suf)] + ("y" if suf == "ies" else "")
            if len(out) >= 4:
                return out
    return word


def word_forms(term: str) -> list[str]:
    """`term`'s exact inflected forms for the sweep (deduplicated, order kept): the
    term itself; its plural (`term + "s"`, or `term + "es"` when it ends in
    s/x/z/ch/sh) -- unless `term` already ends in "s" and is longer than 4
    characters, in which case its singular (`term[:-1]`) instead, since a plural
    noun needs no double plural; and, when that singular/unchanged base stems to
    something shorter (a verb), that stem's -ed and -ing forms. restore -> restore,
    restores, restored, restoring; agent -> agent, agents; windows -> windows,
    window (the singular stems to itself, so no verb forms); box -> box, boxes."""
    forms = [term]
    if term.endswith("s") and len(term) > 4:
        base = term[:-1]
    else:
        base = term
        variant = term + "es" if term.endswith(("s", "x", "z", "ch", "sh")) else term + "s"
        if variant not in forms:
            forms.append(variant)
    if base not in forms:
        forms.append(base)
    s = stem(base)
    if s != base:
        for suf in ("ed", "ing"):
            w = s + suf
            if w not in forms:
                forms.append(w)
    return forms


def stem_prefixes(word: str) -> list[str]:
    """The prefixes a word's forms share: its stem, plus the -ie- form of a stem
    ending in y (copy -> copie, so "copies" is found as well as "copying")."""
    s = stem(word)
    return [s, s[:-1] + "ie"] if s.endswith("y") and len(s) >= 4 else [s]


def scope_words(scope: Scope | None) -> set[str]:
    """Every lower-cased word of the scope's vendor/product/source names: never
    distinctive, since the scope already restricts to them."""
    if scope is None:
        return set()
    return {w for name in (*scope.vendors, *scope.products, *scope.sources)
            for w in _WORD_RE.findall(name.lower())}


def question_terms(q: str, scope: Scope | None) -> list[str]:
    """The question's distinctive words: no stopwords or generic verbs, no list item
    noun, no word of the scope's vendor/product/source names (the scope already
    restricts to them)."""
    drop = set(STOPWORDS) | _ITEM_NOUN_WORDS | scope_words(scope)
    m = _ITEM_NOUN_RE.search(q)
    if m:
        drop.add(m.group(1).lower())
    out: list[str] = []
    for w in _WORD_RE.findall(q.lower()):
        if w not in drop and w not in out:
            out.append(w)
    return out


def item_kind_words(q: str) -> set[str]:
    """The lower-cased words between the leading "which/what" and the matched item
    noun of `_ITEM_NOUN_RE` (its prefix), plus the noun itself: "Which resource types
    can AWS Backup protect?" -> {"resource", "types"}; "Which workloads does Veeam
    support immutability for?" -> {"workloads"}; no "which/what ... <noun>" match ->
    set(). These name the *kind* of thing being listed, not evidence an item's facts
    must contain -- `question_terms` keeps them, since the sweep still wants them for
    recall (spec 2026-10-15 §13.1)."""
    m = _ITEM_NOUN_RE.search(q)
    if not m:
        return set()
    return set(_WORD_RE.findall(m.group(0).lower())[1:])


def grading_terms(q: str, scope: Scope | None) -> list[str]:
    """`question_terms`, minus `item_kind_words`: the words naming what kind of item
    the question lists are never grading qualifiers (spec §13.1)."""
    kind = item_kind_words(q)
    return [t for t in question_terms(q, scope) if t not in kind]


def lucene_query(terms: list[str]) -> str:
    """OR of every term's inflected exact forms (`word_forms`), plus its stem as a
    prefix wildcard when the stem is a plain word of at least 4 characters; '-' is
    escaped (Lucene syntax) and never wildcarded."""
    parts: list[str] = []
    for t in terms:
        if "-" in t:
            parts.append(t.replace("-", "\\-"))
            continue
        parts.extend(word_forms(t))
        parts.extend(f"{s}*" for s in stem_prefixes(t) if len(s) >= 4)
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
    fetched = len(rows)
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
    logger.info("list keyword sweep: terms=%s fetched=%d in_scope=%d kept=%d",
                terms, fetched, len(rows), len(kept))
    if not kept:
        return []
    resolved = await Provenance(driver).resolve_citations(list(kept))
    return [{"fact": r["fact"], "fact_uuid": u,
             "valid_at": resolved.get(u, {}).get("valid_at"),
             "invalid_at": resolved.get(u, {}).get("invalid_at"),
             "sources": resolved.get(u, {}).get("sources", [])} for u, r in kept.items()]
