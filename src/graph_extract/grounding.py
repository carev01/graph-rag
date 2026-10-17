"""Deterministic grounding gate for graphiti's entity extraction.

Traced 2026-10-17 (docs/superpowers/specs/2026-10-17-extraction-grounding-design.md):
the extraction model invented entities the chunk never mentions -- a forced subject
("AWS Backup" over a bare Securiti.ai connector list) and the instructions' own
example vocabulary ("Hyper-V", "East US", "LRS" in a short AWS Security Hub control).
Facts then attached to those inventions.

`dedup_guard`'s wrapper already intercepts every graphiti LLM call; for the
`extract_nodes.extract_text` reply it calls `filter_entities`, which keeps an
extracted entity only when the chunk text mentions it (`is_grounded`). graphiti then
extracts edges only among the kept entities, so no fact can attach to an invented one.

The chunk text and the document's product/vendor names arrive through
`CURRENT_GROUNDING`, which IngestDriver sets around each `add_text_episode`. With no
context open (probe, compat, pilot scripts), the gate is a no-op counted as unscoped.

Two more corrections ride on the same reply (final-review fix wave, 2026-10-17):
an entity typed Product that is not the documented product needs its whole name (or
a naming alias) in the text -- most of its words is not enough, because product names
share generic words ("Backup", "Azure") that a chunk mentions anyway; and an entity
named by a known abbreviation of the documented product ("VBR") is renamed to the
documented product, so one product never becomes two nodes.

Never reads a failure as absence: a reply the gate cannot parse is passed through
unchanged (and counted), never emptied. A missing or malformed alias file fails the
import rather than silently disabling the rename.
"""
from __future__ import annotations

import json
import logging
import math
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from graph_extract.ontology import ENTITY_TYPES, NAMING_ALIASES

logger = logging.getLogger(__name__)

EXTRACT_TEXT_PROMPT_NAME = "extract_nodes.extract_text"

DROPPED_NAMES_CAP = 50

_STOPWORDS = frozenset(
    {"the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with", "by"})
_TOKEN = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("es", "s", "ed", "ing")
_B = r"(?<![a-z0-9])"     # word start
_E = r"(?![a-z0-9])"      # word end
# What may follow a 3+ character token or stem and still be "the same word":
# a plural, a past or progressive form, or one doubled/added letter before
# ed/ing (snapshot -> snapshotting, gap -> gapped). Nothing else: "hyper" does not
# match "hypervisor", "nas" does not match "nasa".
_INFLECTION = r"(?:s|es|d|ed|ing|[a-z](?:ed|ing))?"

# The entity_type_id graphiti assigns the Product type. graphiti numbers the types it
# shows the extraction model as 0 = its default "Entity", then the custom types in
# the order of the `entity_types` dict starting at 1 (node_operations.
# _build_entity_types_context, 0.30.1); add_text_episode passes ENTITY_TYPES.
# Computed, not hard-coded; a tripwire test builds graphiti's own context.
PRODUCT_ENTITY_TYPE_ID = list(ENTITY_TYPES).index("Product") + 1

# answer_api's scope alias table (alias -> one target name or a list of them), read
# as DATA: graph_extract must not import answer_api code. Both packages ship in the
# same wheel/image, so the file sits beside this package.
SCOPE_ALIASES_PATH = Path(__file__).resolve().parent.parent / "answer_api" / "scope_aliases.json"


def load_product_aliases(path: Path = SCOPE_ALIASES_PATH) -> dict[str, tuple[str, ...]]:
    """Lower-cased alias -> the names it stands for.

    Accepts what answer_api's ScopeResolver accepts, so the two consumers of the
    file agree: a target is a name or a list of names; a blank name is ignored and
    an entry left with no names (an empty list) is skipped, not an error. Raises
    on a missing file, a non-object file, or a target that is neither a string nor
    a list of strings (ScopeResolver crashes on those too) -- at import, so a bad
    committed file fails CI and startup instead of silently turning the rename off.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object of alias -> target(s)")
    table: dict[str, tuple[str, ...]] = {}
    for alias, targets in raw.items():
        names = [targets] if isinstance(targets, str) else targets
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"{path}: alias {alias!r} must map to a name or a list of names")
        kept = tuple(n for n in names if n.strip())
        if kept:
            table[alias.strip().lower()] = kept
    return table


PRODUCT_ALIASES: dict[str, tuple[str, ...]] = load_product_aliases()

# Warned once per process: an extraction call reached a guarded graphiti with no
# grounding context open, i.e. the gate was bypassed (the call is counted either way).
_unscoped_warned = False


@dataclass
class GroundingStats:
    """Counters for one scope (an article, or a batch/run when merged)."""
    calls: int = 0            # extract_text replies filtered
    dropped: int = 0          # entities removed from those replies
    parse_failures: int = 0   # replies not shaped like ExtractedEntities -> passed through
    injected: int = 0         # documented-product entities supplied by the gate
    renamed: int = 0          # entities named by an alias of the documented product, folded into it
    unscoped: int = 0         # extract_text calls with no grounding context open
    dropped_names: list[str] = field(default_factory=list)   # first DROPPED_NAMES_CAP

    def _add_names(self, names: list[str]) -> None:
        room = DROPPED_NAMES_CAP - len(self.dropped_names)
        if room > 0:
            self.dropped_names.extend(names[:room])

    def merge(self, other: GroundingStats) -> None:
        self.calls += other.calls
        self.dropped += other.dropped
        self.parse_failures += other.parse_failures
        self.unscoped += other.unscoped
        self.injected += other.injected
        self.renamed += other.renamed
        self._add_names(other.dropped_names)


@dataclass
class GroundingContext:
    text: str                    # the episode body exactly as graphiti receives it
    allowed: tuple[str, ...]     # the document's product and vendor names
    stats: GroundingStats
    # The documented product ("" when unknown). Supplied as an entity when the
    # extraction reply omits it: the edge step names the implied subject anyway,
    # and graphiti discards every edge whose source entity was not extracted.
    product: str = ""


# Set by IngestDriver around each add_text_episode; read by dedup_guard's wrapper.
# A ContextVar for the same reason as CURRENT_DEDUP_STATS: articles run concurrently,
# and add_episode's inner tasks inherit the context.
CURRENT_GROUNDING: ContextVar[GroundingContext | None] = ContextVar(
    "current_grounding", default=None)


def _distinctive_tokens(name: str) -> list[str]:
    return list(dict.fromkeys(_name_tokens(name)))


def _name_tokens(name: str) -> tuple[str, ...]:
    """Every distinctive token in order (duplicates kept) -- the identity rule (b)
    compares, so "Veeam Backup and Replication" equals "Veeam Backup & Replication"."""
    return tuple(t for t in _TOKEN.findall(name.lower())
                 if len(t) >= 2 and t not in _STOPWORDS)


def _stems(token: str) -> list[str]:
    """The token with one trailing suffix removed, for every suffix that leaves at
    least 3 characters ("objectives" -> "objectiv", "objective"), plus the -y
    singular of an -ies/-ied form ("policies" -> "policy")."""
    stems = [token[: -len(suf)] for suf in _SUFFIXES
             if token.endswith(suf) and len(token) - len(suf) >= 3]
    for suf in ("ies", "ied"):
        if token.endswith(suf) and len(token) - len(suf) + 1 >= 3:
            stems.append(token[: -len(suf)] + "y")
    return stems


def _form_pattern(form: str) -> str:
    """A 3+ character token or stem with its accepted inflections, as a whole word.
    A form ending in "y" also matches its -ies/-ied form ("policy" ~ "policies")."""
    alts = re.escape(form) + _INFLECTION
    if form.endswith("y"):
        alts = f"(?:{alts}|{re.escape(form[:-1])}(?:ies|ied))"
    return _B + alts + _E


def _whole_word(phrase: str, text: str) -> bool:
    return re.search(_B + re.escape(phrase) + _E, text) is not None


# Phrase matching on normalised text: "&" reads as "and", and any run of
# punctuation/whitespace is one space, so "Veeam Backup & Replication" matches
# "Veeam Backup and Replication" and "Hyper-V" matches "Hyper V". A phrase never
# crosses a hard boundary, though: a line break, a table cell ("|"), a comma or
# semicolon, or a sentence/label end (".", ":", "!", "?" before whitespace). Those
# separate list items and cells -- "+ AWS\n+ Backup" is two connectors, not the
# product "AWS Backup".
_HARD_BOUNDARY = re.compile(r"[\r\n|,;]+|[.:!?]+(?=\s|$)")
_SOFT_SEPARATORS = re.compile(r"[\W_]+")


def _normalise(s: str) -> str:
    return _SOFT_SEPARATORS.sub(" ", s.lower().replace("&", " and ")).strip()


@lru_cache(maxsize=32)
def _normalised_text(text: str) -> str:
    """The text's segments (split at hard boundaries), each normalised, one per line."""
    return "\n".join(_normalise(seg) for seg in _HARD_BOUNDARY.split(text))


def _phrase_in(phrase: str, text: str) -> bool:
    """`phrase` occurs in `text` as a whole-word phrase: raw (the original rule (a),
    case-insensitive) or after normalising both sides."""
    lphrase = phrase.strip().lower()
    if not lphrase:
        return False
    if _whole_word(lphrase, text.lower()):
        return True
    nphrase = _normalise(phrase)
    return bool(nphrase) and re.search(
        r"(?<!\S)" + re.escape(nphrase) + r"(?!\S)", _normalised_text(text)) is not None


def _present(token: str, text: str) -> bool:
    if len(token) == 2:
        # whole word or its plural: "vm" matches "vm"/"vms", never "kvm" or "vmware"
        return re.search(_B + re.escape(token) + "s?" + _E, text) is not None
    return any(re.search(_form_pattern(form), text) is not None
               for form in [token, *_stems(token)])


def same_name(a: str, b: str) -> bool:
    """One name spelled two ways: equal ignoring case, or the same distinctive-token
    sequence ("Veeam Backup and Replication" == "Veeam Backup & Replication").
    Blank names are never the same as anything."""
    la, lb = a.strip().lower(), b.strip().lower()
    if not la or not lb:
        return False
    if la == lb:
        return True
    ta = _name_tokens(la)
    return bool(ta) and ta == _name_tokens(lb)


def note_unscoped(stats: GroundingStats) -> None:
    """Count an extract_text call made with no grounding context, and warn the
    first time it happens in this process."""
    global _unscoped_warned
    stats.unscoped += 1
    if not _unscoped_warned:
        _unscoped_warned = True
        logger.warning("grounding gate bypassed: extraction call without a grounding context")


def _single_target_aliases(
        aliases: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """Normalised target name -> the aliases that name it ALONE. A multi-target
    alias ("Veeam Agent" -> several guides) names none of its targets."""
    out: dict[str, list[str]] = {}
    for alias, targets in aliases.items():
        if len(targets) == 1:
            out.setdefault(_normalise(targets[0]), []).append(alias)
    return {k: tuple(v) for k, v in out.items()}


SINGLE_TARGET_ALIASES: dict[str, tuple[str, ...]] = _single_target_aliases(PRODUCT_ALIASES)


def is_product_alias(name: str, product: str,
                     aliases: dict[str, tuple[str, ...]] | None = None) -> bool:
    """True when `name` is, case-insensitively and exactly, an alias the table maps
    to `product` ("VBR" -> "Veeam Backup & Replication"). Initials count only when
    the table lists them; nothing is derived. The product's own name is not an alias."""
    if not product.strip() or same_name(name, product):
        return False
    targets = (PRODUCT_ALIASES if aliases is None else aliases).get(name.strip().lower(), ())
    return any(same_name(t, product) for t in targets)


def is_grounded(name: str, text: str, allowed: tuple[str, ...], *,
                product_typed: bool = False,
                scope_aliases: dict[str, tuple[str, ...]] | None = None) -> bool:
    """True when any of these holds:

    (a) the whole name occurs in the text as a whole-word phrase, case-insensitive,
        compared raw or normalised ("&" = "and", punctuation runs = one space, never
        across a line, cell, comma or sentence end -- see `_phrase_in`);
    (b) the name is the document's product or vendor (the implied subject),
        compared by distinctive tokens so "and" and "&" spell the same name;
    (c) at least ceil(2n/3) of its n distinctive words are present as whole words,
        a simple inflection counting ("air gap" ~ "air-gapped", "VM" ~ "VMs");
    (d) one of its naming aliases (ontology.NAMING_ALIASES) occurs as a whole-word
        phrase (as in (a)) -- the instructions tell the model to write "Kubernetes"
        for K8s, so a chunk saying only K8s still grounds "Kubernetes".

    `product_typed` (the entity was typed Product): rule (c) does not apply, so a
    product other than the documented one needs (a) or (d). Product names are
    built from generic words -- "AWS Backup" and "Google Cloud Backup" both ride on
    a connector list naming AWS RDS, Google Cloud Storage and a stray "Backup".
    In exchange, a Product-typed name is also grounded by a scope alias that names
    it ALONE (answer_api's scope_aliases.json, single-target entries only) found as
    in (a): "VBR" grounds "Veeam Backup & Replication". `scope_aliases` overrides
    the table (lower-cased alias -> targets, as `load_product_aliases` returns).
    """
    lname = name.strip().lower()
    ltext = text.lower()
    if lname and _phrase_in(lname, text):
        return True
    if any(same_name(name, a) for a in allowed):
        return True
    if any(_phrase_in(alias, text) for alias in NAMING_ALIASES.get(lname, ())):
        return True
    if product_typed:
        single = (SINGLE_TARGET_ALIASES if scope_aliases is None
                  else _single_target_aliases(scope_aliases))
        return any(_phrase_in(alias, text) for alias in single.get(_normalise(name), ()))
    toks = _distinctive_tokens(lname)
    if not toks:
        return False
    present = sum(1 for t in toks if _present(t, ltext))
    return present >= math.ceil(2 * len(toks) / 3)


def filter_entities(response: Any, ctx: GroundingContext) -> Any:
    """`response` with `extracted_entities` reduced to the grounded ones, an alias
    of the documented product renamed to it, and the documented product supplied
    when absent. A reply not shaped like ExtractedEntities is returned unchanged
    and counted; graphiti's own `ExtractedEntities(**response)` then decides what
    it is worth."""
    stats = ctx.stats
    entities = response.get("extracted_entities") if isinstance(response, dict) else None
    if not isinstance(entities, list):
        stats.parse_failures += 1
        logger.warning(
            "grounding: extract_text reply is not {'extracted_entities': [...]} (%s); "
            "passed through unfiltered", type(response).__name__)
        return response
    stats.calls += 1
    kept: list[Any] = []
    dropped: list[str] = []
    for ent in entities:
        name = ent.get("name") if isinstance(ent, dict) else None
        # An item without a (non-blank) string name is not ours to judge: keep it;
        # graphiti's validation and its own empty-name filter decide.
        if (not isinstance(name, str) or not name.strip()
                or is_grounded(name, ctx.text, ctx.allowed,
                               product_typed=ent.get("entity_type_id") == PRODUCT_ENTITY_TYPE_ID)):
            kept.append(ent)
        else:
            dropped.append(name)
    renamed = 0
    if ctx.product.strip():
        # "VBR" in a Veeam Backup & Replication chunk is the documented product, not
        # a second product: rename it (only ever INTO ctx.product). When the product
        # is already in the reply, the alias entity is folded away instead of
        # becoming a same-named duplicate.
        has_product = any(isinstance(ent, dict) and isinstance(ent.get("name"), str)
                          and same_name(ent["name"], ctx.product) for ent in kept)
        folded: list[Any] = []
        for ent in kept:
            if (isinstance(ent, dict) and isinstance(ent.get("name"), str)
                    and is_product_alias(ent["name"], ctx.product)):
                renamed += 1
                if has_product:
                    continue
                ent = {**ent, "name": ctx.product, "entity_type_id": PRODUCT_ENTITY_TYPE_ID}
                has_product = True
            folded.append(ent)
        kept = folded
        stats.renamed += renamed
    injected = bool(ctx.product.strip()) and not any(
        isinstance(ent, dict) and isinstance(ent.get("name"), str)
        and same_name(ent["name"], ctx.product) for ent in kept)
    if injected:
        # The model left out the implied subject; the edge step will use it as the
        # source of the chunk's facts regardless (the 2026-10-17 smoke lost ~28
        # facts to "Source entity not found in nodes"). Supply it deterministically.
        kept.append({"name": ctx.product, "entity_type_id": PRODUCT_ENTITY_TYPE_ID})
        stats.injected += 1
    if not dropped and not injected and not renamed:
        return response
    if dropped:
        stats.dropped += len(dropped)
        stats._add_names(dropped)
        logger.debug("grounding dropped %d of %d entities: %s",
                     len(dropped), len(entities), dropped)
    return {**response, "extracted_entities": kept}
