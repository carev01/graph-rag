"""Per-item evidence grades for list answers (spec 2026-10-15 §12.2).

Every item line of a list answer is graded from the facts it cites -- the fact text
plus the source article titles -- with no model involved:

- confirmed: a cited fact names the item, its rerank score is at least `floor` (or
  unscored after a degraded rerank), and it contains every question term;
- needs validation: as confirmed but some question terms are missing (reason
  "missing_terms"); the line is kept with a fixed note naming them, spelled as in the
  question. Or: no cited fact names the item, but a non-limitation cited fact scores
  at least `unnamed_floor` (reason "unnamed"); the line is kept with a fixed note
  saying the source does not name the item. Named support always wins when both would
  apply;
- dropped: neither of the above -- no cited fact at all (reason "no_marker"), or some
  exist but none clears its floor (reason "below_floor"; an unscored fact never
  clears the unnamed floor).

A dropped line takes its deeper-indented children (sub-bullets, wrapped lines) with
it, and each drop is logged with its reason. Words of the scope's vendor/product/
source names never count towards naming an item. If every item is dropped, the
caller treats the answer as ungrounded (list_mode.answer_list refuses).

No grade text is model-written: the note is fixed wording plus the question's own
words. Lines from the "Not supported / limitations" section on are never graded."""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import NamedTuple

from answer_api.list_sweep import (
    _WORD_RE,
    STOPWORDS,
    is_limitation,
    scope_words,
    stem,
    stem_prefixes,
)
from answer_api.scope import Scope
from answer_api.synthesize import _MARKER_LIST_RE

logger = logging.getLogger(__name__)

CONFIRMED = "confirmed"
NEEDS_VALIDATION = "needs_validation"
DROPPED = "dropped"

# A bulleted ("-", "*") or numbered ("1.", "1)") line.
_BULLET = r"(?:[-*]|\d+[.)])"
_ITEM_RE = re.compile(r"^(\s*" + _BULLET + r"\s+)\*\*(?P<item>[^*]+)\*\*(?P<rest>.*)$")
_BULLET_RE = re.compile(r"^\s*" + _BULLET + r"\s+")
_LIMITS_RE = re.compile(r"not supported\s*/\s*limitations", re.IGNORECASE)
# The phrase alone on its line: optional "#"s, optional "**" around it, optional colon.
_LIMITS_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?\s*not supported\s*/\s*limitations\s*:?\s*(?:\*\*)?"
    r"\s*:?\s*$", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s*(#{1,6}\s+\S.*|\*\*[^*]+\*\*:?)\s*$")
# A bold-line heading ("**Clouds**") nests under every "#" heading.
_BOLD_LEVEL = 7
# By stem; a term in a group is present when any member is.
_SYNONYMS = ({"restor", "recover"}, {"protect", "backup"}, {"vms", "vm", "virtual machine"})
# Words that describe the kind of thing listed, never which item it is.
_GENERIC = frozenset({"vm", "vms", "machine", "machines", "instance", "instances",
                      "environment", "backup", "backups", "restore", "recovery"})
_NOTE = " *(needs validation: the cited source does not mention {terms})*"
_NOTE_UNNAMED = " *(needs validation: the cited source does not name this item)*"


@dataclass
class ItemGrade:
    item: str
    grade: str
    missing_terms: list[str] = field(default_factory=list)
    markers: list[int] = field(default_factory=list)
    reason: str = ""

    def as_dict(self, marker_map: dict[int, dict]) -> dict:
        return {"item": self.item, "grade": self.grade,
                "missing_terms": list(self.missing_terms), "markers": list(self.markers),
                "reason": self.reason,
                "fact_uuids": [marker_map[m]["fact_uuid"] for m in self.markers
                               if m in marker_map]}


def _hit(needle: str, text: str) -> bool:
    """`needle` as a word prefix; a needle shorter than 4 characters must be the
    whole word, optionally plural ("vm" matches "VM" and "VMs", never "VMware")."""
    tail = r"s?(?![a-z0-9])" if len(needle) < 4 else ""
    return re.search(r"(?<![a-z0-9])" + re.escape(needle) + tail, text) is not None


def _present(term: str, text: str) -> bool:
    """The term's stem as a word prefix in `text`, or any member of its synonym group."""
    if any(_hit(p, text) for p in stem_prefixes(term)):
        return True
    s = stem(term)
    return any(_hit(member, text) for group in _SYNONYMS
               if s in group or term in group for member in group)


def _evidence(fact: dict) -> str:
    titles = [s.get("title") or "" for s in fact.get("sources") or ()]
    return " ".join([fact.get("fact") or "", *titles]).lower()


def _markers(line: str, marker_map: dict[int, dict]) -> list[int]:
    out: list[int] = []
    for m in _MARKER_LIST_RE.finditer(line):
        for n in map(int, re.findall(r"\d+", m.group(0))):
            if n in marker_map and n not in out:
                out.append(n)
    return out


def _item_words(item: str, term_stems: set[str], excluded: set[str]) -> list[str]:
    """The item's distinctive words: no stopwords, no generic kind-of-thing words, no
    question term and no word of the scope's names (`excluded`) -- "Veeam" says
    nothing about which Veeam item a fact names."""
    return [w for w in _WORD_RE.findall(item.lower())
            if w not in STOPWORDS and w not in _GENERIC and w not in excluded
            and stem(w) not in term_stems]


def _names(item: str, text: str, term_stems: set[str], excluded: set[str]) -> bool:
    """At least half (rounded up) of the item's distinctive words are present. An item
    made only of excluded words ("Veeam Backup" under the product scope "Veeam Backup &
    Replication") falls back to its words minus stopwords, so it still has to be
    named."""
    words = (_item_words(item, term_stems, excluded)
             or [w for w in _WORD_RE.findall(item.lower()) if w not in STOPWORDS])
    if not words:
        return True
    return sum(_present(w, text) for w in words) >= math.ceil(len(words) / 2)


def _as_in_question(term: str, q: str) -> str:
    m = re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", q, re.IGNORECASE)
    return m.group(0) if m else term


def _grade_line(item: str, markers: list[int], marker_map: dict[int, dict], *,
                terms: list[str], term_stems: set[str], excluded: set[str],
                scores: dict[str, float | None], floor: float,
                unnamed_floor: float) -> ItemGrade:
    candidates: list[dict] = []        # every non-limitation cited fact, named or not
    first_missing: list[str] | None = None
    for m in markers:
        fact = marker_map[m]
        # A limitation ("X is not supported for ...") never supports an item.
        if is_limitation(fact.get("fact") or ""):
            continue
        candidates.append(fact)
        score = scores.get(fact["fact_uuid"])
        if score is not None and score < floor:
            continue
        text = _evidence(fact)
        if not _names(item, text, term_stems, excluded):
            continue
        missing = [t for t in terms if not _present(t, text)]
        if not missing:
            return ItemGrade(item, CONFIRMED, [], markers, "named")
        if first_missing is None:
            first_missing = missing
    if first_missing is not None:
        return ItemGrade(item, NEEDS_VALIDATION, first_missing, markers, "missing_terms")
    if not candidates:
        return ItemGrade(item, DROPPED, [], markers, "no_marker")
    # No fact names the item at or above the floor: an unscored or low-scoring fact
    # still flags the item (never drops it) once it clears the unnamed floor.
    for fact in candidates:
        score = scores.get(fact["fact_uuid"])
        if score is not None and score >= unnamed_floor:
            return ItemGrade(item, NEEDS_VALIDATION, [], markers, "unnamed")
    return ItemGrade(item, DROPPED, [], markers, "below_floor")


class GradedAnswer(NamedTuple):
    text: str                  # dropped lines removed, flagged lines annotated
    grades: list[ItemGrade]    # one per item line, in text order
    ungraded: int              # bullet lines in the graded region not in item form


def _indent(line: str) -> int:
    return len(line.expandtabs(4)) - len(line.expandtabs(4).lstrip())


def _is_bullet_heading(line: str) -> bool:
    """A group heading written as a bullet: `- **Hypervisors**` or `- **Clouds:**` /
    `- **Clouds**:` -- item form with nothing after the bold name but an optional
    colon, so no description and no marker. Never graded or counted; removed only
    when the items nested under it are all dropped (`_emptied_headings`)."""
    m = _ITEM_RE.match(line)
    return m is not None and m.group("rest").strip() in ("", ":")


def _children(lines: list[str], i: int, end: int) -> list[int]:
    """The lines that belong to item line `i`: following lines indented deeper than
    it (sub-bullets, wrapped continuation lines), across a blank line only when the
    next non-blank line is still deeper; stops at a line at the same or a lower
    indent, a heading, an item-form line (a nested item is graded on its own), or
    the end of the graded region."""
    base = _indent(lines[i])

    def stops(ln: str) -> bool:
        return (_indent(ln) <= base or _level(ln) is not None
                or _ITEM_RE.match(ln) is not None)

    out: list[int] = []
    j = i + 1
    while j < end:
        ln = lines[j]
        if not ln.strip():
            k = next((k for k in range(j + 1, end) if lines[k].strip()), None)
            if k is None or stops(lines[k]):
                break
            out.append(j)
        elif stops(ln):
            break
        else:
            out.append(j)
        j += 1
    return out


def grade_items(raw: str, marker_map: dict[int, dict], *, q: str, terms: list[str],
                scores: dict[str, float | None], floor: float, unnamed_floor: float,
                scope: Scope | None = None) -> GradedAnswer:
    """Grade every `- **<item>** ... [N]` line before the limitations section; return
    the text with dropped lines -- their deeper-indented children, and the headings
    they leave empty -- removed and flagged lines annotated, one grade per graded
    item line in text order, and the count of bullet lines not in item form. Words
    of the `scope`'s names never count towards naming an item."""
    lines = raw.split("\n")
    end = next((i for i, ln in enumerate(lines) if _is_limits_heading(ln)), len(lines))
    term_stems = {stem(t) for t in terms}
    excluded = scope_words(scope)
    grades: list[ItemGrade] = []
    removed: set[int] = set()
    ungraded = 0
    for i in range(end):
        if i in removed:               # a child of a dropped item goes with it
            continue
        m = _ITEM_RE.match(lines[i])
        if m is None:
            if _BULLET_RE.match(lines[i]):
                ungraded += 1
            continue
        if _is_bullet_heading(lines[i]):
            continue
        g = _grade_line(m.group("item").strip(), _markers(lines[i], marker_map), marker_map,
                        terms=terms, term_stems=term_stems, excluded=excluded, scores=scores,
                        floor=floor, unnamed_floor=unnamed_floor)
        grades.append(g)
        if g.grade == DROPPED:
            children = _children(lines, i, end)
            removed.add(i)
            removed.update(children)
            logger.info("list item dropped (%s): %s%s", g.reason, g.item[:80],
                        f" (+{len(children)} indented line(s))" if children else "")
        elif g.grade == NEEDS_VALIDATION:
            if g.reason == "unnamed":
                lines[i] = lines[i].rstrip() + _NOTE_UNNAMED
            else:
                shown = ", ".join(_as_in_question(t, q) for t in g.missing_terms)
                lines[i] = lines[i].rstrip() + _NOTE.format(terms=shown)
    if ungraded:
        logger.warning("list answer: %d bullet line(s) not in item form; left ungraded",
                       ungraded)
    if removed:
        removed |= _emptied_headings(lines, end, removed)
        kept: list[str] = []
        for i, ln in enumerate(lines):
            if i in removed:
                continue
            # Collapse the blank runs a removal leaves behind (graded region only).
            if i < end and not ln.strip() and kept and not kept[-1].strip():
                continue
            kept.append(ln)
        raw = "\n".join(kept).strip("\n")
    else:
        raw = "\n".join(lines)
    logger.info("list items: %d confirmed, %d needs validation, %d dropped",
                sum(g.grade == CONFIRMED for g in grades),
                sum(g.grade == NEEDS_VALIDATION for g in grades),
                sum(g.grade == DROPPED for g in grades))
    return GradedAnswer(raw, grades, ungraded)


def _is_limits_heading(line: str) -> bool:
    """The "Not supported / limitations" section heading; the same phrase inside a
    sentence or a bullet does not end the graded region."""
    if _LIMITS_LINE_RE.match(line):     # "Not supported / limitations:" on its own
        return True
    if _is_bullet_heading(line):        # "- **Not supported / limitations**"
        return bool(_LIMITS_RE.search(line))
    shaped = bool(_HEADING_RE.match(line)) or line.lstrip().startswith(("#", "**"))
    return shaped and bool(_LIMITS_RE.search(line))


def _level(line: str) -> int | None:
    if not _HEADING_RE.match(line):
        return None
    hashes = len(line.lstrip()) - len(line.lstrip().lstrip("#"))
    return hashes if hashes else _BOLD_LEVEL


def _emptied_headings(lines: list[str], end: int, removed: set[int]) -> set[int]:
    """Headings in the graded region whose section -- every line up to the next
    heading of the same or a higher level, the limitations line or the end, deeper
    headings included -- had content and has none once the dropped lines (and the
    headings they empty, repeated until stable) are gone. A heading the model wrote
    with nothing beneath it stays.

    A bulleted group heading (`- **Clouds**`) sits one level deeper than the
    enclosing heading; its section also ends at the first non-blank line indented
    no deeper than it (the items nested under it are the indented ones)."""
    levels: dict[int, int] = {}
    bulleted: set[int] = set()
    enclosing = 0
    for i in range(end):
        if (lv := _level(lines[i])) is not None:
            levels[i] = enclosing = lv
        elif _is_bullet_heading(lines[i]):
            levels[i] = enclosing + 1
            bulleted.add(i)

    def section(i: int) -> range:
        def ends(j: int) -> bool:
            if j in levels and j not in bulleted and levels[j] <= levels[i]:
                return True
            if i in bulleted:
                return bool(lines[j].strip()) and _indent(lines[j]) <= _indent(lines[i])
            return False
        stop = next((j for j in range(i + 1, end) if ends(j)), end)
        return range(i + 1, stop)

    gone = set(removed)
    while True:
        newly = {i for i in levels if i not in gone
                 and any(lines[j].strip() for j in section(i))
                 and all(j in gone for j in section(i) if lines[j].strip())}
        if not newly:
            return gone - removed
        gone |= newly
