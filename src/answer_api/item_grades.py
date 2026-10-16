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

No grade text is model-written: the note is fixed wording plus the question's own
words. Lines from the "Not supported / limitations" section on are never graded."""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field

from answer_api.list_sweep import _WORD_RE, STOPWORDS, is_limitation, stem, stem_prefixes
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


def _names(item: str, text: str, term_stems: set[str]) -> bool:
    """At least half (rounded up) of the item's distinctive words are present."""
    words = [w for w in _WORD_RE.findall(item.lower())
             if w not in STOPWORDS and w not in _GENERIC and stem(w) not in term_stems]
    if not words:
        return True
    return sum(_present(w, text) for w in words) >= math.ceil(len(words) / 2)


def _as_in_question(term: str, q: str) -> str:
    m = re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", q, re.IGNORECASE)
    return m.group(0) if m else term


def _grade_line(item: str, markers: list[int], marker_map: dict[int, dict], *,
                terms: list[str], term_stems: set[str], scores: dict[str, float | None],
                floor: float, unnamed_floor: float) -> ItemGrade:
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
        if not _names(item, text, term_stems):
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


def grade_items(raw: str, marker_map: dict[int, dict], *, q: str, terms: list[str],
                scores: dict[str, float | None], floor: float,
                unnamed_floor: float) -> tuple[str, list[ItemGrade]]:
    """Grade every `- **<item>** ... [N]` line before the limitations section; return
    the text with dropped lines (and headings they leave empty) removed and flagged
    lines annotated, plus one grade per item line in text order."""
    lines = raw.split("\n")
    end = next((i for i, ln in enumerate(lines) if _is_limits_heading(ln)), len(lines))
    term_stems = {stem(t) for t in terms}
    grades: list[ItemGrade] = []
    removed: set[int] = set()
    ungraded = 0
    for i in range(end):
        m = _ITEM_RE.match(lines[i])
        if m is None:
            if _BULLET_RE.match(lines[i]):
                ungraded += 1
            continue
        g = _grade_line(m.group("item").strip(), _markers(lines[i], marker_map), marker_map,
                        terms=terms, term_stems=term_stems, scores=scores, floor=floor,
                        unnamed_floor=unnamed_floor)
        grades.append(g)
        if g.grade == DROPPED:
            removed.add(i)
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
    return raw, grades


def _is_limits_heading(line: str) -> bool:
    """The "Not supported / limitations" section heading; the same phrase inside a
    sentence or a bullet does not end the graded region."""
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
    with nothing beneath it stays."""
    levels = {i: lv for i in range(end) if (lv := _level(lines[i])) is not None}

    def section(i: int) -> range:
        stop = next((j for j in range(i + 1, end) if j in levels and levels[j] <= levels[i]),
                    end)
        return range(i + 1, stop)

    gone = set(removed)
    while True:
        newly = {i for i in levels if i not in gone
                 and any(lines[j].strip() for j in section(i))
                 and all(j in gone for j in section(i) if lines[j].strip())}
        if not newly:
            return gone - removed
        gone |= newly
