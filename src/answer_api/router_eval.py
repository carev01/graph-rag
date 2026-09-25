"""Golden-set scoring for the /answer router: routing accuracy, citation grounding
(reuses golden.precision_at_k), and LLM-judge faithfulness. Pure aggregation only;
the @live harness that runs the router + judge lives in eval_router.py."""
from __future__ import annotations

import re
from collections.abc import Sequence


def routing_hit(chosen: str, expected_modes: list[str]) -> bool:
    return chosen in expected_modes


def _parse_judge_score(raw: str) -> int:
    m = re.search(r"-?\d+", raw or "")
    if m is None:
        return 0
    return max(0, min(5, int(m.group())))


def _parse_attribution(text: str | None) -> int | None:
    """The first non-negative integer in the judge's reply, or None -- never a
    coerced 0 (memory: "LLM empty reply coerced to a value"). Callers that need
    to distinguish "judge said 0" from "unusable reply" must do so before
    calling this (see `judge_attribution`'s `_attempt`), not here."""
    if not text:
        return None
    m = re.search(r"\d+", text)
    return int(m.group()) if m else None


def _mean(xs: Sequence[int | float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


_MARKER_RE = re.compile(r"\[\d+\]")
# A sentence ends at . ! ? followed by whitespace, or at a line break (headings
# and bullet lines are their own segments).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_LEADING_MARKERS_RE = re.compile(r"^(?:\[\d+\]\s*)+")


def markers_per_sentence(text: str) -> list[int]:
    """Marker count of every sentence that carries at least one [N] marker, in
    order. BACKLOG 0b: 19 markers pasted onto one sentence and one marker per
    claim were indistinguishable in every measure the eval had (`cited`,
    `ranges`, grounding, the judge); this is the distribution that tells them
    apart. Markers written AFTER the full stop ("Claim. [1] [2] Next...") are
    folded into the sentence before them -- the model put the stop first,
    nothing more."""
    counts: list[int] = []
    for seg in _SENTENCE_SPLIT_RE.split(text or ""):
        seg = seg.strip()
        if not seg:
            continue
        lead = _LEADING_MARKERS_RE.match(seg)
        if lead and counts:
            counts[-1] += len(_MARKER_RE.findall(lead.group(0)))
            seg = seg[lead.end():]
        counts.append(len(_MARKER_RE.findall(seg)))
    return [c for c in counts if c > 0]


BAG_MARKERS = 8
"""A sentence carrying this many markers or more is counted as bag-pasting.
Set from the 0b measurement: five of ten global-mode answers carried a sentence
of 16, 9, 9, 10 and 19 markers, and no answer cited between 7 and 9 -- the
distribution is bimodal, and 8 sits in the gap."""


def bag_share(text: str) -> float | None:
    """Share of an answer's citation MASS that sits in >=`BAG_MARKERS`-marker
    sentences, or None when the answer cites nothing.

    `markers_per_sentence` says a bag exists; it cannot say how much of the
    answer is bag. One 19-marker sentence beside thirty well-cited claims and an
    answer that is one 19-marker sentence share the same `mps_max`. Weighting by
    markers rather than by sentences separates them.

    None, not 0.0: an uncited answer or a refusal has no citation mass, and 0.0
    is the BEST score on this scale -- reporting it for an answer that cited
    nothing would read as perfect citation hygiene."""
    counts = markers_per_sentence(text)
    total = sum(counts)
    if total == 0:
        return None
    return sum(c for c in counts if c >= BAG_MARKERS) / total


def aggregate(per_question: list[dict]) -> dict:
    n = len(per_question)
    # A failed question (e.g. an APIConnectionError) is not a routing miss --
    # it never got a chance to route. `ran` mirrors the `grounded`/`faithed`
    # filters below: exclude what could not be measured rather than scoring it
    # as a failure.
    ran = [r for r in per_question if not r.get("failed")]
    by_intent: dict[str, list[bool]] = {}
    for r in ran:
        by_intent.setdefault(r["intent"], []).append(bool(r["routing_hit"]))
    grounded = [r for r in per_question if r["grounding_hit"] is not None]
    ground_by_mode: dict[str, list[bool]] = {}
    for r in grounded:
        ground_by_mode.setdefault(r["chosen"], []).append(bool(r["grounding_hit"]))

    # Faithfulness can be None -- the judge could not be measured (e.g. the
    # reasoning model burned its whole budget and returned no content). An
    # unmeasured answer must never silently average in as a 0; it is excluded
    # from every mean below and its count is surfaced via faithfulness_unscored.
    faithed = [r for r in per_question if r["faithfulness"] is not None]
    faithfulness_unscored = n - len(faithed)
    faith_by_mode_scored: dict[str, list[int]] = {}
    faith_by_mode_seen: list[str] = []
    for r in per_question:
        if r["chosen"] not in faith_by_mode_seen:
            faith_by_mode_seen.append(r["chosen"])
        if r["faithfulness"] is not None:
            faith_by_mode_scored.setdefault(r["chosen"], []).append(r["faithfulness"])
    faith_by_mode = {m: (_mean(faith_by_mode_scored[m]) if m in faith_by_mode_scored else None)
                      for m in faith_by_mode_seen}

    # Markers per sentence, by mode: mean of the per-question means, and the
    # single worst sentence seen. Records written before the metric existed
    # carry no mps keys and are skipped, not scored as 0.
    mps_by_mode: dict[str, dict] = {}
    mps_rows: dict[str, list[dict]] = {}
    for r in per_question:
        if r.get("mps_mean") is not None:
            mps_rows.setdefault(r["chosen"], []).append(r)
    for mode, rows in mps_rows.items():
        mps_by_mode[mode] = {"mean": _mean([r["mps_mean"] for r in rows]),
                             "max": max(r["mps_max"] for r in rows)}

    # Bag share, by mode: mean of the per-question shares. A record with no key
    # (written before the metric existed) and a None share (an answer that cited
    # nothing) are both skipped -- neither is a 0.0, which is the best score.
    bag_rows: dict[str, list[float]] = {}
    for r in per_question:
        if r.get("bag_share") is not None:
            bag_rows.setdefault(r["chosen"], []).append(r["bag_share"])
    bag_share_by_mode = {m: _mean(v) for m, v in bag_rows.items()}

    # Misattributed claims (spec §6): an answer that attributes a claim to a
    # vendor/product not among its cited facts' labels. `.get`, not `[...]` --
    # records written before this metric existed carry no key at all, and those
    # must read as unscored rather than raise KeyError, same as bag_share.
    misattributed_seen = [r.get("misattributed") for r in per_question]
    misattributed_scored = [x for x in misattributed_seen if x is not None]
    misattributed_unscored = sum(1 for x in misattributed_seen if x is None)
    misattributed_total = sum(misattributed_scored)
    misattributed_answers = sum(1 for x in misattributed_scored if x > 0)

    comp_rows = [r for r in per_question if "comparative" in r]
    comparative: dict | None = None
    drift_wins: bool | None = None
    if comp_rows:
        comparative = {}
        for m in ("local", "global", "drift"):
            faiths = [r["comparative"][m]["faithfulness"] for r in comp_rows
                      if r["comparative"][m]["faithfulness"] is not None]
            gh = [r["comparative"][m]["grounding_hit"] for r in comp_rows
                  if r["comparative"][m]["grounding_hit"] is not None]
            comparative[m] = {
                "faithfulness": _mean(faiths) if faiths else None,
                "grounding": (sum(1 for x in gh if x) / len(gh)) if gh else None,
            }
        d, lo, gl = comparative["drift"], comparative["local"], comparative["global"]
        if d["faithfulness"] is None or lo["faithfulness"] is None or gl["faithfulness"] is None:
            drift_wins = None
        else:
            grounding_ok = (d["grounding"] is None
                            or ((lo["grounding"] is None or d["grounding"] >= lo["grounding"])
                                and (gl["grounding"] is None or d["grounding"] >= gl["grounding"])))
            drift_wins = (d["faithfulness"] >= lo["faithfulness"]
                          and d["faithfulness"] >= gl["faithfulness"] and grounding_ok)

    return {
        "n": n,
        "questions_failed": n - len(ran),
        "routing_accuracy": (sum(1 for r in ran if r["routing_hit"]) / len(ran)) if ran else 0.0,
        "routing_by_intent": {k: sum(v) / len(v) for k, v in by_intent.items()},
        "grounding_precision": (sum(1 for r in grounded if r["grounding_hit"]) / len(grounded)
                                if grounded else None),
        "grounding_by_mode": {k: sum(v) / len(v) for k, v in ground_by_mode.items()},
        "faithfulness_mean": _mean([r["faithfulness"] for r in faithed]) if faithed else None,
        "faithfulness_by_mode": faith_by_mode,
        "faithfulness_unscored": faithfulness_unscored,
        "mps_by_mode": mps_by_mode,
        "bag_share_by_mode": bag_share_by_mode,
        "misattributed_total": misattributed_total,
        "misattributed_answers": misattributed_answers,
        "misattributed_unscored": misattributed_unscored,
        "comparative": comparative,
        "drift_wins": drift_wins,
        "per_question": per_question,
    }
