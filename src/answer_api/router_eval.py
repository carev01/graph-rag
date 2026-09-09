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


def _mean(xs: Sequence[int | float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(per_question: list[dict]) -> dict:
    n = len(per_question)
    by_intent: dict[str, list[bool]] = {}
    for r in per_question:
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
        "routing_accuracy": (sum(1 for r in per_question if r["routing_hit"]) / n) if n else 0.0,
        "routing_by_intent": {k: sum(v) / len(v) for k, v in by_intent.items()},
        "grounding_precision": (sum(1 for r in grounded if r["grounding_hit"]) / len(grounded)
                                if grounded else None),
        "grounding_by_mode": {k: sum(v) / len(v) for k, v in ground_by_mode.items()},
        "faithfulness_mean": _mean([r["faithfulness"] for r in faithed]) if faithed else None,
        "faithfulness_by_mode": faith_by_mode,
        "faithfulness_unscored": faithfulness_unscored,
        "comparative": comparative,
        "drift_wins": drift_wins,
        "per_question": per_question,
    }
