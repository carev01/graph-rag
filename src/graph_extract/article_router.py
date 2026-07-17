"""Deterministic per-article tier routing for hybrid extraction.

`is_dense_matrix` decides whether an article is a dense availability/support
matrix that the cheap model (ling-2.6-flash) cannot extract reliably (it
over-generates on such tables -> 16k truncation -> lost chunks). Dense articles
route to the strong model; everything else to the cheap one. No I/O, no LLM.
"""
from __future__ import annotations


def is_dense_matrix(markdown: str, *, ratio_threshold: float = 0.25,
                    pipe_threshold: int = 200) -> bool:
    """True if `markdown` is a dense table the cheap tier can't handle.

    A line is a table row when its left-stripped form starts with '|'. Routes
    to the strong tier when the table-line ratio >= ratio_threshold OR the total
    '|' count >= pipe_threshold. Calibration (2026-07): dense articles measured
    33-35% table-line ratio / 264-1415 pipes; clean articles <=17% / <=80.
    """
    lines = markdown.splitlines()
    if not lines:
        return False
    table_lines = sum(1 for ln in lines if ln.lstrip().startswith("|"))
    ratio = table_lines / len(lines)
    pipes = markdown.count("|")
    return ratio >= ratio_threshold or pipes >= pipe_threshold
