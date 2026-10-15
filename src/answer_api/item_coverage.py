"""Deterministic item matching for list-question ground truth (spec 2026-10-15 §7)."""
from __future__ import annotations

import re


def match_items(text: str, items: list[dict]) -> tuple[list[str], list[str]]:
    low = (text or "").lower()
    found: list[str] = []
    missing: list[str] = []
    for it in items:
        hit = any(re.search(rf"(?<![a-z0-9]){re.escape(a.lower())}(?![a-z0-9])", low)
                  for a in it["aliases"])
        (found if hit else missing).append(it["item"])
    return found, missing
