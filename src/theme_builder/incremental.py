"""Incremental community refresh: decide which communities actually changed since
the last theme-build (via created_at timestamps) and match fresh Leiden communities
to persisted ones (Jaccard) so their stable ids carry across refreshes and only
new/changed communities are LLM-regenerated."""
from __future__ import annotations

from dataclasses import dataclass

from neo4j import AsyncDriver

from theme_builder.detect import Community


@dataclass
class PersistedCommunity:
    community_id: str
    level: int
    members: set[str]
    title: str
    summary: str
    full_report: str
    rating: float
    rating_explanation: str
    tags: list[str]
    cited_fact_uuids: list[str]
    embedding: list[float]
    generated_at: object          # neo4j DateTime; passed back unchanged on reuse


def _jaccard(a: set[str], b: set[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def match_communities(fresh: list[Community], persisted: list[PersistedCommunity],
                      *, tau: float) -> dict[int, "PersistedCommunity | None"]:
    """Greedy 1:1 Jaccard match of fresh->persisted within the same level. Each
    fresh community maps to its best-overlapping persisted community (>= tau) or
    None (new); a persisted community is claimed by at most one fresh."""
    cands: list[tuple[float, int, int]] = []
    for fi, fc in enumerate(fresh):
        fmembers = set(fc.member_uuids)
        for pi, pc in enumerate(persisted):
            if pc.level != fc.level:
                continue
            j = _jaccard(fmembers, pc.members)
            if j >= tau:
                cands.append((j, fi, pi))
    cands.sort(key=lambda t: t[0], reverse=True)
    out: dict[int, "PersistedCommunity | None"] = {i: None for i in range(len(fresh))}
    used_fresh: set[int] = set()
    used_persisted: set[int] = set()
    for _j, fi, pi in cands:
        if fi in used_fresh or pi in used_persisted:
            continue
        out[fi] = persisted[pi]
        used_fresh.add(fi)
        used_persisted.add(pi)
    return out


def classify(fresh: list[Community], matches: dict[int, "PersistedCommunity | None"],
             touched: set[str] | None) -> tuple[list[int], list[int]]:
    """Split fresh community indexes into (dirty, clean). touched=None -> all dirty
    (cold start). A matched community is clean iff none of its members were touched."""
    dirty: list[int] = []
    clean: list[int] = []
    for i, fc in enumerate(fresh):
        if touched is None or matches[i] is None:
            dirty.append(i)
        elif set(fc.member_uuids) & touched:
            dirty.append(i)
        else:
            clean.append(i)
    return dirty, clean
