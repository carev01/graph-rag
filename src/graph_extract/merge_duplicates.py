"""Exact-name duplicate entities: the read-only planner.

Why this exists (measured, `docs/superpowers/specs/2026-09-14-concurrency-
duplicate-mitigation-design.md` §1): re-ingesting the 83 pilot articles at
article concurrency 4 produced 1027 entities against the 999-entity
sequential baseline. Every one of the 30 excess nodes was an **exact-name
collision** (997 distinct names over 1027 nodes; 999/999 sequentially), and
they concentrated on hub entities (`AWS Backup` x3, `Amazon EC2` x3,
`Azure Backup` x3, `Microsoft Azure`, `Amazon RDS`, `Azure Files`) -- the
nodes cross-vendor questions resolve through, which design invariant #4 (one
corpus-wide `group_id`) exists to keep whole. Because every observed
duplicate was byte-identical in name, repair needs no LLM and no semantic
judgement: two `:Entity` nodes form a group iff `a.name = b.name` (Cypher
string equality, no `toLower`, no `trim`) and `a.group_id = b.group_id`.

Why it cannot wait (spec §1.2, the escalation tax): graphiti's node
resolution tries an exact normalized-name match against its candidate set
first; with one candidate it resolves for free, with more than one of the
same normalized name it escalates to the LLM ("Ambiguous: multiple candidates
share the same normalized name") and lands on whichever copy the model
picks. A duplicate is therefore a permanent per-mention LLM tax on the
hottest names in the corpus and a source of further fragmentation.

Why case variants are NOT merged (spec §4.1): the sequential baseline -- the
reference graph every future comparison is made against -- itself holds 6
case-insensitive pairs graphiti's own pipeline left apart (`Access Control`/
`Access control`, `AWS BackInt`/`AWS Backint`, `point-in-time recovery`/
`Point-in-Time Recovery`, `Multi-user authorization`/`Multi-User
Authorization`, `Redundancy`/`redundancy`, `Backup Reader`/`Backup reader`).
They are not concurrency artefacts, and a pass whose purpose is to repair
concurrency artefacts must not alter the reference graph. They are surfaced
under `near_duplicates_not_merged` for the operator and never merged.

Survivor rule (spec §4.3): the member with the earliest `created_at`, ties
broken by ascending `uuid`. That is a total order, so the survivor is a pure
function of the group, and it is the node sequential ingestion would have
produced (the first extraction creates it; every later one resolves onto it).

This module's planner **writes nothing**. Every query runs through a
READ-routed `execute_query`, so even a stray write clause would be refused
by the server rather than silently applied. The destructive half
(`apply_merges`) lives alongside it and consumes `plan_merges` verbatim.
"""

from __future__ import annotations

import math
from typing import Any

from neo4j import AsyncDriver, RoutingControl

# Excess = sum over duplicated names of (copies - 1): the number of nodes a
# merge pass would remove, NOT the number of duplicated names. Tasks 6/7 read
# this as "how many duplicates are there" and Task 7 gates on it.
_COUNT = (
    "MATCH (e:Entity {group_id:$group_id}) "
    "WITH e.name AS name, count(*) AS c WHERE c > 1 "
    "RETURN coalesce(sum(c - 1), 0) AS excess"
)

# One row per duplicated name with every member's identity, ordering key,
# labels (node labels AND graphiti's mirroring `labels` property), summary,
# name embedding and degree. `created_at` is a zoned datetime written by
# graphiti; the epoch pair is the exact ordering key (no string comparison of
# offsets), the string form is what the report carries.
#
# The ORDER BY before `collect(e)` is deliberate and deliberately BACKWARDS:
# members arrive latest-first, largest-uuid-first -- the exact opposite of the
# survivor rule. `plan_merges` sorts in Python and that sort is the only
# authority on the survivor.
#
# Why pin the order at all: an unordered `collect()` returns rows in scan
# order, and scan order is unspecified -- Cypher promises nothing about it,
# and it is free to change with the plan, the store, or the Neo4j version.
# That alone is the reason for the clause; it does not rest on any claim
# about which order the scan produces today.
#
# Why adverse rather than favourable: if the collected order matched the
# survivor rule, a planner that stopped sorting, or dropped the uuid
# tie-break, would still pick the right survivor by luck and no test could
# tell. Latest-first, largest-uuid-first means the collected order is wrong
# for BOTH keys, so only the Python sort can be right.
#
# What was observed (not guaranteed), on neo4j:2026.07.1-community: the plan
# is Sort -> EagerAggregation and the collected order followed the sort in
# every configuration tried, including a 200-member shuffled group. On the
# live graph the query plans as a NodeIndexSeek on graphiti's
# `entity_group_id` index -- `group_id` equality is its only predicate -- and
# the `entity_uuid` index is not used by it; the unordered variant returned
# insertion order there, not uuid order. A different version or plan may
# behave differently, which is exactly why the order is pinned and a test
# (`test_groups_query_hands_members_back_in_adverse_order`) checks it.
_GROUPS = (
    "MATCH (e:Entity {group_id:$group_id}) "
    "WITH e ORDER BY e.created_at DESC, e.uuid DESC "
    "WITH e.name AS name, collect(e) AS nodes WHERE size(nodes) > 1 "
    "RETURN name, [n IN nodes | {uuid: n.uuid, created_at: toString(n.created_at), "
    "  created_at_epoch: [n.created_at.epochSeconds, n.created_at.nanosecond], "
    "  labels: labels(n), labels_prop: n.labels, summary: n.summary, "
    "  name_embedding: n.name_embedding, degree: COUNT { (n)--() }}] AS members "
    "ORDER BY name"
)

# Names that collide only after graphiti-style normalisation. Report-only.
_NEAR = (
    "MATCH (e:Entity {group_id:$group_id}) "
    "WITH toLower(trim(e.name)) AS norm, collect(DISTINCT e.name) AS names "
    "WHERE size(names) > 1 RETURN norm, names ORDER BY norm"
)


async def _read(driver: AsyncDriver, query: str, **params: Any) -> list[dict[str, Any]]:
    """Run one query READ-routed: on a single instance a write clause inside
    it fails with "Writing in read access mode not allowed" instead of being
    applied -- the planner's no-write guarantee is structural, not by care."""
    result = await driver.execute_query(query, params, routing_=RoutingControl.READ)
    return [dict(rec) for rec in result.records]


async def count_duplicates(driver: AsyncDriver, group_id: str) -> int:
    """Number of excess `:Entity` nodes in `group_id`: sum of (copies - 1) over
    every byte-identical name that occurs more than once. Zero on a clean graph."""
    rows = await _read(driver, _COUNT, group_id=group_id)
    return int(rows[0]["excess"]) if rows else 0


def _order_key(member: dict[str, Any]) -> tuple[bool, int, int, str]:
    """(created_at, uuid) as a total order. A member with no `created_at`
    sorts LAST: it cannot honestly claim to be the earliest."""
    epoch = member.get("created_at_epoch") or [None, None]
    seconds, nanos = epoch[0], epoch[1]
    missing = seconds is None
    return (missing, 0 if missing else int(seconds), 0 if nanos is None else int(nanos),
            str(member["uuid"]))


def _cosine(a: list[float], b: list[float]) -> float | None:
    if not a or not b or len(a) != len(b):
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return None
    return dot / (na * nb)


def _min_pairwise_cosine(vectors: list[list[float] | None]) -> float | None:
    """The WORST pairwise cosine among the members' name embeddings (report
    only). Same string through the same embedder should give identical
    vectors, so the expected value is ~1.0; anything lower means the "one
    embedding space everywhere" assumption does not hold for this group.
    None if any member lacks a vector -- unknown is not the same as 1.0."""
    if any(v is None for v in vectors):
        return None
    present = [v for v in vectors if v is not None]
    worst: float | None = None
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            c = _cosine(present[i], present[j])
            if c is None:
                return None
            worst = c if worst is None else min(worst, c)
    return worst


def _custom_labels(labels: list[str]) -> set[str]:
    return {label for label in labels if label != "Entity"}


def _label_conflict(survivor: dict[str, Any], losers: list[dict[str, Any]]) -> bool:
    """A conflict is a custom label some loser carries that the survivor does
    not AND that the merge will NOT promote. Promotion (spec §4.4, graphiti's
    `_promote_resolved_node` mirror) happens only onto a bare `:Entity`
    survivor, so a typed survivor facing any foreign loser label is a conflict;
    a bare survivor never is, and a bare loser never contributes one."""
    survivor_custom = _custom_labels(survivor["labels"])
    if not survivor_custom:
        return False
    return any(_custom_labels(loser["labels"]) - survivor_custom for loser in losers)


def _public_member(member: dict[str, Any]) -> dict[str, Any]:
    """The report shape. Never the embedding (768 floats per node), never the
    raw summary (it is emitted by the apply step's audit when dropped)."""
    return {
        "uuid": member["uuid"],
        "created_at": member["created_at"],
        "labels": list(member["labels"]),
        "degree": int(member["degree"]),
        "summary_len": len(member.get("summary") or ""),
    }


async def plan_merges(driver: AsyncDriver, group_id: str) -> dict[str, Any]:
    """Find every exact-name duplicate group in `group_id` and decide what a
    merge would do, without touching the graph. See the module docstring for
    the payload's consumers and the survivor rule."""
    group_rows = await _read(driver, _GROUPS, group_id=group_id)
    near_rows = await _read(driver, _NEAR, group_id=group_id)

    groups: list[dict[str, Any]] = []
    label_conflicts: list[dict[str, Any]] = []
    for row in group_rows:
        members = sorted(row["members"], key=_order_key)
        survivor, losers = members[0], members[1:]
        groups.append({
            "name": row["name"],
            "survivor": survivor["uuid"],
            "losers": [m["uuid"] for m in losers],
            "members": [_public_member(m) for m in members],
            "name_embedding_cosine": _min_pairwise_cosine(
                [m.get("name_embedding") for m in members]),
        })
        if _label_conflict(survivor, losers):
            label_conflicts.append({
                "name": row["name"],
                "labels": [list(m["labels"]) for m in members],
            })

    return {
        "groups": groups,
        "totals": {
            "groups": len(groups),
            "excess": sum(len(g["members"]) - 1 for g in groups),
        },
        "near_duplicates_not_merged": [
            {"normalized": row["norm"], "names": sorted(row["names"])} for row in near_rows
        ],
        "label_conflicts": label_conflicts,
    }
