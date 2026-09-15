"""Exact-name duplicate entities: the read-only planner and the transactional merge.

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

The destructive half (spec §4.5) rests on two mechanics, both load-bearing:

1. **Every edge is re-created with its full property map INCLUDING its
   original uuid, then the original is deleted.** Neo4j cannot re-point a
   relationship. The 655 baseline episodes carry 4,886 `Episodic.entity_edges`
   references BY EDGE UUID, and `Community.cited_fact_uuids`,
   `resolve_citations` and the staleness sweep all address facts by uuid. A
   regenerated uuid would orphan every one of them silently -- the graph
   would still answer, just wrongly. There is no uniqueness constraint on
   `RELATES_TO.uuid`, so old and new sharing a uuid inside the transaction is
   legal.
2. **The loser is removed with `DELETE`, never `DETACH DELETE`.** Any
   relationship type this module did not enumerate makes Neo4j refuse the
   delete and abort the transaction instead of dropping the edge with the
   loser. A guard before the `DELETE` names the surviving types so the error
   is readable; the plain `DELETE` is the belt under those braces.

One explicit transaction per name group: a crash leaves a group either
untouched or fully merged, never half-rewired. The run STOPS on the first
failing group and raises (unlike `ingest_source`, which continues): a
failure in a destructive pass means the graph is not what the design
assumed, and the operator should look before more is changed.

Edges that meet after a merge (`S->X` and `L->X`) are KEPT as two edges, and
`L->S` / `S->L` / `L->L` facts become self-loops on `S`, kept and counted:
each carries its own fact, episodes and provenance, and the baseline already
holds 440 multi-edge pairs and 10 self-loops from ordinary sequential ingest
(spec §4.6). Collapsing them would violate design invariant #3.
"""

from __future__ import annotations

import math
from typing import Any

from neo4j import AsyncDriver, AsyncTransaction, RoutingControl

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


def _label_decision(survivor_labels: list[str], loser_labels: list[list[str]],
                    ) -> tuple[list[str], bool]:
    """Spec §4.4's label rule as ONE function, so the planner's
    `label_conflicts` and the apply step's promotion can never disagree:
    returns `(labels to promote, whether the group is a label conflict)`.

    Promotion happens only onto a bare `:Entity` survivor, and only when every
    TYPED loser carries the same custom labels -- that set is then promoted. A
    bare loser has nothing to say and never contributes a conflict (graphiti's
    `_promote_resolved_node` returns the canonical unchanged for a label-less
    extracted node). Anything else is a disagreement the merge does NOT
    resolve: it promotes nothing and reports the group. That covers a typed
    survivor facing a foreign loser label, and two typed losers that disagree
    with each other.

    At two members this is graphiti's `_promote_resolved_node` rule. At three
    or more it deliberately is NOT: sequential resolution would let the first
    typed loser win and drop the second silently; reading the rule per loser
    would union the two into a node shape graphiti never writes. Promote-
    nothing is chosen because a bare survivor is what graphiti's own promotion
    repairs on the next ingest that resolves a typed extraction onto it, while
    an arbitrary winner is permanent (spec §4.4). The union matters
    downstream: `theme_builder.cli._fetch_members` (`src/theme_builder/cli.py:43`)
    reads `[0]` of a node's custom labels as its `type`, so a two-label node
    would be typed by an unspecified pick in every community report that cites
    it. (`graph_cleanup.prune_noise_entities` reads the same `[0]`, but
    `is_noise` never consults its `type` argument, so nothing changes there.)"""
    survivor_custom = _custom_labels(survivor_labels)
    typed = [custom for custom in map(_custom_labels, loser_labels) if custom]
    if survivor_custom:
        return [], any(custom - survivor_custom for custom in typed)
    if not typed:
        return [], False
    if all(custom == typed[0] for custom in typed):
        return sorted(typed[0]), False
    return [], True


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
        if _label_decision(survivor["labels"], [m["labels"] for m in losers])[1]:
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


# --- the destructive half ------------------------------------------------------


class MergeAborted(RuntimeError):
    """A group could not be merged safely; its transaction was rolled back and
    the run stopped. The message names the group and the reason."""


# Properties graphiti itself writes on an `:Entity`, plus this module's own
# audit stamps. Anything else on a loser is a custom attribute; it is listed in
# the audit and NOT copied onto the survivor (spec §4.4: no silent overwrite
# in either direction).
_ENTITY_STANDARD_KEYS = frozenset({
    "uuid", "name", "group_id", "created_at", "summary", "name_embedding", "labels",
    "merged_from", "merged_at",
})

# The edge types this pass knows how to rewire. Step 6 refuses to delete a
# loser that still carries ANY relationship; this tuple is only the order the
# report lists counts in.
_EDGE_TYPES = ("RELATES_TO", "MENTIONS", "IN_COMMUNITY", "SAME_AS")

# Step 0 -- re-verify the plan inside the transaction: every member must still
# exist under this uuid, in this group, with THIS name. `count(e)` is 0 for a
# member that vanished or was renamed since the plan and >1 if two nodes share
# a uuid in the group (never seen; refused rather than guessed at). The row also
# carries what step 7 needs from each member (labels, summary, custom keys),
# read before any loser is deleted. The name embedding is masked: 768 floats
# per member the caller never uses.
_STEP0 = (
    "UNWIND $uuids AS u "
    "OPTIONAL MATCH (e:Entity {uuid:u, group_id:$g, name:$name}) "
    "RETURN u, count(e) AS n, collect(labels(e))[0] AS labels, "
    "collect(e{.*, name_embedding: null})[0] AS props"
)

# Step 1 -- outgoing facts: `L->X`, `L->S` (becomes a self-loop on S) and
# `L->L` (likewise). The new edge is created with the old edge's ENTIRE
# property map (uuid included), then the stored endpoints are rewritten to the
# new topology, then the vector is re-set through the same procedure graphiti
# uses (the plain-`SET` copy leaves a float list whose storage form cannot be
# told apart from the schema; the procedure removes the question), and only
# then is the old edge deleted. The CALL is a scoped unit subquery so a fact
# with no embedding (never seen from graphiti, but not this pass's problem)
# skips the procedure instead of failing it.
_STEP1_OUT = (
    "MATCH (l:Entity {uuid:$loser, group_id:$g}), (s:Entity {uuid:$survivor, group_id:$g}) "
    "MATCH (l)-[old:RELATES_TO]->(t) "
    "WITH s, l, old, properties(old) AS props, old.fact_embedding AS emb, "
    "     CASE WHEN t = l THEN s ELSE t END AS target "
    "CREATE (s)-[new:RELATES_TO]->(target) "
    "SET new = props "
    "SET new.source_node_uuid = s.uuid, new.target_node_uuid = target.uuid "
    "WITH s, old, new, emb, target "
    "CALL (new, emb) { "
    "  WITH new, emb WHERE emb IS NOT NULL "
    "  CALL db.create.setRelationshipVectorProperty(new, 'fact_embedding', emb) "
    "} "
    "DELETE old "
    "RETURN count(new) AS moved, sum(CASE WHEN target = s THEN 1 ELSE 0 END) AS loops"
)

# Step 2 -- incoming facts: `X->L` and `S->L` (a self-loop on S). Step 1
# already removed every edge that STARTED at the loser, so a loser self-loop
# is not seen twice here.
_STEP2_IN = (
    "MATCH (l:Entity {uuid:$loser, group_id:$g}), (s:Entity {uuid:$survivor, group_id:$g}) "
    "MATCH (x)-[old:RELATES_TO]->(l) "
    "WITH s, x, old, properties(old) AS props, old.fact_embedding AS emb "
    "CREATE (x)-[new:RELATES_TO]->(s) "
    "SET new = props "
    "SET new.source_node_uuid = x.uuid, new.target_node_uuid = s.uuid "
    "WITH s, x, old, new, emb "
    "CALL (new, emb) { "
    "  WITH new, emb WHERE emb IS NOT NULL "
    "  CALL db.create.setRelationshipVectorProperty(new, 'fact_embedding', emb) "
    "} "
    "DELETE old "
    "RETURN count(new) AS moved, sum(CASE WHEN x = s THEN 1 ELSE 0 END) AS loops"
)

# Step 3 -- MENTIONS. An episode that already mentions the survivor keeps BOTH
# edges: each has its own uuid and created_at, and collapsing them is a
# judgement this pass does not make.
_STEP3_MENTIONS = (
    "MATCH (l:Entity {uuid:$loser, group_id:$g}), (s:Entity {uuid:$survivor, group_id:$g}) "
    "MATCH (ep:Episodic)-[old:MENTIONS]->(l) "
    "CREATE (ep)-[new:MENTIONS]->(s) "
    "SET new = properties(old) "
    "DELETE old "
    "RETURN count(new) AS moved"
)

# Step 4 -- IN_COMMUNITY: membership is a set, so MERGE. `stale` is the flag
# `theme_builder.incremental` already treats as "regenerate this community";
# without it the merge would be invisible to the incremental refresh, which
# keys `touched_entities` on `created_at` (the survivor keeps its own).
_STEP4_COMMUNITY = (
    "MATCH (l:Entity {uuid:$loser, group_id:$g}), (s:Entity {uuid:$survivor, group_id:$g}) "
    "MATCH (l)-[old:IN_COMMUNITY]->(c) "
    "MERGE (s)-[:IN_COMMUNITY]->(c) "
    "SET c.stale = true "
    "DELETE old "
    "RETURN count(old) AS moved"
)

# Step 5 -- SAME_AS from a structural Vendor/Product: a set as well, so MERGE.
_STEP5_SAME_AS = (
    "MATCH (l:Entity {uuid:$loser, group_id:$g}), (s:Entity {uuid:$survivor, group_id:$g}) "
    "MATCH (v)-[old:SAME_AS]->(l) "
    "MERGE (v)-[:SAME_AS]->(s) "
    "DELETE old "
    "RETURN count(old) AS moved"
)

# Step 6 -- the guard, then the delete. Any relationship still on the loser is
# one this pass did not enumerate; name its type and abort. And `DELETE`, NOT
# `DETACH DELETE`: even if the guard were wrong, Neo4j refuses to delete a node
# with relationships, so an unknown edge type can never vanish with the loser.
_STEP6_REMAINING = (
    "MATCH (l:Entity {uuid:$loser, group_id:$g}) "
    "OPTIONAL MATCH (l)-[r]-() "
    "RETURN collect(DISTINCT type(r)) AS types"
)
_STEP6_DELETE = "MATCH (l:Entity {uuid:$loser, group_id:$g}) DELETE l RETURN count(*) AS deleted"

# Step 7 -- audit stamps and the field rules of spec §4.4 on the survivor.
# `merged_from` also folds in a loser's own `merged_from`, so a chain of
# merges keeps every uuid that ever pointed at this name. `coalesce($summary,
# s.summary)` keeps the survivor's summary unless the caller chose a loser's
# (only when the survivor's is empty). The label promotion clause is appended
# only when there is something to promote.
_STEP7_STAMP = (
    "MATCH (s:Entity {uuid:$survivor, group_id:$g, name:$name}) "
    "SET s.merged_from = coalesce(s.merged_from, []) + $merged_from, "
    "    s.merged_at = datetime(), "
    "    s.summary = coalesce($summary, s.summary) "
)
_STEP7_PROMOTE = (
    "WITH s, coalesce(s.labels, labels(s)) AS had "
    "SET s:$($promote) "
    "SET s.labels = had + [x IN $promote WHERE NOT x IN had] "
)


async def _tx_rows(tx: AsyncTransaction, query: str, **params: Any) -> list[dict[str, Any]]:
    result = await tx.run(query, params)
    return [dict(rec) async for rec in result]


async def _tx_one(tx: AsyncTransaction, query: str, **params: Any) -> dict[str, Any]:
    rows = await _tx_rows(tx, query, **params)
    return rows[0] if rows else {}


def _verify_members(group: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Step 0's Python half: every member exactly once, or abort naming why."""
    name = group["name"]
    by_uuid = {row["u"]: row for row in rows}
    missing = [u for u, row in by_uuid.items() if row["n"] == 0]
    ambiguous = [u for u, row in by_uuid.items() if row["n"] > 1]
    if missing or ambiguous:
        raise MergeAborted(
            f"group {name!r}: plan no longer matches the graph "
            f"(missing or renamed: {missing}, ambiguous uuid: {ambiguous}); "
            "nothing was changed for this group")
    return by_uuid


def _decide_step7(group: dict[str, Any], members: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The survivor's field rules (spec §4.4), decided from the rows step 0
    read before any loser was deleted.

    Labels: `_label_decision` -- promoted onto a bare `:Entity` survivor only,
    and only the labels every typed loser agrees on; a disagreement promotes
    nothing (the planner has reported it under `label_conflicts`). Decided
    from the whole group at once, not loser by loser, so the outcome is the
    same list the planner's rule saw.

    Summary: the survivor's, unless empty, in which case the first loser
    (in survivor order) with a non-empty summary lends its own. Every other
    non-empty loser summary that differs from what the survivor ends up with
    is emitted in the audit -- nothing vanishes unseen.

    Custom properties on a loser are listed by key, never copied."""
    name = group["name"]
    survivor = members[group["survivor"]]
    survivor_props = survivor["props"] or {}
    promote, _ = _label_decision(
        survivor["labels"] or [], [members[u]["labels"] or [] for u in group["losers"]])

    adopted: str | None = None
    summary = survivor_props.get("summary") or ""
    dropped: list[dict[str, Any]] = []
    properties_dropped: list[dict[str, Any]] = []
    merged_from: list[str] = []
    for loser_uuid in group["losers"]:
        loser = members[loser_uuid]
        props = loser["props"] or {}
        merged_from.append(loser_uuid)
        merged_from.extend(str(u) for u in (props.get("merged_from") or []))
        loser_summary = props.get("summary") or ""
        if loser_summary:
            if not summary:
                adopted = summary = loser_summary
            elif loser_summary != summary:
                dropped.append({"name": name, "survivor": group["survivor"],
                                "loser": loser_uuid, "summary": loser_summary})
        extra = sorted(set(props) - _ENTITY_STANDARD_KEYS)
        if extra:
            properties_dropped.append({"name": name, "loser": loser_uuid, "keys": extra})
    return {
        "merged_from": merged_from,
        "promote": promote,
        "summary": adopted,
        "summary_dropped": dropped,
        "properties_dropped": properties_dropped,
    }


async def _merge_group_in(tx: AsyncTransaction, group_id: str, group: dict[str, Any],
                          ) -> dict[str, Any]:
    """Steps 0-7 of spec §4.5 inside an already-open transaction. Raises on
    anything unexpected; the caller's transaction then rolls back."""
    name, survivor = group["name"], group["survivor"]
    rows = await _tx_rows(tx, _STEP0, uuids=[survivor, *group["losers"]], g=group_id, name=name)
    members = _verify_members(group, rows)
    decision = _decide_step7(group, members)

    # `moved` counts MOVE OPERATIONS, not distinct edges. A fact between two
    # losers is relocated twice (once per loser: L1->L2 becomes S->L2, then
    # S->S), so `RELATES_TO` can exceed the number of distinct edges in a group
    # of three or more. `self_loops_created` is exact (a loop is counted at the
    # one move that closes it) and `totals.merged` is nodes, not edges.
    moved = dict.fromkeys(_EDGE_TYPES, 0)
    self_loops = 0
    for loser in group["losers"]:
        params = {"loser": loser, "survivor": survivor, "g": group_id}
        out = await _tx_one(tx, _STEP1_OUT, **params)
        inc = await _tx_one(tx, _STEP2_IN, **params)
        moved["RELATES_TO"] += int(out.get("moved", 0)) + int(inc.get("moved", 0))
        self_loops += int(out.get("loops", 0)) + int(inc.get("loops", 0))
        moved["MENTIONS"] += int((await _tx_one(tx, _STEP3_MENTIONS, **params)).get("moved", 0))
        moved["IN_COMMUNITY"] += int(
            (await _tx_one(tx, _STEP4_COMMUNITY, **params)).get("moved", 0))
        moved["SAME_AS"] += int((await _tx_one(tx, _STEP5_SAME_AS, **params)).get("moved", 0))

        remaining = (await _tx_one(tx, _STEP6_REMAINING, loser=loser, g=group_id)).get("types")
        if remaining:
            raise MergeAborted(
                f"group {name!r}: loser {loser} still has relationships of type(s) "
                f"{sorted(remaining)} this pass does not know how to rewire; "
                "refusing to delete it, rolling the group back")
        deleted = await _tx_one(tx, _STEP6_DELETE, loser=loser, g=group_id)
        if int(deleted.get("deleted", 0)) != 1:
            raise MergeAborted(f"group {name!r}: loser {loser} was not deleted (matched "
                               f"{deleted.get('deleted', 0)} nodes); rolling the group back")

    stamp = _STEP7_STAMP + (_STEP7_PROMOTE if decision["promote"] else "")
    await _tx_rows(tx, stamp, survivor=survivor, g=group_id, name=name,
                   merged_from=decision["merged_from"], summary=decision["summary"],
                   promote=decision["promote"])
    return {
        "name": name,
        "survivor": survivor,
        "losers": list(group["losers"]),
        "edges_moved": moved,
        "self_loops_created": self_loops,
        "labels_promoted": ([{"name": name, "survivor": survivor, "labels": decision["promote"]}]
                            if decision["promote"] else []),
        "summary_dropped": decision["summary_dropped"],
        "properties_dropped": decision["properties_dropped"],
    }


async def merge_group(driver: AsyncDriver, group_id: str, group: dict[str, Any]) -> dict[str, Any]:
    """Merge one planned group in ONE explicit transaction. The group is a
    `plan_merges` entry (`name`, `survivor`, `losers`). Committed only if
    every step succeeded; any exception -- a Python-level abort or a server
    refusal -- leaves the group exactly as it was.

    Returns `{name, survivor, losers, edges_moved, self_loops_created,
    labels_promoted, summary_dropped, properties_dropped}`. `edges_moved`
    is move operations by type (see `_merge_group_in`: a loser-to-loser fact
    is moved twice). The three audit lists carry the SAME element shapes
    `apply_merges` returns -- `labels_promoted` is `[{name, survivor, labels}]`
    (empty or one entry), `summary_dropped` is `[{name, survivor, loser,
    summary}]`, `properties_dropped` is `[{name, loser, keys}]` -- so a caller
    of either function reads one type under one key."""
    async with driver.session() as session:
        async with await session.begin_transaction() as tx:
            outcome = await _merge_group_in(tx, group_id, group)
            await tx.commit()
    return outcome


async def apply_merges(driver: AsyncDriver, group_id: str) -> dict[str, Any]:
    """Plan, then merge every exact-name group in `group_id`, one transaction
    per group, stopping at the first failure. Returns the `plan_merges`
    payload plus `totals.merged` (loser nodes removed), `edges_moved` by
    type (move operations, not distinct edges -- a fact between two losers
    counts twice), `self_loops_created`, `summary_dropped`, `labels_promoted`
    and `properties_dropped` (each a list of the dicts `merge_group`
    documents, concatenated over groups). A graph with no duplicates is not
    touched at all."""
    plan = await plan_merges(driver, group_id)
    edges_moved = dict.fromkeys(_EDGE_TYPES, 0)
    self_loops = 0
    merged = 0
    summary_dropped: list[dict[str, Any]] = []
    labels_promoted: list[dict[str, Any]] = []
    properties_dropped: list[dict[str, Any]] = []
    for group in plan["groups"]:
        outcome = await merge_group(driver, group_id, group)
        merged += len(outcome["losers"])
        for edge_type, n in outcome["edges_moved"].items():
            edges_moved[edge_type] += n
        self_loops += outcome["self_loops_created"]
        summary_dropped.extend(outcome["summary_dropped"])
        properties_dropped.extend(outcome["properties_dropped"])
        labels_promoted.extend(outcome["labels_promoted"])
    plan["totals"]["merged"] = merged
    return {
        **plan,
        "edges_moved": edges_moved,
        "self_loops_created": self_loops,
        "summary_dropped": summary_dropped,
        "labels_promoted": labels_promoted,
        "properties_dropped": properties_dropped,
    }
