"""Exact-name duplicate planner and the transactional merge.

Every fixture here is built so the correct behaviour produces a DIFFERENT
observable result from the plausible broken ones: the survivor test's
earliest node is neither the smallest uuid nor the highest degree; the
cosine test's drifted group has three members so the worst pair (0.0) and
the best pair (1.0) are different numbers; the label test seeds one
promotable and one conflicting group; the zero-duplicate test asserts the
snapshot, not just an empty list.

And that difference must not depend on luck. `_GROUPS` hands members back in
an order ADVERSE to the survivor rule (latest first, largest uuid first), so
a planner that stopped sorting could never land on the right survivor because
the scan happened to produce it first. The survivor and tie-break tests seed
in the FAVOURABLE insertion order on purpose: if the query's ORDER BY were
lost, insertion order would hide the missing sort -- which is why a separate
test pins the collected order itself.

The apply half (second section) is destructive, so a count is never enough:
the conservation test compares every edge's FULL property map keyed by uuid
before and after; the foreign-edge test's loser carries edges that steps 1-5
move BEFORE step 6 refuses (so an auto-commit mutant leaves a visibly
half-rewired group) and one foreign edge in EACH direction (so a guard that
only looked outward would miss the incoming one); the other-group test gives
the other group the SAME uuids and one edge of every shape steps 1-5 match
(outgoing fact, incoming fact, MENTIONS, IN_COMMUNITY, SAME_AS), so a `MATCH`
in any of steps 0-7 that dropped `group_id` has something to hit -- steps 1-5
would move the other group's edge, step 0 would find two nodes per uuid,
step 6 would meet the other loser's edges, step 7 would stamp the other
survivor; and the three-member tests seed the production shape, because a
fact between two losers and a label disagreement between two losers do not
exist at two members.
"""

import json

import pytest
import pytest_asyncio

from graph_extract import merge_duplicates
from graph_extract.merge_duplicates import (
    _GROUPS,
    _STEP6_DELETE,
    MergeAborted,
    _read,
    apply_merges,
    count_duplicates,
    merge_group,
    plan_merges,
)

pytestmark = pytest.mark.asyncio(loop_scope="module")

GROUP = "backup-docs"


@pytest_asyncio.fixture(loop_scope="module")
async def neo4j_driver(extract_driver):
    """The module-scoped testcontainer driver, wiped before each test -- the
    container is shared, and these tests all use the same group id. The two
    indexes one test creates are dropped here too, so a failure inside that
    test cannot leave them behind to change the plan of the others."""
    await extract_driver.execute_query("MATCH (n) DETACH DELETE n")
    await extract_driver.execute_query("DROP INDEX entity_group_id IF EXISTS")
    await extract_driver.execute_query("DROP INDEX entity_uuid IF EXISTS")
    yield extract_driver


async def _seed(driver, *, uuid, name, created_at="2026-01-01T00:00:00Z", degree=0,
                group_id=GROUP, labels=(), summary="", name_embedding=None):
    """One `:Entity` the way graphiti writes it: zoned `created_at`, a `labels`
    property mirroring the node labels, the vector set through the procedure.
    `degree` fact edges go to fresh uniquely-named neighbours."""
    extra = "".join(f":`{label}`" for label in labels)
    await driver.execute_query(
        f"CREATE (e:Entity{extra} {{uuid:$uuid, name:$name, group_id:$g, "
        f"created_at:datetime($created_at), labels:$labels, summary:$summary}})",
        uuid=uuid, name=name, g=group_id, created_at=created_at,
        labels=["Entity", *labels], summary=summary,
    )
    if name_embedding is not None:
        await driver.execute_query(
            "MATCH (e:Entity {uuid:$uuid}) "
            "CALL db.create.setNodeVectorProperty(e, 'name_embedding', $vec)",
            uuid=uuid, vec=name_embedding,
        )
    for i in range(degree):
        await driver.execute_query(
            "MATCH (e:Entity {uuid:$uuid}) "
            "CREATE (e)-[:RELATES_TO {uuid:$edge, group_id:$g, fact:'f'}]->"
            "(:Entity {uuid:$nbr, name:$nbr, group_id:$g, created_at:datetime()})",
            uuid=uuid, edge=f"{uuid}-edge-{i}", nbr=f"{uuid}-nbr-{i}", g=group_id,
        )


async def _seed_duplicate_group(driver, *, name, n, **kwargs):
    """`n` exact-name copies with strictly increasing `created_at`, so the
    survivor is `<name>-0` by construction. Returns the uuids in that order."""
    uuids = []
    for i in range(n):
        uuid = f"{name}-{i}"
        await _seed(driver, uuid=uuid, name=name,
                    created_at=f"2026-01-{i + 1:02d}T00:00:00Z", **kwargs)
        uuids.append(uuid)
    return uuids


async def _snapshot(driver):
    """The whole graph as data: every node's labels + full property map, every
    relationship's type + full property map + endpoints. Compared, not counted."""
    nodes = await driver.execute_query(
        "MATCH (n) RETURN elementId(n) AS id, labels(n) AS labels, properties(n) AS props")
    rels = await driver.execute_query(
        "MATCH (a)-[r]->(b) RETURN elementId(a) AS a, type(r) AS t, "
        "properties(r) AS props, elementId(b) AS b")
    def rows(result):
        return sorted(json.dumps(dict(r), sort_keys=True, default=str) for r in result.records)
    return {"nodes": rows(nodes), "rels": rows(rels)}


# --- the brief's tests, verbatim ---------------------------------------------

async def test_the_planner_writes_nothing(neo4j_driver):
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=3)
    before = await _snapshot(neo4j_driver)
    await plan_merges(neo4j_driver, "backup-docs")
    assert await _snapshot(neo4j_driver) == before, "report mode must not write"


async def test_the_survivor_is_the_earliest_created_at(neo4j_driver):
    """Deliberately built so the earliest node is NEITHER the smallest uuid NOR
    the highest degree -- otherwise the test cannot tell the rules apart."""
    await _seed(neo4j_driver, uuid="zzz", name="AWS Backup",
                created_at="2026-01-01T00:00:00Z", degree=1)
    await _seed(neo4j_driver, uuid="aaa", name="AWS Backup",
                created_at="2026-06-01T00:00:00Z", degree=9)
    plan = await plan_merges(neo4j_driver, "backup-docs")
    group = plan["groups"][0]
    assert group["survivor"] == "zzz"
    assert group["losers"] == ["aaa"]


async def test_ties_on_created_at_break_by_ascending_uuid(neo4j_driver):
    """`aaa` is seeded FIRST: under insertion order a planner with no
    tie-break would pick it by accident. `_GROUPS` orders uuid DESC before
    collecting, so `bbb` arrives first and only the tie-break can choose `aaa`."""
    await _seed(neo4j_driver, uuid="aaa", name="X", created_at="2026-01-01T00:00:00Z")
    await _seed(neo4j_driver, uuid="bbb", name="X", created_at="2026-01-01T00:00:00Z")
    plan = await plan_merges(neo4j_driver, "backup-docs")
    assert plan["groups"][0]["survivor"] == "aaa"


async def test_groups_query_hands_members_back_in_adverse_order(neo4j_driver):
    """The collected order is specified, not incidental: latest `created_at`
    first, then largest uuid first -- the reverse of the survivor rule.

    The fixture is built so every plausible wrong order is a DIFFERENT
    sequence from the right one. The latest node (`aaa`, June) has the
    SMALLEST uuid, so `created_at` order and uuid order disagree:

        (created_at DESC, uuid DESC)  -> aaa, ccc, bbb   the clause as written
        (uuid DESC, created_at DESC)  -> ccc, bbb, aaa   keys swapped
        (created_at ASC, uuid ASC)    -> bbb, ccc, aaa   favourable direction
        insertion order (no ORDER BY) -> bbb, ccc, aaa   seeded survivor-first

    Both indexes graphiti creates on `:Entity` (`entity_group_id`, which this
    query's plan seeks on, and `entity_uuid`, which it does not use) are
    present, so the query plans as it does on the live graph."""
    await neo4j_driver.execute_query(
        "CREATE INDEX entity_group_id IF NOT EXISTS FOR (n:Entity) ON (n.group_id)")
    await neo4j_driver.execute_query(
        "CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)")
    await neo4j_driver.execute_query("CALL db.awaitIndexes(60)")
    await _seed(neo4j_driver, uuid="bbb", name="X", created_at="2026-01-01T00:00:00Z")
    await _seed(neo4j_driver, uuid="ccc", name="X", created_at="2026-01-01T00:00:00Z")
    await _seed(neo4j_driver, uuid="aaa", name="X", created_at="2026-06-01T00:00:00Z")
    rows = await _read(neo4j_driver, _GROUPS, group_id=GROUP)
    assert [m["uuid"] for m in rows[0]["members"]] == ["aaa", "ccc", "bbb"]
    plan = await plan_merges(neo4j_driver, GROUP)
    assert plan["groups"][0]["survivor"] == "bbb", "the Python sort stays authoritative"


async def test_case_variants_are_not_a_group_but_are_reported(neo4j_driver):
    """The sequential baseline holds 6 such pairs that graphiti itself left
    apart. They are not concurrency artefacts, and a pass whose purpose is to
    repair concurrency artefacts must not alter the reference graph."""
    await _seed(neo4j_driver, uuid="a", name="Access Control")
    await _seed(neo4j_driver, uuid="b", name="Access control")
    plan = await plan_merges(neo4j_driver, "backup-docs")
    assert plan["groups"] == []
    assert "Access Control" in str(plan["near_duplicates_not_merged"])


async def test_another_group_id_is_not_a_duplicate(neo4j_driver):
    await _seed(neo4j_driver, uuid="a", name="X", group_id="backup-docs")
    await _seed(neo4j_driver, uuid="b", name="X", group_id="other")
    plan = await plan_merges(neo4j_driver, "backup-docs")
    assert plan["groups"] == []
    assert plan["totals"]["excess"] == 0
    assert await count_duplicates(neo4j_driver, "backup-docs") == 0


async def test_count_duplicates_counts_excess_not_groups(neo4j_driver):
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=3)
    await _seed_duplicate_group(neo4j_driver, name="Amazon EC2", n=2)
    # 3 nodes of one name + 2 of another = 5 nodes, 2 names, 3 excess
    assert await count_duplicates(neo4j_driver, "backup-docs") == 3


# --- payload shape and the report-only diagnostics ----------------------------

async def test_totals_agree_with_count_duplicates_and_groups_are_sorted(neo4j_driver):
    """`totals.excess` is what Task 7 gates on; it must be the same number
    `count_duplicates` gives, and it is excess nodes, not group count."""
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=3)
    await _seed_duplicate_group(neo4j_driver, name="Amazon EC2", n=2)
    await _seed(neo4j_driver, uuid="solo", name="Unique")
    plan = await plan_merges(neo4j_driver, GROUP)
    assert plan["totals"] == {"groups": 2, "excess": 3}
    assert plan["totals"]["excess"] == await count_duplicates(neo4j_driver, GROUP)
    assert [g["name"] for g in plan["groups"]] == ["AWS Backup", "Amazon EC2"]
    assert plan["near_duplicates_not_merged"] == []
    assert plan["label_conflicts"] == []


async def test_members_are_survivor_first_and_carry_the_report_fields(neo4j_driver):
    await _seed(neo4j_driver, uuid="zzz", name="AWS Backup", labels=("Product",),
                created_at="2026-01-01T00:00:00Z", degree=1, summary="abc")
    await _seed(neo4j_driver, uuid="aaa", name="AWS Backup",
                created_at="2026-06-01T00:00:00Z", degree=9)
    group = (await plan_merges(neo4j_driver, GROUP))["groups"][0]
    assert [m["uuid"] for m in group["members"]] == ["zzz", "aaa"]
    assert group["members"][0] == {
        "uuid": "zzz", "created_at": "2026-01-01T00:00:00Z",
        "labels": ["Entity", "Product"], "degree": 1, "summary_len": 3}
    assert group["members"][1]["degree"] == 9
    assert group["members"][1]["summary_len"] == 0


async def test_the_payload_is_json_and_never_carries_an_embedding(neo4j_driver):
    """Tasks 6/7 print and persist this payload. 768 floats per member is not
    a report, and a Neo4j DateTime object is not JSON."""
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=2,
                                name_embedding=[0.5, 0.5, 0.5])
    plan = await plan_merges(neo4j_driver, GROUP)
    json.dumps(plan)  # DateTime objects or float vectors of any size would raise / bloat
    for member in plan["groups"][0]["members"]:
        assert set(member) == {"uuid", "created_at", "labels", "degree", "summary_len"}


async def test_name_embedding_cosine_checks_the_same_vector_assumption(neo4j_driver):
    """Identical vectors -> 1.0; a group with ONE drifted member -> the worst
    pair, not the best; a member without a vector -> None (unknown is not 1.0).

    The drifted group has three members: two identical (cosine 1.0 with each
    other) and one orthogonal to both (0.0). The field must report 0.0 -- an
    instrument that reported the best pair (1.0) would say "nothing to see
    here" about a group whose assumption has demonstrably failed. The drifted
    member is the LATEST, so it is last after sorting and no "first pair only"
    shortcut can see it either."""
    await _seed_duplicate_group(neo4j_driver, name="Same", n=2,
                                name_embedding=[0.1, 0.2, 0.3])
    await _seed(neo4j_driver, uuid="d0", name="Drifted", name_embedding=[1.0, 0.0, 0.0])
    await _seed(neo4j_driver, uuid="d1", name="Drifted", name_embedding=[1.0, 0.0, 0.0],
                created_at="2026-02-01T00:00:00Z")
    await _seed(neo4j_driver, uuid="d2", name="Drifted", name_embedding=[0.0, 1.0, 0.0],
                created_at="2026-03-01T00:00:00Z")
    await _seed(neo4j_driver, uuid="m0", name="Missing", name_embedding=[1.0, 0.0, 0.0])
    await _seed(neo4j_driver, uuid="m1", name="Missing", created_at="2026-02-01T00:00:00Z")
    by_name = {g["name"]: g["name_embedding_cosine"]
               for g in (await plan_merges(neo4j_driver, GROUP))["groups"]}
    assert abs(by_name["Same"] - 1.0) < 1e-9
    assert abs(by_name["Drifted"]) < 1e-9, "the worst pair is orthogonal; the best pair is 1.0"
    assert by_name["Missing"] is None


async def test_label_conflicts_only_when_the_survivor_is_typed(neo4j_driver):
    """Bare `:Entity` survivor + `:Product` loser is a PROMOTION (spec §4.4),
    not a conflict. `:Tool` survivor + `:Product` loser drops a label and
    must be reported. Same labels both sides is nothing."""
    await _seed(neo4j_driver, uuid="p0", name="Promotable")
    await _seed(neo4j_driver, uuid="p1", name="Promotable", labels=("Product",),
                created_at="2026-02-01T00:00:00Z")
    await _seed(neo4j_driver, uuid="c0", name="Conflict", labels=("Tool",))
    await _seed(neo4j_driver, uuid="c1", name="Conflict", labels=("Product",),
                created_at="2026-02-01T00:00:00Z")
    await _seed(neo4j_driver, uuid="s0", name="Agree", labels=("Product",))
    await _seed(neo4j_driver, uuid="s1", name="Agree", labels=("Product",),
                created_at="2026-02-01T00:00:00Z")
    plan = await plan_merges(neo4j_driver, GROUP)
    assert plan["totals"]["groups"] == 3
    assert plan["label_conflicts"] == [
        {"name": "Conflict", "labels": [["Entity", "Tool"], ["Entity", "Product"]]}]


async def test_a_graph_without_duplicates_reports_nothing_and_is_untouched(neo4j_driver):
    await _seed(neo4j_driver, uuid="a", name="Unique", degree=2)
    await _seed(neo4j_driver, uuid="b", name="Other", labels=("Product",))
    before = await _snapshot(neo4j_driver)
    plan = await plan_merges(neo4j_driver, GROUP)
    assert plan == {"groups": [], "totals": {"groups": 0, "excess": 0},
                    "near_duplicates_not_merged": [], "label_conflicts": []}
    assert await count_duplicates(neo4j_driver, GROUP) == 0
    assert await _snapshot(neo4j_driver) == before


# =============================================================================
# The destructive half: apply_merges / merge_group
# =============================================================================

# `_seed_duplicate_group(name="AWS Backup", n=2)` names its members
# `AWS Backup-0` (earliest, the survivor) and `AWS Backup-1` (the loser).
_S = "AWS Backup-0"
_LOSER = "AWS Backup-1"
_X = "Amazon S3"
_Y = "Veeam"


async def _fact(driver, *, src, dst, uuid, embedding=None, **props):
    """One `RELATES_TO` edge the way graphiti writes it: the full property map
    (endpoints stored as properties too), the vector through the procedure so
    its stored form is graphiti's (float32), not a Python float list."""
    base = {"uuid": uuid, "name": f"n-{uuid}", "fact": f"fact {uuid}", "group_id": GROUP,
            "source_node_uuid": src, "target_node_uuid": dst, "episodes": []}
    await driver.execute_query(
        "MATCH (a:Entity {uuid:$src, group_id:$g}), (b:Entity {uuid:$dst, group_id:$g}) "
        "CREATE (a)-[f:RELATES_TO]->(b) SET f = $props, f.created_at = datetime()",
        src=src, dst=dst, g=GROUP, props={**base, **props})
    if embedding is not None:
        await driver.execute_query(
            "MATCH ()-[f:RELATES_TO {uuid:$uuid}]->() "
            "CALL db.create.setRelationshipVectorProperty(f, 'fact_embedding', $vec)",
            uuid=uuid, vec=embedding)


async def _episode(driver, *, uuid, mentions, entity_edges):
    await driver.execute_query(
        "CREATE (ep:Episodic {uuid:$uuid, group_id:$g, entity_edges:$edges, name:$uuid}) "
        "WITH ep UNWIND $mentions AS m "
        "MATCH (e:Entity {uuid:m, group_id:$g}) "
        "CREATE (ep)-[:MENTIONS {uuid:$uuid + '->' + m, group_id:$g, created_at:datetime()}]->(e)",
        uuid=uuid, g=GROUP, edges=entity_edges, mentions=mentions)


async def _community(driver, *, uuid, members, cited=(), stale=False):
    await driver.execute_query(
        "CREATE (c:Community {uuid:$uuid, group_id:$g, stale:$stale, cited_fact_uuids:$cited}) "
        "WITH c UNWIND $members AS m MATCH (e:Entity {uuid:m, group_id:$g}) "
        "CREATE (e)-[:IN_COMMUNITY]->(c)",
        uuid=uuid, g=GROUP, stale=stale, cited=list(cited), members=members)


async def _product(driver, *, uuid, same_as):
    await driver.execute_query(
        "CREATE (p:Product {id:$uuid, name:$uuid}) WITH p UNWIND $targets AS t "
        "MATCH (e:Entity {uuid:t, group_id:$g}) CREATE (p)-[:SAME_AS]->(e)",
        uuid=uuid, targets=same_as, g=GROUP)


def _vec(seed, dim=8):
    return [round(0.1 * ((seed + i) % 7 + 1), 3) for i in range(dim)]


async def _seed_realistic_group(driver):
    """S and L (exact-name copies), two neighbours, facts in every direction the
    merge must handle, episodes citing those facts by uuid, communities,
    structural twins:

        S->X  L->X   meet after the merge (kept as two)
        X->L         incoming
        L->S  S->L  L->L   become self-loops on S
        Y->S         untouched
    """
    await _seed_duplicate_group(driver, name="AWS Backup", n=2,
                                name_embedding=[0.5, 0.5, 0.5], summary="")
    await _seed(driver, uuid=_X, name=_X)
    await _seed(driver, uuid=_Y, name=_Y)
    await _fact(driver, src=_S, dst=_X, uuid="f-sx", embedding=_vec(1), episodes=["ep1"],
                valid_at="2025-01-01T00:00:00Z")
    await _fact(driver, src=_LOSER, dst=_X, uuid="f-lx", embedding=_vec(2), episodes=["ep1", "ep2"],
                valid_at="2025-02-01T00:00:00Z", invalid_at="2025-06-01T00:00:00Z")
    await _fact(driver, src=_X, dst=_LOSER, uuid="f-xl", embedding=_vec(3), episodes=["ep2"],
                expired_at="2025-07-01T00:00:00Z", reference_time="2025-02-02T00:00:00Z")
    await _fact(driver, src=_LOSER, dst=_S, uuid="f-ls", embedding=_vec(4), episodes=["ep2"])
    await _fact(driver, src=_S, dst=_LOSER, uuid="f-sl", embedding=_vec(5), episodes=["ep1"])
    await _fact(driver, src=_LOSER, dst=_LOSER, uuid="f-ll", embedding=_vec(6), episodes=["ep2"])
    await _fact(driver, src=_Y, dst=_S, uuid="f-ys", embedding=_vec(7), episodes=["ep1"])
    await _episode(driver, uuid="ep1", mentions=[_S, _LOSER, _X],
                   entity_edges=["f-sx", "f-lx", "f-sl", "f-ys"])
    await _episode(driver, uuid="ep2", mentions=[_LOSER, _X],
                   entity_edges=["f-lx", "f-xl", "f-ls", "f-ll"])
    await _community(driver, uuid="c-both", members=[_S, _LOSER], cited=["f-sx", "f-lx"])
    await _community(driver, uuid="c-loser", members=[_LOSER, _X], cited=["f-xl", "f-ll"])
    await _community(driver, uuid="c-other", members=[_X, _Y], cited=["f-ys"])
    await _product(driver, uuid="p-loser", same_as=[_LOSER])
    await _product(driver, uuid="p-both", same_as=[_S, _LOSER])


async def _seed_group_with_fact_embedding(driver, vec):
    await _seed_duplicate_group(driver, name="AWS Backup", n=2)
    await _seed(driver, uuid=_X, name=_X)
    await _fact(driver, src=_LOSER, dst=_X, uuid="f-lx", embedding=vec)


async def _edge_snapshot(driver):
    """Every RELATES_TO and MENTIONS edge as `{uuid, type, a, b, props}` --
    the FULL property map, `fact_embedding` included."""
    r = await driver.execute_query(
        "MATCH (a)-[f:RELATES_TO|MENTIONS]->(b) "
        "RETURN f.uuid AS uuid, type(f) AS type, a.uuid AS a, b.uuid AS b, properties(f) AS props")
    return [dict(rec) for rec in r.records]


async def _snapshot_group(driver, group_id):
    """`_snapshot` restricted to one group's nodes and every edge that touches
    one of them. "Touches", not "among": a structural `:Product` carries no
    `group_id`, so its `SAME_AS` edge into the group is only seen this way."""
    nodes = await driver.execute_query(
        "MATCH (n {group_id:$g}) RETURN elementId(n) AS id, labels(n) AS labels, "
        "properties(n) AS props", g=group_id)
    rels = await driver.execute_query(
        "MATCH (a)-[r]->(b) WHERE a.group_id = $g OR b.group_id = $g "
        "RETURN elementId(a) AS a, type(r) AS t, properties(r) AS props, elementId(b) AS b",
        g=group_id)
    def rows(result):
        return sorted(json.dumps(dict(r), sort_keys=True, default=str) for r in result.records)
    return {"nodes": rows(nodes), "rels": rows(rels)}


async def _entity(driver, uuid, group_id=GROUP):
    r = await driver.execute_query(
        "MATCH (e:Entity {uuid:$uuid, group_id:$g}) RETURN labels(e) AS labels, "
        "properties(e) AS props", uuid=uuid, g=group_id)
    return dict(r.records[0]) if r.records else None


# --- the brief's tests, verbatim ---------------------------------------------

async def test_every_edge_survives_with_its_full_property_map_and_uuid(neo4j_driver):
    await _seed_realistic_group(neo4j_driver)  # facts both ways, MENTIONS, episodes
    before = {r["uuid"]: r for r in await _edge_snapshot(neo4j_driver)}
    await apply_merges(neo4j_driver, "backup-docs")
    after = {r["uuid"]: r for r in await _edge_snapshot(neo4j_driver)}
    assert set(before) == set(after), "every edge uuid must survive"
    for uuid, old in before.items():
        new = after[uuid]
        for key, value in old["props"].items():
            if key in ("source_node_uuid", "target_node_uuid"):
                continue
            assert new["props"][key] == value, f"{uuid}.{key} changed"


async def test_stored_endpoints_agree_with_topology_for_every_edge(neo4j_driver):
    await _seed_realistic_group(neo4j_driver)
    await apply_merges(neo4j_driver, "backup-docs")
    r = await neo4j_driver.execute_query(
        "MATCH (a:Entity)-[f:RELATES_TO]->(b:Entity) "
        "WHERE f.source_node_uuid <> a.uuid OR f.target_node_uuid <> b.uuid "
        "RETURN count(f) AS n")
    assert r.records[0]["n"] == 0


async def test_the_fact_embedding_stays_scorable(neo4j_driver):
    vec = [0.1] * 768
    await _seed_group_with_fact_embedding(neo4j_driver, vec)
    await apply_merges(neo4j_driver, "backup-docs")
    r = await neo4j_driver.execute_query(
        "MATCH ()-[f:RELATES_TO]->() "
        "RETURN vector.similarity.cosine(f.fact_embedding, $v) AS sim", v=vec)
    assert abs(r.records[0]["sim"] - 1.0) < 1e-6


async def test_a_foreign_edge_type_aborts_the_group_and_stops_the_run(neo4j_driver):
    """The safety property. DELETE (not DETACH DELETE) makes an unenumerated
    edge type fail loudly instead of vanishing with the loser.

    `degree=2` is one step past the brief: the loser must carry edges that
    steps 1-5 move BEFORE step 6 refuses, otherwise "rolls back whole" is
    indistinguishable from "auto-committed every step and then stopped".

    One foreign edge in EACH direction, and the abort must name both. The
    realistic unenumerated type is incoming -- graphiti's community membership
    is `(:Community)-[:HAS_MEMBER]->(:Entity)` -- and a guard that only looked
    outward would miss `BAR`: the plain `DELETE` would still refuse, but with
    Neo4j's constraint error instead of the type's name, which is exactly the
    degradation the guard exists to prevent."""
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=2, degree=2)
    await neo4j_driver.execute_query(
        "MATCH (l:Entity {uuid:$l}) CREATE (l)-[:FOO]->(:Thing) CREATE (:Thing)-[:BAR]->(l)",
        l=_LOSER)
    before = await _snapshot(neo4j_driver)
    with pytest.raises(MergeAborted, match=r"\['BAR', 'FOO'\]"):
        await apply_merges(neo4j_driver, "backup-docs")
    assert await _snapshot(neo4j_driver) == before, "the group must roll back whole"


async def test_the_delete_statement_itself_refuses_a_node_with_relationships(neo4j_driver):
    """The belt under the braces. The step-6 guard raises BEFORE the delete
    whenever a loser still carries an edge, so no end-to-end test can tell
    `DELETE` from `DETACH DELETE` -- behind an intact guard they look the
    same (the `DETACH DELETE` mutant survived the foreign-edge test for
    exactly that reason). This pins the statement on its own: against a node
    that still has an edge, Neo4j must refuse and the edge must survive.
    Under `DETACH DELETE` there is no error and the edge is gone."""
    await _seed(neo4j_driver, uuid="l", name="L")
    await neo4j_driver.execute_query("MATCH (l:Entity {uuid:'l'}) CREATE (l)-[:FOO]->(:Thing)")
    with pytest.raises(Exception, match="still has relationships"):
        await neo4j_driver.execute_query(_STEP6_DELETE, loser="l", g=GROUP)
    r = await neo4j_driver.execute_query(
        "MATCH (:Entity {uuid:'l'})-[f:FOO]->(:Thing) RETURN count(f) AS n")
    assert r.records[0]["n"] == 1


async def test_applying_twice_changes_nothing_the_second_time(neo4j_driver):
    await _seed_realistic_group(neo4j_driver)
    await apply_merges(neo4j_driver, "backup-docs")
    once = await _snapshot(neo4j_driver)
    res = await apply_merges(neo4j_driver, "backup-docs")
    assert res["totals"]["merged"] == 0
    assert await _snapshot(neo4j_driver) == once, "a no-op must not re-stamp merged_at"


async def test_a_zero_duplicate_graph_is_untouched(neo4j_driver):
    await _seed(neo4j_driver, uuid="a", name="Unique")
    before = await _snapshot(neo4j_driver)
    res = await apply_merges(neo4j_driver, "backup-docs")
    assert res["totals"]["merged"] == 0
    assert await _snapshot(neo4j_driver) == before


# --- spec §7.2, the rest ------------------------------------------------------

async def test_the_loser_is_gone_and_the_report_counts_what_moved(neo4j_driver):
    """On the realistic group: one loser removed, every edge type counted
    exactly, three self-loops, and the payload is JSON (Task 6 prints it)."""
    await _seed_realistic_group(neo4j_driver)
    res = await apply_merges(neo4j_driver, GROUP)
    json.dumps(res)
    assert res["totals"] == {"groups": 1, "excess": 1, "merged": 1}
    # L->X, X->L, L->S, S->L, L->L = 5 facts; ep1 + ep2 = 2 mentions;
    # c-both + c-loser = 2 memberships; p-loser + p-both = 2 twins
    assert res["edges_moved"] == {"RELATES_TO": 5, "MENTIONS": 2, "IN_COMMUNITY": 2, "SAME_AS": 2}
    assert res["self_loops_created"] == 3
    assert await _entity(neo4j_driver, _LOSER) is None
    assert await count_duplicates(neo4j_driver, GROUP) == 0


async def test_meeting_facts_survive_as_two_edges(neo4j_driver):
    """`S->X` and `L->X` each carry their own fact and provenance; the
    baseline holds 440 such multi-edge pairs from sequential ingest and
    collapsing them would overwrite history (invariant #3)."""
    await _seed_realistic_group(neo4j_driver)
    await apply_merges(neo4j_driver, GROUP)
    r = await neo4j_driver.execute_query(
        "MATCH (:Entity {uuid:$s})-[f:RELATES_TO]->(:Entity {uuid:$x}) "
        "RETURN f.uuid AS uuid, f.fact AS fact ORDER BY uuid", s=_S, x=_X)
    assert [dict(rec) for rec in r.records] == [
        {"uuid": "f-lx", "fact": "fact f-lx"}, {"uuid": "f-sx", "fact": "fact f-sx"}]


async def test_facts_between_survivor_and_loser_become_self_loops(neo4j_driver):
    """`L->S`, `S->L` and `L->L` end up as `S->S`, uuids intact, counted --
    an operator can inspect them, but deleting a cited fact on a guess is
    not this pass's call (spec §4.6)."""
    await _seed_realistic_group(neo4j_driver)
    res = await apply_merges(neo4j_driver, GROUP)
    r = await neo4j_driver.execute_query(
        "MATCH (s:Entity {uuid:$s})-[f:RELATES_TO]->(s) RETURN f.uuid AS uuid ORDER BY uuid", s=_S)
    assert [rec["uuid"] for rec in r.records] == ["f-ll", "f-ls", "f-sl"]
    assert res["self_loops_created"] == 3


async def test_mentions_are_rewired_and_every_entity_edges_uuid_still_resolves(neo4j_driver):
    """ep1 mentioned both S and L: it keeps TWO MENTIONS edges to S, each with
    its own uuid. And every uuid any episode lists in `entity_edges` must
    still name an existing fact -- that list is how citations resolve."""
    await _seed_realistic_group(neo4j_driver)
    await apply_merges(neo4j_driver, GROUP)
    r = await neo4j_driver.execute_query(
        "MATCH (ep:Episodic)-[m:MENTIONS]->(e:Entity) "
        "RETURN ep.uuid AS ep, m.uuid AS m, e.uuid AS e ORDER BY ep, m")
    assert [tuple(rec.values()) for rec in r.records] == [
        ("ep1", "ep1->AWS Backup-0", _S), ("ep1", "ep1->AWS Backup-1", _S), ("ep1", "ep1->Amazon S3", _X),
        ("ep2", "ep2->AWS Backup-1", _S), ("ep2", "ep2->Amazon S3", _X)]
    r = await neo4j_driver.execute_query(
        "MATCH (ep:Episodic) UNWIND ep.entity_edges AS u "
        "WITH DISTINCT u OPTIONAL MATCH ()-[f:RELATES_TO {uuid:u}]->() "
        "RETURN u, count(f) AS n ORDER BY u")
    assert {rec["u"]: rec["n"] for rec in r.records} == {
        "f-sx": 1, "f-lx": 1, "f-sl": 1, "f-ys": 1, "f-xl": 1, "f-ls": 1, "f-ll": 1}


async def test_in_community_is_merged_and_the_community_flagged_stale(neo4j_driver):
    """S already belonged to `c-both`: MERGE keeps ONE membership edge. Both
    communities the loser belonged to are flagged `stale` (what the
    incremental theme-build treats as dirty); the unrelated one is not.
    `cited_fact_uuids` still resolve to existing edges."""
    await _seed_realistic_group(neo4j_driver)
    await apply_merges(neo4j_driver, GROUP)
    r = await neo4j_driver.execute_query(
        "MATCH (c:Community) OPTIONAL MATCH (e:Entity)-[:IN_COMMUNITY]->(c) "
        "WITH c, collect(e.uuid) AS members "
        "RETURN c.uuid AS c, c.stale AS stale, members ORDER BY c")
    rows = {rec["c"]: (rec["stale"], sorted(rec["members"])) for rec in r.records}
    assert rows == {
        "c-both": (True, [_S]),
        "c-loser": (True, sorted([_X, _S])),
        "c-other": (False, sorted([_X, _Y])),
    }
    r = await neo4j_driver.execute_query(
        "MATCH (c:Community) UNWIND c.cited_fact_uuids AS u "
        "OPTIONAL MATCH ()-[f:RELATES_TO {uuid:u}]->() RETURN u, count(f) AS n")
    assert {rec["u"]: rec["n"] for rec in r.records} == {
        "f-sx": 1, "f-lx": 1, "f-xl": 1, "f-ll": 1, "f-ys": 1}


async def test_same_as_is_rewired(neo4j_driver):
    """`p-loser` pointed only at L and now points at S; `p-both` pointed at
    both and keeps ONE edge (MERGE)."""
    await _seed_realistic_group(neo4j_driver)
    await apply_merges(neo4j_driver, GROUP)
    r = await neo4j_driver.execute_query(
        "MATCH (p:Product)-[:SAME_AS]->(e:Entity) RETURN p.id AS p, e.uuid AS e ORDER BY p")
    assert [tuple(rec.values()) for rec in r.records] == [("p-both", _S), ("p-loser", _S)]


async def test_label_promotion_only_onto_a_bare_survivor(neo4j_driver):
    """Bare `:Entity` survivor + `:Product` loser -> survivor gains `:Product`
    as a node label AND in the `labels` property (graphiti's
    `_promote_resolved_node` mirror). `:Tool` survivor + `:Product` loser ->
    unchanged in both places, and the planner's report names the conflict."""
    await _seed(neo4j_driver, uuid="p0", name="Promotable")
    await _seed(neo4j_driver, uuid="p1", name="Promotable", labels=("Product",),
                created_at="2026-02-01T00:00:00Z")
    await _seed(neo4j_driver, uuid="c0", name="Conflict", labels=("Tool",))
    await _seed(neo4j_driver, uuid="c1", name="Conflict", labels=("Product",),
                created_at="2026-02-01T00:00:00Z")
    res = await apply_merges(neo4j_driver, GROUP)
    p0 = await _entity(neo4j_driver, "p0")
    assert sorted(p0["labels"]) == ["Entity", "Product"]
    assert p0["props"]["labels"] == ["Entity", "Product"]
    c0 = await _entity(neo4j_driver, "c0")
    assert sorted(c0["labels"]) == ["Entity", "Tool"]
    assert c0["props"]["labels"] == ["Entity", "Tool"]
    assert res["label_conflicts"] == [
        {"name": "Conflict", "labels": [["Entity", "Tool"], ["Entity", "Product"]]}]
    assert res["labels_promoted"] == [{"name": "Promotable", "survivor": "p0", "labels": ["Product"]}]
    assert await _entity(neo4j_driver, "p1") is None
    assert await _entity(neo4j_driver, "c1") is None


async def test_an_empty_survivor_summary_takes_the_losers_and_a_full_one_keeps_its_own(neo4j_driver):
    """Deterministic and loses least: the loser's text is either adopted or
    printed in the audit -- never silently gone."""
    await _seed(neo4j_driver, uuid="e0", name="Empty", summary="")
    await _seed(neo4j_driver, uuid="e1", name="Empty", summary="the loser's words",
                created_at="2026-02-01T00:00:00Z")
    await _seed(neo4j_driver, uuid="k0", name="Kept", summary="the survivor's words")
    await _seed(neo4j_driver, uuid="k1", name="Kept", summary="the loser's other words",
                created_at="2026-02-01T00:00:00Z")
    res = await apply_merges(neo4j_driver, GROUP)
    assert (await _entity(neo4j_driver, "e0"))["props"]["summary"] == "the loser's words"
    assert (await _entity(neo4j_driver, "k0"))["props"]["summary"] == "the survivor's words"
    assert res["summary_dropped"] == [
        {"name": "Kept", "survivor": "k0", "loser": "k1", "summary": "the loser's other words"}]


async def test_the_survivor_is_untouched_except_for_the_audit_stamps(neo4j_driver):
    """uuid, name, group_id, created_at, labels (both), summary and
    name_embedding are byte-identical; `merged_from` lists the loser (and
    folds in the loser's own `merged_from` chain); the loser's custom
    property is NOT copied but IS listed by key in the audit."""
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=2, labels=("Product",),
                                summary="kept", name_embedding=[0.3, 0.2, 0.1])
    await neo4j_driver.execute_query(
        "MATCH (l:Entity {uuid:$l}) SET l.demoted_from_region = true, l.merged_from = ['ghost'], "
        "l.summary = 'not adopted'", l=_LOSER)
    before = await _entity(neo4j_driver, _S)
    res = await apply_merges(neo4j_driver, GROUP)
    after = await _entity(neo4j_driver, _S)
    assert sorted(after["labels"]) == sorted(before["labels"]) == ["Entity", "Product"]
    stamped = {k: v for k, v in after["props"].items() if k not in ("merged_from", "merged_at")}
    assert stamped == before["props"]
    assert after["props"]["merged_from"] == [_LOSER, "ghost"]
    assert after["props"]["merged_at"] is not None
    assert "demoted_from_region" not in after["props"]
    assert res["properties_dropped"] == [
        {"name": "AWS Backup", "loser": _LOSER, "keys": ["demoted_from_region"]}]
    assert res["summary_dropped"] == [
        {"name": "AWS Backup", "survivor": _S, "loser": _LOSER, "summary": "not adopted"}]


async def test_the_survivor_rule_holds_at_apply(neo4j_driver):
    """The earliest node survives even though it has the LARGEST uuid and the
    LOWEST degree; its edges and the loser's edges all end on it."""
    await _seed(neo4j_driver, uuid="zzz", name="AWS Backup",
                created_at="2026-01-01T00:00:00Z", degree=1)
    await _seed(neo4j_driver, uuid="aaa", name="AWS Backup",
                created_at="2026-06-01T00:00:00Z", degree=3)
    await apply_merges(neo4j_driver, GROUP)
    assert await _entity(neo4j_driver, "aaa") is None
    r = await neo4j_driver.execute_query(
        "MATCH (s:Entity {uuid:'zzz'})-[f:RELATES_TO]->() RETURN f.uuid AS uuid ORDER BY uuid")
    assert [rec["uuid"] for rec in r.records] == [
        "aaa-edge-0", "aaa-edge-1", "aaa-edge-2", "zzz-edge-0"]


async def test_the_same_name_in_another_group_is_untouched(neo4j_driver):
    """Invariant #4 scoping: the other group holds the SAME names AND the SAME
    uuids, and its loser carries one edge of every shape steps 1-5 match -- an
    outgoing fact, an incoming fact, a MENTIONS from an episode, an
    IN_COMMUNITY, a SAME_AS from a structural product -- so a `MATCH` in any
    step that dropped `group_id` has something to hit: steps 1-5 would move
    the other group's edge onto this group's survivor, step 0 would find two
    nodes per uuid and abort, step 6's guard would meet the other loser's
    edges and abort, its delete would be refused, and step 7 would stamp the
    other survivor. Every one of those was run as a mutant against this
    fixture and each is killed. The other group's snapshot (nodes and every
    edge touching them) must be byte-identical afterwards."""
    await _seed_realistic_group(neo4j_driver)
    for uuid, name in ((_S, "AWS Backup"), (_LOSER, "AWS Backup"), (_X, _X)):
        await _seed(neo4j_driver, uuid=uuid, name=name, group_id="other")
    await neo4j_driver.execute_query(
        "MATCH (l:Entity {uuid:$l, group_id:'other'}), (x:Entity {uuid:$x, group_id:'other'}) "
        "CREATE (l)-[:RELATES_TO {uuid:'other-f-lx', group_id:'other', fact:'o'}]->(x) "
        "CREATE (x)-[:RELATES_TO {uuid:'other-f-xl', group_id:'other', fact:'o'}]->(l) "
        "CREATE (:Episodic {uuid:'other-ep', group_id:'other', entity_edges:['other-f-lx']})"
        "-[:MENTIONS {uuid:'other-ep->l', group_id:'other'}]->(l) "
        "CREATE (l)-[:IN_COMMUNITY]->(:Community {uuid:'other-c', group_id:'other', stale:false}) "
        "CREATE (:Product {id:'other-p', name:'other-p'})-[:SAME_AS]->(l)",
        l=_LOSER, x=_X)
    other_before = await _snapshot_group(neo4j_driver, "other")
    assert len(other_before["rels"]) == 5, "the fixture must hold one edge per step 1-5 pattern"
    res = await apply_merges(neo4j_driver, GROUP)
    assert res["totals"]["merged"] == 1
    assert await _snapshot_group(neo4j_driver, "other") == other_before
    assert await count_duplicates(neo4j_driver, "other") == 1, "the other group is still theirs"


async def test_case_variants_are_not_merged_by_apply(neo4j_driver):
    await _seed(neo4j_driver, uuid="a", name="Access Control", degree=1)
    await _seed(neo4j_driver, uuid="b", name="Access control", degree=1)
    before = await _snapshot(neo4j_driver)
    res = await apply_merges(neo4j_driver, GROUP)
    assert res["totals"]["merged"] == 0
    assert res["near_duplicates_not_merged"] == [
        {"normalized": "access control", "names": ["Access Control", "Access control"]}]
    assert await _snapshot(neo4j_driver) == before


async def test_a_renamed_member_aborts_its_group_after_earlier_groups_merged(neo4j_driver, monkeypatch):
    """Step 0 re-verifies uuid, group AND name inside the transaction. `Zeta-1`
    is renamed between plan and apply: `Alpha` (earlier by name) merges,
    `Zeta` aborts naming itself, the run stops, and both Zeta nodes -- the
    renamed one with its edge -- are still there."""
    await _seed_duplicate_group(neo4j_driver, name="Alpha", n=2)
    await _seed_duplicate_group(neo4j_driver, name="Zeta", n=2, degree=1)
    real_plan = merge_duplicates.plan_merges

    async def plan_then_rename(driver, group_id):
        plan = await real_plan(driver, group_id)
        await driver.execute_query(
            "MATCH (e:Entity {uuid:'Zeta-1'}) SET e.name = 'Zeta (renamed)'")
        return plan

    monkeypatch.setattr(merge_duplicates, "plan_merges", plan_then_rename)
    with pytest.raises(MergeAborted, match="Zeta"):
        await apply_merges(neo4j_driver, GROUP)
    assert await _entity(neo4j_driver, "Alpha-1") is None
    assert (await _entity(neo4j_driver, "Alpha-0"))["props"]["merged_from"] == ["Alpha-1"]
    zeta0, zeta1 = await _entity(neo4j_driver, "Zeta-0"), await _entity(neo4j_driver, "Zeta-1")
    assert "merged_from" not in zeta0["props"]
    assert zeta1["props"]["name"] == "Zeta (renamed)"
    r = await neo4j_driver.execute_query(
        "MATCH (:Entity {uuid:'Zeta-1'})-[f:RELATES_TO]->() RETURN count(f) AS n")
    assert r.records[0]["n"] == 1, "the renamed node keeps its own edges"


async def test_merge_group_refuses_a_plan_whose_survivor_vanished(neo4j_driver):
    """`merge_group` on a stale plan entry: the survivor is gone, so nothing
    happens to the loser either."""
    await _seed_duplicate_group(neo4j_driver, name="AWS Backup", n=2, degree=1)
    group = (await plan_merges(neo4j_driver, GROUP))["groups"][0]
    await neo4j_driver.execute_query("MATCH (s:Entity {uuid:$s}) DETACH DELETE s", s=_S)
    before = await _snapshot(neo4j_driver)
    with pytest.raises(MergeAborted, match=_S):
        await merge_group(neo4j_driver, GROUP, group)
    assert await _snapshot(neo4j_driver) == before


# --- the production shape: three members --------------------------------------
# The live graph's duplicates are `AWS Backup` x3, `Amazon EC2` x3, `Azure
# Backup` x3. Two things exist only from three members up: a fact BETWEEN two
# losers (relocated once per loser), and two losers whose labels disagree.

_L1 = "AWS Backup-1"
_L2 = "AWS Backup-2"


async def _seed_three_member_group(driver):
    """S, L1, L2 (survivor order by `created_at`) and an outside node X, with
    a fact in every direction the per-loser loop must handle:

        S->X  L1->X  L2->X    meet after the merge (kept as three)
        X->L1  X->L2          incoming
        L1->L2  L2->L1        loser-to-loser: moved TWICE, end as S->S
        L1->S  S->L2  L2->L2  become S->S

    Nine facts are relocated, two of them twice: eleven move operations and
    five self-loops. One episode mentions all three and cites every fact.
    S has no summary; both losers do, and they differ."""
    await _seed_duplicate_group(driver, name="AWS Backup", n=3, name_embedding=[0.5, 0.5, 0.5])
    await driver.execute_query(
        "MATCH (l1:Entity {uuid:$l1}), (l2:Entity {uuid:$l2}) SET l1.summary = $s1, l2.summary = $s2",
        l1=_L1, l2=_L2, s1="the first loser's words", s2="the second loser's words")
    await _seed(driver, uuid=_X, name=_X)
    facts = [("f-sx", _S, _X), ("f-l1x", _L1, _X), ("f-l2x", _L2, _X),
             ("f-xl1", _X, _L1), ("f-xl2", _X, _L2),
             ("f-l1l2", _L1, _L2), ("f-l2l1", _L2, _L1),
             ("f-l1s", _L1, _S), ("f-sl2", _S, _L2), ("f-l2l2", _L2, _L2)]
    for i, (uuid, src, dst) in enumerate(facts):
        await _fact(driver, src=src, dst=dst, uuid=uuid, embedding=_vec(i), episodes=["ep1"],
                    valid_at=f"2025-0{i % 9 + 1}-01T00:00:00Z",
                    **({"invalid_at": "2025-12-01T00:00:00Z"} if i % 3 == 0 else {}))
    await _episode(driver, uuid="ep1", mentions=[_S, _L1, _L2, _X],
                   entity_edges=[uuid for uuid, _, _ in facts])
    return [uuid for uuid, _, _ in facts]


async def test_three_members_conserve_every_edge_and_count_moves_and_loops(neo4j_driver):
    """The conservation properties the two-member tests assert, at the
    production shape: every uuid and full property map survives, every fact
    ends on S or X with stored endpoints agreeing, `merged_from` accumulates
    both losers in survivor order, the FIRST loser with a non-empty summary
    lends it and the second's is audited, `self_loops_created` is exact, and
    `edges_moved.RELATES_TO` is the documented move-operation count (eleven,
    for nine distinct facts -- the two loser-to-loser facts move twice)."""
    facts = await _seed_three_member_group(neo4j_driver)
    before = {r["uuid"]: r for r in await _edge_snapshot(neo4j_driver)}
    res = await apply_merges(neo4j_driver, GROUP)
    json.dumps(res)
    after = {r["uuid"]: r for r in await _edge_snapshot(neo4j_driver)}
    assert set(before) == set(after), "every edge uuid must survive"
    for uuid, old in before.items():
        for key, value in old["props"].items():
            if key not in ("source_node_uuid", "target_node_uuid"):
                assert after[uuid]["props"][key] == value, f"{uuid}.{key} changed"
    assert {u: (r["a"], r["b"]) for u, r in after.items() if r["type"] == "RELATES_TO"} == {
        "f-sx": (_S, _X), "f-l1x": (_S, _X), "f-l2x": (_S, _X),
        "f-xl1": (_X, _S), "f-xl2": (_X, _S),
        "f-l1l2": (_S, _S), "f-l2l1": (_S, _S), "f-l1s": (_S, _S), "f-sl2": (_S, _S),
        "f-l2l2": (_S, _S)}
    r = await neo4j_driver.execute_query(
        "MATCH (a:Entity)-[f:RELATES_TO]->(b:Entity) "
        "WHERE f.source_node_uuid <> a.uuid OR f.target_node_uuid <> b.uuid RETURN count(f) AS n")
    assert r.records[0]["n"] == 0
    r = await neo4j_driver.execute_query(
        "MATCH (ep:Episodic) UNWIND ep.entity_edges AS u "
        "WITH DISTINCT u OPTIONAL MATCH ()-[f:RELATES_TO {uuid:u}]->() RETURN u, count(f) AS n")
    assert {rec["u"]: rec["n"] for rec in r.records} == dict.fromkeys(facts, 1)
    assert res["totals"] == {"groups": 1, "excess": 2, "merged": 2}
    assert res["self_loops_created"] == 5
    assert res["edges_moved"] == {"RELATES_TO": 11, "MENTIONS": 2, "IN_COMMUNITY": 0, "SAME_AS": 0}
    survivor = await _entity(neo4j_driver, _S)
    assert survivor["props"]["merged_from"] == [_L1, _L2]
    assert survivor["props"]["summary"] == "the first loser's words"
    assert res["summary_dropped"] == [
        {"name": "AWS Backup", "survivor": _S, "loser": _L2, "summary": "the second loser's words"}]
    assert res["labels_promoted"] == []
    assert await _entity(neo4j_driver, _L1) is None
    assert await _entity(neo4j_driver, _L2) is None
    assert await count_duplicates(neo4j_driver, GROUP) == 0


async def test_three_members_promote_labels_only_when_the_typed_losers_agree(neo4j_driver):
    """Spec §4.4: a bare survivor adopts a typed loser's labels; ANY other
    disagreement is dropped and reported under `label_conflicts`. Two losers
    disagreeing with each other is such a disagreement -- nothing is promoted
    (not the union, not the first), and the group is reported. A bare loser
    has no say. Invisible at two members: one loser has nobody to disagree
    with. Bare survivors throughout:

        Agree     :Product, :Product  -> :Product promoted
        Mixed     bare, :Product      -> :Product promoted
        Disagree  :Product, :Tool     -> nothing promoted, reported

    Run through `merge_group` directly, so the per-group payload carries the
    same `labels_promoted` element shape `apply_merges` reports."""
    for name, labels in (("Agree", ((), ("Product",), ("Product",))),
                         ("Mixed", ((), (), ("Product",))),
                         ("Disagree", ((), ("Product",), ("Tool",)))):
        for i, member_labels in enumerate(labels):
            await _seed(neo4j_driver, uuid=f"{name}-{i}", name=name, labels=member_labels,
                        created_at=f"2026-01-0{i + 1}T00:00:00Z")
    plan = await plan_merges(neo4j_driver, GROUP)
    assert plan["label_conflicts"] == [
        {"name": "Disagree", "labels": [["Entity"], ["Entity", "Product"], ["Entity", "Tool"]]}]
    outcomes = {g["name"]: await merge_group(neo4j_driver, GROUP, g) for g in plan["groups"]}
    assert outcomes["Agree"]["labels_promoted"] == [
        {"name": "Agree", "survivor": "Agree-0", "labels": ["Product"]}]
    assert outcomes["Mixed"]["labels_promoted"] == [
        {"name": "Mixed", "survivor": "Mixed-0", "labels": ["Product"]}]
    assert outcomes["Disagree"]["labels_promoted"] == []
    for name, expected in (("Agree", ["Entity", "Product"]), ("Mixed", ["Entity", "Product"]),
                           ("Disagree", ["Entity"])):
        survivor = await _entity(neo4j_driver, f"{name}-0")
        assert sorted(survivor["labels"]) == expected, name
        assert survivor["props"]["labels"] == expected, name
        assert survivor["props"]["merged_from"] == [f"{name}-1", f"{name}-2"]
    assert await count_duplicates(neo4j_driver, GROUP) == 0
