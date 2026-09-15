"""Exact-name duplicate planner (read-only half of the merge pass).

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
"""

import json

import pytest
import pytest_asyncio

from graph_extract.merge_duplicates import _GROUPS, _read, count_duplicates, plan_merges

pytestmark = pytest.mark.asyncio(loop_scope="module")

GROUP = "backup-docs"


@pytest_asyncio.fixture(loop_scope="module")
async def neo4j_driver(extract_driver):
    """The module-scoped testcontainer driver, wiped before each test -- the
    container is shared, and these tests all use the same group id. The uuid
    index one test creates is dropped here too, so a failure inside that test
    cannot leave it behind to change the scan order of the others."""
    await extract_driver.execute_query("MATCH (n) DETACH DELETE n")
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
    first, then largest uuid first -- the reverse of the survivor rule. Seeded
    in survivor-first insertion order AND with graphiti's `:Entity(uuid)` index
    present, so neither a label scan nor an index scan could produce this
    order on its own; only the ORDER BY can."""
    await neo4j_driver.execute_query(
        "CREATE INDEX entity_uuid IF NOT EXISTS FOR (n:Entity) ON (n.uuid)")
    await neo4j_driver.execute_query("CALL db.awaitIndexes(60)")
    await _seed(neo4j_driver, uuid="aaa", name="X", created_at="2026-01-01T00:00:00Z")
    await _seed(neo4j_driver, uuid="bbb", name="X", created_at="2026-01-01T00:00:00Z")
    await _seed(neo4j_driver, uuid="ccc", name="X", created_at="2026-06-01T00:00:00Z")
    rows = await _read(neo4j_driver, _GROUPS, group_id=GROUP)
    assert [m["uuid"] for m in rows[0]["members"]] == ["ccc", "bbb", "aaa"]
    plan = await plan_merges(neo4j_driver, GROUP)
    assert plan["groups"][0]["survivor"] == "aaa", "the Python sort stays authoritative"


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
