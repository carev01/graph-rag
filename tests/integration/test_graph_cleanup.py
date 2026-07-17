import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_prune_removes_noise_keeps_real(extract_driver):
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity {name:'arn:aws:ec2:us-east-1::snapshot/snap-1', group_id:'g'})")
        await s.run("CREATE (:Entity {name:'CreatorRequestId', group_id:'g'})")
        await s.run("CREATE (:Entity {name:'immutability', group_id:'g'})")
        await s.run("CREATE (:Entity {name:'Amazon S3', group_id:'g'})")
        # a fact edge on a noise node -> must go with it
        await s.run("MATCH (a:Entity {name:'arn:aws:ec2:us-east-1::snapshot/snap-1'}), "
                    "(b:Entity {name:'immutability'}) "
                    "CREATE (a)-[:RELATES_TO {group_id:'g'}]->(b)")
    from graph_extract.graph_cleanup import prune_noise_entities
    res = await prune_noise_entities(extract_driver, "g")
    assert res["pruned"] == 2
    async with extract_driver.session() as s:
        r = await s.run("MATCH (e:Entity {group_id:'g'}) RETURN count(e) AS n")
        assert (await r.single())["n"] == 2                 # only the 2 real kept
        r = await s.run("MATCH ()-[x:RELATES_TO {group_id:'g'}]->() RETURN count(x) AS n")
        assert (await r.single())["n"] == 0                 # noise fact gone


async def test_retype_region_entities(extract_driver):
    from graph_extract.graph_cleanup import retype_region_entities
    g = "rgn"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity:Platform {group_id:$g, name:'Germany West Central'})", g=g)
        await s.run("CREATE (:Entity:Workload {group_id:$g, name:'Amazon S3'})", g=g)
        await s.run("CREATE (:Entity:Region {group_id:$g, name:'East US'})", g=g)
    res = await retype_region_entities(extract_driver, g)
    assert res["retyped"] == 1 and res["retyped_names"] == ["Germany West Central"]
    assert res["demoted"] == 0 and res["demote_guard_tripped"] is False
    async with extract_driver.session() as s:
        lbls = {r["name"]: set(r["l"]) async for r in await s.run(
            "MATCH (e:Entity {group_id:$g}) RETURN e.name AS name, labels(e) AS l", g=g)}
    assert "Region" in lbls["Germany West Central"] and "Platform" not in lbls["Germany West Central"]
    assert lbls["Amazon S3"] == {"Entity", "Workload"}
    assert "Region" in lbls["East US"]


async def _labels_by_name(driver, g):
    async with driver.session() as s:
        return {r["name"]: set(r["l"]) async for r in await s.run(
            "MATCH (e:Entity {group_id:$g}) RETURN e.name AS name, labels(e) AS l", g=g)}


async def test_demotes_selftyped_nonregion_to_bare_entity(extract_driver):
    """A :Region the model self-typed but the gazetteer doesn't recognise is
    demoted to a bare :Entity + audit stamp; its fact edge survives (no delete)."""
    from graph_extract.graph_cleanup import retype_region_entities
    g = "dem-core"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity:Region {group_id:$g, name:'Availability Zone'})", g=g)
        await s.run("CREATE (:Entity {group_id:$g, name:'AWS Backup'})", g=g)
        await s.run("MATCH (a:Entity {name:'Availability Zone', group_id:$g}),"
                    "(b:Entity {name:'AWS Backup', group_id:$g}) "
                    "CREATE (a)-[:RELATES_TO {group_id:$g}]->(b)", g=g)
    res = await retype_region_entities(extract_driver, g)
    assert res["demoted"] == 1 and res["demoted_names"] == ["Availability Zone"]
    lbls = await _labels_by_name(extract_driver, g)
    assert lbls["Availability Zone"] == {"Entity"}          # :Region gone, bare Entity
    async with extract_driver.session() as s:
        rec = await (await s.run(
            "MATCH (e:Entity {name:'Availability Zone', group_id:$g}) "
            "RETURN e.demoted_from_region AS d, e.demoted_at IS NOT NULL AS t", g=g)).single()
        assert rec["d"] is True and rec["t"] is True
        n = (await (await s.run(
            "MATCH ()-[x:RELATES_TO {group_id:$g}]->() RETURN count(x) AS n", g=g)).single())["n"]
        assert n == 1                                        # fact edge preserved


async def test_genuine_regions_untouched_by_demote(extract_driver):
    from graph_extract.graph_cleanup import retype_region_entities
    g = "dem-genuine"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity:Region {group_id:$g, name:'us-east-1'})", g=g)
        await s.run("CREATE (:Entity:Region {group_id:$g, name:'Africa (Cape Town)'})", g=g)
    res = await retype_region_entities(extract_driver, g)
    assert res["demoted"] == 0
    lbls = await _labels_by_name(extract_driver, g)
    assert "Region" in lbls["us-east-1"] and "Region" in lbls["Africa (Cape Town)"]


async def test_promote_and_demote_in_one_pass(extract_driver):
    from graph_extract.graph_cleanup import retype_region_entities
    g = "dem-both"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity:Platform {group_id:$g, name:'Poland Central'})", g=g)  # promote
        await s.run("CREATE (:Entity:Region {group_id:$g, name:'subscriptions'})", g=g)     # demote
    res = await retype_region_entities(extract_driver, g)
    assert res["retyped"] == 1 and res["retyped_names"] == ["Poland Central"]
    assert res["demoted"] == 1 and res["demoted_names"] == ["subscriptions"]
    lbls = await _labels_by_name(extract_driver, g)
    assert lbls["Poland Central"] == {"Entity", "Region"}
    assert lbls["subscriptions"] == {"Entity"}


async def test_demote_idempotent(extract_driver):
    from graph_extract.graph_cleanup import retype_region_entities
    g = "dem-idem"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity:Region {group_id:$g, name:'Protected resources'})", g=g)
    first = await retype_region_entities(extract_driver, g)
    second = await retype_region_entities(extract_driver, g)
    assert first["demoted"] == 1 and second["demoted"] == 0
    lbls = await _labels_by_name(extract_driver, g)
    assert lbls["Protected resources"] == {"Entity"}


async def test_repromotion_clears_audit_stamp(extract_driver, monkeypatch):
    """Self-healing: a demoted name later added to the gazetteer is re-promoted
    to :Region and its demote audit stamp is cleared."""
    import graph_extract.graph_cleanup as gc
    from graph_extract.graph_cleanup import retype_region_entities
    g = "dem-heal"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Entity:Region {group_id:$g, name:'Nowhere Region 9'})", g=g)
    await retype_region_entities(extract_driver, g)          # demoted (gazetteer says no)
    lbls = await _labels_by_name(extract_driver, g)
    assert lbls["Nowhere Region 9"] == {"Entity"}
    monkeypatch.setattr(gc, "is_region", lambda name: name == "Nowhere Region 9")
    res = await retype_region_entities(extract_driver, g)    # now recognised -> re-promote
    assert res["retyped"] == 1
    lbls = await _labels_by_name(extract_driver, g)
    assert lbls["Nowhere Region 9"] == {"Entity", "Region"}
    async with extract_driver.session() as s:
        rec = await (await s.run(
            "MATCH (e:Entity {name:'Nowhere Region 9', group_id:$g}) "
            "RETURN e.demoted_from_region AS d", g=g)).single()
        assert rec["d"] is None                              # stamp cleared


async def test_demote_preserves_other_custom_labels(extract_driver):
    from graph_extract.graph_cleanup import retype_region_entities
    g = "dem-multi"
    async with extract_driver.session() as s:
        # a node the model over-labelled :Region AND :Concept; name isn't a region
        await s.run("CREATE (:Entity:Region:Concept {group_id:$g, name:'private IP address'})", g=g)
    res = await retype_region_entities(extract_driver, g)
    assert res["demoted"] == 1
    lbls = await _labels_by_name(extract_driver, g)
    assert lbls["private IP address"] == {"Entity", "Concept"}   # only :Region removed


async def test_demote_guard_trips_on_mass_demotion(extract_driver):
    """If demotions would exceed the guard threshold (a sign is_region regressed),
    skip ALL demotions, still promote, and report the tripped guard."""
    from graph_extract.graph_cleanup import retype_region_entities
    g = "dem-guard"
    async with extract_driver.session() as s:
        # 3 genuine + 11 junk :Region -> 11 > max(2, ceil(0.5*14)=7) -> trips
        for n in ("us-east-1", "East US", "Japan East"):
            await s.run("CREATE (:Entity:Region {group_id:$g, name:$n})", g=g, n=n)
        for i in range(11):
            await s.run("CREATE (:Entity:Region {group_id:$g, name:$n})", g=g, n=f"junk-{i}")
    res = await retype_region_entities(extract_driver, g)
    assert res["demote_guard_tripped"] is True
    assert res["demoted"] == 0 and len(res["demote_skipped_names"]) == 11
    lbls = await _labels_by_name(extract_driver, g)
    assert lbls["junk-0"] == {"Entity", "Region"}           # nothing demoted


async def test_demote_guard_boundary_at_limit_does_not_trip(extract_driver):
    """Exactly `guard_limit` demotions must NOT trip (guards on '>', not '>=')."""
    from graph_extract.graph_cleanup import retype_region_entities
    g = "dem-boundary"
    async with extract_driver.session() as s:
        # 5 genuine + 5 junk = 10 :Region -> limit=max(2, ceil(0.5*10)=5)=5;
        # demote==5 is NOT > 5 -> proceeds.
        for n in ("us-east-1", "East US", "Japan East", "Korea Central", "India West"):
            await s.run("CREATE (:Entity:Region {group_id:$g, name:$n})", g=g, n=n)
        for i in range(5):
            await s.run("CREATE (:Entity:Region {group_id:$g, name:$n})", g=g, n=f"jk-{i}")
    res = await retype_region_entities(extract_driver, g)
    assert res["demote_guard_tripped"] is False
    assert res["demoted"] == 5


async def test_demote_force_bypasses_guard(extract_driver):
    """force=True demotes a legitimate large batch the guard would otherwise block."""
    from graph_extract.graph_cleanup import retype_region_entities
    g = "dem-force"
    async with extract_driver.session() as s:
        for n in ("us-east-1", "East US"):
            await s.run("CREATE (:Entity:Region {group_id:$g, name:$n})", g=g, n=n)
        for i in range(11):
            await s.run("CREATE (:Entity:Region {group_id:$g, name:$n})", g=g, n=f"jf-{i}")
    res = await retype_region_entities(extract_driver, g, force=True)
    assert res["demote_guard_tripped"] is False and res["demoted"] == 11
    lbls = await _labels_by_name(extract_driver, g)
    assert lbls["jf-0"] == {"Entity"}                       # demoted despite volume


async def test_demote_guard_catches_isregion_regression_on_small_graph(
        extract_driver, monkeypatch):
    """The guard must protect a SMALL graph too: a total is_region regression
    (rejecting every genuine region) trips even with only 4 :Region nodes --
    the case a fixed floor of 10 could never catch."""
    import graph_extract.graph_cleanup as gc
    from graph_extract.graph_cleanup import retype_region_entities
    g = "dem-small-regress"
    async with extract_driver.session() as s:
        for n in ("us-east-1", "East US", "Japan East", "Korea Central"):
            await s.run("CREATE (:Entity:Region {group_id:$g, name:$n})", g=g, n=n)
    monkeypatch.setattr(gc, "is_region", lambda name: False)   # regression
    res = await retype_region_entities(extract_driver, g)
    # 4 demote candidates > max(2, ceil(0.5*4)=2)=2 -> trips, nothing stripped
    assert res["demote_guard_tripped"] is True and res["demoted"] == 0
    lbls = await _labels_by_name(extract_driver, g)
    assert lbls["us-east-1"] == {"Entity", "Region"}           # genuine region safe
