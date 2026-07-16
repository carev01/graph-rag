import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")


async def test_reconcile_links_aliased_vendor(extract_driver):
    from graph_extract.reconcile import reconcile_same_as

    g = "rec-links"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Vendor {name:'AWS'})")
        await s.run(
            "CREATE (:Entity:Vendor {group_id:$g, name:'Amazon Web Services'})", g=g
        )
        await s.run("CREATE (:Entity:Vendor {group_id:$g, name:'vaults'})", g=g)  # noise
        await s.run("CREATE (:Vendor {name:'Veeam'})")  # unmatched (no semantic twin)

    res = await reconcile_same_as(extract_driver, g)

    async with extract_driver.session() as s:
        n = (
            await (
                await s.run(
                    "MATCH (:Vendor {name:'AWS'})-[:SAME_AS]->"
                    "(e:Entity {name:'Amazon Web Services'}) RETURN count(*) AS c"
                )
            ).single()
        )["c"]
    assert n == 1
    assert res["linked"] >= 1
    assert "Veeam" in res["unmatched_structural"]

    # noise entity never gets a link
    async with extract_driver.session() as s:
        m = (
            await (
                await s.run(
                    "MATCH (:Vendor)-[:SAME_AS]->(e:Entity {name:'vaults'}) "
                    "RETURN count(*) AS c"
                )
            ).single()
        )["c"]
    assert m == 0


async def test_reconcile_is_idempotent(extract_driver):
    from graph_extract.reconcile import reconcile_same_as

    # Distinct names from test_reconcile_links_aliased_vendor above -- the
    # extract_driver fixture is module-scoped (shared Neo4j container across
    # this file's tests), and structural nodes carry no group_id, so a
    # structural-node scan is never filtered by group. Reusing 'AWS'/'Veeam'
    # here would let this test's reconcile call also touch the OTHER test's
    # leftover nodes, breaking the "exactly one SAME_AS" assertion below.
    g = "rec-idem"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Vendor {name:'IdemCorp'})")
        await s.run(
            "CREATE (:Entity:Vendor {group_id:$g, name:'IdemCorp'})", g=g
        )

    await reconcile_same_as(extract_driver, g)
    await reconcile_same_as(extract_driver, g)  # rerun -- MERGE must not duplicate

    async with extract_driver.session() as s:
        n = (
            await (
                await s.run(
                    "MATCH (:Vendor {name:'IdemCorp'})-[:SAME_AS]->"
                    "(e:Entity {name:'IdemCorp'}) RETURN count(*) AS c"
                )
            ).single()
        )["c"]
    assert n == 1
