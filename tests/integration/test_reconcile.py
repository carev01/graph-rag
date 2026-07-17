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
    assert "Vendor:Veeam" in res["unmatched_structural"]

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


async def test_reconcile_links_aliased_product(extract_driver):
    from graph_extract.reconcile import reconcile_same_as

    # Distinct name -- exercises the :Product code path (only :Vendor was
    # covered above). Exact normalized-name match, no alias needed.
    g = "rec-product"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Product {name:'DistinctProdX'})")
        await s.run(
            "CREATE (:Entity:Product {group_id:$g, name:'DistinctProdX'})", g=g
        )

    res = await reconcile_same_as(extract_driver, g)

    async with extract_driver.session() as s:
        n = (
            await (
                await s.run(
                    "MATCH (:Product {name:'DistinctProdX'})-[:SAME_AS]->"
                    "(e:Entity:Product {name:'DistinctProdX'}) RETURN count(*) AS c"
                )
            ).single()
        )["c"]
    assert n == 1
    assert res["linked"] >= 1


async def test_reconcile_unmatched_is_kind_aware(extract_driver):
    from graph_extract.reconcile import reconcile_same_as

    # Same name, two different structural kinds, neither has a semantic
    # twin -- without kind-qualifying, both would collapse into a single
    # "ZetaX" entry under sorted(set(unmatched)).
    g = "rec-kind-aware"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Vendor {name:'ZetaX'})")
        await s.run("CREATE (:Product {name:'ZetaX'})")

    res = await reconcile_same_as(extract_driver, g)

    assert "Vendor:ZetaX" in res["unmatched_structural"]
    assert "Product:ZetaX" in res["unmatched_structural"]


async def test_unmatched_no_candidate_classified(extract_driver):
    """A structural node whose aliases match NO semantic entity of any type is
    classified 'no_candidate' -- the expected, not-an-error case (e.g. a vendor
    the corpus never mentions as an actor)."""
    from graph_extract.reconcile import reconcile_same_as

    g = "rec-nocand"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Vendor {name:'OrphanVendorZZ'})")

    res = await reconcile_same_as(extract_driver, g)

    assert "Vendor:OrphanVendorZZ" in res["unmatched_structural"]
    assert res["unmatched_detail"]["Vendor:OrphanVendorZZ"] == {"reason": "no_candidate"}


async def test_unmatched_wrong_type_candidate_and_no_crosstype_link(extract_driver):
    """A structural :Vendor whose alias matches a semantic entity of a DIFFERENT
    kind (Microsoft's 'azure' form -> Azure typed :Platform) is classified
    'wrong_type_candidate' with the candidate + its labels, and NO cross-type
    SAME_AS is created (that would assert a false vendor==platform identity)."""
    from graph_extract.reconcile import reconcile_same_as

    g = "rec-wrongtype"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Vendor {name:'Microsoft'})")
        await s.run("CREATE (:Entity:Platform {group_id:$g, name:'Azure'})", g=g)

    res = await reconcile_same_as(extract_driver, g)

    detail = res["unmatched_detail"]["Vendor:Microsoft"]
    assert detail["reason"] == "wrong_type_candidate"
    cand = {c["name"]: set(c["labels"]) for c in detail["candidates"]}
    assert "Azure" in cand and cand["Azure"] == {"Entity", "Platform"}
    # read-only probe: no SAME_AS edge minted from Microsoft
    async with extract_driver.session() as s:
        n = (await (await s.run(
            "MATCH (:Vendor {name:'Microsoft'})-[:SAME_AS]->() RETURN count(*) AS c"
        )).single())["c"]
    assert n == 0


async def test_matched_structural_absent_from_detail(extract_driver):
    """A structural node that DID link is in neither unmatched list nor detail."""
    from graph_extract.reconcile import reconcile_same_as

    g = "rec-detail-matched"
    async with extract_driver.session() as s:
        await s.run("CREATE (:Product {name:'ReconDetailProd'})")
        await s.run("CREATE (:Entity:Product {group_id:$g, name:'ReconDetailProd'})", g=g)

    res = await reconcile_same_as(extract_driver, g)

    assert "Product:ReconDetailProd" not in res["unmatched_structural"]
    assert "Product:ReconDetailProd" not in res["unmatched_detail"]
