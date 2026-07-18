import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

G = "backup-docs"


async def test_touched_entities_none_cursor_returns_none(extract_driver):
    from theme_builder.incremental import touched_entities
    out = await touched_entities(extract_driver, G, None)
    assert out is None


async def test_touched_entities_from_created_at(extract_driver):
    from theme_builder.incremental import touched_entities
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        # e_new created after cursor; e_old before; a new fact touches e_a and e_b
        await s.run("CREATE (:Entity {group_id:$g, uuid:'e_new', created_at: datetime('2026-02-01')})", g=G)
        await s.run("CREATE (:Entity {group_id:$g, uuid:'e_old', created_at: datetime('2026-01-01')})", g=G)
        await s.run("CREATE (a:Entity {group_id:$g, uuid:'e_a', created_at: datetime('2026-01-01')}), "
                    "(b:Entity {group_id:$g, uuid:'e_b', created_at: datetime('2026-01-01')}) "
                    "CREATE (a)-[:RELATES_TO {group_id:$g, uuid:'f1', created_at: datetime('2026-02-01')}]->(b)", g=G)
    out = await touched_entities(extract_driver, G, "2026-01-15T00:00:00Z")
    assert out == {"e_new", "e_a", "e_b"}      # e_old excluded; fact endpoints included


async def test_load_persisted_and_prev_cursor(extract_driver):
    from theme_builder.incremental import load_persisted, prev_corpus_cursor
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        await s.run("CREATE (c:Community {group_id:$g, community_id:'s1', level:1, title:'T', "
                    "summary:'S', full_report:'[]', rating:7.0, rating_explanation:'why', "
                    "tags:['x'], cited_fact_uuids:['fa'], embedding:[1.0,2.0], "
                    "generated_at: datetime('2026-03-01'), corpus_cursor:'2026-03-01T00:00:00Z', member_count:2}) "
                    "CREATE (e1:Entity {group_id:$g, uuid:'e1'})-[:IN_COMMUNITY]->(c) "
                    "CREATE (e2:Entity {group_id:$g, uuid:'e2'})-[:IN_COMMUNITY]->(c)", g=G)
    persisted = await load_persisted(extract_driver, G)
    assert len(persisted) == 1
    p = persisted[0]
    assert p.community_id == "s1" and p.level == 1
    assert p.members == {"e1", "e2"}
    assert p.title == "T" and p.rating == 7.0 and p.cited_fact_uuids == ["fa"]
    assert p.embedding == [1.0, 2.0]
    assert (await prev_corpus_cursor(extract_driver, G)) == "2026-03-01T00:00:00Z"


async def test_touched_includes_swept_fact_endpoints(extract_driver):
    from theme_builder.incremental import touched_entities
    g = "backup-docs"
    async with extract_driver.session() as s:
        await s.run("MATCH (n) DETACH DELETE n")
        # swept fact: expired_by_sweep + invalid_at AFTER cursor -> endpoints touched
        await s.run("CREATE (a:Entity {group_id:$g, uuid:'sa', created_at: datetime('2026-01-01')})"
                    "-[:RELATES_TO {group_id:$g, uuid:'fs', created_at: datetime('2026-01-01'), "
                    "expired_by_sweep: true, invalid_at: datetime('2026-02-01')}]->"
                    "(b:Entity {group_id:$g, uuid:'sb', created_at: datetime('2026-01-01')})", g=g)
        # graphiti-style invalidation: invalid_at after cursor but NO expired_by_sweep -> NOT touched
        await s.run("CREATE (a:Entity {group_id:$g, uuid:'ga', created_at: datetime('2026-01-01')})"
                    "-[:RELATES_TO {group_id:$g, uuid:'fg', created_at: datetime('2026-01-01'), "
                    "invalid_at: datetime('2026-02-01')}]->"
                    "(b:Entity {group_id:$g, uuid:'gb', created_at: datetime('2026-01-01')})", g=g)
        # swept fact but invalid_at BEFORE cursor -> already accounted -> NOT touched
        await s.run("CREATE (a:Entity {group_id:$g, uuid:'oa', created_at: datetime('2026-01-01')})"
                    "-[:RELATES_TO {group_id:$g, uuid:'fo', created_at: datetime('2026-01-01'), "
                    "expired_by_sweep: true, invalid_at: datetime('2026-01-10')}]->"
                    "(b:Entity {group_id:$g, uuid:'ob', created_at: datetime('2026-01-01')})", g=g)
    out = await touched_entities(extract_driver, g, "2026-01-15T00:00:00Z")
    assert out == {"sa", "sb"}
