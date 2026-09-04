"""The scenario nobody has ever run: a real article, updated, through the pipeline.

Chunk 0 is edited and chunk 1 is dropped, then the temporal rule is asserted end to
end. Uses its own group_id so the pilot corpus is untouched.
"""
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.asyncio(loop_scope="module")

_ARTICLE_ID = "temporal-live-article"
_ARTICLE_URL = "https://example.invalid/temporal-live"
_GROUP = "temporal-live"

_V1_CHUNK0 = "Vault Lock enforces a minimum retention period of 7 days."
_V1_CHUNK1 = "Vault Lock is available in all commercial AWS Regions."
_V2_CHUNK0 = "Vault Lock enforces a minimum retention period of 30 days."


@pytest.mark.live
async def test_article_update_preserves_citations_and_expires_dropped_content():
    from neo4j import AsyncGraphDatabase

    from graph_extract.config import get_extract_settings
    from graph_extract.graphiti_client import add_text_episode, build_graphiti
    from graph_extract.ingest_driver import IngestDriver
    from graph_extract.provenance import Provenance
    from graph_extract.staleness_sweep import sweep_stale_facts

    settings = get_extract_settings().model_copy(update={"group_id": _GROUP})
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    graphiti = build_graphiti(settings)
    prov = Provenance(driver)
    # Only the two Cypher-only helpers are used, so the other deps are unused here.
    ing = IngestDriver(settings, None, None, None, prov, driver)
    now = datetime.now(timezone.utc)

    async def _ingest(chunk_index: int, body: str) -> str:
        added = await add_text_episode(
            graphiti, settings, name=f"{_ARTICLE_ID}-c{chunk_index}", body=body,
            source_description=_ARTICLE_URL, reference_time=now)
        await prov.link(_ARTICLE_ID, added.episode.uuid, chunk_index=chunk_index,
                        heading_path="Retention", token_count=len(body.split()),
                        content_hash=f"h-{chunk_index}-{len(body)}")
        return added.episode.uuid

    async def _facts_for(episode_uuid: str) -> list[str]:
        async with driver.session() as s:
            r = await s.run(
                "MATCH ()-[f:RELATES_TO {group_id:$g}]->() "
                "WHERE $u IN f.episodes RETURN f.uuid AS uuid", g=_GROUP, u=episode_uuid)
            return [rec["uuid"] async for rec in r]

    try:
        async with driver.session() as s:
            await s.run("MATCH (n {group_id:$g}) DETACH DELETE n", g=_GROUP)
            await s.run(
                "MERGE (a:Article {id:$a}) SET a.source_url=$u, a.removed=false",
                a=_ARTICLE_ID, u=_ARTICLE_URL)

        # --- v1: two chunks ---
        ep0_old = await _ingest(0, _V1_CHUNK0)
        ep1 = await _ingest(1, _V1_CHUNK1)
        await ing._supersede_trailing_episodes(_ARTICLE_ID, 2)
        old_facts = await _facts_for(ep0_old)
        dropped_facts = await _facts_for(ep1)
        assert old_facts, "v1 chunk 0 produced no facts; cannot test supersession"
        assert dropped_facts, "v1 chunk 1 produced no facts; cannot test expiry"

        # --- v2: chunk 0 edited, chunk 1 dropped ---
        ep0_new = await _ingest(0, _V2_CHUNK0)
        await ing._supersede_trailing_episodes(_ARTICLE_ID, 1)

        # 1 (defect A): the superseded episode KEEPS its edge, flagged
        async with driver.session() as s:
            rows = [rec async for rec in await s.run(
                "MATCH (:Article {id:$a})-[r:HAS_EPISODE]->(e:Episodic) "
                "RETURN e.uuid AS uuid, coalesce(r.superseded, false) AS sup",
                a=_ARTICLE_ID)]
        flags = {r["uuid"]: r["sup"] for r in rows}
        assert flags.get(ep0_old) is True, "defect A: superseded edge was deleted"
        assert flags.get(ep0_new) is False, "the replacement edge must be live"
        assert flags.get(ep1) is True, "defect B: dropped chunk edge not flagged"

        # 2 (defect A): its facts still resolve to the article URL
        resolved = await prov.resolve_citations(old_facts)
        sources = resolved.get(old_facts[0], {}).get("sources") or []
        assert sources, "defect A: superseded fact lost its citation"
        assert sources[0]["url"] == _ARTICLE_URL

        # 3 (defect B): the sweep expires the dropped chunk's facts
        await sweep_stale_facts(driver, _GROUP)
        async with driver.session() as s:
            r = await s.run(
                "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $u "
                "RETURN f.uuid AS uuid, f.invalid_at IS NOT NULL AS dead, "
                "coalesce(f.expired_by_sweep, false) AS swept",
                g=_GROUP, u=dropped_facts + old_facts)
            state = {rec["uuid"]: (rec["dead"], rec["swept"]) async for rec in r}
        for uuid in dropped_facts:
            assert state[uuid] == (True, True), f"defect B: {uuid} survived the sweep"

        # 4: the superseded chunk-0 facts are also dead (their only edge is flagged)
        for uuid in old_facts:
            assert state[uuid][0] is True, f"{uuid} should be dead after supersession"

        # 5: the NEW chunk-0 facts are alive
        new_facts = await _facts_for(ep0_new)
        assert new_facts, "v2 chunk 0 produced no facts"
        async with driver.session() as s:
            r = await s.run(
                "MATCH ()-[f:RELATES_TO {group_id:$g}]->() WHERE f.uuid IN $u "
                "RETURN count(f) AS dead", g=_GROUP,
                u=new_facts)
            # none of the new facts may be invalidated
            r2 = await s.run(
                "MATCH ()-[f:RELATES_TO {group_id:$g}]->() "
                "WHERE f.uuid IN $u AND f.invalid_at IS NOT NULL RETURN count(f) AS n",
                g=_GROUP, u=new_facts)
            assert [dict(x) async for x in r2][0]["n"] == 0
    finally:
        await graphiti.close()
        await driver.close()
