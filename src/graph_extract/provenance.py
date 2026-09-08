from __future__ import annotations
from neo4j import AsyncDriver

class Provenance:
    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    async def already_ingested(self, article_id: str, chunk_index: int,
                               content_hash: str) -> bool:
        async with self._driver.session() as s:
            r = await s.run(
                "MATCH (:Article {id:$a})-[r:HAS_EPISODE]->() "
                "WHERE r.chunk_index=$i AND r.content_hash=$h "
                "AND coalesce(r.superseded, false) = false RETURN r LIMIT 1",
                a=article_id, i=chunk_index, h=content_hash)
            return await r.single() is not None

    async def link(self, article_id: str, episode_uuid: str, *, chunk_index: int,
                   heading_path: str, token_count: int, content_hash: str) -> None:
        async with self._driver.session() as s:
            await s.run(
                # The old edge is FLAGGED, never deleted: its existence is what makes
                # the superseded fact citable (design decision #2 for history), while
                # the flag is what makes it dead for liveness (see episode_liveness).
                # The leading MATCH on $u means a non-existent episode uuid yields zero
                # rows and the whole statement is a no-op -- a pinned behaviour.
                "MATCH (a:Article {id:$a}) "
                "MATCH (e:Episodic {uuid:$u}) "
                "OPTIONAL MATCH (a)-[old:HAS_EPISODE {chunk_index:$i}]->(oldE:Episodic) "
                "WHERE oldE.uuid <> $u "
                "SET old.superseded = true "
                "WITH DISTINCT a, e "
                "MERGE (a)-[r:HAS_EPISODE {chunk_index:$i}]->(e) "
                "SET r.heading_path=$hp, r.token_count=$tc, r.content_hash=$h, "
                "    r.superseded = false",
                a=article_id, u=episode_uuid, i=chunk_index, hp=heading_path,
                tc=token_count, h=content_hash)

    async def resolve_citations(self, fact_uuids: list[str]) -> dict[str, dict]:
        async with self._driver.session() as s:
            r = await s.run(
                "MATCH ()-[f:RELATES_TO]->() WHERE f.uuid IN $uuids "
                "OPTIONAL MATCH (a:Article)-[he:HAS_EPISODE]->(e:Episodic) "
                "  WHERE e.uuid IN f.episodes "
                "OPTIONAL MATCH (v:Vendor)-[:HAS_PRODUCT]->(p:Product)-[:HAS_SOURCE]->"
                "  (:Source)-[:HAS_ARTICLE]->(a) "
                "WITH f.uuid AS uuid, toString(f.valid_at) AS valid_at, "
                "     toString(f.invalid_at) AS invalid_at, "
                "     collect(DISTINCT CASE WHEN a IS NULL THEN NULL ELSE "
                "       {url:a.source_url, title:a.title, article_id:a.id, "
                "        section:he.heading_path, vendor:v.name, product:p.name} END) AS raw "
                "RETURN uuid, valid_at, invalid_at, [x IN raw WHERE x IS NOT NULL] AS sources",
                uuids=fact_uuids)
            return {rec["uuid"]: {"valid_at": rec["valid_at"],
                                  "invalid_at": rec["invalid_at"],
                                  "sources": rec["sources"]} async for rec in r}

    async def resolve_chain(self, fact_edge_uuid: str) -> list[dict]:
        async with self._driver.session() as s:
            r = await s.run(
                "MATCH ()-[f:RELATES_TO {uuid:$u}]-() "
                "WITH f, f.episodes AS eps "
                "UNWIND eps AS epu "
                "MATCH (a:Article)-[:HAS_EPISODE]->(e:Episodic {uuid:epu}) "
                "RETURN DISTINCT a.source_url AS url, a.title AS title, "
                "a.id AS article_id, f.fact AS fact", u=fact_edge_uuid)
            return [dict(rec) async for rec in r]
