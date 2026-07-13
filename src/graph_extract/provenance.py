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
                "WHERE r.chunk_index=$i AND r.content_hash=$h RETURN r LIMIT 1",
                a=article_id, i=chunk_index, h=content_hash)
            return await r.single() is not None

    async def link(self, article_id: str, episode_uuid: str, *, chunk_index: int,
                   heading_path: str, token_count: int, content_hash: str) -> None:
        async with self._driver.session() as s:
            await s.run(
                "MATCH (a:Article {id:$a}) "
                "MATCH (e:Episodic {uuid:$u}) "
                "MERGE (a)-[r:HAS_EPISODE {chunk_index:$i}]->(e) "
                "SET r.heading_path=$hp, r.token_count=$tc, r.content_hash=$h",
                a=article_id, u=episode_uuid, i=chunk_index, hp=heading_path,
                tc=token_count, h=content_hash)

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
