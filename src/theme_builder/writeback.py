"""Persist the :Community subgraph (full rebuild: delete then write). Derived &
disposable — the theme-builder is the only writer of :Community."""
from __future__ import annotations

from neo4j import AsyncDriver

from theme_builder.detect import Community
from theme_builder.report import CommunityReport


async def write_communities(driver: AsyncDriver, embedder, group_id: str,
                            communities: list[Community],
                            reports: dict[str, CommunityReport],
                            corpus_cursor: str | None) -> dict:
    written = [c for c in communities if c.community_id in reports]
    texts = [f"{reports[c.community_id].title}\n{reports[c.community_id].summary}" for c in written]
    # graphiti's OpenAIEmbedder.create returns ONE vector; create_batch returns
    # one per input (list[list[float]]) — use it for the batch of community texts.
    embeddings = await embedder.create_batch(texts) if texts else []
    by_level: dict[int, int] = {}
    async with driver.session() as s:
        await s.run("MATCH (c:Community {group_id:$g}) DETACH DELETE c", g=group_id)
        for c, emb in zip(written, embeddings):
            r = reports[c.community_id]
            by_level[c.level] = by_level.get(c.level, 0) + 1
            await s.run(
                "MERGE (c:Community {community_id:$cid, group_id:$g}) "
                "SET c += {level:$level, title:$title, summary:$summary, "
                "full_report:$full_report, rating:$rating, rating_explanation:$re, "
                "tags:$tags, cited_fact_uuids:$cited, embedding:$emb, "
                "member_count:$mc, generated_at:datetime(), corpus_cursor:$cur}",
                cid=c.community_id, g=group_id, level=c.level, title=r.title,
                summary=r.summary, full_report=r.full_report, rating=r.rating,
                re=r.rating_explanation, tags=r.tags, cited=r.cited_fact_uuids,
                emb=emb, mc=len(c.member_uuids), cur=corpus_cursor)
            await s.run(
                "MATCH (c:Community {community_id:$cid, group_id:$g}) "
                "UNWIND $members AS mu MATCH (e:Entity {uuid:mu, group_id:$g}) "
                "MERGE (e)-[:IN_COMMUNITY]->(c)",
                cid=c.community_id, g=group_id, members=c.member_uuids)
        # parent edges after all community nodes exist
        for c in written:
            if c.parent_id and c.parent_id in reports:
                await s.run(
                    "MATCH (p:Community {community_id:$pid, group_id:$g}), "
                    "(c:Community {community_id:$cid, group_id:$g}) "
                    "MERGE (p)-[:PARENT_OF]->(c)",
                    pid=c.parent_id, cid=c.community_id, g=group_id)
    return {"reports_written": len(written), "by_level": by_level,
            "facts_cited": sum(len(reports[c.community_id].cited_fact_uuids) for c in written)}
