"""Persist the :Community subgraph (full rebuild: delete then write). Derived &
disposable — the theme-builder is the only writer of :Community."""
from __future__ import annotations

from neo4j import AsyncDriver

from theme_builder.detect import Community
from theme_builder.report import CommunityReport


async def write_communities(driver: AsyncDriver, embedder, group_id: str,
                            communities: list[Community],
                            reports: dict[str, CommunityReport],
                            corpus_cursor: str | None, *,
                            pending: dict[str, CommunityReport] | None = None) -> dict:
    # A cid in BOTH maps would be written twice: the verified row first (with an
    # embedding), then the staged `SET c +=` on the same node -- which leaves the
    # embedding in place, making a staged report retrievable. Staging's whole
    # guarantee is the ABSENT embedding, so make the caller's invariant explicit.
    assert not (set(reports) & set(pending or {})), (
        "a community cannot be both verified and staged: "
        f"{sorted(set(reports) & set(pending or {}))}")
    written = [c for c in communities if c.community_id in reports]
    texts = [f"{reports[c.community_id].title}\n{reports[c.community_id].summary}" for c in written]
    # graphiti's OpenAIEmbedder.create returns ONE vector; create_batch returns
    # one per input (list[list[float]]) — use it for the batch of community texts.
    embeddings = await embedder.create_batch(texts) if texts else []
    by_level: dict[int, int] = {}
    for c in written:
        by_level[c.level] = by_level.get(c.level, 0) + 1

    # A community that is neither written (no report) nor staged (pending) is
    # genuinely dropped by the DETACH DELETE rebuild below -- gone from the graph,
    # not just unwritten this run. by_level only ever counted the survivors, which
    # is how level 1 lost a 219-entity community for two slices without anyone
    # noticing: reports_skipped moved, but nothing said WHICH level absorbed it,
    # and global search reads level 1 only. Keyed by level so an operator can see
    # a retrieval-critical level draining without cross-referencing report logs.
    pending_ids = set(pending or {})
    lost_by_level: dict[int, int] = {}
    for c in communities:
        if c.community_id not in reports and c.community_id not in pending_ids:
            lost_by_level[c.level] = lost_by_level.get(c.level, 0) + 1

    async def _rebuild(tx):
        # atomic full rebuild: old :Community subgraph deleted and the new one
        # written in a single transaction, so a mid-run failure rolls back to the
        # previous good layer instead of leaving a truncated one.
        await tx.run("MATCH (c:Community {group_id:$g}) DETACH DELETE c", g=group_id)
        for c, emb in zip(written, embeddings):
            r = reports[c.community_id]
            await tx.run(
                "MERGE (c:Community {community_id:$cid, group_id:$g}) "
                "SET c += {level:$level, title:$title, summary:$summary, "
                "full_report:$full_report, rating:$rating, rating_explanation:$re, "
                "tags:$tags, cited_fact_uuids:$cited, embedding:$emb, verified:true, "
                "member_count:$mc, generated_at:datetime(), corpus_cursor:$cur}",
                cid=c.community_id, g=group_id, level=c.level, title=r.title,
                summary=r.summary, full_report=r.full_report, rating=r.rating,
                re=r.rating_explanation, tags=r.tags, cited=r.cited_fact_uuids,
                emb=emb, mc=len(c.member_uuids), cur=corpus_cursor)
            await tx.run(
                "MATCH (c:Community {community_id:$cid, group_id:$g}) "
                "UNWIND $members AS mu MATCH (e:Entity {uuid:mu, group_id:$g}) "
                "MERGE (e)-[:IN_COMMUNITY]->(c)",
                cid=c.community_id, g=group_id, members=c.member_uuids)
        for c in (communities if pending else []):
            pr = (pending or {}).get(c.community_id)
            if pr is None:
                continue
            # Staged: kept so a transient verifier outage does not cost a full
            # regeneration, but written WITHOUT an embedding. shortlist_communities
            # drops rows with no embedding, and DRIFT sources its ids from that same
            # shortlist, so this is unreachable from every answering path by
            # construction -- not by a flag someone could forget to check.
            await tx.run(
                "MERGE (c:Community {community_id:$cid, group_id:$g}) "
                "SET c += {level:$level, title:$title, pending_summary:$summary, "
                "pending_full_report:$full_report, rating:$rating, "
                "rating_explanation:$re, tags:$tags, cited_fact_uuids:$cited, "
                "verified:false, member_count:$mc, generated_at:datetime(), "
                "corpus_cursor:$cur}",
                cid=c.community_id, g=group_id, level=c.level, title=pr.title,
                summary=pr.summary, full_report=pr.full_report, rating=pr.rating,
                re=pr.rating_explanation, tags=pr.tags, cited=pr.cited_fact_uuids,
                mc=len(c.member_uuids), cur=corpus_cursor)
            await tx.run(
                "MATCH (c:Community {community_id:$cid, group_id:$g}) "
                "UNWIND $members AS mu MATCH (e:Entity {uuid:mu, group_id:$g}) "
                "MERGE (e)-[:IN_COMMUNITY]->(c)",
                cid=c.community_id, g=group_id, members=c.member_uuids)
        for c in written:
            if c.parent_id and c.parent_id in reports:
                await tx.run(
                    "MATCH (p:Community {community_id:$pid, group_id:$g}), "
                    "(c:Community {community_id:$cid, group_id:$g}) "
                    "MERGE (p)-[:PARENT_OF]->(c)",
                    pid=c.parent_id, cid=c.community_id, g=group_id)

    async with driver.session() as s:
        await s.execute_write(_rebuild)
    return {"reports_written": len(written), "by_level": by_level,
            "lost_by_level": lost_by_level,
            "facts_cited": sum(len(reports[c.community_id].cited_fact_uuids) for c in written),
            "reports_staged": len(pending or {})}


async def write_communities_incremental(driver: AsyncDriver, group_id: str,
                                        entries: list[dict], *,
                                        corpus_cursor: str | None) -> dict:
    """Atomic rewrite of the :Community layer from complete entries (each carrying
    its stable id, members, report fields, embedding, and generated_at). Clean
    entries carry their prior embedding/generated_at; dirty ones carry fresh values.
    Communities absent from `entries` are dropped (dissolved).

    An entry marked `staged: True` is one whose verification could NOT be completed.
    It is written with pending_summary / pending_full_report, verified=false and NO
    embedding, exactly as write_communities' `pending` path does -- present in the
    graph (so the rebuild's DETACH DELETE does not destroy it) but unreachable from
    shortlist_communities, and therefore from DRIFT, by construction. Staging it
    here is what keeps the DEFAULT incremental path from losing a community to a
    transient verifier blip.

    An entry marked `stale: True` is a previously verified report carried over
    because a NEW attempt to regenerate it failed (BACKLOG 5c). It is written as a
    normal, retrievable report -- embedding and all, because it was verified and
    losing it from retrieval would be the very harm being avoided -- but the flag
    makes `incremental.classify` treat it as dirty on the next run, so it is
    retried rather than silently accepted as current.

    No lost_by_level here (unlike write_communities): `entries` only holds the
    survivors -- clean/reused, regenerated, preserved and staged rows -- so a
    community the CLI genuinely lost (no report, nothing staged, nothing
    persisted) never shows up in `entries` at all, not even its level. It is the
    caller, theme_builder.cli._run_theme_build_incremental, that accumulates
    `lost_by_level` from its own loop and merges it into the result."""
    by_level: dict[int, int] = {}
    for e in entries:
        by_level[e["level"]] = by_level.get(e["level"], 0) + 1
    staged = [e for e in entries if e.get("staged")]

    async def _rebuild(tx):
        await tx.run("MATCH (c:Community {group_id:$g}) DETACH DELETE c", g=group_id)
        for e in entries:
            if e.get("staged"):
                await tx.run(
                    "MERGE (c:Community {community_id:$cid, group_id:$g}) "
                    "SET c += {level:$level, title:$title, pending_summary:$summary, "
                    "pending_full_report:$full_report, rating:$rating, "
                    "rating_explanation:$re, tags:$tags, cited_fact_uuids:$cited, "
                    "verified:false, member_count:$mc, generated_at:$ga, "
                    "corpus_cursor:$cur}",
                    cid=e["community_id"], g=group_id, level=e["level"], title=e["title"],
                    summary=e["pending_summary"], full_report=e["pending_full_report"],
                    rating=e["rating"], re=e["rating_explanation"], tags=e["tags"],
                    cited=e["cited_fact_uuids"], mc=len(e["member_uuids"]),
                    ga=e["generated_at"], cur=corpus_cursor)
            else:
                await tx.run(
                    "MERGE (c:Community {community_id:$cid, group_id:$g}) "
                    "SET c += {level:$level, title:$title, summary:$summary, "
                    "full_report:$full_report, rating:$rating, rating_explanation:$re, "
                    "tags:$tags, cited_fact_uuids:$cited, embedding:$emb, verified:true, "
                    "stale:$stale, member_count:$mc, generated_at:$ga, corpus_cursor:$cur}",
                    cid=e["community_id"], g=group_id, level=e["level"], title=e["title"],
                    summary=e["summary"], full_report=e["full_report"], rating=e["rating"],
                    re=e["rating_explanation"], tags=e["tags"], cited=e["cited_fact_uuids"],
                    emb=e["embedding"], stale=bool(e.get("stale")),
                    mc=len(e["member_uuids"]), ga=e["generated_at"],
                    cur=corpus_cursor)
            await tx.run(
                "MATCH (c:Community {community_id:$cid, group_id:$g}) "
                "UNWIND $members AS mu MATCH (ent:Entity {uuid:mu, group_id:$g}) "
                "MERGE (ent)-[:IN_COMMUNITY]->(c)",
                cid=e["community_id"], g=group_id, members=e["member_uuids"])
        for e in entries:
            if e.get("parent_id"):
                await tx.run(
                    "MATCH (p:Community {community_id:$pid, group_id:$g}), "
                    "(c:Community {community_id:$cid, group_id:$g}) "
                    "MERGE (p)-[:PARENT_OF]->(c)",
                    pid=e["parent_id"], cid=e["community_id"], g=group_id)

    async with driver.session() as s:
        await s.execute_write(_rebuild)
    return {"reports_written": len(entries) - len(staged), "by_level": by_level,
            "reports_staged": len(staged)}
