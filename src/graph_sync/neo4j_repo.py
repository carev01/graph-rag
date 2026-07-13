from __future__ import annotations

from neo4j import AsyncGraphDatabase

from graph_sync.models import StructuralWrite, Tombstone, TocSnapshot

_CONSTRAINTS = [
    "CREATE CONSTRAINT vendor_id IF NOT EXISTS FOR (n:Vendor) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (n:Product) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (n:Article) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (n:Chapter) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX article_source IF NOT EXISTS FOR (n:Article) ON (n.source_id)",
    "CREATE INDEX chapter_source IF NOT EXISTS FOR (n:Chapter) ON (n.source_id)",
]

_APPLY_STRUCTURAL = """
MERGE (v:Vendor {id: $vendor.id}) SET v += $vendor
MERGE (p:Product {id: $product.id}) SET p += $product
MERGE (s:Source {id: $source.id}) SET s += $source
MERGE (a:Article {id: $article.id}) SET a += $article
MERGE (v)-[:HAS_PRODUCT]->(p)
MERGE (p)-[:HAS_SOURCE]->(s)
MERGE (s)-[:HAS_ARTICLE]->(a)
"""

# Deviates from the brief's single chained-WITH statement. Two Neo4j 5.22 Cypher
# semantics broke the brief's one-query design:
#
# 1. A `MATCH ... DELETE` (or `UNWIND $empty_list ...`) that matches/produces zero rows
#    collapses the row stream to zero rows. Any `WITH` immediately after it then carries
#    zero rows forward, so everything chained afterwards -- even clauses like a later
#    UNWIND that don't depend on the deleted/matched data -- silently never executes.
#    This bit the "clear old IN_CHAPTER links" step: on the very first `apply_toc` call
#    for a source there are no existing IN_CHAPTER rels to delete, so the brief's
#    `MATCH ...-[r:IN_CHAPTER]->() DELETE r WITH sid UNWIND $article_links ...` ran the
#    MATCH/DELETE, found 0 rows, and then never executed the UNWIND that creates the new
#    links -- the rewire silently no-op'd. Same hazard applies to `UNWIND $root_ids`
#    when root_ids could be empty followed by more work in the same WITH-chain.
# 2. `CALL { WITH sid ... }` subqueries need an explicit importing WITH and add extra
#    variable-scoping rules across the surrounding WITHs that make one giant statement
#    fragile and hard to reason about.
#
# Fix: use OPTIONAL MATCH where a "maybe zero rows" step must not gate later work, and
# split the TOC apply into four independent, idempotent `session.run()` statements
# instead of one giant WITH-chained query. Each statement is simple enough to not hit
# the row-collapse trap. Semantics (upsert chapters, wire HAS_CHAPTER nesting, rewire
# IN_CHAPTER, prune stale chapters) are unchanged from the brief's intent.
_UPSERT_CHAPTERS = """
UNWIND $chapters AS ch
MERGE (c:Chapter {id: ch.id})
SET c += ch
"""

_WIRE_ROOT_CHAPTERS = """
MATCH (s:Source {id: $source_id})
UNWIND $root_ids AS rid
MATCH (c:Chapter {id: rid})
MERGE (s)-[:HAS_CHAPTER]->(c)
"""

_WIRE_NESTING = """
UNWIND $nesting AS pair
MATCH (pc:Chapter {id: pair[0]}), (cc:Chapter {id: pair[1]})
MERGE (pc)-[:HAS_CHAPTER]->(cc)
"""

_REWIRE_ARTICLE_LINKS = """
OPTIONAL MATCH (a:Article {source_id: $source_id})-[r:IN_CHAPTER]->()
DELETE r
WITH DISTINCT 1 AS _
UNWIND $article_links AS link
MATCH (a:Article {id: link[0]}), (c:Chapter {id: link[1]})
MERGE (a)-[:IN_CHAPTER]->(c)
"""

_PRUNE_CHAPTERS = """
MATCH (old:Chapter {source_id: $source_id})
WHERE NOT old.id IN $keep
DETACH DELETE old
"""


class Neo4jRepo:
    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

    async def close(self) -> None:
        await self._driver.close()

    async def init_schema(self) -> None:
        async with self._driver.session() as sess:
            for stmt in _CONSTRAINTS:
                await sess.run(stmt)

    async def apply_structural(self, w: StructuralWrite) -> None:
        async with self._driver.session() as sess:
            await sess.run(
                _APPLY_STRUCTURAL,
                vendor=w.vendor,
                product=w.product,
                source=w.source,
                article=w.article,
            )

    async def get_content_hash(self, article_id: str) -> str | None:
        async with self._driver.session() as sess:
            r = await sess.run(
                "MATCH (a:Article {id:$id}) RETURN a.content_hash AS h", id=article_id
            )
            rec = await r.single()
            return rec["h"] if rec else None

    async def tombstone_article(self, t: Tombstone) -> None:
        async with self._driver.session() as sess:
            await sess.run(
                "MERGE (a:Article {id:$id}) SET a.removed=true, a.removed_at=$ts",
                id=t.article_id,
                ts=t.removed_at,
            )

    async def apply_toc(self, snap: TocSnapshot) -> None:
        chapters = [vars(c) for c in snap.chapters]
        nesting = [list(p) for p in snap.nesting]
        article_links = [list(p) for p in snap.article_links]
        keep = [c["id"] for c in chapters]
        async with self._driver.session() as sess:
            await sess.run(_UPSERT_CHAPTERS, chapters=chapters)
            await sess.run(
                _WIRE_ROOT_CHAPTERS,
                source_id=snap.source_id,
                root_ids=snap.root_ids,
            )
            await sess.run(_WIRE_NESTING, nesting=nesting)
            await sess.run(
                _REWIRE_ARTICLE_LINKS,
                source_id=snap.source_id,
                article_links=article_links,
            )
            await sess.run(_PRUNE_CHAPTERS, source_id=snap.source_id, keep=keep)

    async def article_count_by_source(self, source_id: str) -> int:
        async with self._driver.session() as sess:
            r = await sess.run(
                "MATCH (a:Article {source_id:$s}) RETURN count(a) AS n", s=source_id
            )
            return (await r.single())["n"]

    async def chapter_exists(self, chapter_id: str) -> bool:
        async with self._driver.session() as sess:
            r = await sess.run("MATCH (c:Chapter {id:$id}) RETURN c LIMIT 1", id=chapter_id)
            return await r.single() is not None

    async def article_chapter_id(self, article_id: str) -> str | None:
        async with self._driver.session() as sess:
            r = await sess.run(
                "MATCH (a:Article {id:$id})-[:IN_CHAPTER]->(c) RETURN c.id AS id",
                id=article_id,
            )
            rec = await r.single()
            return rec["id"] if rec else None
