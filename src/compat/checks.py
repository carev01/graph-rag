"""The compatibility check registry.

Registry ORDER IS SIGNIFICANT: group 5 (`graphiti-write`) creates the synthetic
graph that groups 6 (`graphiti-search`) and 7 (`our-cypher`) query. run_all
executes sequentially in this order."""
from __future__ import annotations

from compat.model import (
    COMPAT_GROUP_ID, CallableCheck, Check, CheckContext, CypherCheck, SkipCheck,
)

# --- synthetic fixture identifiers (stable so checks can reference each other) ---
EP_UUID = "compat-ep-1"
ENT_A, ENT_B, ENT_C = "compat-ent-a", "compat-ent-b", "compat-ent-c"
FACT_AB, FACT_BC = "compat-fact-ab", "compat-fact-bc"

_GRAPHITI_FULLTEXT_INDEXES = [
    "episode_content", "node_name_and_summary", "community_name", "edge_name_and_fact",
]

_LUCENE_METACHARACTERS = r'+ - && || ! ( ) { } [ ] ^ " ~ * ? : \ /'


# --------------------------------------------------------------------------- #
# Group 1: server
# --------------------------------------------------------------------------- #

async def _gds_version(ctx: CheckContext) -> str:
    """GDS is a plugin, not a Neo4j capability: absence is a skip, not a failure."""
    try:
        async with ctx.driver.session() as s:
            result = await s.run("CALL gds.version() YIELD gdsVersion RETURN gdsVersion")
            rows = [dict(rec) async for rec in result]
    except Exception as exc:  # noqa: BLE001
        raise SkipCheck(f"GDS not available: {type(exc).__name__}: {exc}") from exc
    return f"GDS {rows[0]['gdsVersion']}" if rows else "GDS present, no version row"


def server_checks() -> list[Check]:
    return [
        CypherCheck(
            "kernel version and edition", "server",
            "CALL dbms.components() YIELD name, versions, edition "
            "WHERE name = 'Neo4j Kernel' RETURN versions[0] AS version, edition",
            expect=lambda rows: bool(rows)),
        CypherCheck(
            "default Cypher language", "server",
            "SHOW SETTINGS YIELD name, value WHERE name = 'db.query.default_language' "
            "RETURN value",
            expect=lambda rows: bool(rows)),
        CallableCheck("gds version", "server", _gds_version),
        CypherCheck(
            "apoc installed", "server",
            "SHOW PROCEDURES YIELD name WHERE name STARTS WITH 'apoc.' "
            "RETURN count(*) AS n",
            expect=lambda rows: bool(rows) and rows[0]["n"] > 0,
            # Informational: nothing in graphiti-core or src/ calls apoc.*, so its
            # absence on the target is a fact to record, never a blocker.
            informational=True),
        CypherCheck(
            "fulltext procedures available", "server",
            "SHOW PROCEDURES YIELD name "
            "WHERE name IN ['db.index.fulltext.queryNodes', "
            "'db.index.fulltext.queryRelationships'] RETURN count(*) AS n",
            expect=lambda rows: bool(rows) and rows[0]["n"] == 2),
        CypherCheck(
            "vector similarity function available", "server",
            "SHOW FUNCTIONS YIELD name WHERE name = 'vector.similarity.cosine' "
            "RETURN count(*) AS n",
            expect=lambda rows: bool(rows) and rows[0]["n"] == 1),
    ]


# --------------------------------------------------------------------------- #
# Group 2: bootstrap
# --------------------------------------------------------------------------- #

async def _graphiti_bootstrap(ctx: CheckContext) -> str:
    from graph_extract.graphiti_client import init_indices
    await init_indices(ctx.graphiti)
    return "build_indices_and_constraints() completed"


async def _structural_schema(ctx: CheckContext) -> str:
    from compat.runner import compat_target
    from graph_sync.neo4j_repo import Neo4jRepo
    uri, user, password = compat_target(ctx.settings)
    repo = Neo4jRepo(uri, user, password)
    try:
        await repo.init_schema()
    finally:
        await repo.close()
    return "graph_sync init_schema() completed"


def bootstrap_checks() -> list[Check]:
    return [
        CallableCheck("graphiti build_indices_and_constraints", "bootstrap",
                      _graphiti_bootstrap),
        CypherCheck(
            "graphiti fulltext indexes exist", "bootstrap",
            "SHOW INDEXES YIELD name, type WHERE type = 'FULLTEXT' "
            "RETURN collect(name) AS names",
            expect=lambda rows: bool(rows) and all(
                ix in rows[0]["names"] for ix in _GRAPHITI_FULLTEXT_INDEXES)),
        CypherCheck(
            "graphiti range indexes exist", "bootstrap",
            "SHOW INDEXES YIELD name, type WHERE type = 'RANGE' "
            "RETURN count(*) AS n",
            expect=lambda rows: bool(rows) and rows[0]["n"] > 0),
        CallableCheck("graph_sync structural schema", "bootstrap", _structural_schema),
    ]


# --------------------------------------------------------------------------- #
# Group 3: vector
# --------------------------------------------------------------------------- #
# Narrow by design: graphiti 0.29.2 creates NO vector indexes and scores similarity
# with brute-force `vector.similarity.cosine` in Cypher, so an ANN index is never
# consulted. We verify the function over both a node and a relationship property
# (fact embeddings live on RELATES_TO), plus graphiti's exact min_score query shape.

def vector_checks() -> list[Check]:
    return [
        CypherCheck(
            "seed vector probe nodes", "vector",
            "CREATE (a:CompatVec {uuid:'compat-vec-a', group_id:$g, emb:$v}), "
            "       (b:CompatVec {uuid:'compat-vec-b', group_id:$g, emb:$v}) "
            "CREATE (a)-[:COMPAT_REL {uuid:'compat-vec-rel', group_id:$g, emb:$v}]->(b) "
            "RETURN count(*) AS created",
            params={"g": COMPAT_GROUP_ID, "v": None}),   # v injected by all_checks()
        CypherCheck(
            "cosine similarity over a node property", "vector",
            "MATCH (n:CompatVec {group_id:$g}) "
            "RETURN vector.similarity.cosine(n.emb, $v) AS score LIMIT 1",
            params={"g": COMPAT_GROUP_ID, "v": None},
            expect=lambda rows: bool(rows) and rows[0]["score"] is not None),
        CypherCheck(
            "cosine similarity over a relationship property", "vector",
            "MATCH ()-[r:COMPAT_REL {group_id:$g}]->() "
            "RETURN vector.similarity.cosine(r.emb, $v) AS score LIMIT 1",
            params={"g": COMPAT_GROUP_ID, "v": None},
            expect=lambda rows: bool(rows) and rows[0]["score"] is not None),
        CypherCheck(
            "graphiti min_score query shape", "vector",
            "MATCH (n:CompatVec {group_id:$g}) "
            "WITH n, vector.similarity.cosine(n.emb, $v) AS score "
            "WHERE score > $min_score "
            "RETURN n.uuid AS uuid, score ORDER BY score DESC LIMIT 10",
            params={"g": COMPAT_GROUP_ID, "v": None, "min_score": 0.0},
            expect=lambda rows: len(rows) >= 1),
        CypherCheck(
            "CREATE VECTOR INDEX accepted", "vector",
            "CREATE VECTOR INDEX compat_vec_probe IF NOT EXISTS "
            "FOR (n:CompatVec) ON (n.emb) OPTIONS {indexConfig: "
            "{`vector.dimensions`: 768, `vector.similarity_function`: 'cosine'}}",
            # Informational: nothing in the stack uses a vector index today. This
            # probe exists to confirm the remediation path for the brute-force-scan
            # follow-up (spec section 9) is open on this server.
            informational=True),
    ]


# --------------------------------------------------------------------------- #
# Group 4: fulltext
# --------------------------------------------------------------------------- #

async def _lucene_escaping(ctx: CheckContext) -> str:
    """Every Lucene metacharacter, pushed through graphiti's own sanitiser and into
    the fulltext procedure. An unsanitised metacharacter raises a Lucene
    ParseException at query time — a break that only shows up on real questions."""
    from graphiti_core.helpers import lucene_sanitize
    sanitized = lucene_sanitize(f"vault {_LUCENE_METACHARACTERS} lock")
    async with ctx.driver.session() as s:
        result = await s.run(
            "CALL db.index.fulltext.queryNodes('compat_ft_probe', $q) "
            "YIELD node, score RETURN count(*) AS n", q=sanitized)
        rows = [dict(rec) async for rec in result]
    return f"sanitised query accepted, {rows[0]['n'] if rows else 0} hits"


def fulltext_checks() -> list[Check]:
    # Two distinct awaits are needed, for two distinct reasons:
    #   db.awaitIndexes waits for the just-created index to finish its initial
    #   POPULATING phase and come ONLINE. Skipping this risks querying an index
    #   that isn't ready yet — a false compatibility failure, not a real one.
    #   awaitEventuallyConsistentIndexRefresh (below, after seeding) instead waits
    #   for newly-written nodes to become visible to an already-ONLINE index.
    return [
        CypherCheck(
            "CREATE FULLTEXT INDEX accepted", "fulltext",
            "CREATE FULLTEXT INDEX compat_ft_probe IF NOT EXISTS "
            "FOR (n:CompatFt) ON EACH [n.name, n.summary]"),
        CypherCheck(
            "await fulltext index online", "fulltext",
            "CALL db.awaitIndexes(120)"),
        CypherCheck(
            "seed fulltext probe node", "fulltext",
            "CREATE (n:CompatFt {uuid:'compat-ft-a', group_id:$g, "
            "name:'vault lock', summary:'immutability and retention'}) "
            "RETURN n.uuid AS uuid",
            params={"g": COMPAT_GROUP_ID}),
        CypherCheck(
            "await fulltext index refresh", "fulltext",
            "CALL db.index.fulltext.awaitEventuallyConsistentIndexRefresh()"),
        CypherCheck(
            "db.index.fulltext.queryNodes returns ranked hits", "fulltext",
            "CALL db.index.fulltext.queryNodes('compat_ft_probe', 'vault') "
            "YIELD node, score RETURN node.uuid AS uuid, score",
            expect=lambda rows: len(rows) >= 1),
        CallableCheck("lucene metacharacter escaping", "fulltext", _lucene_escaping),
    ]
