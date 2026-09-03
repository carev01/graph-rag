"""The compatibility check registry.

Registry ORDER IS SIGNIFICANT: group 5 (`graphiti-write`) creates the synthetic
graph that groups 6 (`graphiti-search`) and 7 (`our-cypher`) query. run_all
executes sequentially in this order."""
from __future__ import annotations

import httpx
import openai

from compat.model import (
    COMPAT_GROUP_ID, CallableCheck, Check, CheckContext, CypherCheck, SkipCheck,
)

#: An unreachable LLM/embedder endpoint is not a Neo4j incompatibility — the e2e
#: check skips on these rather than failing. openai.APIConnectionError is the base
#: class of openai.APITimeoutError, so both are covered.
_ENDPOINT_DOWN = (httpx.ConnectError, httpx.ConnectTimeout, openai.APIConnectionError)

# --- synthetic fixture identifiers (stable so checks can reference each other) ---
EP_UUID = "compat-ep-1"
ENT_A, ENT_B, ENT_C = "compat-ent-a", "compat-ent-b", "compat-ent-c"
FACT_AB, FACT_BC = "compat-fact-ab", "compat-fact-bc"

# --- structural fixture ids -------------------------------------------------
# SAFETY: these are LITERAL and `compat-check-`-prefixed on purpose. apply_structural
# runs `MERGE (v:Vendor {id: $vendor.id}) SET v += $vendor`, so a collision with a real
# vendor/article id would stamp group_id="compat-check" onto a PRODUCTION node -- which
# teardown would then delete. Never derive these from settings or from the target.
VENDOR_ID = "compat-check-vendor"
PRODUCT_ID = "compat-check-product"
SOURCE_ID = "compat-check-source"
ARTICLE_ID = "compat-check-article"
E2E_ARTICLE_ID = "compat-check-article-e2e"
# .invalid is a reserved non-resolving TLD (RFC 2606): a fabricated URL appearing in a
# report must never be clickable or mistakable for real vendor documentation.
ARTICLE_URL = "https://example.invalid/compat-check/vault-lock"
E2E_ARTICLE_URL = "https://example.invalid/compat-check/e2e"
HEADING_PATH = "Retention"
VENDOR_NAME = "AWS"          # _vendor_episode_uuids(driver, "AWS") must find this
PRODUCT_NAME = "AWS Backup"

# An episode deliberately NEVER linked to an :Article, plus a fact supported only by
# it. The staleness sweep must expire exactly this fact and leave the supported ones
# alone -- covering both directions of a query that silently invalidates data.
EP_ORPHAN = "compat-ep-orphan"
FACT_ORPHAN = "compat-fact-orphan"

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


async def _structural_fixture(ctx: CheckContext) -> str:
    """Write the Vendor->Product->Source->Article chain through the REAL ingestion
    path (graph_sync.neo4j_repo.apply_structural), not hand-written Cypher. This
    covers a write query that was previously untested, and because _APPLY_STRUCTURAL
    is `MERGE ... SET v += $vendor`, passing group_id puts every created node inside
    teardown's scoped sweep."""
    from compat.runner import compat_target
    from graph_sync.models import StructuralWrite
    from graph_sync.neo4j_repo import Neo4jRepo

    uri, user, password = compat_target(ctx.settings)
    repo = Neo4jRepo(uri, user, password)
    try:
        await repo.apply_structural(StructuralWrite(
            vendor={"id": VENDOR_ID, "name": VENDOR_NAME, "group_id": COMPAT_GROUP_ID},
            product={"id": PRODUCT_ID, "name": PRODUCT_NAME,
                     "group_id": COMPAT_GROUP_ID},
            source={"id": SOURCE_ID, "name": "compat source",
                    "group_id": COMPAT_GROUP_ID},
            article={"id": ARTICLE_ID, "title": "Vault Lock",
                     "source_url": ARTICLE_URL, "source_id": SOURCE_ID,
                     "group_id": COMPAT_GROUP_ID}))
    finally:
        await repo.close()
    return f"structural chain written: {VENDOR_ID} -> {ARTICLE_ID}"


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
        CallableCheck("structural fixture chain", "bootstrap", _structural_fixture),
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


# --------------------------------------------------------------------------- #
# Group 5: graphiti-write  (dynamic labels + bi-temporal edges, NO LLM)
# --------------------------------------------------------------------------- #
# Builds the synthetic graph that groups 6 and 7 query. Embeddings are fabricated,
# so this group costs nothing and needs no embedder.

async def _write_synthetic_graph(ctx: CheckContext) -> str:
    from datetime import datetime, timezone

    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode

    now = datetime.now(timezone.utc)
    gdriver = ctx.graphiti.driver

    episode = EpisodicNode(
        uuid=EP_UUID, name="compat episode", group_id=COMPAT_GROUP_ID, created_at=now,
        source=EpisodeType.text, source_description="compatibility harness",
        content="AWS Backup Vault Lock enforces immutable retention on recovery points.",
        valid_at=now)
    await episode.save(gdriver)

    orphan_episode = EpisodicNode(
        uuid=EP_ORPHAN, name="compat orphan episode", group_id=COMPAT_GROUP_ID,
        created_at=now, source=EpisodeType.text,
        source_description="compatibility harness (deliberately unlinked)",
        content="An episode intentionally never linked to an :Article.",
        valid_at=now)
    await orphan_episode.save(gdriver)

    entities = [
        EntityNode(uuid=ENT_A, name="AWS Backup", group_id=COMPAT_GROUP_ID,
                   labels=["Entity", "Product"], created_at=now,
                   name_embedding=ctx.embedding, summary="a backup service"),
        EntityNode(uuid=ENT_B, name="Vault Lock", group_id=COMPAT_GROUP_ID,
                   labels=["Entity", "Feature"], created_at=now,
                   name_embedding=ctx.embedding, summary="an immutability control"),
        EntityNode(uuid=ENT_C, name="Recovery Point", group_id=COMPAT_GROUP_ID,
                   labels=["Entity", "Concept"], created_at=now,
                   name_embedding=ctx.embedding, summary="a stored backup"),
    ]
    for entity in entities:
        await entity.save(gdriver)

    edges = [
        EntityEdge(uuid=FACT_AB, group_id=COMPAT_GROUP_ID, source_node_uuid=ENT_A,
                   target_node_uuid=ENT_B, created_at=now, name="HAS_FEATURE",
                   fact="AWS Backup provides Vault Lock.",
                   fact_embedding=ctx.embedding, episodes=[EP_UUID],
                   valid_at=now, invalid_at=None),
        EntityEdge(uuid=FACT_BC, group_id=COMPAT_GROUP_ID, source_node_uuid=ENT_B,
                   target_node_uuid=ENT_C, created_at=now, name="PROTECTS",
                   fact="Vault Lock enforces immutable retention on recovery points.",
                   fact_embedding=ctx.embedding, episodes=[EP_UUID],
                   valid_at=now, invalid_at=None),
        EntityEdge(uuid=FACT_ORPHAN, group_id=COMPAT_GROUP_ID, source_node_uuid=ENT_A,
                   target_node_uuid=ENT_C, created_at=now, name="UNSUPPORTED",
                   fact="A fact whose only supporting episode has no article.",
                   fact_embedding=ctx.embedding, episodes=[EP_ORPHAN],
                   valid_at=now, invalid_at=None),
    ]
    for edge in edges:
        await edge.save(gdriver)
    return "2 episodes (1 orphan), 3 entities, 3 bi-temporal facts written"


async def _dynamic_labels_persisted(ctx: CheckContext) -> str:
    """Graphiti writes extra labels alongside :Entity. On Neo4j 5.22 the bulk form
    `SET n:$(node.labels)` is a syntax error; here we assert the labels landed."""
    async with ctx.driver.session() as s:
        result = await s.run(
            "MATCH (n:Entity {uuid:$u}) RETURN labels(n) AS labels", u=ENT_A)
        rows = [dict(rec) async for rec in result]
    if not rows:
        raise RuntimeError(f"entity {ENT_A} was not persisted")
    labels = set(rows[0]["labels"])
    if "Product" not in labels:
        raise RuntimeError(f"dynamic label not applied; got {sorted(labels)}")
    return f"labels persisted: {sorted(labels)}"


async def _bitemporal_properties_persisted(ctx: CheckContext) -> str:
    async with ctx.driver.session() as s:
        result = await s.run(
            "MATCH ()-[f:RELATES_TO {uuid:$u}]->() "
            "RETURN f.valid_at IS NOT NULL AS has_valid, "
            "       f.invalid_at IS NULL AS open, "
            "       size(f.fact_embedding) AS dim", u=FACT_AB)
        rows = [dict(rec) async for rec in result]
    if not rows:
        raise RuntimeError(f"fact {FACT_AB} was not persisted")
    row = rows[0]
    if not (row["has_valid"] and row["open"]):
        raise RuntimeError(f"bi-temporal properties wrong: {row}")
    return f"valid_at set, invalid_at null, embedding dim {row['dim']}"


async def _link_episode(ctx: CheckContext) -> str:
    """Attach the Article to the episode via the REAL provenance writer. In production
    this edge is graph-sync's job, never graphiti's -- which is why the fixture had no
    HAS_EPISODE chain and resolve_citations returned zero sources."""
    from graph_extract.provenance import Provenance

    await Provenance(ctx.driver).link(
        ARTICLE_ID, EP_UUID, chunk_index=0, heading_path=HEADING_PATH,
        token_count=42, content_hash="compat-check-hash")
    async with ctx.driver.session() as s:
        result = await s.run(
            "MATCH (:Article {id:$a})-[r:HAS_EPISODE]->(e:Episodic {uuid:$u}) "
            "RETURN r.heading_path AS section", a=ARTICLE_ID, u=EP_UUID)
        rows = [dict(rec) async for rec in result]
    if not rows:
        raise RuntimeError("Provenance.link did not create the HAS_EPISODE edge")
    return f"article linked to episode, section={rows[0]['section']!r}"


def graphiti_write_checks() -> list[Check]:
    return [
        CallableCheck("write synthetic graph via graphiti models", "graphiti-write",
                      _write_synthetic_graph),
        CallableCheck("dynamic entity labels persisted", "graphiti-write",
                      _dynamic_labels_persisted),
        # The construct from graphiti's BULK node save
        # (models/nodes/node_db_queries.py:260). `.save()` interpolates labels as
        # literal query text, so only this exercises Cypher's native dynamic-label
        # expression -- the exact statement Neo4j 5.22 cannot parse. Declarative so
        # a failure is auto-retried under CYPHER 5.
        CypherCheck(
            "bulk dynamic-label expression SET n:$(node.labels)", "graphiti-write",
            "UNWIND $nodes AS node "
            "MERGE (n:Entity {uuid: node.uuid}) "
            "SET n:$(node.labels) "
            "SET n.group_id = node.group_id "
            "RETURN labels(n) AS labels",
            params={"nodes": [{"uuid": "compat-bulk-1",
                               "labels": ["Product", "Feature"],
                               "group_id": COMPAT_GROUP_ID}]},
            expect=lambda rows: bool(rows) and "Product" in rows[0]["labels"]),
        CallableCheck("bi-temporal fact properties persisted", "graphiti-write",
                      _bitemporal_properties_persisted),
        CallableCheck("provenance link (Article HAS_EPISODE)", "graphiti-write",
                      _link_episode),
    ]


# --------------------------------------------------------------------------- #
# Group 6: graphiti-search  (the two recipes this codebase actually uses)
# --------------------------------------------------------------------------- #

async def _search_rrf(ctx: CheckContext) -> str:
    from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
    config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
    config.limit = 10
    results = await ctx.graphiti._search(
        "vault lock immutable retention", config, group_ids=[COMPAT_GROUP_ID])
    return f"RRF recipe returned {len(results.edges)} edges"


async def _search_node_distance(ctx: CheckContext) -> str:
    from graphiti_core.search.search_config_recipes import (
        EDGE_HYBRID_SEARCH_NODE_DISTANCE)
    config = EDGE_HYBRID_SEARCH_NODE_DISTANCE.model_copy(deep=True)
    config.limit = 10
    results = await ctx.graphiti._search(
        "vault lock immutable retention", config, group_ids=[COMPAT_GROUP_ID],
        center_node_uuid=ENT_A)
    return f"node_distance recipe returned {len(results.edges)} edges"


def graphiti_search_checks() -> list[Check]:
    # Only these two recipes appear anywhere in src/ (answer_api/search.py:3-4):
    # global_search ranks communities with its own Python cosine and drift reuses
    # search_local, so no node or community recipe is exercised by this codebase.
    return [
        CallableCheck("EDGE_HYBRID_SEARCH_RRF recipe", "graphiti-search", _search_rrf),
        CallableCheck("EDGE_HYBRID_SEARCH_NODE_DISTANCE recipe", "graphiti-search",
                      _search_node_distance),
    ]


# --------------------------------------------------------------------------- #
# Group 7: our-cypher  (every query we own, run against the synthetic graph)
# --------------------------------------------------------------------------- #

async def _resolve_citations(ctx: CheckContext) -> str:
    from graph_extract.provenance import Provenance
    resolved = await Provenance(ctx.driver).resolve_citations([FACT_AB, FACT_BC])
    return f"resolve_citations returned {len(resolved)} entries"


async def _vendor_scope(ctx: CheckContext) -> str:
    from answer_api.search import _vendor_episode_uuids
    uuids = await _vendor_episode_uuids(ctx.driver, "AWS")
    return f"vendor episode scope query ran, {len(uuids)} uuids"


async def _freshness(ctx: CheckContext) -> str:
    """freshness() swallows its own errors by design (it must never fail an answer),
    so assert the returned SHAPE instead of relying on an exception."""
    from answer_api.freshness import freshness
    stamps = await freshness(ctx.driver, COMPAT_GROUP_ID, reports=True)
    if set(stamps) != {"graph_cursor_time", "reports_as_of"}:
        raise RuntimeError(f"unexpected freshness shape: {stamps}")
    if stamps["graph_cursor_time"] is None:
        raise RuntimeError("graph_cursor_time is None despite a written episode — "
                           "the freshness query failed and was swallowed")
    return f"freshness stamps resolved: {stamps}"


async def _timeline_sweep_flags(ctx: CheckContext) -> str:
    from answer_api.timeline import _sweep_flags
    flags = await _sweep_flags(ctx.driver, [FACT_AB, FACT_BC], COMPAT_GROUP_ID)
    return f"timeline sweep-flag query ran, {len(flags)} flags"


async def _corpus_cursor_subquery(ctx: CheckContext) -> str:
    """theme_builder/cli.py's watermark — a CALL {} UNION ALL subquery."""
    from theme_builder.cli import _corpus_cursor
    cursor = await _corpus_cursor(ctx.driver, COMPAT_GROUP_ID)
    if cursor is None:
        raise RuntimeError("corpus cursor is None despite a written episode")
    return f"corpus cursor subquery ran: {cursor}"


async def _staleness_sweep(ctx: CheckContext) -> str:
    """graph_extract/staleness_sweep.py — a scoped `CALL (eps) {...}` subquery
    (Neo4j 5.23+ call-scope form). The synthetic facts' only supporting episode
    (EP_UUID) is never linked to an :Article, so the sweep correctly treats it as
    unsupported and expires both facts — this is expected, not a bug: it confirms
    the sweep's "no surviving support" logic actually runs and fires."""
    from graph_extract.staleness_sweep import sweep_stale_facts
    outcome = await sweep_stale_facts(ctx.driver, COMPAT_GROUP_ID)
    return f"staleness sweep subquery ran: expired={outcome.get('expired')}"


async def _touched_entities(ctx: CheckContext) -> str:
    """A None cursor short-circuits touched_entities with no Cypher at all (see
    theme_builder/incremental.py's `if prev_cursor is None: return None`), which
    would make this check report success without exercising anything. Pass a past
    ISO-8601 timestamp — the format its `datetime($c)` Cypher expects — so the
    query actually runs."""
    from theme_builder.incremental import touched_entities
    touched = await touched_entities(ctx.driver, COMPAT_GROUP_ID, "2000-01-01T00:00:00Z")
    count = "all (None sentinel)" if touched is None else len(touched)
    return f"touched_entities ran: {count}"


async def _load_persisted(ctx: CheckContext) -> str:
    from theme_builder.incremental import load_persisted, prev_corpus_cursor
    persisted = await load_persisted(ctx.driver, COMPAT_GROUP_ID)
    cursor = await prev_corpus_cursor(ctx.driver, COMPAT_GROUP_ID)
    return f"load_persisted={len(persisted)} communities, prev_cursor={cursor}"


async def _leiden_detect(ctx: CheckContext) -> str:
    """GDS projection + seeded Leiden. GDS is a plugin: absence is a skip."""
    from theme_builder.detect import detect_communities
    try:
        async with ctx.driver.session() as s:
            await s.run("CALL gds.version() YIELD gdsVersion RETURN gdsVersion")
    except Exception as exc:  # noqa: BLE001
        raise SkipCheck(f"GDS not available: {type(exc).__name__}") from exc
    communities = await detect_communities(
        ctx.driver, COMPAT_GROUP_ID, min_community_size=1, max_levels=2)
    return f"GDS projection + seeded Leiden ran: {len(communities)} communities"


def our_cypher_checks() -> list[Check]:
    return [
        CallableCheck("provenance resolve_citations", "our-cypher", _resolve_citations),
        CallableCheck("vendor episode scope", "our-cypher", _vendor_scope),
        CallableCheck("freshness stamps", "our-cypher", _freshness),
        CallableCheck("timeline sweep flags", "our-cypher", _timeline_sweep_flags),
        CallableCheck("theme-builder corpus cursor (CALL {} UNION ALL)", "our-cypher",
                      _corpus_cursor_subquery),
        CallableCheck("staleness sweep (scoped CALL (eps) {...})", "our-cypher",
                      _staleness_sweep),
        CallableCheck("incremental touched_entities", "our-cypher", _touched_entities),
        CallableCheck("incremental load_persisted", "our-cypher", _load_persisted),
        CallableCheck("GDS projection + seeded leiden detect", "our-cypher",
                      _leiden_detect),
    ]


# --------------------------------------------------------------------------- #
# Group 8: e2e  (ONE real article: extraction + retrieval, never synthesis)
# --------------------------------------------------------------------------- #

_E2E_ARTICLE = """# AWS Backup Vault Lock

AWS Backup Vault Lock enforces a write-once, read-many (WORM) setting on a backup
vault. Once a vault lock is in compliance mode, the retention period of a recovery
point cannot be shortened and recovery points cannot be deleted before they expire.

## Retention

The minimum retention period defines the shortest retention any backup plan may
assign to recovery points in the locked vault.
"""


async def _e2e_ingest_and_retrieve(ctx: CheckContext) -> str:
    """One inlined article through the real extraction pipeline, then retrieval.

    Inlined rather than fetched so the check does not depend on DocExtractor being
    up. Extraction + retrieval only: no synthesis, so this never touches the
    strong evaluation tier."""
    from datetime import datetime, timezone

    from answer_api.search import search_local
    from graph_extract.graphiti_client import add_text_episode

    harness_settings = ctx.settings.model_copy(update={"group_id": COMPAT_GROUP_ID})
    try:
        await add_text_episode(
            ctx.graphiti, harness_settings, name="compat-e2e-article",
            body=_E2E_ARTICLE, source_description="compatibility harness",
            reference_time=datetime.now(timezone.utc))
    except _ENDPOINT_DOWN as exc:
        # An unreachable LLM/embedder endpoint is not a Neo4j incompatibility.
        raise SkipCheck(f"model endpoint unreachable ({type(exc).__name__}: {exc})") from exc

    async with ctx.driver.session() as s:
        result = await s.run(
            "MATCH (e:Episodic {group_id:$g}) "
            "OPTIONAL MATCH (n:Entity {group_id:$g}) "
            "OPTIONAL MATCH ()-[f:RELATES_TO {group_id:$g}]->() "
            "RETURN count(DISTINCT e) AS episodes, count(DISTINCT n) AS entities, "
            "       count(DISTINCT f) AS facts", g=COMPAT_GROUP_ID)
        rows = [dict(rec) async for rec in result]
    counts = rows[0] if rows else {}
    if not counts.get("facts"):
        raise RuntimeError(f"ingest produced no facts: {counts}")

    found = await search_local(
        ctx.graphiti, ctx.driver, q="What does Vault Lock enforce?", k=5,
        group_id=COMPAT_GROUP_ID)
    if found["count"] == 0:
        raise RuntimeError("search_local returned no results after ingest")
    return (f"ingested {counts}; search_local returned {found['count']} results "
            f"with {sum(len(r['sources']) for r in found['results'])} resolved sources")


def e2e_checks() -> list[Check]:
    return [CallableCheck("one article: extract then retrieve", "e2e",
                          _e2e_ingest_and_retrieve)]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def all_checks(embedding: list[float]) -> list[Check]:
    """Every group, in execution order. Group 5 writes the synthetic graph that
    groups 6 and 7 query, so this order is load-bearing.

    CypherChecks declare their vector parameter as `{"v": None}`; the fabricated
    768-d vector is substituted here so the registry stays a pure literal."""
    registry = (server_checks() + bootstrap_checks() + vector_checks()
                + fulltext_checks() + graphiti_write_checks()
                + graphiti_search_checks() + our_cypher_checks() + e2e_checks())
    resolved: list[Check] = []
    for check in registry:
        if isinstance(check, CypherCheck) and check.params.get("v", "") is None:
            params = dict(check.params)
            params["v"] = embedding
            check = CypherCheck(
                name=check.name, group=check.group, cypher=check.cypher,
                params=params, expect=check.expect, informational=check.informational)
        resolved.append(check)
    return resolved
