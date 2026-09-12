"""Stop shipping `fact_embedding` in candidate-search results.

graphiti's edge return query ends `properties(e) AS attributes`, which on Neo4j
sweeps in EVERY relationship property -- including `fact_embedding`, 768 floats.
`get_entity_edge_from_record` then does `attributes.pop('fact_embedding', None)`:
the vector is serialised from the database, parsed into Python, and thrown away.

Measured 2026-09-12 against the live graph (3,096 facts, 20 candidates):

    with properties(e)      751 ms
    without                 269 ms      2.8x, identical rows

graphiti issues TWO of these searches per extracted fact (duplicate candidates
and invalidation candidates) at ~67 facts per article, so the waste is 15,360
floats per search on the path that dominates ingestion time.

**Why this cannot change behaviour.** `get_entity_edge_from_record` reads the
vector from a TOP-LEVEL record key (`record.get('fact_embedding')`), which the
search return query never provides -- only `properties(e)` carried it, into a
dict whose very next statement discards it. Search-derived `EntityEdge`s already
have `fact_embedding=None` both before and after this patch. Anything that
genuinely needs the vector calls `load_fact_embedding()`, which is unaffected.

Scope: the SEARCH path only (`search_utils`). `edges.py`'s own uses of the same
return query -- `get_by_uuid`, etc. -- are left alone: they are not the hot path,
and narrowing the patch narrows the blast radius.
"""
from __future__ import annotations

import logging

from graphiti_core.models.edges.edge_db_queries import get_entity_edge_return_query
from graphiti_core.search import search_utils

logger = logging.getLogger(__name__)

_PROPERTIES = "properties(e) AS attributes"

# Every key graphiti's own `get_entity_edge_from_record` pops, minus
# `fact_embedding`. Listed explicitly because Cypher has no "map without key":
# `map.drop`, `map.remove`, map subtraction and APOC are all unavailable on
# Neo4j 2026.07.1 Community (probed), and `e {.*}` re-includes the vector.
_LEAN_PROJECTION = (
    "e {.uuid, .fact, .name, .group_id, .episodes, .created_at, .expired_at, "
    ".valid_at, .invalid_at, .reference_time, .source_node_uuid, "
    ".target_node_uuid} AS attributes"
)

# The keys the projection above covers. A relationship property outside this set
# is a CUSTOM edge attribute, which the projection would silently drop -- see
# `assert_no_custom_edge_attributes`.
KNOWN_EDGE_KEYS = frozenset({
    "uuid", "fact", "fact_embedding", "name", "group_id", "episodes",
    "created_at", "expired_at", "valid_at", "invalid_at", "reference_time",
    "source_node_uuid", "target_node_uuid",
})


def lean_entity_edge_return_query(provider) -> str:
    """graphiti's return query with the embedding-bearing `properties(e)` swapped
    for an explicit projection.

    If the library's string ever stops containing `properties(e) AS attributes`,
    this returns the original UNCHANGED and logs a warning: a slower search is a
    cost, a malformed query is an outage.
    """
    original = get_entity_edge_return_query(provider)
    if _PROPERTIES not in original:
        logger.warning(
            "graphiti's edge return query no longer contains %r; leaving it "
            "unpatched (searches keep shipping fact_embedding)", _PROPERTIES)
        return original
    return original.replace(_PROPERTIES, _LEAN_PROJECTION)


def install_lean_edge_search() -> bool:
    """Point `search_utils`' copy of the return query at the lean one.

    `search_utils` imports the function by name, so the patch must be applied to
    that module's attribute -- patching the defining module would not be seen.
    Returns True if the patch is active.
    """
    search_utils.get_entity_edge_return_query = lean_entity_edge_return_query
    return _PROPERTIES not in lean_entity_edge_return_query(
        search_utils.GraphProvider.NEO4J)


async def assert_no_custom_edge_attributes(driver, group_id: str) -> None:
    """Fail loudly if any `:RELATES_TO` carries a property the lean projection
    does not list -- that would be a custom edge attribute, and the projection
    would drop it from search results silently.

    Verified empty on 2026-09-12. Known window: a custom attribute written DURING
    a run is only caught on the next start, because this runs once at startup.
    Adding one requires an ontology change, so the next run catches it before the
    attribute is ever read back through search.
    """
    async with driver.session() as s:
        r = await s.run(
            "MATCH ()-[e:RELATES_TO {group_id:$g}]->() "
            "UNWIND keys(e) AS k RETURN collect(DISTINCT k) AS ks", g=group_id)
        rec = await r.single()
    extra = sorted(set(rec["ks"] if rec else []) - KNOWN_EDGE_KEYS)
    if extra:
        raise ValueError(
            f"custom :RELATES_TO attributes found {extra}; the lean edge "
            "projection in graph_extract.lean_edge_search does not list them and "
            "would drop them from search results. Add them to _LEAN_PROJECTION "
            "and KNOWN_EDGE_KEYS, or disable the patch.")
