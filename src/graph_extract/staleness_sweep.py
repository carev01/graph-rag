"""Residual-staleness sweep: expire facts whose supporting episodes are all dead.

Graphiti invalidates a fact (`invalid_at`) when it sees a *contradicting*
fact during extraction -- but a fact whose only supporting episode(s) were
silently removed (a `removed` tombstone from DocExtractor, or an article
that shrank and dropped a chunk-episode) leaves no contradiction for
Graphiti to observe. This module is the deterministic, Cypher-only backstop
for that case: a weekly sweep that expires facts with zero live supporting
episodes.

Correctness rule (see task brief "#6"): a supporting episode is DEAD iff
`removed=true` OR it has NO `HAS_EPISODE` edge from a non-removed
`:Article`. Liveness is keyed on HAS_EPISODE linkage, NEVER on the
`superseded` flag -- a superseded episode had its HAS_EPISODE edge deleted
by `Provenance.link`'s re-key (so it's dead by linkage anyway), but a
shrink-then-restore episode can be wrongly left `superseded=true` while
STILL holding its HAS_EPISODE edge, and that episode is ALIVE. Consulting
`superseded` directly would wrongly expire facts still backed by real
content -- so this sweep never reads it.

A fact is expired iff ALL its supporting episodes are dead (no alive
episode survives). Only facts with `invalid_at IS NULL` and a non-empty
`episodes` list are candidates -- already-invalidated facts (by Graphiti or
a prior sweep run) are left untouched.
"""

from __future__ import annotations

from neo4j import AsyncDriver

# Two-step form (compute expirable fact uuids, then SET) rather than doing
# the SET inside the same query as the CALL subquery: kept the write
# separate from the read/aggregate so the "which facts qualify" logic can be
# verified independently. Also note: the newer `CALL (eps) { ... }`
# call-scope syntax is rejected by the Neo4j 5.22 testcontainer used in
# tests/integration/conftest.py ("expected an identifier or '{'") -- the
# classic `CALL { WITH eps ... }` importing form below is what's portable
# and was verified against the real container.
_FIND_EXPIRABLE = """
MATCH ()-[f:RELATES_TO {group_id:$g}]->()
WHERE f.invalid_at IS NULL AND f.episodes IS NOT NULL AND size(f.episodes) > 0
WITH f, f.episodes AS eps
CALL {
  WITH eps
  UNWIND eps AS epu
  OPTIONAL MATCH (e:Episodic {uuid: epu})
  OPTIONAL MATCH (a:Article)-[:HAS_EPISODE]->(e) WHERE coalesce(a.removed, false) = false
  WITH e, count(a) AS live_links
  RETURN sum(
    CASE WHEN e IS NOT NULL AND coalesce(e.removed, false) = false AND live_links > 0
    THEN 1 ELSE 0 END
  ) AS alive
}
WITH f WHERE alive = 0
RETURN f.uuid AS uuid
"""

_EXPIRE = """
MATCH ()-[f:RELATES_TO {group_id:$g}]->()
WHERE f.uuid IN $uuids
SET f.invalid_at = datetime(), f.expired_by_sweep = true
RETURN f.uuid AS uuid
"""

_SCAN_COUNT = """
MATCH ()-[f:RELATES_TO {group_id:$g}]->()
WHERE f.invalid_at IS NULL
RETURN count(f) AS c
"""


async def sweep_stale_facts(driver: AsyncDriver, group_id: str) -> dict:
    """Expire RELATES_TO facts with no live supporting episode.

    Returns `{"scanned": int, "expired": int, "expired_sample": list[str]}`.
    `scanned` counts every not-yet-invalidated fact in the group (the sweep's
    candidate pool); `expired` and `expired_sample` describe what this run
    actually flipped.
    """
    async with driver.session() as s:
        scan_record = await (await s.run(_SCAN_COUNT, g=group_id)).single()
        assert scan_record is not None  # count() always returns exactly one row
        scanned = scan_record["c"]

        find_result = await s.run(_FIND_EXPIRABLE, g=group_id)
        expirable_uuids = [rec["uuid"] async for rec in find_result]

        expired_sample: list[str] = []
        if expirable_uuids:
            expire_result = await s.run(_EXPIRE, g=group_id, uuids=expirable_uuids)
            expired_sample = [rec["uuid"] async for rec in expire_result][:20]

    return {
        "scanned": scanned,
        "expired": len(expirable_uuids),
        "expired_sample": expired_sample,
    }
